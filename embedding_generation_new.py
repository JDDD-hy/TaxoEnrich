import torch
import networkx as nx
from tqdm import tqdm
from gensim.models import FastText
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
import dgl
from gensim.models import KeyedVectors
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from transformers import *
from collections import defaultdict, deque
from typing import List, Optional
import pickle
from networkx.algorithms import descendants, ancestors
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import time
from itertools import chain, product, combinations
import os
import multiprocessing as mp
from functools import partial
import more_itertools as mit

# --------------------------
# 子图构造：统一使用 NetworkX + taxo_id 列表
# --------------------------
def _get_holdout_subgraph(g_full, node_ids):
    # 原版：假定是 DGLGraph，一刀切转 nx；如果本来就是 nx，会重复转
    # full_graph = g_full.to_networkx()
    # → 修正：如果传进来的是 DGLGraph，就转成 networkx；否则直接用
    full_graph = g_full.to_networkx() if hasattr(g_full, "to_networkx") else g_full

    node_to_remove = [n for n in full_graph.nodes if n not in node_ids]
    subgraph = full_graph.subgraph(node_ids).copy()

    # 为保持可达性：移除节点时连父子
    for node in node_to_remove:
        parents, children = set(), set()
        ps, cs = deque(full_graph.predecessors(node)), deque(full_graph.successors(node))
        while ps:
            p = ps.popleft()
            if p in subgraph:
                parents.add(p)
            else:
                ps += list(full_graph.predecessors(p))
        while cs:
            c = cs.popleft()
            if c in subgraph:
                children.add(c)
            else:
                cs += list(full_graph.successors(c))
        for p in parents:
            for c in children:
                subgraph.add_edge(p, c)

    # 去掉“跳边”
    node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
    for node in list(subgraph.nodes()):
        if subgraph.out_degree(node) > 1:
            successors1 = set(subgraph.successors(node))
            successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
            for s in successors1.intersection(successors2):
                subgraph.remove_edge(node, s)
    return subgraph

def get_root(graph):
    # 原版：直接取 topo_sort 的第一个，图为空会 IndexError
    # return list(nx.topological_sort(graph))[0]
    # → 修正：给出更清晰的错误信息
    topo = list(nx.topological_sort(graph))
    assert len(topo) > 0, "core_subgraph 为空：请检查 train_ids 是否真的出现在 taxonomy 节点里（注意 ordinal→taxo_id 的映射）"
    return topo[0]

def get_parents(graph, node):
    return [edge[0] for edge in graph.in_edges(node)]

def get_children(graph, node):
    return [edge[1] for edge in graph.out_edges(node)]

def get_leaf(graph):
    return [node for node in graph if graph.out_degree(node) == 0]

def _get_path_to_root(graph):
    node2root_path = {n: [] for n in graph.nodes}
    r = get_root(graph)
    q = deque([r])
    node2root_path[r] = [[r]]
    visit = set()
    while q:
        i = q.popleft()
        if i in visit:
            continue
        visit.add(i)
        for c in get_children(graph, i):
            if c not in q:
                q.append(c)
            for path in node2root_path[i]:
                node2root_path[c].append([c] + path)
    return node2root_path

def _get_path_to_leaf(graph):
    leafs = [n for n in graph.nodes if graph.out_degree(n) == 0]
    node2leaf_path = {n: [] for n in graph.nodes}
    q = deque(leafs)
    for n in leafs:
        node2leaf_path[n] = [[n]]
    visit = set()
    while q:
        i = q.popleft()
        if i in visit:
            continue
        visit.add(i)
        for p in get_parents(graph, i):
            if p not in q:
                q.append(p)
            for path in node2leaf_path[i]:
                node2leaf_path[p].append([p] + path)
    return node2leaf_path

# --------------------------
# 句子生成：统一用 taxo_id -> term
# --------------------------
def generate_sentences(root_path, leaf_path, graph, node, taxoid2term):
    parents = root_path.get(node, [])
    children = leaf_path.get(node, [])

    parent_sentence, children_sentence = [], []

    # 父路径：A, B, C is a super class of X
    for path in parents:
        ascendants = path[1:]
        if not ascendants:
            continue
        parts = [taxoid2term.get(aid, str(aid)) for aid in ascendants]
        sentence = ", ".join(parts) + f" is a super class of {taxoid2term.get(node, str(node))}"
        parent_sentence.append(sentence)

    # 子路径：Y, Z is a subclass of X
    for path in children:
        descendants_ = path[1:]
        if not descendants_:
            continue
        parts = [taxoid2term.get(cid, str(cid)) for cid in descendants_]
        sentence = ", ".join(parts) + f" is a subclass of {taxoid2term.get(node, str(node))}"
        children_sentence.append(sentence)

    return parent_sentence, children_sentence

# --------------------------
# BERT 抽取：补齐缺失的 model/tokenizer 传参
# --------------------------
def emb_extract(sentence, key_words, device, model, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)], device=device)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # [1, T, H]

    # 合并 WordPiece 成词级
    token_list_new, idx_new2old_map = [], []
    n = 0
    for i, token in enumerate(tokens):
        if '##' not in token:
            token_list_new.append(token)
            idx_new2old_map.append([i + 1])  # +1 是因为 [CLS] 在第 0 位
            n += 1
        else:
            token_list_new[n - 1] += token.replace('##', '')
            idx_new2old_map[n - 1].append(i + 1)

    emb_list = []
    for tgt in key_words:
        target = ''.join([t.replace("##", '') for t in tokenizer.tokenize(tgt)])
        idx = token_list_new.index(target)  # 若匹配不到会 ValueError
        old_idx = idx_new2old_map[idx]
        embs = last_hidden_states[0, old_idx, :].mean(dim=0)  # 词的平均向量
        emb_list.append(embs.to('cpu'))
    return emb_list

def phrase_emb_extract(sentence: str, phrase_list: List[str], device, model, tokenizer) -> torch.Tensor:
    # 原版：def phrase_emb_extract(..., device='cuda:0')，但没把 model/tokenizer 传进去，
    # 后面调用 phrase_emb_extract(...) 也没传 model/tokenizer → 运行时一定报错。
    # → 修正：把 model/tokenizer 显式传入，device 与上游一致。
    emb_list = []
    for phrase in phrase_list:
        words = phrase.split(' ')
        embs = emb_extract(sentence, words, device, model, tokenizer)
        phrase_emb = torch.stack(embs).mean(dim=0)
        emb_list.append(phrase_emb)
    return emb_list[0]

def generate_embeddings(taxoid2sent, node, taxoid2term, device, model, tokenizer):
    # 原版：用 taxoord2term（ordinal→term），而 node 是 taxo_id，键不对 → KeyError
    parent_sentence, children_sentence = taxoid2sent.get(node, [[], []])

    vecs = []
    if parent_sentence:
        v = torch.mean(torch.stack([phrase_emb_extract(s, [taxoid2term[node]], device, model, tokenizer)
                                    for s in parent_sentence]), dim=0)
        vecs.append(v)
    if children_sentence:
        v = torch.mean(torch.stack([phrase_emb_extract(s, [taxoid2term[node]], device, model, tokenizer)
                                    for s in children_sentence]), dim=0)
        vecs.append(v)
    if vecs:
        return torch.mean(torch.stack(vecs), dim=0)
    else:
        return torch.zeros(768)

def extract_bert_embedding(sentence, model, tokenizer, device):
    # 原版签名有 key_words 参数，但未使用；这里去掉，避免误导
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)], device=device)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0][0]  # [T, H]
    return torch.mean(last_hidden_states, dim=0)

def main():
    with open('./data/MAG_CS/computer_science.pickle.bin', 'rb') as f:
        data = pickle.load(f)
        g_full = data['g_full']
        name = data['name']
        vocab = data["vocab"]
        train_node_ids = data["train_node_ids"]           # ordinal 列表
        validation_node_ids = data["validation_node_ids"] # ordinal 列表
        test_node_ids = data["test_node_ids"]             # ordinal 列表

    # 映射：taxo_id <-> term 与 ordinal <-> taxo_id
    taxoid2term, taxoterm2id = {}, {}
    taxoid2ord, taxoord2id, taxoord2term = {}, {}, {}
    ord_ = 0
    with open('./data/MAG_CS/computer_science.terms', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            taxo_id = line.split('\t')[0]
            taxo_term = line.split('\t')[1].split('||')[0]
            # 清洗术语（注意：只对 term 清洗，不要误用到 id 上）
            taxo_term = taxo_term.replace('_', ' ').replace('-', ' ').replace("'", '').replace('/', ' ')
            taxoid2term[taxo_id] = taxo_term
            taxoterm2id[taxo_term] = taxo_id
            taxoid2ord[taxo_id] = ord_
            taxoord2id[ord_] = taxo_id
            taxoord2term[ord_] = taxo_term
            ord_ += 1

    # 读 taxonomy（parent_id \t child_id），注意不要对 id 做“清洗”
    taxonomy = nx.DiGraph()
    with open('./data/MAG_CS/computer_science.taxo', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parent = line.split('\t')[0].split('||')[0]
            child  = line.split('\t')[1].split('||')[0]
            # 原版：这里错误地对 taxo_term 做清洗（变量本身不存在），会误导读者且无意义
            # if '_' in taxo_term:
            #     taxo_term = taxo_term.replace('_', ' ')
            # ...
            taxonomy.add_edge(parent, child)

    # terms 中的孤立点也补到图里
    for node in taxoid2term:
        if node not in taxonomy:
            taxonomy.add_node(node)

    print("taxonomy nodes:", taxonomy.number_of_nodes())
    print("taxonomy edges:", taxonomy.number_of_edges())

    # 设备 + 模型
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # -------- 构造训练核心子图：用 taxo_id 列表 --------
    # 原版：core_subgraph = _get_holdout_subgraph(taxonomy, train_node_ids)
    # 错因：train_node_ids 是 ordinal，图节点是 taxo_id → 空子图 / IndexError
    train_taxo_ids = [taxoord2id[o] for o in train_node_ids if o in taxoord2id]
    core_subgraph = _get_holdout_subgraph(taxonomy, train_taxo_ids)

    assert core_subgraph.number_of_nodes() > 0, (
        "core_subgraph 为空。常见原因：train_node_ids（ordinal）未正确映射为 taxo_id；"
        "或 terms/taxo 与 pickle 的数据集不对应。"
    )

    root_path = _get_path_to_root(core_subgraph)
    leaf_path = _get_path_to_leaf(core_subgraph)

    # -------- 生成句子（仅对子图节点）--------
    taxoid2sent = {}
    nodes = list(core_subgraph.nodes)
    for i in tqdm(range(len(nodes)), desc="Generating sentences"):
        n = nodes[i]  # taxo_id
        parent_sentence, children_sentence = generate_sentences(
            root_path, leaf_path, core_subgraph, n, taxoid2term
        )
        taxoid2sent[n] = [parent_sentence, children_sentence]

    count = sum(len(v[0]) + len(v[1]) for v in taxoid2sent.values())
    print("Total number of sentences: ", count)

    # -------- 生成上下文句子嵌入（对所有 taxonomy 节点；不在子图里的记零向量）--------
    taxoid2emb = {}
    nodes_all = list(taxonomy.nodes)  # taxo_id 列表
    for i in tqdm(range(len(nodes_all)), desc="Loading embeddings"):
        taxo_id = nodes_all[i]
        emb = generate_embeddings(taxoid2sent, taxo_id, taxoid2term, device, model, tokenizer)
        taxoid2emb[taxo_id] = emb

    with open('./data/MAG_CS/computer_science.terms.embed', 'w') as f:
        f.write(f"{len(taxoid2emb)} 768\n")
        for k in sorted(taxoid2emb.keys()):  # k 是 taxo_id
            embed_string = " ".join([str(a.item()) for a in taxoid2emb[k]])
            f.write(f"{k} {embed_string}\n")

    # -------- 生成词面 BERT 平均嵌入（对所有节点）--------
    taxoid2bertemb = {}
    for node in tqdm(taxonomy.nodes, desc="Loading BERT name embeddings"):
        name = taxoid2term.get(node, str(node))
        emb = extract_bert_embedding(name, model, tokenizer, device)
        taxoid2bertemb[node] = emb

    with open('./data/MAG_CS/computer_science.terms.bertembed', 'w') as f:
        f.write(f"{len(taxoid2bertemb)} 768\n")
        for k in sorted(taxoid2bertemb.keys()):
            embed_string = " ".join([str(a.item()) for a in taxoid2bertemb[k]])
            f.write(f"{k} {embed_string}\n")

    # -------- 写回 train/valid/test 清单：此处才做 ordinal→taxo_id 的映射 --------
    with open('./data/MAG_CS/computer_science.terms.train', 'w') as f:
        for o in train_node_ids:
            if o in taxoord2id:
                tid = taxoord2id[o]
                f.write(f"{tid}\t{taxoid2term[tid]}\n")

    with open('./data/MAG_CS/computer_science.terms.validation', 'w') as f:
        for o in validation_node_ids:
            if o in taxoord2id:
                tid = taxoord2id[o]
                f.write(f"{tid}\t{taxoid2term[tid]}\n")

    with open('./data/MAG_CS/computer_science.terms.test', 'w') as f:
        for o in test_node_ids:
            if o in taxoord2id:
                tid = taxoord2id[o]
                f.write(f"{tid}\t{taxoid2term[tid]}\n")

if __name__ == "__main__":
    main()
