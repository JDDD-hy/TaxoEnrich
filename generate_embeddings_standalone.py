#!/usr/bin/env python3
"""
独立的嵌入文件生成脚本
不依赖pickle文件，直接从terms和taxo文件生成
"""

import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import networkx as nx
from collections import deque
import os

# 配置
DATA_DIR = "data/MAG_CS_backup"
TAXONOMY_NAME = "computer_science"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 768

print(f"Using device: {DEVICE}")

# 1. 加载术语
print("\n1. 加载术语...")
taxoid2term = {}
with open(f'{DATA_DIR}/{TAXONOMY_NAME}.terms', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split('\t')
            taxo_id = parts[0]
            taxo_term = parts[1]
            # 清理术语名称
            taxo_term = taxo_term.replace('_', ' ').replace('-', ' ')
            taxo_term = taxo_term.replace("'", '').replace("/", " ")
            taxoid2term[taxo_id] = taxo_term

print(f"   加载了 {len(taxoid2term)} 个术语")

# 2. 加载分类关系构建图
print("\n2. 构建分类图...")
taxonomy = nx.DiGraph()
with open(f'{DATA_DIR}/{TAXONOMY_NAME}.taxo', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split('\t')
            parent = parts[0]
            child = parts[1]
            taxonomy.add_edge(parent, child)

# 添加孤立节点
for node_id in taxoid2term:
    if node_id not in taxonomy.nodes:
        taxonomy.add_node(node_id)

print(f"   节点数: {taxonomy.number_of_nodes()}")
print(f"   边数: {taxonomy.number_of_edges()}")

# 3. 加载BERT模型
print("\n3. 加载BERT模型...")
# 使用镜像站点加速下载
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("   BERT模型加载完成")
except Exception as e:
    print(f"   下载失败，尝试使用本地缓存或镜像...")
    # 备用方案：指定缓存目录
    cache_dir = "/tmp/huggingface_cache"
    model = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir).to(DEVICE)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    print("   BERT模型加载完成（使用缓存）")

# 4. 生成BERT嵌入（基于术语名称）
print("\n4. 生成BERT嵌入（基于术语名称）...")
taxoid2bertemb = {}

for taxo_id, term in tqdm(taxoid2term.items(), desc="   生成BERT嵌入"):
    # 编码术语
    input_ids = torch.tensor([tokenizer.encode(term, add_special_tokens=True)], device=DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids)
        # 使用所有token的平均作为嵌入
        last_hidden_states = outputs[0][0]  # [seq_len, hidden_size]
        emb = torch.mean(last_hidden_states, dim=0)  # [hidden_size]
    
    taxoid2bertemb[taxo_id] = emb.cpu()

print(f"   生成了 {len(taxoid2bertemb)} 个BERT嵌入")

# 5. 生成结构嵌入（基于图结构和上下文）
print("\n5. 生成结构嵌入...")

# 获取训练集节点
train_ids = set()
with open(f'{DATA_DIR}/{TAXONOMY_NAME}.terms.train', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            train_ids.add(line.split('\t')[0])

print(f"   训练集大小: {len(train_ids)}")

def get_context_embedding(node_id, taxonomy, taxoid2bertemb, taxoid2term):
    """基于节点的上下文（父节点和子节点）生成嵌入"""
    embeddings = []
    
    # 节点自身的嵌入
    if node_id in taxoid2bertemb:
        embeddings.append(taxoid2bertemb[node_id])
    
    # 父节点嵌入
    parents = list(taxonomy.predecessors(node_id))
    for p in parents[:3]:  # 最多3个父节点
        if p in taxoid2bertemb:
            embeddings.append(taxoid2bertemb[p] * 0.5)  # 降权
    
    # 子节点嵌入
    children = list(taxonomy.successors(node_id))
    for c in children[:3]:  # 最多3个子节点
        if c in taxoid2bertemb:
            embeddings.append(taxoid2bertemb[c] * 0.5)  # 降权
    
    if embeddings:
        return torch.mean(torch.stack(embeddings), dim=0)
    else:
        return torch.zeros(EMBED_DIM)

taxoid2emb = {}
for taxo_id in tqdm(taxoid2term.keys(), desc="   生成结构嵌入"):
    taxoid2emb[taxo_id] = get_context_embedding(taxo_id, taxonomy, taxoid2bertemb, taxoid2term)

print(f"   生成了 {len(taxoid2emb)} 个结构嵌入")

# 6. 保存嵌入文件
print("\n6. 保存嵌入文件...")

# 保存结构嵌入
embed_file = f'{DATA_DIR}/{TAXONOMY_NAME}.terms.embed'
with open(embed_file, 'w', encoding='utf-8') as f:
    f.write(f"{len(taxoid2emb)} {EMBED_DIM}\n")
    for taxo_id in sorted(taxoid2emb.keys()):
        emb = taxoid2emb[taxo_id]
        embed_string = " ".join([str(a.item()) for a in emb])
        f.write(f"{taxo_id} {embed_string}\n")

print(f"   ✓ 结构嵌入已保存: {embed_file}")

# 保存BERT嵌入
bertembed_file = f'{DATA_DIR}/{TAXONOMY_NAME}.terms.bertembed'
with open(bertembed_file, 'w', encoding='utf-8') as f:
    f.write(f"{len(taxoid2bertemb)} {EMBED_DIM}\n")
    for taxo_id in sorted(taxoid2bertemb.keys()):
        emb = taxoid2bertemb[taxo_id]
        embed_string = " ".join([str(a.item()) for a in emb])
        f.write(f"{taxo_id} {embed_string}\n")

print(f"   ✓ BERT嵌入已保存: {bertembed_file}")

print("\n" + "="*80)
print("✓ 嵌入生成完成！")
print("="*80)
print(f"\n生成的文件:")
print(f"  - {embed_file} ({len(taxoid2emb) + 1} 行)")
print(f"  - {bertembed_file} ({len(taxoid2bertemb) + 1} 行)")

