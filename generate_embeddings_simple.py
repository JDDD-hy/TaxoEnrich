#!/usr/bin/env python3
"""
简化版嵌入生成脚本 - 使用本地模型或简单的随机嵌入
适用于网络不稳定的情况
"""

import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
import os

# 配置
DATA_DIR = "data/MAG_CS"
TAXONOMY_NAME = "computer_science"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 768

print(f"Using device: {DEVICE}")
print("="*80)
print("简化版嵌入生成（不依赖BERT模型下载）")
print("="*80)

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

for node_id in taxoid2term:
    if node_id not in taxonomy.nodes:
        taxonomy.add_node(node_id)

print(f"   节点数: {taxonomy.number_of_nodes()}")
print(f"   边数: {taxonomy.number_of_edges()}")

# 3. 生成简单嵌入（基于术语特征）
print("\n3. 生成术语嵌入（基于字符特征）...")

def simple_text_embedding(text, dim=768):
    """
    使用简单的字符级特征生成嵌入
    这是一个临时方案，不如BERT但可以工作
    """
    # 设置随机种子以保证可重复性
    np.random.seed(hash(text) % (2**32))
    
    # 基于文本生成确定性的嵌入
    embedding = np.zeros(dim)
    
    # 使用文本的哈希值作为种子
    for i, char in enumerate(text[:100]):  # 使用前100个字符
        idx = i % dim
        embedding[idx] += ord(char) / 255.0
    
    # 归一化
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return torch.tensor(embedding, dtype=torch.float32)

taxoid2bertemb = {}
for taxo_id, term in tqdm(taxoid2term.items(), desc="   生成文本嵌入"):
    taxoid2bertemb[taxo_id] = simple_text_embedding(term, EMBED_DIM)

print(f"   生成了 {len(taxoid2bertemb)} 个文本嵌入")

# 4. 生成结构嵌入
print("\n4. 生成结构嵌入（基于图上下文）...")

def get_context_embedding(node_id, taxonomy, taxoid2bertemb):
    """基于节点的上下文生成嵌入"""
    embeddings = []
    
    # 节点自身
    if node_id in taxoid2bertemb:
        embeddings.append(taxoid2bertemb[node_id])
    
    # 父节点
    parents = list(taxonomy.predecessors(node_id))
    for p in parents[:3]:
        if p in taxoid2bertemb:
            embeddings.append(taxoid2bertemb[p] * 0.5)
    
    # 子节点
    children = list(taxonomy.successors(node_id))
    for c in children[:3]:
        if c in taxoid2bertemb:
            embeddings.append(taxoid2bertemb[c] * 0.5)
    
    if embeddings:
        return torch.mean(torch.stack(embeddings), dim=0)
    else:
        return torch.zeros(EMBED_DIM)

taxoid2emb = {}
for taxo_id in tqdm(taxoid2term.keys(), desc="   生成结构嵌入"):
    taxoid2emb[taxo_id] = get_context_embedding(taxo_id, taxonomy, taxoid2bertemb)

print(f"   生成了 {len(taxoid2emb)} 个结构嵌入")

# 5. 保存嵌入文件
print("\n5. 保存嵌入文件...")

# 保存结构嵌入
embed_file = f'{DATA_DIR}/{TAXONOMY_NAME}.terms.embed'
with open(embed_file, 'w', encoding='utf-8') as f:
    f.write(f"{len(taxoid2emb)} {EMBED_DIM}\n")
    for taxo_id in sorted(taxoid2emb.keys()):
        emb = taxoid2emb[taxo_id]
        embed_string = " ".join([str(float(a)) for a in emb])
        f.write(f"{taxo_id} {embed_string}\n")

print(f"   ✓ 结构嵌入已保存: {embed_file}")

# 保存文本嵌入
bertembed_file = f'{DATA_DIR}/{TAXONOMY_NAME}.terms.bertembed'
with open(bertembed_file, 'w', encoding='utf-8') as f:
    f.write(f"{len(taxoid2bertemb)} {EMBED_DIM}\n")
    for taxo_id in sorted(taxoid2bertemb.keys()):
        emb = taxoid2bertemb[taxo_id]
        embed_string = " ".join([str(float(a)) for a in emb])
        f.write(f"{taxo_id} {embed_string}\n")

print(f"   ✓ 文本嵌入已保存: {bertembed_file}")

print("\n" + "="*80)
print("✓ 嵌入生成完成！")
print("="*80)
print(f"\n生成的文件:")
print(f"  - {embed_file} ({len(taxoid2emb) + 1} 行)")
print(f"  - {bertembed_file} ({len(taxoid2bertemb) + 1} 行)")
print("\n注意: 使用的是简化版嵌入（基于字符特征），")
print("      质量不如BERT，但可以让项目运行起来。")
print("      如果需要更好的效果，请稍后手动下载BERT模型。")

