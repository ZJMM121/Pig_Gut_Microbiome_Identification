import pandas as pd
from ete3 import NCBITaxa
import networkx as nx
import numpy as np
import json
from tqdm import tqdm
import os
from sklearn.decomposition import PCA

# 目标级别及对应文件名
levels = [
    #("Genus", "/user_data/yezy/zhangjm/FE_kg/Merge/standardization0630/taxonomy_unique/genus_name_nodup.txt"),
    #("Family", "/user_data/yezy/zhangjm/FE_kg/Merge/standardization0630/taxonomy_unique/family_name_nodup.txt"),
    ("Species", "/user_data/yezy/zhangjm/FE_kg/Merge/standardization0630/taxonomy_unique/species_name_nodup_cleaned.txt")
]

out_dir = "/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector"
os.makedirs(out_dir, exist_ok=True)

ncbi = NCBITaxa()

def semantic_contribution(graph, node, weight_decay=0.8):
    contributions = {}
    queue = [(node, 1.0)]
    while queue:
        current_node, weight = queue.pop(0)
        if current_node in contributions:
            contributions[current_node] += weight
        else:
            contributions[current_node] = weight
        for parent in graph.predecessors(current_node):
            queue.append((parent, weight * weight_decay))
    return contributions

for level, namefile in levels:
    print(f"\n==== 处理 {level} ====")
    # 读取名单
    df = pd.read_csv(namefile)
    names = df[level].dropna().unique().tolist()
    print(f"{level}待处理数量: {len(names)}")

    # 名称转taxid
    name_to_taxid = {}
    for name in tqdm(names, desc=f"Mapping {level} to TaxID"):
        try:
            taxid = ncbi.get_name_translator([name])[name][0]
            name_to_taxid[name] = taxid
        except:
            name_to_taxid[name] = None
    with open(os.path.join(out_dir, f"{level.lower()}_taxid.json"), "w") as f:
        json.dump(name_to_taxid, f)

    # 构建DAG
    taxids = [tid for tid in name_to_taxid.values() if tid is not None]
    G = nx.DiGraph()
    for taxid in tqdm(taxids, desc=f"Building {level} taxonomy DAG"):
        try:
            lineage = ncbi.get_lineage(taxid)
            for i in range(len(lineage) - 1):
                parent, child = lineage[i], lineage[i + 1]
                G.add_edge(parent, child)
        except:
            continue

    # 生成Wang向量
    all_terms = set()
    contributions_dict = {}
    valid_names = []  # 存储有效名称
    
    for name, taxid in tqdm(name_to_taxid.items(), desc=f"Computing {level} Wang vectors"):
        if taxid is None or taxid not in G:
            continue
        contrib = semantic_contribution(G, taxid)
        contributions_dict[name] = contrib
        all_terms.update(contrib.keys())
        valid_names.append(name)
    
    term_list = sorted(list(all_terms))
    term_index = {term: i for i, term in enumerate(term_list)}
    
    # 构建完整特征矩阵（密集）
    num_samples = len(valid_names)
    num_terms = len(term_list)
    feature_matrix = np.zeros((num_samples, num_terms), dtype=np.float32)
    
    for idx, name in enumerate(tqdm(valid_names, desc="Building feature matrix")):
        contrib = contributions_dict[name]
        for term, score in contrib.items():
            j = term_index[term]
            feature_matrix[idx, j] = score
    
    # 保存原始特征和名称
    np.save(os.path.join(out_dir, f"{level.lower()}_wang_features.npy"), feature_matrix)
    with open(os.path.join(out_dir, f"{level.lower()}_wang_names.txt"), "w") as f:
        for name in valid_names:
            f.write(name + "\n")
    
    print(f"{level} 原始特征矩阵形状: {feature_matrix.shape}")
    print(f"有效样本数量: {num_samples}")

    # ========== PCA降维 ==========
    n_components = 1164
    print(f"{level} 开始PCA降维...")
    
    # 调整组件数不超过样本数
    if n_components > num_samples:
        n_components = num_samples
        print(f"调整n_components为样本数: {n_components}")
    
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(feature_matrix)
    
    # 保存降维结果
    np.save(os.path.join(out_dir, f"{level.lower()}_wang_features_pca.npy"), reduced_matrix)
    
    print(f"降维后矩阵形状: {reduced_matrix.shape}")
    print(f"累计解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"{level} 处理完成")