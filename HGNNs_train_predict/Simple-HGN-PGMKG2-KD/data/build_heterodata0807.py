import torch
import numpy as np
import pickle
import os
from torch_geometric.data import HeteroData
from scipy import sparse
from tqdm import tqdm

# 定义特征和名称文件的根目录
BASE_FEATURE_DIR = "/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/"
# 另外，name_wang_names.txt 的路径是 /user_data/yezy/zhangjm/Simple-HGN-PGMKG2-KD/data/
# 为了简化，我们可以假设它在项目根目录下，或者明确指定绝对路径
NEW_BACTERIA_NAMES_PATH = "/user_data/yezy/zhangjm/Simple-HGN-PGMKG2-KD/data/name_wang_names.txt" # 明确指定新细菌名称文件路径


# 初始化 HeteroData 容器
data = HeteroData()

# ===== 加载节点特征 (包括合并新旧细菌) =====
print("加载节点特征...")

# 原始细菌特征和名称
original_bacteria_features_path = os.path.join(BASE_FEATURE_DIR, "bacteria_wang_features0724_128d.npy")
original_bacteria_names_path = os.path.join(BASE_FEATURE_DIR, "bacteria_wang_names0724.txt") # 假设也有原始细菌名称文件

# 新细菌特征和名称
new_bacteria_features_path = os.path.join(BASE_FEATURE_DIR, "name_wang_features0810_reduced_128.npy")
# NEW_BACTERIA_NAMES_PATH 已在文件顶部定义

all_node_types_config = [
    ('Bacteria', None, 'npy'), # Bacteria 将特殊处理，这里设为None
    ('Gene', os.path.join(BASE_FEATURE_DIR, "gene_wang_similarity.npy"), 'npy'),
    ('Metabolite', os.path.join(BASE_FEATURE_DIR, "metabolite_features0725.npy"), 'npy'),
    ('Pathway', os.path.join(BASE_FEATURE_DIR, "pathway_onehot_features.npz"), 'npz'),
    ('Trait', os.path.join(BASE_FEATURE_DIR, "trait_onehot_features0724.npz"), 'npz'),
    ('PMID', os.path.join(BASE_FEATURE_DIR, "pmid_onehot_features0725.npz"), 'npz'),
    ('ID', os.path.join(BASE_FEATURE_DIR, "id_onehot_features.npz"), 'npz'),
    ('Taxonomy', os.path.join(BASE_FEATURE_DIR, "taxonomy_onehot_features.npz"), 'npz'),
    ('Segment', os.path.join(BASE_FEATURE_DIR, "segment_onehot_features.npz"), 'npz'),
]

# --- 特殊处理 Bacteria 节点类型以合并新旧细菌 ---
print("正在合并新旧细菌特征...")
combined_bacteria_features = None

if os.path.exists(original_bacteria_features_path):
    orig_bact_feats = np.load(original_bacteria_features_path)
    combined_bacteria_features = orig_bact_feats
    print(f"加载原始细菌特征：{orig_bact_feats.shape[0]} 个。")
else:
    print(f"警告：原始细菌特征文件 '{original_bacteria_features_path}' 未找到。")

if os.path.exists(new_bacteria_features_path):
    new_bact_feats = np.load(new_bacteria_features_path)
    if combined_bacteria_features is not None:
        combined_bacteria_features = np.vstack((combined_bacteria_features, new_bact_feats))
    else:
        combined_bacteria_features = new_bact_feats
    print(f"加载新细菌特征：{new_bact_feats.shape[0]} 个。")
else:
    print(f"警告：新细菌特征文件 '{new_bacteria_features_path}' 未找到。将不会合并新细菌。")

if combined_bacteria_features is not None:
    data['Bacteria'].x = torch.tensor(combined_bacteria_features, dtype=torch.float)
    print(f"合并后的细菌特征形状: {data['Bacteria'].x.shape}")
else:
    print("错误：无法为 'Bacteria' 类型加载任何特征。请检查文件路径。")
    # 如果没有任何细菌特征，可能需要退出或采取其他错误处理措施

# 处理其他节点类型
for ntype, path, ftype in tqdm(all_node_types_config, desc="加载其他节点类型特征"):
    if ntype == 'Bacteria': # Bacteria 已经特殊处理过了
        continue
    
    if path is None:
        print(f"警告：节点类型 '{ntype}' 没有指定特征文件路径。跳过。")
        continue

    if os.path.exists(path):
        if ftype == 'npy':
            data[ntype].x = torch.tensor(np.load(path), dtype=torch.float)
        else: # npz
            data[ntype].x = torch.tensor(sparse.load_npz(path).toarray(), dtype=torch.float)
    else:
        print(f"警告：节点类型 '{ntype}' 的特征文件 '{path}' 未找到。跳过。")

# ===== 加载边索引 =====
print("\n加载边索引...")
edges_dict_path = os.path.join(BASE_FEATURE_DIR, "edges_dict0725.pkl")
edge_index_dict = {}
if os.path.exists(edges_dict_path):
    with open(edges_dict_path, "rb") as f:
        edge_index_dict = pickle.load(f)
    print(f"加载边字典：{edges_dict_path}")
else:
    print(f"错误：边字典文件 '{edges_dict_path}' 未找到。无法构建边。")
    # 如果边文件丢失，可能需要退出或采取其他错误处理措施

# 注意：此处的边索引仍然是原始图中的局部ID。
# 如果新细菌需要与其他节点形成边，您需要在 edges_dict0725.pkl 中更新这些边的局部ID，
# 或者在 convert_to_hgb.py 之前添加这些边。
# 目前的修改仅确保细菌节点特征被正确合并。

for (src_type, rel_type, dst_type), edge_list in tqdm(edge_index_dict.items(), desc="加载边类型"):
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() # shape [2, num_edges]
    data[(src_type, rel_type, dst_type)].edge_index = edge_index

print("\n✅ HeteroData 构建完成。节点类型:")
for node_type in data.node_types:
    if hasattr(data[node_type], 'x') and data[node_type].x is not None: # 确保节点类型有特征数据
        print(f" - {node_type}: {data[node_type].x.shape}")
    else:
        print(f" - {node_type}: 无特征数据")

print("✅ 边类型:")
for edge_type in data.edge_types:
    print(f" - {edge_type}: {data[edge_type].edge_index.shape[1] if data[edge_type].edge_index is not None else 0} 条边")

# 可选：保存构建好的图对象
output_graph_path = "hetero_graph0810.pt"
torch.save(data, output_graph_path)
print(f"\n✅ 已保存 HeteroData 为 {output_graph_path}")