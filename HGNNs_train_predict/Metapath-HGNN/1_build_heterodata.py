import torch
import numpy as np
import pickle
from torch_geometric.data import HeteroData
from scipy import sparse
from tqdm import tqdm
# 初始化 HeteroData 容器
data = HeteroData()

# ===== 加载节点特征 =====
print("加载节点特征...")
node_types = [
    ('Bacteria', "../feature_vector/bacteria_wang_features0724.npy", 'npy'),
    ('Gene', "../feature_vector/gene_wang_similarity.npy", 'npy'),
    ('Metabolite', "../feature_vector/metabolite_features0725.npy", 'npy'),
    ('Pathway', "../feature_vector/pathway_onehot_features.npz", 'npz'),
    ('Trait', "../feature_vector/trait_onehot_features0724.npz", 'npz'),
    ('PMID', "../feature_vector/pmid_onehot_features0725.npz", 'npz'),
    ('ID', "../feature_vector/id_onehot_features.npz", 'npz'),
    ('Taxonomy', "../feature_vector/taxonomy_onehot_features.npz", 'npz'),
    ('Segment', "../feature_vector/segment_onehot_features.npz", 'npz'),
]
for ntype, path, ftype in tqdm(node_types, desc="节点类型加载"):
    if ftype == 'npy':
        data[ntype].x = torch.tensor(np.load(path), dtype=torch.float)
    else:
        data[ntype].x = torch.tensor(sparse.load_npz(path).toarray(), dtype=torch.float)

# ===== 加载边索引 =====
print("加载边索引...")
with open("../feature_vector/edges_dict0725.pkl", "rb") as f:
    edge_index_dict = pickle.load(f)

for (src_type, rel_type, dst_type), edge_list in tqdm(edge_index_dict.items(), desc="边类型加载"):
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    data[(src_type, rel_type, dst_type)].edge_index = edge_index

print("✅ HeteroData 构建完成。节点类型:")
for node_type in data.node_types:
    print(f" - {node_type}: {data[node_type].x.shape}")

print("✅ 边类型:")
for edge_type in data.edge_types:
    print(f" - {edge_type}: {data[edge_type].edge_index.shape[1]} 条边")

# 可选：保存构建好的图对象
torch.save(data, "hetero_graph0725.pt")
print("✅ 已保存 HeteroData 为 hetero_graph0725.pt")
