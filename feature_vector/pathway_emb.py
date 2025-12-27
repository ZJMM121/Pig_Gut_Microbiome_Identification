import pandas as pd
import numpy as np
from scipy import sparse

# 读取路径名称文件
df = pd.read_csv("/user_data/yezy/zhangjm/FE_kg/Merge/entity_names/Pathway_names.txt")
pathway_names = df['name'].unique().tolist()

# 生成稀疏 one-hot 矩阵
n = len(pathway_names)
one_hot_matrix = sparse.eye(n, dtype=np.float32, format='csr')

# 保存稀疏矩阵
sparse.save_npz("pathway_onehot_features.npz", one_hot_matrix)

with open("pathway_names.txt", "w") as f:
    for name in pathway_names:
        f.write(name + "\n")

print(f"保存完成：{n} 个 pathway 的稀疏 one-hot 特征")