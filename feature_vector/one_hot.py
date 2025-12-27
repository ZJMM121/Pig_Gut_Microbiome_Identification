import pandas as pd
import numpy as np
from scipy import sparse

# 实体类型及其文件路径（每行一个实体名）
entity_files = {
     "PMID": "/user_data/yezy/zhangjm/FE_kg/Merge/standardization/entity_names0724/PMID_names.txt",
    # "ID": "/user_data/yezy/zhangjm/FE_kg/Merge/entity_names/ID_names.txt",
    # "Segment": "/user_data/yezy/zhangjm/FE_kg/Merge/entity_names/Segment_names.txt",
    # "Trait": "/user_data/yezy/zhangjm/FE_kg/Merge/standardization/entity_names0724/Trait_names.txt",
    # "Taxonomy": "/user_data/yezy/zhangjm/FE_kg/Merge/entity_names/Taxonomy_names.txt"
}

for entity, path in entity_files.items():
    # 读取实体名
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    n = len(names)
    one_hot = sparse.eye(n, dtype=np.float32, format='csr')
    # 保存稀疏矩阵
    sparse.save_npz(f"{entity.lower()}_onehot_features0725.npz", one_hot)
    # 保存实体名
    with open(f"{entity.lower()}_names0725.txt", "w") as f2:
        for name in names:
            f2.write(name + "\n")
    print(f"{entity}: {n} 个实体的one-hot特征已保存。")