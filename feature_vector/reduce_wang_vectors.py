import numpy as np
from sklearn.decomposition import PCA
import os

# === 路径配置 ===
base_dir = "/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/"

# 向量文件路径
species_vec_path = os.path.join(base_dir, "species_wang_features_pca.npy")  # 已PCA
name_vec_path = os.path.join(base_dir, "name_wang_features.npy")

# 名称文件路径
species_name_path = os.path.join(base_dir, "species_wang_names.txt")
name_name_path = os.path.join(base_dir, "name_wang_names.txt")

# === 加载基准维度 ===
species_vec = np.load(species_vec_path)
target_dim = species_vec.shape[1]
print(f"📏 基准维度为: {target_dim}")

# === 准备总向量和总名称 ===
all_vecs = [species_vec]
all_names = []

with open(species_name_path, "r", encoding="utf-8") as f:
    all_names.extend([line.strip() for line in f])

# === 定义中间输出路径配置 ===
middle_outputs = {
    "name": {"vec": name_vec_path, "name": name_name_path}

}

# === 遍历 genus 和 family，处理、保存中间文件 ===
for rank in ["name"]:
    vec_path = middle_outputs[rank]["vec"]
    name_path = middle_outputs[rank]["name"]

    print(f"\n🔄 处理向量文件: {vec_path}")
    vec = np.load(vec_path)
    print(f"原始 shape: {vec.shape}")

    # 降维或补零
    if vec.shape[1] > target_dim:
        print("➡ PCA 降维中...")
        pca = PCA(n_components=target_dim)
        vec_reduced = pca.fit_transform(vec)
    elif vec.shape[1] < target_dim:
        print("➡ 列数不足，补零中...")
        vec_reduced = np.pad(vec, ((0, 0), (0, target_dim - vec.shape[1])), mode='constant')
    else:
        vec_reduced = vec
        print("✅ 维度一致，无需处理。")

    # === 保存降维后的特征文件 ===
    reduced_vec_path = os.path.join(base_dir, f"{rank}_wang_features_pca.npy")
    np.save(reduced_vec_path, vec_reduced)
    print(f"✅ 已保存降维向量: {reduced_vec_path} (shape: {vec_reduced.shape})")

    # === 加载并保存对应名称 ===
    with open(name_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f]
        assert len(names) == vec.shape[0], f"{rank} 名称数与向量行数不符"

    reduced_name_path = os.path.join(base_dir, f"{rank}_wang_names_pca.txt")
    with open(reduced_name_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
    print(f"✅ 已保存降维名称列表: {reduced_name_path}")

#     # === 加入到总量中
#     all_vecs.append(vec_reduced)
#     all_names.extend(names)

# # === 拼接所有向量 ===
# all_features = np.concatenate(all_vecs, axis=0)
# print(f"\n🔗 拼接后总向量 shape: {all_features.shape}")

# # === 保存总向量和总名称 ===
# out_vec_path = os.path.join(base_dir, "all_wang_features.npy")
# out_name_path = os.path.join(base_dir, "all_wang_names.txt")

# np.save(out_vec_path, all_features)
# with open(out_name_path, "w", encoding="utf-8") as f:
#     for name in all_names:
#         f.write(name + "\n")

# print(f"\n✅ 向量保存: {out_vec_path}")
# print(f"✅ 名称保存: {out_name_path}")
# print(f"🎉 总计细菌数量: {len(all_names)}")
