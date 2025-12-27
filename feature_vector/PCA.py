import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import joblib # To save/load the PCA model

# Define paths
original_graph_path = 'hetero_graph0725_standardized.pt'
new_bacteria_features_path = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/name_wang_features0810.npy'
output_reduced_features_path = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/name_wang_features0810_reduced_128.npy'
pca_model_path = 'bacteria_pca_model.pkl' # Path to save/load PCA model

TARGET_DIMENSION = 128
BACTERIA_NODE_TYPE = 'Bacteria'

print("--- Starting Feature Dimension Reduction ---")

# 1. 确认目标维度 (从原始细菌特征中获取，这一步仅为确认维度，不用于PCA训练)
original_bacteria_feature_dim = None
try:
    print(f"Loading original graph data from: {original_graph_path} to confirm target dimension...")
    original_data = torch.load(original_graph_path, weights_only=False) # 显式设置 weights_only=False

    if BACTERIA_NODE_TYPE in original_data.node_types and \
       hasattr(original_data[BACTERIA_NODE_TYPE], 'x') and \
       original_data[BACTERIA_NODE_TYPE].x is not None:
        
        original_bacteria_feature_dim = original_data[BACTERIA_NODE_TYPE].x.shape[1]
        print(f"确认目标维度 (原始细菌特征维度): {original_bacteria_feature_dim}")
        if original_bacteria_feature_dim != TARGET_DIMENSION:
            print(f"警告: 原始细菌特征维度 ({original_bacteria_feature_dim}) 与预设目标维度 ({TARGET_DIMENSION}) 不符。")
            print("将使用预设的 TARGET_DIMENSION 进行降维。")
    else:
        print(f"错误: 无法从 '{original_graph_path}' 中获取 '{BACTERIA_NODE_TYPE}' 特征来确认目标维度。")
        print("请手动设置 TARGET_DIMENSION 或确保文件正确。")

except FileNotFoundError:
    print(f"错误: 原始图文件 '{original_graph_path}' 未找到。请检查路径。")
except Exception as e:
    print(f"加载原始图文件时发生错误: {e}")

# 2. 加载新细菌特征 (717维，这是我们要降维的数据)
new_bacteria_features = None
try:
    print(f"\nLoading new bacteria features from: {new_bacteria_features_path}")
    if os.path.exists(new_bacteria_features_path):
        new_bacteria_features = np.load(new_bacteria_features_path)
        if new_bacteria_features.ndim == 2:
            print(f"新细菌特征已加载。形状: {new_bacteria_features.shape}")
        else:
            print(f"错误: 新细菌特征文件 '{new_bacteria_features_path}' 不是二维数组。形状: {new_bacteria_features.shape}")
            new_bacteria_features = None
    else:
        print(f"错误: 新细菌特征文件 '{new_bacteria_features_path}' 未找到。")

except Exception as e:
    print(f"加载新细菌特征文件时发生错误: {e}")

if new_bacteria_features is None:
    print("无法继续降维，因为未能成功加载新细菌特征。")
    exit()

# 3. 训练 PCA 模型 (使用717维的新细菌特征进行训练，将其降到128维)
print(f"\n训练 PCA 模型，将 {new_bacteria_features.shape[1]} 维特征降至 {TARGET_DIMENSION} 维...")
pca = PCA(n_components=TARGET_DIMENSION)
# *** 关键修改：用 new_bacteria_features 来训练 PCA ***
pca.fit(new_bacteria_features) 

# 保存训练好的 PCA 模型
joblib.dump(pca, pca_model_path)
print(f"PCA 模型已训练并保存到: {pca_model_path}")
print(f"训练后的 PCA 解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

# 4. 应用 PCA 变换到新细菌特征
print(f"\n正在对新细菌特征应用 PCA 变换...")
reduced_new_bacteria_features = pca.transform(new_bacteria_features)

print(f"降维后的新细菌特征形状: {reduced_new_bacteria_features.shape}")

# 5. 保存降维后的特征
try:
    np.save(output_reduced_features_path, reduced_new_bacteria_features)
    print(f"降维后的特征已保存到: {output_reduced_features_path}")
except Exception as e:
    print(f"保存降维特征时发生错误: {e}")

print("\n--- 特征降维完成 ---")
print(f"现在你可以在 'run_new.py' 脚本中使用 '{output_reduced_features_path}' 文件了。")