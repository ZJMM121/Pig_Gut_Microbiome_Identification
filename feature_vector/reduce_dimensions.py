#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降维脚本：将细菌特征向量从原始维度降到128维
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def reduce_dimensions(input_file, output_file, target_dim=128):
    """
    使用PCA将特征向量降维到指定维度
    
    Args:
        input_file: 输入的.npy文件路径
        output_file: 输出的.npy文件路径
        target_dim: 目标维度，默认128
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    print(f"正在加载数据: {input_file}")
    # 加载原始特征向量
    features = np.load(input_file)
    print(f"原始数据形状: {features.shape}")
    
    # 检查是否需要降维
    if features.shape[1] <= target_dim:
        print(f"原始维度 {features.shape[1]} 已经小于或等于目标维度 {target_dim}，无需降维")
        return
    
    # 标准化数据
    print("正在标准化数据...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 使用PCA降维
    print(f"正在使用PCA降维到 {target_dim} 维...")
    pca = PCA(n_components=target_dim, random_state=42)
    features_reduced = pca.fit_transform(features_scaled)
    
    # 输出降维信息
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    print(f"降维后数据形状: {features_reduced.shape}")
    print(f"保留的方差比例: {cumulative_variance_ratio[-1]:.4f}")
    print(f"前10个主成分的方差比例: {explained_variance_ratio[:10]}")
    
    # 保存降维后的特征向量
    print(f"正在保存降维后的数据到: {output_file}")
    np.save(output_file, features_reduced)
    
    # 保存PCA模型和标准化器（可选）
    model_dir = os.path.dirname(output_file)
    pca_file = os.path.join(model_dir, 'pca_model.pkl')
    scaler_file = os.path.join(model_dir, 'scaler_model.pkl')
    
    import pickle
    with open(pca_file, 'wb') as f:
        pickle.dump(pca, f)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"PCA模型已保存到: {pca_file}")
    print(f"标准化器已保存到: {scaler_file}")
    print("降维完成！")

def main():
    # 输入和输出文件路径
    input_file = "/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/bacteria_wang_features0724.npy"
    output_file = "/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/bacteria_wang_features0724_128d.npy"
    
    # 执行降维
    reduce_dimensions(input_file, output_file, target_dim=128)

if __name__ == "__main__":
    main()