#!/usr/bin/env python3
"""
节点特征统一预处理脚本
将所有节点类型的特征向量统一到相同维度
"""

import torch
import logging
import numpy as np
from torch_geometric.data import HeteroData

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def standardize_node_features(hetero_data, target_dim=128):
    """
    标准化所有节点类型的特征到统一维度
    
    Args:
        hetero_data: 异质图数据
        target_dim: 目标特征维度
    """
    
    logging.info(f"开始统一节点特征维度到 {target_dim}")
    
    for node_type in hetero_data.node_types:
        num_nodes = hetero_data[node_type].num_nodes
        logging.info(f"处理节点类型: {node_type} (节点数: {num_nodes})")
        
        if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
            original_features = hetero_data[node_type].x
            orig_shape = original_features.shape
            logging.info(f"  原始特征形状: {orig_shape}")
            
            # 处理各种不规则的特征形状
            if len(orig_shape) == 2:
                num_nodes_feat, feat_dim = orig_shape
                
                if feat_dim == num_nodes:
                    # 方阵特征 (如 Gene: [10474, 10474])
                    logging.info(f"  检测到方阵特征，提取统计信息...")
                    
                    # 提取多种统计特征
                    diag_feat = torch.diag(original_features)  # 对角线
                    row_sum = original_features.sum(dim=1)     # 行和
                    row_mean = original_features.mean(dim=1)   # 行均值
                    row_std = original_features.std(dim=1)     # 行标准差
                    row_max = original_features.max(dim=1)[0]  # 行最大值
                    row_min = original_features.min(dim=1)[0]  # 行最小值
                    
                    # 合并统计特征
                    statistical_features = torch.stack([
                        diag_feat, row_sum, row_mean, row_std, row_max, row_min
                    ], dim=1)  # [num_nodes, 6]
                    
                    # 如果还需要更多维度，添加随机特征
                    if target_dim > 6:
                        torch.manual_seed(42)  # 确保可重复性
                        random_features = torch.randn(num_nodes, target_dim - 6) * 0.1
                        new_features = torch.cat([statistical_features, random_features], dim=1)
                    else:
                        new_features = statistical_features[:, :target_dim]
                        
                elif feat_dim > target_dim:
                    # 特征维度过大，使用PCA降维或截取
                    logging.info(f"  特征维度过大 ({feat_dim}), 降维到 {target_dim}")
                    
                    if feat_dim > 1000:
                        # 对于极大的维度，先采样再降维
                        sample_indices = torch.linspace(0, feat_dim-1, min(target_dim*2, feat_dim), dtype=torch.long)
                        sampled_features = original_features[:, sample_indices]
                        
                        if sampled_features.shape[1] > target_dim:
                            new_features = sampled_features[:, :target_dim]
                        else:
                            padding = torch.zeros(num_nodes, target_dim - sampled_features.shape[1])
                            new_features = torch.cat([sampled_features, padding], dim=1)
                    else:
                        # 直接截取前target_dim维
                        new_features = original_features[:, :target_dim]
                        
                elif feat_dim < target_dim:
                    # 特征维度不足，填充
                    logging.info(f"  特征维度不足 ({feat_dim}), 填充到 {target_dim}")
                    padding = torch.zeros(num_nodes, target_dim - feat_dim)
                    new_features = torch.cat([original_features, padding], dim=1)
                    
                else:
                    # 维度刚好匹配
                    new_features = original_features
                    
            else:
                # 处理一维或其他维度的特征
                logging.info(f"  处理非二维特征...")
                if len(orig_shape) == 1:
                    # 一维特征，扩展到二维
                    new_features = original_features.unsqueeze(-1).repeat(1, target_dim)
                else:
                    # 多维特征，展平后处理
                    flattened = original_features.view(num_nodes, -1)
                    if flattened.shape[1] > target_dim:
                        new_features = flattened[:, :target_dim]
                    else:
                        padding = torch.zeros(num_nodes, target_dim - flattened.shape[1])
                        new_features = torch.cat([flattened, padding], dim=1)
        else:
            # 没有特征的节点，创建随机特征
            logging.info(f"  节点类型 {node_type} 没有原始特征，创建随机特征")
            torch.manual_seed(hash(node_type) % 2**32)  # 基于节点类型的可重复随机数
            new_features = torch.randn(num_nodes, target_dim) * 0.1
        
        # 确保特征是float32类型
        new_features = new_features.float()
        
        # 归一化特征
        if new_features.std() > 0:
            new_features = (new_features - new_features.mean(dim=0)) / (new_features.std(dim=0) + 1e-8)
        
        # 更新特征
        hetero_data[node_type].x = new_features
        logging.info(f"  ✅ 更新后特征形状: {new_features.shape}")
    
    return hetero_data

def validate_features(hetero_data):
    """验证所有特征是否已正确标准化"""
    logging.info("验证特征标准化结果...")
    
    all_valid = True
    feature_dims = set()
    
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
            features = hetero_data[node_type].x
            shape = features.shape
            feature_dims.add(shape[1])
            
            # 检查是否有异常值
            has_nan = torch.isnan(features).any()
            has_inf = torch.isinf(features).any()
            
            if len(shape) != 2:
                logging.error(f"节点类型 {node_type} 特征不是二维: {shape}")
                all_valid = False
            elif has_nan:
                logging.error(f"节点类型 {node_type} 特征包含NaN")
                all_valid = False
            elif has_inf:
                logging.error(f"节点类型 {node_type} 特征包含Inf")
                all_valid = False
            else:
                logging.info(f"✅ {node_type}: {shape} (range: [{features.min():.4f}, {features.max():.4f}])")
    
    if len(feature_dims) == 1:
        logging.info(f"✅ 所有节点特征维度统一: {feature_dims.pop()}")
    else:
        logging.error(f"❌ 特征维度不统一: {feature_dims}")
        all_valid = False
    
    return all_valid

def main():
    """主函数"""
    print("🔧 节点特征统一预处理")
    print("="*50)
    
    # 加载原始数据
    data_path = "hetero_graph0810.pt"
    
    try:
        torch.serialization.add_safe_globals([torch.nn.Module])
        try:
            from torch_geometric.data.hetero_data import HeteroData
            torch.serialization.add_safe_globals([HeteroData])
        except ImportError:
            pass
        
        hetero_data = torch.load(data_path, map_location='cpu', weights_only=False)
        logging.info("✅ 原始数据加载成功")
        
    except Exception as e:
        logging.error(f"❌ 数据加载失败: {e}")
        return
    
    # 显示原始特征信息
    print("\n📊 原始特征信息:")
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
            shape = hetero_data[node_type].x.shape
            print(f"  {node_type}: {shape}")
        else:
            print(f"  {node_type}: 无特征")
    
    # 标准化特征
    target_dim = 128  # 统一特征维度
    hetero_data = standardize_node_features(hetero_data, target_dim)
    
    # 验证结果
    print(f"\n🔍 验证标准化结果:")
    is_valid = validate_features(hetero_data)
    
    if is_valid:
        # 保存标准化后的数据
        output_path = "hetero_graph0810_standardized.pt"
        torch.save(hetero_data, output_path)
        logging.info(f"✅ 标准化数据已保存到: {output_path}")
        
        print(f"\n🎉 特征标准化完成!")
        print(f"📁 标准化数据文件: hetero_graph0807_standardized.pt")
        print(f"🎯 统一特征维度: {target_dim}")
        print(f"📈 现在可以使用标准化后的数据进行训练")
        
    else:
        logging.error("❌ 特征标准化失败，请检查错误信息")

if __name__ == "__main__":
    main()
