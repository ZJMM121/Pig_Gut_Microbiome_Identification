#!/usr/bin/env python3
"""
将PyTorch Geometric的HeteroData转换为HGB项目格式
用于链接预测任务
"""

import torch
import numpy as np
import os
import pickle
from collections import defaultdict, Counter
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

def convert_hetero_data_to_hgb(data_path, output_dir):
    """
    将HeteroData转换为HGB格式
    
    Args:
        data_path: HeteroData文件路径
        output_dir: 输出目录
    """
    
    # 加载数据
    print("正在加载数据...")
    data = torch.load(data_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 处理节点信息
    print("处理节点信息...")
    
    # 节点类型映射
    node_type_map = {ntype: i for i, ntype in enumerate(data.node_types)}
    print(f"节点类型映射: {node_type_map}")
    
    # 计算节点偏移量
    node_shifts = {}
    node_counts = {}
    current_shift = 0
    
    for ntype in data.node_types:
        node_shifts[ntype] = current_shift
        node_counts[ntype] = data[ntype].x.shape[0]
        current_shift += node_counts[ntype]
    
    print(f"节点偏移量: {node_shifts}")
    print(f"节点数量: {node_counts}")
    
    # 2. 生成node.dat文件
    print("生成node.dat文件...")
    with open(os.path.join(output_dir, 'node.dat'), 'w', encoding='utf-8') as f:
        for ntype in data.node_types:
            features = data[ntype].x.numpy()
            type_id = node_type_map[ntype]
            shift = node_shifts[ntype]
            
            for i in range(features.shape[0]):
                node_id = shift + i
                node_name = f"{ntype}_{i}"
                # 将特征转换为逗号分隔的字符串
                feature_str = ','.join([f"{x:.6f}" for x in features[i]])
                f.write(f"{node_id}\t{node_name}\t{type_id}\t{feature_str}\n")
    
    # 3. 处理边信息
    print("处理边信息...")
    
    # 边类型映射
    edge_type_map = {etype: i for i, etype in enumerate(data.edge_types)}
    print(f"边类型映射: {edge_type_map}")
    
    # 收集所有边
    all_edges = []
    edge_statistics = {}
    
    for etype in data.edge_types:
        edge_index = data[etype].edge_index.numpy()
        src_type, rel_type, dst_type = etype
        
        # 调整节点索引（加上偏移量）
        src_indices = edge_index[0] + node_shifts[src_type]
        dst_indices = edge_index[1] + node_shifts[dst_type]
        
        rel_id = edge_type_map[etype]
        edge_count = edge_index.shape[1]
        
        edge_statistics[etype] = {
            'count': edge_count,
            'src_type': src_type,
            'dst_type': dst_type,
            'rel_id': rel_id
        }
        
        # 添加边信息
        for i in range(edge_count):
            all_edges.append((src_indices[i], dst_indices[i], rel_id, 1.0))  # 权重设为1.0
    
    print(f"总边数: {len(all_edges)}")
    
    # 4. 数据集分割（训练/验证/测试）
    print("进行数据集分割...")
    
    # 按边类型分组
    edges_by_type = defaultdict(list)
    for edge in all_edges:
        src, dst, rel_id, weight = edge
        edges_by_type[rel_id].append(edge)
    
    train_edges = []
    valid_edges = []
    test_edges = []
    
    # 对每种边类型进行分割
    for rel_id, edges in edges_by_type.items():
        if len(edges) < 10:  # 如果边数太少，全部用于训练
            train_edges.extend(edges)
            continue
            
        # 70% 训练，15% 验证，15% 测试
        train_e, temp_e = train_test_split(edges, test_size=0.3, random_state=42)
        valid_e, test_e = train_test_split(temp_e, test_size=0.5, random_state=42)
        
        train_edges.extend(train_e)
        valid_edges.extend(valid_e)
        test_edges.extend(test_e)
    
    print(f"训练边数: {len(train_edges)}")
    print(f"验证边数: {len(valid_edges)}")
    print(f"测试边数: {len(test_edges)}")
    
    # 5. 生成link.dat文件（训练+验证数据）
    print("生成link.dat文件...")
    with open(os.path.join(output_dir, 'link.dat'), 'w', encoding='utf-8') as f:
        for edge in train_edges + valid_edges:
            src, dst, rel_id, weight = edge
            f.write(f"{src}\t{dst}\t{rel_id}\t{weight}\n")
    
    # 6. 生成link.dat.test文件（测试数据）
    print("生成link.dat.test文件...")
    with open(os.path.join(output_dir, 'link.dat.test'), 'w', encoding='utf-8') as f:
        for edge in test_edges:
            src, dst, rel_id, weight = edge
            f.write(f"{src}\t{dst}\t{rel_id}\t{weight}\n")
    
    # 7. 保存元数据
    print("保存元数据...")
    metadata = {
        'node_type_map': node_type_map,
        'edge_type_map': edge_type_map,
        'node_shifts': node_shifts,
        'node_counts': node_counts,
        'edge_statistics': edge_statistics,
        'original_node_types': data.node_types,
        'original_edge_types': data.edge_types,
        'total_nodes': sum(node_counts.values()),
        'total_edges': len(all_edges),
        'train_edges': len(train_edges),
        'valid_edges': len(valid_edges),
        'test_edges': len(test_edges)
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"转换完成！数据保存在: {output_dir}")
    print(f"节点类型数: {len(data.node_types)}")
    print(f"边类型数: {len(data.edge_types)}")
    print(f"总节点数: {metadata['total_nodes']}")
    print(f"总边数: {metadata['total_edges']}")
    
    return metadata

if __name__ == "__main__":
    # 设置路径
    data_path = "/user_data/yezy/zhangjm/Simple-HGN-PGMKG2-KD/data/hetero_graph0810_standardized.pt"
    output_dir = "PGMKG_HGB_format"
    
    # 执行转换
    metadata = convert_hetero_data_to_hgb(data_path, output_dir)
    
    print("\n转换后的文件结构:")
    print("- node.dat: 节点信息文件")
    print("- link.dat: 训练和验证边数据")
    print("- link.dat.test: 测试边数据")
    print("- metadata.pkl: 元数据信息")
