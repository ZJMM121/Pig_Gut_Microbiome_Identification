#!/usr/bin/env python3
"""
将PyTorch Geometric的HeteroData转换为HGB项目格式
用于链接预测任务，并将新的细菌数据合并到现有'Bacteria'类型中。
"""

import torch
import numpy as np
import os
import pickle
from collections import defaultdict, Counter
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

# Helper function for defaultdict default_factory
def _create_default_edge_stats():
    return {'count': 0, 'train': 0, 'valid': 0, 'test': 0}

def convert_hetero_data_to_hgb(data_path, output_dir):
    """
    将HeteroData转换为HGB格式，并将新的细菌数据合并到现有'Bacteria'类型中。
    
    Args:
        data_path: HeteroData文件路径 (应是构建图时只包含原始数据的图)
        output_dir: 输出目录
    """
    
    # 加载现有图数据 (此图应只包含原始的细菌数据)
    print("正在加载现有图数据...")
    data = torch.load(data_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 新细菌文件路径 ---
    NEW_BACTERIA_NAMES_PATH = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/name_wang_names0810.txt'
    # NEW_BACTERIA_FEATURES_PATH = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/name_wang_features_reduced_128.npy'
    # 注意：根据您的调试输出，新细菌特征实际上是包含在 hetero_graph0725.pt 的1694个细菌特征中的
    # 所以我们不再直接加载 new_bacteria_features.npy 并将其叠加。
    
    # 加载新细菌名称
    print("\n加载新的细菌名称...")
    new_bacteria_names = []
    num_new_bacteria_from_name_file = 0 # 统计从名称文件加载的新细菌数量

    if os.path.exists(NEW_BACTERIA_NAMES_PATH):
        try:
            with open(NEW_BACTERIA_NAMES_PATH, 'r', encoding='utf-8') as f:
                new_bacteria_names = [line.strip() for line in f if line.strip()]
            num_new_bacteria_from_name_file = len(new_bacteria_names)
            print(f"DEBUG: 加载了 {num_new_bacteria_from_name_file} 个新细菌名称。")
        except Exception as e:
            print(f"错误：加载新细菌名称文件失败: {e}")
            new_bacteria_names = []
            num_new_bacteria_from_name_file = 0
    else:
        print(f"警告：新细菌名称文件 '{NEW_BACTERIA_NAMES_PATH}' 未找到。将不会加载新细菌名称。")


    # 1. 处理节点信息 (合并新旧细菌)
    print("\n处理节点信息...")
    
    # 节点类型映射 (基于原始图的类型)
    node_type_map = {ntype: i for i, ntype in enumerate(data.node_types)}
    all_node_types_ordered = list(data.node_types) # 确保顺序与 node_type_map 匹配
    
    print(f"DEBUG: 原始节点类型映射: {node_type_map}")
    
    # 计算节点偏移量和数量
    node_shifts = {}
    node_counts = {}
    current_shift = 0
    
    all_node_features_list = [] # 存储所有节点类型的特征数组
    combined_bacteria_names_list = [] # 用于保存合并后的细菌名称

    for ntype in all_node_types_ordered:
        node_shifts[ntype] = current_shift
        
        if ntype == 'Bacteria': # 特殊处理 Bacteria 类型
            # 直接使用 PyG 图中已有的 Bacteria 特征，因为其数量 (1694) 是我们期望的最终数量
            combined_features = data['Bacteria'].x.numpy()
            print(f"DEBUG: 从PyG图中加载的'Bacteria'特征形状 (已包含合并数据): {combined_features.shape}")

            # 加载原始细菌的名称
            base_feature_vector_dir = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/'
            original_bacteria_names_path = os.path.join(base_feature_vector_dir, "bacteria_wang_names0724.txt")
            orig_bacteria_names = []
            if os.path.exists(original_bacteria_names_path):
                try:
                    with open(original_bacteria_names_path, 'r', encoding='utf-8') as f:
                        orig_bacteria_names = [line.strip() for line in f if line.strip()]
                    print(f"DEBUG: 加载原始细菌名称 (bacteria_wang_names0724.txt)：{len(orig_bacteria_names)} 个。")
                except Exception as e:
                    print(f"错误：加载原始细菌名称文件 '{original_bacteria_names_path}' 失败: {e}")
                    orig_bacteria_names = []
            else:
                print(f"警告：原始细菌名称文件 '{original_bacteria_names_path}' 未找到。")

            # 合并原始细菌名称和新细菌名称
            combined_bacteria_names_list = orig_bacteria_names + new_bacteria_names
            
            # 检查合并后的名称数量是否与特征数量匹配 (预期是 1098 + 596 = 1694)
            if combined_features.shape[0] != len(combined_bacteria_names_list):
                 print(f"⚠️ 严重警告：'Bacteria'的特征数量 ({combined_features.shape[0]}) 与名称数量 ({len(combined_bacteria_names_list)}) 不匹配！")
                 print("请检查'hetero_graph0725.pt'中'Bacteria'特征数量，以及原始细菌名称和新细菌名称文件是否正确。")
                 # 强制调整名称列表长度以匹配特征数量 (以特征为准)
                 if len(combined_bacteria_names_list) < combined_features.shape[0]:
                     combined_bacteria_names_list.extend([f"MissingBacteriaName_{i}" for i in range(len(combined_bacteria_names_list), combined_features.shape[0])])
                     print(f"DEBUG: 名称列表已填充至 {len(combined_bacteria_names_list)} 个。")
                 elif len(combined_bacteria_names_list) > combined_features.shape[0]:
                     combined_bacteria_names_list = combined_bacteria_names_list[:combined_features.shape[0]]
                     print(f"DEBUG: 名称列表已截断至 {len(combined_bacteria_names_list)} 个。")

            node_counts[ntype] = combined_features.shape[0]
            current_shift += node_counts[ntype]
            all_node_features_list.append(combined_features)
            print(f"DEBUG: '{ntype}' (细菌) 最终特征形状: {combined_features.shape}，对应总名称数: {len(combined_bacteria_names_list)}")

        else: # 处理其他节点类型
            count = data[ntype].x.shape[0]
            node_counts[ntype] = count
            current_shift += count
            all_node_features_list.append(data[ntype].x.numpy())
            print(f"DEBUG: '{ntype}' 特征形状: {data[ntype].x.shape}")

    print(f"DEBUG: 节点偏移量: {node_shifts}")
    print(f"DEBUG: 节点数量: {node_counts}")

    # --- 加载所有节点名称并按照 node_type_map 顺序填充 ---
    base_feature_vector_dir = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/'
    
    full_name_file_map = {
        'Gene': os.path.join(base_feature_vector_dir, 'gene_wang_similarity_name.txt'),
        'Metabolite': os.path.join(base_feature_vector_dir, 'metabolite_names0725.txt'),
        'Pathway': os.path.join(base_feature_vector_dir, 'pathway_names.txt'),
        'Trait': os.path.join(base_feature_vector_dir, 'trait_names0724.txt'),
        'PMID': os.path.join(base_feature_vector_dir, 'pmid_names0725.txt'),
        'Segment': os.path.join(base_feature_vector_dir, 'segment_names.txt'),
        'Taxonomy': os.path.join(base_feature_vector_dir, 'taxonomy_names.txt'),
        # 'ID' 类型如果也有名称文件，请在此处添加
        'ID': os.path.join(base_feature_vector_dir, 'id_names.txt'),
        # 'ID': 'path/to/id_names.txt',
    }
    
    original_node_types_data = [None] * len(all_node_types_ordered)
    print("\n加载所有节点名称...")

    for ntype_name_iter in all_node_types_ordered:
        type_idx = node_type_map[ntype_name_iter]
        num_expected_names = node_counts.get(ntype_name_iter, 0)

        if ntype_name_iter == 'Bacteria': # 使用我们已合并的细菌名称列表
            original_node_types_data[type_idx] = combined_bacteria_names_list
            print(f"DEBUG: - 'Bacteria': 成功加载 {len(combined_bacteria_names_list)} 个合并名称。")
            if len(combined_bacteria_names_list) != num_expected_names:
                print(f"⚠️ 警告：'Bacteria' 的名称数量 ({len(combined_bacteria_names_list)}) 与预期节点数量 ({num_expected_names}) 不匹配！ (已在之前处理)")
        elif ntype_name_iter in full_name_file_map: # 处理其他原始节点类型
            name_file_path = full_name_file_map[ntype_name_iter]
            if os.path.exists(name_file_path):
                try:
                    with open(name_file_path, 'r', encoding='utf-8') as f:
                        names = [line.strip() for line in f if line.strip()]
                    original_node_types_data[type_idx] = names
                    print(f"DEBUG: - {ntype_name_iter}: 加载了 {len(names)} 个名称。")
                    if len(names) != num_expected_names:
                        print(f"⚠️ 警告：{ntype_name_iter} 的名称数量 ({len(names)}) 与预期节点数量 ({num_expected_names}) 不匹配！")
                        # 强制调整名称列表长度以匹配特征数量 (以特征为准)
                        if len(names) < num_expected_names:
                            names.extend([f"{ntype_name_iter}_MissingName_{i}" for i in range(len(names), num_expected_names)])
                        elif len(names) > num_expected_names:
                            names = names[:num_expected_names]
                        original_node_types_data[type_idx] = names
                        print(f"DEBUG: {ntype_name_iter} 名称列表已调整至 {len(names)} 个。")
                except Exception as e:
                    print(f"DEBUG: - 错误：加载 {ntype_name_iter} 的名称文件 {name_file_path} 失败: {e}")
                    original_node_types_data[type_idx] = [f"{ntype_name_iter}_ErrorName_{i}" for i in range(num_expected_names)]
            else:
                print(f"DEBUG: - 警告：{ntype_name_iter} 的名称文件 {name_file_path} 未找到。将使用默认占位符。")
                original_node_types_data[type_idx] = [f"{ntype_name_iter}_Placeholder_{i}" for i in range(num_expected_names)]
        else: # 没有配置名称文件的节点类型
            print(f"DEBUG: - 警告：节点类型 '{ntype_name_iter}' 没有对应的名称文件配置。将使用默认占位符。")
            original_node_types_data[type_idx] = [f"{ntype_name_iter}_Placeholder_{i}" for i in range(num_expected_names)]

    # 2. 处理边信息
    print("\n处理边信息...")
    all_edges = []
    edge_type_map = {}
    edge_statistics = defaultdict(_create_default_edge_stats)
    current_edge_type_id = 0

    # 遍历现有图的边
    for (src_type, rel_type, dst_type), edge_index_tensor in data.edge_index_dict.items():
        edge_map_key = f"{src_type}-{rel_type}-{dst_type}"
        if edge_map_key not in edge_type_map:
            edge_type_map[edge_map_key] = current_edge_type_id
            current_edge_type_id += 1
        
        rel_id = edge_type_map[edge_map_key]
        
        # 将局部ID转换为全局ID
        src_global_ids = edge_index_tensor[0] + node_shifts[src_type]
        dst_global_ids = edge_index_tensor[1] + node_shifts[dst_type]
        
        num_edges = edge_index_tensor.shape[1]
        edge_statistics[edge_map_key]['count'] = num_edges # Total count
        
        for i in range(num_edges):
            all_edges.append((src_global_ids[i].item(), dst_global_ids[i].item(), rel_id, 1.0))

    # TODO: 如果您有新细菌与性状或其它节点之间的新边，您需要在这里将其添加到 all_edges 中
    # 这是非常重要的，因为 PyG图（hetero_graph0725.pt）中的边可能没有包含新细菌相关的链接。
    # 例如，如果新细菌与某些性状有新的关联，您需要从其他文件加载这些边，并将其转换为全局ID后加入 all_edges。
    # 示例：
    # from your_custom_loader import load_new_bacteria_to_trait_edges
    # new_bact_trait_edges = load_new_bacteria_to_trait_edges() # 假设这个函数返回 (src_local_bact_idx, dst_local_trait_idx, weight)
    # for src_local_bact, dst_local_trait, weight in new_bact_trait_edges:
    #     # 确保 src_local_bact 是新细菌在合并后细菌列表中的局部索引
    #     # 它的全局ID = (原始细菌数量) + (新细菌的局部ID)
    #     src_global = src_local_bact + len(orig_bacteria_names) # 这里假设新细菌的局部ID从0开始
    #     dst_global = dst_local_trait + node_shifts['Trait']
    #     rel_type_str = 'Bacteria-interacts_with-Trait' # 对应您的边类型
    #     if rel_type_str not in edge_type_map:
    #         edge_type_map[rel_type_str] = current_edge_type_id
    #         current_edge_type_id += 1
    #     rel_id = edge_type_map[rel_type_str]
    #     all_edges.append((src_global, dst_global, rel_id, weight))
    #     edge_statistics[rel_type_str]['count'] += 1


    print(f"DEBUG: 边类型映射: {edge_type_map}")
    print(f"DEBUG: 总边数 (当前包含): {len(all_edges)}")

    # 3. 划分训练集、验证集和测试集 (这里仅对现有边进行划分)
    print("\n划分数据集 (8:1:1)...")
    train_edges, temp_edges = train_test_split(all_edges, test_size=0.2, random_state=42)
    valid_edges, test_edges = train_test_split(temp_edges, test_size=0.5, random_state=42)

    # 更新边统计信息
    for edge_list, split_type in [(train_edges, 'train'), (valid_edges, 'valid'), (test_edges, 'test')]:
        for edge in edge_list:
            rel_id = edge[2]
            # 找到对应的 edge_map_key
            edge_map_key = next((k for k, v in edge_type_map.items() if v == rel_id), None)
            if edge_map_key:
                edge_statistics[edge_map_key][split_type] += 1

    print(f"DEBUG: 训练集边数: {len(train_edges)}")
    print(f"DEBUG: 验证集边数: {len(valid_edges)}")
    print(f"DEBUG: 测试集边数: {len(test_edges)}")

    # 4. 保存节点特征 (包含合并后的细菌特征)
    print("\n保存节点特征...")
    all_node_features = np.vstack(all_node_features_list)
    np.save(os.path.join(output_dir, 'feature.npy'), all_node_features)
    print(f"DEBUG: 节点特征保存为: {os.path.join(output_dir, 'feature.npy')}")

    # 5. 保存标签 (如果适用)
    print("保存标签 (如果适用)...")
    with open(os.path.join(output_dir, 'labels.dat'), 'w') as f:
        f.write("# No labels for this dataset\n")
    print(f"DEBUG: 标签文件保存为: {os.path.join(output_dir, 'labels.dat')}")

    # 6. 保存链接数据
    print("保存链接数据...")
    with open(os.path.join(output_dir, 'link.dat.train'), 'w', encoding='utf-8') as f:
        for edge in train_edges:
            src, dst, rel_id, weight = edge
            f.write(f"{src}\t{dst}\t{rel_id}\t{weight}\n")
            
    with open(os.path.join(output_dir, 'link.dat.valid'), 'w', encoding='utf-8') as f:
        for edge in valid_edges:
            src, dst, rel_id, weight = edge
            f.write(f"{src}\t{dst}\t{rel_id}\t{weight}\n")
            
    with open(os.path.join(output_dir, 'link.dat.test'), 'w', encoding='utf-8') as f:
        for edge in test_edges:
            src, dst, rel_id, weight = edge
            f.write(f"{src}\t{dst}\t{rel_id}\t{weight}\n")
    print(f"DEBUG: 链接文件保存到: {output_dir}")
            
    # 7. 保存元数据
    print("保存元数据...")
    metadata = {
        'node_type_map': node_type_map,
        'edge_type_map': edge_type_map,
        'node_shifts': node_shifts,
        'node_counts': node_counts,
        'edge_statistics': dict(edge_statistics),
        'original_node_types': original_node_types_data, # 现在这里包含了合并后的细菌名称列表
        'original_edge_types': data.edge_types, # 边类型名称 (e.g., ('Bacteria', 'interacts_with', 'Trait'))
        'total_nodes': sum(node_counts.values()),
        'total_edges': len(all_edges),
        'train_edges': len(train_edges),
        'valid_edges': len(valid_edges),
        'test_edges': len(test_edges)
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"转换完成！数据保存在: {output_dir}")
    print(f"节点类型数: {len(all_node_types_ordered)}")
    print(f"边类型数: {len(data.edge_types)}")
    print(f"总节点数: {metadata['total_nodes']}")
    print(f"总边数: {metadata['total_edges']}")

# 示例用法 (根据您的实际情况调整路径)
if __name__ == "__main__":
    # 这是您PyTorch Geometric图文件的路径
    hetero_data_path = 'hetero_graph0810_standardized.pt' 
    # 这是您希望保存HGB格式数据的目录
    output_directory = 'PGMKG_HGB0810' 
    
    convert_hetero_data_to_hgb(hetero_data_path, output_directory)