#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import argparse
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import dgl
import scipy.sparse as sp # Still needed for mat2tensor if it's generally used for other feature types
from tqdm import tqdm
import inspect # For debugging
import traceback # For detailed error info
import random # For setting random seed

# 添加父目录到sys.path
sys.path.append('../../') 

from utils.pytorchtools import EarlyStopping
# from utils.data import load_data # This will be replaced, so it's commented out
from GNN import myGAT, myGAT_Student

# --- Helper functions (kept as is) ---
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

# --- NEW: HGB Data Loading Function ---
def load_hgb_data_and_graph(hgb_data_dir, device):
    """
    加载 HGB 格式的数据和元数据，并构建 DGL 图。
    返回 DGL 图、特征列表、元数据和排序后的节点类型名称。
    """
    metadata_path = os.path.join(hgb_data_dir, 'metadata.pkl')
    feature_path = os.path.join(hgb_data_dir, 'feature.npy')
    
    print(f"正在从 {hgb_data_dir} 加载 HGB 数据...")

    # 加载元数据
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print("metadata.pkl 已加载。")
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：未找到 {metadata_path}。请确保 HGB 格式文件已生成。")

    # 加载所有节点特征
    all_node_features_np = np.load(feature_path)
    all_node_features = torch.tensor(all_node_features_np, dtype=torch.float).to(device)
    print(f"feature.npy 已加载。总节点特征形状: {all_node_features.shape}")

    # 根据 metadata 划分特征为列表
    features_list = []
    # 按照 node_type_map 中的索引顺序组织 features_list
    sorted_node_types = sorted(metadata['node_type_map'].keys(), key=lambda k: metadata['node_type_map'][k])
    
    current_idx = 0
    for ntype in sorted_node_types:
        count = metadata['node_counts'][ntype]
        features = all_node_features[current_idx : current_idx + count]
        features_list.append(features)
        current_idx += count
    
    print(f"已将所有节点特征按类型分离为 {len(features_list)} 个张量。")
    for i, f in enumerate(features_list):
        print(f"  类型 {sorted_node_types[i]} 特征形状: {f.shape}")

    # 构建 DGL 图 (从 link.dat 文件)
    # HGB link.dat 文件格式: src_global_id \t dst_global_id \t rel_type_id \t [weight]
    all_edges_src = []
    all_edges_dst = []
    all_edges_rel = []

    link_files = {
        'train': os.path.join(hgb_data_dir, 'link.dat.train'),
        'valid': os.path.join(hgb_data_dir, 'link.dat.valid'),
        'test': os.path.join(hgb_data_dir, 'link.dat.test')
    }

    # 用于获取训练、验证、测试集的边列表
    train_pos_edges = defaultdict(lambda: [[], []]) # rel_id -> [src_list, dst_list]
    valid_pos_edges = defaultdict(lambda: [[], []])
    test_pos_edges = defaultdict(lambda: [[], []])

    for split_name, link_file_path in link_files.items():
        if os.path.exists(link_file_path):
            print(f"正在读取边文件: {link_file_path}")
            with open(link_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        src = int(parts[0])
                        dst = int(parts[1])
                        rel_id = int(parts[2])
                        all_edges_src.append(src)
                        all_edges_dst.append(dst)
                        all_edges_rel.append(rel_id)

                        # Populate pos_edges for train/valid/test splits
                        if split_name == 'train':
                            train_pos_edges[rel_id][0].append(src)
                            train_pos_edges[rel_id][1].append(dst)
                        elif split_name == 'valid':
                            valid_pos_edges[rel_id][0].append(src)
                            valid_pos_edges[rel_id][1].append(dst)
                        elif split_name == 'test':
                            test_pos_edges[rel_id][0].append(src)
                            test_pos_edges[rel_id][1].append(dst)
        else:
            print(f"警告: 边文件 '{link_file_path}' 未找到。")

    if not all_edges_src:
        raise ValueError("错误: 未找到任何边数据来构建 DGL 图。请检查 link.dat 文件。")

    total_nodes = metadata['total_nodes']
    g = dgl.graph((torch.tensor(all_edges_src), torch.tensor(all_edges_dst)), num_nodes=total_nodes)
    g.edata['type'] = torch.tensor(all_edges_rel) # 存储关系类型作为边特征
    
    print(f"DGL 图已构建。总节点数: {g.num_nodes()}，总边数: {g.num_edges()}")
    
    # 将图移动到设备上并添加自循环
    g = g.to(device)
    g = dgl.add_self_loop(g)
    print("已为DGL图添加自循环。")

    # 返回加载的数据，用于替代原有的 dl 对象
    # 这里我们返回 train_pos_edges, valid_pos_edges, test_pos_edges
    # 负采样需要独立实现或调整
    data_for_loops = {
        'train_pos': train_pos_edges,
        'valid_pos': valid_pos_edges,
        'test_pos': test_pos_edges, # Test positive edges for evaluation
        'edge_types_in_metadata': list(metadata['edge_type_map'].values()) # All relation IDs
    }

    return g, features_list, metadata, sorted_node_types, data_for_loops # Return graph g


# --- NEW: Helper for Negative Sampling (Simplified for this context) ---
# This is a simplified version. In a real scenario, you might want more sophisticated
# negative sampling (e.g., node type aware, or based on specific rules).
# This version generates random negative samples from all possible nodes.
# You might need to adjust this based on your specific 'dl.get_train_neg' logic.
def get_random_neg_samples(pos_heads, pos_tails, num_neg_samples, total_nodes, device, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    neg_heads = []
    neg_tails = []
    # Simple random negative sampling: replace tail
    # More advanced: based on node types, or by using relation-specific negative sampling
    for i in range(len(pos_heads)):
        neg_heads.append(pos_heads[i])
        # Randomly sample a node that is NOT the positive tail
        while True:
            rand_tail = np.random.randint(0, total_nodes)
            if rand_tail != pos_tails[i]:
                neg_tails.append(rand_tail)
                break
    return np.array(neg_heads), np.array(neg_tails)


# --- 简化后的 predict_new_associations 函数 (不需要 new_bacteria_names, original_node_types) ---
def predict_new_associations(args, student_net,
                             features_list, g, device, # Add 'g' here as it's passed from run_model_DBLP
                             trait_node_type_name, node_shifts, node_counts,
                             association_relation_type_id,
                             new_bacteria_global_ids_start, num_new_bacteria,
                             relation_type_name_for_display):
    """
    使用训练好的学生模型预测新细菌与所有性状的关联性，并返回Top N结果。只输出ID。
    """
    print(f"\n--- 开始预测新的细菌与所有性状的关联性 ({relation_type_name_for_display}) ---")

    new_bacteria_global_ids = np.arange(new_bacteria_global_ids_start, new_bacteria_global_ids_start + num_new_bacteria)
    print(f"将要预测关联性的新细菌数量: {num_new_bacteria}")
    print(f"关联关系类型ID (mid): {association_relation_type_id}")

    trait_start_id = node_shifts[trait_node_type_name]
    trait_count = node_counts[trait_node_type_name]
    all_trait_global_ids = np.arange(trait_start_id, trait_start_id + trait_count)
    print(f"总共有 {trait_count} 个性状节点，其全局ID范围为: [{trait_start_id}, {trait_start_id + trait_count})")
    
    # 调试信息：确认Bacteria的ID范围
    bacteria_node_type_name = 'Bacteria'
    bacteria_start_id = node_shifts[bacteria_node_type_name]
    bacteria_count = node_counts[bacteria_node_type_name]
    print(f"合并后的Bacteria节点ID范围 (包含原始和新): [{bacteria_start_id}, {bacteria_start_id + bacteria_count})")

    student_net.eval()
    predictions_per_trait = defaultdict(list) 

    with torch.no_grad():
        batch_size = args.batch_size
        for i in tqdm(range(0, num_new_bacteria, batch_size), desc="预测批次"):
            batch_bacteria_global_ids = new_bacteria_global_ids[i:i + batch_size]
            
            left_nodes_batch = np.repeat(batch_bacteria_global_ids, trait_count)
            right_nodes_batch = np.tile(all_trait_global_ids, len(batch_bacteria_global_ids))
            relation_types_batch_np = np.full(left_nodes_batch.shape[0], fill_value=association_relation_type_id, dtype=np.int32)

            if left_nodes_batch.shape[0] == 0:
                print(f"Warning: Empty batch for prediction. Skipping.")
                continue

            left_nodes_t = torch.tensor(left_nodes_batch, dtype=torch.long).to(device)
            right_nodes_t = torch.tensor(right_nodes_batch, dtype=torch.long).to(device)
            relation_types_t = torch.tensor(relation_types_batch_np, dtype=torch.long).to(device)

            # --- PREVIOUSLY: student_net(g, features_list, ...)
            # --- NOW: student_net(features_list, ...) as 'g' is handled in model's __init__
            scores_batch = student_net(features_list, left_nodes_t, right_nodes_t, relation_types_t, return_logits=False)
            
            current_batch_idx = 0
            for b_idx in range(len(batch_bacteria_global_ids)):
                current_bacteria_id = batch_bacteria_global_ids[b_idx]
                for t_idx in range(trait_count):
                    current_trait_id = all_trait_global_ids[t_idx]
                    score = scores_batch[current_batch_idx].item()
                    predictions_per_trait[current_trait_id].append((current_bacteria_id, score))
                    current_batch_idx += 1

    top_n_bacteria = 262
    
    # Output to a file for structured saving
    output_predictions_file = os.path.join(args.output_dir, f"predictions_{relation_type_name_for_display}_{args.run}.txt")
    print(f"\n--- 正在保存预测结果到: {output_predictions_file} ---")
    
    all_top_associations = [] # To store all top N associations for all traits

    with open(output_predictions_file, 'w', encoding='utf-8') as f_out:
        f_out.write(f"预测关系类型: {relation_type_name_for_display}\n")
        f_out.write(f"每个性状Top {top_n_bacteria}个新细菌关联结果:\n")
        
        for trait_id in sorted(predictions_per_trait.keys()):
            predictions_per_trait[trait_id].sort(key=lambda x: x[1], reverse=True)
            top_bacteria_for_trait = predictions_per_trait[trait_id][:top_n_bacteria]
            
            f_out.write(f"\n------------------------------------------------------------------\n")
            f_out.write(f"性状ID: {trait_id}\n")
            f_out.write("{:<15} {:<15}\n".format("新细菌ID", "关联分数"))
            f_out.write("------------------------------------------------------------------\n")
            
            for bacteria_id, score in top_bacteria_for_trait:
                f_out.write("{:<15} {:<15.6f}\n".format(bacteria_id, score))
                all_top_associations.append((trait_id, bacteria_id, score))
            f_out.write("------------------------------------------------------------------\n")
            
    print(f"预测结果已保存到: {output_predictions_file}")
    return all_top_associations


# --- Knowledge Distillation Loss Function (kept as is) ---
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
    student_logits_2d = torch.stack([torch.zeros_like(student_logits), student_logits], dim=1)
    teacher_logits_2d = torch.stack([torch.zeros_like(teacher_logits), teacher_logits], dim=1)
    soft_target_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits_2d / temperature, dim=1),
        F.softmax(teacher_logits_2d / temperature, dim=1)
    ) * (temperature * temperature)
    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_target_loss
    return total_loss

def run_model_DBLP(args):
    # --- 设置随机种子确保可复现性 ---
    def set_random_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dgl.seed(seed)
    
    # 设置全局随机种子
    global_seed = args.run * 1000 + 42  # 使用run参数确保每次运行的随机种子不同但可复现
    set_random_seeds(global_seed)
    print(f"设置全局随机种子为: {global_seed}")
    
    # --- 日志文件设置 ---
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"prediction_log_{args.dataset}_{args.run}_{int(time.time())}.txt")
    print(f"所有输出将被记录到: {log_file_path}")

    # Output directory for prediction results
    args.output_dir = './predictions0812'
    os.makedirs(args.output_dir, exist_ok=True)

    original_stdout = sys.stdout # 保存原始 stdout
    sys.stdout = open(log_file_path, 'w', encoding='utf-8', buffering=1) # 1 为行缓冲

    print(f"Starting prediction run at: {time.ctime()}")
    print(f"Arguments: {args}")

    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # --- NEW: Load data from HGB format ---
        # Update this path to where your convert_to_hgb.py outputs the HGB data
        hgb_data_dir = '/user_data/yezy/zhangjm/Simple-HGN-PGMKG2-KD/data/PGMKG_HGB0810'
        g, features_list, metadata, sorted_node_types, data_for_loops = \
            load_hgb_data_and_graph(hgb_data_dir, device) # <-- Now returns graph 'g'
        
        node_type_map = metadata['node_type_map']
        edge_type_map = metadata['edge_type_map']
        node_shifts = metadata['node_shifts']
        node_counts = metadata['node_counts']
        original_node_types = metadata['original_node_types'] # For ID to name mapping later (not in this script)
        original_edge_types = metadata['original_edge_types']
        total_nodes_in_graph = metadata['total_nodes'] # Total nodes after all merges
        
        print(f"Loaded metadata from: {os.path.join(hgb_data_dir, 'metadata.pkl')}")
        print(f"Node Type Map: {node_type_map}")
        print(f"Edge Type Map: {edge_type_map}")
        print(f"Node Shifts: {node_shifts}")
        print(f"Node Counts: {node_counts}")
        print(f"Total nodes in graph: {total_nodes_in_graph}")

        # Ensure required node types exist
        bacteria_node_type_name = 'Bacteria' 
        if bacteria_node_type_name not in node_type_map:
            raise ValueError(f"'{bacteria_node_type_name}' 节点类型未在 metadata.pkl 中找到。")
        
        trait_node_type_name = 'Trait' 
        if trait_node_type_name not in node_type_map:
            raise ValueError(f"'{trait_node_type_name}' 节点类型未在 metadata.pkl 中找到。")

        # Define target relation types using the edge_type_map
        # IMPORTANT: Convert these to the string format used in metadata.pkl
        all_target_relation_tuples_str = [
            'Bacteria-related_to-Trait',
            'Bacteria-positive_relate-Trait',
            'Bacteria-negative_relate-Trait'
        ]
        
        # Map relation strings to their integer IDs using edge_type_map
        all_target_relation_type_ids = {}
        for rel_str in all_target_relation_tuples_str: # <-- 遍历字符串格式
            if rel_str in edge_type_map: # <-- 现在可以正确匹配了
                # 存储时仍然用元组作为键，方便后续显示和处理，但查找时用字符串
                original_tuple = tuple(rel_str.split('-'))
                all_target_relation_type_ids[original_tuple] = edge_type_map[rel_str]
            else:
                print(f"警告: 边类型 '{rel_str}' 未在 metadata.pkl 中找到。跳过此关系类型的预测。")

        # --- Determine new bacteria global IDs ---
        # Get count of original bacteria from its name file
        original_bacteria_names_path = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/bacteria_wang_names0724.txt'
        original_bacteria_count_actual = 0
        if os.path.exists(original_bacteria_names_path):
            with open(original_bacteria_names_path, 'r', encoding='utf-8') as f:
                original_bacteria_count_actual = len([line.strip() for line in f if line.strip()])
            print(f"从 '{original_bacteria_names_path}' 确定原始细菌数量: {original_bacteria_count_actual}")
        else:
            print(f"警告: 原始细菌名称文件 '{original_bacteria_names_path}' 未找到。无法准确确定新细菌的起始ID。")
            # Fallback: assume all bacteria in the combined list beyond original_node_types['Bacteria'] are 'new'
            # This is less robust but might work if the count is always consistent
            bacteria_node_idx = node_type_map[bacteria_node_type_name]
            original_bacteria_count_actual = len(original_node_types[bacteria_node_idx])
            print(f"DEBUG: Fallback to original_node_types to get original bacteria count: {original_bacteria_count_actual}")


        # Get count of new bacteria from its feature file (as it's already reduced to 128D)
        new_bacteria_features_path = '/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/name_wang_features0810_reduced_128.npy'
        new_bacteria_features_np = None
        num_new_bacteria_to_predict = 0

        if os.path.exists(new_bacteria_features_path):
            new_bacteria_features_np = np.load(new_bacteria_features_path)
            num_new_bacteria_to_predict = new_bacteria_features_np.shape[0]
            print(f"成功加载 {num_new_bacteria_to_predict} 个新细菌的特征 (已是128维)。")
        else:
            print(f"错误：未找到新细菌特征文件: {new_bacteria_features_path}。无法进行新细菌预测。")
            # If no features, can't predict. Exit or provide a mock prediction.
            return
        
        # Calculate global ID start for new bacteria within the combined 'Bacteria' node type
        # This assumes 'Bacteria' nodes are ordered: original_bacteria then new_bacteria
        bacteria_node_global_shift = node_shifts[bacteria_node_type_name]
        new_bacteria_global_ids_start = bacteria_node_global_shift + original_bacteria_count_actual
        print(f"新细菌全局ID范围预期为: [{new_bacteria_global_ids_start}, {new_bacteria_global_ids_start + num_new_bacteria_to_predict})")


        # --- Model Initialization ---
        # in_dims for teacher and student models will be derived from the loaded features_list
        in_dims_for_models = [f.shape[1] for f in features_list]
        num_etypes_for_model = len(edge_type_map) # Use the total count of edge types from metadata
        print(f"模型边类型数量 (来自metadata): {num_etypes_for_model}")
        print(f"DEBUG: in_dims for models: {in_dims_for_models}")

        final_student_net = None
        edge_types_to_evaluate = data_for_loops['edge_types_in_metadata'] # Use all edge types found in metadata
        res_random = defaultdict(float)
        total_evaluated_edge_types = 0

        for i, test_edge_type_id in enumerate(edge_types_to_evaluate):
            # Convert edge type ID back to tuple for logging if needed, or just use ID
            current_rel_tuple = None
            for k, v in edge_type_map.items():
                if v == test_edge_type_id:
                    current_rel_tuple = k
                    break

            print(f"\n{'='*50}")
            print(f"处理边类型 {i+1}/{len(edge_types_to_evaluate)} (ID: {test_edge_type_id}, Tuple: {current_rel_tuple})")
            print(f"{'='*50}")
            
            # --- Get train/valid positive samples for current edge type from data_for_loops ---
            train_pos_head = np.array(data_for_loops['train_pos'][test_edge_type_id][0])
            train_pos_tail = np.array(data_for_loops['train_pos'][test_edge_type_id][1])
            valid_pos_head = np.array(data_for_loops['valid_pos'][test_edge_type_id][0])
            valid_pos_tail = np.array(data_for_loops['valid_pos'][test_edge_type_id][1])

            # Skip if no training data for this edge type
            if len(train_pos_head) == 0:
                print(f"警告: 边类型 {test_edge_type_id} 无训练数据。跳过此边类型的训练和评估。")
                continue

            # --- Teacher Model Training ---
            # --- FIX: Add 'g' as the first argument to myGAT __init__ ---
            teacher_net = myGAT(g, # ADDED: Pass the DGL graph 'g' here
                                 args.edge_feats, num_etypes_for_model, 
                                 in_dims_for_models, 
                                 args.hidden_dim, 1, args.num_layers, [args.num_heads] * args.num_layers + [args.num_heads],
                                 F.elu, args.dropout, args.dropout, args.slope, args.residual,
                                 args.residual_att, decode=args.decoder)
            teacher_net.to(device)
            teacher_optimizer = torch.optim.Adam(teacher_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            teacher_checkpoint_path = 'checkpoint/teacher_checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers, test_edge_type_id)
            teacher_early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                    save_path=teacher_checkpoint_path)
            teacher_loss_func = nn.BCELoss()

            print("--- 检查 Teacher Model forward 签名 ---")
            print(inspect.signature(teacher_net.forward))
            print("---------------------------------------")
            for epoch in range(args.epoch):
                # Get negative samples for training
                epoch_seed = global_seed + epoch * 100  # 为每个epoch设置不同但可复现的种子
                train_neg_head, train_neg_tail = get_random_neg_samples(train_pos_head, train_pos_tail, len(train_pos_head), total_nodes_in_graph, device, random_seed=epoch_seed)
                
                train_idx = np.arange(len(train_pos_head))
                np.random.seed(epoch_seed)  # 确保shuffle结果可复现
                np.random.shuffle(train_idx)
                batch_size = args.batch_size
                for start in range(0, len(train_pos_head), args.batch_size):
                    teacher_net.train()
                    left_batch = np.concatenate([train_pos_head[train_idx[start:start+batch_size]], train_neg_head[train_idx[start:start+batch_size]]])
                    right_batch = np.concatenate([train_pos_tail[train_idx[start:start+batch_size]], train_neg_tail[train_idx[start:start+batch_size]]])
                    mid_batch_np = np.full(left_batch.shape[0], fill_value=test_edge_type_id, dtype=np.int32)
                    
                    if left_batch.shape[0] == 0:
                        continue

                    mid_batch_t = torch.tensor(mid_batch_np, dtype=torch.long).to(device)
                    labels_batch = torch.FloatTensor(np.concatenate([np.ones(train_pos_head[train_idx[start:start+batch_size]].shape[0]), np.zeros(train_neg_head[train_idx[start:start+batch_size]].shape[0])])).to(device)

                    try:
                        # --- FIX: Removed 'g' from forward call ---
                        logits = teacher_net(features_list, left_batch, right_batch, mid_batch_t, return_logits=True)

                    except TypeError as e:
                        print(f"\n--- 捕获到 TypeError！ ---")
                        print(f"错误信息: {e}")
                        traceback.print_exc() # 打印完整的堆栈跟踪
                        print("---------------------------\n")
                        exit(1) # 错误发生后退出程序
                    logp = F.sigmoid(logits)
                    train_loss = teacher_loss_func(logp, labels_batch)

                    teacher_optimizer.zero_grad()
                    train_loss.backward()
                    teacher_optimizer.step()
                    
                # Validation for Teacher model
                teacher_net.eval()
                with torch.no_grad():
                    if len(valid_pos_head) == 0:
                        print(f"Warning: Teacher model validation: No validation data for edge type {test_edge_type_id} in epoch {epoch}. Skipping validation.")
                        break # Break epoch loop if no validation data

                    valid_neg_head, valid_neg_tail = get_random_neg_samples(valid_pos_head, valid_pos_tail, len(valid_pos_head), total_nodes_in_graph, device, random_seed=epoch_seed)
                    left_val = np.concatenate([valid_pos_head, valid_neg_head])
                    right_val = np.concatenate([valid_pos_tail, valid_neg_tail])
                    mid_val_np = np.full(left_val.shape[0], fill_value=test_edge_type_id, dtype=np.int32)
                    
                    mid_val_t = torch.tensor(mid_val_np, dtype=torch.long).to(device)
                    labels_val = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                    
                    # --- FIX: Removed 'g' from forward call ---
                    logits_val = teacher_net(features_list, left_val, right_val, mid_val_t, return_logits=True)
                    logp_val = F.sigmoid(logits_val)
                    val_loss = teacher_loss_func(logp_val, labels_val)
                
                teacher_early_stopping(val_loss, teacher_net)
                if teacher_early_stopping.early_stop:
                    break
            
            if os.path.exists(teacher_checkpoint_path):
                teacher_net.load_state_dict(torch.load(teacher_checkpoint_path))
                teacher_net.eval()
                print("\n--- Teacher Model Training Complete ---")
            else:
                print(f"Warning: No teacher checkpoint found for edge type {test_edge_type_id}. Skipping student training and evaluation for this edge type.")
                continue

            # --- Student Model Training (Knowledge Distillation) ---
            print("\n--- Training Student Model (Knowledge Distillation) ---")
            student_num_hidden = args.hidden_dim // 2
            student_num_layers = args.num_layers - 1 if args.num_layers > 1 else 1
            student_heads = [args.num_heads] * student_num_layers + [args.num_heads]

            # --- FIX: Add 'g' as the first argument to myGAT_Student __init__ ---
            student_net = myGAT_Student(g, # ADDED: Pass the DGL graph 'g' here
                                        args.edge_feats, num_etypes_for_model, in_dims_for_models, 
                                        student_num_hidden, 1, student_num_layers, student_heads,
                                        F.elu, args.dropout, args.dropout, args.slope, args.residual,
                                        args.residual_att, decode=args.decoder)
            student_net.to(device)
            print(f"DEBUG: Student myGAT initialized with in_dims: {in_dims_for_models} and hidden_dim: {student_num_hidden}")

            student_optimizer = torch.optim.Adam(student_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            student_checkpoint_path = 'checkpoint/student_checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers, test_edge_type_id)
            student_early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                    save_path=student_checkpoint_path)
            
            temperature = args.temperature
            alpha = args.alpha

            for epoch in range(args.epoch):
                epoch_seed = global_seed + epoch * 100 + test_edge_type_id * 10000  # 为学生模型设置不同的种子基数
                train_neg_head, train_neg_tail = get_random_neg_samples(train_pos_head, train_pos_tail, len(train_pos_head), total_nodes_in_graph, device, random_seed=epoch_seed)
                
                train_idx = np.arange(len(train_pos_head))
                np.random.seed(epoch_seed)  # 确保shuffle结果可复现
                np.random.shuffle(train_idx)
                batch_size = args.batch_size

                for start in range(0, len(train_pos_head), args.batch_size):
                    student_net.train()
                    teacher_net.eval()

                    left_batch = np.concatenate([train_pos_head[train_idx[start:start+batch_size]], train_neg_head[train_idx[start:start+batch_size]]])
                    right_batch = np.concatenate([train_pos_tail[train_idx[start:start+batch_size]], train_neg_tail[train_idx[start:start+batch_size]]])
                    mid_batch_np = np.full(left_batch.shape[0], fill_value=test_edge_type_id, dtype=np.int32)
                    
                    if left_batch.shape[0] == 0:
                        continue

                    mid_batch_t = torch.tensor(mid_batch_np, dtype=torch.long).to(device)
                    labels_batch = torch.FloatTensor(np.concatenate([np.ones(train_pos_head[train_idx[start:start+batch_size]].shape[0]), np.zeros(train_neg_head[train_idx[start:start+batch_size]].shape[0])])).to(device)

                    with torch.no_grad():
                        # --- FIX: Removed 'g' from forward call ---
                        teacher_logits = teacher_net(features_list, left_batch, right_batch, mid_batch_t, return_logits=True)

                    # --- FIX: Removed 'g' from forward call ---
                    student_logits = student_net(features_list, left_batch, right_batch, mid_batch_t, return_logits=True) 

                    train_loss = distillation_loss(student_logits, teacher_logits, labels_batch, temperature, alpha)

                    student_optimizer.zero_grad()
                    train_loss.backward()
                    student_optimizer.step()
                    
                # Validation for Student model
                student_net.eval()
                with torch.no_grad():
                    if len(valid_pos_head) == 0:
                        print(f"Warning: Student model validation: No validation data for edge type {test_edge_type_id} in epoch {epoch}. Skipping validation.")
                        break # Break epoch loop if no validation data
                    
                    valid_neg_head, valid_neg_tail = get_random_neg_samples(valid_pos_head, valid_pos_tail, len(valid_pos_head), total_nodes_in_graph, device, random_seed=epoch_seed)
                    left_val = np.concatenate([valid_pos_head, valid_neg_head])
                    right_val = np.concatenate([valid_pos_tail, valid_neg_tail])
                    mid_val_np = np.full(left_val.shape[0], fill_value=test_edge_type_id, dtype=np.int32)
                    
                    mid_val_t = torch.tensor(mid_val_np, dtype=torch.long).to(device)
                    labels_val = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                    
                    # --- FIX: Removed 'g' from forward call ---
                    student_val_logits = student_net(features_list, left_val, right_val, mid_val_t, return_logits=True)
                    val_loss = F.binary_cross_entropy_with_logits(student_val_logits, labels_val)
                
                student_early_stopping(val_loss, student_net)
                if student_early_stopping.early_stop:
                    break
            
            if os.path.exists(student_checkpoint_path):
                student_net.load_state_dict(torch.load(student_checkpoint_path))
                student_net.eval()
                print("\n--- Student Model Training Complete ---")
            else:
                print(f"Warning: No student checkpoint found for edge type {test_edge_type_id}. Skipping evaluation for this edge type.")
                continue

            final_student_net = student_net
            # 同理，检查学生模型
            if final_student_net is not None:
                print("--- 检查 Student Model forward 签名 ---")
                print(inspect.signature(final_student_net.forward))
                print("---------------------------------------")
            # --- Evaluate Student Model (using test data from HGB) ---
            print(f"\n开始测试学生模型针对边类型 {test_edge_type_id}...")
            with torch.no_grad():
                test_pos_head = np.array(data_for_loops['test_pos'][test_edge_type_id][0])
                test_pos_tail = np.array(data_for_loops['test_pos'][test_edge_type_id][1])

                if len(test_pos_head) == 0:
                    print(f"Warning: Student model testing: No test data for edge type {test_edge_type_id}. Skipping evaluation for this edge type.")
                    continue

                test_neg_head, test_neg_tail = get_random_neg_samples(test_pos_head, test_pos_tail, len(test_pos_head), total_nodes_in_graph, device, random_seed=global_seed+test_edge_type_id)
                
                left_test = np.concatenate([test_pos_head, test_neg_head])
                right_test = np.concatenate([test_pos_tail, test_neg_tail])
                mid_test_np = np.full(left_test.shape[0], fill_value=test_edge_type_id, dtype=np.int32)
                
                mid_test_t = torch.tensor(mid_test_np, dtype=torch.long).to(device)
                labels_test = torch.FloatTensor(np.concatenate([np.ones(test_pos_head.shape[0]), np.zeros(test_neg_head.shape[0])])).to(device)
                
                # --- FIX: Removed 'g' from forward call ---
                pred = student_net(features_list, left_test, right_test, mid_test_t, return_logits=False).cpu().numpy() 
                
                print(f"学生模型边类型 {test_edge_type_id} 测试完成。预测结果形状: {pred.shape}")
                total_evaluated_edge_types += 1 

        if final_student_net is not None:
            print("\n" + "="*60)
            print("所有模型训练和评估完成。现在开始进行自定义预测。")
            print("="*60 + "\n")
            
            # Perform predictions for new bacteria on all target relations
            for rel_tuple, rel_id in all_target_relation_type_ids.items():
                predict_new_associations(args, final_student_net, features_list, g, device, # Pass 'g' here
                                         trait_node_type_name, node_shifts, node_counts,
                                         rel_id,
                                         new_bacteria_global_ids_start, num_new_bacteria_to_predict,
                                         f"{rel_tuple[0]}-{rel_tuple[1]}-{rel_tuple[2]}")
        else:
            print("\n没有训练好的学生模型进行自定义预测。")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sys.stdout.close() # 关闭日志文件
        sys.stdout = original_stdout # 恢复原始 stdout
        print(f"预测运行结束。详细日志请查看：{log_file_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset with Knowledge Distillation')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                          '0 - loaded features; ' +
                          '1 - only target node features (zero vec for others); ' +
                          '2 - only target node features (id vec for others); ' +
                          '3 - all id vec. Default is 2;' +
                          '4 - only term features (id vec for others);' + 
                          '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=2, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=40, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=40, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=2e-4)
    ap.add_argument('--slope', type=float, default=0.01)
    ap.add_argument('--dataset', type=str, default='PGMKG') # 默认值，如果命令行不指定
    ap.add_argument('--edge-feats', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--decoder', type=str, default='dot')
    ap.add_argument('--residual-att', type=float, default=0.)
    ap.add_argument('--residual', type=bool, default=False)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=2.0, help='Temperature for distillation.')
    ap.add_argument('--alpha', type=float, default=0.5, help='Weight for hard target loss in distillation (1-alpha for soft target loss).')

    args = ap.parse_args()
    os.makedirs('checkpoint', exist_ok=True)
    run_model_DBLP(args)