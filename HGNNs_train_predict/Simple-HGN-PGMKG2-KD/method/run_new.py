#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../')
import time
import argparse
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from GNN import myGAT, myGAT_Student # Import myGAT_Student
import dgl
import os

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

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    Computes the knowledge distillation loss.

    Args:
        student_logits (torch.Tensor): Logits from the student model.
        teacher_logits (torch.Tensor): Logits from the teacher model.
        labels (torch.Tensor): True labels.
        temperature (float): Temperature for softening the teacher's logits.
        alpha (float): Weight for the hard target loss. (1 - alpha) for soft target loss.
    """
    # Hard target loss (e.g., Binary Cross-Entropy)
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

    # Soft target loss (KL Divergence between softened student and teacher probabilities)
    # Apply sigmoid to logits for probabilities, then softmax for distillation on probabilities
    # For binary classification, we can use BCEWithLogitsLoss for soft targets directly or KLDivLoss.
    # Let's use BCEWithLogitsLoss on softened probabilities, or KLDivLoss on log-softmax of logits.

    # Soften teacher probabilities
    teacher_soft_probs = torch.sigmoid(teacher_logits / temperature)
    student_soft_logits = student_logits / temperature # Soften student logits for consistency

    # Using BCEWithLogitsLoss for soft targets
    soft_loss = F.binary_cross_entropy_with_logits(student_soft_logits, teacher_soft_probs)

    # For multi-class classification, typically:
    # soft_loss = nn.KLDivLoss(reduction='batchmean')(
    #     F.log_softmax(student_logits / temperature, dim=-1),
    #     F.softmax(teacher_logits / temperature, dim=-1)
    # ) * (temperature * temperature)

    return alpha * hard_loss + (1. - alpha) * soft_loss

def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, dl = load_data(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    
    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    res_2hop = defaultdict(float)
    res_random = defaultdict(float)
    total = len(list(dl.links_test['data'].keys()))

    first_flag = True
    edge_types = list(dl.links_test['data'].keys())
    print(f"总共需要处理 {len(edge_types)} 种边类型: {edge_types}")
    
    for i, test_edge_type in enumerate(edge_types):
        print(f"\n{'='*50}")
        print(f"处理边类型 {i+1}/{len(edge_types)}: {test_edge_type}")
        print(f"{'='*50}")
        
        train_pos, valid_pos = dl.get_train_valid_pos()#edge_types=[test_edge_type])
        train_pos = train_pos[test_edge_type]
        valid_pos = valid_pos[test_edge_type]
        num_classes = args.hidden_dim
        heads = [args.num_heads] * args.num_layers + [args.num_heads]#[1]

        # --- Teacher Model Training ---
        print("\n--- Training Teacher Model ---")
        teacher_net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims,
                            args.hidden_dim, num_classes, args.num_layers, heads,
                            F.elu, args.dropout, args.dropout, args.slope, args.residual,
                            args.residual_att, decode=args.decoder)
        teacher_net.to(device)
        teacher_optimizer = torch.optim.Adam(teacher_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        teacher_early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                               save_path='checkpoint/teacher_checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers, test_edge_type))
        teacher_loss_func = nn.BCELoss()

        for epoch in range(args.epoch):
          train_neg = dl.get_train_neg(edge_types=[test_edge_type])[test_edge_type]
          train_pos_head_full = np.array(train_pos[0])
          train_pos_tail_full = np.array(train_pos[1])
          train_neg_head_full = np.array(train_neg[0])
          train_neg_tail_full = np.array(train_neg[1])
          train_idx = np.arange(len(train_pos_head_full))
          np.random.shuffle(train_idx)
          batch_size = args.batch_size
          for step, start in enumerate(range(0, len(train_pos_head_full), args.batch_size)):
            t_start = time.time()
            teacher_net.train()
            train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
            train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
            train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
            train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
            left = np.concatenate([train_pos_head, train_neg_head])
            right = np.concatenate([train_pos_tail, train_neg_tail])
            mid = np.zeros(train_pos_head.shape[0]+train_neg_head.shape[0], dtype=np.int32)
            labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

            logits = teacher_net(features_list, e_feat, left, right, mid, return_logits=True) # Get logits for teacher
            logp = F.sigmoid(logits)
            train_loss = teacher_loss_func(logp, labels)

            teacher_optimizer.zero_grad()
            train_loss.backward()
            teacher_optimizer.step()

            t_end = time.time()
            print('Teacher Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))

            t_start = time.time()
            teacher_net.eval()
            with torch.no_grad():
                valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[test_edge_type]
                valid_pos_head = np.array(valid_pos[0])
                valid_pos_tail = np.array(valid_pos[1])
                valid_neg_head = np.array(valid_neg[0])
                valid_neg_tail = np.array(valid_neg[1])
                left = np.concatenate([valid_pos_head, valid_neg_head])
                right = np.concatenate([valid_pos_tail, valid_neg_tail])
                mid = np.zeros(valid_pos_head.shape[0]+valid_neg_head.shape[0], dtype=np.int32)
                labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                
                logits = teacher_net(features_list, e_feat, left, right, mid, return_logits=True)
                logp = F.sigmoid(logits)
                val_loss = teacher_loss_func(logp, labels)
            t_end = time.time()
            print('Teacher Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            teacher_early_stopping(val_loss, teacher_net)
            if teacher_early_stopping.early_stop:
                print('Teacher Early stopping!')
                break
          if teacher_early_stopping.early_stop:
              print('Teacher Early stopping!')
              break
        
        teacher_net.load_state_dict(torch.load('checkpoint/teacher_checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers, test_edge_type)))
        teacher_net.eval()
        print("\n--- Teacher Model Training Complete ---")


        # --- Student Model Training (Knowledge Distillation) ---
        print("\n--- Training Student Model (Knowledge Distillation) ---")
        student_num_hidden = args.hidden_dim // 2 # Example: half the hidden dim
        student_num_layers = args.num_layers - 1 if args.num_layers > 1 else 1 # Example: one less layer
        student_heads = [args.num_heads] * student_num_layers + [args.num_heads]

        student_net = myGAT_Student(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims,
                                    student_num_hidden, num_classes, student_num_layers, student_heads,
                                    F.elu, args.dropout, args.dropout, args.slope, args.residual,
                                    args.residual_att, decode=args.decoder)
        student_net.to(device)
        student_optimizer = torch.optim.Adam(student_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        student_early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                               save_path='checkpoint/student_checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers, test_edge_type))
        
        # Distillation parameters
        temperature = args.temperature
        alpha = args.alpha

        for epoch in range(args.epoch):
            train_neg = dl.get_train_neg(edge_types=[test_edge_type])[test_edge_type]
            train_pos_head_full = np.array(train_pos[0])
            train_pos_tail_full = np.array(train_pos[1])
            train_neg_head_full = np.array(train_neg[0])
            train_neg_tail_full = np.array(train_neg[1])
            train_idx = np.arange(len(train_pos_head_full))
            np.random.shuffle(train_idx)
            batch_size = args.batch_size

            for step, start in enumerate(range(0, len(train_pos_head_full), args.batch_size)):
                t_start = time.time()
                student_net.train()
                teacher_net.eval() # Teacher is in eval mode during student training

                train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
                train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
                train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
                train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
                left = np.concatenate([train_pos_head, train_neg_head])
                right = np.concatenate([train_pos_tail, train_neg_tail])
                mid = np.zeros(train_pos_head.shape[0]+train_neg_head.shape[0], dtype=np.int32)
                labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

                # Get teacher logits
                with torch.no_grad():
                    teacher_logits = teacher_net(features_list, e_feat, left, right, mid, return_logits=True)

                # Get student logits
                student_logits = student_net(features_list, e_feat, left, right, mid, return_logits=True)

                # Calculate distillation loss
                train_loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)

                student_optimizer.zero_grad()
                train_loss.backward()
                student_optimizer.step()

                t_end = time.time()
                print('Student Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))

                t_start = time.time()
                student_net.eval()
                with torch.no_grad():
                    valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[test_edge_type]
                    valid_pos_head = np.array(valid_pos[0])
                    valid_pos_tail = np.array(valid_pos[1])
                    valid_neg_head = np.array(valid_neg[0])
                    valid_neg_tail = np.array(valid_neg[1])
                    left = np.concatenate([valid_pos_head, valid_neg_head])
                    right = np.concatenate([valid_pos_tail, valid_neg_tail])
                    mid = np.zeros(valid_pos_head.shape[0]+valid_neg_head.shape[0], dtype=np.int32)
                    labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                    
                    student_val_logits = student_net(features_list, e_feat, left, right, mid, return_logits=True)
                    # For validation, we typically just use the hard target loss to monitor performance
                    val_loss = F.binary_cross_entropy_with_logits(student_val_logits, labels)
                t_end = time.time()
                print('Student Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                    epoch, val_loss.item(), t_end - t_start))
                student_early_stopping(val_loss, student_net)
                if student_early_stopping.early_stop:
                    print('Student Early stopping!')
                    break
            if student_early_stopping.early_stop:
                print('Student Early stopping!')
                break
        
        student_net.load_state_dict(torch.load('checkpoint/student_checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers, test_edge_type)))
        student_net.eval()
        print("\n--- Student Model Training Complete ---")


        # --- Testing with Student Model ---
        print(f"\n开始测试学生模型针对边类型 {test_edge_type}...")
        test_logits = []
        with torch.no_grad():
            print(f"获取测试数据...")
            test_neigh, test_label = dl.get_test_neigh_w_random()
            print(f"测试数据获取完成，边类型: {test_edge_type}")
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            print(f"测试样本数: {len(test_neigh[0])}")
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            mid[:] = test_edge_type
            labels = torch.FloatTensor(test_label).to(device)
            print(f"开始学生模型推理...")
            pred = student_net(features_list, e_feat, left, right, mid, return_logits=False).cpu().numpy() # Student outputs probabilities directly
            edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
            labels = labels.cpu().numpy()
            print(f"生成评估文件...")
            dl.gen_file_for_evaluate(test_neigh, pred, test_edge_type, file_path=f"{args.dataset}_{args.run}_student.txt", flag=first_flag)
            first_flag = False
            print(f"开始评估...")
            res = dl.evaluate(edge_list, pred, labels)
            print(f"学生模型边类型 {test_edge_type} 评估结果: {res}")
            for k in res:
                res_random[k] += res[k]
    # Note: res_2hop is not calculated in the original code's modified test block.
    # If you need it, uncomment the relevant block from the original run_new.py.
    # For now, only res_random is used for overall average.
    for k in res_random:
        res_random[k] /= total
    print("Final Student Model Average Results (Random Negative Sampling):")
    print(res_random)

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
    ap.add_argument('--dataset', type=str)
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