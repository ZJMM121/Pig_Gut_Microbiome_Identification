from helper import *
from data_loader import *
from model.models import *
import torch
import argparse
import numpy as np
import logging
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class ModelEvaluator(object):
    def __init__(self, model_path, dataset='PGMKG2.0'):
        # 设置参数，与训练时保持一致
        self.p = argparse.Namespace()
        self.p.dataset = dataset
        self.p.model = 'compgcn'
        self.p.score_func = 'conve'
        self.p.opn = 'corr'
        self.p.batch_size = 128
        self.p.gamma = 40.0
        self.p.gpu = '0'
        self.p.max_epochs = 500
        self.p.l2 = 0.0
        self.p.lr = 0.001
        self.p.lbl_smooth = 0.1
        self.p.num_workers = 10
        self.p.seed = 41504
        self.p.restore = False
        self.p.bias = False
        self.p.num_bases = -1
        self.p.init_dim = 100
        self.p.gcn_dim = 200
        self.p.embed_dim = None
        self.p.gcn_layer = 1
        self.p.dropout = 0.1
        self.p.hid_drop = 0.3
        self.p.hid_drop2 = 0.3
        self.p.feat_drop = 0.3
        self.p.k_w = 10
        self.p.k_h = 20
        self.p.num_filt = 200
        self.p.ker_sz = 7
        self.p.log_dir = './log/'
        self.p.config_dir = './config/'
        self.p.name = 'evaluation'
        
        # 设置设备 - 强制使用CPU避免CUDA兼容性问题
        self.device = torch.device('cpu')
        
        # 设置日志记录
        log_dir = './evaluation_log/'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'evaluation_{self.p.dataset}.log')
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # 同时输出到终端
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Using CPU for evaluation to avoid CUDA compatibility issues")
        self.logger.info(f"Evaluation log will be saved to: {log_file}")
            
        # 加载数据
        self.load_data()
        
        # 创建模型
        self.model = self.add_model(self.p.model, self.p.score_func)
        
        # 加载训练好的模型权重
        self.load_model(model_path)
        
    def load_data(self):
        """加载数据，与训练时相同的逻辑"""
        ent_set, rel_set = OrderedSet(), OrderedSet()
        data_dir = '/user_data/yezy/zhangjm/CompGCN/data/PGMKG2.0'
        
        for split in ['train', 'test', 'valid']:
            file_path = os.path.join(data_dir, '{}.txt'.format(split))
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    ent_set.add(sub)
                    rel_set.add(rel)
                    ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            file_path = os.path.join(data_dir, '{}.txt'.format(split))
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                    self.data[split].append((sub, rel, obj))

                    if split == 'train': 
                        sr2o[(sub, rel)].add(obj)
                        sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """构建邻接矩阵"""
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def add_model(self, model, score_func):
        """创建模型"""
        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'compgcn_transe':
            model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_distmult':
            model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_conve':
            model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def load_model(self, load_path):
        """加载模型"""
        state = torch.load(load_path, map_location='cpu')  # 强制加载到CPU
        state_dict = state['state_dict']
        self.model.load_state_dict(state_dict)
        self.logger.info(f"Successfully loaded model from {load_path} to CPU")

    def read_batch(self, batch, split):
        """读取批次数据"""
        triple, label = [_.to(self.device) for _ in batch]
        return triple[:, 0], triple[:, 1], triple[:, 2], label

    def predict(self, split='test', mode='tail_batch'):
        """模型预测"""
        self.model.eval()

        with torch.no_grad():
            results = {}
            data_iter = self.data_iter['{}_{}'.format(split, mode.split('_')[0])]
            
            # 添加进度条
            total_batches = len(data_iter)
            progress_bar = tqdm(data_iter, desc=f'{split.title()} {mode.title()}', total=total_batches)

            all_ranks = []
            all_predictions = []
            all_labels = []
            
            for step, batch in enumerate(progress_bar):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                
                # 收集预测分数和标签用于AUC计算
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_labels.extend(label.cpu().numpy().flatten())
                
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

                ranks = ranks.float()
                all_ranks.extend(ranks.cpu().numpy())
                
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0/ranks).item() + results.get('mrr', 0.0)
                
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                # 更新进度条描述
                if step % 50 == 0:
                    current_mrr = results['mrr'] / results['count'] if results['count'] > 0 else 0
                    progress_bar.set_postfix({'Current MRR': f'{current_mrr:.4f}'})

            # 计算最终指标
            count = results['count']
            results['mr'] = results['mr'] / count
            results['mrr'] = results['mrr'] / count
            
            for k in range(10):
                results['hits@{}'.format(k+1)] = results['hits@{}'.format(k+1)] / count
            
            # 计算AUC
            try:
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                # 将标签转换为二进制（正例为1，负例为0）
                binary_labels = 1 - all_labels  # 因为原始标签中1表示负例，0表示正例
                results['auc'] = roc_auc_score(binary_labels, all_predictions)
            except Exception as e:
                self.logger.warning(f"AUC calculation failed: {e}")
                results['auc'] = 0.0

        return results, all_ranks

    def evaluate_detailed(self, split='test'):
        """详细评估模型"""
        self.logger.info(f"\n=== Evaluating on {split} set ===")
        
        # 尾实体预测
        self.logger.info("Starting tail entity prediction...")
        tail_results, tail_ranks = self.predict(split=split, mode='tail_batch')
        self.logger.info(f"\nTail Prediction Results:")
        self.logger.info(f"MRR: {tail_results['mrr']:.5f}")
        self.logger.info(f"MR: {tail_results['mr']:.2f}")
        self.logger.info(f"AUC: {tail_results['auc']:.5f}")
        self.logger.info(f"Hits@1: {tail_results['hits@1']:.5f}")
        self.logger.info(f"Hits@3: {tail_results['hits@3']:.5f}")
        self.logger.info(f"Hits@10: {tail_results['hits@10']:.5f}")
        
        # 头实体预测
        self.logger.info("Starting head entity prediction...")
        head_results, head_ranks = self.predict(split=split, mode='head_batch')
        self.logger.info(f"\nHead Prediction Results:")
        self.logger.info(f"MRR: {head_results['mrr']:.5f}")
        self.logger.info(f"MR: {head_results['mr']:.2f}")
        self.logger.info(f"AUC: {head_results['auc']:.5f}")
        self.logger.info(f"Hits@1: {head_results['hits@1']:.5f}")
        self.logger.info(f"Hits@3: {head_results['hits@3']:.5f}")
        self.logger.info(f"Hits@10: {head_results['hits@10']:.5f}")
        
        # 计算平均结果
        avg_results = get_combined_results(tail_results, head_results)
        self.logger.info(f"\n=== Combined Results ===")
        self.logger.info(f"Average MRR: {avg_results['mrr']:.5f}")
        self.logger.info(f"Average MR: {(tail_results['mr'] + head_results['mr'])/2:.2f}")
        self.logger.info(f"Average AUC: {(tail_results['auc'] + head_results['auc'])/2:.5f}")
        self.logger.info(f"Average Hits@1: {(tail_results['hits@1'] + head_results['hits@1'])/2:.5f}")
        self.logger.info(f"Average Hits@3: {(tail_results['hits@3'] + head_results['hits@3'])/2:.5f}")
        self.logger.info(f"Average Hits@10: {(tail_results['hits@10'] + head_results['hits@10'])/2:.5f}")
        
        return avg_results

if __name__ == '__main__':
    # 模型路径
    model_path = '/user_data/yezy/zhangjm/CompGCN/checkpoints/testrun_30_07_2025_11:06:19'
    
    print("Loading model and data...")
    evaluator = ModelEvaluator(model_path, dataset='PGMKG2.0')
    
    evaluator.logger.info("Starting evaluation...")
    results = evaluator.evaluate_detailed('test')
    
    evaluator.logger.info("\nEvaluation completed!")
