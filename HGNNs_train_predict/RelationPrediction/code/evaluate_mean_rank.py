#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mean Rank 评估脚本
用于评估训练好的关系预测模型的 Mean Rank 指标
"""

import argparse
import numpy as np
import tensorflow as tf
import common.settings_reader as settings_reader
import common.io as io
import common.model_builder as model_builder
import common.evaluation as evaluation
from model import Model
import sys
import os

def calculate_mean_rank(ranks):
    """计算Mean Rank"""
    return np.mean(ranks)

def calculate_metrics_with_mean_rank(raw_ranks, filtered_ranks):
    """计算包含Mean Rank的所有指标"""
    results = {}
    
    # Mean Rank
    results['Raw Mean Rank'] = calculate_mean_rank(raw_ranks)
    results['Filtered Mean Rank'] = calculate_mean_rank(filtered_ranks)
    
    # MRR (现有指标)
    results['Raw MRR'] = np.mean([1.0/rank for rank in raw_ranks])
    results['Filtered MRR'] = np.mean([1.0/rank for rank in filtered_ranks])
    
    # Hits@K
    for k in [1, 3, 10]:
        results['Raw Hits@{}'.format(k)] = np.mean([1.0 if rank <= k else 0.0 for rank in raw_ranks])
        results['Filtered Hits@{}'.format(k)] = np.mean([1.0 if rank <= k else 0.0 for rank in filtered_ranks])
    
    return results

class MeanRankScorer(evaluation.Scorer):
    """扩展原有Scorer类以支持Mean Rank计算"""
    
    def compute_mean_rank_scores(self, triples, verbose=False):
        """计算Mean Rank分数"""
        score = evaluation.MrrScore(triples)
        
        chunk_size = 1000
        n_chunks = int(np.ceil(len(triples) / chunk_size))
        
        if verbose:
            print("正在评估 {} 个三元组，分为 {} 个批次...".format(len(triples), n_chunks))
        
        for chunk in range(n_chunks):
            i_b = chunk * chunk_size
            i_e = min(i_b + chunk_size, len(triples))
            
            triple_chunk = triples[i_b:i_e]
            self.evaluate_mrr(score, triple_chunk, verbose and chunk % 10 == 0)
            
            if verbose and chunk % 10 == 0:
                print("已完成批次 {}/{}".format(chunk+1, n_chunks))
        
        return score

def load_model_and_data(settings_path, dataset_path, model_path):
    """加载模型和数据"""
    print("正在加载设置...")
    settings = settings_reader.read(settings_path)
    
    print("正在加载数据...")
    # 数据路径
    relations_path = dataset_path + '/relations.dict'
    entities_path = dataset_path + '/entities.dict'
    test_path = dataset_path + '/test.txt'
    train_path = dataset_path + '/train.txt'
    valid_path = dataset_path + '/valid.txt'
    
    # 读取数据
    test_triplets = np.array(io.read_triplets_as_list(test_path, entities_path, relations_path))
    train_triplets = np.array(io.read_triplets_as_list(train_path, entities_path, relations_path))
    valid_triplets = np.array(io.read_triplets_as_list(valid_path, entities_path, relations_path))
    
    entities = io.read_dictionary(entities_path)
    relations = io.read_dictionary(relations_path)
    
    print("测试三元组数量: {}".format(len(test_triplets)))
    print("实体数量: {}".format(len(entities)))
    print("关系数量: {}".format(len(relations)))
    
    print("正在构建模型...")
    # 构建模型
    encoder_settings = settings['Encoder']
    decoder_settings = settings['Decoder']
    shared_settings = settings['Shared']
    general_settings = settings['General']
    
    # 设置必要的参数
    general_settings.put('EntityCount', len(entities))
    general_settings.put('RelationCount', len(relations))
    general_settings.put('EdgeCount', len(train_triplets))
    
    # 合并设置
    encoder_settings.merge(shared_settings)
    encoder_settings.merge(general_settings)
    decoder_settings.merge(shared_settings)
    decoder_settings.merge(general_settings)
    
    # 构建编码器和解码器
    encoder = model_builder.build_encoder(encoder_settings, train_triplets)
    model = model_builder.build_decoder(encoder, decoder_settings)
    
    print("正在加载训练好的模型权重...")
    # 创建TensorFlow会话
    sess = tf.Session()
    
    # 设置模型的会话
    model.session = sess
    
    # 初始化模型的计算图（这会创建TensorFlow变量）
    # 创建一个小的测试批次来初始化变量
    dummy_triplets = train_triplets[:1]  # 只用第一个三元组来初始化
    try:
        model.score(dummy_triplets)  # 这会创建必要的TensorFlow变量
    except:
        # 如果失败，尝试其他初始化方法
        pass
    
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    # 现在创建Saver并加载权重
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    
    return model, test_triplets, train_triplets, valid_triplets, entities, relations, settings

def main():
    parser = argparse.ArgumentParser(description="评估模型的Mean Rank指标")
    parser.add_argument("--settings", help="设置文件路径", 
                       default="/user_data/yezy/zhangjm/RelationPrediction/settings/gcn_basis_optimized.exp")
    parser.add_argument("--dataset", help="数据集路径", 
                       default="/user_data/yezy/zhangjm/RelationPrediction/data/PGMKG2.0")
    parser.add_argument("--model", help="模型文件路径", 
                       default="/user_data/yezy/zhangjm/RelationPrediction/models/GcnBlock_optimized-3")
    parser.add_argument("--output", help="结果输出文件路径", 
                       default="/user_data/yezy/zhangjm/RelationPrediction/mean_rank_results.txt")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.settings):
        print("错误：设置文件不存在 {}".format(args.settings))
        sys.exit(1)
    
    if not os.path.exists(args.dataset):
        print("错误：数据集路径不存在 {}".format(args.dataset))
        sys.exit(1)
        
    model_files = [f for f in os.listdir(os.path.dirname(args.model)) 
                   if os.path.basename(args.model) in f]
    if not model_files:
        print("错误：模型文件不存在 {}.*".format(args.model))
        sys.exit(1)
    
    try:
        # 加载模型和数据
        model, test_triplets, train_triplets, valid_triplets, entities, relations, settings = \
            load_model_and_data(args.settings, args.dataset, args.model)
        
        # 创建评估器
        print("正在初始化评估器...")
        scorer = MeanRankScorer(settings['Evaluation'])
        
        # 注册数据
        all_triplets = np.concatenate([train_triplets, valid_triplets, test_triplets])
        scorer.register_data(all_triplets)
        scorer.register_degrees(all_triplets)
        scorer.finalize_frequency_computation(all_triplets)
        scorer.register_model(model)
        
        # 计算测试集上的Mean Rank
        print("正在计算测试集的Mean Rank指标...")
        score = scorer.compute_mean_rank_scores(test_triplets, verbose=args.verbose)
        
        # 获取原始结果
        summary = score.get_summary()
        
        # 计算包含Mean Rank的指标
        results = calculate_metrics_with_mean_rank(score.raw_ranks, score.filtered_ranks)
        
        # 打印结果
        print("\n" + "="*60)
        print("Mean Rank 评估结果")
        print("="*60)
        
        print("{:<20} {:<15} {:<15}".format('指标', 'Raw', 'Filtered'))
        print("-" * 50)
        print("{:<20} {:<15.3f} {:<15.3f}".format('Mean Rank', results['Raw Mean Rank'], results['Filtered Mean Rank']))
        print("{:<20} {:<15.3f} {:<15.3f}".format('MRR', results['Raw MRR'], results['Filtered MRR']))
        
        for k in [1, 3, 10]:
            print("{:<20} {:<15.3f} {:<15.3f}".format('Hits@' + str(k), results['Raw Hits@{}'.format(k)], results['Filtered Hits@{}'.format(k)]))
        
        # 保存结果到文件
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("Mean Rank 评估结果\n")
            f.write("="*60 + "\n")
            f.write("模型路径: {}\n".format(args.model))
            f.write("数据集路径: {}\n".format(args.dataset))
            f.write("测试三元组数量: {}\n\n".format(len(test_triplets)))
            
            f.write("{:<20} {:<15} {:<15}\n".format('指标', 'Raw', 'Filtered'))
            f.write("-" * 50 + "\n")
            f.write("{:<20} {:<15.3f} {:<15.3f}\n".format('Mean Rank', results['Raw Mean Rank'], results['Filtered Mean Rank']))
            f.write("{:<20} {:<15.3f} {:<15.3f}\n".format('MRR', results['Raw MRR'], results['Filtered MRR']))
            
            for k in [1, 3, 10]:
                f.write("{:<20} {:<15.3f} {:<15.3f}\n".format('Hits@' + str(k), results['Raw Hits@{}'.format(k)], results['Filtered Hits@{}'.format(k)]))
        
        print("\n结果已保存到: {}".format(args.output))
        
        # 关闭TensorFlow会话
        model.session.close()
        
    except Exception as e:
        print("评估过程中出现错误: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
