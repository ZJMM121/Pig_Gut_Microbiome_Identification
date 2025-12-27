#!/usr/bin/env python3
"""
优化的训练脚本，解决内存和兼容性问题
"""

import os
import warnings

# 在导入其他模块之前设置环境变量和警告过滤
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置TensorFlow环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误
os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32,optimizer=fast_compile'

import argparse
import random
import tensorflow as tf
import numpy as np

# 配置TensorFlow会话
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
tf_config.log_device_placement = False

from optimization.optimize import build_tensorflow
from common import settings_reader, io, model_builder, optimizer_parameter_parser, evaluation, auxilliaries
from model import Model

def main():
    parser = argparse.ArgumentParser(description="Train a model on a given dataset.")
    parser.add_argument("--settings", help="Filepath for settings file.", required=True)
    parser.add_argument("--dataset", help="Filepath for dataset.", required=True)
    args = parser.parse_args()

    settings = settings_reader.read(args.settings)
    print("设置已加载:", settings)

    # 检查内存使用情况
    try:
        import psutil
        memory = psutil.virtual_memory()
        print("可用内存: {:.2f} GB".format(memory.available / (1024**3)))
        if memory.available < 2 * (1024**3):
            print("警告: 可用内存不足2GB，建议减少BatchSize")
    except ImportError:
        print("psutil未安装，无法检查内存使用情况")

    # 加载数据集
    dataset = args.dataset
    relations_path = dataset + '/relations.dict'
    entities_path = dataset + '/entities.dict'
    train_path = dataset + '/train.txt'
    valid_path = dataset + '/valid.txt'
    test_path = dataset + '/test.txt'

    # 扩展路径以进行准确性评估
    if settings['Evaluation']['Metric'] == 'Accuracy':
        valid_path = dataset + '/valid_accuracy.txt'
        test_path = dataset + '/test_accuracy.txt'

    print("正在加载数据...")
    train_triplets = io.read_triplets_as_list(train_path, entities_path, relations_path)
    valid_triplets = io.read_triplets_as_list(valid_path, entities_path, relations_path)
    test_triplets = io.read_triplets_as_list(test_path, entities_path, relations_path)

    train_triplets = np.array(train_triplets)
    valid_triplets = np.array(valid_triplets)
    test_triplets = np.array(test_triplets)

    entities = io.read_dictionary(entities_path)
    relations = io.read_dictionary(relations_path)

    print("训练三元组数量: {}".format(len(train_triplets)))
    print("验证三元组数量: {}".format(len(valid_triplets)))
    print("测试三元组数量: {}".format(len(test_triplets)))
    print("实体数量: {}".format(len(entities)))
    print("关系数量: {}".format(len(relations)))

    # 加载通用设置
    encoder_settings = settings['Encoder']
    decoder_settings = settings['Decoder']
    shared_settings = settings['Shared']
    general_settings = settings['General']
    optimizer_settings = settings['Optimizer']
    evaluation_settings = settings['Evaluation']

    general_settings.put('EntityCount', len(entities))
    general_settings.put('RelationCount', len(relations))
    general_settings.put('EdgeCount', len(train_triplets))

    encoder_settings.merge(shared_settings)
    encoder_settings.merge(general_settings)
    decoder_settings.merge(shared_settings)
    decoder_settings.merge(general_settings)

    optimizer_settings.merge(general_settings)
    evaluation_settings.merge(general_settings)

    # 构建编码器-解码器对
    print("正在构建模型...")
    encoder = model_builder.build_encoder(encoder_settings, train_triplets)
    model = model_builder.build_decoder(encoder, decoder_settings)

    # 构建优化器
    opp = optimizer_parameter_parser.Parser(optimizer_settings)
    opp.set_save_function(model.save)

    scorer = evaluation.Scorer(evaluation_settings)
    scorer.register_data(train_triplets)
    scorer.register_data(valid_triplets)
    scorer.register_data(test_triplets)
    scorer.register_degrees(train_triplets)
    scorer.register_model(model)
    scorer.finalize_frequency_computation(np.concatenate((train_triplets, valid_triplets, test_triplets), axis=0))

    def score_validation_data(validation_data):
        score_summary = scorer.compute_scores(validation_data, verbose=False).get_summary()
        
        if evaluation_settings['Metric'] == 'MRR':
            lookup_string = score_summary.mrr_string()
        elif evaluation_settings['Metric'] == 'Accuracy':
            lookup_string = score_summary.accuracy_string()

        early_stopping = score_summary.results['Filtered'][lookup_string]
        
        # 定期显示测试结果
        test_score_summary = scorer.compute_scores(test_triplets, verbose=False).get_summary()
        test_score_summary.pretty_print()

        return early_stopping

    opp.set_early_stopping_score_function(score_validation_data)

    # 构建邻接列表
    print("正在构建邻接列表...")
    adj_list = [[] for _ in entities]
    for i, triplet in enumerate(train_triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]

    # 采样函数（保持原有逻辑）
    def sample_edge_neighborhood(triplets, sample_size):
        edges = np.zeros((sample_size), dtype=np.int32)
        sample_counts = np.array([d for d in degrees])
        picked = np.array([False for _ in triplets])
        seen = np.array([False for _ in degrees])

        for i in range(0, sample_size):
            weights = sample_counts * seen

            if np.sum(weights) == 0:
                weights = np.ones_like(weights)
                weights[np.where(sample_counts == 0)] = 0

            probabilities = (weights) / np.sum(weights)
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
            chosen_adj_list = adj_list[chosen_vertex]
            seen[chosen_vertex] = True

            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

            while picked[edge_number]:
                chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
                chosen_edge = chosen_adj_list[chosen_edge]
                edge_number = chosen_edge[0]

            edges[i] = edge_number
            other_vertex = chosen_edge[1]
            picked[edge_number] = True
            sample_counts[chosen_vertex] -= 1
            sample_counts[other_vertex] -= 1
            seen[other_vertex] = True

        return edges

    # 负采样设置
    if 'NegativeSampleRate' in general_settings:
        ns = auxilliaries.NegativeSampler(int(general_settings['NegativeSampleRate']), general_settings['EntityCount'])
        ns.set_known_positives(train_triplets)

        def t_func(x):
            arr = np.array(x)
            if not encoder.needs_graph():
                return ns.transform(arr)
            else:
                if 'GraphBatchSize' in general_settings:
                    graph_batch_size = int(general_settings['GraphBatchSize'])
                    graph_batch_ids = sample_edge_neighborhood(arr, graph_batch_size)
                else:
                    graph_batch_size = arr.shape[0]
                    graph_batch_ids = np.arange(graph_batch_size)

                graph_batch = np.array(train_triplets)[graph_batch_ids]

                # 应用dropout
                graph_percentage = float(general_settings['GraphSplitSize'])
                split_size = int(graph_percentage * graph_batch.shape[0])
                graph_split_ids = np.random.choice(graph_batch_ids, size=split_size, replace=False)
                graph_split = np.array(train_triplets)[graph_split_ids]

                t = ns.transform(graph_batch)

                if 'StoreEdgeData' in encoder_settings and encoder_settings['StoreEdgeData'] == "Yes":
                    return (graph_split, graph_split_ids, t[0], t[1])
                else:
                    return (graph_split, t[0], t[1])

        opp.set_sample_transform_function(t_func)

    # 初始化训练
    print("正在初始化训练...")
    model.preprocess(train_triplets)
    model.register_for_test(train_triplets)
    model.initialize_train()

    optimizer_weights = model.get_weights()
    optimizer_input = model.get_train_input_variables()
    loss = model.get_loss(mode='train') + model.get_regularization()

    # 添加额外操作
    for add_op in model.get_additional_ops():
        opp.additional_ops.append(add_op)

    optimizer_parameters = opp.get_parametrization()

    # 开始训练
    print("开始训练...")
    with tf.Session(config=tf_config) as sess:
        model.session = sess
        optimizer = build_tensorflow(loss, optimizer_weights, optimizer_parameters, optimizer_input)
        optimizer.set_session(model.session)
        
        try:
            optimizer.fit(train_triplets, validation_data=valid_triplets)
            print("训练完成!")
        except KeyboardInterrupt:
            print("训练被用户中断")
        except Exception as e:
            print("训练过程中出现错误: {}".format(e))
            raise

if __name__ == "__main__":
    main()
