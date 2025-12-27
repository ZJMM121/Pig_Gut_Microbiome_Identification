# 知识蒸馏实现说明

## 概述

本项目实现了基于图注意力网络(GAT)的知识蒸馏框架，用于异构图上的链接预测任务。通过Teacher-Student模型架构，实现了模型压缩和性能优化。

## 知识蒸馏架构

### 模型设计

- **Teacher Model**: 完整的多层GAT网络，参数量大，性能高
- **Student Model**: 轻量化的GAT网络，参数量约为Teacher的一半

```python
# Teacher Model: 完整网络
teacher_net = myGAT(g, args.edge_feats, num_etypes_for_model, 
                    in_dims_for_models, args.hidden_dim, 1, 
                    args.num_layers, [args.num_heads] * args.num_layers + [args.num_heads],
                    ...)

# Student Model: 压缩网络
student_num_hidden = args.hidden_dim // 2  # 隐藏层维度减半
student_num_layers = args.num_layers - 1   # 层数减少
student_net = myGAT_Student(g, args.edge_feats, num_etypes_for_model,
                           in_dims_for_models, student_num_hidden, 1,
                           student_num_layers, student_heads, ...)
```

### 训练流程

#### 第一阶段：Teacher模型训练

1. **目标**: 训练一个性能最优的Teacher模型
2. **数据**: 使用正样本和负样本进行二分类训练
3. **损失函数**: 标准的二元交叉熵损失
4. **优化**: 早停机制防止过拟合

```python
# Teacher训练过程
teacher_loss_func = nn.BCELoss()
logits = teacher_net(features_list, left_batch, right_batch, mid_batch_t, return_logits=True)
logp = F.sigmoid(logits)
train_loss = teacher_loss_func(logp, labels_batch)
```

#### 第二阶段：Student模型知识蒸馏

1. **目标**: 让Student模型学习Teacher的知识
2. **输入**: 相同的训练数据
3. **监督信号**: Teacher的soft targets + 真实标签的hard targets

## 核心蒸馏损失函数

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Hard Loss: 学生模型对真实标签的损失
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
    
    # Soft Loss: 学生模型学习教师模型的知识
    student_logits_2d = torch.stack([torch.zeros_like(student_logits), student_logits], dim=1)
    teacher_logits_2d = torch.stack([torch.zeros_like(teacher_logits), teacher_logits], dim=1)
    
    soft_target_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits_2d / temperature, dim=1),
        F.softmax(teacher_logits_2d / temperature, dim=1)
    ) * (temperature * temperature)
    
    # 总损失 = α * 硬损失 + (1-α) * 软损失
    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_target_loss
    return total_loss
```

### 损失函数组件说明

1. **Hard Loss (硬损失)**
   - 学生模型对真实标签的预测损失
   - 确保学生模型能够完成基本任务

2. **Soft Loss (软损失)**
   - 学生模型学习教师模型的概率分布
   - 使用KL散度衡量两个分布的差异
   - `temperature`参数控制分布的平滑程度

3. **温度参数 (Temperature)**
   - 较高的温度使概率分布更加平滑
   - 帮助学生模型学习教师的"不确定性"知识
   - 默认值：2.0

4. **权重参数 (Alpha)**
   - 控制硬损失和软损失的平衡
   - α = 0.5 表示两者权重相等
   - 可根据具体任务调整

## 训练过程

### Student模型训练循环

```python
for epoch in range(args.epoch):
    for batch in data_loader:
        student_net.train()
        teacher_net.eval()  # 保持Teacher为评估模式
        
        # 获取Teacher的预测(无梯度)
        with torch.no_grad():
            teacher_logits = teacher_net(features_list, left_batch, right_batch, 
                                       mid_batch_t, return_logits=True)
        
        # 获取Student的预测
        student_logits = student_net(features_list, left_batch, right_batch, 
                                   mid_batch_t, return_logits=True)
        
        # 计算蒸馏损失
        train_loss = distillation_loss(student_logits, teacher_logits, 
                                     labels_batch, temperature, alpha)
        
        # 反向传播和优化
        student_optimizer.zero_grad()
        train_loss.backward()
        student_optimizer.step()
```

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      知识蒸馏架构图                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    训练阶段1    ┌─────────────────┐         │
│  │   HGB数据集     │ ──────────────→ │  Teacher Model  │         │
│  │                 │                 │  (完整GAT网络)   │         │
│  │ • 节点特征      │                 │                 │         │
│  │ • 边关系        │                 │ • 隐藏层: 64维   │         │
│  │ • 训练/验证集   │                 │ • 层数: 3层     │         │
│  └─────────────────┘                 │ • 注意力头: 2个  │         │
│                                      └─────────────────┘         │
│                                              │                   │
│                                              │ 知识传递           │
│                                              ▼                   │
│  ┌─────────────────┐    训练阶段2    ┌─────────────────┐         │
│  │   相同数据      │ ──────────────→ │  Student Model  │         │
│  │                 │                 │  (轻量GAT网络)   │         │
│  │                 │                 │                 │         │
│  │                 │                 │ • 隐藏层: 32维   │         │
│  │                 │                 │ • 层数: 2层     │         │
│  │                 │                 │ • 注意力头: 2个  │         │
│  └─────────────────┘                 └─────────────────┘         │
│                                              │                   │
│                                              ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               蒸馏损失函数                                    │ │
│  │                                                             │ │
│  │  Loss = α × Hard_Loss + (1-α) × Soft_Loss                  │ │
│  │                                                             │ │
│  │  Hard_Loss = BCE(student_logits, true_labels)              │ │
│  │  Soft_Loss = KL_Div(student_probs, teacher_probs)          │ │
│  │                                                             │ │
│  │  Temperature = 2.0, Alpha = 0.5                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                              │                   │
│                                              ▼                   │
│  ┌─────────────────┐     预测阶段     ┌─────────────────┐         │
│  │   新细菌数据    │ ──────────────→ │ 训练好的Student │         │
│  │                 │                 │                 │         │
│  │ • 262个新细菌   │                 │ • 快速推理     │         │
│  │ • 128维特征     │                 │ • 低资源消耗   │         │
│  │                 │                 │ • 保持精度     │         │
│  └─────────────────┘                 └─────────────────┘         │
│                                              │                   │
│                                              ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    预测结果                                  │ │
│  │                                                             │ │
│  │  • 新细菌-性状关联预测                                       │ │
│  │  • Top N 排序结果                                           │ │
│  │  • 多种关系类型支持                                         │ │
│  │    - related_to                                             │ │
│  │    - positive_relate                                        │ │
│  │    - negative_relate                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 关键特性

### 1. 多关系类型支持
- 针对每种边类型(关系)独立训练Teacher-Student模型对
- 支持异构图中的多种关系预测

### 2. 模型压缩效果
- 隐藏层维度减半：`hidden_dim // 2`
- 网络层数减少：`num_layers - 1`
- 显著减少参数量和计算开销

### 3. 可复现性保证
- 设置全局随机种子：`global_seed = args.run * 1000 + 42`
- 每个epoch使用不同但可复现的种子
- 确保实验结果的一致性

## 使用方法

### 命令行参数

```bash
python run_predict0818.py \
    --hidden-dim 64 \         # Teacher模型隐藏维度
    --num-layers 3 \          # Teacher模型层数
    --temperature 2.0 \       # 蒸馏温度
    --alpha 0.5 \            # 损失权重
    --epoch 40 \             # 训练轮数
    --lr 5e-4 \              # 学习率
    --batch-size 8192        # 批次大小
```

### 关键参数说明

- `--temperature`: 蒸馏温度，控制软目标的平滑程度
- `--alpha`: 硬损失权重，(1-alpha)为软损失权重
- `--hidden-dim`: Teacher模型隐藏层维度，Student自动减半
- `--num-layers`: Teacher模型层数，Student自动减1

## 输出结果

1. **模型检查点**: 保存在`checkpoint/`目录
   - Teacher模型：`teacher_checkpoint_{dataset}_{layers}_{edge_type}.pt`
   - Student模型：`student_checkpoint_{dataset}_{layers}_{edge_type}.pt`

2. **预测结果**: 保存在`predictions0818/`目录
   - 每种关系类型的预测结果独立保存
   - 格式：`predictions_{relation_type}_{run}.txt`

3. **训练日志**: 保存在`log/`目录
   - 详细记录训练过程和错误信息

## 优势

1. **模型压缩**: Student模型参数量显著减少，推理速度更快
2. **性能保持**: 通过知识蒸馏保持较高的预测精度
3. **灵活部署**: 轻量化的Student模型更适合资源受限环境
4. **知识传递**: 有效传递Teacher模型学到的复杂特征表示

## 应用场景

- 异构图链接预测
- 生物信息学中的关联预测
- 推荐系统中的物品关联
- 社交网络分析

这个知识蒸馏框架特别适用于需要在保持预测精度的同时减少模型复杂度的场景。

## 文件结构

```
Simple-HGN-PGMKG2-KD/
├── method/
│   ├── run_predict0818.py          # 主程序文件
│   └── GNN.py                      # GAT模型定义
├── utils/
│   └── pytorchtools.py             # 早停工具
├── data/
│   └── PGMKG_HGB0810/              # HGB格式数据
├── checkpoint/                     # 模型检查点
├── predictions0818/                # 预测结果
├── log/                           # 训练日志
└── README_Knowledge_Distillation.md # 本文档
```

## 依赖环境

- Python 3.7+
- PyTorch 1.8+
- DGL 0.7+
- NumPy
- Scipy
- tqdm

## 更新日志

- 2024-08-18: 初始版本发布
- 支持HGB格式数据加载
- 实现Teacher-Student知识蒸馏
- 新细菌关联预测功能