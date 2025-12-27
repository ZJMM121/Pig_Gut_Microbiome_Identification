# method/GNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import myGATConv # 确保路径正确
print("DEBUG: GNN.py 版本：2025年8月7日调试版本") # 添加这一行
class myGAT(nn.Module):
    def __init__(self, g, edge_feats_dim, num_etypes, in_dims, hidden_dim, num_classes, num_layers, heads, activation, feat_drop, attn_drop, negative_slope, residual, res_attn, decode):
        super(myGAT, self).__init__()
        self.g = g # Store the graph reference
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feats_dim = edge_feats_dim # Store edge_feats_dim
        
        # 定义边类型嵌入层
        self.edge_emb_feats = nn.Embedding(num_etypes, edge_feats_dim)

        self.gat_layers = nn.ModuleList()
        # Input projection (features_list processing)
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=True) for in_dim in in_dims])
        # Output projection
        if decode == 'dot':
            self.fc_out = nn.Linear(hidden_dim * 2, num_classes, bias=True)
        else: # mlp decoder
            self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
            self.fc_out_mlp = nn.Linear(hidden_dim, num_classes, bias=True)

        # GAT layers
        for l in range(num_layers):
            # 将 edge_feats_dim 传递给 myGATConv
            self.gat_layers.append(myGATConv(hidden_dim, edge_feats_dim, heads[l], activation, feat_drop, attn_drop, negative_slope, residual))
            
        self.activation = activation
        self.decode = decode


    # 移除 e_feat 参数
    def forward(self, features_list, left, right, mid, return_logits=False):
        # *** 在这里添加以下调试代码 ***
        # import inspect
        # frame = inspect.currentframe()
        # args, _, _, values = inspect.getargvalues(frame)
        # print("\n--- DEBUG: forward 方法接收到的参数 ---")
        # for arg in args:
        #     print(f"  {arg}: {values[arg]}")
        # print("---------------------------------------\n")
        h = []

        # DEBUG Print: 初始 features_list 的形状
        # print(f"DEBUG (GNN.py): Initial features_list shapes: {[f.shape for f in features_list]}")
        for i, (fc, feature) in enumerate(zip(self.fc_list, features_list)):
            transformed_feature = fc(feature)
            h.append(transformed_feature)
            # DEBUG Print: 每次 Linear 变换后的特征形状
            # print(f"DEBUG (GNN.py): Transformed feature {i} shape: {transformed_feature.shape} (expected {self.hidden_dim})")
        
        h = torch.cat(h, 0)
        # DEBUG Print: 拼接后 h 的形状
        #print(f"DEBUG (GNN.py): h shape after concatenation: {h.shape} (expected {h.shape[0]}x{self.hidden_dim})")

        res_attn = None
        
        # 修复TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not numpy.int32
        # 确保 specific_relation_type_id 是一个 torch.LongTensor
        specific_relation_type_id = mid[0].to(torch.long)
        
        # 从 embedding 层获取这个特定边类型的嵌入
        # 形状为 (edge_feats_dim,)
        single_edge_embedding = self.edge_emb_feats(specific_relation_type_id)
        
        # 将这个嵌入复制为图 g 中所有边的特征
        # 形状为 (num_edges_in_graph, edge_feats_dim)
        # 这样 myGATConv 就能接收到正确形状的边特征
        all_graph_edge_features = single_edge_embedding.unsqueeze(0).repeat(self.g.number_of_edges(), 1)
        # DEBUG Print: 边特征的形状
        #print(f"DEBUG (GNN.py): all_graph_edge_features shape: {all_graph_edge_features.shape}")


        for l in range(self.num_layers):
            # DEBUG Print: 进入 GAT 层前 h 的形状
            #print(f"DEBUG (GNN.py): Before GAT layer {l}, h shape: {h.shape}")
            h, res_attn = self.gat_layers[l](self.g, h, all_graph_edge_features, res_attn=res_attn)
            # DEBUG Print: 离开 GAT 层后 h 的形状
            #print(f"DEBUG (GNN.py): After GAT layer {l}, h shape: {h.shape}")
        
        # Link prediction part
        left_h = h[left]
        right_h = h[right]
        # DEBUG Print: link prediction inputs 的形状
        #print(f"DEBUG (GNN.py): left_h shape: {left_h.shape}, right_h shape: {right_h.shape}")


        if self.decode == 'dot':
            logits = torch.sum(left_h * right_h, dim=1)
        else: # mlp decoder
            logits = self.fc_out(torch.cat((left_h, right_h), dim=1))
            logits = self.activation(logits)
            logits = self.fc_out_mlp(logits)
            logits = logits.squeeze(1)

        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)


class myGAT_Student(myGAT): # myGAT_Student 继承自 myGAT，因此也需要更新其构造函数
    def __init__(self, g, edge_feats_dim, num_etypes, in_dims, hidden_dim, num_classes, num_layers, heads, activation, feat_drop, attn_drop, negative_slope, residual, res_attn, decode):
        # Student model has reduced layers and hidden_dim
        super().__init__(g, edge_feats_dim, num_etypes, in_dims, hidden_dim, num_classes, num_layers, heads, activation, feat_drop, attn_drop, negative_slope, residual, res_attn, decode)
        # Override gat_layers to match student architecture if needed, or rely on super's init logic.
        # Ensure student model also gets edge_emb_feats and correctly handles edge features in forward.
        # The forward method of myGAT_Student will inherit the changes from myGAT.