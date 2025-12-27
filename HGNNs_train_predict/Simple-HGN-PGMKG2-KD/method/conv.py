# method/conv.py

import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class myGATConv(nn.Module):
    # 移除 num_etypes 参数，因为不再在 Conv 层中进行 embedding lookup
    def __init__(self, in_dim, edge_feats, num_heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(myGATConv, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.edge_feats = edge_feats # edge_feats 现在是边特征的维度
        self.out_dim = in_dim // num_heads # Assuming out_dim is in_dim / num_heads for each head
        
        # QKV transformations
        # 这些线性层应该将输入的 in_dim 映射到 num_heads * out_dim_per_head
        self.fc_q = nn.Linear(in_dim, self.num_heads * self.out_dim, bias=False)
        self.fc_k = nn.Linear(in_dim, self.num_heads * self.out_dim, bias=False)
        self.fc_v = nn.Linear(in_dim, self.num_heads * self.out_dim, bias=False)
        
        # Edge feature transformation for attention
        self.fc_e = nn.Linear(edge_feats, self.num_heads, bias=False) # Maps edge_feats to attention weights per head

        # Attention mechanism
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self.out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self.out_dim)))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual

        if residual:
            # 残差连接的线性层，确保其输出维度与多头拼接后的维度一致
            self.res_fc = nn.Linear(in_dim, num_heads * self.out_dim, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain) # Initialize edge feature transform
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.residual:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    # 将 e_feat 参数重命名为 edge_features，它现在是形状为 (num_edges, edge_feats) 的张量
    def forward(self, graph, h, edge_features, res_attn=None):
        with graph.local_scope():
            # Apply feature dropout
            h_prime = self.feat_drop(h)

            # Project to Q, K, V for multi-head attention
            # h_prime 的形状应为 (num_nodes, in_dim)
            q = self.fc_q(h_prime).view(-1, self.num_heads, self.out_dim)
            k = self.fc_k(h_prime).view(-1, self.num_heads, self.out_dim)
            v = self.fc_v(h_prime).view(-1, self.num_heads, self.out_dim)

            # Project edge features for attention
            ee = self.fc_e(edge_features).view(-1, self.num_heads, 1)

            # Store node and edge features in graph.ndata and graph.edata
            graph.srcdata.update({'q': q, 'k': k, 'v': v})
            graph.edata.update({'ee': ee})

            # Compute attention scores
            # u_mul_e('q', 'ee', 'm_att') computes element-wise multiplication of q and ee
            graph.apply_edges(fn.u_mul_e('q', 'ee', 'm_att'))
            # v_dot_u('q', 'k', 'e_score') computes dot product between dst node's q and src node's k
            graph.apply_edges(fn.v_dot_u('q', 'k', 'e_score'))

            # Attention score (alpha)
            e = self.leaky_relu(graph.edata['e_score']) # (num_edges, num_heads, 1)
            alpha = self.attn_drop(dgl.nn.functional.edge_softmax(graph, e))

            # Store attention scores on edges
            graph.edata['alpha'] = alpha

            # Message passing
            # fn.src_mul_edge('v', 'alpha', 'm') computes v * alpha as message
            # fn.sum('m', 'h_aggregated') aggregates messages to destination nodes
            graph.update_all(fn.src_mul_edge('v', 'alpha', 'm'), fn.sum('m', 'h_aggregated'))
            
            # rst has shape (num_nodes, num_heads, out_dim)
            rst = graph.dstdata['h_aggregated']

            # Crucial Fix: Concatenate the heads to produce a 2D tensor for the next layer
            # Output shape will be (num_nodes, num_heads * self.out_dim)
            rst = rst.view(rst.shape[0], -1) 

            # Optional: Residual connection
            if self.residual:
                # The 'h' here is the original input to this GAT layer, which is (num_nodes, in_dim)
                # We need to ensure that self.res_fc(h) has the same shape as rst after concatenation.
                # Since in_dim = hidden_dim and num_heads * out_dim = num_heads * (hidden_dim / num_heads) = hidden_dim,
                # the dimensions should match (64 for your current setup).
                rst = self.res_fc(h) + rst

            # Activation after GAT layer (typically in GNN.py after calling the layer)
            # return self.activation(rst), res_attn # If activation is handled here
            return rst, res_attn # Assuming activation is handled by myGAT after this layer