import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from torch_geometric.nn import global_mean_pool
import math


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_views=2, temperature=1.0):
        """
        Args:
            input_dim: 输入特征维度
            num_views: 视图数量
            temperature: softmax 温度参数
        """
        super(AttentionFusion, self).__init__()
        self.num_views = num_views
        self.input_dim = input_dim
        self.temperature = temperature
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim * num_views, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_views)
        )
        
       
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias) 
        
    def forward(self, view_features, batch=None):
        """
        强制节点级融合（按照新框架要求）
        view_features: list of [N, dim] 节点级表示
        batch: [N] 节点所属图ID（用于后续pooling，此处不使用）
        返回: (fused_node_features [N, dim], attention_weights [N, num_views])
        """
        return self._node_level_attention(view_features)
    
    def _graph_level_attention(self, view_features, batch):
        graph_embs = []
        for view_feat in view_features:
            graph_emb = global_mean_pool(view_feat, batch)  
            graph_embs.append(graph_emb)
        
        concatenated = torch.cat(graph_embs, dim=1)  
        
        scores = self.attention_mlp(concatenated) 
       
        attention_weights = F.softmax(scores / self.temperature, dim=1)  
        
        fused_feature = torch.zeros_like(view_features[0])
        batch_size = len(torch.unique(batch))
        
        for i in range(self.num_views):
            node_weights = torch.zeros_like(view_features[0][:, 0:1])  
            
            for graph_idx in range(batch_size):
                graph_mask = (batch == graph_idx)
                graph_weight = attention_weights[graph_idx, i]
                node_weights[graph_mask] = graph_weight
            
            fused_feature += node_weights * view_features[i]
        
        return fused_feature, attention_weights
    
    def _node_level_attention(self, view_features):
       
        concatenated = torch.cat(view_features, dim=1)  
        
        scores = self.attention_mlp(concatenated) 
      
        attention_weights = F.softmax(scores / self.temperature, dim=1)  
        
        fused_feature = torch.zeros_like(view_features[0])
        for i in range(self.num_views):
            weight = attention_weights[:, i].unsqueeze(1)  
            fused_feature += weight * view_features[i]  
        
        return fused_feature, attention_weights