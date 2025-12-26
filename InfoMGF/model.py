import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_learner import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse 
EPS = 1e-12
def _adj_to_dense(adj, n_nodes=None, device=None):
    """
    支持输入：
      - DGLGraph (dgl.DGLGraph / DGLHeteroGraph)
      - torch.sparse_coo_tensor
      - torch.Tensor (dense)
      - edge_index (Tensor 2 x E) 或 tuple (edge_index, num_nodes)
    返回 dense torch.FloatTensor (n_nodes x n_nodes) 在指定 device 上。
    """
    # DGL 图
    if isinstance(adj, (dgl.DGLGraph, dgl.DGLHeteroGraph)):
        g = adj
        n = n_nodes if n_nodes is not None else g.num_nodes()
        device = device if device is not None else (g.device if hasattr(g, 'device') else torch.device('cpu'))
        # 若有边权 'w' 则用它，否则用 1
        if 'w' in g.edata:
            vals = g.edata['w'].to(device)
        else:
            u, v = g.edges()
            vals = torch.ones(u.shape[0], device=device)
        u, v = g.edges()
        indices = torch.stack([u, v], dim=0).to(device)
        A_sparse = torch.sparse_coo_tensor(indices, vals, (n, n), device=device).coalesce()
        return A_sparse.to_dense().float()

    # torch.sparse_coo_tensor
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        device = device if device is not None else adj.device
        return adj.coalesce().to_dense().to(device).float()

    # dense torch tensor
    if isinstance(adj, torch.Tensor):
        device = device if device is not None else adj.device
        return adj.to(device).float()

    # edge_index 2xE
    if isinstance(adj, (list, tuple)) and len(adj) >= 1:
        edge_index = adj[0]
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            if len(adj) == 2 and adj[1] is not None:
                n = adj[1]
            else:
                # infer n from max index
                n = int(edge_index.max().item()) + 1
            device = device if device is not None else edge_index.device
            rows = edge_index[0].to(device)
            cols = edge_index[1].to(device)
            vals = torch.ones(rows.size(0), device=device)
            idx = torch.stack([rows, cols], dim=0)
            A_sparse = torch.sparse_coo_tensor(idx, vals, (n, n), device=device).coalesce()
            return A_sparse.to_dense().float()

    raise TypeError(f"Unsupported adj type: {type(adj)}. Provide DGLGraph / sparse_coo / dense tensor / (edge_index, n_nodes).")


def _adj_to_edge_index_and_weight(adj, device):
    """
    将多种邻接表示统一为 PyG GCNConv 可用的 (edge_index, edge_weight)。
    支持：DGLGraph / sparse_coo / dense / (edge_index, num_nodes / edge_weight)
    """
    if adj is None:
        empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
        return empty_idx, None

    if isinstance(adj, (dgl.DGLGraph, dgl.DGLHeteroGraph)):
        u, v = adj.edges()
        edge_index = torch.stack([u, v], dim=0).to(device)
        edge_weight = adj.edata['w'].to(device) if 'w' in adj.edata else None
        return edge_index, edge_weight

    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        adj = adj.coalesce()
        return adj.indices().to(device), adj.values().to(device)

    if isinstance(adj, torch.Tensor):
        rows, cols = torch.nonzero(adj, as_tuple=True)
        if rows.numel() == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            return empty_idx, None
        edge_index = torch.stack([rows, cols], dim=0).to(device)
        edge_weight = adj[rows, cols].to(device)
        return edge_index, edge_weight

    if isinstance(adj, (list, tuple)) and len(adj) >= 1:
        edge_index = adj[0].to(device)
        edge_weight = None
        # tuple可能是 (edge_index, num_nodes) 或 (edge_index, edge_weight)
        if len(adj) >= 2 and isinstance(adj[1], torch.Tensor):
            if adj[1].dim() == 1 and adj[1].numel() == edge_index.shape[1]:
                edge_weight = adj[1].to(device)
        return edge_index, edge_weight

    raise TypeError(f"Unsupported adj type for edge_index conversion: {type(adj)}")


    
class ViewEncoder(nn.Module):
    """
    三层GCN编码器：在静态视图上进行特征提取
    返回节点级表示，用于后续的多视图融合
    """
    def __init__(self, in_dim, hidden_dim, emb_dim, dropout, sparse=False):
        super().__init__()
        self.dropout = dropout
        self.sparse = sparse  
        
        # 三层GCN（PyG实现，关闭内部自环与归一化，保持与输入邻接一致）
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.conv3 = GCNConv(hidden_dim, emb_dim, add_self_loops=False, normalize=False)
        
        self.act = nn.ReLU()
        
    def forward(self, x, adj):
        """
        x: [N, in_dim] 节点特征
        adj: 静态邻接矩阵（sparse_coo_tensor或DGLGraph）
        返回: [N, emb_dim] 节点级表示
        """
        edge_index, edge_weight = _adj_to_edge_index_and_weight(adj, x.device)

        # Layer 1
        h1 = self.conv1(x, edge_index, edge_weight)
        h1 = self.act(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # Layer 2
        h2 = self.conv2(h1, edge_index, edge_weight)
        h2 = self.act(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + h1  
        
        # Layer 3
        h3 = self.conv3(h2, edge_index, edge_weight)
        
        return h3  #


class GraphClassifierHead(torch.nn.Module):
    def __init__(self, in_dim, nclasses):
        super().__init__()
        out_dim = 1 if nclasses == 2 else nclasses
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)  
    
    def forward(self, graph_emb):
        graph_emb = F.normalize(graph_emb, p=2, dim=1)
        logits = self.linear(graph_emb)
        if self.out_dim == 1:
            return logits.squeeze(-1)
        return logits

def compute_self_supervised_loss(z_specific_adjs, z_fused_adj, specific_adjs, 
                                  temperature=1.0, h=1.0, alpha=1.0, beta=0.0):
    """
    计算自监督损失（LFD + S_high），用于新框架
    
    Args:
        z_specific_adjs: list of [N, d] 每个视图的节点表示
        z_fused_adj: [N, d] 融合后的节点表示
        specific_adjs: list of adj 静态视图邻接矩阵
        temperature: LFD温度参数
        h: （保留参数，未使用）
        alpha: LFD权重
        beta: S_high权重
    
    Returns:
        total_loss, loss_details
    """
    num_views = len(z_specific_adjs)
    
    # 1. LFD损失: 让融合嵌入学习各视图的邻域聚合知识
    lfd_loss_total = 0
    for i in range(num_views):
        lfd_loss = compute_lfd_loss_optimized(
            z_teacher=z_specific_adjs[i],
            h_student=z_fused_adj,
            adj_teacher=specific_adjs[i],
            temperature=temperature
        )
        lfd_loss_total += lfd_loss
    lfd_loss_avg = lfd_loss_total / num_views
    
    # 2. S_High损失: 鼓励每个view在自己的图上保持平滑性（低高频能量）
    s_high_same = [] 
    s_high_cross = [] 
    
    for i in range(num_views):
        s_high_same.append(compute_s_high(z_specific_adjs[i], specific_adjs[i]))
    
        for j in range(num_views):
            if i != j:
                s_high_cross.append(compute_s_high(z_specific_adjs[i], specific_adjs[j]))

    s_high_same_avg = torch.stack(s_high_same).mean()
    s_high_cross_avg = torch.stack(s_high_cross).mean()
   
    s_high_loss_avg = torch.clamp(s_high_same_avg - s_high_cross_avg + 1.0, min=0.0)
    
    total_loss = (alpha * lfd_loss_avg + 
                 beta * s_high_loss_avg)
    
    loss_details = {
        'lfd_loss': lfd_loss_avg,
        's_high_loss': s_high_loss_avg,
        'total_loss': total_loss
    }
    return total_loss, loss_details


#这个是对于融合前后一个点将融合前的一个点聚合它的邻居的特征值然后取平均值用kl散度取拉近这两者之间的距离
def compute_lfd_loss_optimized(z_teacher, h_student, adj_teacher, temperature=1.0):
        device = z_teacher.device
        num_nodes = z_teacher.shape[0]
        adj_teacher = adj_teacher.float()
        if adj_teacher.is_sparse:
            indices = adj_teacher.coalesce().indices()
            values = adj_teacher.coalesce().values()
            self_loop_indices = torch.arange(0, num_nodes, device=device).repeat(2, 1)
            self_loop_values = torch.ones(num_nodes, device=device)

            all_indices = torch.cat([indices, self_loop_indices], dim=1)
            all_values = torch.cat([values, self_loop_values])
            adj_plus_self_loop = torch.sparse_coo_tensor(
                all_indices, all_values, (num_nodes, num_nodes)
            ).coalesce()
        else:
            adj_plus_self_loop = adj_teacher + torch.eye(num_nodes, device=device)
        if adj_plus_self_loop.is_sparse:
            row, col = adj_plus_self_loop.indices()
            deg = torch.zeros(num_nodes, device=device)
            deg = deg.scatter_add(0, row, adj_plus_self_loop.values())
            deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
            norm_values = deg_inv_sqrt[row] * adj_plus_self_loop.values() * deg_inv_sqrt[col]
            adj_normalized = torch.sparse_coo_tensor(
                adj_plus_self_loop.indices(), norm_values, adj_plus_self_loop.size()
            )
            z_teacher_agg = torch.sparse.mm(adj_normalized, z_teacher)
        else:

            deg = torch.diag(adj_plus_self_loop.sum(dim=1))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            adj_normalized = torch.mm(torch.mm(deg_inv_sqrt, adj_plus_self_loop), deg_inv_sqrt)
            z_teacher_agg = torch.mm(adj_normalized, z_teacher)

        p_teacher = F.softmax(z_teacher_agg / temperature, dim=1).detach()
        log_p_student = F.log_softmax(h_student / temperature, dim=1)

        loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean', log_target=False)
        loss = loss * (temperature ** 2)      
        return loss
    #下面这个是对于S_high的计算，然后它的原理是让内部的图信号（经过卷积得到的节点嵌入）
    #去放到全局视图上去评估 同样的交叉操作一下
    #这里是到loss的时候要进行相减的操作
def compute_s_high(x, adj):
    """
    计算图信号的高频能量度量（S_high）。
    x: (N, F) 节点嵌入 torch.Tensor
    adj: DGLGraph / sparse coo / dense tensor / (edge_index, n_nodes)
    返回 scalar torch.Tensor
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()

    device = x.device
    N = x.size(0)

    A = _adj_to_dense(adj, n_nodes=N, device=device)  # (N,N) float
    # 如果 A 全零（孤立图），返回 0 防止数值异常
    if torch.allclose(A, torch.zeros_like(A)):
        return torch.tensor(0.0, device=device)

    # 归一化对称拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1)  # (N,)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    I = torch.eye(N, device=device, dtype=A_norm.dtype)
    L = I - A_norm  # (N,N)

    # numerator per feature: x_f^T L x_f
    XLX = x.t() @ L @ x  # (F, F)
    numerator = torch.diag(XLX)  # (F,)
    denom_mat = x.t() @ x  # (F, F)
    denominator = torch.diag(denom_mat)  # (F,)

    s_high_vals = numerator / (denominator + EPS)
    s_high = s_high_vals.mean()
    return s_high
