import argparse
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from data_loader import BrainGraphDataset
from model import *
from graph_learner import *
from utils import *
from params import *
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans as KMeans_py
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import os
import json
from collections import Counter
import pickle

EOS = 1e-10
args = set_params()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', weight=None):
        """
        alpha / weight: per-class weighting (tensor/list/float). If `alpha` is None, `weight`
                        is used as alpha for compatibility with caller.
        """
        super().__init__()
        if alpha is None and weight is not None:
            alpha = weight
        if alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha_tensor)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Supports binary (logits shape [B] or [B,1]) and multi-class (logits shape [B, C]).
        targets: long tensor shape [B].
        """
        targets = targets.long()
        # Multi-class focal
        if inputs.dim() > 1 and inputs.size(1) > 1:
            log_probs = F.log_softmax(inputs, dim=1)
            probs = log_probs.exp()
            ce_loss = F.nll_loss(log_probs, targets, reduction='none')
            pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
            focal = (1 - pt) ** self.gamma * ce_loss
            if self.alpha is not None:
                alpha_t = self.alpha.to(inputs.device)
                if alpha_t.numel() > 1:
                    focal = focal * alpha_t.gather(0, targets)
                else:
                    focal = focal * alpha_t
        else:
            logits = inputs.view(-1)
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
            pt = torch.exp(-ce_loss)
            focal = (1 - pt) ** self.gamma * ce_loss
            if self.alpha is not None:
                alpha_t = self.alpha.to(inputs.device)
                if alpha_t.numel() > 1:
                    # assume alpha[0]=neg, alpha[1]=pos
                    alpha_targets = alpha_t[targets.clamp(max=alpha_t.numel() - 1)]
                    focal = focal * alpha_targets
                else:
                    focal = focal * alpha_t

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

def quick_diagnose(logits, batch, model_modules, optimizer, criterion, attention_weights=None, graph_emb=None, 
                   attention_balance_loss=None, view_diversity_loss=None):
    # Support both multi-class and binary (single-logit) cases
    if logits.dim() == 2 and logits.size(1) > 1:
        preds = logits.argmax(dim=1)
        print("\n" + "="*50)
        print("Pred distribution:", torch.bincount(preds).cpu().numpy())
        print("Label distribution:", torch.bincount(batch.y.view(-1)).cpu().numpy())
        probs = F.softmax(logits, dim=1)
        class_conf = probs.mean(dim=0).detach().cpu().numpy()
        print(f"Avg pred confidence per class: {[f'{c:.3f}' for c in class_conf]}")
        print("Logits stats mean/std/min/max:", float(logits.mean().item()), float(logits.std().item()),
              float(logits.min().item()), float(logits.max().item()))
        print("Logits per class:")
        for i in range(logits.shape[1]):
            class_logits = logits[:, i]
            print(f"  Class {i}: mean={class_logits.mean().item():.4f}, std={class_logits.std().item():.4f}")
    else:
        probs_pos = torch.sigmoid(logits.view(-1))
        preds = (probs_pos > 0.5).long()
        print("\n" + "="*50)
        print("Pred distribution:", torch.bincount(preds).cpu().numpy())
        print("Label distribution:", torch.bincount(batch.y.view(-1)).cpu().numpy())
        print(f"Avg positive probability: {probs_pos.mean().item():.3f}")
        print("Logits stats mean/std/min/max:", float(logits.mean().item()), float(logits.std().item()),
              float(logits.min().item()), float(logits.max().item()))
    
    per_sample_std = logits.std(dim=1)
    print("Per-sample logits std mean:", float(per_sample_std.mean().item()),
          "frac zero-std:", float((per_sample_std==0).float().mean().item()))
    # Compute criterion loss preview with correct shapes/types
    try:
        if logits.dim() == 2 and logits.size(1) > 1:
            loss_val = criterion(logits, batch.y.view(-1))
        else:
            loss_val = F.binary_cross_entropy_with_logits(logits.view(-1), batch.y.float())
    except Exception as e:
        loss_val = torch.tensor(float('nan'))
    print("Loss value:", float(loss_val.item()))
    
    # 分析图嵌入的区分度
    if graph_emb is not None:
        print("\nGraph embeddings stats:")
        print(f"  Mean: {graph_emb.mean().item():.4f}, Std: {graph_emb.std().item():.4f}")
        
        graph_emb_norm = F.normalize(graph_emb, p=2, dim=1)
        sim_matrix = torch.mm(graph_emb_norm, graph_emb_norm.t())
       
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        off_diag_sim = sim_matrix[mask]
        print(f"  Inter-graph similarity: mean={off_diag_sim.mean().item():.4f}, std={off_diag_sim.std().item():.4f}")
        if off_diag_sim.mean().item() > 0.9:
            print(" WARNING: Graph embeddings are too similar! Model may not distinguish different graphs.")
    
    if attention_weights is not None:
        print("\nAttention weights stats:")
        print("  Mean per view:", attention_weights.mean(dim=0).detach().cpu().numpy())
        print("  Std per view:", attention_weights.std(dim=0).detach().cpu().numpy())
  
        attn_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(float(attention_weights.size(1))))
        print(f"  Attention entropy: {attn_entropy.item():.4f} / {max_entropy.item():.4f} (higher=more balanced)")
        if attention_weights.mean(dim=0).max().item() > 0.75:
            print(" WARNING: Attention is heavily biased towards one view!")
  
    if attention_balance_loss is not None:
        print(f"\nRegularization losses:")
        print(f"  Attention balance loss: {attention_balance_loss.item():.4f} (lower=more balanced)")
    if view_diversity_loss is not None:
        print(f"  View diversity loss: {view_diversity_loss.item():.4f} (lower=more diverse)")
    
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    for name, m in model_modules.items():
        for n,p in m.named_parameters():
            if id(p) not in opt_param_ids:
                print("WARNING: param not in optimizer:", name, n)

    if 'fusion' in model_modules:
        print("\nFusion layer gradients:")
        for n,p in model_modules['fusion'].named_parameters():
            grad_norm = 0.0 if p.grad is None else float(p.grad.norm().item())
            print(f"  {n}: grad_norm={grad_norm:.6f}")
    print("="*50 + "\n")
    return preds
class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        self.training = False
        self.writer = None

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def callculate_detailed(self,all_labels,all_preds,all_probs,n_classes,trial,split='test',log_file=None,epoch=None):
        accuracy = accuracy_score(all_labels, all_preds)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        try:
            if n_classes == 2:
                if all_probs.ndim == 1:
                    auc_roc_macro = roc_auc_score(all_labels, all_probs)
                else:
                    auc_roc_macro = roc_auc_score(all_labels, all_probs[:, 1])
                auc_roc_micro = auc_roc_macro
            else:
                
                auc_roc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                auc_roc_micro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='micro')
        except Exception as e:
            error_msg = f"AUC calculation error: {str(e)}"
            if log_file:
                log_file.write(f"{error_msg}\n")
            else:
                print(f"⚠️ {error_msg}")

            unique_labels = np.unique(all_labels)
            if len(unique_labels) < 2:
                print(f"   Reason: Only one class present in y_true. Classes found: {unique_labels}")
            
            auc_roc_macro = 0.0
            auc_roc_micro = 0.0
        
        cm = confusion_matrix(all_labels, all_preds)
        
        # 准备输出内容
        epoch_info = f" - Epoch {epoch}" if epoch is not None else ""
        output = f"\n{'='*80}\n"
        output += f"=== {split.upper()} Metrics - Trial {trial}{epoch_info} ===\n"
        output += f"{'='*80}\n"
        output += f"Accuracy: {accuracy:.4f}\n"
        output += f"Precision - Macro: {precision_macro:.4f}, Micro: {precision_micro:.4f}\n"
        output += f"Recall    - Macro: {recall_macro:.4f}, Micro: {recall_micro:.4f}\n"
        output += f"F1-Score  - Macro: {f1_macro:.4f}, Micro: {f1_micro:.4f}\n"
        output += f"AUC-ROC   - Macro: {auc_roc_macro:.4f}, Micro: {auc_roc_micro:.4f}\n"
        output += f"\nPer-class metrics:\n"
        for i in range(n_classes):
            output += f"  Class {i}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}\n"
        output += f"\nConfusion Matrix (shape: {cm.shape}):\n"
        output += f"{cm}\n"
        output += f"{'='*80}\n"
        
        if log_file:
            log_file.write(output)
            log_file.flush()  # 立即写入磁盘
        
        # 控制台只显示简要信息
        print(f"[{split.upper()}] Trial {trial}{epoch_info}: Acc={accuracy:.4f}, F1_macro={f1_macro:.4f}, AUC_macro={auc_roc_macro:.4f}")
        return {
            'acc': accuracy,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'auc_macro': auc_roc_macro,
            'auc_micro': auc_roc_micro,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }

    def test_cls_graphlevel(self, view_encoders, classifier, loader, attention_fusion, args,
                            trial=0, split='test', log_file=None, epoch=None, threshold=None,
                            return_probs=False):
        view_encoders.eval()
        classifier.eval()
        attention_fusion.eval()
        
        all_preds, all_labels = [], []
        all_probs = []
        total_loss = 0.0
        nb = 0
        n_classes = None
        
        device = self.device if hasattr(self, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                adjs_list, node2graph, N = build_adjs_from_batch(batch, device=self.device)
                feat = batch.x

                z_specifics = []
                for i in range(len(adjs_list)):
                    z_view = view_encoders(feat, adjs_list[i])
                    z_specifics.append(z_view)
                
                # ====== 节点级融合 ======
                fused_z, attention_weight = attention_fusion(z_specifics, batch=node2graph)
                
                # ====== 图级pooling ======
                graph_emb = global_mean_pool(fused_z, node2graph)
                graph_emb = F.normalize(graph_emb, p=2, dim=1)
                
                # ====== 分类 ======
                logits = classifier(graph_emb)
                if n_classes is None:
                    if logits.dim() == 2 and logits.size(1) > 1:
                        n_classes = logits.size(1)
                    else:
                        n_classes = 2
                if n_classes == 2 and (logits.dim() == 1 or logits.size(1) == 1):
                    # Binary: single-logit
                    loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch.y.float())
                    prob_pos = torch.sigmoid(logits.view(-1))
                    th = 0.5 if threshold is None else float(threshold)
                    preds = (prob_pos > th).long().cpu().numpy()
                    probs_np = prob_pos.detach().cpu().numpy()
                    probs_2col = np.stack([1.0 - probs_np, probs_np], axis=1)
                    all_probs.append(probs_2col)
                else:
                    probs = F.softmax(logits, dim=1)
                    loss = F.cross_entropy(logits, batch.y.view(-1))
                    
                    # 修复：如果是二分类且双输出，应用阈值
                    if n_classes == 2:
                        prob_pos = probs[:, 1] # 取出正类概率
                        th = 0.5 if threshold is None else float(threshold)
                        preds = (prob_pos > th).long().cpu().numpy()
                    else:
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        
                    all_probs.append(probs.cpu().numpy())
                
                total_loss += float(loss.item())
                nb += 1
                all_preds.extend(preds.tolist())
                all_labels.extend(batch.y.cpu().numpy().reshape(-1).tolist())
                
        
        if nb == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        avg_loss = total_loss / nb
        all_probs = np.concatenate(all_probs, axis=0)
        
        detailed_metrics = self.callculate_detailed(all_labels, all_preds, all_probs, n_classes, trial, split, log_file=log_file, epoch=epoch)
        
        if return_probs:
            return avg_loss, detailed_metrics, {
                'labels': np.array(all_labels),
                'probs': np.array(all_probs),
                'n_classes': n_classes
            }
        return avg_loss, detailed_metrics

    def search_best_threshold(self, view_encoders, classifier, loader, attention_fusion, args,
                              trial=0, split='val', threshold_min=None, threshold_max=None, step=None, verbose=False):
        """
        Search best decision threshold using F1-macro (binary).
        Threshold candidates come from validation probabilities.
        """
        loss0, metrics0, raw = self.test_cls_graphlevel(
            view_encoders, classifier, loader, attention_fusion, args,
            trial=trial, split=split, log_file=None, epoch=None, threshold=None,
            return_probs=True
        )
        labels = raw['labels']
        probs = raw['probs']
        n_classes = raw.get('n_classes', 2)

        if n_classes != 2 or probs.shape[1] < 2:
            best_results = dict(metrics0)
            best_results['best_threshold'] = 0.5
            return best_results, 0.5, loss0

        prob_pos = probs[:, 1]
        prob_unique = np.unique(np.sort(prob_pos))

        thresholds = []
        if prob_unique.size >= 2:
            mids = (prob_unique[:-1] + prob_unique[1:]) / 2.0
            thresholds.extend(mids.tolist())
        thresholds.extend([0.0, 1.0, prob_unique[0] - 1e-6, prob_unique[-1] + 1e-6])
        thresholds = sorted({float(np.clip(t, 0.0, 1.0)) for t in thresholds})

        best_score = -1.0
        best_threshold = 0.5
        best_results = metrics0
        best_loss = loss0

        for th in thresholds:
            preds = (prob_pos > th).astype(int)
            metrics = self.callculate_detailed(
                labels.tolist(), preds.tolist(), probs, n_classes,
                trial, split, log_file=None, epoch=None
            )
            f1m = metrics.get('f1_macro', 0.0)
            if f1m > best_score:
                best_score = f1m
                best_threshold = th
                best_results = metrics
                best_loss = loss0

        best_results = dict(best_results)
        best_results['best_threshold'] = best_threshold
        return best_results, best_threshold, best_loss

    # ==== 读取 processed_{split}.pkl 并转为 Data 列表 ====
    def load_processed_dict(self, root, split):
        p = os.path.join(root, f"processed_{split}.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found")
        with open(p, 'rb') as f:
            return pickle.load(f)

    def dict_to_datalist(self, data_dict):
        data_list = []
        for sid, graph_data in data_dict.items():
            node_features = torch.from_numpy(graph_data['node_feats']).float()
            raw_label = int(graph_data['label'])
            A_intra = graph_data['A_intra']
            A_global = graph_data['A_global']
            edge_index_intra = dense_to_sparse(torch.from_numpy(A_intra).float())[0]
            edge_index_global = dense_to_sparse(torch.from_numpy(A_global).float())[0]
            data = Data(
                x=node_features,
                y=torch.tensor(raw_label, dtype=torch.long),
                edge_index_intra=edge_index_intra,
                edge_index_global=edge_index_global,
                sid=sid,
                original_label=raw_label
            )
            data_list.append(data)
        return data_list

    def train_single(self, args):
        print("="*80)
        print("使用 train/val/test 原始划分（不做 K-Fold）")
        print("="*80)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        torch.cuda.set_device(args.gpu)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/{args.dataset}_single_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        
        results_log_path = f"results_{args.dataset}_single_{timestamp}.txt"
        results_log = open(results_log_path, 'w', encoding='utf-8')
        results_log.write(f"Single-Split Results (train/val/test)\n")
        results_log.write(f"无动态图学习的\n")
        results_log.write(f"先融合再池化\n")
        results_log.write(f"Dataset: {args.dataset}\n")
        results_log.write(f"Timestamp: {timestamp}\n")
        results_log.write(f"loss_a1: {args.lambda1}\n")
        results_log.write(f"loss_a2: {args.lambda2}\n")
        results_log.write(f"dropout: {args.dropout}\n")
        results_log.write(f"loss_type: {args.loss_type}\n")
        results_log.write(f"focal_gamma: {args.focal_gamma}\n")
        results_log.write(f"epoch: {args.epochs}\n")
        results_log.write(f"{'='*80}\n\n")
        results_log.flush()
        
        # 使用预处理数据路径（train/val/test）
        data_root = args.out_dir
        
        # 加载Test集 (只需加载一次)
        # 注意：使用label_mode='binary'将多分类标签转为二分类（0=正常, 1-4=异常）
        test_dataset = BrainGraphDataset(
            root=data_root,
            split='test',
            label_mode='binary',  # 二分类：0=正常, 1=异常
            smart_resample=False       # 禁用Dataset内的重采样
        )
        
        print(f"\n{'='*80}")
        print(f"Single-split training using preprocessed data")
        print(f"Data root: {data_root}")
        print(f"Test set: {len(test_dataset)} samples")
        print(f"{'='*80}\n")
        
        # 获取特征维度和类别数 (从test集推断)
        sample = test_dataset[0]
        nfeats = sample.x.shape[1]
        nclasses = len(set([data.y.item() for data in test_dataset]))
        num_views = 2  # intra + global
        
        print(f"Model config: nfeats={nfeats}, nclasses={nclasses}, num_views={num_views}")

        def compute_class_weights_from_dataset(dataset, nclasses, device):
            labels = [int(data.y.item()) for data in dataset]
            counts = np.bincount(labels, minlength=nclasses).astype(np.float32)
            weights = counts.sum() / (counts + 1e-6)
            weights = weights / np.mean(weights)
            pos_weight = None
            if nclasses == 2:
                neg = counts[0]
                pos = counts[1] if len(counts) > 1 else 0.0
                pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32, device=device)
            return torch.tensor(weights, dtype=torch.float32, device=device), counts, pos_weight

        fold_results = []

        for fold in range(1):

            train_dataset = BrainGraphDataset(
                root=data_root,
                split='train',
                label_mode='binary',   # 二分类
                smart_resample=False,  # 禁用二次重采样！
                augment=True,          # 启用训练集动态增强
                drop_edge_p=0.05,      
                noise_std=0.05          
            )
            
            val_dataset = BrainGraphDataset(
                root=data_root,
                split='val',
                label_mode='binary',
                smart_resample=False
            )
            
            # 统计标签分布
            train_labels = [data.y.item() for data in train_dataset]
            val_labels = [data.y.item() for data in val_dataset]
            
            print(f"\n{'='*80}")
            print(f"Single run")
            print(f"{'='*80}")
            print(f"Train: {len(train_dataset)} samples")
            print(f"  Label distribution: {dict(Counter(train_labels))}")
            print(f"Val: {len(val_dataset)} samples")
            print(f"  Label distribution: {dict(Counter(val_labels))}")
            
            results_log.write(f"\n{'='*80}\n")
            results_log.write(f"Single run\n")
            results_log.write(f"Train: {len(train_dataset)} samples, distribution: {dict(Counter(train_labels))}\n")
            results_log.write(f"Val: {len(val_dataset)} samples, distribution: {dict(Counter(val_labels))}\n")
            results_log.flush()

            class_weights = None
            pos_weight = None
            if args.loss_type in ['weighted_ce', 'focal']:
                class_weights, weight_counts, pos_weight = compute_class_weights_from_dataset(
                    train_dataset, nclasses, device
                )
                print(f"Class weights: {class_weights.detach().cpu().numpy().tolist()}")
                results_log.write(f"Class weights: {class_weights.detach().cpu().numpy().tolist()}\n")
                results_log.flush()

            if args.loss_type == 'ce':
                if nclasses == 2:
                    criterion = torch.nn.BCEWithLogitsLoss()
                else:
                    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
            elif args.loss_type == 'weighted_ce':
                if nclasses == 2:
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
            
            self.setup_seed(fold)
            
            # 为每个 Fold 创建单独的 TensorBoard writer
            fold_log_dir = f"{log_dir}/fold_{fold+1}"
            fold_writer = SummaryWriter(log_dir=fold_log_dir)
            
            # 创建当前fold的数据加载器
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            

            view_encoders = ViewEncoder(
                    in_dim=nfeats, hidden_dim=args.hidden_dim,
                    emb_dim=args.emb_dim, dropout=args.dropout, sparse=args.sparse
                ).to(device)
  
            
            attention_fusion = AttentionFusion(input_dim=args.emb_dim, num_views=num_views).to(device)
            classifier = GraphClassifierHead(in_dim=args.emb_dim, nclasses=nclasses).to(device)
            
            # 优化器参数
            params = []
            params.append({'params': view_encoders.parameters()})
            params.append({'params': classifier.parameters()})
            params.append({'params': attention_fusion.parameters()})
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.w_decay)
            
            total_epochs = max(1, args.epochs)
            warmup_epochs = max(0, min(args.warmup_epochs, total_epochs))
            if warmup_epochs > 0:
                warmup = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / float(warmup_epochs))
            else:
                warmup = LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=args.min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
            lr_mode = 'cosine'
            best_val = -1.0
            best_state = None
            best_epoch = 0
            
            # 用于记录每个 epoch 的损失（用于绘图）
            epoch_losses = {
                'total': [],
                'supervised': [],
                'self_supervised': [],
                'lfd': [],
                's_high': [],
                'train_acc': []
            }

            patience_counter = 0
            stop_training = False
            
            # 训练循环
            for epoch in range(1, args.epochs + 1):

                view_encoders.train()
                classifier.train()
                attention_fusion.train()
                
                total_loss = 0.0
                total_sup_loss = 0.0
                total_self_loss = 0.0
                total_lfd_loss = 0.0
                total_s_high_loss = 0.0
                train_correct = 0
                train_total = 0
                n_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    batch = batch.to(device)
                    adj_list, node2graph, N = build_adjs_from_batch(batch, device)
                    feat = batch.x
                    
                    z_specifics = []  # 节点级表示
                    for i in range(len(adj_list)):
                        # 直接在静态视图上编码
                        z_view = view_encoders(feat, adj_list[i])
                        z_specifics.append(z_view)
                    
                    # ====== 节点级多视图融合 ======
                    fused_z, attention_weights = attention_fusion(z_specifics, batch=node2graph)
                    
                    # ====== 图级pooling ======
                    graph_emb = global_mean_pool(fused_z, node2graph)
                    graph_emb = F.normalize(graph_emb, p=2, dim=1)
                    
                    # ====== 分类 ======
                    logits = classifier(graph_emb)
                    if torch.isnan(logits).any():
                        continue
                    
                    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                        loss_sup = criterion(logits.view(-1), batch.y.float())
                        prob_pos = torch.sigmoid(logits.view(-1))
                        preds = (prob_pos > 0.5).long()
                    else:
                        loss_sup = criterion(logits, batch.y.view(-1))
                        preds = logits.argmax(dim=1)
                    train_correct += (preds == batch.y.view(-1)).sum().item()
                    train_total += batch.y.view(-1).size(0)
                    
                    # ====== 自监督损失（LFD + S_high） ======
                    if args.loss_mode == 'ce_only':
                        loss = loss_sup
                        loss_self_val = 0.0
                        lfd_loss_val = 0.0
                        s_high_loss_val = 0.0
                    else:
                        # 计算LFD和S_high（使用静态adj_list）
                        loss_self, loss_details = compute_self_supervised_loss(
                            z_specifics, fused_z, adj_list, args.tau, args.h, args.alpha, args.beta
                        )
                        loss = loss_sup + args.lambda1 * loss_details['lfd_loss'] + args.lambda2 * loss_details['s_high_loss']
                        loss_self_val = loss_self.item() if torch.is_tensor(loss_self) else loss_self
                        lfd_loss_val = loss_details['lfd_loss'].item() if torch.is_tensor(loss_details['lfd_loss']) else loss_details['lfd_loss']
                        s_high_loss_val = loss_details['s_high_loss'].item() if torch.is_tensor(loss_details['s_high_loss']) else loss_details['s_high_loss']
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(view_encoders.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(attention_fusion.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_sup_loss += loss_sup.item()
                    total_self_loss += loss_self_val
                    total_lfd_loss += lfd_loss_val
                    total_s_high_loss += s_high_loss_val
                    n_batches += 1
                
                # 计算平均损失
                avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
                avg_sup_loss = total_sup_loss / n_batches if n_batches > 0 else 0.0
                avg_self_loss = total_self_loss / n_batches if n_batches > 0 else 0.0
                avg_lfd_loss = total_lfd_loss / n_batches if n_batches > 0 else 0.0
                avg_s_high_loss = total_s_high_loss / n_batches if n_batches > 0 else 0.0
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                
                # TensorBoard记录训练损失
                fold_writer.add_scalar('Training/Loss', avg_loss, epoch)
                fold_writer.add_scalar('Training/Loss_CE', avg_sup_loss, epoch)
                fold_writer.add_scalar('Training/Loss_Self', avg_self_loss, epoch)
                fold_writer.add_scalar('Training/Loss_LFD', avg_lfd_loss, epoch)
                fold_writer.add_scalar('Training/Loss_S_High', avg_s_high_loss, epoch)
                fold_writer.add_scalar('Training/Accuracy', train_acc, epoch)
                fold_writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], epoch)
                
                # 验证
                if epoch % args.eval_freq == 0:
                    # 使用阈值搜索
                    val_metrics, best_th, val_loss = self.search_best_threshold(
                        view_encoders, classifier, val_loader, 
                        attention_fusion, args, trial=fold, split='val',
                        threshold_min=args.threshold_min, threshold_max=args.threshold_max,
                        step=args.threshold_step, verbose=False
                    )
                    current_f1 = val_metrics['f1_macro']  # 使用F1作为主要指标
                    
                    # Balanced Accuracy = 0.5*(recall0+recall1) - 仅用于记录
                    cm = val_metrics.get('confusion_matrix', None)
                    if cm is None or cm.shape != (2, 2):
                        recall_per_class = val_metrics.get('recall_per_class', [0, 0])
                        recall0 = recall_per_class[0] if len(recall_per_class) > 0 else 0.0
                        recall1 = recall_per_class[1] if len(recall_per_class) > 1 else 0.0
                    else:
                        tn, fp, fn, tp = cm.ravel()
                        recall0 = tn / (tn + fp + 1e-12)
                        recall1 = tp / (tp + fn + 1e-12)
                    current_ba = 0.5 * (recall0 + recall1)
                    
                    # TensorBoard记录验证指标
                    fold_writer.add_scalar('Validation/Loss', val_loss, epoch)
                    fold_writer.add_scalar('Validation/Accuracy', val_metrics['acc'], epoch)
                    fold_writer.add_scalar('Validation/F1_Macro', val_metrics['f1_macro'], epoch)
                    fold_writer.add_scalar('Validation/Balanced_Acc', current_ba, epoch)
                    fold_writer.add_scalar('Validation/Precision_Macro', val_metrics['precision_macro'], epoch)
                    fold_writer.add_scalar('Validation/Recall_Macro', val_metrics['recall_macro'], epoch)
                    fold_writer.add_scalar('Validation/AUC_Macro', val_metrics['auc_macro'], epoch)
                    
                    print(f"Fold {fold+1} Epoch {epoch}: Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={current_f1:.4f}, Val BA={current_ba:.4f}, Val Acc={val_metrics['acc']:.4f}, Best Th={best_th:.2f}")
                    
                    # 使用F1作为选择标准（而不是BA）
                    if current_f1 > best_val:
                        best_val = current_f1
                        best_epoch = epoch
                        best_state = {
                            'view_encoders': copy.deepcopy(view_encoders.state_dict()),
                            'classifier': copy.deepcopy(classifier.state_dict()),
                            'attention_fusion': copy.deepcopy(attention_fusion.state_dict()),
                            'best_threshold': best_th
                        }
                        print(f"  -> New best! Saved (F1={current_f1:.4f}).")
                        patience_counter = 0
                    else:
                        # 如果是第一次验证且没有best_state，保存当前模型
                        if best_state is None:
                            best_val = current_f1
                            best_epoch = epoch
                            best_state = {
                                'view_encoders': copy.deepcopy(view_encoders.state_dict()),
                                'classifier': copy.deepcopy(classifier.state_dict()),
                                'attention_fusion': copy.deepcopy(attention_fusion.state_dict()),
                                'best_threshold': best_th
                            }
                            print(f"  -> Initial model saved.")
                        patience_counter += 1
                        if epoch >= args.min_training_epochs and patience_counter >= args.patience:
                            print(f"  -> Early stopping at epoch {epoch} (no improvement for {patience_counter} evals).")
                            stop_training = True

                if stop_training:
                    scheduler.step()
                    break
                
                # 每个epoch结束后更新学习率（余弦退火）
                scheduler.step()
            
            # 恢复最佳模型并测试 (使用F1而不是BA)
            best_th_final = 0.5
            if best_state is not None:
                view_encoders.load_state_dict(best_state['view_encoders'])
                classifier.load_state_dict(best_state['classifier'])
                attention_fusion.load_state_dict(best_state['attention_fusion'])
                if 'best_threshold' in best_state:
                    best_th_final = best_state['best_threshold']
                print(f"\n✓ Loaded best model from epoch {best_epoch} (Val F1: {best_val:.4f}, Th: {best_th_final:.2f})")
            else:
                print(f"\n⚠️ Warning: No improvement found during training. Using final model state.")
                # 即使没有找到更好的模型，也保存当前状态作为best_state
                best_val = 0.0
                best_epoch = args.epochs
            
            test_loss, test_metrics = self.test_cls_graphlevel(
                view_encoders, classifier, test_loader,
                attention_fusion, args, fold, 'test', log_file=results_log, epoch=None, threshold=best_th_final
            )
            
            # TensorBoard记录测试结果（作为该折的最终指标）
            fold_writer.add_scalar('Test/Accuracy', test_metrics['acc'], best_epoch)
            fold_writer.add_scalar('Test/F1_Macro', test_metrics['f1_macro'], best_epoch)
            fold_writer.add_scalar('Test/Precision_Macro', test_metrics['precision_macro'], best_epoch)
            fold_writer.add_scalar('Test/Recall_Macro', test_metrics['recall_macro'], best_epoch)
            fold_writer.add_scalar('Test/AUC_Macro', test_metrics['auc_macro'], best_epoch)
            
            # 记录超参数和最终指标
            fold_writer.add_hparams(
                {
                    'lr': args.lr,
                    'hidden_dim': args.hidden_dim,
                    'lambda1': args.lambda1,
                    'lambda2': args.lambda2,
                    'fold': fold + 1,
                    'best_epoch': best_epoch
                },
                {
                    'hparam/val_ba': best_val,
                    'hparam/test_acc': test_metrics['acc'],
                    'hparam/test_f1': test_metrics['f1_macro'],
                    'hparam/test_auc': test_metrics['auc_macro']
                }
            )
            
            # 关闭该折的 TensorBoard writer
            fold_writer.close()
            print(f"  TensorBoard log saved to: {log_dir}/fold_{fold+1}")
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_ba': best_val,
                'best_epoch': best_epoch,
                'test_acc': test_metrics['acc'],
                'test_f1': test_metrics['f1_macro'],
                'test_auc': test_metrics['auc_macro'],
                'metrics': test_metrics
            })
            
            print(f"\nFold {fold+1} Results:")
            print(f"  Best Val BA: {best_val:.4f} (Epoch {best_epoch})")
            print(f"  Test Acc: {test_metrics['acc']:.4f}")
            print(f"  Test F1: {test_metrics['f1_macro']:.4f}")
            print(f"  Test AUC: {test_metrics['auc_macro']:.4f}")
        
        # 汇总单次结果
        if fold_results:
            r = fold_results[0]
            summary = f"""
{'='*80}
SINGLE SPLIT SUMMARY
{'='*80}
TensorBoard Log Directory: {log_dir}

Results:
  Val BA: {r['best_val_ba']:.4f} (Best Epoch: {r['best_epoch']})
  Test Acc: {r['test_acc']:.4f}
  Test F1:  {r['test_f1']:.4f}
  Test AUC: {r['test_auc']:.4f}
{'='*80}
"""
            print(summary)
            results_log.write(summary)
        results_log.close()
        print(f"\nResults saved to: {results_log_path}")
        print(f"To view TensorBoard, run: tensorboard --logdir={log_dir}")

    def train_kfold_random(self, args):
        """
        K-Fold Random Validation: 
        - 进行k次独立实验
        - 每次实验：将所有数据混合后随机划分为 train:val:test = 7:1:2
        - 使用验证集F1值选择最佳模型
        - 在测试集上评估
        - 最后报告k次实验的平均结果和标准差
        """
        print("="*80)
        print(f"使用 K-Fold Random Validation (k={args.num_folds})")
        print("每个fold: 7:1:2 随机划分 train/val/test")
        print("选择标准: 验证集F1-macro")
        print("="*80)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        torch.cuda.set_device(args.gpu)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/{args.dataset}_kfold_random_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        
        results_log_path = f"results_{args.dataset}_kfold_random_{timestamp}.txt"
        results_log = open(results_log_path, 'w', encoding='utf-8')
        results_log.write(f"K-Fold Random Validation Results (k={args.num_folds})\n")
        results_log.write(f"Selection criterion: F1-macro on validation set\n")
        results_log.write(f"Split ratio: train:val:test = 7:1:2\n")
        results_log.write(f"Dataset: {args.dataset}\n")
        results_log.write(f"Timestamp: {timestamp}\n")
        results_log.write(f"loss_a1: {args.lambda1}\n")
        results_log.write(f"loss_a2: {args.lambda2}\n")
        results_log.write(f"dropout: {args.dropout}\n")
        results_log.write(f"loss_type: {args.loss_type}\n")
        results_log.write(f"focal_gamma: {args.focal_gamma}\n")
        results_log.write(f"epochs: {args.epochs}\n")
        results_log.write(f"{'='*80}\n\n")
        results_log.flush()
        
        # 加载所有数据（train+val+test混合）
        data_root = args.out_dir
        print(f"\nLoading all data from: {data_root}")
        
        # 分别加载三个集合
        train_dict = self.load_processed_dict(data_root, 'train')
        val_dict = self.load_processed_dict(data_root, 'val')
        test_dict = self.load_processed_dict(data_root, 'test')
        
        # 合并所有数据
        all_data_dict = {}
        all_data_dict.update(train_dict)
        all_data_dict.update(val_dict)
        all_data_dict.update(test_dict)
        
        # 转换为Data列表
        all_data_list = self.dict_to_datalist(all_data_dict)
        
        print(f"Total samples loaded: {len(all_data_list)}")
        all_labels = [int(data.y.item()) for data in all_data_list]
        print(f"Label distribution: {dict(Counter(all_labels))}")
        
        # 获取特征维度和类别数
        sample = all_data_list[0]
        nfeats = sample.x.shape[1]
        nclasses = len(set(all_labels))
        num_views = 2  # intra + global
        
        print(f"Model config: nfeats={nfeats}, nclasses={nclasses}, num_views={num_views}\n")
        
        # 存储每个fold的结果
        fold_results = []
        
        # 进行k次独立实验
        for fold in range(args.num_folds):
            print(f"\n{'='*80}")
            print(f"Fold {fold+1}/{args.num_folds}")
            print(f"{'='*80}")
            
            # 设置该fold的随机种子
            fold_seed = args.seed + fold
            self.setup_seed(fold_seed)
            np.random.seed(fold_seed)
            random.seed(fold_seed)
            
            # 随机打乱数据
            indices = np.arange(len(all_data_list))
            np.random.shuffle(indices)
            
            # 计算划分点: 7:1:2
            n_total = len(all_data_list)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.1)
            # n_test = n_total - n_train - n_val  # 剩余的作为测试集
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train+n_val]
            test_indices = indices[n_train+n_val:]
            
            # 创建数据子集
            train_data = [all_data_list[i] for i in train_indices]
            val_data = [all_data_list[i] for i in val_indices]
            test_data = [all_data_list[i] for i in test_indices]
            
            # 统计标签分布
            train_labels = [int(d.y.item()) for d in train_data]
            val_labels = [int(d.y.item()) for d in val_data]
            test_labels = [int(d.y.item()) for d in test_data]
            
            print(f"Train: {len(train_data)} samples, distribution: {dict(Counter(train_labels))}")
            print(f"Val:   {len(val_data)} samples, distribution: {dict(Counter(val_labels))}")
            print(f"Test:  {len(test_data)} samples, distribution: {dict(Counter(test_labels))}")
            
            results_log.write(f"\n{'='*80}\n")
            results_log.write(f"Fold {fold+1}/{args.num_folds} (seed={fold_seed})\n")
            results_log.write(f"Train: {len(train_data)} samples, distribution: {dict(Counter(train_labels))}\n")
            results_log.write(f"Val: {len(val_data)} samples, distribution: {dict(Counter(val_labels))}\n")
            results_log.write(f"Test: {len(test_data)} samples, distribution: {dict(Counter(test_labels))}\n")
            results_log.flush()
            
            # 计算类别权重
            class_weights = None
            pos_weight = None
            if args.loss_type in ['weighted_ce', 'focal']:
                counts = np.bincount(train_labels, minlength=nclasses).astype(np.float32)
                weights = counts.sum() / (counts + 1e-6)
                weights = weights / np.mean(weights)
                class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
                if nclasses == 2:
                    neg = counts[0]
                    pos = counts[1] if len(counts) > 1 else 0.0
                    pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32, device=device)
                print(f"Class weights: {class_weights.detach().cpu().numpy().tolist()}")
                results_log.write(f"Class weights: {class_weights.detach().cpu().numpy().tolist()}\n")
                results_log.flush()
            
            # 创建损失函数
            if args.loss_type == 'ce':
                if nclasses == 2:
                    criterion = torch.nn.BCEWithLogitsLoss()
                else:
                    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
            elif args.loss_type == 'weighted_ce':
                if nclasses == 2:
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
            
            # 创建数据加载器
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
            
            # 创建该fold的TensorBoard writer
            fold_log_dir = f"{log_dir}/fold_{fold+1}"
            fold_writer = SummaryWriter(log_dir=fold_log_dir)
            
            # 初始化模型
            view_encoders = ViewEncoder(
                in_dim=nfeats, hidden_dim=args.hidden_dim,
                emb_dim=args.emb_dim, dropout=args.dropout, sparse=args.sparse
            ).to(device)
            
            attention_fusion = AttentionFusion(input_dim=args.emb_dim, num_views=num_views).to(device)
            classifier = GraphClassifierHead(in_dim=args.emb_dim, nclasses=nclasses).to(device)
            
            # 优化器
            params = []
            params.append({'params': view_encoders.parameters()})
            params.append({'params': classifier.parameters()})
            params.append({'params': attention_fusion.parameters()})
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.w_decay)
            
            # 学习率调度器
            total_epochs = max(1, args.epochs)
            warmup_epochs = max(0, min(args.warmup_epochs, total_epochs))
            if warmup_epochs > 0:
                warmup = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / float(warmup_epochs))
            else:
                warmup = LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=args.min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
            
            best_val_f1 = -1.0
            best_state = None
            best_epoch = 0
            patience_counter = 0
            stop_training = False
            
            # 训练循环
            for epoch in range(1, args.epochs + 1):
                view_encoders.train()
                classifier.train()
                attention_fusion.train()
                
                total_loss = 0.0
                total_sup_loss = 0.0
                total_self_loss = 0.0
                total_lfd_loss = 0.0
                total_s_high_loss = 0.0
                train_correct = 0
                train_total = 0
                n_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    batch = batch.to(device)
                    adj_list, node2graph, N = build_adjs_from_batch(batch, device)
                    feat = batch.x
                    
                    z_specifics = []
                    for i in range(len(adj_list)):
                        z_view = view_encoders(feat, adj_list[i])
                        z_specifics.append(z_view)
                    
                    # 节点级多视图融合
                    fused_z, attention_weights = attention_fusion(z_specifics, batch=node2graph)
                    
                    # 图级pooling
                    graph_emb = global_mean_pool(fused_z, node2graph)
                    graph_emb = F.normalize(graph_emb, p=2, dim=1)
                    
                    # 分类
                    logits = classifier(graph_emb)
                    if torch.isnan(logits).any():
                        continue
                    
                    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                        loss_sup = criterion(logits.view(-1), batch.y.float())
                        prob_pos = torch.sigmoid(logits.view(-1))
                        preds = (prob_pos > 0.5).long()
                    else:
                        loss_sup = criterion(logits, batch.y.view(-1))
                        preds = logits.argmax(dim=1)
                    train_correct += (preds == batch.y.view(-1)).sum().item()
                    train_total += batch.y.view(-1).size(0)
                    
                    # 自监督损失
                    if args.loss_mode == 'ce_only':
                        loss = loss_sup
                        loss_self_val = 0.0
                        lfd_loss_val = 0.0
                        s_high_loss_val = 0.0
                    else:
                        loss_self, loss_details = compute_self_supervised_loss(
                            z_specifics, fused_z, adj_list, args.tau, args.h, args.alpha, args.beta
                        )
                        loss = loss_sup + args.lambda1 * loss_details['lfd_loss'] + args.lambda2 * loss_details['s_high_loss']
                        loss_self_val = loss_self.item() if torch.is_tensor(loss_self) else loss_self
                        lfd_loss_val = loss_details['lfd_loss'].item() if torch.is_tensor(loss_details['lfd_loss']) else loss_details['lfd_loss']
                        s_high_loss_val = loss_details['s_high_loss'].item() if torch.is_tensor(loss_details['s_high_loss']) else loss_details['s_high_loss']
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(view_encoders.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(attention_fusion.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_sup_loss += loss_sup.item()
                    total_self_loss += loss_self_val
                    total_lfd_loss += lfd_loss_val
                    total_s_high_loss += s_high_loss_val
                    n_batches += 1
                
                # 计算平均损失
                avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
                avg_sup_loss = total_sup_loss / n_batches if n_batches > 0 else 0.0
                avg_self_loss = total_self_loss / n_batches if n_batches > 0 else 0.0
                avg_lfd_loss = total_lfd_loss / n_batches if n_batches > 0 else 0.0
                avg_s_high_loss = total_s_high_loss / n_batches if n_batches > 0 else 0.0
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                
                # TensorBoard记录训练损失
                fold_writer.add_scalar('Training/Loss', avg_loss, epoch)
                fold_writer.add_scalar('Training/Loss_CE', avg_sup_loss, epoch)
                fold_writer.add_scalar('Training/Loss_Self', avg_self_loss, epoch)
                fold_writer.add_scalar('Training/Loss_LFD', avg_lfd_loss, epoch)
                fold_writer.add_scalar('Training/Loss_S_High', avg_s_high_loss, epoch)
                fold_writer.add_scalar('Training/Accuracy', train_acc, epoch)
                fold_writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], epoch)
                
                # 验证 - 使用F1作为选择标准
                if epoch % args.eval_freq == 0:
                    val_metrics, best_th, val_loss = self.search_best_threshold(
                        view_encoders, classifier, val_loader, 
                        attention_fusion, args, trial=fold, split='val',
                        threshold_min=args.threshold_min, threshold_max=args.threshold_max,
                        step=args.threshold_step, verbose=False
                    )
                    current_f1 = val_metrics['f1_macro']
                    
                    # TensorBoard记录验证指标
                    fold_writer.add_scalar('Validation/Loss', val_loss, epoch)
                    fold_writer.add_scalar('Validation/Accuracy', val_metrics['acc'], epoch)
                    fold_writer.add_scalar('Validation/F1_Macro', val_metrics['f1_macro'], epoch)
                    fold_writer.add_scalar('Validation/Precision_Macro', val_metrics['precision_macro'], epoch)
                    fold_writer.add_scalar('Validation/Recall_Macro', val_metrics['recall_macro'], epoch)
                    fold_writer.add_scalar('Validation/AUC_Macro', val_metrics['auc_macro'], epoch)
                    
                    print(f"Fold {fold+1} Epoch {epoch}: Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={current_f1:.4f}, Val Acc={val_metrics['acc']:.4f}, Best Th={best_th:.2f}")
                    
                    # 使用F1作为选择标准
                    if current_f1 > best_val_f1:
                        best_val_f1 = current_f1
                        best_epoch = epoch
                        best_state = {
                            'view_encoders': copy.deepcopy(view_encoders.state_dict()),
                            'classifier': copy.deepcopy(classifier.state_dict()),
                            'attention_fusion': copy.deepcopy(attention_fusion.state_dict()),
                            'best_threshold': best_th
                        }
                        print(f"  -> New best F1! Saved (F1={current_f1:.4f}).")
                        patience_counter = 0
                    else:
                        if best_state is None:
                            best_val_f1 = current_f1
                            best_epoch = epoch
                            best_state = {
                                'view_encoders': copy.deepcopy(view_encoders.state_dict()),
                                'classifier': copy.deepcopy(classifier.state_dict()),
                                'attention_fusion': copy.deepcopy(attention_fusion.state_dict()),
                                'best_threshold': best_th
                            }
                            print(f"  -> Initial model saved.")
                        patience_counter += 1
                        if epoch >= args.min_training_epochs and patience_counter >= args.patience:
                            print(f"  -> Early stopping at epoch {epoch} (no improvement for {patience_counter} evals).")
                            stop_training = True
                
                if stop_training:
                    scheduler.step()
                    break
                
                # 更新学习率
                scheduler.step()
            
            # 恢复最佳模型并在测试集上评估
            best_th_final = 0.5
            if best_state is not None:
                view_encoders.load_state_dict(best_state['view_encoders'])
                classifier.load_state_dict(best_state['classifier'])
                attention_fusion.load_state_dict(best_state['attention_fusion'])
                if 'best_threshold' in best_state:
                    best_th_final = best_state['best_threshold']
                print(f"\n✓ Loaded best model from epoch {best_epoch} (Val F1: {best_val_f1:.4f}, Th: {best_th_final:.2f})")
            else:
                print(f"\n⚠️ Warning: No improvement found during training. Using final model state.")
                best_val_f1 = 0.0
                best_epoch = args.epochs
            
            # 在测试集上评估
            test_loss, test_metrics = self.test_cls_graphlevel(
                view_encoders, classifier, test_loader,
                attention_fusion, args, fold, 'test', log_file=results_log, epoch=None, threshold=best_th_final
            )
            
            # TensorBoard记录测试结果
            fold_writer.add_scalar('Test/Accuracy', test_metrics['acc'], best_epoch)
            fold_writer.add_scalar('Test/F1_Macro', test_metrics['f1_macro'], best_epoch)
            fold_writer.add_scalar('Test/Precision_Macro', test_metrics['precision_macro'], best_epoch)
            fold_writer.add_scalar('Test/Recall_Macro', test_metrics['recall_macro'], best_epoch)
            fold_writer.add_scalar('Test/AUC_Macro', test_metrics['auc_macro'], best_epoch)
            
            # 记录超参数和最终指标
            fold_writer.add_hparams(
                {
                    'lr': args.lr,
                    'hidden_dim': args.hidden_dim,
                    'lambda1': args.lambda1,
                    'lambda2': args.lambda2,
                    'fold': fold + 1,
                    'best_epoch': best_epoch
                },
                {
                    'hparam/val_f1': best_val_f1,
                    'hparam/test_acc': test_metrics['acc'],
                    'hparam/test_f1': test_metrics['f1_macro'],
                    'hparam/test_auc': test_metrics['auc_macro']
                }
            )
            
            fold_writer.close()
            print(f"  TensorBoard log saved to: {fold_log_dir}")
            
            # 保存该fold的结果
            fold_results.append({
                'fold': fold + 1,
                'best_val_f1': best_val_f1,
                'best_epoch': best_epoch,
                'test_acc': test_metrics['acc'],
                'test_f1_macro': test_metrics['f1_macro'],
                'test_f1_micro': test_metrics['f1_micro'],
                'test_precision_macro': test_metrics['precision_macro'],
                'test_recall_macro': test_metrics['recall_macro'],
                'test_auc_macro': test_metrics['auc_macro'],
                'test_auc_micro': test_metrics['auc_micro'],
                'best_threshold': best_th_final
            })
            
            print(f"\nFold {fold+1} Results:")
            print(f"  Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
            print(f"  Test Acc: {test_metrics['acc']:.4f}")
            print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
            print(f"  Test F1 (micro): {test_metrics['f1_micro']:.4f}")
            print(f"  Test AUC (macro): {test_metrics['auc_macro']:.4f}")
        
        # 汇总所有fold的结果
        print(f"\n{'='*80}")
        print(f"K-FOLD RANDOM VALIDATION SUMMARY (k={args.num_folds})")
        print(f"{'='*80}")
        
        # 计算平均值和标准差
        metrics_to_report = ['test_acc', 'test_f1_macro', 'test_f1_micro', 
                            'test_precision_macro', 'test_recall_macro',
                            'test_auc_macro', 'test_auc_micro', 'best_val_f1']
        
        summary_stats = {}
        for metric in metrics_to_report:
            values = [r[metric] for r in fold_results]
            summary_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # 输出汇总结果
        summary_text = f"""
{'='*80}
K-FOLD RANDOM VALIDATION SUMMARY
{'='*80}
Number of folds: {args.num_folds}
Selection criterion: F1-macro on validation set
TensorBoard Log Directory: {log_dir}

Per-fold Results:
"""
        for r in fold_results:
            summary_text += f"""
  Fold {r['fold']}:
    Val F1:  {r['best_val_f1']:.4f} (Epoch {r['best_epoch']}, Threshold {r['best_threshold']:.3f})
    Test Acc: {r['test_acc']:.4f}
    Test F1 (macro): {r['test_f1_macro']:.4f}
    Test F1 (micro): {r['test_f1_micro']:.4f}
    Test Precision (macro): {r['test_precision_macro']:.4f}
    Test Recall (macro): {r['test_recall_macro']:.4f}
    Test AUC (macro): {r['test_auc_macro']:.4f}
"""
        
        summary_text += f"""
{'='*80}
AVERAGE RESULTS ACROSS {args.num_folds} FOLDS:
{'='*80}
Validation F1 (macro):
  Mean ± Std: {summary_stats['best_val_f1']['mean']:.4f} ± {summary_stats['best_val_f1']['std']:.4f}
  Range: [{summary_stats['best_val_f1']['min']:.4f}, {summary_stats['best_val_f1']['max']:.4f}]

Test Accuracy:
  Mean ± Std: {summary_stats['test_acc']['mean']:.4f} ± {summary_stats['test_acc']['std']:.4f}
  Range: [{summary_stats['test_acc']['min']:.4f}, {summary_stats['test_acc']['max']:.4f}]

Test F1 (macro):
  Mean ± Std: {summary_stats['test_f1_macro']['mean']:.4f} ± {summary_stats['test_f1_macro']['std']:.4f}
  Range: [{summary_stats['test_f1_macro']['min']:.4f}, {summary_stats['test_f1_macro']['max']:.4f}]

Test F1 (micro):
  Mean ± Std: {summary_stats['test_f1_micro']['mean']:.4f} ± {summary_stats['test_f1_micro']['std']:.4f}
  Range: [{summary_stats['test_f1_micro']['min']:.4f}, {summary_stats['test_f1_micro']['max']:.4f}]

Test Precision (macro):
  Mean ± Std: {summary_stats['test_precision_macro']['mean']:.4f} ± {summary_stats['test_precision_macro']['std']:.4f}
  Range: [{summary_stats['test_precision_macro']['min']:.4f}, {summary_stats['test_precision_macro']['max']:.4f}]

Test Recall (macro):
  Mean ± Std: {summary_stats['test_recall_macro']['mean']:.4f} ± {summary_stats['test_recall_macro']['std']:.4f}
  Range: [{summary_stats['test_recall_macro']['min']:.4f}, {summary_stats['test_recall_macro']['max']:.4f}]

Test AUC (macro):
  Mean ± Std: {summary_stats['test_auc_macro']['mean']:.4f} ± {summary_stats['test_auc_macro']['std']:.4f}
  Range: [{summary_stats['test_auc_macro']['min']:.4f}, {summary_stats['test_auc_macro']['max']:.4f}]

Test AUC (micro):
  Mean ± Std: {summary_stats['test_auc_micro']['mean']:.4f} ± {summary_stats['test_auc_micro']['std']:.4f}
  Range: [{summary_stats['test_auc_micro']['min']:.4f}, {summary_stats['test_auc_micro']['max']:.4f}]
{'='*80}
"""
        
        print(summary_text)
        results_log.write(summary_text)
        results_log.close()
        
        print(f"\nResults saved to: {results_log_path}")
        print(f"To view TensorBoard, run: tensorboard --logdir={log_dir}")


if __name__ == '__main__':
    experiment = Experiment()
    if args.use_kfold_random:
        experiment.train_kfold_random(args)
    else:
        experiment.train_single(args)
