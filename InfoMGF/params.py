import argparse
import torch
import sys

def set_params():
    """设置实验参数"""
    parser = argparse.ArgumentParser(description='InfoMGF: Single-Stage End-to-End fNIRS Classification Framework')
    
    # 基础实验设置
    parser.add_argument('--dataset', type=str, default='fnirs', 
                        help='Dataset name')
    parser.add_argument('--dataset_name', type=str, default='fnirs_exp',
                        help='Dataset name for experiment tracking')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number (used when multi_gpu=False)')
    parser.add_argument('--multi_gpu', type=bool, default=True,
                        help='Whether to use multiple GPUs with DataParallel')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7',
                        help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # K-Fold Random 模式参数
    parser.add_argument('--use_kfold_random', action='store_true',
                        help='Use k-fold random split mode (each fold: 7:1:2 train/val/test split)')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for k-fold random validation')
    
    # 数据相关参数
    parser.add_argument('--out_dir', type=str, default='/data1/cuichenyang/processed_data',
                        help='Directory containing processed data files')
    parser.add_argument('--feature_path', type=str, default='features',
                        help='Path to feature files')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (对齐 laoda 默认=4)')
    parser.add_argument('--preload', type=bool, default=True,
                        help='Whether to preload data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=350,
                        help='Maximum training epochs (对齐 laoda 默认=350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (对齐 laoda 默认=1e-3)')
    parser.add_argument('--w_decay', type=float, default=1e-3,
                        help='Weight decay (对齐 laoda 默认=1e-3)')
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'weighted_ce', 'focal'],
                        help='Classification loss: ce, weighted_ce, or focal')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma for focal loss when loss_type=focal')

    parser.add_argument('--patience', type=int, default=80,
                        help='Early stopping patience (increased to allow more exploration)')
    parser.add_argument('--classification_weight', type=float, default=5.0,
                        help='Weight multiplier for classification loss')
    parser.add_argument('--min_ce_loss_threshold', type=float, default=0.3,
                        help='Minimum CE loss threshold before allowing early stopping (relaxed from 0.05 to be more achievable)')
    parser.add_argument('--min_training_epochs', type=int, default=30,
                        help='Minimum number of training epochs before considering early stopping')
 
    
    # 模型架构参数
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension (reduced to 32 to combat overfitting)')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (balanced regularization)')
    

  
    
    # 在 params.py 中添加缺失的参数
    parser.add_argument('--sparse', type=bool, default=False, 
                        help='Whether to use sparse graph representation')
    parser.add_argument('--nlayer_gnn', type=int, default=2,
                        help='Number of GNN encoder layers')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Embedding dimension (reduced to 32 to combat overfitting)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter for custom loss')
    parser.add_argument('--h', type=float, default=1.0,
                        help='H parameter (reserved for future use)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for LFD loss (MSE-based, scale ~1.0)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Weight for S_high loss (already ~1.0, reduce weight)')
    # gamma参数已移除 - SC loss不再使用
    parser.add_argument('--lambda1', type=float, default=0.0,
                        help='Overall weight for self-supervised loss (balanced with CE loss)')
    parser.add_argument('--lambda2', type=float, default=0.0,
                        help='Overall weight for self-supervised loss (balanced with CE loss)')
    

    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Validation evaluation frequency (every N epochs)')
    
    # K-Fold 交叉验证参数
    parser.add_argument('--n_folds', type=int, default=8,
                        help='Number of folds for K-Fold cross validation')


    parser.add_argument('--scheduler_patience', type=int, default=30,
                        help='ReduceLROnPlateau patience (eval steps) - reduced for faster adaptation')
    parser.add_argument('--lr_schedule', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Learning rate schedule: plateau (ReduceLROnPlateau) or cosine (warmup + CosineAnnealingLR, no restarts)')
    parser.add_argument('--warmup_epochs', type=int, default=70,
                        help='Warmup epochs before cosine annealing (only when lr_schedule=cosine)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--scheduler_T0', type=int, default=15,
                        help='(Unused now) kept for backward compatibility')
    parser.add_argument('--scheduler_Tmult', type=int, default=2,
                        help='(Unused now) kept for backward compatibility')
   
  
    parser.add_argument('--label_mode', type=str, default='binary',
                        choices=['multi', 'binary'],
                        help='How to load labels: multi-class or binary (0 vs others)')
    
    # 损失函数模式
    parser.add_argument('--loss_mode', type=str, default='full',
                        choices=['full', 'ce_only'],
                        help='Loss mode: full (all losses) or ce_only (classification loss only) - DEFAULT ce_only for stability')
    parser.add_argument('--smart_resample', type=bool, default=True,
                        help='Whether to apply smart resampling for binary classification (train set only) - DEFAULT False to check raw data')
    parser.add_argument('--use_original_only', action='store_true',
                        help='Use only original samples, filtering out all _aug augmented samples (train only). Default: False (includes augmented samples).')
    parser.add_argument('--balance_strategy', type=str, default='downsample_class1',
                        choices=['upsample_class0', 'downsample_class1'],
                        help='Strategy for balancing classes: upsample_class0 (increase Class 0) or downsample_class1 (proportionally sample Class 1)')
    parser.add_argument('--attention_use_layer_norm', type=bool, default=True,
                    help='Whether to apply LayerNorm inside the attention fusion module')
    
    # 注意力机制参数
    parser.add_argument('--use_attention_clipping', action='store_true',
                        help='Enable attention weight clipping to prevent one view from dominating')
    parser.add_argument('--min_attention', type=float, default=0.1,
                        help='Minimum attention weight for each view (default: 0.1, range: 0.0-0.5)')
    parser.add_argument('--max_attention', type=float, default=0.9,
                        help='Maximum attention weight for each view (default: 0.9, range: 0.5-1.0)')
    
    # 动态增强参数
    parser.add_argument('--augment', action='store_true',
                        help='Enable dynamic data augmentation (DropEdge + FeatureNoise) during training')
    parser.add_argument('--augment_strength', type=str, default='light',
                        choices=['light', 'medium', 'heavy'],
                        help='Augmentation strength: light (0.1/0.05), medium (0.2/0.1), heavy (0.3/0.2)')
    parser.add_argument('--drop_edge_p', type=float, default=None,
                        help='Manual override for drop edge probability (default: None, use augment_strength)')
    parser.add_argument('--noise_std', type=float, default=None,
                        help='Manual override for feature noise std (default: None, use augment_strength)')
   
  
    
    # ????????????? logit?
    parser.add_argument('--threshold_min', type=float, default=0.1,
                        help='Min threshold when searching best decision boundary (binary)')
    parser.add_argument('--threshold_max', type=float, default=0.9,
                        help='Max threshold when searching best decision boundary (binary)')
    parser.add_argument('--threshold_step', type=float, default=0.02,
                        help='Step size for threshold search (binary)')

    args = parser.parse_args()
    
    # 根据 augment_strength 自动设置增强参数（如果没有手动指定）
    if args.augment and args.drop_edge_p is None:
        strength_map = {
            'light': (0.05, 0.05),
            'medium': (0.2, 0.1),
            'heavy': (0.3, 0.2)
        }
        args.drop_edge_p, args.noise_std = strength_map[args.augment_strength]
    
    return args


