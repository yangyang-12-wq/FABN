"""
K-Fold 数据处理器 (多GPU并行版本)
负责从 train_val_raw.pkl 生成 K 折训练/验证数据
支持: Fold级并行 + Train/Val并行 + 图缓存
"""
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utils_graph_build import *
from utils import safe_makedirs
from concurrent.futures import ProcessPoolExecutor, as_completed

def _ensure_args_object(args):
    """确保 args 是一个可访问属性的对象"""
    if isinstance(args, dict):
        from types import SimpleNamespace
        return SimpleNamespace(**args)
    return args


def build_graph_for_sample(sample_data, sample_id, label, region_map, args):
    """
    为单个样本构建图结构和编码 (可并行调用)
    
    Args:
        sample_data: 时序数据 (T, C)
        sample_id: 样本ID
        label: 标签
        region_map: 脑区映射
        args: 参数 (dict或SimpleNamespace)
    
    Returns:
        处理后的样本字典
    """
    args = _ensure_args_object(args)
    ts_np = sample_data
    
    # 构建图
    G_intra, _, _ = build_intra_region_view_mi(
        ts_np, region_map,
        n_bins=16, strategy="uniform",
        window_size=400, stride=200
    )
    
    S_global, _, _ = build_global_view(ts_np, w1=args.w1, w2=args.w2, fs=args.fs)
    
    A_intra_sp = topk_sparsify_sym_row_normalize(G_intra, args.k_intra)
    A_global_sp = topk_sparsify_sym_row_normalize(S_global, args.k_global)
    
    # 结构编码
    A_global_sp_t = torch.from_numpy(A_global_sp).float()
    from torch_geometric.utils import dense_to_sparse
    from torch_geometric.data import Data
    edge_index, edge_weight = dense_to_sparse(A_global_sp_t)
    g = Data(edge_index=edge_index, num_nodes=A_global_sp.shape[0], edge_attr=edge_weight)
    node_feats_np = compute_structure_encodings(g)
    
    return {
        'id': sample_id,
        'label': label,
        'node_feats': node_feats_np.astype(np.float32),
        'A_intra': A_intra_sp.astype(np.float32),
        'A_global': A_global_sp.astype(np.float32),
    }


def process_samples_parallel(samples, region_map, args, desc="Processing", use_cache=None):
    """
    并行处理多个样本的图构建
    
    Args:
        samples: 样本列表 [{'id', 'data', 'labels'}, ...]
        region_map: 脑区映射
        args: 参数
        desc: 进度条描述
        use_cache: 可选的缓存字典 {sample_id: processed_result}
    
    Returns:
        处理后的字典 {sample_id: processed_data}
    """
    processed = {}
    
    # 识别需要处理的样本
    if use_cache is not None:
        samples_to_process = [s for s in samples if s['id'] not in use_cache]
        cached_samples = [s for s in samples if s['id'] in use_cache]
        print(f"  Using cache: {len(cached_samples)} samples, Processing: {len(samples_to_process)} samples")
        
        # 加载缓存
        for s in cached_samples:
            processed[s['id']] = use_cache[s['id']]
    else:
        samples_to_process = samples
    
    # 并行处理新样本
    if len(samples_to_process) > 0:
        # 使用进程池并行
        max_workers = min(8, len(samples_to_process))  # 最多8个worker
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for entry in samples_to_process:
                future = executor.submit(
                    build_graph_for_sample,
                    entry['data'], entry['id'], entry['labels'],
                    region_map, args
                )
                futures[future] = entry['id']
            
            # 收集结果
            with tqdm(total=len(samples_to_process), desc=desc) as pbar:
                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        result = future.result()
                        processed[result['id']] = result
                    except Exception as e:
                        print(f"Error processing {sid}: {e}")
                    pbar.update(1)
    
    return processed


def process_fold_data_parallel(fold_idx, train_indices, val_indices, train_val_data, 
                                region_map, args, graph_cache=None):
    """
    并行处理单个 fold 的训练和验证数据
    Train和Val的图构建同时进行,原始样本使用缓存
    
    Args:
        fold_idx: Fold 编号
        train_indices: 训练集索引
        val_indices: 验证集索引
        train_val_data: 原始 train+val 数据
        region_map: 脑区映射
        args: 参数
        graph_cache: 图缓存字典 {sample_id: processed_result}
    
    Returns:
        (train_processed, val_processed): 处理后的字典
    """
    print(f"\n{'='*60}")
    print(f"Processing Fold {fold_idx + 1}")
    print(f"{'='*60}")
    
    # 分割数据
    train_rows = [train_val_data[i] for i in train_indices]
    val_rows = [train_val_data[i] for i in val_indices]
    
    print(f"Train: {len(train_rows)} samples")
    print(f"  Distribution: {dict(Counter([r['labels'] for r in train_rows]))}")
    print(f"Val: {len(val_rows)} samples")
    print(f"  Distribution: {dict(Counter([r['labels'] for r in val_rows]))}")
    
    # 处理 Train 和 Val (使用样本级并行,已经在 process_samples_parallel 中实现)
    print(f"\nProcessing training set...")
    train_processed = process_samples_parallel(
        train_rows, region_map, args,
        desc=f"Fold{fold_idx+1} Train",
        use_cache=graph_cache
    )
    
    print(f"\nProcessing validation set...")
    val_processed = process_samples_parallel(
        val_rows, region_map, args,
        desc=f"Fold{fold_idx+1} Val",
        use_cache=graph_cache
    )
    
    return train_processed, val_processed


def build_graph_cache(train_val_data, region_map, args, cache_file=None):
    """
    预先构建所有原始样本的图缓存 (支持磁盘持久化)
    如果缓存文件存在,直接加载;否则构建并保存
    
    Args:
        train_val_data: 原始样本列表
        region_map: 脑区映射
        args: 参数
        cache_file: 缓存文件路径 (可选)
    
    Returns:
        缓存字典 {sample_id: processed_result}
    """
    # 检查是否有缓存文件
    if cache_file and os.path.exists(cache_file):
        print("\n" + "="*60)
        print("Loading existing graph cache from disk...")
        print("="*60)
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"✓ Cache loaded: {len(cache)} samples from {cache_file}")
            
            # 验证缓存是否包含所有需要的样本
            cached_ids = set(cache.keys())
            required_ids = set([r['id'] for r in train_val_data])
            missing_ids = required_ids - cached_ids
            
            if missing_ids:
                print(f"⚠️  Cache incomplete: {len(missing_ids)} samples missing")
                print(f"  Missing samples: {list(missing_ids)[:10]}...")
                print("  Rebuilding missing samples...")
                
                # 只构建缺失的样本
                missing_samples = [r for r in train_val_data if r['id'] in missing_ids]
                missing_cache = process_samples_parallel(
                    missing_samples, region_map, args,
                    desc="Building missing cache"
                )
                cache.update(missing_cache)
                
                # 保存更新后的缓存
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache, f)
                print(f"✓ Cache updated and saved: {len(cache)} samples")
            
            return cache
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            print("  Rebuilding cache from scratch...")
    
    # 构建新缓存
    print("\n" + "="*60)
    print("Building graph cache for all original samples...")
    print("="*60)
    
    cache = process_samples_parallel(
        train_val_data, region_map, args,
        desc="Building cache"
    )
    
    print(f"✓ Cache built: {len(cache)} samples")
    
    # 保存到磁盘
    if cache_file:
        try:
            safe_makedirs(os.path.dirname(cache_file))
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            print(f"✓ Cache saved to: {cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
    
    return cache


def process_single_fold(fold_data):
    """
    处理单个fold的包装函数 (用于多进程)
    
    Args:
        fold_data: 包含所有必要参数的字典
    
    Returns:
        (fold_idx, train_processed, val_processed, train_file, val_file)
    """
    fold_idx = fold_data['fold_idx']
    train_idx = fold_data['train_idx']
    val_idx = fold_data['val_idx']
    train_val_data = fold_data['train_val_data']
    region_map = fold_data['region_map']
    args_dict = fold_data['args']
    graph_cache = fold_data['graph_cache']
    processed_dir = fold_data['processed_dir']
    
    # 将字典转换为SimpleNamespace对象(具有属性访问)
    from types import SimpleNamespace
    args = SimpleNamespace(**args_dict)
    
    # 处理数据
    train_processed, val_processed = process_fold_data_parallel(
        fold_idx, train_idx, val_idx, train_val_data,
        region_map, args, graph_cache
    )
    
    # 保存文件
    train_file = os.path.join(processed_dir, f'processed_train_fold{fold_idx}.pkl')
    val_file = os.path.join(processed_dir, f'processed_val_fold{fold_idx}.pkl')
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_processed, f)
    with open(val_file, 'wb') as f:
        pickle.dump(val_processed, f)
    
    return fold_idx, len(train_processed), len(val_processed), train_file, val_file


def generate_kfold_data(processed_dir, n_splits=8, random_seed=42, n_jobs=None):
    """
    从 train_val_raw.pkl 生成 K-Fold 数据 (多进程并行版本)
    
    Args:
        processed_dir: 预处理数据目录
        n_splits: K 值(默认 8)
        random_seed: 随机种子
        n_jobs: 并行job数 (默认None自动检测GPU数,或使用CPU核心数)
    """
    print("\n" + "="*80)
    print(f"K-Fold Data Generation (K={n_splits}) - Parallel Mode")
    print("="*80)
    
    # 自动检测并行数
    if n_jobs is None:
        if torch.cuda.is_available():
            n_jobs = torch.cuda.device_count()
            print(f"🚀 Detected {n_jobs} GPUs, will process {n_jobs} folds in parallel")
        else:
            n_jobs = min(4, os.cpu_count() or 1)  # CPU模式最多4个进程
            print(f"💻 Using CPU mode with {n_jobs} parallel jobs")
    else:
        print(f"⚙️  Using {n_jobs} parallel jobs (user specified)")
    
    # 加载 train_val_raw.pkl
    train_val_file = os.path.join(processed_dir, 'train_val_raw.pkl')
    if not os.path.exists(train_val_file):
        raise FileNotFoundError(f"train_val_raw.pkl not found at {train_val_file}")
    
    with open(train_val_file, 'rb') as f:
        train_val_data = pickle.load(f)
    
    print(f"Loaded {len(train_val_data)} samples from train_val_raw.pkl")
    
    # 提取标签用于分层
    labels = np.array([r['labels'] for r in train_val_data])
    print(f"Label distribution: {dict(Counter(labels))}")
    
    # 创建 K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # 准备参数 (使用简单字典代替argparse对象,便于序列化)
    args_dict = {
        'k_intra': 8,
        'k_global': 8,
        'w1': 0.5,
        'w2': 0.5,
        'fs': 1.0
    }
    
    # 加载 region_map
    region_map_path = os.path.join(os.path.dirname(processed_dir), 'region_map.npy')
    if os.path.exists(region_map_path):
        region_map = np.load(region_map_path)
    else:
        C = train_val_data[0]['data'].shape[1]
        from utils_graph_build import brain_regions, make_region_map
        region_map = make_region_map(brain_regions, C)
        np.save(region_map_path, region_map)
    
    # 构建图缓存 (只对原始样本构建一次,支持磁盘持久化)
    # 创建临时的SimpleNamespace对象供build_graph_cache使用
    from types import SimpleNamespace
    args_obj = SimpleNamespace(**args_dict)
    
    # 缓存文件路径
    cache_file = os.path.join(processed_dir, 'graph_cache.pkl')
    graph_cache = build_graph_cache(train_val_data, region_map, args_obj, cache_file=cache_file)
    
    # 准备所有fold的参数
    fold_tasks = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_tasks.append({
            'fold_idx': fold_idx,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'train_val_data': train_val_data,
            'region_map': region_map,
            'args': args_dict,  # 传递字典而不是argparse对象
            'graph_cache': graph_cache,
            'processed_dir': processed_dir
        })
    
    # 并行处理所有fold
    print("\n" + "="*60)
    print(f"Processing {n_splits} folds in parallel...")
    print("="*60)
    
    if n_jobs == 1:
        # 串行模式
        results = []
        for task in fold_tasks:
            result = process_single_fold(task)
            results.append(result)
    else:
        # 并行模式
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_single_fold, fold_tasks))
    
    # 打印结果
    print("\n" + "="*60)
    print("All folds completed!")
    print("="*60)
    for fold_idx, n_train, n_val, train_file, val_file in sorted(results):
        print(f"✓ Fold {fold_idx + 1}:")
        print(f"    Train: {os.path.basename(train_file)} ({n_train} samples)")
        print(f"    Val: {os.path.basename(val_file)} ({n_val} samples)")
    
    print("\n" + "="*80)
    print("K-Fold data generation complete!")
    print("="*80)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python kfold_processor.py <processed_dir> [n_splits] [random_seed] [n_jobs] [--clear-cache]")
        print("Example: python kfold_processor.py ./processed_fnirs 8 42 4")
        print("  n_jobs: Number of parallel jobs (default: auto-detect GPUs or use 4 CPUs)")
        print("  --clear-cache: Delete existing cache and rebuild from scratch")
        sys.exit(1)
    
    processed_dir = sys.argv[1]
    n_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    random_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '--clear-cache' else None
    
    # 检查是否需要清除缓存
    clear_cache = '--clear-cache' in sys.argv
    if clear_cache:
        cache_file = os.path.join(processed_dir, 'graph_cache.pkl')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"✓ Cache cleared: {cache_file}")
    
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    generate_kfold_data(processed_dir, n_splits, random_seed, n_jobs)
