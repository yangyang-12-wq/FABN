import os
import argparse
import pickle
from tqdm import tqdm
from collections import Counter
import shutil
from utils import *
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from utils_graph_build import *
import random
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, degree
class DataAugmentor:
    @staticmethod
    def time_shifting(data):
        shift = random.randint(-200, 200)
        if isinstance(data, torch.Tensor):
            return torch.roll(data, shifts=shift, dims=0)
        else:
            return np.roll(data, shift=shift, axis=0)
    
    @staticmethod
    def time_reversal(data):
        if isinstance(data, torch.Tensor):
            return torch.flip(data, dims=[0])
        else:
            return np.flip(data, axis=0)

    @staticmethod
    def noise_injection(data, rel_std=0.05, min_abs_noise=1e-6):
        if isinstance(data, torch.Tensor):
            arr = data.cpu().numpy()
            is_torch = True
        else:
            arr = data
            is_torch = False

        T, C = arr.shape
        noise = np.zeros_like(arr)
        channel_stds = np.std(arr, axis=0)
        for c in range(C):
            sigma = channel_stds[c] * rel_std
            if sigma < min_abs_noise:
                sigma = min_abs_noise
            noise[:, c] = np.random.normal(0.0, sigma, size=T)
        out = arr + noise
        if is_torch:
            return torch.from_numpy(out).to(data.dtype)
        return out
        
    @staticmethod
    def time_masking(data, min_nonzero_ratio=0.8, max_mask_span=100):
        if isinstance(data, torch.Tensor):
            arr = data.cpu().numpy()
            is_torch = True
        else:
            arr = data
            is_torch = False

        T = arr.shape[0]
        max_mask_len = int(T * (1 - min_nonzero_ratio))
        max_mask_len = max(5, min(max_mask_len, max_mask_span))  # 更灵活的限制
        if max_mask_len < 1:
            # 无需掩码
            return data

        mask_len = random.randint(1, max_mask_len)
        start = random.randint(0, T - mask_len)
        end = start + mask_len

        out = arr.copy()
        for c in range(arr.shape[1]):
            # 插值前后端点
            left_idx = start - 1
            right_idx = end
            if left_idx < 0 and right_idx >= T:
                # 整列都被mask，跳过（返回原始）
                continue
            elif left_idx < 0:
                out[start:end, c] = out[right_idx, c]
            elif right_idx >= T:
                out[start:end, c] = out[left_idx, c]
            else:
                # 线性插值
                left_val = out[left_idx, c]
                right_val = out[right_idx, c]
                span = np.linspace(0, 1, mask_len, endpoint=False)
                out[start:end, c] = left_val * (1 - span) + right_val * span

        if is_torch:
            return torch.from_numpy(out).to(data.dtype)
        return out
    @staticmethod
    def scaling(data, min_scale=0.9, max_scale=1.1):
        """按通道乘以小幅度因子，避免把数据变平"""
        if isinstance(data, torch.Tensor):
            arr = data.cpu().numpy()
            is_torch = True
        else:
            arr = data
            is_torch = False
        scales = np.random.uniform(min_scale, max_scale, size=(1, arr.shape[1]))
        out = arr * scales
        if is_torch:
            return torch.from_numpy(out).to(data.dtype)
        return out
    @staticmethod
    def augment_data(data, allow_identity=False):
        total_std = np.std(data) if not isinstance(data, torch.Tensor) else float(torch.std(data).item())
        # 可用的方法（不包含恒等）
        methods = [
            DataAugmentor.time_shifting,
            DataAugmentor.noise_injection,
            DataAugmentor.time_masking,
            DataAugmentor.time_reversal,
            DataAugmentor.scaling
        ]
        if allow_identity:
            methods.append(lambda x: x)

        # 若原始方差极低，优先选择 noise 或 scaling，避免掩码导致更低方差
        if total_std < 1e-5:
            choice = random.choice([DataAugmentor.noise_injection, DataAugmentor.scaling, DataAugmentor.time_shifting])
            return choice(data)
        else:
            choice = random.choice(methods)
            return choice(data)

def oversample_training_data(rows, augment_minority=True):
    label_counts = Counter([r['labels'] for r in rows])
    max_count = max(label_counts.values())
    
    print(f"原始类别分布: {dict(label_counts)}")
    
    augmented_data = rows.copy()

    for label, count in label_counts.items():
        if count < max_count:
            minority_samples = [r for r in rows if r['labels'] == label]
            needed = max_count - count
            
            for i in range(needed):
                sample = random.choice(minority_samples).copy()
                if augment_minority:
                    sample['data'] = DataAugmentor.augment_data(sample['data'])
                    sample['id'] = f"{sample['id']}_aug_{i}"
                augmented_data.append(sample)
    
    new_counts = Counter([r['labels'] for r in augmented_data])
    print(f"过采样后类别分布: {dict(new_counts)}")
    print(f"总样本数: {len(augmented_data)} (原始: {len(rows)})")
    return augmented_data

def prepare_and_balance_data(feature_path, split, feature_type, augment_train=True):
    feature_data = load_feature_pickle(feature_path, split)
    rows = prepare_rows_from_feature_data(feature_data, feature_type)
    
    print(f"\n=== {split}集数据信息 ===")
    print(f"样本数量: {len(rows)}")
    
    if rows:
        original_labels = [r['labels'] for r in rows]
        label_counter = Counter(original_labels)
        print(f"多分类标签分布: {dict(label_counter)}")
        
        if split == 'train' and augment_train:
            rows = oversample_training_data(rows, augment_minority=True)
            new_label_counter = Counter([r['labels'] for r in rows])
            print(f"过采样后多分类分布: {dict(new_label_counter)}")
        
        sample_shape = rows[0]['data'].shape
        print(f"数据形状: (时间点, 通道数) = {sample_shape}")
    
    return rows

def process_split(split_name, rows, out_dir, feature_path, region_map, 
                  k_global, w1, w2, fs, device, batch_size, save_npy):
    out = {}
    adjs_dir = os.path.join(feature_path, 'adjs_precomputed')
    safe_makedirs(adjs_dir)
    safe_makedirs(out_dir)

    print(f"Processing {split_name} data on device {device}...")

    progress_bar = tqdm(total=len(rows), desc=f"{split_name} split on {device}")

    for j, entry in enumerate(rows):
        sid = entry['id']
        ts_np = entry['data']
        label = entry['labels']

        G_intra, _, n_windows = build_intra_region_view_mi(
            ts_np, region_map,
            n_bins=16, strategy="uniform",
            window_size=400, stride=200
        )
        print(f"G_intra: shape={G_intra.shape}, non-zero={np.count_nonzero(G_intra)}, mean={np.mean(G_intra):.6f}, valid_windows={n_windows}")                    
        
        S_global, _, _ = build_global_view(ts_np, w1=w1, w2=w2, fs=fs)
        print(f"S_global: shape={S_global.shape}, non-zero={np.count_nonzero(S_global)}, mean={np.mean(S_global):.6f}")
        
        # 确保对角线为0（防止自环消耗边预算）
        np.fill_diagonal(G_intra, 0.0)
        np.fill_diagonal(S_global, 0.0)
        
        # 使用自适应 k_intra 稀疏化（按脑区大小自动调整）
        A_intra_sp = topk_sparsify_adaptive_intra(G_intra, region_map)
        print(f"A_intra_sp (adaptive): non-zero={np.count_nonzero(A_intra_sp)}, mean={np.mean(A_intra_sp):.6f}")
        
        A_global_sp = topk_sparsify_sym_row_normalize(S_global, k_global)
        print(f"A_global_sp: non-zero={np.count_nonzero(A_global_sp)}, mean={np.mean(A_global_sp):.6f}")
        
        A_global_sp_t = torch.from_numpy(A_global_sp).float()
        edge_index, edge_weight = dense_to_sparse(A_global_sp_t)
        g = Data(edge_index=edge_index, num_nodes=A_global_sp.shape[0], edge_attr=edge_weight)
        node_feats_np = compute_structure_encodings(g)
        
        feature_mean = np.mean(node_feats_np)
        feature_std = np.std(node_feats_np)
        print(f"Sample {sid}: feature_mean={feature_mean:.6f}, feature_std={feature_std:.6f}")
        
        if save_npy:
            np.save(os.path.join(adjs_dir, f"{sid}_intra.npy"), A_intra_sp.astype(np.float32))
            np.save(os.path.join(adjs_dir, f"{sid}_global.npy"), A_global_sp.astype(np.float32))
            np.save(os.path.join(adjs_dir, f"{sid}_nodefeat.npy"), node_feats_np.astype(np.float32))

        out[sid] = {
            'id': sid,
            'label': label,
            'node_feats': node_feats_np.astype(np.float32),
            'A_intra': A_intra_sp.astype(np.float32),
            'A_global': A_global_sp.astype(np.float32),
        }
        progress_bar.update(1)

    progress_bar.close()
    return out


def run_on_gpu(rank, args, all_rows, splits, region_map):
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        torch.cuda.set_device(rank)
    print(f"Process {rank} is using {device}")

    results = {}
    for split in splits:
        rows = all_rows[split]
        start = len(rows) // args.nprocs * rank
        end = len(rows) // args.nprocs * (rank + 1)
        if rank == args.nprocs - 1:
            end = len(rows)
        part = rows[start:end]
        if not part:
            continue

        out_dict = process_split(split, part, args.out_dir, args.feature_path, region_map,
                                 k_global=args.k_global,
                                 w1=args.w1, w2=args.w2, fs=args.fs, device=device,
                                 batch_size=args.batch_size, save_npy=args.save_npy)
        results[split] = out_dict

    out_file = os.path.join(args.out_dir, f"processed_rank{rank}.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Rank {rank} saved {out_file}")


def load_and_split_all_data(feature_path, feature_type, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42):
    """
    加载所有数据并按照指定比例重新分割
    train_ratio : val_ratio : test_ratio = 7:1:2
    """
    print("\n" + "="*80)
    print("Step 1: Loading ALL original data and reshuffling...")
    print("="*80)
    
    # 加载所有分割的原始数据
    all_rows = []
    for split in ['train', 'val', 'test']:
        try:
            feature_data = load_feature_pickle(feature_path, split)
            rows = prepare_rows_from_feature_data(feature_data, feature_type)
            all_rows.extend(rows)
            print(f"Loaded {len(rows)} samples from {split}")
        except FileNotFoundError:
            print(f"Warning: {split}_data.pkl not found, skipping...")
    
    total_samples = len(all_rows)
    print(f"\nTotal samples loaded: {total_samples}")
    
    # 统计原始类别分布
    original_label_dist = Counter([r['labels'] for r in all_rows])
    print(f"Original label distribution: {dict(original_label_dist)}")
    
    # 打乱数据
    np.random.seed(random_seed)
    random.seed(random_seed)
    random.shuffle(all_rows)
    print(f"Data shuffled with seed={random_seed}")
    
    # 按比例分割
    n_train_val = int(total_samples * (train_ratio + val_ratio))
    n_test = total_samples - n_train_val
    
    train_val_data = all_rows[:n_train_val]
    test_data = all_rows[n_train_val:]
    
    print(f"\n" + "="*80)
    print(f"Step 2: Initial Split (7:1:2)")
    print(f"="*80)
    print(f"Train+Val: {len(train_val_data)} samples ({len(train_val_data)/total_samples*100:.1f}%)")
    print(f"Test (locked): {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)")
    print(f"  Test label distribution: {dict(Counter([r['labels'] for r in test_data]))}")
    
    return train_val_data, test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='processed_fnirs')
    # k_intra is now adaptive based on region size, no longer a command line argument
    parser.add_argument('--k_global', type=int, default=8)
    parser.add_argument('--w1', type=float, default=0.5)
    parser.add_argument('--w2', type=float, default=0.5)
    parser.add_argument('--fs', type=float, default=20.0)
    parser.add_argument('--feature_type', type=str, default='oxy')
    parser.add_argument('--node_out_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nprocs', type=int, default=torch.cuda.device_count())
    parser.add_argument('--save_npy', type=bool, default=True, help='Save individual .npy files for each sample')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for data splitting')
    args = parser.parse_args()

    # Step 1 & 2: 加载并重新分割数据 (7:1:2)
    train_val_data, test_data = load_and_split_all_data(
        args.feature_path, 
        args.feature_type,
        random_seed=args.random_seed
    )
    
    # Step 3: 处理 Test 集 (独立保存,不做任何增强)
    print("\n" + "="*80)
    print("Step 3: Processing Test set (independent, no augmentation)")
    print("="*80)
    
    # 获取 region_map
    region_map_path = os.path.join(args.feature_path, 'region_map.npy')
    if os.path.exists(region_map_path):
        region_map = np.load(region_map_path)
        print("Loaded region_map from", region_map_path)
    else:
        if len(train_val_data) > 0:
            C = train_val_data[0]['data'].shape[1]
        elif len(test_data) > 0:
            C = test_data[0]['data'].shape[1]
        else:
            raise ValueError("No data available to determine number of channels")
        region_map = make_region_map(brain_regions, C)
        np.save(region_map_path, region_map)
        print("Saved region_map to", region_map_path)
    
    safe_makedirs(args.out_dir)
    
    # 处理 Test 集
    all_rows = {'test': test_data}
    splits = ['test']
    
    if args.nprocs > 1 and torch.cuda.is_available():
        mp.spawn(run_on_gpu, nprocs=args.nprocs, args=(args, all_rows, splits, region_map), join=True)
    else:
        run_on_gpu(0, args, all_rows, splits, region_map)
    
    # 打印自适应 k_intra 信息
    print("\n" + "="*80)
    print("Adaptive k_intra configuration:")
    print("="*80)
    for region_id, channels in enumerate(brain_regions):
        n_r = len(channels)
        k_r = max(1, int(np.ceil(0.5 * (n_r - 1))))
        print(f"Region {region_id}: {n_r} channels -> k_intra = {k_r}")
    print("="*80 + "\n")
    
    # 合并 Test 集结果
    merged_results = {'test': {}}
    merged_files = []
    for fname in os.listdir(args.out_dir):
        if fname.startswith("processed_rank"):
            fpath = os.path.join(args.out_dir, fname)
            try:
                with open(fpath, 'rb') as f:
                    part_dict = pickle.load(f)
                if 'test' in part_dict:
                    merged_results['test'].update(part_dict['test'])
                merged_files.append(fpath)
            except Exception as e:
                print(f"Warning: Could not read {fpath}. Error: {e}")
    
    # 保存 Test 集
    if merged_results['test']:
        final_file = os.path.join(args.out_dir, f"processed_test.pkl")
        with open(final_file, 'wb') as f:
            pickle.dump(merged_results['test'], f)
        print(f"✓ Test set saved: {len(merged_results['test'])} samples -> {final_file}")
    
    for fpath in merged_files:
        os.remove(fpath)
    
    # Step 4: 保存 Train+Val 原始数据 (用于后续 K-Fold)
    print("\n" + "="*80)
    print("Step 4: Saving Train+Val data for K-Fold cross-validation")
    print("="*80)
    
    train_val_file = os.path.join(args.out_dir, 'train_val_raw.pkl')
    with open(train_val_file, 'wb') as f:
        pickle.dump(train_val_data, f)
    print(f"✓ Train+Val raw data saved: {len(train_val_data)} samples -> {train_val_file}")
    print(f"  Label distribution: {dict(Counter([r['labels'] for r in train_val_data]))}")
    
    print("\n" + "="*80)
    print("Preprocessing Complete!")
    print("="*80)
    print(f"Next steps:")
    print(f"  1. Test set is ready at: {args.out_dir}/processed_test.pkl")
    print(f"  2. Train+Val raw data is at: {args.out_dir}/train_val_raw.pkl")
    print(f"  3. Use data_loader.py with K-Fold to split train_val_raw.pkl")
    print(f"  4. Resampling will be done during training (not here)")
    print("="*80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()