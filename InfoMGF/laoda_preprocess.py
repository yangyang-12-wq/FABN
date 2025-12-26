"""
Preprocess fNIRS data with the laoda-style pipeline (Spearman correlation on raw time series),
then package results into hh/InfoMGF-compatible processed_{split}.pkl files.

Usage example:
    python laoda_preprocess.py ^
        --source_dir ../fnirs/fnirs/resting/feature ^
        --out_dir ./configs/data/fnirs/processed_laoda ^
        --feature_type oxy ^
        --k_intra 8 --k_global 8
"""

import argparse
import os
import pickle
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import torch

from utils import safe_makedirs
from utils_graph_build import (
    brain_regions,
    make_region_map,
    topk_sparsify_sym_row_normalize,
    compute_structure_encodings,
)


def load_feature_pickle(feature_path: str, split: str) -> Dict[str, Any]:
    p = os.path.join(feature_path, f"{split}_data.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found")
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data


def compute_spearman_corr(ts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Laoda-style Spearman correlation for a single sample.
    Args:
        ts: np.ndarray with shape (T, C)
    Returns:
        (C, C) correlation matrix
    """
    # torch ops to match laoda implementation
    x = torch.from_numpy(ts)
    # rank along time dimension
    ranks = torch.argsort(torch.argsort(x, dim=0), dim=0).float()
    centered = ranks - ranks.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True, unbiased=False) + eps
    normalized = centered / std
    corr = (normalized.transpose(0, 1) @ normalized) / normalized.size(0)
    corr = torch.clamp(corr, -0.99, 0.99)
    diag_idx = torch.arange(corr.size(0))
    # Laoda keeps self-correlation as 1.0
    corr[diag_idx, diag_idx] = 1.0
    return corr.numpy()


def build_intra_from_corr(corr: np.ndarray, region_map: np.ndarray) -> np.ndarray:
    """Mask correlation to intra-region channels only."""
    C = corr.shape[0]
    mask = np.zeros((C, C), dtype=float)
    assigned = np.where(region_map != -1)[0]
    for i in assigned:
        for j in assigned:
            if i != j and region_map[i] == region_map[j]:
                mask[i, j] = 1.0
    return corr * mask


def binarize_labels(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Match laoda's binary mapping: 0 stays 0, 1/2/3/4 -> 1."""
    out = []
    for r in rows:
        lbl = int(r["labels"])
        if lbl not in [0, 1, 2, 3, 4]:
            continue
        r = dict(r)
        r["labels"] = 0 if lbl == 0 else 1
        out.append(r)
    return out


def prepare_rows(feature_data: Dict[str, Any], feature_type: str) -> List[Dict[str, Any]]:
    rows = []
    ids = list(feature_data[feature_type].keys())
    for sid in ids:
        ts = feature_data[feature_type][sid]
        if isinstance(ts, torch.Tensor):
            ts = ts.cpu().numpy()
        rows.append({"id": sid, "data": ts, "labels": int(feature_data["labels"][sid])})
    rows = binarize_labels(rows)
    return rows


def process_split(
    rows: List[Dict[str, Any]],
    region_map: np.ndarray,
    k_intra: int,
    k_global: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for entry in rows:
        sid = entry["id"]
        ts_np = entry["data"]
        label = entry["labels"]

        corr = compute_spearman_corr(ts_np)
        A_intra_dense = build_intra_from_corr(corr, region_map)
        A_global_dense = corr.copy()

        if k_intra and k_intra > 0:
            A_intra = topk_sparsify_sym_row_normalize(A_intra_dense, k_intra).astype(np.float32)
        else:
            A_intra = A_intra_dense.astype(np.float32)

        if k_global and k_global > 0:
            A_global = topk_sparsify_sym_row_normalize(A_global_dense, k_global).astype(np.float32)
        else:
            A_global = A_global_dense.astype(np.float32)

        # Node features: laoda-style — use each node's correlation profile (dense Spearman row)
        # Shape: (C, C), consistent with laoda using correlation as the fundamental feature.
        node_feats = A_global_dense.astype(np.float32)

        out[sid] = {
            "id": sid,
            "label": label,
            "node_feats": node_feats,
            "A_intra": A_intra,
            "A_global": A_global,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing train/val/test_data.pkl")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for processed_{split}.pkl")
    parser.add_argument("--feature_type", type=str, default="oxy", choices=["oxy", "dxy", "both"])
    parser.add_argument("--k_intra", type=int, default=0, help="top-k for intra; 0 means no sparsify (laoda default)")
    parser.add_argument("--k_global", type=int, default=0, help="top-k for global; 0 means no sparsify (laoda default)")
    parser.add_argument("--region_map", type=str, default=None, help="Optional path to region_map.npy")
    args = parser.parse_args()

    safe_makedirs(args.out_dir)

    # load or create region_map
    if args.region_map and os.path.exists(args.region_map):
        region_map = np.load(args.region_map)
    else:
        # infer channel count from train split
        tmp = load_feature_pickle(args.source_dir, "train")
        sample = next(iter(tmp[args.feature_type].values()))
        C = sample.shape[1] if isinstance(sample, np.ndarray) else sample.shape[1]
        region_map = make_region_map(brain_regions, C)
        # save alongside out_dir for reuse
        np.save(os.path.join(args.out_dir, "region_map.npy"), region_map)

    processed_all: Dict[str, Dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        feature_data = load_feature_pickle(args.source_dir, split)
        rows = prepare_rows(feature_data, args.feature_type)
        print(f"{split}: {len(rows)} samples, label dist {Counter([r['labels'] for r in rows])}")
        processed = process_split(rows, region_map, args.k_intra, args.k_global)
        processed_all[split] = processed
        out_file = os.path.join(args.out_dir, f"processed_{split}.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(processed, f)
        print(f"saved {len(processed)} graphs -> {out_file}")

    print("Done. Use data_loader.BrainGraphDataset with raw_file_path pointing to processed_{split}.pkl")


if __name__ == "__main__":
    main()
