import os
import pickle
import argparse
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold

def load_cache(cache_path: str):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict) or len(cache) == 0:
        raise ValueError("graph_cache.pkl is empty or not a dict.")
    # 基本字段检查
    sid = next(iter(cache.keys()))
    required = {"id", "label", "node_feats", "A_intra", "A_global"}
    missing = required - set(cache[sid].keys())
    if missing:
        raise ValueError(f"Cache sample missing fields: {missing}")
    return cache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True, help="Directory containing graph_cache.pkl")
    parser.add_argument("--cache_name", type=str, default="graph_cache.pkl")
    parser.add_argument("--n_splits", type=int, default=8, help="8 means 7:1 train/val each fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary_stratify", action="store_true",
                        help="Stratify by (label==0 vs label!=0) instead of full multiclass labels")
    parser.add_argument("--export_fold_pkls", action="store_true",
                        help="Export processed_train_fold{i}.pkl and processed_val_fold{i}.pkl")
    args = parser.parse_args()

    cache_path = os.path.join(args.processed_dir, args.cache_name)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    cache = load_cache(cache_path)

    # IDs & labels
    ids = np.array(list(cache.keys()))
    labels = np.array([cache[sid]["label"] for sid in ids])

    print(f"Loaded cache: {cache_path}")
    print(f"Total samples: {len(ids)}")
    print(f"Label distribution (raw): {dict(Counter(labels))}")

    if args.binary_stratify:
        strat_labels = (labels != 0).astype(int)
        print(f"Using binary stratify (0 vs non-0). Dist: {dict(Counter(strat_labels))}")
    else:
        strat_labels = labels
        print("Using multiclass stratify (recommended if class counts allow).")

    # 7:1 = 8-fold
    if args.n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if args.n_splits == 8:
        print("n_splits=8 => each fold is exactly 7/8 train and 1/8 val (7:1).")
    else:
        print(f"n_splits={args.n_splits} => each fold train:val ~= {(args.n_splits-1)}:1")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(strat_labels)), strat_labels)):
        train_ids = ids[train_idx].tolist()
        val_ids = ids[val_idx].tolist()

        # 统计一下每折分布，方便你快速 sanity check
        train_labels = [cache[sid]["label"] for sid in train_ids]
        val_labels = [cache[sid]["label"] for sid in val_ids]

        folds.append({
            "fold": fold_idx,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "train_label_dist": dict(Counter(train_labels)),
            "val_label_dist": dict(Counter(val_labels)),
        })

        print(f"\nFold {fold_idx}: train={len(train_ids)} val={len(val_ids)}")
        print(f"  train dist: {dict(Counter(train_labels))}")
        print(f"  val   dist: {dict(Counter(val_labels))}")

        if args.export_fold_pkls:
            train_dict = {sid: cache[sid] for sid in train_ids}
            val_dict = {sid: cache[sid] for sid in val_ids}
            train_out = os.path.join(args.processed_dir, f"processed_train_fold{fold_idx}.pkl")
            val_out = os.path.join(args.processed_dir, f"processed_val_fold{fold_idx}.pkl")
            with open(train_out, "wb") as f:
                pickle.dump(train_dict, f)
            with open(val_out, "wb") as f:
                pickle.dump(val_dict, f)

    folds_out = os.path.join(args.processed_dir, "folds.pkl")
    with open(folds_out, "wb") as f:
        pickle.dump(folds, f)

    print(f"\nSaved folds to: {folds_out}")
    if args.export_fold_pkls:
        print("Also exported per-fold processed_train/val pkl files.")

if __name__ == "__main__":
    main()