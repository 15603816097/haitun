"""V4 ESM embedding KNN validator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .metrics import macro_f1_skip_empty


def load_split(mode: str):
    if mode == "random":
        p = np.load("outputs/splits/group_random_seed42.npz")
        return p["train_idx"], p["val_idx"]
    if mode == "tail":
        p = np.load("outputs/splits/test_like_tail.npz")
        return p["head_idx"], p["tail_idx"]
    raise ValueError(mode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random","tail"], default="random")
    p.add_argument("--embeddings", default="cache/esm2_150m/train_embeddings.npy")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    x = np.load(args.embeddings, mmap_mode="r")
    df = pd.read_csv(args.train)
    labels = sorted([c for c in df.columns if c.startswith("label_")], key=lambda c:int(c.split("_")[1]))
    y = df[labels].to_numpy(dtype=np.float32)

    tr_idx, va_idx = load_split(args.mode)
    xtr = np.asarray(x[tr_idx], dtype=np.float32)
    xva = np.asarray(x[va_idx], dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=args.k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(xtr)
    dist, ind = nn.kneighbors(xva, return_distance=True)

    sim = np.clip(1.0 - dist, 0.0, 1.0)
    w = np.power(sim + 1e-8, args.power)
    neigh_y = y[tr_idx][ind]
    probs = (neigh_y * w[..., None]).sum(axis=1) / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)

    pred = (probs >= 0.5).astype(np.int8)
    y_true = y[va_idx].astype(np.int8)
    score = macro_f1_skip_empty(y_true, pred)

    out = Path(args.output_root)
    (out/"oof").mkdir(parents=True, exist_ok=True)
    (out/"metrics").mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out/"oof"/f"v4_esm_knn_{args.mode}_scores.npz",
        row_idx=va_idx.astype(np.int64),
        probs=probs.astype(np.float32),
        y_true=y_true,
        labels=np.asarray(labels),
    )
    summary = {
        "version":"V4",
        "mode":args.mode,
        "k":args.k,
        "power":args.power,
        "macro_f1_at_0.5":float(score),
        "true_labels_per_sample":float(y_true.sum(axis=1).mean()),
        "predicted_labels_per_sample":float(pred.sum(axis=1).mean()),
        "mean_top1_similarity":float(sim[:,0].mean()),
        "mean_topk_similarity":float(sim.mean()),
    }
    path = out/"metrics"/f"v4_esm_knn_{args.mode}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("="*80)
    print(f"V4 ESM-KNN {args.mode}")
    print(f"K={args.k}, power={args.power}")
    print(f"Macro F1 @ 0.50         : {score:.6f}")
    print(f"true labels/sample      : {y_true.sum(axis=1).mean():.3f}")
    print(f"pred labels/sample      : {pred.sum(axis=1).mean():.3f}")
    print(f"mean top1 similarity    : {sim[:,0].mean():.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
