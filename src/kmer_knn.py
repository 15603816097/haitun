"""TF-IDF k-mer KNN validation using cached V1 feature matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .metrics import macro_f1_skip_empty
from .train_kmer import config_fingerprint, load_config


def resolve_cache(config: dict, mode: str, cache_dir: Path) -> Path:
    fp = config_fingerprint(config, mode)
    return cache_dir / f"v1_{mode}_{fp}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random", "tail", "tail_clean"], default="random")
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--random-split", default="outputs/splits/group_random_seed42.npz")
    p.add_argument("--tail-split", default="outputs/splits/test_like_tail.npz")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    cache = resolve_cache(cfg, args.mode, Path(args.cache_dir))
    x_fit_path = cache / "X_fit.npz"
    x_eval_path = cache / "X_eval.npz"
    if not x_fit_path.exists() or not x_eval_path.exists():
        raise FileNotFoundError(
            f"missing TF-IDF cache under {cache}. "
            f"Run: python -m src.train_kmer --mode {args.mode}"
        )

    print(f"[cache] {cache}")
    X_fit = sparse.load_npz(x_fit_path).tocsr()
    X_eval = sparse.load_npz(x_eval_path).tocsr()
    print(f"[features] fit={X_fit.shape}, eval={X_eval.shape}")

    df = pd.read_csv(args.train)
    labels = sorted(
        [c for c in df.columns if c.startswith("label_")],
        key=lambda c: int(c.split("_")[1]),
    )
    y = df[labels].to_numpy(dtype=np.float32)

    if args.mode == "random":
        sp = np.load(args.random_split)
        fit_idx, eval_idx = sp["train_idx"], sp["val_idx"]
    else:
        sp = np.load(args.tail_split)
        fit_idx, eval_idx = sp["head_idx"], sp["tail_idx"]

    nn = NearestNeighbors(
        n_neighbors=args.k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X_fit)
    dist, ind = nn.kneighbors(X_eval, return_distance=True)

    sim = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    w = np.power(sim + 1e-8, args.power, dtype=np.float32)
    neigh_y = y[fit_idx][ind]
    probs = (
        (neigh_y * w[..., None]).sum(axis=1)
        / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
    ).astype(np.float32)

    y_true = y[eval_idx].astype(np.int8)
    pred = (probs >= 0.5).astype(np.int8)
    score = macro_f1_skip_empty(y_true, pred)

    out = Path(args.output_root)
    (out / "oof").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "oof" / f"v5_kmer_knn_{args.mode}_scores.npz",
        row_idx=eval_idx.astype(np.int64),
        probs=probs,
        y_true=y_true,
        labels=np.asarray(labels),
    )

    summary = {
        "version": "V5",
        "mode": args.mode,
        "k": args.k,
        "power": args.power,
        "macro_f1_at_0.5": float(score),
        "true_labels_per_sample": float(y_true.sum(axis=1).mean()),
        "predicted_labels_per_sample": float(pred.sum(axis=1).mean()),
        "mean_top1_similarity": float(sim[:, 0].mean()),
        "mean_topk_similarity": float(sim.mean()),
    }
    path = out / "metrics" / f"v5_kmer_knn_{args.mode}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"V5 k-mer TF-IDF KNN {args.mode}")
    print(f"K={args.k}, power={args.power}")
    print(f"Macro F1 @ 0.50         : {score:.6f}")
    print(f"true labels/sample      : {y_true.sum(axis=1).mean():.3f}")
    print(f"pred labels/sample      : {pred.sum(axis=1).mean():.3f}")
    print(f"mean top1 similarity    : {sim[:,0].mean():.4f}")
    print(f"mean topK similarity    : {sim.mean():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
