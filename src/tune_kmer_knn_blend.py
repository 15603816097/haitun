"""Tune TF-IDF KNN hyperparameters and SGD/KNN blend weight.

Reuses the cached TF-IDF matrices. Neighbors are searched once up to max(k_grid),
then each k/power setting is evaluated without refitting TF-IDF.

Examples:
  python -m src.tune_kmer_knn_blend --mode chronological \
    --sgd-scores outputs/oof/v1_kmer_sgd_chronological_scores.npz
  python -m src.tune_kmer_knn_blend --mode random \
    --sgd-scores outputs/oof/v1_kmer_sgd_random_scores.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .metrics import macro_f1_skip_empty
from .train_kmer import config_fingerprint, load_config


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def resolve_split(args):
    if args.mode == "random":
        sp = np.load(args.random_split)
        return sp["train_idx"].astype(np.int64), sp["val_idx"].astype(np.int64)
    if args.mode == "chronological":
        sp = np.load(args.chronological_split)
        return sp["train_idx"].astype(np.int64), sp["val_idx"].astype(np.int64)
    raise ValueError(args.mode)


def compute_knn_probs(
    y_fit: np.ndarray,
    neighbor_idx: np.ndarray,
    similarities: np.ndarray,
    k: int,
    power: float,
    batch_size: int = 256,
) -> np.ndarray:
    n = neighbor_idx.shape[0]
    n_labels = y_fit.shape[1]
    out = np.empty((n, n_labels), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        idx = neighbor_idx[start:end, :k]
        sim = similarities[start:end, :k]
        w = np.power(sim + 1e-8, power).astype(np.float32)
        neigh_y = y_fit[idx]
        numer = np.einsum("bk,bkl->bl", w, neigh_y, optimize=True)
        denom = np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
        out[start:end] = numer / denom
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random", "chronological"], default="chronological")
    p.add_argument("--sgd-scores", required=True)
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--random-split", default="outputs/splits/group_random_seed42.npz")
    p.add_argument("--chronological-split", default="outputs/splits/chronological_group_holdout.npz")
    p.add_argument("--k-grid", default="5,10,20,40,80")
    p.add_argument("--power-grid", default="1,2,4")
    p.add_argument("--blend-step", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    cache = Path(args.cache_dir) / f"v1_{args.mode}_{config_fingerprint(cfg, args.mode)}"
    x_fit_path = cache / "X_fit.npz"
    x_eval_path = cache / "X_eval.npz"
    if not x_fit_path.exists() or not x_eval_path.exists():
        raise FileNotFoundError(
            f"missing cache: {cache}. Run python -m src.train_kmer --mode {args.mode} first."
        )

    print(f"[cache] {cache}")
    X_fit = sparse.load_npz(x_fit_path).tocsr()
    X_eval = sparse.load_npz(x_eval_path).tocsr()

    df = pd.read_csv(args.train)
    labels = sorted(
        [c for c in df.columns if c.startswith("label_")],
        key=lambda c: int(c.split("_")[1]),
    )
    y_all = df[labels].to_numpy(dtype=np.float32)
    fit_idx, eval_idx = resolve_split(args)
    y_fit = y_all[fit_idx]
    y_true = y_all[eval_idx].astype(np.int8)

    sgd_pack = np.load(args.sgd_scores, allow_pickle=False)
    sgd_rows = sgd_pack["row_idx"].astype(np.int64)
    sgd_probs = sgd_pack["probs"].astype(np.float32)
    if not np.array_equal(sgd_rows, eval_idx):
        raise ValueError("SGD score row_idx does not match selected validation split")

    k_grid = sorted(set(parse_ints(args.k_grid)))
    power_grid = parse_floats(args.power_grid)
    max_k = max(k_grid)

    print(f"[neighbors] search max_k={max_k}, fit={X_fit.shape}, eval={X_eval.shape}")
    nn = NearestNeighbors(
        n_neighbors=max_k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X_fit)
    dist, neighbor_idx = nn.kneighbors(X_eval, return_distance=True)
    similarities = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    print(
        f"[neighbors] mean top1={similarities[:,0].mean():.4f}, "
        f"mean top{max_k}={similarities.mean():.4f}"
    )

    results = []
    best = None
    best_probs = None

    blend_weights = np.arange(
        0.0, 1.0 + args.blend_step * 0.5, args.blend_step, dtype=np.float32
    )

    for k in k_grid:
        for power in power_grid:
            knn_probs = compute_knn_probs(
                y_fit=y_fit,
                neighbor_idx=neighbor_idx,
                similarities=similarities,
                k=k,
                power=power,
                batch_size=args.batch_size,
            )
            knn_pred = (knn_probs >= args.threshold).astype(np.int8)
            knn_f1 = macro_f1_skip_empty(y_true, knn_pred)

            local_best = None
            for w_sgd in blend_weights:
                w_knn = 1.0 - float(w_sgd)
                probs = float(w_sgd) * sgd_probs + w_knn * knn_probs
                pred = (probs >= args.threshold).astype(np.int8)
                score = macro_f1_skip_empty(y_true, pred)
                rec = {
                    "k": int(k),
                    "power": float(power),
                    "knn_macro_f1": float(knn_f1),
                    "sgd_weight": float(w_sgd),
                    "knn_weight": float(w_knn),
                    "blend_macro_f1": float(score),
                    "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
                }
                if local_best is None or score > local_best["blend_macro_f1"]:
                    local_best = rec
                    local_best_probs = probs.astype(np.float32)

            results.append(local_best)
            print(
                f"k={k:>2} power={power:g} "
                f"knn={knn_f1:.6f} "
                f"blend={local_best['blend_macro_f1']:.6f} "
                f"w_sgd={local_best['sgd_weight']:.2f} "
                f"pred/sample={local_best['pred_labels_per_sample']:.3f}"
            )

            if best is None or local_best["blend_macro_f1"] > best["blend_macro_f1"]:
                best = local_best
                best_probs = local_best_probs.copy()

    out = Path(args.output_root)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "oof").mkdir(parents=True, exist_ok=True)

    tag = f"v7_tuned_sgd_knn_{args.mode}"
    np.savez_compressed(
        out / "oof" / f"{tag}_scores.npz",
        row_idx=eval_idx,
        probs=best_probs,
        y_true=y_true,
        labels=np.asarray(labels),
    )

    summary = {
        "version": "V7",
        "mode": args.mode,
        "threshold_during_search": args.threshold,
        "best": best,
        "all_settings": results,
        "mean_top1_similarity": float(similarities[:, 0].mean()),
        "max_k_mean_similarity": float(similarities.mean()),
    }
    summary_path = out / "metrics" / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"V7 tuned SGD + k-mer KNN ({args.mode})")
    print(
        f"BEST k={best['k']} power={best['power']} "
        f"SGD={best['sgd_weight']:.2f} KNN={best['knn_weight']:.2f}"
    )
    print(f"Macro F1 @ {args.threshold:.2f}: {best['blend_macro_f1']:.6f}")
    print(f"pred labels/sample      : {best['pred_labels_per_sample']:.3f}")
    print(f"saved scores            : {out/'oof'/f'{tag}_scores.npz'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
