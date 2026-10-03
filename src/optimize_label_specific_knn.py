"""V9: cross-fitted label-specific selection across KNN settings, blend weight and threshold.

For each label, calibration chooses among:
  k in {5,10,20,40,80}
  power in {2,4,6}
  SGD blend weight in {0.2,0.3,0.4,0.5,0.6}
and fits a label threshold. GroupKFold estimates honest Macro F1.

The selected full-data parameters are saved for a later test submission pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from .optimize_thresholds import best_threshold_for_label, fast_macro_f1
from .train_kmer import config_fingerprint, load_config
from .tune_kmer_knn_blend import compute_knn_probs


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def build_candidate_probs(
    y_fit: np.ndarray,
    neighbor_idx: np.ndarray,
    similarities: np.ndarray,
    k_grid: list[int],
    power_grid: list[float],
    batch_size: int,
) -> dict[tuple[int, float], np.ndarray]:
    out = {}
    for k in k_grid:
        for power in power_grid:
            probs = compute_knn_probs(
                y_fit=y_fit,
                neighbor_idx=neighbor_idx,
                similarities=similarities,
                k=k,
                power=power,
                batch_size=batch_size,
            )
            out[(k, power)] = probs.astype(np.float32)
            print(f"[knn] built k={k} power={power:g}")
    return out


def fit_label_params(
    y: np.ndarray,
    sgd: np.ndarray,
    knn_map: dict[tuple[int, float], np.ndarray],
    k_grid: list[int],
    power_grid: list[float],
    weight_grid: list[float],
    fallback_k: int,
    fallback_power: float,
    fallback_weight: float,
    fallback_threshold: float,
    shrink_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_labels = y.shape[1]
    best_k = np.full(n_labels, fallback_k, dtype=np.int16)
    best_power = np.full(n_labels, fallback_power, dtype=np.float32)
    best_weight = np.full(n_labels, fallback_weight, dtype=np.float32)
    best_threshold = np.full(n_labels, fallback_threshold, dtype=np.float32)

    support = y.sum(axis=0).astype(np.float64)

    for j in range(n_labels):
        if support[j] <= 0:
            continue

        top_score = -1.0
        top = (fallback_k, fallback_power, fallback_weight, fallback_threshold)

        for k in k_grid:
            for power in power_grid:
                kp = knn_map[(k, power)][:, j]
                for w in weight_grid:
                    p = w * sgd[:, j] + (1.0 - w) * kp
                    t = best_threshold_for_label(
                        y[:, j],
                        p,
                        min_threshold=0.05,
                        max_threshold=0.95,
                        fallback=fallback_threshold,
                    )
                    pred = p >= t
                    tp = np.logical_and(y[:, j] == 1, pred).sum()
                    pred_pos = pred.sum()
                    denom = support[j] + pred_pos
                    score = 0.0 if denom == 0 else float(2.0 * tp / denom)

                    if score > top_score:
                        top_score = score
                        top = (k, power, w, t)

        # Shrink noisy per-label continuous params toward robust V7 defaults.
        shrink = support[j] / (support[j] + shrink_k)
        best_k[j] = int(top[0])
        best_power[j] = float(top[1])
        best_weight[j] = float(shrink * top[2] + (1.0 - shrink) * fallback_weight)
        best_threshold[j] = float(
            shrink * top[3] + (1.0 - shrink) * fallback_threshold
        )

    return best_k, best_power, best_weight, best_threshold


def apply_label_params(
    sgd: np.ndarray,
    knn_map: dict[tuple[int, float], np.ndarray],
    ks: np.ndarray,
    powers: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    n_rows, n_labels = sgd.shape
    pred = np.zeros((n_rows, n_labels), dtype=np.int8)

    for j in range(n_labels):
        key = (int(ks[j]), float(powers[j]))
        kp = knn_map[key][:, j]
        p = weights[j] * sgd[:, j] + (1.0 - weights[j]) * kp
        pred[:, j] = (p >= thresholds[j]).astype(np.int8)
    return pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sgd-scores", default="outputs/oof/v1_kmer_sgd_chronological_scores.npz")
    p.add_argument("--mode", choices=["chronological"], default="chronological")
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--split", default="outputs/splits/chronological_group_holdout.npz")
    p.add_argument("--k-grid", default="5,10,20,40,80")
    p.add_argument("--power-grid", default="2,4,6")
    p.add_argument("--weight-grid", default="0.2,0.3,0.4,0.5,0.6")
    p.add_argument("--fallback-k", type=int, default=40)
    p.add_argument("--fallback-power", type=float, default=4.0)
    p.add_argument("--fallback-weight", type=float, default=0.40)
    p.add_argument("--fallback-threshold", type=float, default=0.48)
    p.add_argument("--shrink-k", type=float, default=500.0)
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    cache = Path(args.cache_dir) / f"v1_{args.mode}_{config_fingerprint(cfg,args.mode)}"
    X_fit = sparse.load_npz(cache / "X_fit.npz").tocsr()
    X_eval = sparse.load_npz(cache / "X_eval.npz").tocsr()

    df = pd.read_csv(args.train)
    labels = sorted(
        [c for c in df.columns if c.startswith("label_")],
        key=lambda c: int(c.split("_")[1]),
    )
    y_all = df[labels].to_numpy(dtype=np.float32)

    split = np.load(args.split)
    fit_idx = split["train_idx"].astype(np.int64)
    eval_idx = split["val_idx"].astype(np.int64)
    y_fit = y_all[fit_idx]
    y_true = y_all[eval_idx].astype(np.int8)

    sgd_pack = np.load(args.sgd_scores, allow_pickle=False)
    if not np.array_equal(sgd_pack["row_idx"].astype(np.int64), eval_idx):
        raise ValueError("SGD rows do not match chronological validation")
    sgd = sgd_pack["probs"].astype(np.float32)

    k_grid = parse_ints(args.k_grid)
    power_grid = parse_floats(args.power_grid)
    weight_grid = parse_floats(args.weight_grid)

    max_k = max(k_grid)
    print(f"[neighbors] max_k={max_k}")
    nn = NearestNeighbors(
        n_neighbors=max_k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X_fit)
    dist, ind = nn.kneighbors(X_eval, return_distance=True)
    sim = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)

    knn_map = build_candidate_probs(
        y_fit=y_fit,
        neighbor_idx=ind,
        similarities=sim,
        k_grid=k_grid,
        power_grid=power_grid,
        batch_size=args.batch_size,
    )

    groups = df.iloc[eval_idx]["sequence"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=args.folds)

    pred_cf = np.zeros_like(y_true, dtype=np.int8)
    fold_rows = []

    for fold, (cal, ev) in enumerate(gkf.split(sgd, y_true, groups=groups), 1):
        cal_knn = {k: v[cal] for k, v in knn_map.items()}
        ev_knn = {k: v[ev] for k, v in knn_map.items()}

        ks, powers, weights, thresholds = fit_label_params(
            y=y_true[cal],
            sgd=sgd[cal],
            knn_map=cal_knn,
            k_grid=k_grid,
            power_grid=power_grid,
            weight_grid=weight_grid,
            fallback_k=args.fallback_k,
            fallback_power=args.fallback_power,
            fallback_weight=args.fallback_weight,
            fallback_threshold=args.fallback_threshold,
            shrink_k=args.shrink_k,
        )
        pred = apply_label_params(
            sgd=sgd[ev],
            knn_map=ev_knn,
            ks=ks,
            powers=powers,
            weights=weights,
            thresholds=thresholds,
        )
        pred_cf[ev] = pred
        score = fast_macro_f1(y_true[ev], pred)
        fold_rows.append({
            "fold": fold,
            "macro_f1": float(score),
            "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
        })
        print(
            f"fold {fold}: Macro F1={score:.6f}, "
            f"pred/sample={pred.sum(axis=1).mean():.3f}"
        )

    crossfit = fast_macro_f1(y_true, pred_cf)

    ks, powers, weights, thresholds = fit_label_params(
        y=y_true,
        sgd=sgd,
        knn_map=knn_map,
        k_grid=k_grid,
        power_grid=power_grid,
        weight_grid=weight_grid,
        fallback_k=args.fallback_k,
        fallback_power=args.fallback_power,
        fallback_weight=args.fallback_weight,
        fallback_threshold=args.fallback_threshold,
        shrink_k=args.shrink_k,
    )
    pred_fit = apply_label_params(sgd, knn_map, ks, powers, weights, thresholds)
    fit_all = fast_macro_f1(y_true, pred_fit)

    out = Path(args.output_root)
    (out / "adaptive").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    param_path = out / "adaptive" / "v9_label_specific_knn_params.npz"
    np.savez_compressed(
        param_path,
        labels=np.asarray(labels),
        k=ks,
        power=powers,
        sgd_weight=weights,
        thresholds=thresholds,
    )

    unique_k, counts_k = np.unique(ks, return_counts=True)
    unique_p, counts_p = np.unique(powers, return_counts=True)
    summary = {
        "version": "V9",
        "crossfit_macro_f1": float(crossfit),
        "fit_all_macro_f1": float(fit_all),
        "crossfit_pred_labels_per_sample": float(pred_cf.sum(axis=1).mean()),
        "true_labels_per_sample": float(y_true.sum(axis=1).mean()),
        "mean_sgd_weight": float(weights.mean()),
        "median_sgd_weight": float(np.median(weights)),
        "k_distribution": {str(int(k)): int(c) for k, c in zip(unique_k, counts_k)},
        "power_distribution": {str(float(k)): int(c) for k, c in zip(unique_p, counts_p)},
        "folds": fold_rows,
        "shrink_k": args.shrink_k,
    }
    (out / "metrics" / "v9_label_specific_knn_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"Crossfit Macro F1       : {crossfit:.6f}")
    print(f"Fit-all Macro F1*       : {fit_all:.6f}")
    print(
        f"pred labels/sample      : {pred_cf.sum(axis=1).mean():.3f} "
        f"(true={y_true.sum(axis=1).mean():.3f})"
    )
    print(f"mean/median SGD weight  : {weights.mean():.3f}/{np.median(weights):.3f}")
    print(f"k distribution          : {summary['k_distribution']}")
    print(f"power distribution      : {summary['power_distribution']}")
    print(f"saved params            : {param_path}")
    print("* optimistic fit-all")
    print("=" * 80)


if __name__ == "__main__":
    main()
