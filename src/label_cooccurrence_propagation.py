"""V13: label co-occurrence probability propagation for V7 scores.

Builds the label graph only from the chronological training partition (never from
validation labels). Related-label probabilities are propagated through a sparse
top-k association matrix, then blended with the base V7 probabilities.

A small beta grid is evaluated. The best propagated probability matrix is saved
for the existing per-label threshold optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .optimize_thresholds import find_best_global_threshold, fast_macro_f1


def build_label_graph(
    y_fit: np.ndarray,
    topk: int,
    min_cooc: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return column-normalized source->target association matrix."""
    y = y_fit.astype(np.float32, copy=False)
    n = float(y.shape[0])
    support = y.sum(axis=0).astype(np.float64)
    cooc = (y.T @ y).astype(np.float64)
    np.fill_diagonal(cooc, 0.0)

    # Lift > 1 means two labels co-occur more often than independence predicts.
    expected = np.outer(support, support) / max(n, 1.0)
    lift = (cooc + eps) / (expected + eps)
    assoc = np.maximum(np.log(lift), 0.0)

    # Downweight weak/noisy pairs.
    reliability = cooc / (cooc + 100.0)
    assoc *= reliability
    assoc[cooc < min_cooc] = 0.0
    np.fill_diagonal(assoc, 0.0)

    # Keep the strongest source labels for each target label.
    n_labels = assoc.shape[0]
    if topk < n_labels:
        for j in range(n_labels):
            col = assoc[:, j]
            nz = np.flatnonzero(col > 0)
            if len(nz) > topk:
                keep = nz[np.argpartition(col[nz], -topk)[-topk:]]
                mask = np.ones(n_labels, dtype=bool)
                mask[keep] = False
                assoc[mask, j] = 0.0

    # Each target becomes a weighted average of related source probabilities.
    col_sum = assoc.sum(axis=0, keepdims=True)
    graph = np.divide(
        assoc,
        col_sum,
        out=np.zeros_like(assoc, dtype=np.float64),
        where=col_sum > 0,
    )
    return graph.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scores",
        default="outputs/oof/v7_tuned_sgd_knn_chronological_scores.npz",
    )
    p.add_argument("--train", default="train.csv")
    p.add_argument("--split", default="outputs/splits/chronological_group_holdout.npz")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--min-cooc", type=int, default=30)
    p.add_argument("--beta-grid", default="0,0.025,0.05,0.075,0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--tag", default="v13_cooc_chronological")
    args = p.parse_args()

    pack = np.load(args.scores, allow_pickle=False)
    row_idx = pack["row_idx"].astype(np.int64)
    base_probs = pack["probs"].astype(np.float32)
    y_true = pack["y_true"].astype(np.int8)
    labels = pack["labels"].astype(str)

    split = np.load(args.split)
    fit_idx = split["train_idx"].astype(np.int64)
    val_idx = split["val_idx"].astype(np.int64)
    if not np.array_equal(row_idx, val_idx):
        raise ValueError("score rows do not match chronological validation split")

    train = pd.read_csv(args.train)
    y_all = train[labels].to_numpy(dtype=np.int8)
    y_fit = y_all[fit_idx]

    print(
        f"[graph] fit_rows={len(fit_idx)} labels={len(labels)} "
        f"topk={args.topk} min_cooc={args.min_cooc}",
        flush=True,
    )
    graph = build_label_graph(y_fit, args.topk, args.min_cooc)
    nnz = int(np.count_nonzero(graph))
    print(f"[graph] edges={nnz}", flush=True)

    propagated = base_probs @ graph
    betas = [float(x) for x in args.beta_grid.split(",") if x.strip()]

    rows = []
    best = None
    best_probs = None

    for beta in betas:
        probs = ((1.0 - beta) * base_probs + beta * propagated).astype(np.float32)
        t, global_f1 = find_best_global_threshold(
            y_true, probs, 0.10, 0.90, 0.01
        )
        pred = probs >= t
        rec = {
            "beta": beta,
            "global_threshold": float(t),
            "macro_f1_global_fit": float(global_f1),
            "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
        }
        rows.append(rec)
        print(
            f"beta={beta:.3f} global_t={t:.2f} "
            f"F1={global_f1:.6f} pred/sample={pred.sum(axis=1).mean():.3f}",
            flush=True,
        )
        if best is None or global_f1 > best["macro_f1_global_fit"]:
            best = rec
            best_probs = probs

    out = Path(args.output_root)
    (out / "oof").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "adaptive").mkdir(parents=True, exist_ok=True)

    score_path = out / "oof" / f"{args.tag}_scores.npz"
    np.savez_compressed(
        score_path,
        row_idx=row_idx,
        probs=best_probs,
        y_true=y_true,
        labels=labels,
    )
    np.savez_compressed(
        out / "adaptive" / f"{args.tag}_graph.npz",
        graph=graph,
        labels=labels,
        beta=np.float32(best["beta"]),
    )

    summary = {
        "version": "V13",
        "source_scores": args.scores,
        "graph_topk": args.topk,
        "graph_min_cooc": args.min_cooc,
        "graph_edges": nnz,
        "best": best,
        "all": rows,
    }
    summary_path = out / "metrics" / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"BEST beta                 : {best['beta']:.3f}")
    print(f"BEST global threshold     : {best['global_threshold']:.2f}")
    print(f"Macro F1 global fit*      : {best['macro_f1_global_fit']:.6f}")
    print(f"pred labels/sample        : {best['pred_labels_per_sample']:.3f}")
    print(f"saved scores              : {score_path}")
    print("* beta/global threshold selected on this validation set")
    print("=" * 80)


if __name__ == "__main__":
    main()
