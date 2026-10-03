"""V12: blend the best tuned balanced SGD (alpha=2e-6) with tuned KNN.

Reuses existing OOF probabilities. Searches a single blend weight at threshold 0.5,
then saves the best blended scores for per-label threshold optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .optimize_thresholds import fast_macro_f1


def load_scores(path: str):
    p = np.load(path, allow_pickle=False)
    return {
        "row_idx": p["row_idx"].astype(np.int64),
        "probs": p["probs"].astype(np.float32),
        "y_true": p["y_true"].astype(np.int8),
        "labels": p["labels"].astype(str),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sgd",
        default="outputs/oof/v10_sgd_a2e6_bal_chronological_scores.npz",
    )
    p.add_argument(
        "--knn",
        default="outputs/oof/v7_kmer_knn_k40_p4p0_chronological_scores.npz",
    )
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tag", default="v12_a2e6_knn_chronological")
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    A = load_scores(args.sgd)
    B = load_scores(args.knn)

    if not np.array_equal(A["row_idx"], B["row_idx"]):
        raise ValueError("row_idx mismatch")
    if not np.array_equal(A["labels"], B["labels"]):
        raise ValueError("label mismatch")
    if not np.array_equal(A["y_true"], B["y_true"]):
        raise ValueError("y_true mismatch")

    y = A["y_true"]
    weights = np.arange(0.0, 1.0 + args.step * 0.5, args.step, dtype=np.float32)

    best = None
    best_probs = None
    rows = []

    print(f"[search] {len(weights)} SGD weights", flush=True)

    for i, w_sgd in enumerate(weights, start=1):
        w_knn = 1.0 - float(w_sgd)
        probs = float(w_sgd) * A["probs"] + w_knn * B["probs"]
        pred = probs >= args.threshold
        score = fast_macro_f1(y, pred)
        rec = {
            "sgd_weight": float(w_sgd),
            "knn_weight": float(w_knn),
            "macro_f1": float(score),
            "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
        }
        rows.append(rec)

        if best is None or score > best["macro_f1"]:
            best = rec
            best_probs = probs.astype(np.float32)

        if i == 1 or i % 5 == 0 or i == len(weights):
            print(
                f"[search] {i}/{len(weights)} "
                f"current={score:.6f} best={best['macro_f1']:.6f} "
                f"w_sgd={float(w_sgd):.3f}",
                flush=True,
            )

    out = Path(args.output_root)
    (out / "oof").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    score_path = out / "oof" / f"{args.tag}_scores.npz"
    np.savez_compressed(
        score_path,
        row_idx=A["row_idx"],
        probs=best_probs,
        y_true=y,
        labels=A["labels"],
    )

    rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    summary = {
        "version": "V12",
        "threshold": args.threshold,
        "best": best,
        "top10": rows[:10],
    }
    summary_path = out / "metrics" / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(
        f"BEST SGD={best['sgd_weight']:.3f} "
        f"KNN={best['knn_weight']:.3f}"
    )
    print(f"Macro F1 @ {args.threshold:.2f}: {best['macro_f1']:.6f}")
    print(f"pred labels/sample      : {best['pred_labels_per_sample']:.3f}")
    print(f"saved                   : {score_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
