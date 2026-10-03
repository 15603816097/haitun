"""V11: search three-way blend of balanced SGD, unweighted SGD, and tuned KNN.

All inputs are existing chronological OOF probability files, so no retraining is needed.
The script performs a coarse simplex search over three weights at threshold 0.5 and
saves the best blended probabilities for downstream per-label threshold tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import macro_f1_skip_empty


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
    p.add_argument("--balanced", default="outputs/oof/v1_kmer_sgd_chronological_scores.npz")
    p.add_argument("--unweighted", default="outputs/oof/v10_sgd_a5e6_none_chronological_scores.npz")
    p.add_argument("--knn", default="outputs/oof/v7_kmer_knn_k40_p4p0_chronological_scores.npz")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tag", default="v11_threeway_chronological")
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    A = load_scores(args.balanced)
    B = load_scores(args.unweighted)
    C = load_scores(args.knn)

    for other in (B, C):
        if not np.array_equal(A["row_idx"], other["row_idx"]):
            raise ValueError("row_idx mismatch")
        if not np.array_equal(A["labels"], other["labels"]):
            raise ValueError("label mismatch")
        if not np.array_equal(A["y_true"], other["y_true"]):
            raise ValueError("y_true mismatch")

    y = A["y_true"]
    weights = np.arange(0.0, 1.0 + args.step * 0.5, args.step, dtype=np.float32)

    best = None
    best_probs = None
    rows = []

    for wa in weights:
        for wb in weights:
            wc = 1.0 - float(wa) - float(wb)
            if wc < -1e-8:
                continue
            if wc < 0:
                wc = 0.0
            probs = float(wa) * A["probs"] + float(wb) * B["probs"] + wc * C["probs"]
            pred = (probs >= args.threshold).astype(np.int8)
            score = macro_f1_skip_empty(y, pred)
            rec = {
                "balanced_weight": float(wa),
                "unweighted_weight": float(wb),
                "knn_weight": float(wc),
                "macro_f1": float(score),
                "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
            }
            rows.append(rec)
            if best is None or score > best["macro_f1"]:
                best = rec
                best_probs = probs.astype(np.float32)

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
        "version": "V11",
        "threshold": args.threshold,
        "best": best,
        "top10": rows[:10],
    }
    summary_path = out / "metrics" / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(
        f"BEST balanced={best['balanced_weight']:.2f} "
        f"unweighted={best['unweighted_weight']:.2f} "
        f"knn={best['knn_weight']:.2f}"
    )
    print(f"Macro F1 @ {args.threshold:.2f}: {best['macro_f1']:.6f}")
    print(f"pred labels/sample      : {best['pred_labels_per_sample']:.3f}")
    print(f"saved                   : {score_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
