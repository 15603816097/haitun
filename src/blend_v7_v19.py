"""V20: blend V7 main model with V19 4-6mer SGD."""
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
    p.add_argument("--v7", default="outputs/oof/v7_tuned_sgd_knn_chronological_scores.npz")
    p.add_argument("--v19", default="outputs/v19_kmer46/oof/v1_kmer_sgd_chronological_scores.npz")
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tag", default="v20_v7_kmer46_chronological")
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    A = load_scores(args.v7)
    B = load_scores(args.v19)

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

    print(f"[search] {len(weights)} V7 weights", flush=True)

    for i, w_v7 in enumerate(weights, start=1):
        w_v19 = 1.0 - float(w_v7)
        probs = float(w_v7) * A["probs"] + w_v19 * B["probs"]
        pred = probs >= args.threshold
        score = fast_macro_f1(y, pred)
        rec = {
            "v7_weight": float(w_v7),
            "v19_weight": float(w_v19),
            "macro_f1": float(score),
            "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
        }
        rows.append(rec)
        if best is None or score > best["macro_f1"]:
            best = rec
            best_probs = probs.astype(np.float32)

        if i == 1 or i % 5 == 0 or i == len(weights):
            print(
                f"[search] {i}/{len(weights)} current={score:.6f} "
                f"best={best['macro_f1']:.6f} w_v7={float(w_v7):.3f}",
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
    summary = {"version": "V20", "threshold": args.threshold, "best": best, "top10": rows[:10]}
    (out / "metrics" / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 80)
    print(f"BEST V7={best['v7_weight']:.3f} V19={best['v19_weight']:.3f}")
    print(f"Macro F1 @ {args.threshold:.2f}: {best['macro_f1']:.6f}")
    print(f"pred labels/sample      : {best['pred_labels_per_sample']:.3f}")
    print(f"saved                   : {score_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
