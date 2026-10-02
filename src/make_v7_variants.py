"""Create threshold-scaled V7 submission variants without retraining."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import apply_thresholds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="outputs/oof/v7_final_test_scores.npz")
    p.add_argument(
        "--thresholds",
        default="outputs/thresholds/v7_tuned_sgd_knn_chronological_thresholds.npz",
    )
    p.add_argument("--test", default="test.csv")
    p.add_argument("--scales", default="1.00,1.03,1.06")
    p.add_argument("--output-dir", default="outputs/submissions")
    args = p.parse_args()

    score_pack = np.load(args.scores, allow_pickle=False)
    probs = score_pack["probs"].astype(np.float32)
    labels = score_pack["labels"].astype(str)

    th_pack = np.load(args.thresholds, allow_pickle=False)
    th_labels = th_pack["labels"].astype(str)
    base_th = th_pack["thresholds"].astype(np.float32)

    if not np.array_equal(labels, th_labels):
        raise ValueError("label order mismatch between scores and thresholds")

    test = pd.read_csv(args.test, usecols=["protein_id"])
    if len(test) != len(probs):
        raise ValueError("test rows do not match score rows")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scale_text in args.scales.split(","):
        scale = float(scale_text.strip())
        thresholds = np.clip(base_th * scale, 0.001, 0.999)
        pred = apply_thresholds(probs, thresholds)

        out = pd.DataFrame(pred.astype(np.int8), columns=labels)
        out.insert(0, "protein_id", test["protein_id"].astype(str).to_numpy())

        suffix = str(scale).replace(".", "p")
        path = out_dir / f"v7_threshold_scale_{suffix}.csv"
        out.to_csv(path, index=False)

        print(
            f"scale={scale:.3f} "
            f"mean_labels={pred.sum(axis=1).mean():.3f} "
            f"positive_rate={pred.mean():.6f} "
            f"saved={path}"
        )


if __name__ == "__main__":
    main()
