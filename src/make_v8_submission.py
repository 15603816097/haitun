"""Generate the V8 label-adaptive submission from already-computed V7 test scores.

No retraining is required. Uses:
- outputs/oof/v7_final_test_scores.npz
- outputs/adaptive/v8_label_adaptive_chronological.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import apply_thresholds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="test.csv")
    p.add_argument("--scores", default="outputs/oof/v7_final_test_scores.npz")
    p.add_argument("--params", default="outputs/adaptive/v8_label_adaptive_chronological.npz")
    p.add_argument("--output", default="outputs/submissions/v8_label_adaptive.csv")
    args = p.parse_args()

    scores = np.load(args.scores, allow_pickle=False)
    params = np.load(args.params, allow_pickle=False)

    labels = scores["labels"].astype(str)
    sgd = scores["sgd_probs"].astype(np.float32)
    knn = scores["knn_probs"].astype(np.float32)

    p_labels = params["labels"].astype(str)
    weights = params["weights"].astype(np.float32)
    thresholds = params["thresholds"].astype(np.float32)

    if not np.array_equal(labels, p_labels):
        raise ValueError("label order mismatch")
    if sgd.shape != knn.shape:
        raise ValueError("score shape mismatch")
    if sgd.shape[1] != len(weights):
        raise ValueError("parameter size mismatch")

    probs = (
        sgd * weights[None, :]
        + knn * (1.0 - weights[None, :])
    ).astype(np.float32)
    pred = apply_thresholds(probs, thresholds)

    test = pd.read_csv(args.test, usecols=["protein_id"])
    if len(test) != len(pred):
        raise ValueError("test row count mismatch")

    out = pd.DataFrame(pred.astype(np.int8), columns=labels)
    out.insert(0, "protein_id", test["protein_id"].astype(str).to_numpy())

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    print("=" * 80)
    print(f"submission              : {path}")
    print(f"rows                    : {len(out)}")
    print(f"labels                  : {len(labels)}")
    print(f"mean SGD weight         : {weights.mean():.3f}")
    print(f"median SGD weight       : {np.median(weights):.3f}")
    print(f"mean predicted labels   : {pred.sum(axis=1).mean():.3f}")
    print(f"predicted positive rate : {pred.mean():.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
