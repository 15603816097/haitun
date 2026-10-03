"""Generate V9 label-specific KNN/SGD submission without retraining SGD.

Reuses:
- full-data submit TF-IDF cache produced by make_v7_submission.py
- outputs/oof/v7_final_test_scores.npz for full-data SGD test probabilities
- outputs/adaptive/v9_label_specific_knn_params.npz for per-label
  k / power / SGD weight / threshold selected on chronological validation

Only the test KNN neighbor search is recomputed (once up to max selected k).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .train_kmer import config_fingerprint, label_columns, load_config
from .tune_kmer_knn_blend import compute_knn_probs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--v7-test-scores", default="outputs/oof/v7_final_test_scores.npz")
    p.add_argument("--params", default="outputs/adaptive/v9_label_specific_knn_params.npz")
    p.add_argument("--output", default="outputs/submissions/v9_label_specific_knn.csv")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    labels = label_columns(train)
    y = train[labels].to_numpy(dtype=np.float32)

    # Reuse full-data submit TF-IDF cache created by V7.
    cfg = load_config(Path(args.config))
    cache = Path(args.cache_dir) / f"v1_submit_{config_fingerprint(cfg, 'submit')}"
    x_fit_path = cache / "X_fit.npz"
    x_test_path = cache / "X_eval.npz"
    if not x_fit_path.exists() or not x_test_path.exists():
        raise FileNotFoundError(
            f"missing submit TF-IDF cache: {cache}. "
            "Run python -m src.make_v7_submission first."
        )

    print(f"[cache] {cache}", flush=True)
    X_fit = sparse.load_npz(x_fit_path).tocsr()
    X_test = sparse.load_npz(x_test_path).tocsr()

    score_pack = np.load(args.v7_test_scores, allow_pickle=False)
    score_labels = score_pack["labels"].astype(str)
    sgd_probs = score_pack["sgd_probs"].astype(np.float32)

    params = np.load(args.params, allow_pickle=False)
    param_labels = params["labels"].astype(str)
    ks = params["k"].astype(np.int32)
    powers = params["power"].astype(np.float32)
    sgd_weights = params["sgd_weight"].astype(np.float32)
    thresholds = params["thresholds"].astype(np.float32)

    expected = np.asarray(labels)
    if not np.array_equal(score_labels, expected):
        raise ValueError("V7 test score label order mismatch")
    if not np.array_equal(param_labels, expected):
        raise ValueError("V9 parameter label order mismatch")
    if sgd_probs.shape != (len(test), len(labels)):
        raise ValueError(f"unexpected SGD test score shape: {sgd_probs.shape}")

    max_k = int(ks.max())
    print(
        f"[neighbors] searching max_k={max_k} "
        f"fit={X_fit.shape} test={X_test.shape}",
        flush=True,
    )
    nn = NearestNeighbors(
        n_neighbors=max_k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X_fit)
    dist, neighbor_idx = nn.kneighbors(X_test, return_distance=True)
    similarities = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    print(
        f"[neighbors] mean top1={similarities[:,0].mean():.4f} "
        f"mean top{max_k}={similarities.mean():.4f}",
        flush=True,
    )

    unique_pairs = sorted({(int(k), float(p)) for k, p in zip(ks, powers)})
    print(f"[knn] unique (k,power) pairs: {len(unique_pairs)}", flush=True)

    pair_probs: dict[tuple[int, float], np.ndarray] = {}
    for i, (k, power) in enumerate(unique_pairs, 1):
        probs = compute_knn_probs(
            y_fit=y,
            neighbor_idx=neighbor_idx,
            similarities=similarities,
            k=k,
            power=power,
            batch_size=args.batch_size,
        )
        pair_probs[(k, power)] = probs
        print(f"[knn] {i}/{len(unique_pairs)} built k={k} power={power:g}", flush=True)

    n_rows = len(test)
    n_labels = len(labels)
    final_probs = np.empty((n_rows, n_labels), dtype=np.float32)
    pred = np.zeros((n_rows, n_labels), dtype=np.int8)

    for j in range(n_labels):
        key = (int(ks[j]), float(powers[j]))
        knn_col = pair_probs[key][:, j]
        p = sgd_weights[j] * sgd_probs[:, j] + (1.0 - sgd_weights[j]) * knn_col
        final_probs[:, j] = p
        pred[:, j] = (p >= thresholds[j]).astype(np.int8)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(pred, columns=labels)
    out.insert(0, "protein_id", test["protein_id"].astype(str).to_numpy())
    out.to_csv(out_path, index=False)

    prob_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        prob_path,
        probs=final_probs,
        labels=np.asarray(labels),
        protein_id=test["protein_id"].astype(str).to_numpy(),
        k=ks,
        power=powers,
        sgd_weight=sgd_weights,
        thresholds=thresholds,
    )

    uk, ck = np.unique(ks, return_counts=True)
    up, cp = np.unique(powers, return_counts=True)

    print("=" * 80)
    print(f"submission              : {out_path}")
    print(f"rows                    : {len(out)}")
    print(f"labels                  : {len(labels)}")
    print(f"mean predicted labels   : {pred.sum(axis=1).mean():.3f}")
    print(f"predicted positive rate : {pred.mean():.6f}")
    print(f"mean SGD weight         : {sgd_weights.mean():.3f}")
    print(f"median SGD weight       : {np.median(sgd_weights):.3f}")
    print(f"k distribution          : {dict(zip(uk.tolist(), ck.tolist()))}")
    print(f"power distribution      : {dict(zip(up.tolist(), cp.tolist()))}")
    print(f"saved probabilities     : {prob_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
