"""Train full-data V7 k-mer SGD + KNN and generate submission.

Uses:
- full train.csv for TF-IDF fit and SGD training
- test.csv for prediction
- V7 tuned KNN params: k=40, power=4
- V7 blend: SGD=0.40, KNN=0.60
- per-label thresholds learned on chronological validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .metrics import apply_thresholds
from .train_kmer import (
    config_fingerprint,
    label_columns,
    load_config,
    prepare_features,
    train_scores,
)
from .tune_kmer_knn_blend import compute_knn_probs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--output-root", default="outputs")
    p.add_argument(
        "--thresholds",
        default="outputs/thresholds/v7_tuned_sgd_knn_chronological_thresholds.npz",
    )
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--power", type=float, default=4.0)
    p.add_argument("--sgd-weight", type=float, default=0.40)
    p.add_argument("--knn-weight", type=float, default=0.60)
    p.add_argument("--knn-batch-size", type=int, default=256)
    args = p.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    labels = label_columns(train)
    y = train[labels].to_numpy(dtype=np.int8)
    fit_idx = np.arange(len(train), dtype=np.int64)

    cfg = load_config(Path(args.config))
    print("[1/5] building/loading full-data TF-IDF features")
    _, X_fit, X_test = prepare_features(
        train_sequences=train["sequence"],
        eval_sequences=test["sequence"],
        fit_idx=fit_idx,
        config=cfg,
        mode="submit",
        cache_dir=Path(args.cache_dir),
        use_cache=True,
    )
    if X_test is None:
        raise RuntimeError("test features missing")

    print("[2/5] training full-data SGD probabilities")
    sgd_probs, used_labels = train_scores(
        X_fit=X_fit,
        y_fit=y,
        X_eval=X_test,
        labels=labels,
        config=cfg,
        max_labels=None,
    )
    if used_labels != labels:
        raise RuntimeError("label mismatch after SGD training")

    print("[3/5] computing full-data KNN probabilities")
    nn = NearestNeighbors(
        n_neighbors=args.k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X_fit)
    dist, ind = nn.kneighbors(X_test, return_distance=True)
    sim = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    knn_probs = compute_knn_probs(
        y_fit=y.astype(np.float32),
        neighbor_idx=ind,
        similarities=sim,
        k=args.k,
        power=args.power,
        batch_size=args.knn_batch_size,
    )

    print("[4/5] blending probabilities and applying per-label thresholds")
    probs = (
        float(args.sgd_weight) * sgd_probs
        + float(args.knn_weight) * knn_probs
    ).astype(np.float32)

    th_pack = np.load(args.thresholds, allow_pickle=False)
    th_labels = th_pack["labels"].astype(str)
    thresholds = th_pack["thresholds"].astype(np.float32)
    if not np.array_equal(th_labels, np.asarray(labels)):
        raise ValueError("threshold label order does not match training labels")

    pred = apply_thresholds(probs, thresholds)

    out_root = Path(args.output_root)
    sub_dir = out_root / "submissions"
    oof_dir = out_root / "oof"
    sub_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        oof_dir / "v7_final_test_scores.npz",
        probs=probs,
        sgd_probs=sgd_probs.astype(np.float32),
        knn_probs=knn_probs.astype(np.float32),
        labels=np.asarray(labels),
        protein_id=test["protein_id"].astype(str).to_numpy(),
    )

    out = pd.DataFrame(pred.astype(np.int8), columns=labels)
    out.insert(0, "protein_id", test["protein_id"].astype(str).to_numpy())

    csv_path = sub_dir / "v7_sgd40_knn60_k40_p4_thresholds.csv"
    out.to_csv(csv_path, index=False)

    print("[5/5] done")
    print("=" * 80)
    print(f"submission              : {csv_path}")
    print(f"rows                    : {len(out)}")
    print(f"labels                  : {len(labels)}")
    print(f"SGD weight              : {args.sgd_weight:.2f}")
    print(f"KNN weight              : {args.knn_weight:.2f}")
    print(f"KNN k / power           : {args.k} / {args.power:g}")
    print(f"mean predicted labels   : {pred.sum(axis=1).mean():.3f}")
    print(f"predicted positive rate : {pred.mean():.6f}")
    print(f"mean top1 similarity    : {sim[:,0].mean():.4f}")
    print(f"saved probabilities     : {oof_dir / 'v7_final_test_scores.npz'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
