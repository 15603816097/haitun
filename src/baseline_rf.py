from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .metrics import macro_f1_skip_empty

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def extract_features(sequences: pd.Series) -> np.ndarray:
    n = len(sequences)
    feats = np.zeros((n, len(AMINO_ACIDS) + 1), dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    for i, seq in enumerate(sequences.astype(str)):
        length = len(seq)
        feats[i, -1] = np.log1p(length)
        if length == 0:
            continue
        for c in seq:
            j = aa_to_idx.get(c)
            if j is not None:
                feats[i, j] += 1.0
        feats[i, : len(AMINO_ACIDS)] /= length
    return feats


def label_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("label_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def make_model(seed: int, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )


def validate(
    train: pd.DataFrame,
    labels: list[str],
    X: np.ndarray,
    split_path: Path,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_pos: int,
) -> float:
    split = np.load(split_path)
    tr_idx = split["train_idx"]
    va_idx = split["val_idx"]
    y_true = train.iloc[va_idx][labels].to_numpy(dtype=np.int8)
    y_pred = np.zeros_like(y_true, dtype=np.int8)

    start = time.time()
    for j, col in enumerate(labels):
        y = train[col].to_numpy(dtype=np.int8)
        if y[tr_idx].sum() < min_pos:
            continue
        model = make_model(seed, n_estimators, max_depth)
        model.fit(X[tr_idx], y[tr_idx])
        y_pred[:, j] = model.predict(X[va_idx]).astype(np.int8)
        if (j + 1) % 50 == 0:
            print(f"[validate] {j + 1}/{len(labels)} elapsed={time.time() - start:.0f}s")

    score = macro_f1_skip_empty(y_true, y_pred)
    print(f"validation Macro F1: {score:.6f}")
    print(f"validation predicted positive rate: {y_pred.mean():.6f}")
    print(f"validation true positive rate: {y_true.mean():.6f}")
    return score


def submit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    output: Path,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_pos: int,
) -> None:
    pred = np.zeros((len(test), len(labels)), dtype=np.int8)
    start = time.time()

    for j, col in enumerate(labels):
        y = train[col].to_numpy(dtype=np.int8)
        if y.sum() < min_pos:
            continue
        model = make_model(seed, n_estimators, max_depth)
        model.fit(X_train, y)
        pred[:, j] = model.predict(X_test).astype(np.int8)
        if (j + 1) % 50 == 0:
            print(f"[submit] {j + 1}/{len(labels)} elapsed={time.time() - start:.0f}s")

    submission = pd.DataFrame(pred, columns=labels)
    submission.insert(0, "protein_id", test["protein_id"].astype(str).to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"saved submission: {output}")
    print(f"shape: {submission.shape}, predicted positive rate: {pred.mean():.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--split", default="outputs/splits/group_random_seed42.npz")
    p.add_argument("--output", default="outputs/submissions/v0_rf_submission.csv")
    p.add_argument("--mode", choices=["validate", "submit", "both"], default="validate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--min-pos", type=int, default=10)
    args = p.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    labels = label_columns(train)

    print(f"train={train.shape}, test={test.shape}, labels={len(labels)}")
    print("extracting official 21-dimensional composition features...")
    X_train = extract_features(train["sequence"])
    X_test = extract_features(test["sequence"])
    print(f"X_train={X_train.shape}, X_test={X_test.shape}")

    if args.mode in {"validate", "both"}:
        validate(
            train=train,
            labels=labels,
            X=X_train,
            split_path=Path(args.split),
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_pos=args.min_pos,
        )

    if args.mode in {"submit", "both"}:
        submit(
            train=train,
            test=test,
            labels=labels,
            X_train=X_train,
            X_test=X_test,
            output=Path(args.output),
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_pos=args.min_pos,
        )


if __name__ == "__main__":
    main()
