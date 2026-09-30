from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ID_RE = re.compile(r"^[A-Za-z]+(\d+)$")


def _label_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("label_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def _numeric_ids(ids: pd.Series) -> np.ndarray:
    out = []
    for x in ids.astype(str):
        m = ID_RE.match(x)
        out.append(int(m.group(1)) if m else -1)
    return np.asarray(out, dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--output", default="outputs/splits")
    p.add_argument("--val-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    labels = _label_columns(train)
    y = train[labels].to_numpy(dtype=np.int8)
    groups = train["sequence"].astype(str).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.val_size,
        random_state=args.seed,
    )
    tr_idx, va_idx = next(splitter.split(train, y, groups=groups))

    train_groups = set(groups[tr_idx])
    val_groups = set(groups[va_idx])
    overlap = len(train_groups & val_groups)
    if overlap:
        raise RuntimeError(f"group leakage detected: {overlap} sequences overlap")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    split_path = output / f"group_random_seed{args.seed}.npz"
    np.savez_compressed(split_path, train_idx=tr_idx, val_idx=va_idx)

    prevalence_all = y.mean(axis=0)
    prevalence_tr = y[tr_idx].mean(axis=0)
    prevalence_va = y[va_idx].mean(axis=0)

    train_num = _numeric_ids(train["protein_id"])
    test_num = _numeric_ids(test["protein_id"])
    tail_idx = np.array([], dtype=np.int64)
    head_idx = np.arange(len(train), dtype=np.int64)
    test_min = None

    if np.all(train_num >= 0) and np.all(test_num >= 0):
        test_min = int(test_num.min())
        tail_idx = np.flatnonzero(train_num >= test_min)
        head_idx = np.flatnonzero(train_num < test_min)
        np.savez_compressed(output / "test_like_tail.npz", head_idx=head_idx, tail_idx=tail_idx)

    meta = {
        "seed": args.seed,
        "val_size_requested": args.val_size,
        "train_rows": int(len(tr_idx)),
        "val_rows": int(len(va_idx)),
        "group_overlap": overlap,
        "mean_abs_label_prevalence_drift_train_vs_all": float(np.mean(np.abs(prevalence_tr - prevalence_all))),
        "mean_abs_label_prevalence_drift_val_vs_all": float(np.mean(np.abs(prevalence_va - prevalence_all))),
        "test_min_numeric_id": test_min,
        "test_like_tail_rows": int(len(tail_idx)),
        "head_rows": int(len(head_idx)),
        "notes": [
            "Identical sequences are kept in the same random split to prevent exact-sequence leakage.",
            "The test-like tail split is diagnostic only; protein_id is not used as a predictive feature.",
            "V1 will add stronger multilabel-aware split stability checks.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"saved: {split_path}")
    if test_min is not None:
        print(f"saved: {output / 'test_like_tail.npz'}")


if __name__ == "__main__":
    main()
