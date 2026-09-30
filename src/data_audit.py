from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
ID_RE = re.compile(r"^[A-Za-z]+(\d+)$")


def _label_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("label_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def _series_stats(s: pd.Series) -> dict[str, float | int]:
    d = s.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def _nonstandard_counts(sequences: pd.Series) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for seq in sequences.astype(str):
        for ch in seq:
            if ch not in STANDARD_AA:
                counter[ch] += 1
    return dict(sorted(counter.items()))


def _numeric_ids(ids: pd.Series) -> np.ndarray:
    values = []
    for x in ids.astype(str):
        m = ID_RE.match(x)
        values.append(int(m.group(1)) if m else -1)
    return np.asarray(values, dtype=np.int64)


def audit(train_path: Path, test_path: Path) -> dict:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    labels = _label_columns(train)

    if len(labels) != 500:
        print(f"[warning] expected 500 labels, found {len(labels)}")

    train_len = train["sequence"].astype(str).str.len()
    test_len = test["sequence"].astype(str).str.len()
    y = train[labels].to_numpy(dtype=np.int8)

    train_ids_num = _numeric_ids(train["protein_id"])
    test_ids_num = _numeric_ids(test["protein_id"])

    tail = None
    if np.all(train_ids_num >= 0) and np.all(test_ids_num >= 0):
        test_min = int(test_ids_num.min())
        mask = train_ids_num >= test_min
        if mask.any():
            tail_y = y[mask]
            tail = {
                "boundary_test_min_numeric_id": test_min,
                "tail_train_samples": int(mask.sum()),
                "head_train_samples": int((~mask).sum()),
                "tail_sequence_length_mean": float(train_len[mask].mean()),
                "tail_sequence_length_median": float(train_len[mask].median()),
                "tail_labels_per_sample_mean": float(tail_y.sum(axis=1).mean()),
                "tail_label_density": float(tail_y.mean()),
            }

    train_seq = train["sequence"].astype(str)
    test_seq = test["sequence"].astype(str)
    train_seq_set = set(train_seq)
    test_seq_set = set(test_seq)

    duplicate_train_mask = train_seq.duplicated(keep=False)
    duplicate_test_mask = test_seq.duplicated(keep=False)

    result = {
        "train": {
            "shape": [int(train.shape[0]), int(train.shape[1])],
            "duplicate_protein_id": int(train["protein_id"].duplicated().sum()),
            "duplicate_sequence_rows": int(duplicate_train_mask.sum()),
            "duplicate_sequence_groups": int(train.loc[duplicate_train_mask, "sequence"].nunique()),
            "missing_cells": int(train.isna().sum().sum()),
            "sequence_length": _series_stats(train_len),
            "nonstandard_amino_acids": _nonstandard_counts(train_seq),
        },
        "test": {
            "shape": [int(test.shape[0]), int(test.shape[1])],
            "duplicate_protein_id": int(test["protein_id"].duplicated().sum()),
            "duplicate_sequence_rows": int(duplicate_test_mask.sum()),
            "duplicate_sequence_groups": int(test.loc[duplicate_test_mask, "sequence"].nunique()),
            "missing_cells": int(test.isna().sum().sum()),
            "sequence_length": _series_stats(test_len),
            "nonstandard_amino_acids": _nonstandard_counts(test_seq),
        },
        "labels": {
            "count": len(labels),
            "density": float(y.mean()),
            "positive_count_min": int(y.sum(axis=0).min()),
            "positive_count_median": float(np.median(y.sum(axis=0))),
            "positive_count_mean": float(y.sum(axis=0).mean()),
            "positive_count_max": int(y.sum(axis=0).max()),
            "labels_per_sample": _series_stats(pd.Series(y.sum(axis=1))),
        },
        "overlap": {
            "protein_id": int(len(set(train["protein_id"]) & set(test["protein_id"]))),
            "exact_sequence": int(len(train_seq_set & test_seq_set)),
        },
        "id_ranges": {
            "train_numeric_min": int(train_ids_num[train_ids_num >= 0].min()) if np.any(train_ids_num >= 0) else None,
            "train_numeric_max": int(train_ids_num.max()) if np.any(train_ids_num >= 0) else None,
            "test_numeric_min": int(test_ids_num[test_ids_num >= 0].min()) if np.any(test_ids_num >= 0) else None,
            "test_numeric_max": int(test_ids_num.max()) if np.any(test_ids_num >= 0) else None,
        },
        "test_like_tail": tail,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--output", default="outputs/metrics/data_audit.json")
    args = p.parse_args()

    result = audit(Path(args.train), Path(args.test))
    print(json.dumps(result, ensure_ascii=False, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
