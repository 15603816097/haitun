"""V18: cross-fitted per-label model selection among V7, V16 and a2e6 SGD.

For each label, choose the candidate model and threshold using only the calibration
half of a GroupKFold split, then evaluate on the held-out half. To reduce noise,
an alternative model must beat V7 by a configurable F1 margin on calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .optimize_thresholds import best_threshold_for_label, fast_macro_f1


def load_pack(path: str):
    p = np.load(path, allow_pickle=False)
    return {
        "row_idx": p["row_idx"].astype(np.int64),
        "probs": p["probs"].astype(np.float32),
        "y_true": p["y_true"].astype(np.int8),
        "labels": p["labels"].astype(str),
    }


def binary_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = np.logical_and(y == 1, pred).sum()
    fp = np.logical_and(y == 0, pred).sum()
    fn = np.logical_and(y == 1, ~pred).sum()
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else float(2 * tp / d)


def fit_selector(
    y: np.ndarray,
    probs_list: list[np.ndarray],
    fallback_threshold: float,
    switch_margin: float,
):
    n_labels = y.shape[1]
    chosen = np.zeros(n_labels, dtype=np.int8)
    thresholds = np.full(n_labels, fallback_threshold, dtype=np.float32)
    cal_scores = np.zeros((len(probs_list), n_labels), dtype=np.float32)

    for j in range(n_labels):
        if y[:, j].sum() == 0:
            continue

        best_model = 0
        best_t = fallback_threshold
        best_score = -1.0

        for m, probs in enumerate(probs_list):
            t = best_threshold_for_label(
                y[:, j],
                probs[:, j],
                min_threshold=0.02,
                max_threshold=0.98,
                fallback=fallback_threshold,
            )
            pred = probs[:, j] >= t
            score = binary_f1(y[:, j], pred)
            cal_scores[m, j] = score

            if score > best_score:
                best_score = score
                best_model = m
                best_t = t

        # Conservative switch: stay with V7 unless alternative clearly wins.
        v7_t = best_threshold_for_label(
            y[:, j],
            probs_list[0][:, j],
            min_threshold=0.02,
            max_threshold=0.98,
            fallback=fallback_threshold,
        )
        v7_score = binary_f1(y[:, j], probs_list[0][:, j] >= v7_t)

        if best_model != 0 and best_score < v7_score + switch_margin:
            best_model = 0
            best_t = v7_t

        chosen[j] = best_model
        thresholds[j] = np.float32(best_t)

    return chosen, thresholds, cal_scores


def apply_selector(probs_list, chosen, thresholds):
    n_rows = probs_list[0].shape[0]
    n_labels = probs_list[0].shape[1]
    pred = np.zeros((n_rows, n_labels), dtype=np.int8)

    for j in range(n_labels):
        p = probs_list[int(chosen[j])][:, j]
        pred[:, j] = (p >= thresholds[j]).astype(np.int8)
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v7", default="outputs/oof/v7_tuned_sgd_knn_chronological_scores.npz")
    p.add_argument("--v16", default="outputs/v16_kmer24/oof/v1_kmer_sgd_chronological_scores.npz")
    p.add_argument("--a2e6", default="outputs/oof/v10_sgd_a2e6_bal_chronological_scores.npz")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--fallback-threshold", type=float, default=0.48)
    p.add_argument("--switch-margin", type=float, default=0.01)
    p.add_argument("--tag", default="v18_label_selector_chronological")
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    packs = [load_pack(args.v7), load_pack(args.v16), load_pack(args.a2e6)]
    base = packs[0]
    for pck in packs[1:]:
        if not np.array_equal(base["row_idx"], pck["row_idx"]):
            raise ValueError("row_idx mismatch")
        if not np.array_equal(base["labels"], pck["labels"]):
            raise ValueError("label mismatch")
        if not np.array_equal(base["y_true"], pck["y_true"]):
            raise ValueError("y_true mismatch")

    y = base["y_true"]
    probs_all = [pck["probs"] for pck in packs]
    row_idx = base["row_idx"]
    labels = base["labels"]

    train = pd.read_csv(args.train, usecols=["sequence"])
    groups = train.iloc[row_idx]["sequence"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=args.folds)
    pred_cf = np.zeros_like(y, dtype=np.int8)
    fold_rows = []

    for fold, (cal, ev) in enumerate(gkf.split(y, y, groups=groups), 1):
        cal_probs = [p[cal] for p in probs_all]
        ev_probs = [p[ev] for p in probs_all]

        chosen, th, _ = fit_selector(
            y[cal],
            cal_probs,
            fallback_threshold=args.fallback_threshold,
            switch_margin=args.switch_margin,
        )
        pred = apply_selector(ev_probs, chosen, th)
        pred_cf[ev] = pred

        score = fast_macro_f1(y[ev], pred)
        counts = np.bincount(chosen, minlength=3)
        fold_rows.append({
            "fold": fold,
            "macro_f1": float(score),
            "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
            "selected_v7": int(counts[0]),
            "selected_v16": int(counts[1]),
            "selected_a2e6": int(counts[2]),
        })
        print(
            f"fold {fold}: F1={score:.6f} pred/sample={pred.sum(axis=1).mean():.3f} "
            f"selected V7/V16/a2e6={counts.tolist()}",
            flush=True,
        )

    cf_score = fast_macro_f1(y, pred_cf)

    chosen_all, th_all, cal_scores = fit_selector(
        y,
        probs_all,
        fallback_threshold=args.fallback_threshold,
        switch_margin=args.switch_margin,
    )
    pred_fit = apply_selector(probs_all, chosen_all, th_all)
    fit_all = fast_macro_f1(y, pred_fit)
    counts_all = np.bincount(chosen_all, minlength=3)

    out = Path(args.output_root)
    (out / "adaptive").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "adaptive" / f"{args.tag}_params.npz",
        labels=labels,
        chosen_model=chosen_all,
        thresholds=th_all,
        calibration_scores=cal_scores,
        source_names=np.asarray(["v7","v16_kmer24","a2e6_bal"]),
    )

    summary = {
        "version": "V18",
        "crossfit_macro_f1": float(cf_score),
        "fit_all_macro_f1": float(fit_all),
        "crossfit_pred_labels_per_sample": float(pred_cf.sum(axis=1).mean()),
        "true_labels_per_sample": float(y.sum(axis=1).mean()),
        "selected_all": {
            "v7": int(counts_all[0]),
            "v16": int(counts_all[1]),
            "a2e6": int(counts_all[2]),
        },
        "switch_margin": args.switch_margin,
        "folds": fold_rows,
    }
    path = out / "metrics" / f"{args.tag}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"Crossfit Macro F1       : {cf_score:.6f}")
    print(f"Fit-all Macro F1*       : {fit_all:.6f}")
    print(
        f"pred labels/sample      : {pred_cf.sum(axis=1).mean():.3f} "
        f"(true={y.sum(axis=1).mean():.3f})"
    )
    print(f"selected V7/V16/a2e6   : {counts_all.tolist()}")
    print("* optimistic fit-all")
    print("=" * 80)


if __name__ == "__main__":
    main()
