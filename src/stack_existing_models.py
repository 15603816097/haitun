"""V14: cross-fitted per-label logistic stacking of existing model scores.

Inputs:
- baseline balanced SGD (alpha=5e-6)
- tuned balanced SGD (alpha=2e-6)
- unweighted SGD (high AUC)
- tuned KNN (k=40, power=4)

For each label, a tiny logistic meta-model learns how to combine these four
probability signals. GroupKFold gives an honest cross-fitted estimate. The final
full-validation meta parameters are saved for later test prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def check_compatible(packs):
    base = packs[0]
    for p in packs[1:]:
        if not np.array_equal(base["row_idx"], p["row_idx"]):
            raise ValueError("row_idx mismatch")
        if not np.array_equal(base["labels"], p["labels"]):
            raise ValueError("label mismatch")
        if not np.array_equal(base["y_true"], p["y_true"]):
            raise ValueError("y_true mismatch")


def build_features(packs):
    return np.stack([p["probs"] for p in packs], axis=2).astype(np.float32)


def fit_meta_for_labels(X, y, C=1.0, class_weight="balanced"):
    n_labels = y.shape[1]
    n_features = X.shape[2]
    coef = np.zeros((n_labels, n_features), dtype=np.float32)
    intercept = np.zeros(n_labels, dtype=np.float32)

    for j in range(n_labels):
        yy = y[:, j]
        if yy.min() == yy.max():
            continue
        m = LogisticRegression(
            C=C,
            solver="liblinear",
            class_weight=class_weight,
            max_iter=200,
            random_state=42,
        )
        m.fit(X[:, j, :], yy)
        coef[j] = m.coef_[0].astype(np.float32)
        intercept[j] = np.float32(m.intercept_[0])

    return coef, intercept


def predict_meta(X, coef, intercept):
    z = np.einsum("nlf,lf->nl", X, coef, optimize=True) + intercept[None, :]
    z = np.clip(z, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def fit_thresholds_simple(y, probs, fallback=0.5, shrink_k=200.0):
    support = y.sum(axis=0).astype(np.float64)
    th = np.full(y.shape[1], fallback, dtype=np.float32)
    for j in range(y.shape[1]):
        if support[j] <= 0:
            continue
        raw = best_threshold_for_label(
            y[:, j],
            probs[:, j],
            min_threshold=0.02,
            max_threshold=0.98,
            fallback=fallback,
        )
        w = support[j] / (support[j] + shrink_k)
        th[j] = np.float32(w * raw + (1.0 - w) * fallback)
    return th


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="outputs/oof/v1_kmer_sgd_chronological_scores.npz")
    p.add_argument("--a2e6", default="outputs/oof/v10_sgd_a2e6_bal_chronological_scores.npz")
    p.add_argument("--none", default="outputs/oof/v10_sgd_a5e6_none_chronological_scores.npz")
    p.add_argument("--knn", default="outputs/oof/v7_kmer_knn_k40_p4p0_chronological_scores.npz")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--threshold-shrink-k", type=float, default=200.0)
    p.add_argument("--tag", default="v14_logistic_stack_chronological")
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    packs = [load_pack(args.base), load_pack(args.a2e6), load_pack(args.none), load_pack(args.knn)]
    check_compatible(packs)

    row_idx = packs[0]["row_idx"]
    labels = packs[0]["labels"]
    y = packs[0]["y_true"]
    X = build_features(packs)

    train = pd.read_csv(args.train, usecols=["sequence"])
    groups = train.iloc[row_idx]["sequence"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=args.folds)
    pred_cf = np.zeros_like(y, dtype=np.int8)
    prob_cf = np.zeros_like(y, dtype=np.float32)
    fold_rows = []

    for fold, (cal, ev) in enumerate(gkf.split(X, y, groups=groups), 1):
        coef, intercept = fit_meta_for_labels(X[cal], y[cal], C=args.C)
        p_ev = predict_meta(X[ev], coef, intercept)

        # Thresholds are fitted only on the calibration half.
        p_cal = predict_meta(X[cal], coef, intercept)
        th = fit_thresholds_simple(
            y[cal],
            p_cal,
            fallback=0.5,
            shrink_k=args.threshold_shrink_k,
        )
        pred = (p_ev >= th[None, :]).astype(np.int8)

        prob_cf[ev] = p_ev
        pred_cf[ev] = pred

        score = fast_macro_f1(y[ev], pred)
        fold_rows.append({
            "fold": fold,
            "macro_f1": float(score),
            "pred_labels_per_sample": float(pred.sum(axis=1).mean()),
        })
        print(
            f"fold {fold}: Macro F1={score:.6f}, "
            f"pred/sample={pred.sum(axis=1).mean():.3f}",
            flush=True,
        )

    cf_score = fast_macro_f1(y, pred_cf)

    coef_all, intercept_all = fit_meta_for_labels(X, y, C=args.C)
    probs_all = predict_meta(X, coef_all, intercept_all)
    th_all = fit_thresholds_simple(
        y,
        probs_all,
        fallback=0.5,
        shrink_k=args.threshold_shrink_k,
    )
    pred_all = (probs_all >= th_all[None, :]).astype(np.int8)
    fit_all = fast_macro_f1(y, pred_all)

    out = Path(args.output_root)
    (out / "oof").mkdir(parents=True, exist_ok=True)
    (out / "adaptive").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "oof" / f"{args.tag}_scores.npz",
        row_idx=row_idx,
        probs=prob_cf,
        y_true=y,
        labels=labels,
    )
    np.savez_compressed(
        out / "adaptive" / f"{args.tag}_params.npz",
        labels=labels,
        coef=coef_all,
        intercept=intercept_all,
        thresholds=th_all,
        source_names=np.asarray(["base_balanced","a2e6_balanced","a5e6_none","knn"]),
    )

    summary = {
        "version": "V14",
        "crossfit_macro_f1": float(cf_score),
        "fit_all_macro_f1": float(fit_all),
        "crossfit_pred_labels_per_sample": float(pred_cf.sum(axis=1).mean()),
        "true_labels_per_sample": float(y.sum(axis=1).mean()),
        "folds": fold_rows,
        "C": args.C,
        "threshold_shrink_k": args.threshold_shrink_k,
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
    print(f"saved params            : {out/'adaptive'/f'{args.tag}_params.npz'}")
    print("* optimistic fit-all")
    print("=" * 80)


if __name__ == "__main__":
    main()
