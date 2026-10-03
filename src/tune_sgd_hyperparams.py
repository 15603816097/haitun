"""V10: tune SGD hyperparameters on the existing chronological TF-IDF cache.

This does NOT rebuild TF-IDF. It compares several SGDClassifier variants on the
same cached X_fit/X_eval, reports Macro F1 @0.5 and macro ROC-AUC, and saves the
probability matrix for each candidate so threshold/blend stages can reuse it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from .metrics import macro_f1_skip_empty
from .train_kmer import config_fingerprint, label_columns, load_config


CANDIDATES = [
    {"name": "a2e6_bal", "alpha": 2e-6, "class_weight": "balanced", "average": False},
    {"name": "a5e6_bal", "alpha": 5e-6, "class_weight": "balanced", "average": False},
    {"name": "a1e5_bal", "alpha": 1e-5, "class_weight": "balanced", "average": False},
    {"name": "a5e6_bal_avg", "alpha": 5e-6, "class_weight": "balanced", "average": True},
    {"name": "a5e6_none", "alpha": 5e-6, "class_weight": None, "average": False},
]


def macro_auc_skip_invalid(y_true: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        y = y_true[:, j]
        if y.min() == y.max():
            continue
        vals.append(roc_auc_score(y, probs[:, j]))
    return float(np.mean(vals)) if vals else 0.0


def train_candidate(
    X_fit,
    y_fit,
    X_eval,
    alpha,
    class_weight,
    average,
    max_iter,
    tol,
    seed,
):
    n_labels = y_fit.shape[1]
    probs = np.zeros((X_eval.shape[0], n_labels), dtype=np.float32)
    t0 = time.time()
    for j in range(n_labels):
        m = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(alpha),
            class_weight=class_weight,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=int(seed),
            average=bool(average),
        )
        m.fit(X_fit, y_fit[:, j])
        probs[:, j] = m.predict_proba(X_eval)[:, 1].astype(np.float32)
        if (j + 1) % 50 == 0 or j + 1 == n_labels:
            elapsed = time.time() - t0
            print(f"  labels {j+1}/{n_labels} elapsed={elapsed:.1f}s")
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["chronological"], default="chronological")
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--split", default="outputs/splits/chronological_group_holdout.npz")
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    cache = Path(args.cache_dir) / f"v1_{args.mode}_{config_fingerprint(cfg,args.mode)}"
    X_fit = sparse.load_npz(cache / "X_fit.npz").tocsr()
    X_eval = sparse.load_npz(cache / "X_eval.npz").tocsr()

    df = pd.read_csv(args.train)
    labels = label_columns(df)
    y_all = df[labels].to_numpy(dtype=np.int8)
    sp = np.load(args.split)
    fit_idx = sp["train_idx"].astype(np.int64)
    eval_idx = sp["val_idx"].astype(np.int64)
    y_fit = y_all[fit_idx]
    y_true = y_all[eval_idx]

    out = Path(args.output_root)
    (out / "oof").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    base_model = cfg["model"]
    rows = []

    for cand in CANDIDATES:
        print("=" * 80)
        print(f"[candidate] {cand['name']} alpha={cand['alpha']} "
              f"class_weight={cand['class_weight']} average={cand['average']}")
        probs = train_candidate(
            X_fit=X_fit,
            y_fit=y_fit,
            X_eval=X_eval,
            alpha=cand["alpha"],
            class_weight=cand["class_weight"],
            average=cand["average"],
            max_iter=base_model.get("max_iter", 30),
            tol=base_model.get("tol", 1e-3),
            seed=base_model.get("random_state", 42),
        )
        pred = (probs >= 0.5).astype(np.int8)
        f1 = macro_f1_skip_empty(y_true, pred)
        auc = macro_auc_skip_invalid(y_true, probs)
        mean_labels = float(pred.sum(axis=1).mean())

        score_path = out / "oof" / f"v10_sgd_{cand['name']}_chronological_scores.npz"
        np.savez_compressed(
            score_path,
            row_idx=eval_idx,
            probs=probs,
            y_true=y_true,
            labels=np.asarray(labels),
        )

        row = {
            **cand,
            "macro_f1_at_0_5": float(f1),
            "macro_auc": float(auc),
            "pred_labels_per_sample": mean_labels,
            "score_path": str(score_path),
        }
        rows.append(row)
        print(f"Macro F1 @0.5 : {f1:.6f}")
        print(f"Macro ROC-AUC  : {auc:.6f}")
        print(f"pred/sample    : {mean_labels:.3f}")

    rows = sorted(rows, key=lambda x: (x["macro_auc"], x["macro_f1_at_0_5"]), reverse=True)
    summary = {"version": "V10", "results": rows}
    path = out / "metrics" / "v10_sgd_tuning_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("V10 ranking by Macro ROC-AUC")
    for r in rows:
        print(
            f"{r['name']}: AUC={r['macro_auc']:.6f} "
            f"F1@0.5={r['macro_f1_at_0_5']:.6f} "
            f"pred/sample={r['pred_labels_per_sample']:.3f}"
        )
    print(f"saved: {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
