from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import apply_thresholds, macro_f1_skip_empty


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yb = y_true.astype(bool, copy=False)
    pb = y_pred.astype(bool, copy=False)
    support = yb.sum(axis=0)
    valid = support > 0
    if not np.any(valid):
        return 0.0

    tp = np.logical_and(yb, pb).sum(axis=0, dtype=np.int64)
    pred_pos = pb.sum(axis=0, dtype=np.int64)
    denom = support + pred_pos
    f1 = np.divide(
        2.0 * tp,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    return float(f1[valid].mean())


def find_best_global_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    t_min: float,
    t_max: float,
    step: float,
) -> tuple[float, float]:
    best_t = 0.5
    best_score = -1.0
    thresholds = np.arange(t_min, t_max + step * 0.5, step, dtype=np.float32)
    for t in thresholds:
        score = fast_macro_f1(y_true, probs >= t)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


def best_threshold_for_label(
    y: np.ndarray,
    p: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    fallback: float,
) -> float:
    positives = int(y.sum())
    if positives == 0:
        return fallback

    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order].astype(np.int64)
    p_sorted = p[order]

    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1, dtype=np.int64)
    fp = k - tp
    fn = positives - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        denom,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denom > 0,
    )

    valid = (p_sorted >= min_threshold) & (p_sorted <= max_threshold)
    if not np.any(valid):
        return fallback

    masked = np.where(valid, f1, -1.0)
    idx = int(np.argmax(masked))
    return float(np.clip(p_sorted[idx], min_threshold, max_threshold))


def fit_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    global_threshold: float,
    min_threshold: float,
    max_threshold: float,
    shrink_k: float,
    min_support_for_local: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_labels = y_true.shape[1]
    raw = np.full(n_labels, global_threshold, dtype=np.float32)
    shrunk = np.full(n_labels, global_threshold, dtype=np.float32)

    support = y_true.sum(axis=0).astype(np.float64)
    for j in range(n_labels):
        if support[j] < min_support_for_local:
            continue
        local = best_threshold_for_label(
            y_true[:, j],
            probs[:, j],
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            fallback=global_threshold,
        )
        raw[j] = local
        weight = support[j] / (support[j] + shrink_k)
        shrunk[j] = float(weight * local + (1.0 - weight) * global_threshold)

    return raw, shrunk


def threshold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: float | np.ndarray,
) -> dict:
    pred = apply_thresholds(probs, thresholds)
    return {
        "macro_f1": fast_macro_f1(y_true, pred),
        "true_positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(pred.mean()),
        "true_labels_per_sample": float(y_true.sum(axis=1).mean()),
        "predicted_labels_per_sample": float(pred.sum(axis=1).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="outputs/oof/v1_kmer_sgd_random_scores.npz")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--config", default="configs/v2_threshold.json")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--tag", default="v2_kmer")
    args = p.parse_args()

    pack = np.load(args.scores, allow_pickle=False)
    row_idx = pack["row_idx"].astype(np.int64)
    probs = pack["probs"].astype(np.float32)
    y_true = pack["y_true"].astype(np.int8)
    labels = pack["labels"].astype(str)

    train = pd.read_csv(args.train, usecols=["sequence"])
    groups = train.iloc[row_idx]["sequence"].astype(str).to_numpy()

    cfg = load_json(Path(args.config))
    gcfg = cfg["global_grid"]
    pcfg = cfg["per_label"]
    ccfg = cfg["crossfit"]

    fixed = threshold_metrics(y_true, probs, 0.5)
    print("=" * 80)
    print(f"V1 fixed 0.5 Macro F1: {fixed['macro_f1']:.6f}")
    print(
        "fixed predicted labels/sample: "
        f"{fixed['predicted_labels_per_sample']:.3f} "
        f"(true={fixed['true_labels_per_sample']:.3f})"
    )

    global_t, global_fit_score = find_best_global_threshold(
        y_true,
        probs,
        float(gcfg["min"]),
        float(gcfg["max"]),
        float(gcfg["step"]),
    )
    global_metrics = threshold_metrics(y_true, probs, global_t)
    print(f"best global threshold (fit-all): {global_t:.4f}")
    print(f"global Macro F1 (fit-all, optimistic): {global_fit_score:.6f}")

    n_splits = int(ccfg.get("n_splits", 2))
    gkf = GroupKFold(n_splits=n_splits)

    crossfit_pred = np.zeros_like(y_true, dtype=np.int8)
    fold_rows = []
    for fold, (cal_pos, eval_pos) in enumerate(gkf.split(probs, y_true, groups=groups), start=1):
        y_cal = y_true[cal_pos]
        p_cal = probs[cal_pos]

        fold_global_t, _ = find_best_global_threshold(
            y_cal,
            p_cal,
            float(gcfg["min"]),
            float(gcfg["max"]),
            float(gcfg["step"]),
        )
        _, fold_shrunk = fit_thresholds(
            y_cal,
            p_cal,
            global_threshold=fold_global_t,
            min_threshold=float(pcfg["min_threshold"]),
            max_threshold=float(pcfg["max_threshold"]),
            shrink_k=float(pcfg["shrink_k"]),
            min_support_for_local=int(pcfg["min_support_for_local"]),
        )
        fold_pred = apply_thresholds(probs[eval_pos], fold_shrunk)
        crossfit_pred[eval_pos] = fold_pred

        fold_score = fast_macro_f1(y_true[eval_pos], fold_pred)
        fold_rows.append({
            "fold": fold,
            "calibration_rows": int(len(cal_pos)),
            "evaluation_rows": int(len(eval_pos)),
            "global_threshold": float(fold_global_t),
            "macro_f1": float(fold_score),
            "predicted_positive_rate": float(fold_pred.mean()),
        })
        print(
            f"crossfit fold {fold}/{n_splits}: "
            f"global_t={fold_global_t:.4f} Macro F1={fold_score:.6f}"
        )

    crossfit_metrics = {
        "macro_f1": fast_macro_f1(y_true, crossfit_pred),
        "true_positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(crossfit_pred.mean()),
        "true_labels_per_sample": float(y_true.sum(axis=1).mean()),
        "predicted_labels_per_sample": float(crossfit_pred.sum(axis=1).mean()),
    }

    raw_all, shrunk_all = fit_thresholds(
        y_true,
        probs,
        global_threshold=global_t,
        min_threshold=float(pcfg["min_threshold"]),
        max_threshold=float(pcfg["max_threshold"]),
        shrink_k=float(pcfg["shrink_k"]),
        min_support_for_local=int(pcfg["min_support_for_local"]),
    )
    fit_all_metrics = threshold_metrics(y_true, probs, shrunk_all)

    metrics_dir = Path(args.output_root) / "metrics"
    model_dir = Path(args.output_root) / "thresholds"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    threshold_df = pd.DataFrame({
        "label": labels,
        "support": y_true.sum(axis=0).astype(int),
        "raw_threshold": raw_all,
        "shrunk_threshold": shrunk_all,
    })
    threshold_df.to_csv(model_dir / "{args.tag}_thresholds.csv", index=False)

    np.savez_compressed(
        model_dir / "{args.tag}_thresholds.npz",
        labels=labels,
        global_threshold=np.float32(global_t),
        raw_thresholds=raw_all.astype(np.float32),
        thresholds=shrunk_all.astype(np.float32),
    )

    summary = {
        "version": args.tag,
        "source_scores": args.scores,
        "fixed_0_5": fixed,
        "global_fit_all": {
            "threshold": global_t,
            **global_metrics,
        },
        "per_label_crossfit": crossfit_metrics,
        "per_label_fit_all_optimistic": fit_all_metrics,
        "folds": fold_rows,
        "config": cfg,
        "notes": [
            "Use per_label_crossfit as the honest estimate of threshold-tuning gain.",
            "per_label_fit_all_optimistic is expected to be higher because thresholds are fitted and evaluated on the same rows.",
            "{args.tag}_thresholds.npz contains thresholds fitted on the full random validation set for later test prediction.",
        ],
    }
    summary_path = metrics_dir / f"{args.tag}_threshold_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"Fixed 0.5 Macro F1             : {fixed['macro_f1']:.6f}")
    print(f"Global threshold Macro F1*     : {global_metrics['macro_f1']:.6f}")
    print(f"Per-label crossfit Macro F1    : {crossfit_metrics['macro_f1']:.6f}")
    print(f"Per-label fit-all Macro F1*    : {fit_all_metrics['macro_f1']:.6f}")
    print("* optimistic: fitted and evaluated on the same validation rows")
    print(
        "Crossfit predicted labels/sample: "
        f"{crossfit_metrics['predicted_labels_per_sample']:.3f} "
        f"(true={crossfit_metrics['true_labels_per_sample']:.3f})"
    )
    print(f"saved: {summary_path}")
    print(f"saved: {model_dir / '{args.tag}_thresholds.npz'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
