from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from .metrics import apply_thresholds, macro_f1_skip_empty


def label_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("label_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def config_fingerprint(config: dict, mode: str) -> str:
    payload = json.dumps({"config": config, "mode": mode}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def build_vectorizer(config: dict) -> TfidfVectorizer:
    f = config["features"]
    dtype = np.float32 if f.get("dtype", "float32") == "float32" else np.float64
    return TfidfVectorizer(
        analyzer=f.get("analyzer", "char"),
        ngram_range=(int(f.get("ngram_min", 3)), int(f.get("ngram_max", 5))),
        min_df=int(f.get("min_df", 5)),
        max_features=int(f.get("max_features", 120000)),
        sublinear_tf=bool(f.get("sublinear_tf", True)),
        lowercase=bool(f.get("lowercase", False)),
        dtype=dtype,
    )


def build_model(config: dict) -> SGDClassifier:
    m = config["model"]
    return SGDClassifier(
        loss=m.get("loss", "log_loss"),
        penalty=m.get("penalty", "l2"),
        alpha=float(m.get("alpha", 5e-6)),
        class_weight=m.get("class_weight", "balanced"),
        max_iter=int(m.get("max_iter", 30)),
        tol=float(m.get("tol", 1e-3)),
        random_state=int(m.get("random_state", 42)),
        average=bool(m.get("average", False)),
    )


def resolve_indices(
    mode: str,
    n_train: int,
    random_split_path: Path,
    tail_split_path: Path,
    chronological_split_path: Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "random":
        split = np.load(random_split_path)
        return split["train_idx"], split["val_idx"]
    if mode in {"tail", "tail_clean"}:
        split = np.load(tail_split_path)
        return split["head_idx"], split["tail_idx"]
    if mode == "chronological":
        split = np.load(chronological_split_path)
        return split["train_idx"], split["val_idx"]
    if mode == "submit":
        return np.arange(n_train, dtype=np.int64), None
    raise ValueError(f"unsupported mode: {mode}")


def prepare_features(
    train_sequences: pd.Series,
    eval_sequences: pd.Series | None,
    fit_idx: np.ndarray,
    config: dict,
    mode: str,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[TfidfVectorizer, sparse.csr_matrix, sparse.csr_matrix | None]:
    fp = config_fingerprint(config, mode)
    cache = cache_dir / f"v1_{mode}_{fp}"
    vectorizer_path = cache / "vectorizer.joblib"
    x_fit_path = cache / "X_fit.npz"
    x_eval_path = cache / "X_eval.npz"

    if use_cache and vectorizer_path.exists() and x_fit_path.exists():
        print(f"[features] loading cache: {cache}")
        vectorizer = joblib.load(vectorizer_path)
        X_fit = sparse.load_npz(x_fit_path).tocsr()
        X_eval = sparse.load_npz(x_eval_path).tocsr() if x_eval_path.exists() else None
        return vectorizer, X_fit, X_eval

    cache.mkdir(parents=True, exist_ok=True)
    vectorizer = build_vectorizer(config)

    fit_sequences = train_sequences.iloc[fit_idx].astype(str)
    t0 = time.time()
    print(
        "[features] fitting TF-IDF "
        f"rows={len(fit_sequences)} ngram={vectorizer.ngram_range} "
        f"max_features={vectorizer.max_features} min_df={vectorizer.min_df}"
    )
    X_fit = vectorizer.fit_transform(fit_sequences).tocsr()
    print(
        f"[features] fit done in {time.time()-t0:.1f}s; "
        f"shape={X_fit.shape}; nnz={X_fit.nnz:,}; "
        f"dtype={X_fit.dtype}"
    )

    X_eval = None
    if eval_sequences is not None:
        t1 = time.time()
        X_eval = vectorizer.transform(eval_sequences.astype(str)).tocsr()
        print(
            f"[features] eval transform done in {time.time()-t1:.1f}s; "
            f"shape={X_eval.shape}; nnz={X_eval.nnz:,}"
        )

    if use_cache:
        print(f"[features] saving cache: {cache}")
        joblib.dump(vectorizer, vectorizer_path, compress=3)
        sparse.save_npz(x_fit_path, X_fit, compressed=True)
        if X_eval is not None:
            sparse.save_npz(x_eval_path, X_eval, compressed=True)

    return vectorizer, X_fit, X_eval


def train_scores(
    X_fit: sparse.csr_matrix,
    y_fit: np.ndarray,
    X_eval: sparse.csr_matrix,
    labels: list[str],
    config: dict,
    max_labels: int | None,
) -> tuple[np.ndarray, list[str]]:
    n_labels = len(labels) if max_labels is None else min(len(labels), max_labels)
    used_labels = labels[:n_labels]
    probs = np.zeros((X_eval.shape[0], n_labels), dtype=np.float32)

    t0 = time.time()
    for j in range(n_labels):
        model = build_model(config)
        model.fit(X_fit, y_fit[:, j])
        probs[:, j] = model.predict_proba(X_eval)[:, 1].astype(np.float32)

        if (j + 1) % 25 == 0 or j + 1 == n_labels:
            elapsed = time.time() - t0
            rate = elapsed / (j + 1)
            eta = rate * (n_labels - j - 1)
            print(
                f"[model] {j+1}/{n_labels} "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s"
            )
    return probs, used_labels


def evaluate_and_save(
    mode: str,
    row_idx: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    threshold: float,
    output_root: Path,
    config: dict,
) -> None:
    pred = apply_thresholds(probs, threshold)
    score = macro_f1_skip_empty(y_true, pred)

    valid = y_true.sum(axis=0) > 0
    per_label_f1 = np.zeros(len(labels), dtype=np.float64)
    for j in range(len(labels)):
        if valid[j]:
            per_label_f1[j] = f1_score(y_true[:, j], pred[:, j], zero_division=0)
        else:
            per_label_f1[j] = np.nan

    true_rate = float(y_true.mean())
    pred_rate = float(pred.mean())

    metrics_dir = output_root / "metrics"
    oof_dir = output_root / "oof"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "version": "V1",
        "mode": mode,
        "model": "3-5mer TF-IDF + SGDClassifier(log_loss)",
        "rows": int(len(row_idx)),
        "label_count": int(len(labels)),
        "fixed_threshold": float(threshold),
        "macro_f1": float(score),
        "true_positive_rate": true_rate,
        "predicted_positive_rate": pred_rate,
        "mean_true_labels_per_sample": float(y_true.sum(axis=1).mean()),
        "mean_predicted_labels_per_sample": float(pred.sum(axis=1).mean()),
        "config": config,
    }

    summary_path = metrics_dir / f"v1_kmer_sgd_{mode}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    per_label = pd.DataFrame({
        "label": labels,
        "support": y_true.sum(axis=0).astype(int),
        "f1_at_0.5": per_label_f1,
        "predicted_positive_rate": pred.mean(axis=0),
        "true_positive_rate": y_true.mean(axis=0),
    })
    per_label.to_csv(metrics_dir / f"v1_kmer_sgd_{mode}_per_label.csv", index=False)

    np.savez_compressed(
        oof_dir / f"v1_kmer_sgd_{mode}_scores.npz",
        row_idx=row_idx.astype(np.int64),
        probs=probs.astype(np.float32),
        y_true=y_true.astype(np.int8),
        labels=np.asarray(labels),
    )

    print("=" * 80)
    print(f"V1 {mode} validation")
    print(f"Macro F1 @ {threshold:.2f}: {score:.6f}")
    print(f"true positive rate      : {true_rate:.6f}")
    print(f"predicted positive rate : {pred_rate:.6f}")
    print(f"true labels/sample      : {y_true.sum(axis=1).mean():.3f}")
    print(f"pred labels/sample      : {pred.sum(axis=1).mean():.3f}")
    print(f"saved summary           : {summary_path}")
    print("=" * 80)


def make_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: list[str],
    X_fit: sparse.csr_matrix,
    y_fit: np.ndarray,
    X_test: sparse.csr_matrix,
    config: dict,
    max_labels: int | None,
    output_root: Path,
    threshold: float,
) -> None:
    probs, used_labels = train_scores(
        X_fit=X_fit,
        y_fit=y_fit,
        X_eval=X_test,
        labels=labels,
        config=config,
        max_labels=max_labels,
    )
    pred = apply_thresholds(probs, threshold)

    if len(used_labels) != len(labels):
        raise ValueError("submission mode requires all 500 labels; do not use --max-labels")

    out = pd.DataFrame(pred, columns=labels)
    out.insert(0, "protein_id", test["protein_id"].astype(str).to_numpy())

    submission_dir = output_root / "submissions"
    oof_dir = output_root / "oof"
    submission_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    csv_path = submission_dir / "v1_kmer_sgd_threshold0.5.csv"
    out.to_csv(csv_path, index=False)
    np.savez_compressed(
        oof_dir / "v1_kmer_sgd_test_scores.npz",
        probs=probs.astype(np.float32),
        labels=np.asarray(labels),
        protein_id=test["protein_id"].astype(str).to_numpy(),
    )
    print(f"saved submission: {csv_path}")
    print(f"predicted positive rate: {pred.mean():.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--mode", choices=["random", "tail", "tail_clean", "chronological", "submit"], default="random")
    p.add_argument("--random-split", default="outputs/splits/group_random_seed42.npz")
    p.add_argument("--tail-split", default="outputs/splits/test_like_tail.npz")
    p.add_argument("--chronological-split", default="outputs/splits/chronological_group_holdout.npz")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Smoke-test only: train the first N labels instead of all 500.",
    )
    args = p.parse_args()

    config = load_config(Path(args.config))
    threshold = float(config.get("validation", {}).get("fixed_threshold", 0.5))

    print(f"[load] train={args.train}")
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    labels = label_columns(train)
    if len(labels) != 500:
        print(f"[warning] expected 500 labels, found {len(labels)}")

    fit_idx, eval_idx = resolve_indices(
        mode=args.mode,
        n_train=len(train),
        random_split_path=Path(args.random_split),
        tail_split_path=Path(args.tail_split),
        chronological_split_path=Path(args.chronological_split),
    )

    y_all = train[labels].to_numpy(dtype=np.int8)
    y_fit = y_all[fit_idx]

    if args.mode in {"random", "tail", "tail_clean", "chronological"}:
        assert eval_idx is not None
        eval_sequences = train.iloc[eval_idx]["sequence"]
    else:
        eval_sequences = test["sequence"]

    _, X_fit, X_eval = prepare_features(
        train_sequences=train["sequence"],
        eval_sequences=eval_sequences,
        fit_idx=fit_idx,
        config=config,
        mode=args.mode,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    )
    assert X_eval is not None

    if args.mode in {"random", "tail", "tail_clean", "chronological"}:
        assert eval_idx is not None
        probs, used_labels = train_scores(
            X_fit=X_fit,
            y_fit=y_fit[:, : (len(labels) if args.max_labels is None else args.max_labels)],
            X_eval=X_eval,
            labels=labels,
            config=config,
            max_labels=args.max_labels,
        )
        y_true = y_all[eval_idx, : len(used_labels)]
        evaluate_and_save(
            mode=args.mode,
            row_idx=eval_idx,
            y_true=y_true,
            probs=probs,
            labels=used_labels,
            threshold=threshold,
            output_root=Path(args.output_root),
            config=config,
        )
    else:
        make_submission(
            train=train,
            test=test,
            labels=labels,
            X_fit=X_fit,
            y_fit=y_fit,
            X_test=X_eval,
            config=config,
            max_labels=args.max_labels,
            output_root=Path(args.output_root),
            threshold=threshold,
        )


if __name__ == "__main__":
    main()
