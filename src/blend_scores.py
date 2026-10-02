"""Blend aligned validation probability files and search model weight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import macro_f1_skip_empty


def load_scores(path: str):
    p=np.load(path, allow_pickle=False)
    return {
        "row_idx": p["row_idx"].astype(np.int64),
        "probs": p["probs"].astype(np.float32),
        "y_true": p["y_true"].astype(np.int8),
        "labels": p["labels"].astype(str),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="sgd")
    ap.add_argument("--name-b", default="knn")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tag", default="v6_blend")
    ap.add_argument("--output-root", default="outputs")
    args=ap.parse_args()

    A=load_scores(args.a); B=load_scores(args.b)
    if not np.array_equal(A["row_idx"], B["row_idx"]):
        raise ValueError("row_idx mismatch between score files")
    if not np.array_equal(A["labels"], B["labels"]):
        raise ValueError("label mismatch between score files")
    if not np.array_equal(A["y_true"], B["y_true"]):
        raise ValueError("y_true mismatch between score files")

    best=None
    rows=[]
    weights=np.arange(0.0, 1.0+args.step*0.5, args.step)
    for wa in weights:
        wb=1.0-wa
        probs=wa*A["probs"]+wb*B["probs"]
        pred=(probs>=args.threshold).astype(np.int8)
        score=macro_f1_skip_empty(A["y_true"], pred)
        rec={
            "weight_a":float(wa),
            "weight_b":float(wb),
            "macro_f1":float(score),
            "pred_labels_per_sample":float(pred.sum(axis=1).mean()),
        }
        rows.append(rec)
        if best is None or score>best["macro_f1"]:
            best=rec

    out=Path(args.output_root)
    (out/"metrics").mkdir(parents=True, exist_ok=True)
    (out/"oof").mkdir(parents=True, exist_ok=True)

    wa=best["weight_a"]; wb=best["weight_b"]
    probs=(wa*A["probs"]+wb*B["probs"]).astype(np.float32)
    np.savez_compressed(
        out/"oof"/f"{args.tag}_scores.npz",
        row_idx=A["row_idx"],
        probs=probs,
        y_true=A["y_true"],
        labels=A["labels"],
    )

    summary={
        "tag":args.tag,
        "model_a":args.name_a,
        "model_b":args.name_b,
        "threshold":args.threshold,
        "best":best,
        "grid":rows,
    }
    path=out/"metrics"/f"{args.tag}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("="*80)
    print(f"Best blend: {args.name_a}={wa:.2f}, {args.name_b}={wb:.2f}")
    print(f"Macro F1 @ {args.threshold:.2f}: {best['macro_f1']:.6f}")
    print(f"pred labels/sample: {best['pred_labels_per_sample']:.3f}")
    print(f"saved: {out/'oof'/f'{args.tag}_scores.npz'}")
    print("="*80)


if __name__=="__main__":
    main()
