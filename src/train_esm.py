"""Train a multilabel MLP on cached ESM2 embeddings.

Examples:
    python -m src.train_esm --mode random
    python -m src.train_esm --mode tail
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import macro_f1_skip_empty


class ESMClassifier(nn.Module):
    def __init__(self, dim: int, labels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "random":
        pack = np.load("outputs/splits/group_random_seed42.npz")
        return pack["train_idx"], pack["val_idx"]
    if mode == "tail":
        pack = np.load("outputs/splits/test_like_tail.npz")
        return pack["head_idx"], pack["tail_idx"]
    raise ValueError(mode)


def predict_probs(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(x).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(xb)
            chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "tail"], default="random")
    parser.add_argument("--config", default="configs/v3_esm.json")
    parser.add_argument("--embeddings", default="cache/esm2_150m/train_embeddings.npy")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ccfg = cfg["classifier"]
    seed = int(ccfg["seed"])
    set_seed(seed)

    x = np.load(args.embeddings, mmap_mode="r")
    df = pd.read_csv(args.train)
    label_cols = sorted(
        [c for c in df.columns if c.startswith("label_")],
        key=lambda c: int(c.split("_")[1]),
    )
    y = df[label_cols].to_numpy(dtype=np.float32)

    if len(x) != len(df):
        raise ValueError(
            f"embedding rows ({len(x)}) != train rows ({len(df)}). "
            "Use the full train_embeddings.npy."
        )
    if x.shape[1] != int(cfg["embedding_dim"]):
        raise ValueError(
            f"embedding dim ({x.shape[1]}) != config embedding_dim "
            f"({cfg['embedding_dim']})"
        )

    tr_idx, va_idx = load_split(args.mode)
    x_train = np.asarray(x[tr_idx], dtype=np.float32)
    x_val = np.asarray(x[va_idx], dtype=np.float32)
    y_train = y[tr_idx]
    y_val = y[va_idx]

    batch_size = int(ccfg["batch_size"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(ccfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMClassifier(
        dim=x.shape[1],
        labels=len(label_cols),
        hidden_dim=int(ccfg["hidden_dim"]),
        dropout=float(ccfg["dropout"]),
    ).to(device)

    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, float(ccfg["max_pos_weight"]))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight).float().to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ccfg["learning_rate"]),
        weight_decay=float(ccfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_root = Path(args.output_root)
    model_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    oof_dir = out_root / "oof"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = -1
    patience = 0
    history = []
    model_path = model_dir / f"v3_esm_mlp_{args.mode}.pt"

    for epoch in range(1, int(ccfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach()) * len(xb)
            seen += len(xb)

        probs = predict_probs(model, x_val, batch_size, device)
        pred = (probs >= 0.5).astype(np.int8)
        score = macro_f1_skip_empty(y_val.astype(np.int8), pred)
        avg_loss = running_loss / max(seen, 1)
        pred_labels = float(pred.sum(axis=1).mean())

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "macro_f1_at_0.5": score,
            "predicted_labels_per_sample": pred_labels,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={avg_loss:.6f} "
            f"val_macro_f1@0.5={score:.6f} "
            f"pred_labels/sample={pred_labels:.3f}"
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "embedding_dim": int(x.shape[1]),
                    "label_cols": label_cols,
                    "config": cfg,
                    "mode": args.mode,
                    "best_epoch": best_epoch,
                    "best_macro_f1_at_0.5": best_score,
                },
                model_path,
            )
        else:
            patience += 1
            if patience >= int(ccfg["early_stopping_patience"]):
                print(f"early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    probs = predict_probs(model, x_val, batch_size, device)
    pred = (probs >= 0.5).astype(np.int8)
    final_score = macro_f1_skip_empty(y_val.astype(np.int8), pred)

    np.savez_compressed(
        oof_dir / f"v3_esm_mlp_{args.mode}_scores.npz",
        row_idx=va_idx.astype(np.int64),
        probs=probs,
        y_true=y_val.astype(np.int8),
        labels=np.asarray(label_cols),
    )

    summary = {
        "version": "V3",
        "mode": args.mode,
        "embedding_model": cfg["model"],
        "embedding_dim": int(x.shape[1]),
        "train_rows": int(len(tr_idx)),
        "validation_rows": int(len(va_idx)),
        "best_epoch": int(best_epoch),
        "macro_f1_at_0.5": float(final_score),
        "true_positive_rate": float(y_val.mean()),
        "predicted_positive_rate": float(pred.mean()),
        "true_labels_per_sample": float(y_val.sum(axis=1).mean()),
        "predicted_labels_per_sample": float(pred.sum(axis=1).mean()),
        "history": history,
        "config": cfg,
    }
    summary_path = metrics_dir / f"v3_esm_mlp_{args.mode}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"V3 ESM MLP {args.mode}")
    print(f"best epoch                 : {best_epoch}")
    print(f"Macro F1 @ 0.50            : {final_score:.6f}")
    print(f"true labels/sample         : {y_val.sum(axis=1).mean():.3f}")
    print(f"predicted labels/sample    : {pred.sum(axis=1).mean():.3f}")
    print(f"saved model                : {model_path}")
    print(f"saved probabilities        : outputs/oof/v3_esm_mlp_{args.mode}_scores.npz")
    print("=" * 80)


if __name__ == "__main__":
    main()
