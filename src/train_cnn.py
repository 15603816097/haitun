"""V15: lightweight multi-scale 1D CNN for protein multi-label classification.

Directly models amino-acid sequences and provides a complementary signal to
TF-IDF/KNN. Supports chronological validation first; test inference can be added
after validation proves useful.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .optimize_thresholds import fast_macro_f1


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA)}
UNK_ID = len(AA_TO_ID) + 1
PAD_ID = 0


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    seq = str(seq)
    ids = np.fromiter((AA_TO_ID.get(ch, UNK_ID) for ch in seq), dtype=np.int16)
    if len(ids) <= max_len:
        out = np.zeros(max_len, dtype=np.int16)
        out[: len(ids)] = ids
        return out

    # Deterministic head+middle+tail coverage instead of simple truncation.
    third = max_len // 3
    remain = max_len - 2 * third
    mid_start = max(0, len(ids) // 2 - remain // 2)
    return np.concatenate([
        ids[:third],
        ids[mid_start:mid_start + remain],
        ids[-third:],
    ]).astype(np.int16)


class ProteinDataset(Dataset):
    def __init__(self, seqs, y, max_len):
        self.seqs = seqs
        self.y = y
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.from_numpy(encode_sequence(self.seqs[idx], self.max_len).astype(np.int64))
        if self.y is None:
            return x
        return x, torch.from_numpy(self.y[idx].astype(np.float32))


class MultiScaleCNN(nn.Module):
    def __init__(self, n_labels=500, emb_dim=64, channels=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(UNK_ID + 1, emb_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, channels, k, padding=k // 2)
            for k in (3, 5, 9)
        ])
        self.norm = nn.LayerNorm(channels * 6)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels * 6, n_labels)

    def forward(self, x):
        mask = (x != PAD_ID).unsqueeze(1)
        z = self.embedding(x).transpose(1, 2)

        feats = []
        for conv in self.convs:
            h = torch.nn.functional.gelu(conv(z))
            neg_inf = torch.finfo(h.dtype).min
            h_max = h.masked_fill(~mask, neg_inf).amax(dim=2)
            h_sum = h.masked_fill(~mask, 0).sum(dim=2)
            denom = mask.sum(dim=2).clamp_min(1)
            h_mean = h_sum / denom
            feats.extend([h_max, h_mean])

        h = torch.cat(feats, dim=1)
        h = self.norm(h)
        h = self.dropout(h)
        return self.head(h)


def evaluate(model, loader, device):
    model.eval()
    probs = []
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys).astype(np.int8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--split", default="outputs/splits/chronological_group_holdout.npz")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--max-len", type=int, default=1536)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(args.train)
    labels = sorted([c for c in df.columns if c.startswith("label_")],
                    key=lambda c: int(c.split("_")[1]))
    y_all = df[labels].to_numpy(dtype=np.int8)
    seqs = df["sequence"].astype(str).to_numpy()

    sp = np.load(args.split)
    tr_idx = sp["train_idx"].astype(np.int64)
    va_idx = sp["val_idx"].astype(np.int64)

    train_ds = ProteinDataset(seqs[tr_idx], y_all[tr_idx], args.max_len)
    val_ds = ProteinDataset(seqs[va_idx], y_all[va_idx], args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model = MultiScaleCNN(n_labels=len(labels)).to(device)

    y_train = y_all[tr_idx]
    pos = y_train.sum(axis=0).astype(np.float64)
    neg = len(tr_idx) - pos
    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = Path(args.output_root)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "oof").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    best_probs = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * len(x)
            seen += len(x)

            if step % 200 == 0:
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={running/seen:.5f}",
                    flush=True,
                )

        probs, y_val = evaluate(model, val_loader, device)

        # Search one global threshold for epoch selection only.
        epoch_best = (-1.0, 0.5)
        for t in np.arange(0.10, 0.91, 0.02):
            f1 = fast_macro_f1(y_val, probs >= t)
            if f1 > epoch_best[0]:
                epoch_best = (f1, float(t))

        pred_count = (probs >= epoch_best[1]).sum(axis=1).mean()
        print(
            f"[epoch {epoch}] loss={running/seen:.5f} "
            f"val_F1={epoch_best[0]:.6f} t={epoch_best[1]:.2f} "
            f"pred/sample={pred_count:.3f}",
            flush=True,
        )

        if epoch_best[0] > best_f1:
            best_f1 = epoch_best[0]
            best_epoch = epoch
            best_probs = probs.astype(np.float32)
            torch.save(model.state_dict(), out / "models" / "v15_cnn_best.pt")

    np.savez_compressed(
        out / "oof" / "v15_cnn_chronological_scores.npz",
        row_idx=va_idx,
        probs=best_probs,
        y_true=y_all[va_idx],
        labels=np.asarray(labels),
    )

    summary = {
        "version": "V15",
        "best_epoch": best_epoch,
        "best_global_threshold_macro_f1": float(best_f1),
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }
    (out / "metrics" / "v15_cnn_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 80)
    print(f"BEST epoch               : {best_epoch}")
    print(f"BEST global-threshold F1 : {best_f1:.6f}")
    print(f"saved OOF                : {out/'oof'/'v15_cnn_chronological_scores.npz'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
