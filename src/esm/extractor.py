"""V3 ESM2 embedding extractor.

Examples:
    python -m src.esm.extractor --mode train --max-samples 100
    python -m src.esm.extractor --mode train
    python -m src.esm.extractor --mode test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def clean_sequence(seq: str) -> str:
    return (
        str(seq)
        .upper()
        .replace("B", "X")
        .replace("O", "X")
        .replace("U", "X")
        .replace("Z", "X")
    )


def make_chunks(seq: str, max_len: int, stride: int) -> list[str]:
    if len(seq) <= max_len:
        return [seq]
    parts = []
    for start in range(0, len(seq), stride):
        part = seq[start : start + max_len]
        if len(part) > 10:
            parts.append(part)
        if start + max_len >= len(seq):
            break
    return parts


def masked_mean_without_special_tokens(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.clone()
    lengths = mask.sum(dim=1)
    for i, length in enumerate(lengths.tolist()):
        if length >= 1:
            mask[i, 0] = 0
        if length >= 2:
            mask[i, int(length) - 1] = 0

    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (hidden * mask_f).sum(dim=1) / denom


@torch.inference_mode()
def encode_sequences(
    sequences: list[str],
    model_name: str,
    max_len: int,
    stride: int,
    fp16: bool,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    outputs = np.empty((len(sequences), hidden_size), dtype=np.float32)

    amp_enabled = fp16 and device.type == "cuda"
    amp_dtype = torch.float16

    for row, seq in enumerate(tqdm(sequences, desc="ESM2 embedding")):
        part_embeddings = []
        for part in make_chunks(clean_sequence(seq), max_len=max_len, stride=stride):
            token = tokenizer(
                part,
                return_tensors="pt",
                truncation=True,
                max_length=max_len + 2,
                padding=False,
            )
            token = {k: v.to(device) for k, v in token.items()}

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                hidden = model(**token).last_hidden_state
                emb = masked_mean_without_special_tokens(
                    hidden,
                    token["attention_mask"],
                )

            part_embeddings.append(emb.float().cpu().numpy()[0])

        if not part_embeddings:
            raise RuntimeError(f"sequence at row {row} produced no chunks")
        outputs[row] = np.mean(part_embeddings, axis=0, dtype=np.float32)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "random"], default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config", default="configs/v3_esm.json")
    parser.add_argument("--output-dir", default="cache/esm2_150m")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    actual_mode = "train" if args.mode == "random" else args.mode
    csv_path = Path(f"{actual_mode}.csv")
    df = pd.read_csv(csv_path, usecols=["protein_id", "sequence"])

    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    sequences = df["sequence"].astype(str).tolist()
    emb = encode_sequences(
        sequences=sequences,
        model_name=cfg["model"],
        max_len=int(cfg["max_length"]),
        stride=int(cfg["stride"]),
        fp16=bool(cfg.get("fp16", True)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.max_samples}" if args.max_samples is not None else ""
    out_path = output_dir / f"{actual_mode}_embeddings{suffix}.npy"
    ids_path = output_dir / f"{actual_mode}_protein_ids{suffix}.npy"

    np.save(out_path, emb)
    np.save(ids_path, df["protein_id"].astype(str).to_numpy())

    print(f"saved embeddings: {out_path}")
    print(f"shape: {emb.shape}, dtype={emb.dtype}")
    print(f"saved ids: {ids_path}")


if __name__ == "__main__":
    main()
