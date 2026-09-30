"""V3 ESM2 embedding extractor.

Usage:
python -m src.esm.extractor --mode random --max-samples 100
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
MAX_LEN = 1022
STRIDE = 512


def clean_sequence(seq):
    return str(seq).upper().replace("B", "X").replace("O", "X").replace("U", "X").replace("Z", "X")


def chunks(seq):
    if len(seq) <= MAX_LEN:
        return [seq]
    return [seq[i:i+MAX_LEN] for i in range(0, len(seq), STRIDE) if len(seq[i:i+MAX_LEN]) > 10]


@torch.no_grad()
def encode_sequences(sequences, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    outputs = []
    for seq in tqdm(sequences):
        part_embeddings = []
        for part in chunks(clean_sequence(seq)):
            token = tokenizer(part, return_tensors="pt", truncation=True,
                              max_length=1024, padding=True)
            token = {k: v.to(device) for k, v in token.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                hidden = model(**token).last_hidden_state
            mask = token["attention_mask"].unsqueeze(-1)
            emb = (hidden * mask).sum(1) / mask.sum(1)
            part_embeddings.append(emb.float().cpu().numpy()[0])
        outputs.append(np.mean(part_embeddings, axis=0))
    return np.asarray(outputs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="random")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv("train.csv")
    seqs = df.sequence.tolist()
    if args.max_samples:
        seqs = seqs[:args.max_samples]

    emb = encode_sequences(seqs)
    os.makedirs("cache/esm2_150m", exist_ok=True)
    np.save("cache/esm2_150m/train_embeddings.npy", emb)
    print("saved", emb.shape)


if __name__ == "__main__":
    main()
