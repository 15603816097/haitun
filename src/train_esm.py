"""Train multilabel classifier on cached ESM2 embeddings."""
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score


class ESMClassifier(nn.Module):
    def __init__(self, dim=1280, labels=500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, labels)
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    x = np.load('cache/esm2_150m/train_embeddings.npy')
    df = pd.read_csv('train.csv')
    y = df[[c for c in df.columns if c.startswith('label_')]].values.astype('float32')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ESMClassifier(labels=y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(x).to(device)
    yt = torch.tensor(y).to(device)
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        print('epoch', epoch, 'loss', float(loss))

    torch.save(model.state_dict(), 'outputs/models/esm_mlp.pt')


if __name__ == '__main__':
    main()
