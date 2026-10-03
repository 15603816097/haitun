"""Build standalone TF-IDF KNN probability scores for a validation split."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from .train_kmer import config_fingerprint, load_config
from .tune_kmer_knn_blend import compute_knn_probs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random","chronological"], default="chronological")
    p.add_argument("--config", default="configs/v1_kmer_sgd.json")
    p.add_argument("--cache-dir", default="cache/tfidf")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--random-split", default="outputs/splits/group_random_seed42.npz")
    p.add_argument("--chronological-split", default="outputs/splits/chronological_group_holdout.npz")
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--power", type=float, default=4.0)
    p.add_argument("--output-root", default="outputs")
    args=p.parse_args()

    cfg=load_config(Path(args.config))
    cache=Path(args.cache_dir)/f"v1_{args.mode}_{config_fingerprint(cfg,args.mode)}"
    X_fit=sparse.load_npz(cache/"X_fit.npz").tocsr()
    X_eval=sparse.load_npz(cache/"X_eval.npz").tocsr()

    df=pd.read_csv(args.train)
    labels=sorted([c for c in df.columns if c.startswith("label_")], key=lambda c:int(c.split("_")[1]))
    y=df[labels].to_numpy(dtype=np.float32)
    if args.mode=="random":
        sp=np.load(args.random_split); fit_idx,eval_idx=sp["train_idx"],sp["val_idx"]
    else:
        sp=np.load(args.chronological_split); fit_idx,eval_idx=sp["train_idx"],sp["val_idx"]

    nn=NearestNeighbors(n_neighbors=args.k,metric="cosine",algorithm="brute",n_jobs=-1)
    nn.fit(X_fit)
    dist,ind=nn.kneighbors(X_eval,return_distance=True)
    sim=np.clip(1.0-dist,0.0,1.0).astype(np.float32)
    probs=compute_knn_probs(y[fit_idx],ind,sim,args.k,args.power,256)

    out=Path(args.output_root)/"oof"; out.mkdir(parents=True,exist_ok=True)
    path=out/f"v7_kmer_knn_k{args.k}_p{str(args.power).replace('.','p')}_{args.mode}_scores.npz"
    np.savez_compressed(path,row_idx=eval_idx.astype(np.int64),probs=probs.astype(np.float32),
                        y_true=y[eval_idx].astype(np.int8),labels=np.asarray(labels))
    print(f"saved: {path}")
    print(f"mean top1 similarity: {sim[:,0].mean():.4f}")

if __name__=="__main__":
    main()
