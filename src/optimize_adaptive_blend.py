"""Cross-fitted label-adaptive SGD/KNN blending and thresholds for Macro F1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .optimize_thresholds import fast_macro_f1

def best_threshold(y,p,fallback=0.48,tmin=0.05,tmax=0.95):
    pos=int(y.sum())
    if pos==0: return fallback,0.0
    order=np.argsort(-p,kind="mergesort")
    ys=y[order].astype(np.int64); ps=p[order]
    tp=np.cumsum(ys); k=np.arange(1,len(ys)+1)
    denom=pos+k
    f1=np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=np.float64),where=denom>0)
    valid=(ps>=tmin)&(ps<=tmax)
    if not np.any(valid): return fallback,0.0
    idx=int(np.argmax(np.where(valid,f1,-1.0)))
    return float(ps[idx]),float(f1[idx])

def fit_label_params(y,sgd,knn,weights,global_w,global_t,shrink_k):
    L=y.shape[1]
    out_w=np.full(L,global_w,dtype=np.float32)
    out_t=np.full(L,global_t,dtype=np.float32)
    support=y.sum(axis=0).astype(np.float64)
    for j in range(L):
        if support[j]<=0: continue
        best=(-1.0,global_w,global_t)
        for w in weights:
            p=w*sgd[:,j]+(1.0-w)*knn[:,j]
            t,score=best_threshold(y[:,j],p,fallback=global_t)
            if score>best[0]:
                best=(score,float(w),float(t))
        shrink=support[j]/(support[j]+shrink_k)
        out_w[j]=shrink*best[1]+(1-shrink)*global_w
        out_t[j]=shrink*best[2]+(1-shrink)*global_t
    return out_w,out_t

def predict(sgd,knn,w,t):
    probs=sgd*w[None,:]+knn*(1.0-w[None,:])
    return probs,(probs>=t[None,:]).astype(np.int8)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sgd", required=True)
    p.add_argument("--knn", required=True)
    p.add_argument("--train", default="train.csv")
    p.add_argument("--global-weight", type=float, default=0.40)
    p.add_argument("--global-threshold", type=float, default=0.48)
    p.add_argument("--weight-step", type=float, default=0.10)
    p.add_argument("--shrink-k", type=float, default=300.0)
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--tag", default="v8_label_adaptive_chronological")
    p.add_argument("--output-root", default="outputs")
    args=p.parse_args()

    A=np.load(args.sgd,allow_pickle=False); B=np.load(args.knn,allow_pickle=False)
    row=A["row_idx"].astype(np.int64)
    if not np.array_equal(row,B["row_idx"]): raise ValueError("row mismatch")
    y=A["y_true"].astype(np.int8)
    if not np.array_equal(y,B["y_true"]): raise ValueError("y mismatch")
    labels=A["labels"].astype(str)
    sgd=A["probs"].astype(np.float32); knn=B["probs"].astype(np.float32)

    train=pd.read_csv(args.train,usecols=["sequence"])
    groups=train.iloc[row]["sequence"].astype(str).to_numpy()
    weights=np.arange(0,1+args.weight_step*0.5,args.weight_step,dtype=np.float32)

    pred_cf=np.zeros_like(y,dtype=np.int8)
    fold_rows=[]
    gkf=GroupKFold(n_splits=args.folds)
    for fold,(cal,ev) in enumerate(gkf.split(sgd,y,groups),1):
        w,t=fit_label_params(y[cal],sgd[cal],knn[cal],weights,args.global_weight,args.global_threshold,args.shrink_k)
        _,pred=predict(sgd[ev],knn[ev],w,t)
        pred_cf[ev]=pred
        score=fast_macro_f1(y[ev],pred)
        fold_rows.append({"fold":fold,"macro_f1":float(score),"pred_labels_per_sample":float(pred.sum(1).mean())})
        print(f"fold {fold}: Macro F1={score:.6f}, pred/sample={pred.sum(1).mean():.3f}")

    cf_score=fast_macro_f1(y,pred_cf)
    full_w,full_t=fit_label_params(y,sgd,knn,weights,args.global_weight,args.global_threshold,args.shrink_k)
    probs_full,pred_full=predict(sgd,knn,full_w,full_t)
    fit_score=fast_macro_f1(y,pred_full)

    out=Path(args.output_root)
    (out/"adaptive").mkdir(parents=True,exist_ok=True)
    (out/"metrics").mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"adaptive"/f"{args.tag}.npz",labels=labels,weights=full_w,thresholds=full_t,
                        global_weight=np.float32(args.global_weight),global_threshold=np.float32(args.global_threshold))
    summary={"tag":args.tag,"crossfit_macro_f1":float(cf_score),"fit_all_macro_f1":float(fit_score),
             "crossfit_pred_labels_per_sample":float(pred_cf.sum(1).mean()),
             "true_labels_per_sample":float(y.sum(1).mean()),"folds":fold_rows,
             "mean_sgd_weight":float(full_w.mean()),"median_sgd_weight":float(np.median(full_w)),
             "shrink_k":args.shrink_k}
    (out/"metrics"/f"{args.tag}_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("="*80)
    print(f"Crossfit Macro F1       : {cf_score:.6f}")
    print(f"Fit-all Macro F1*       : {fit_score:.6f}")
    print(f"pred labels/sample      : {pred_cf.sum(1).mean():.3f} (true={y.sum(1).mean():.3f})")
    print(f"mean/median SGD weight  : {full_w.mean():.3f}/{np.median(full_w):.3f}")
    print(f"saved params            : {out/'adaptive'/f'{args.tag}.npz'}")
    print("* optimistic fit-all")
    print("="*80)

if __name__=="__main__":
    main()
