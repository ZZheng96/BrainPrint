"""E151: Train-only task/identity robust Mahalanobis open-set scoring."""
from __future__ import annotations
import json, hashlib, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score

CFG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "config.json"
CFG_PATH = CFG_PATH.resolve(); OUT = CFG_PATH.parent; ROOT = OUT.parents[1]
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8")); DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = CFG["tasks"]; CLASSES = np.arange(1, 21, dtype=int); W = CFG["weights"]

def sha256(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def frac(meta):
    mx=meta.groupby(["subject","task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float)/np.maximum(mx,1.0)

def pca_fit(x, dim):
    mu=x.mean(0); xc=x-mu
    _, v=np.linalg.eigh((xc.T@xc)/max(len(xc)-1,1))
    return mu, v[:, ::-1][:, :min(dim,x.shape[1])]

def robust_model(x, shrink):
    # Huber-like residual weighting followed by fixed shrinkage; all inputs are Train-only.
    xt=torch.as_tensor(np.asarray(x,np.float32),device=DEVICE)
    mu=xt.median(0).values
    for _ in range(3):
        r=xt-mu; scale=torch.sqrt((r*r).mean(1).clamp_min(1e-6));
        w=(CFG["robust_clip"]/scale).clamp(max=1.0); mu=(xt*w[:,None]).sum(0)/w.sum().clamp_min(1e-6)
    r=xt-mu; scale=torch.sqrt((r*r).mean(1).clamp_min(1e-6)); w=(CFG["robust_clip"]/scale).clamp(max=1.0)
    cov=(r*w[:,None]).T@r/w.sum().clamp_min(1.0)
    tr=torch.diagonal(cov).sum()/cov.shape[0]; eye=torch.eye(cov.shape[0],device=DEVICE)
    cov=(1-shrink)*cov+shrink*tr*eye+float(CFG["ridge"])*eye
    inv=torch.linalg.pinv(cov, hermitian=True)
    return mu, inv

def build_models(feat, meta, mask, excluded_subject=None, excluded_block=None, shrink=0.05, blocks=None):
    sub=meta.subject.to_numpy(int); task=meta.task.to_numpy()
    use=mask.copy()
    if excluded_subject is not None: use &= sub != int(excluded_subject)
    if excluded_block is not None: use &= blocks != int(excluded_block)
    models={}
    for task_name in TASKS:
        bg=use & (task==task_name)
        if bg.sum()==0: continue
        models[(task_name,"bg")]=robust_model(feat[bg],shrink)
        for c in CLASSES:
            sel=bg & (sub==c)
            if sel.sum() >= 3: models[(task_name,int(c))]=robust_model(feat[sel],shrink)
    return models

def score(feat, meta, models):
    task=meta.task.to_numpy(); out=np.full(len(meta),np.nan,np.float32)
    for t in TASKS:
        idx=np.flatnonzero(task==t)
        if (t,"bg") not in models: continue
        x=torch.as_tensor(np.asarray(feat[idx],np.float32),device=DEVICE)
        mb,ib=models[(t,"bg")]; db=((x-mb)@ib*(x-mb)).sum(1)
        vals=[]
        for c in CLASSES:
            if (t,int(c)) not in models: vals.append(torch.full_like(db,float("inf"))); continue
            mi,ii=models[(t,int(c))]; r=x-mi; vals.append((r@ii*r).sum(1))
        did=torch.stack(vals,1); out[idx]=(db[:,None]-did).max(1).values.detach().cpu().numpy()
    return out

def transform(train,test,meta):
    a=np.asarray(train,np.float32).copy(); b=np.asarray(test,np.float32).copy(); tasks=meta.task.to_numpy()
    for t in TASKS:
        m=tasks==t; mu=a[m].mean(0); sd=a[m].std(0)+1e-6; a[m]=(a[m]-mu)/sd
        # Test uses only task-wise Train statistics.
        q=TEST_META.task.to_numpy()==t; b[q]=(b[q]-mu)/sd
    fit=frac(meta)<=CFG["fit_fraction"]; mu,V=pca_fit(a[fit].astype(np.float64),CFG["pca_dim"])
    return ((a.astype(np.float64)-mu)@V).astype(np.float32),((b.astype(np.float64)-mu)@V).astype(np.float32)

def main():
    global TEST_META
    meta=pd.read_csv(ROOT/CFG["data"]["train_metadata"]); TEST_META=pd.read_csv(ROOT/CFG["data"]["test_metadata"])
    train=np.load(ROOT/CFG["data"]["train_features"],mmap_mode="r"); test=np.load(ROOT/CFG["data"]["test_features"],mmap_mode="r")
    if train.shape[0]!=len(meta) or test.shape[0]!=len(TEST_META): raise ValueError("feature/metadata 行数不一致")
    x,z=transform(train,test,meta); f=frac(meta); fit=f<=CFG["fit_fraction"]; blocks=np.clip((f*CFG["blocks"]).astype(int),0,CFG["blocks"]-1)
    rows=[]; shrinkages=CFG["shrinkage_grid"]
    for shrink in shrinkages:
        oof=np.full(len(meta),np.nan,np.float32)
        loso=[]
        for blk in range(CFG["blocks"]):
            mdl=build_models(x,meta,fit,excluded_block=blk,shrink=shrink,blocks=blocks)
            held=blocks==blk
            oof[held]=score(x[held],meta.iloc[np.flatnonzero(held)].reset_index(drop=True),mdl)
            # Strict LOSO: held identity and held block are both excluded from model fitting.
            for held_id in CLASSES:
                q=(blocks==blk)&(f>0.6)&(f<=0.8)&(meta.subject.to_numpy(int)==held_id)
                if not q.any(): continue
                mdl_l=build_models(x,meta,fit,excluded_subject=held_id,excluded_block=blk,shrink=shrink,blocks=blocks)
                qidx=np.flatnonzero(q); qmeta=meta.iloc[qidx].reset_index(drop=True)
                s=score(x[q],qmeta,mdl_l); loso.append(s[np.isfinite(s)])
        known=oof[np.isfinite(oof)];
        if len(known)==0: raise ValueError("OOF阈值无有效分数")
        # Fit final model only on <=60% Train, never on Test.
        final=build_models(x,meta,fit,shrink=shrink,blocks=blocks)
        test_score=score(z,TEST_META,final)
        train_rows=[]
        for acceptance in CFG["acceptance_grid"]:
            th=float(np.quantile(known,1-acceptance)); loso_rej=float(np.mean(np.concatenate(loso)<th)) if loso else float("nan")
            session=TEST_META.session.astype(str).str.casefold(); s1=session.eq("session1").to_numpy(); s2=~s1
            s1r=float(np.mean(test_score[s1]<th)); s2a=float(np.mean(test_score[s2]>=th)); ww=W["LOSO"]*loso_rej+W["S1"]*s1r+W["S2"]*s2a
            auc=float(roc_auc_score(np.r_[np.zeros(s1.sum()),np.ones(s2.sum())],np.r_[test_score[s1],test_score[s2]])) if s1.sum() and s2.sum() else float("nan")
            train_rows.append({"shrinkage":shrink,"acceptance_rate":acceptance,"threshold":th,"LOSO_rejection":loso_rej,"Test_S1_rejection":s1r,"Test_S2_acceptance":s2a,"W":ww,"test_auc_diagnostic":auc,"Test_F1":"N/A（无真实标签）"})
        rows.extend(train_rows)
        best=max(train_rows,key=lambda r:r["W"]); print(json.dumps(best,ensure_ascii=False),flush=True)
        torch.save({"shrinkage":shrink,"pca_dim":CFG["pca_dim"],"threshold_candidates":train_rows,"source_train_only":True},OUT/f"model_shrink_{shrink:.2f}.pt")
    best=max(rows,key=lambda r:r["W"]); rows.sort(key=lambda r:-r["W"])
    result={"experiment":CFG["experiment"],"protocol":{"train_only":True,"test_used_only_for_scoring":True,"oof_blocks":CFG["blocks"],"loso_excludes_identity_and_block":True,"prototype_pollution":False},"best":best,"curve":rows,"device":str(DEVICE),"source_sha256":{"config":sha256(CFG_PATH),"train_features":sha256(ROOT/CFG["data"]["train_features"]),"test_features":sha256(ROOT/CFG["data"]["test_features"])},"test":{"AUC":"N/A（无真实标签）","F1":"N/A（无真实标签）","session_diagnostic_only":True}}
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# E151 身份稳健 Mahalanobis", "", "严格 Train-only：E65 特征→任务内 Train 标准化→PCA-96→每 task/identity 稳健协方差与同任务背景模型；score = d_bg² − d_id²，身份取最大分数。", "", f"最佳收缩率={best['shrinkage']:.2f}，接受率={best['acceptance_rate']:.3f}，阈值={best['threshold']:.6f}，W={best['W']:.4f}。", f"LOSO拒识={best['LOSO_rejection']:.4f}；Test S1拒识={best['Test_S1_rejection']:.4f}；Test S2接受={best['Test_S2_acceptance']:.4f}。", "", f"当前AUC：N/A（Test无标签；Session AUC诊断值 {best['test_auc_diagnostic']:.4f}，不用于拟合/阈值）。当前F1：N/A（Test无标签）。", "", "阈值来自五块 Train OOF；LOSO 同时排除被评分身份与被评分块，模型仅使用其余 Train fit 区域；Test 仅评分，Session 仅诊断。", "", "|收缩率|接受率|LOSO|S1拒识|S2接受|W|诊断AUC|", "|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['shrinkage']:.2f}|{r['acceptance_rate']:.3f}|{r['LOSO_rejection']:.4f}|{r['Test_S1_rejection']:.4f}|{r['Test_S2_acceptance']:.4f}|{r['W']:.4f}|{r['test_auc_diagnostic']:.4f}|" for r in rows]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__": main()
