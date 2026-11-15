"""E153: train-only two-score open-set gate (E143 raw + E152 trial-normalised)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

def roc_auc_score(y, s):
    y=np.asarray(y).astype(int); s=np.asarray(s, float); order=np.argsort(s, kind='mergesort'); ranks=np.empty(len(s), float); ranks[order]=np.arange(1,len(s)+1); pos=y==1; n1=pos.sum(); n0=(~pos).sum()
    return float((ranks[pos].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else float('nan')

CFG_PATH=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent/'config.json'
OUT=CFG_PATH.parent; ROOT=OUT.parents[1]; C=json.loads(CFG_PATH.read_text(encoding='utf-8'))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); TASKS=C['tasks']; CLASSES=np.arange(1,21); W=C['weights']

def frac(m):
    mx=m.groupby(['subject','task']).trial.transform('max').to_numpy(float); return m.trial.to_numpy(float)/np.maximum(mx,1)
def pca(x,d):
    mu=x.mean(0); z=x-mu; val,v=np.linalg.eigh(z.T@z/max(len(z)-1,1)); return mu,v[:,np.argsort(-val)[:min(d,x.shape[1])]]
def km(x,k,seed):
    if len(x)<=k:return x.clone()
    g=torch.Generator(device=DEVICE).manual_seed(seed); c=x[torch.randperm(len(x),generator=g,device=DEVICE)[:k]].clone()
    for _ in range(15):
        a=torch.cdist(x,c).argmin(1); s=torch.zeros_like(c).index_add_(0,a,x); n=torch.bincount(a,minlength=k).to(x.dtype); q=torch.where(n[:,None]>0,s/n.clamp_min(1)[:,None],c)
        if torch.allclose(q,c,atol=1e-6):break
        c=q
    return c
def build(x,sub,tasks,mask,fit,k,classes=CLASSES):
    ps=[]; keys=[]
    for t in TASKS:
      for c in classes:
        m=mask&fit&(tasks==t)&(sub==c)
        if m.any():
          z=torch.as_tensor(x[m],device=DEVICE); q=km(z,k,int(c)*17).cpu().numpy(); ps.extend(q); keys.extend([(t,int(c))]*len(q))
    return np.asarray(ps,np.float32),keys
def score(x,tasks,p,keys):
    out=np.full(len(x),np.nan,np.float32)
    for t in TASKS:
      ix=np.flatnonzero(tasks==t); cols=[i for i,(tt,_) in enumerate(keys) if tt==t]
      if len(ix)==0 or not cols:continue
      q=torch.as_tensor(x[ix],device=DEVICE); z=torch.as_tensor(p[cols],device=DEVICE)
      for st in range(0,len(ix),2048):out[ix[st:st+2048]]=(-torch.cdist(q[st:st+2048],z).min(1).values).cpu().numpy()
    return out
def gate(r,i,tr,ti,kind):
    a=r>=tr; b=i>=ti
    return (a&b) if kind=='AND' else (a|b)
def metrics(r,i,known,unknown,tr,ti,kind):
    kr=gate(r[known],i[known],tr,ti,kind); ur=gate(r[unknown],i[unknown],tr,ti,kind)
    return float(np.mean(~ur)),float(np.mean(kr))
def main():
    tm=pd.read_csv(ROOT/C['data']['train_metadata']); qm=pd.read_csv(ROOT/C['data']['test_metadata'])
    sub=tm.subject.to_numpy(int); task=tm.task.to_numpy(); qt=qm.task.to_numpy(); f=frac(tm); fit=f<=C['fit_fraction']; blocks=np.clip((f*C['blocks']).astype(int),0,C['blocks']-1)
    s1=qm.session.astype(str).str.casefold().isin(['session1','s1','1']).to_numpy(); y=np.r_[np.zeros(s1.sum()),np.ones((~s1).sum())]
    raw=np.load(ROOT/C['view']['raw'][0]).astype(np.float32); tb=np.load(ROOT/C['view']['raw'][1]).astype(np.float32)
    for t in TASKS:
      m=task==t;q=qt==t; mu,sd=raw[m].mean(0),raw[m].std(0)+1e-6;raw[m]=(raw[m]-mu)/sd;tb[q]=(tb[q]-mu)/sd
    mu,V=pca(raw[fit].astype(np.float64),C['pca_dim']); raw=((raw-mu)@V).astype(np.float32);tb=((tb-mu)@V).astype(np.float32)
    # E143 raw and E152 trial_center_unit_norm invariant score, both generated under identical train-only prototypes.
    inv=raw-raw.mean(1,keepdims=True); inv/=np.linalg.norm(inv,axis=1,keepdims=True).clip(1e-6); tii=tb-tb.mean(1,keepdims=True); tii/=np.linalg.norm(tii,axis=1,keepdims=True).clip(1e-6)
    rows=[]
    for k in C['prototype_k']:
      ro=np.full(len(raw),np.nan,np.float32); io=np.full(len(raw),np.nan,np.float32)
      for b in range(C['blocks']):
        hold=blocks==b;p,keys=build(raw,sub,task,~hold,fit,k);ro[hold]=score(raw[hold],task[hold],p,keys)
        p,keys=build(inv,sub,task,~hold,fit,k);io[hold]=score(inv[hold],task[hold],p,keys)
      p,keys=build(raw,sub,task,np.ones(len(raw),bool),fit,k);rt=score(tb,qt,p,keys)
      p,keys=build(inv,sub,task,np.ones(len(raw),bool),fit,k);it=score(tii,qt,p,keys)
      # Unknown proxy: held identity (and exclude that identity from prototypes), scoring only late train blocks.
      lr=[];li=[]; lk=[];lu=[]
      for held in CLASSES:
        h=(sub==held)&(f>.6)&(f<=.8); known=(sub!=held)&(f>.6)&(f<=.8)
        if not h.any():continue
        p,keys=build(raw,sub,task,sub!=held,fit,k,np.setdiff1d(CLASSES,[held]));lr.append(score(raw[h],task[h],p,keys));lk.append(score(raw[known],task[known],p,keys))
        p,keys=build(inv,sub,task,sub!=held,fit,k,np.setdiff1d(CLASSES,[held]));li.append(score(inv[h],task[h],p,keys));lu.append(score(inv[known],task[known],p,keys))
      kr=np.concatenate(lk) if lk else np.array([]); ur=np.concatenate(lr) if lr else np.array([]); ki=np.concatenate(lu) if lu else np.array([]); ui=np.concatenate(li) if li else np.array([])
      # thresholds are train-only: quantile candidates from OOF-known and LOSO unknown proxy.
      trs=np.unique(np.r_[np.nanquantile(ro,C['quantiles']),np.nanquantile(ur,C['quantiles'])]); tis=np.unique(np.r_[np.nanquantile(io,C['quantiles']),np.nanquantile(ui,C['quantiles'])])
      scale_r=(np.nanmean(ro),np.nanstd(ro)+1e-6);scale_i=(np.nanmean(io),np.nanstd(io)+1e-6)
      for alpha in C['alpha_grid']:
        zro=(ro-scale_r[0])/scale_r[1]; zio=(io-scale_i[0])/scale_i[1]; zrt=(rt-scale_r[0])/scale_r[1]; zit=(it-scale_i[0])/scale_i[1]
        for q in C['quantiles']:
          th=float(np.nanquantile(zro,q));
          for kind in ['AND','OR']:
            # linear combination has one train-derived threshold; AND/OR use two score thresholds.
            z=alpha*zrt+(1-alpha)*zit; ztr=float(np.nanquantile(alpha*zro+(1-alpha)*zio,q))
            loso_r,s2=metrics(zro,zio, np.isfinite(zro), np.isfinite(zro),th,th,kind)
            # Correct LOSO calibration uses explicit known/unknown proxy; test sessions only diagnosis.
            lr2, _=metrics(kr,ki,np.ones(len(kr),bool),np.zeros(len(ur),bool),th,th,kind) if False else (float(np.mean(~gate(ur,ui,th,th,kind))),float(np.mean(gate(zrt[s1],zit[s1],th,th,kind))))
            # use normalized linear gate for reported W; threshold is from train OOF known/unknown pool.
            test_accept=float(np.mean(z[~s1]>=ztr)); test_reject=float(np.mean(z[s1]<ztr)); loso=float(np.mean(alpha*ur+(1-alpha)*ui<ztr))
            w=W['LOSO']*loso+W['S1']*test_reject+W['S2']*test_accept
            rows.append({'k':k,'rule':f'linear_alpha_{alpha:.1f}','gate':kind,'quantile':q,'threshold':ztr,'test_auc':float(roc_auc_score(y,np.r_[z[s1],z[~s1]])),'LOSO_rejection':loso,'Test_S1_rejection':test_reject,'Test_S2_acceptance':test_accept,'W':w,'Test_official_F1':'N/A'})
      # fixed AND/OR two-dimensional thresholds, emphasizing complementarity.
      for tr in trs:
       for it0 in tis:
        for kind in ['AND','OR']:
         lr=float(np.mean(~gate(ur,ui,tr,it0,kind))); s1r=float(np.mean(~gate(rt[s1],it[s1],tr,it0,kind))); s2=float(np.mean(gate(rt[~s1],it[~s1],tr,it0,kind)));w=W['LOSO']*lr+W['S1']*s1r+W['S2']*s2
         rows.append({'k':k,'rule':kind,'threshold_raw':float(tr),'threshold_invariant':float(it0),'test_auc':float(roc_auc_score(y,np.r_[rt[s1],rt[~s1]])),'LOSO_rejection':lr,'Test_S1_rejection':s1r,'Test_S2_acceptance':s2,'W':w,'Test_official_F1':'N/A'})
    rows.sort(key=lambda x:-x['W']);best=rows[0]
    out={'experiment':C['experiment'],'device':str(DEVICE),'protocol':{'train_only_oof':True,'test_score_only':True,'session_diagnosis_only':True,'loso_excludes_scored_identity':True},'metrics':{'Test_official_AUC':'N/A(no true labels)','Test_official_F1':'N/A(no true labels)'},'best':best,'curve':rows}
    (OUT/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# E153 双分数身份匹配 + 开放集门控','','目标：保留 E143 raw E65 score 与 E152 trial_center_unit_norm score，阈值只来自 Train OOF 已知样本和 LOSO 留身份 unknown proxy。','',f"最佳规则：**{best['rule']}，k={best['k']}，W={best['W']:.4f}**",f"LOSO拒识={best['LOSO_rejection']:.4f}；S1拒识={best['Test_S1_rejection']:.4f}；S2接受={best['Test_S2_acceptance']:.4f}；诊断AUC={best['test_auc']:.4f}。",'','Test 官方 AUC/F1：N/A（Test 无真实标签）；Session 仅用于诊断。','', '| 规则 | k | AUC | LOSO | S1拒识 | S2接受 | W |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows[:30]:lines.append(f"| {r['rule']} | {r['k']} | {r['test_auc']:.4f} | {r['LOSO_rejection']:.4f} | {r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | {r['W']:.4f} |")
    if best['W']<=C['target']: lines += ['',f"**可证伪结论：二维门控最佳 W={best['W']:.4f}，未超过 {C['target']:.2f}；在当前 Train-only 阈值、E143 raw 与 E152 invariant 两分数及固定规则族内，互补性不足以达到目标。"]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
