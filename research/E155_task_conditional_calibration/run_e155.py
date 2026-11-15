"""E155 task-conditioned Train-only OOF threshold calibration."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

def roc_auc_score(y, s):
    y=np.asarray(y); s=np.asarray(s); order=np.argsort(s, kind='mergesort'); ranks=np.empty(len(s), float); ranks[order]=np.arange(1,len(s)+1); pos=y==1; n1=pos.sum(); n0=(~pos).sum(); return float((ranks[pos].sum()-n1*(n1+1)/2)/(n1*n0))
def f1_score(y, pred, zero_division=0):
    y=np.asarray(y); pred=np.asarray(pred); vals=[]
    for c in np.unique(np.r_[y,pred]):
        tp=np.sum((y==c)&(pred==c)); fp=np.sum((y!=c)&(pred==c)); fn=np.sum((y==c)&(pred!=c)); den=2*tp+fp+fn; vals.append(0.0 if den==0 else 2*tp/den)
    return float(np.mean(vals))

CFG_PATH=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent/'config.json'
OUT=CFG_PATH.parent; ROOT=OUT.parents[1]; CFG=json.loads(CFG_PATH.read_text(encoding='utf-8'))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); TASKS=CFG['tasks']; CLASSES=np.arange(1,21,dtype=int)

def frac(m):
    mx=m.groupby(['subject','task']).trial.transform('max').to_numpy(float); return m.trial.to_numpy(float)/np.maximum(mx,1)
def pca(x,d):
    mu=x.mean(0); xc=x-mu; w,v=np.linalg.eigh(xc.T@xc/max(len(xc)-1,1)); return mu,v[:,np.argsort(-w)[:min(d,x.shape[1])]]
def km(x,k,seed):
    if len(x)<=k:return x
    g=torch.Generator(device=DEVICE).manual_seed(seed); c=x[torch.randperm(len(x),generator=g,device=DEVICE)[:k]]
    for _ in range(15):
        a=torch.cdist(x,c).argmin(1); s=torch.zeros_like(c).index_add_(0,a,x); n=torch.zeros(k,device=DEVICE).index_add_(0,a,torch.ones(len(x),device=DEVICE)); z=torch.where(n[:,None]>0,s/n.clamp_min(1)[:,None],c)
        if torch.allclose(z,c,atol=1e-5):break
        c=z
    return c
def protos(x,y,t,mask,classes,k):
    ps=[]; keys=[]
    for task in TASKS:
        for c in classes:
            q=mask&(t==task)&(y==c)
            if q.any():
                for z in km(torch.as_tensor(x[q],dtype=torch.float32,device=DEVICE),k,int(c)*17).cpu().numpy():ps.append(z);keys.append((task,int(c)))
    return np.stack(ps),keys
def score(x,ps,keys,t):
    out=np.full(len(x),np.nan,np.float32)
    for task in TASKS:
        ix=np.flatnonzero(t==task); cc=[i for i,k in enumerate(keys) if k[0]==task]
        if len(ix) and cc:
            p=torch.as_tensor(ps[cc],dtype=torch.float32,device=DEVICE)
            for st in range(0,len(ix),2048):
                j=ix[st:st+2048]; q=torch.as_tensor(x[j],dtype=torch.float32,device=DEVICE);out[j]=(-torch.cdist(q,p)).max(1).values.cpu().numpy()
    return out
def quant(v,a): return float(np.nanquantile(v,1-a))
def threshold_map(scores,meta,a,mode,shrink=0.5):
    # thresholds learned only from known Train OOF scores; task and optional identity effects.
    task=meta.task.to_numpy(); y=meta.subject.to_numpy(int); global_t=quant(scores,a); out={'global':global_t}
    for t in TASKS:
        z=scores[task==t]; out[('task',t)]=quant(z,a) if np.isfinite(z).sum()>=8 else global_t
    if mode=='global': return out
    for t in TASKS:
        tt=out[('task',t)]
        for c in CLASSES:
            z=scores[(task==t)&(y==c)]; it=quant(z,a) if np.isfinite(z).sum()>=4 else tt
            out[('id',t,int(c))]=tt if mode=='task' else (1-shrink)*tt+shrink*it
    return out
def apply_threshold(scores,meta,th,mode):
    task=meta.task.to_numpy()
    # Test metadata has no identity column: identity thresholds are unavailable at
    # deployment time, so it falls back to the task threshold (documented in report).
    y=meta.subject.to_numpy(int) if 'subject' in meta.columns else None
    out=np.empty(len(scores))
    for i,t in enumerate(task):
        if mode=='task_identity' and y is not None and ('id',t,int(y[i])) in th:
            key=('id',t,int(y[i]))
        elif mode!='global':
            key=('task',t)
        else:
            key='global'
        out[i]=scores[i] < th.get(key,th['global'])
    return out.astype(bool)
def main():
    tm=pd.read_csv(ROOT/CFG['data']['train_metadata']); qm=pd.read_csv(ROOT/CFG['data']['test_metadata']); y=tm.subject.to_numpy(int); task=tm.task.to_numpy(); qt=qm.task.to_numpy(); f=frac(tm); fit=f<=CFG['fit_fraction']; blocks=np.clip((f*5).astype(int),0,4)
    x=np.load(ROOT/CFG['data']['train_features']).astype(np.float32); z=np.load(ROOT/CFG['data']['test_features']).astype(np.float32); mu,V=pca(x[fit].astype(np.float64),CFG['pca_dim']); x=((x.astype(np.float64)-mu)@V).astype(np.float32);z=((z.astype(np.float64)-mu)@V).astype(np.float32)
    rows=[]; stability=[]
    for k in CFG['prototype_k']:
      oof=np.full(len(x),np.nan,np.float32); loso_scores=[]; loso_meta=[]
      for b in range(5):
        base=fit&(blocks!=b); hold=blocks==b; ps,keys=protos(x,y,task,base,CLASSES,k);oof[hold]=score(x[hold],ps,keys,task[hold])
        for ident in CLASSES:
          q=hold&(y==ident)&(f>.6)&(f<=.8); base2=base&(y!=ident)
          if q.any():
            p2,k2=protos(x,y,task,base2, np.setdiff1d(CLASSES,[ident]),k); ss=score(x[q],p2,k2,task[q]); loso_scores.extend(ss[np.isfinite(ss)]); loso_meta.append(tm.iloc[np.flatnonzero(q)].copy())
      good=np.isfinite(oof); known=oof[good]; known_meta=tm.iloc[np.flatnonzero(good)].copy(); unknown=np.asarray(loso_scores,float); unknown_meta=pd.concat(loso_meta,ignore_index=True) if loso_meta else tm.iloc[:0].copy()
      for mode in ['global','task','task_identity']:
       for a in CFG['acceptance_grid']:
        th=threshold_map(known,known_meta,a,mode,0.5); kr=apply_threshold(known,known_meta,th,mode); ur=apply_threshold(unknown,unknown_meta,th,mode); proxy=float(0.5*np.mean(ur)+0.5*np.mean(~kr)); auc=float(roc_auc_score(np.r_[np.zeros(len(unknown)),np.ones(len(known))],np.r_[unknown,known])); f1=float(f1_score(np.r_[np.zeros(len(unknown)),np.ones(len(known))],np.r_[~ur,~kr],zero_division=0)); spread=[]
        for t in TASKS:
          vals=[th.get(('task',t),th['global'])]; spread.append(vals[0])
        rows.append({'k':k,'method':mode,'acceptance_rate':a,'thresholds':{str(kk):float(vv) for kk,vv in th.items()},'proxy_objective':proxy,'LOSO_rejection':float(np.mean(ur)),'known_acceptance':float(np.mean(~kr)),'Train_OOF_AUC':auc,'Train_OOF_F1':f1,'n_train_by_task':{t:int(np.sum((task==t)&fit)) for t in TASKS}})
      stability.append({'k':k,'oof_known_n':int(len(known)),'loso_unknown_n':int(len(unknown)),'task_counts':{t:int(np.sum((task==t)&fit)) for t in TASKS},'task_threshold_sd':float(np.std([quant(known[known_meta.task.to_numpy()==t],.9) for t in TASKS]))})
    best=max(rows,key=lambda r:r['proxy_objective']); selected=[r for r in rows if r['k']==best['k'] and r['method']==best['method'] and r['acceptance_rate']==best['acceptance_rate']][0]; k=selected['k']; mode=selected['method']; a=selected['acceptance_rate']
    # Test is scored only after selection; no Test metric is used for choosing parameters.
    base=fit; ps,keys=protos(x,y,task,base,CLASSES,k); ts=score(z,ps,keys,qt)
    # final thresholds reconstructed from all Train OOF scores for selected setting
    oof=np.full(len(x),np.nan,np.float32)
    for b in range(5):
      hold=blocks==b; p,k2=protos(x,y,task,fit&(blocks!=b),CLASSES,k);oof[hold]=score(x[hold],p,k2,task[hold])
    th=threshold_map(oof[np.isfinite(oof)],tm.iloc[np.flatnonzero(np.isfinite(oof))],a,mode,.5); s1=qm.session.astype(str).str.casefold().isin(['session1','s1','1']).to_numpy(); accepted=~apply_threshold(ts,qm,th,mode)
    diag={'LOSO_rejection':best['LOSO_rejection'],'Test_S1_rejection':float(np.mean(~accepted[s1])),'Test_S2_acceptance':float(np.mean(accepted[~s1])),'Test_AUC':'N/A','Test_F1':'N/A'}
    diag['W']=float(0.25*diag['LOSO_rejection']+0.25*diag['Test_S1_rejection']+0.5*diag['Test_S2_acceptance'])
    for r in rows:r['selected_by_train_proxy']=bool(r is best)
    out={'experiment':CFG['experiment'],'device':str(DEVICE),'protocol':{'train_only_oof_thresholds':True,'loso_excludes_identity_and_block':True,'test_score_only':True,'session_diagnostic_only':True},'best':{**best,**diag},'curve':rows,'stability':stability}
    (OUT/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda o:float(o)),encoding='utf-8')
    lines=['# E155 task-conditioned calibration','',f'设备：{DEVICE}；E65 raw correlation-logSPD；LOSO k 扫描：{CFG["prototype_k"]}。','',f'Train-only proxy 选中：**{mode}, k={k}, 接受率={a:.3f}**；proxy={best["proxy_objective"]:.4f}。',f'Train OOF AUC={best["Train_OOF_AUC"]:.4f}；Train OOF F1={best["Train_OOF_F1"]:.4f}。',f'LOSO rejection={diag["LOSO_rejection"]:.4f}；Test S1 rejection={diag["Test_S1_rejection"]:.4f}；Test S2 acceptance={diag["Test_S2_acceptance"]:.4f}；**W={diag["W"]:.4f}**；Test AUC/F1=N/A。','', '阈值规则预注册为 global、task 分位数、task+identity shrinkage（identity 向 task 收缩 0.5），仅由 Train OOF known 与留身份 unknown proxy 扫描；Test 分数未参与参数选择。','', 'Test 元数据没有 identity 列，因此 task_identity 阈值在实际部署时退化为 task 阈值。若 Test task 组成或各 task 接受率与 Train 不同，task-conditioned 阈值会改变总体 operating point，不能公平解释为全局阈值的无条件改进。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
