from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
CFG=json.loads((OUT/'config.json').read_text(encoding='utf-8')); DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TASKS=CFG['tasks']; CLASSES=np.arange(1,21); FIT=CFG['fit_fraction']; ACC=CFG['acceptance_grid']; MODEL=OUT/'models'; MODEL.mkdir(exist_ok=True)

def log(x): print(x,flush=True)
def frac(meta):
    mx=meta.groupby(['subject','task']).trial.transform('max').to_numpy(float); return meta.trial.to_numpy(float)/np.maximum(mx,1)
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s); pos=y==1; neg=~pos
    order=np.argsort(s,kind='mergesort'); ss=s[order]; r=np.empty(len(s)); i=0
    while i<len(s):
        j=i+1
        while j<len(s) and ss[j]==ss[i]: j+=1
        r[order[i:j]]=(i+1+j)/2; i=j
    return float((r[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))
def f1(y,p):
    tp=((y==1)&(p==1)).sum(); den=2*tp+((y==0)&(p==1)).sum()+((y==1)&(p==0)).sum(); return float(2*tp/den) if den else 0.
def repr_fit(raws, tasks, fit, dims):
    reps={}; states={}
    for n in raws:
        x=raws[n].astype(np.float32,copy=True); mu=[]; sd=[]
        for t in TASKS:
            m=(tasks==t)&fit; a=x[m].mean(0); b=x[m].std(0)+1e-6; x[tasks==t]=(x[tasks==t]-a)/b; mu.append(a); sd.append(b)
        z=x[fit].astype(np.float64); pm=z.mean(0); _,_,v=np.linalg.svd(z-pm,full_matrices=False); d=min(dims[n],z.shape[1],len(z)-1); basis=v[:d].T
        p=((x.astype(np.float64)-pm)@basis).astype(np.float32); scale=float(np.sqrt(np.mean(p[fit]**2))+1e-6)
        reps[n]=p/scale; states[n]=(np.asarray(mu,dtype=object),np.asarray(sd,dtype=object),pm,basis,scale)
    return reps,states
def transform(raws,tasks,states):
    reps={}
    for n in raws:
        x=raws[n].astype(np.float32,copy=True); mu,sd,pm,basis,scale=states[n]
        for i,t in enumerate(TASKS): x[tasks==t]=(x[tasks==t]-mu[i])/sd[i]
        reps[n]=(((x.astype(np.float64)-pm)@basis).astype(np.float32))/scale
    return reps
def protos(x,subjects,tasks,mask,classes,k):
    ps=[]; keys=[]
    for t in TASKS:
        for s in classes:
            m=mask&(tasks==t)&(subjects==s)
            if m.any():
                q=x[m]; rng=np.random.default_rng(int(s)*17+len(t)); c=q[rng.choice(len(q),min(k,len(q)),False)].copy()
                for _ in range(20):
                    a=torch.cdist(torch.as_tensor(q,device=DEVICE),torch.as_tensor(c,device=DEVICE)).argmin(1).cpu().numpy(); nc=np.array([q[a==j].mean(0) if np.any(a==j) else c[j] for j in range(len(c))],np.float32)
                    if np.allclose(nc,c,atol=1e-6): break
                    c=nc
                ps.append(c); keys += [(t,int(s))]*len(c)
    return np.concatenate(ps).astype(np.float32),keys
def score(x,tasks,ps,keys):
    out=np.full(len(x),-np.inf,np.float32); kt=np.array([a for a,b in keys])
    for t in TASKS:
        r=np.flatnonzero(tasks==t); c=np.flatnonzero(kt==t)
        if len(r) and len(c): out[r]=(-torch.cdist(torch.as_tensor(x[r],device=DEVICE),torch.as_tensor(ps[c],device=DEVICE))).max(1).values.cpu().numpy()
    return out
def metric(sk,su,ref,a):
    th=float(np.quantile(ref,1-a)); ka=sk>=th; rej=su<th; return {'acceptance':a,'threshold':th,'rejection':float(rej.mean()),'known_acceptance':float(ka.mean()),'rejection_f1':f1(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[ka,rej]),'n_known':len(sk),'n_unknown':len(su),'accepted_known':int(ka.sum()),'rejected_unknown':int(rej.sum())}
def run_candidate(name,streams,raw,te_raw,meta,tmeta,fr,subjects,blocks):
    root=MODEL/name; root.mkdir(exist_ok=True); fold_data=[]; per=[]; tasks=meta.task.to_numpy(); tt=tmeta.task.to_numpy(); window=(fr>0.6)&(fr<=0.8)
    for held in CLASSES:
        fp=root/f'held_subject_{held:02d}.npz'
        outer=subjects!=held; ofit=outer&(fr<=FIT); qblocks=sorted(np.unique(blocks[ofit]).tolist()); oof={n:np.full(len(meta),np.nan,np.float32) for n in streams}
        for b in qblocks:
            fit=ofit&(blocks!=b); q=ofit&(blocks==b); x,st=repr_fit({n:raw[n] for n in streams},tasks,fit,{n:CFG['streams'][n]['pca'] for n in streams});
            for n in streams:
                ps,keys=protos(x[n],subjects,tasks,fit,np.setdiff1d(CLASSES,[held]),CFG['streams'][n]['k']); oof[n][q]=score(x[n][q],tasks[q],ps,keys)
        mus={n:float(np.nanmean(oof[n][ofit])) for n in streams}; sds={n:float(np.nanstd(oof[n][ofit])+1e-6) for n in streams}; ref=np.zeros(len(meta),np.float32)
        for n in streams: ref+=np.nan_to_num((oof[n]-mus[n])/sds[n])
        x,st=repr_fit({n:raw[n] for n in streams},tasks,ofit,{n:CFG['streams'][n]['pca'] for n in streams})
        known=window&outer; unknown=window&(subjects==held); sk=np.zeros(known.sum(),np.float32); su=np.zeros(unknown.sum(),np.float32)
        for n in streams:
            ps,keys=protos(x[n],subjects,tasks,ofit,np.setdiff1d(CLASSES,[held]),CFG['streams'][n]['k']); sk+=(score(x[n][known],tasks[known],ps,keys)-mus[n])/sds[n]; su+=(score(x[n][unknown],tasks[unknown],ps,keys)-mus[n])/sds[n]
        curve=[metric(sk,su,ref[ofit],a) for a in ACC]; fold_data.append({'sk':sk,'su':su,'curve':curve,'auc':auc(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[sk,su])}); q=next(z for z in curve if z['acceptance']==.90); per.append({'subject':int(held),'fold_auc':fold_data[-1]['auc'],'rejection_090':q['rejection'],'known_acceptance_090':q['known_acceptance'],'rejection_f1_090':q['rejection_f1']}); np.savez_compressed(fp,known_scores=sk,unknown_scores=su,reference=ref[ofit],curve=np.array([json.dumps(z) for z in curve]))
        log(f'{name} outer {held}/20 done')
    curve=[]
    for a in ACC:
        rows=[next(z for z in f['curve'] if z['acceptance']==a) for f in fold_data]; nk=sum(z['n_known'] for z in rows); nu=sum(z['n_unknown'] for z in rows); q={'acceptance':a,'LOSO_rejection_micro':sum(z['rejected_unknown'] for z in rows)/nu,'LOSO_rejection_macro':float(np.mean([z['rejection'] for z in rows])),'known_acceptance_micro':sum(z['accepted_known'] for z in rows)/nk,'known_acceptance_macro':float(np.mean([z['known_acceptance'] for z in rows])),'rejection_f1_mean':float(np.mean([z['rejection_f1'] for z in rows]))}; curve.append(q)
    fit=fr<=FIT; x,st=repr_fit({n:raw[n] for n in streams},tasks,fit,{n:CFG['streams'][n]['pca'] for n in streams}); final={n:np.full(len(meta),np.nan,np.float32) for n in streams};
    for b in sorted(np.unique(blocks[fit])):
        fm=fit&(blocks!=b); qm=fit&(blocks==b); xx,ss=repr_fit({n:raw[n] for n in streams},tasks,fm,{n:CFG['streams'][n]['pca'] for n in streams})
        for n in streams:
            ps,keys=protos(xx[n],subjects,tasks,fm,CLASSES,CFG['streams'][n]['k']); final[n][qm]=score(xx[n][qm],tasks[qm],ps,keys)
    mus={n:float(np.nanmean(final[n][fit])) for n in streams}; sds={n:float(np.nanstd(final[n][fit])+1e-6) for n in streams}; ref=sum((final[n]-mus[n])/sds[n] for n in streams); xt=transform({n:te_raw[n] for n in streams},tt,st); ts=np.zeros(len(tmeta),np.float32)
    for n in streams:
        ps,keys=protos(x[n],subjects,tasks,fit,CLASSES,CFG['streams'][n]['k']); ts+=(score(xt[n],tt,ps,keys)-mus[n])/sds[n]
    s1=tmeta.session.astype(str).str.casefold().isin(['session1','s1','1']).to_numpy();
    for q in curve:
        th=float(np.quantile(ref[fit],1-q['acceptance'])); q['Test_S1_rejection']=float(np.mean(ts[s1]<th)); q['Test_S2_acceptance']=float(np.mean(ts[~s1]>=th)); q['threshold_final']=th
    r={'config':name,'device':str(DEVICE),'streams':streams,'loso_auc_fold_mean':float(np.mean([f['auc'] for f in fold_data])),'loso_auc_fold_values':[f['auc'] for f in fold_data],'acceptance_curve':curve,'reference_090':next(z for z in curve if z['acceptance']==.90),'per_subject':per,'test_official':{'AUC':'N/A','F1':'N/A'},'strict_protocol':True,'cache':'E217 outer fold caches; E215/E189 source caches not used for scores'}
    np.savez_compressed(root/'final_deployment_scores.npz',test_scores=ts,final_reference=ref[fit]); return r

def main():
    sys.stdout=open(OUT/'run.log','w',encoding='utf-8'); t0=time.time(); meta=pd.read_csv(ROOT/CFG['data']['train_metadata']); tm=pd.read_csv(ROOT/CFG['data']['test_metadata']); fr=frac(meta); blocks=np.floor(fr*CFG['blocks']).astype(int); subjects=meta.subject.to_numpy(int); tasks=meta.task.to_numpy(); raw={}; te={}
    for n,s in CFG['streams'].items(): raw[n]=np.load(ROOT/s['train']).astype(np.float32); te[n]=np.load(ROOT/s['test']).astype(np.float32)
    results={}; log(f'device={DEVICE}, train={len(meta)}, test={len(tm)}, fit={(fr<=FIT).sum()}')
    for name,streams in CFG['candidates'].items(): results[name]=run_candidate(name,streams,raw,te,meta,tm,fr,subjects,blocks)
    results['runtime_seconds']=time.time()-t0; (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# E217 strict fusion audit','',f'- 设备：`{DEVICE}`；严格 outer LOSO，held identity 不参与标准化、PCA、原型或阈值；Test 仅冻结后审计。','- E189/E215 缓存仅作历史参考，E217 从原始特征缓存按 outer fold 重新拟合；官方 Test AUC/F1：N/A（无身份标签）。','', '|候选|LOSO AUC|acceptance|LOSO reject micro|LOSO reject macro|known accept|reject F1|Test S1 rejection|Test S2 acceptance|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in results.items():
        if isinstance(r,dict):
            for q in r['acceptance_curve']: lines.append(f'|{n}|{r["loso_auc_fold_mean"]:.6f}|{q["acceptance"]:.2f}|{q["LOSO_rejection_micro"]:.6f}|{q["LOSO_rejection_macro"]:.6f}|{q["known_acceptance_micro"]:.6f}|{q["rejection_f1_mean"]:.6f}|{q["Test_S1_rejection"]:.6f}|{q["Test_S2_acceptance"]:.6f}|')
    lines += ['',f'运行耗时：{time.time()-t0:.1f}s。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
