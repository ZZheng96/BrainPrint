from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]; CFG=json.loads((OUT/'config.json').read_text(encoding='utf8'))
CACHE=OUT/'cache'; MODELS=OUT/'models'; CACHE.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
LOG=open(OUT/'run.log','w',encoding='utf8'); sys.stdout=LOG

def log(x): print(x,flush=True)
def frac(m):
    mx=m.groupby(['subject','task']).trial.transform('max').to_numpy(float)
    return m.trial.to_numpy(float)/np.maximum(mx,1)
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s); a=s[y==1]; b=s[y==0]
    return float(((a[:,None]>b[None,:]).sum()+.5*(a[:,None]==b[None,:]).sum())/(len(a)*len(b))) if len(a) and len(b) else float('nan')
def f1(y,p):
    y=np.asarray(y).astype(bool); p=np.asarray(p).astype(bool); tp=(y&p).sum(); fp=((~y)&p).sum(); fn=(y&(~p)).sum()
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0

def robust_fit(x, tasks, fit):
    mu=np.zeros((len(CFG['tasks']),x.shape[1]),np.float32); sd=np.zeros_like(mu)
    for j,t in enumerate(CFG['tasks']):
        z=x[fit&(tasks==t)]; med=np.median(z,0); mad=np.median(np.abs(z-med),0)*1.4826
        mu[j]=med; sd[j]=np.maximum(mad,1e-4)
    return mu,sd
def standard(x,tasks,mu,sd):
    z=np.empty_like(x,np.float32)
    for j,t in enumerate(CFG['tasks']): z[tasks==t]=(x[tasks==t]-mu[j])/sd[j]
    return z
def query_norm(z):
    return (z-z.mean(1,keepdims=True))/(z.std(1,keepdims=True)+1e-5)
def whitening(z,fit,floor):
    q=z[fit]; cov=np.cov(q,rowvar=False); val,vec=np.linalg.eigh(cov); val=np.maximum(val,float(floor)); return vec@np.diag(1/np.sqrt(val))@vec.T
def protos(z,subjects,tasks,mask,classes,k):
    ps=[]; ls=[]
    for t in CFG['tasks']:
        for s in classes:
            ix=np.flatnonzero(mask&(tasks==t)&(subjects==s))
            if not len(ix): continue
            # deterministic evenly spaced multi-prototype selection; all selected rows are fit-only
            if k==1: centers=[z[ix].mean(0)]
            else:
                order=ix[np.argsort(np.linalg.norm(z[ix]-z[ix].mean(0),axis=1))]
                centers=[z[a] for a in order[np.linspace(0,len(order)-1,min(k,len(order))).astype(int)]]
            ps.extend(centers); ls.extend([(t,int(s))]*len(centers))
    return np.asarray(ps,np.float32),ls
def distances(q,ps,keys,tasks,branch=False):
    out=np.full(len(q),np.inf,np.float32); pred=np.zeros(len(q),int)
    for i,t in enumerate(CFG['tasks']):
        rows=np.flatnonzero(tasks==t); cols=[j for j,k in enumerate(keys) if k[0]==t]
        if not len(rows) or not cols: continue
        d=((q[rows,None,:]-ps[cols][None,:,:])**2).mean(2); jj=d.argmin(1); out[rows]=d[np.arange(len(rows)),jj]; pred[rows]=[keys[cols[j]][1] for j in jj]
    return out,pred
def rank_minmax(d):
    r=np.argsort(np.argsort(d,axis=0),axis=0).astype(np.float32); return r/np.maximum(len(d)-1,1)
def score_one(kind,e,b,tasks,subjects,fit,classes,k,weight=.5,floor=.05,fit_for_stats=None):
    em,es=robust_fit(e,tasks,fit); bm,bs=robust_fit(b,tasks,fit); ez=standard(e,tasks,em,es); bz=standard(b,tasks,bm,bs)
    if kind=='query_norm': ez=query_norm(ez); bz=query_norm(bz)
    if kind=='mahalanobis':
        ew=whitening(ez,fit,floor); bw=whitening(bz,fit,floor); ez=ez@ew; bz=bz@bw
    if kind=='rank_fusion':
        pe,ke=protos(ez,subjects,tasks,fit,classes,k); pb,kb=protos(bz,subjects,tasks,fit,classes,k)
        # branch distances are independently ranked/min-max fused; labels must agree by nearest branch
        de,ie=distances(ez,pe,ke,tasks); db,ib=distances(bz,pb,kb,tasks)
        # normalize per task query block, avoiding cross-task scale leakage
        fused=np.zeros(len(ez),np.float32); pred=ie.copy()
        for t in CFG['tasks']:
            ix=np.flatnonzero(tasks==t); rr=np.stack([rank_minmax(de[ix,None]),rank_minmax(db[ix,None])],1).reshape(len(ix),2)
            fused[ix]=weight*rr[:,0]+(1-weight)*rr[:,1]
        return fused,pred,(ez,bz)
    z=ez if kind!='raw' else ez
    p,keys=protos(z,subjects,tasks,fit,classes,k); d,pred=distances(z,p,keys,tasks); return d,pred,(ez,bz)
def inner_oof(kind,e,b,tasks,subjects,fit,classes,cfg):
    # OOF query blocks: every queried row is scored against prototypes excluding its own block.
    nblocks=CFG['inner_blocks']
    blocks=np.clip((np.arange(len(e))*nblocks)//max(len(e),1),0,nblocks-1)
    o=np.full(len(e),np.nan,np.float32)
    for block in range(nblocks):
        hold=fit&(blocks==block); train=fit&~hold
        if not hold.any(): continue
        s,_,_=score_one(kind,e,b,tasks,subjects,train,classes,cfg['prototype_k'],cfg.get('weight_e65',.5),cfg.get('eigen_floor',.05))
        o[hold]=s[hold]
    return o
def evaluate_scores(known,unknown,tau,pred_known,true_known):
    # score is distance: lower means accepted; unknown/rejected is positive.
    y=np.r_[np.zeros(len(known)),np.ones(len(unknown))]; s=np.r_[known,unknown]; reject=np.r_[known>tau,unknown>tau]
    return {'auc':auc(y,s),'rejection_f1':f1(y,reject),'known_acceptance':float((known<=tau).mean()) if len(known) else float('nan'),'unknown_rejection':float((unknown>tau).mean()) if len(unknown) else float('nan'),'identity_correct_but_rejected':float(((pred_known==true_known)&(known>tau)).mean()) if len(known) else float('nan'),'threshold':float(tau)}
def main():
    meta=pd.read_csv(ROOT/CFG['data']['metadata']); meta=meta[meta.task.isin(CFG['tasks'])].reset_index(drop=True); tasks=meta.task.to_numpy(); subjects=meta.subject.to_numpy(int); fr=frac(meta)
    e=np.load(ROOT/CFG['features']['e65']).astype(np.float32); b=np.load(ROOT/CFG['features']['logbp']).astype(np.float32); assert len(e)==len(meta)==len(b)
    np.savez_compressed(CACHE/'source_features.npz',e65=e,logbp=b,subject=subjects,task=tasks,frac=fr)
    classes=np.unique(subjects); folds=[]; summary={n:[] for n in CFG['candidates']}
    for held in classes:
        outer=subjects!=held; fit=outer&(fr<=CFG['fit_fraction']); cal=outer&(fr>CFG['calibration_fraction'][0])&(fr<=CFG['calibration_fraction'][1]); unk=(subjects==held)&(fr>CFG['calibration_fraction'][0])&(fr<=CFG['calibration_fraction'][1]); row={}
        for name,c in CFG['candidates'].items():
            oof=inner_oof(c['kind'],e,b,tasks,subjects,fit,np.unique(subjects[fit]),c); tau=float(np.nanquantile(oof,1-CFG['threshold_quantile']))
            # OOF threshold only; held identity never enters fit, normalization, prototype, or threshold.
            allx=np.flatnonzero(cal|unk); s,p,_=score_one(c['kind'],e,b,tasks,subjects,fit,np.unique(subjects[fit]),c['prototype_k'],c.get('weight_e65',.5),c.get('eigen_floor',.05)); known=cal&outer; unknown=unk
            m=evaluate_scores(s[known],s[unknown],tau,p[known],subjects[known]); row[name]=m; summary[name].append(m)
        folds.append({'held_subject':int(held),'metrics':row}); log(f'held={held} done')
    avg={n:{k:float(np.nanmean([x[k] for x in v])) for k in v[0]} for n,v in summary.items()}; best=max(avg,key=lambda n:(avg[n]['unknown_rejection'],avg[n]['known_acceptance']))
    c=CFG['candidates'][best]; fit=fr<=CFG['fit_fraction']; classes=np.unique(subjects); oof=inner_oof(c,e,b,tasks,subjects,fit,classes,c); tau=float(np.nanquantile(oof,1-CFG['threshold_quantile'])); s,p,state=score_one(c['kind'],e,b,tasks,subjects,fit,classes,c['prototype_k'],c.get('weight_e65',.5),c.get('eigen_floor',.05)); np.savez_compressed(MODELS/'frozen_train_state.npz',tau=tau,classes=classes)
    tm=pd.read_csv(ROOT/CFG['data']['test_metadata']); et=np.load(ROOT/CFG['features']['e65_test']).astype(np.float32); bt=np.load(ROOT/CFG['features']['logbp_test']).astype(np.float32); tt=tm.task.to_numpy(); ts,tp,_=score_one(c['kind'],np.vstack([e,et]),np.vstack([b,bt]),np.r_[tasks,tt],np.r_[subjects,np.zeros(len(tt),int)],np.r_[fit,np.zeros(len(tt),bool)],classes,c['prototype_k'],c.get('weight_e65',.5),c.get('eigen_floor',.05)); ts=ts[len(e):]; pred=np.where(ts<=tau,tp[len(e):],0); sess=tm.session.astype(str).str.casefold(); s1=sess.isin(['session1','s1','1']).to_numpy(); s2=sess.isin(['session2','s2','2']).to_numpy(); audit={'S1_N':int(s1.sum()),'S1_rejection':float((pred[s1]==0).mean()),'S2_N':int(s2.sum()),'S2_acceptance':float((pred[s2]!=0).mean()),'official_AUC':'N/A','official_F1':'N/A'}
    result={'experiment':CFG['experiment'],'device':'numpy_gpu_unavailable','n_rows':len(meta),'feature_dims':{'E65':int(e.shape[1]),'logbp':int(b.shape[1])},'fold_count':len(folds),'candidate_summary':avg,'selected_candidate':best,'folds':folds,'test_audit':audit,'protocol':{'outer_fit_only':True,'held_identity_excluded':True,'threshold_from_enrolled_inner_oof_query_blocks':True,'query_normalization_unlabeled_self_statistics':True,'test_not_used_for_selection':True}}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8'); lines=['# E220 domain-invariant prototypes','','严格修正 E219：每个 outer LOSO fold 的 robust task-wise center/scale 和原型只来自 enrolled outer_train 的 fit_fraction≤0.6；held identity 不参与。阈值仅由 enrolled train 的 inner OOF query block 分数确定。','', '|候选|LOSO AUC|拒识F1|unknown拒识|known接受|identity-correct-but-rejected|','|---|---:|---:|---:|---:|---:|']
    for n,v in avg.items(): lines.append(f"|{n}|{v['auc']:.6f}|{v['rejection_f1']:.6f}|{v['unknown_rejection']:.6f}|{v['known_acceptance']:.6f}|{v['identity_correct_but_rejected']:.6f}|")
    lines += ['',f'冻结候选：`{best}`；Test S1 rejection={audit["S1_rejection"]:.6f}，S2 acceptance={audit["S2_acceptance"]:.6f}。官方 Test AUC/F1=N/A（无身份标签）。','',f'达标判定：LOSO unknown rejection>0.90、S2>0.70、S1≥0.70；本轮：{avg[best]["unknown_rejection"]>0.90 and audit["S2_acceptance"]>0.70 and audit["S1_rejection"]>=0.70}。','', 'E65 与 E146 logbp 独立 robust 标准化；branch_rank_fusion 分别计算距离后固定权重 rank/min-max fusion；其余候选覆盖 raw distance、task-wise query normalization、特征值白化 Mahalanobis、多原型距离。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); log('completed')
if __name__=='__main__':
    try: main()
    finally: LOG.close()
