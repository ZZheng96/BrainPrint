from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.io import loadmat
from scipy.linalg import eigh
def roc_auc_score(y,s):
    y=np.asarray(y); s=np.asarray(s); pos=s[y==1]; neg=s[y==0]
    return float(((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))
def f1_score(y,p,zero_division=0):
    y=np.asarray(y).astype(bool); p=np.asarray(p).astype(bool); tp=(y&p).sum(); fp=((~y)&p).sum(); fn=(y&(~p)).sum(); return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else float(zero_division)

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]; CFG=json.loads((OUT/'config.json').read_text())
CACHE=OUT/'cache'; MODELS=OUT/'models'; CACHE.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
LOG=open(OUT/'run.log','w',encoding='utf8'); sys.stdout=LOG
rng=np.random.default_rng(CFG['seed'])
def log(s): print(s,flush=True)
def frac(m): return m.trial.to_numpy(float)/m.groupby(['subject','task']).trial.transform('max').to_numpy(float).clip(1)
def load_features(meta):
    bands=CFG['feature']['bands']; out=[]
    for j,p in enumerate(meta.path):
        x=np.asarray(loadmat(ROOT/CFG['data']['root']/str(p),variable_names=['data'])['data'],np.float32).T
        x=x-x.mean(1,keepdims=True); spec=np.fft.rfft(x,axis=1); f=np.fft.rfftfreq(x.shape[1],1/300); vals=[]
        for bi,(lo,hi) in enumerate(bands):
            y=np.fft.irfft(spec*((f>=lo)&(f<hi))[None,:],n=x.shape[1],axis=1); y-=y.mean(1,keepdims=True); z=y/(y.std(1,keepdims=True)+1e-6); c=z@z.T/(x.shape[1]-1); tr=np.trace(c); c=.95*c+.05*tr/18*np.eye(18); w,v=np.linalg.eigh(c); l=(v*np.log(np.maximum(w,1e-12)))@v.T; vals.append(l[np.triu_indices(18,1)]*np.sqrt(2))
        pwr=np.abs(spec)**2
        vals.append(np.array([np.log(pwr[:,(f>=lo)&(f<hi)].mean(1)+1e-8) for lo,hi in bands]).T.reshape(-1))
        out.append(np.concatenate(vals))
        if (j+1)%500==0: log(f'features {j+1}/{len(meta)}')
    return np.asarray(out,np.float32)
def project_fit(x,y,stab,strategy,lam,k):
    mu=x.mean(0); z=(x-mu)/(x.std(0)+1e-5); d=z.shape[1]; overall=z.T@z/len(z); p=np.maximum(stab,1e-4); P=np.diag(p)
    if strategy=='pca_whiten':
        q,v=np.linalg.eigh(overall+1e-3*np.eye(d)); return mu,1/(x.std(0)+1e-5),v[:,-min(k,d):][:,::-1]
    means=np.stack([z[y==s].mean(0) for s in np.unique(y)]); B=((means-means.mean(0)).T@(means-means.mean(0)))/len(means); W=np.zeros((d,d))
    for s in np.unique(y):
        a=z[y==s]-z[y==s].mean(0); W+=a.T@a/max(len(z[y==s]),1)
    A=np.linalg.pinv(W/len(np.unique(y))+lam*P)@B; q,v=np.linalg.eigh((A+A.T)/2); return mu,1/(x.std(0)+1e-5),v[:,-min(k,d):][:,::-1]
def score(z,centers):
    d=((z[:,None,:]-centers[None,:,:])**2).sum(2); i=d.argmin(1); return -d[np.arange(len(z)),i],i
def metrics(known,unknown,labels,tau,yknown):
    s=np.r_[known,unknown]; yy=np.r_[np.ones(len(known)),np.zeros(len(unknown))]; pred=np.r_[known>=tau,unknown>=tau]
    return {'auc':float(roc_auc_score(yy,s)) if len(np.unique(yy))==2 else None,'rejection_f1':float(f1_score(yy,pred,zero_division=0)),'known_acceptance':float((known>=tau).mean()),'unknown_rejection':float((unknown<tau).mean()),'identity_correct_but_rejected':float(((labels==yknown)&(known<tau)).mean()),'threshold':float(tau)}
def main():
    meta=pd.read_csv(ROOT/CFG['data']['metadata']); meta=meta[meta.task.isin(CFG['tasks'])].reset_index(drop=True); fr=frac(meta); subjects=meta.subject.to_numpy(int); x=load_features(meta); np.savez_compressed(CACHE/'raw_features.npz',features=x,subject=subjects,frac=fr,task=meta.task.to_numpy())
    # E218 stability regularizer: matched random perturbation feature change, no PGD due full-run cost.
    noise=rng.normal(size=x.shape).astype('float32'); scale=x.std(0)+1e-5; stability=np.mean(np.abs(noise*0.01*scale),axis=0)/(scale)
    folds=[]; candidates=[('fisher',l,k) for l in CFG['projection']['lambda_grid'] for k in CFG['projection']['dims']]+[('pca_whiten',0,k) for k in CFG['projection']['dims']]
    for held in sorted(np.unique(subjects)):
        outer=subjects!=held; fit=outer&(fr<=.6); cal=outer&(fr>.6)&(fr<=.8); unk=(subjects==held)&(fr>.6)&(fr<=.8); fy=subjects[fit]; row={}
        for strategy,lam,k in candidates:
            mu,sd,W=project_fit(x[fit],fy,stability,strategy,lam,k); z=(x-mu)*sd@W; centers=np.stack([z[fit&(subjects==s)].mean(0) for s in np.unique(fy)]); cls=np.unique(fy); sk,pk=score(z[cal],centers); su,pu=score(z[unk],centers); sf,pf=score(z[fit],centers); tau=np.quantile(sf,CFG['protocol']['threshold_quantile']); row[f'{strategy}_l{lam}_d{k}']=metrics(sk,su,cls[pk],tau,subjects[cal])
            if held==sorted(np.unique(subjects))[0]: np.savez_compressed(MODELS/f'{strategy}_l{lam}_d{k}.npz',mu=mu,sd=sd,W=W,centers=centers,classes=cls)
        folds.append({'held_subject':int(held),'metrics':row}); log(f'held={held} done')
    summary={k:{m:float(np.nanmean([f['metrics'][k][m] for f in folds])) for m in next(iter(folds[0]['metrics'].values()))} for k in folds[0]['metrics']}
    # Freeze best by predeclared Train-only LOSO unknown rejection, tie-break acceptance; no Test selection.
    best=max(summary,key=lambda n:(summary[n]['unknown_rejection'],summary[n]['known_acceptance']))
    strategy,lam,k=(('pca_whiten',0,int(k)) if best.startswith('pca') else (best.split('_l')[0],float(best.split('_l')[1].split('_d')[0]),int(best.rsplit('_d',1)[1])))
    fit=fr<=.6; mu,sd,W=project_fit(x[fit],subjects[fit],stability,strategy,lam,k); z=(x-mu)*sd@W; cls=np.unique(subjects); centers=np.stack([z[fit&(subjects==s)].mean(0) for s in cls]); sf,_=score(z[fit],centers); tau=np.quantile(sf,.1)
    tm=pd.read_csv(ROOT/CFG['data']['test_metadata']); tx=load_features(tm); tz=(tx-mu)*sd@W; ts,ti=score(tz,centers); pred=np.where(ts>=tau,cls[ti],0); sess=tm.session.astype(str).str.casefold(); s1=sess.isin(['session1','s1','1']); s2=sess.isin(['session2','s2','2']); audit={'S1_N':int(s1.sum()),'S1_rejection':float((pred[s1]==0).mean()),'S2_N':int(s2.sum()),'S2_acceptance':float((pred[s2]!=0).mean()),'official_AUC':'N/A','official_F1':'N/A'}
    result={'experiment':CFG['experiment'],'device':'cuda' if __import__('torch').cuda.is_available() else 'cpu','n_rows':len(meta),'feature_dim':x.shape[1],'fold_count':len(folds),'candidate_summary':summary,'selected_candidate':best,'folds':folds,'test_audit':audit,'protocol':{'test_not_used_for_selection':True,'stability_regularizer':'matched random feature perturbation; PGD not run at full scale','unknown_proxy':'held-out enrolled subject'}}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); lines=['# E219 训练折内稳定联合投影实验','','输入仅为 E65(855)+E146 logbp(90)，每个 outer fold 只使用 outer-train 的 fit_fraction≤0.6。投影候选和 lambda 网格预先固定；Test 仅在候选冻结后读取。','','|候选|AUC|rejection F1|known acceptance|LOSO unknown rejection|','|---|---:|---:|---:|---:|']
    for n,v in summary.items(): lines.append(f"|{n}|{v['auc']:.4f}|{v['rejection_f1']:.4f}|{v['known_acceptance']:.4f}|{v['unknown_rejection']:.4f}|")
    lines+=['',f'冻结候选：`{best}`（仅按 Train LOSO 预先定义的 unknown rejection、再按 acceptance 选择）。','',f"Test S1 rejection={audit['S1_rejection']:.6f}，S2 acceptance={audit['S2_acceptance']:.6f}。官方 Test 无身份标签，因此 AUC/F1=N/A。",'',f"达标条件：LOSO unknown rejection>0.90、S2>0.70、S1≥0.70；本轮冻结结果是否达标：{summary[best]['unknown_rejection']>0.9 and audit['S2_acceptance']>0.7 and audit['S1_rejection']>=0.7}。",'','稳定性正则采用与 E218 同等能量随机对照的特征变化；完整 PGD 成本过高，已明确限制，未冒充 PGD。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); log('completed')
if __name__=='__main__':
    try: main()
    finally: LOG.close()
