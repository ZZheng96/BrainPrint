"""Train-only：逐查询背景归一化，比较固定 E254 打分和归一化打分。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from scipy.stats import rankdata

def roc_auc_score(y,s):
    n=y.sum(); return (rankdata(s)[y].sum()-n*(n+1)/2)/(n*(len(y)-n))

def f1_score(y,p,average):
    return np.mean([2*((y==i)&(p==i)).sum()/((y==i).sum()+(p==i).sum()) for i in np.union1d(y,p)])

HERE=Path(__file__).resolve().parent
CACHE=HERE.parents[1]/'BrainPrint_cache'
D='cuda' if torch.cuda.is_available() else 'cpu'
M=pd.read_csv(CACHE/'train_metadata.csv')
X=torch.from_numpy(np.load(CACHE/'e253_subwindow/train_feat.npy').mean(1)).to(D)
def tens(a): return torch.as_tensor(a,device=D)
y=tens(M.subject.to_numpy()); t=tens(pd.factorize(M.task,sort=True)[0]); order=tens(M.trial.to_numpy()); f=order/tens(M.groupby(['subject','task']).trial.transform('max').to_numpy())
ids=list(range(1,21))

def project(train):
    mu=torch.stack([X[train&(t==i)].mean(0) for i in range(4)]); sd=torch.stack([X[train&(t==i)].std(0).clamp_min(1e-6) for i in range(4)])
    z=(X-mu[t])/sd[t]; m=z[train].mean(0); z=z-m
    _,v=torch.linalg.eigh(z[train].T@z[train]); v=v[:,-256:]; q=z@v
    keys=y[train].unique(); r=torch.cat([q[train&(y==i)]-q[train&(y==i)].mean(0) for i in keys]); c=r.T@r/(len(r)-len(keys)); e,u=torch.linalg.eigh(.95*c+.05*c.diagonal().mean()*torch.eye(256,device=D)); w=(u*e.clamp_min(1e-6).rsqrt())@u.T
    return q@w,dict(mu=mu,sd=sd,center=m,basis=v,white=w)

def templates(q,enroll,people):
    p=q.new_zeros(4,len(people),4,256); valid=torch.zeros(p.shape[:-1],device=D,dtype=torch.bool)
    for i in range(4):
        for j,person in enumerate(people):
            ix=torch.where(enroll&(y==person)&(t==i))[0]; ix=ix[order[ix].argsort()]; k=min(len(ix),4)
            for n in range(k):
                p[i,j,n]=q[ix[n*len(ix)//k:(n+1)*len(ix)//k]].mean(0); valid[i,j,n]=True
    return normalize(p,dim=-1),valid

def score(q,p,v,rows,people):
    s=q.new_empty(int(rows.sum()),len(people)); tt=t[rows]; xx=normalize(q[rows]);
    for i in range(4): s[tt==i]=(xx[tt==i]@p[i].flatten(0,1).T).view(-1,len(people),4).masked_fill(~v[i],-torch.inf).amax(-1)
    top,idx=s.topk(2); base=top[:,0]+.5*(top[:,0]-top[:,1]); bg=s.sort().values[:,2:-2]; rel=(base-bg.mean(1))/bg.std(1).clamp_min(.03)
    return torch.stack((base,rel)).cpu().numpy(),tens(people)[idx[:,0]].cpu().numpy()

def evaluate(old,new,unknown,save=False):
    known=torch.isin(y,tens(old)); fit=known&(f<=.6); q,art=project(fit); en=fit.clone()
    for person in new:
        for i in range(4):
            ix=torch.where((y==person)&(t==i)&(f<=.6))[0]; en[ix[order[ix].argsort()][:5]]=True
        assert int((en&(y==person)).sum())==20
    people=old+new; p,v=templates(q,en,people); cal=known&(f>.6)&(f<=.8); ev=f>.8
    cs,_=score(q,p,v,cal,people); ss,pred=score(q,p,v,ev,people); truth=y[ev].cpu().numpy(); known=~np.isin(truth,unknown); correct=pred==truth; out=[]
    for ca,s in zip(cs,ss):
        tau=np.quantile(ca,.10); acc=s>=tau
        row=dict(tau=float(tau),F1=float(f1_score(truth[known],pred[known],average='macro')),old_e2e=float((correct&acc)[np.isin(truth,old)].mean()),known_rejected_correct=float((correct&~acc)[known].mean()),closed=float(correct[known].mean()))
        if new: row['new_e2e']=float((correct&acc)[np.isin(truth,new)].mean())
        if unknown: row.update(AUC=float(roc_auc_score(known,s)),U95=float((s[~known]<np.quantile(s[known],.05)).mean()),R90=float((s[~known]<np.quantile(s[known],.10)).mean()),unknown=float((~acc)[~known].mean()))
        out.append(row)
    if save: torch.save({**{k:a.cpu() for k,a in art.items()},'proto':p.cpu(),'valid':v.cpu(),'tau':torch.tensor([a['tau'] for a in out]),'tasks':sorted(M.task.unique().tolist()),'people':people},HERE/'weights.pt')
    return out

if __name__=='__main__':
    with torch.no_grad():
        folds=[evaluate([i for i in ids if i!=h],[],[h]) for h in ids]
        sim=[]
        for h in range(20):
            a=ids[h:]+ids[:h]; sim.append(evaluate(a[:14],a[14:17],a[17:]))
        evaluate(ids,[],[],True)
        def summary(rows):return [{k:{'mean':np.mean([r[j][k] for r in rows]),'min':min(r[j][k] for r in rows)} for k in rows[0][j]} for j in range(2)]
        obj={'rules':['E254','query_cohort'],'folds':folds,'sim':sim,'loso':summary(folds),'registration':summary(sim)}; (HERE/'results.json').write_text(json.dumps(obj,indent=2)); print(json.dumps(obj['loso']));print(json.dumps(obj['registration']))
