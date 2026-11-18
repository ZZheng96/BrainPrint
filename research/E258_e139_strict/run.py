"""Strict E139 replication; only the six named Train files are inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json

OUT = Path(__file__).resolve().parent
CACHE = OUT.parents[1] / 'BrainPrint_cache'
VIEWS = ['e65_correlation_spd/train_correlation_logspd.npy',
         *[f'e62_multiwindow/train_multiwindow_{v}_dinov2.npy' for v in ('cov8','corr8','cov4')],
         'e21_scale_invariant/train_cov_detrend_zband.npy']
TASKS = ['EC','EO','Imagery','SSVEP']
D = 'cuda' if torch.cuda.is_available() else 'cpu'


def centers(x, k, s):
    if len(x) <= k:
        return x
    c = x[np.random.default_rng(17*s).choice(len(x), k, replace=False)]
    for _ in range(20):
        a = torch.cdist(x,c).argmin(1)
        w = (a[None,:] == torch.arange(k, device=D)[:,None]).float()
        n = w.sum(1,keepdim=True)
        new = torch.where(n>0,(w@x)/n.clamp_min(1),c)
        if torch.allclose(new,c,atol=1e-6):
            break
        c = new
    return c


def run():
    torch.set_float32_matmul_precision('highest')
    modeldir = OUT/'models'; modeldir.mkdir(exist_ok=True)
    m = pd.read_csv(CACHE/'train_metadata.csv'); s = m.subject.to_numpy(); ids = sorted(map(int,np.unique(s)))
    t = pd.Categorical(m.task,categories=TASKS).codes
    assert ids == list(range(1,21)) and min(t)>=0
    f = (m.trial/m.groupby(['subject','task']).trial.transform('max')).to_numpy()
    ev = f>.8
    xs = [torch.from_numpy(np.load(CACHE/p)).to(D,dtype=torch.float32) for p in VIEWS]
    assert all(len(x)==len(m) and torch.isfinite(x).all() for x in xs)
    rows=[]
    for name,nv in [('C_plus_cov',5),('B_logspd_multiwin',4)]:
        raw=torch.cat(xs[:nv],1)
        for h in ids:
            fit=(s!=h)&(f<=.6); cal=(s!=h)&(f>.6)&(f<=.8)
            mu=torch.stack([raw[fit&(t==j)].mean(0) for j in range(4)])
            sd=torch.stack([raw[fit&(t==j)].std(0,unbiased=False)+1e-6 for j in range(4)])
            z=(raw-mu[t.astype(int)])/sd[t.astype(int)]
            a=z[fit].double(); pm=a.mean(0); a-=pm
            _,v=torch.linalg.eigh(a.T@a); v=v[:,-96:].flip([1]).contiguous(); q=((z.double()-pm)@v).float()
            state=dict(views=VIEWS[:nv],tasks=TASKS,held=h,mu=mu.cpu(),sd=sd.cpu(),pca_mean=pm.cpu(),basis=v.cpu(),rules={})
            for k in (32,12):
                score=np.full(len(s),np.nan); pred=np.zeros(len(s),int); protos=[]; labels=[]
                for j in range(4):
                    cs=[centers(q[fit&(t==j)&(s==i)],k,i) for i in ids if i!=h]; p=torch.cat(cs)
                    lab=np.repeat([i for i in ids if i!=h],[len(c) for c in cs]); idx=(cal|ev)&(t==j)
                    d,ix=torch.cdist(q[idx],p).min(1); score[idx]=-d.cpu().numpy(); pred[idx]=lab[ix.cpu().numpy()]; protos.append(p.cpu()); labels.append(torch.tensor(lab))
                tau95,tau90=np.quantile(score[cal],[.05,.1]); known=ev&(s!=h); unk=ev&(s==h); y=unk[ev]; late=score[ev]; ranks=pd.Series(late).rank().to_numpy(); nk=known.sum(); nu=unk.sum()
                auc=(ranks[~y].sum()-nk*(nk+1)/2)/(nk*nu); rejects=late<tau90; tp=(y&rejects).sum(); yp=np.where(rejects,0,pred[ev]); yt=np.where(y,0,s[ev]); macro=np.mean([2*((yt==i)&(yp==i)).sum()/((yt==i).sum()+(yp==i).sum()) for i in [0]+[i for i in ids if i!=h]])
                rows.append(dict(view=name,k=k,held=h,nfit=fit.sum(),ncal=cal.sum(),nk=nk,nu=nu,tau95=tau95,tau90=tau90,AUROC=auc,U95=(score[unk]<tau95).mean(),rej90=rejects[y].mean(),accept90=(~rejects[~y]).mean(),F1_unknown=2*tp/(y.sum()+rejects.sum()),F1_macro=macro))
                state['rules'][k]=dict(centers=protos,labels=labels,tau95=float(tau95),tau90=float(tau90))
            torch.save(state,modeldir/f'{name}_{h:02}.pt'); print(name,h,flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'folds.csv',index=False)
    keys=['AUROC','U95','rej90','F1_unknown','F1_macro','accept90']; summary=df.groupby(['view','k'])[keys].agg(['mean','min']); summary.to_csv(OUT/'summary.csv'); print(summary.to_string())

if __name__=='__main__':
    run()
