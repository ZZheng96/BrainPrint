"""E271 Train-only paired evaluation. Invoke with .venv/Scripts/python -B."""
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import normalize

H=Path(__file__).resolve().parent; R=H.parents[1]; C=R/'BrainPrint_cache'
m=pd.read_csv(C/'train_metadata.csv'); g=m.groupby(['subject','task'])
assert len(m)==11304 and m.path.is_unique and m.path.str.startswith('Train/').all() and g.trial.apply(lambda v:v.is_unique and v.is_monotonic_increasing).all()
Y=m.subject.to_numpy(); T=pd.Categorical(m.task,categories=['EC','EO','Imagery','SSVEP']).codes.astype(int); assert T.min()==0
frac=(m.trial/g.trial.transform('max')).to_numpy(); block=np.minimum((5*frac).astype(int),4); allrows=np.arange(len(m))
X=[torch.from_numpy(np.load(p)).float().cuda() for p in (R/'research/E171_multiscale_fusion/train_subwin_mean_n6.npy', C/'e262_wave_pair/clean.npy')]; assert all(x.shape==(11304,855) and torch.isfinite(x).all() for x in X)

def fit(i,rank):
    x=X[rank][i]; t=T[i]; y=Y[i]; mu=torch.stack([x[t==j].mean(0) for j in range(4)]); sd=torch.stack([x[t==j].std(0,unbiased=bool(rank)) for j in range(4)]); sd=sd.clamp_min(1e-6) if rank else sd+1e-6
    z=(x-mu[t])/sd[t]; pm=z.mean(0); z-=pm; dim=256 if rank else 96
    b=torch.linalg.eigh(z.T@z/(len(i)-1))[1][:,-dim:].flip([1]).contiguous(); q=z@b
    a=dict(task_mu=mu,task_sd=sd,pca_mean=pm,pca_basis=b)
    if rank:
        r=torch.cat([q[y==k]-q[y==k].mean(0) for k in range(1,21)]); cov=r.T@r/(len(i)-20); v,e=torch.linalg.eigh(.95*cov+.05*cov.diag().mean()*torch.eye(dim,device='cuda')); w=(e*v.clamp_min(1e-6).rsqrt())@e.T; a['whiten']=w; q=q@w
    def proto(j,k):
        u=q[(t==j)&(y==k)]; n=len(u); assert n>=4
        return torch.stack([u[l*n//4:(l+1)*n//4].mean(0) for l in range(4)]) if rank else u.mean(0)
    a['prototypes']=torch.stack([torch.stack([proto(j,k) for k in range(1,21)]) for j in range(4)]); return a

def score(a,i,rank):
    s=torch.empty(len(i),20,device='cuda'); t=T[i]
    for j in range(4):
        mask=t==j; z=(X[rank][i[mask]]-a['task_mu'][j])/a['task_sd'][j]; q=(z-a['pca_mean'])@a['pca_basis']; p=a['prototypes'][j]
        if rank:
            w=a['whiten']; s[mask]=(normalize(q@w)@normalize(p.flatten(0,1)@w).T).reshape(-1,20,4).amax(2)
        else: s[mask]=-torch.cdist(q,p)
    v,k=s.max(1); return -v.cpu().numpy(),k.cpu().numpy()+1

def measure(y,p):
    return dict(C=float(np.mean(y==p)),macro_F1=float(np.mean([2*np.sum((y==k)&(p==k))/(np.sum(y==k)+np.sum(p==k)) for k in range(1,21)])))

def summarize(y,d,b,r,tau):
    ok=d<tau
    return dict(n=len(y),accepted=int(ok.sum()),fixed=int(np.sum(ok&(b!=y)&(r==y))),broken=int(np.sum(ok&(b==y)&(r!=y))),**{key:dict(**measure(y,np.where(ok,p,0)),closed=measure(y,p)) for key,p in [('final',b),('E271',r)]})

@torch.inference_mode()
def main():
    result=dict(device=torch.cuda.get_device_name(),python=sys.executable,torch=str(torch.__version__),unknown_AUC=None); fits={}; acc=[]
    def evaluate(name,i,v):
        if i is None:
            a,b=[torch.load(R/s/'team_hymn/weights.pt',map_location='cuda',weights_only=True) for s in ('final','final2')]; tau=a['tau_dist']
        else:
            assert not np.intersect1d(i,v).size; gg=m.iloc[i].groupby(['subject','task']); inner=(5*gg.cumcount()/gg.trial.transform('size')).to_numpy().astype(int); ds=np.empty(len(i))
            for j in range(5): ds[inner==j]=score(fit(i[inner!=j],0),i[inner==j],0)[0]
            tau=float(np.quantile(ds,.90)); a=fit(i,0); b=fit(i,1); fits[name]=dict(gate=a,rank=b,tau=tau,train=i.tolist(),valid=v.tolist(),inner=inner.tolist())
        d,p=score(a,v,0); _,r=score(b,v,1); s=summarize(Y[v],d,p,r,tau); result[name]=dict(tau=tau,**s); print(name,s,flush=True)
        if name.startswith('fold'):acc.append((v,d,p,r,np.full(len(v),tau)))
    evaluate('frozen',None,allrows)
    for j in range(5):evaluate(f'fold{j}',allrows[block!=j],allrows[block==j])
    v,d,p,r,tau=map(np.concatenate,zip(*acc)); assert np.array_equal(np.sort(v),allrows); result['OOF']=summarize(Y[v],d,p,r,tau); print('OOF',result['OOF'])
    evaluate('forward',allrows[frac<=.6],allrows[frac>.6]); torch.save(fits,H/'gates_fits.pt'); (H/'gates.json').write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__': main()
