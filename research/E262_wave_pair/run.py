"""Train-only paired waveform views; freeze recipe before reading late scores."""
import sys,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'E259_cohort_enroll'))
import run as ref


def train(q,a,fit,held,paired):
    torch.manual_seed(262+held)
    z=F.normalize(q[fit]); az=F.normalize(a[fit]); labels=ref.y[fit]; tasks=ref.t[fit]
    people=labels.unique(); targets=torch.searchsorted(people,labels)
    centers=F.normalize(torch.stack([torch.stack([z[(labels==k)&(tasks==t)].mean(0) for k in people]) for t in range(4)]),dim=-1)
    net=nn.Sequential(nn.Linear(256,96),nn.GELU(),nn.Linear(96,256)).to(q.device)
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    for _ in range(160):
        ix=torch.randint(len(z),(256,),device=q.device); clean=z[ix]; other=(az if paired else z)[ix]
        def loss(x):
            out=F.normalize(x+net(x)); logits=torch.einsum('bd,bkd->bk',out,centers[tasks[ix]])*24
            return F.cross_entropy(logits,targets[ix])+8*(1-(out*clean).sum(1)).mean()
        obj=(loss(clean)+loss(other))*.5; opt.zero_grad(); obj.backward(); opt.step()
    return net.eval()


def metrics(scores,pred,truth,held,tau):
    k=truth!=held; ok=pred==truth; accept=scores>=tau; sortedk=np.sort(scores[k])[::-1]
    def reject_at(r):return float((scores[~k]<sortedk[(int(round(100*r))*len(sortedk)+99)//100-1]).mean())
    return dict(AUC=float(ref.roc_auc_score(k,scores)),U95=reject_at(.95),R90=reject_at(.90),F1=float(ref.f1_score(truth[k],pred[k],average='macro')),old_e2e=float((ok&accept)[k].mean()),misrank=float((~ok)[k].mean()),reject_correct=float((ok&~accept)[k].mean()),unknown_reject=float((~accept)[~k].mean()))


def main():
    cache=ref.CACHE/'e262_wave_pair'; data=[ref.tens(np.load(cache/(name+'.npy'))) for name in ['train_aug','stress']]; folds=[]
    weights=H/'weights'; weights.mkdir(exist_ok=True)
    for h in ref.ids:
        fit=(ref.f<=.6)&(ref.y!=h); cal=(ref.f>.6)&(ref.f<=.8)&(ref.y!=h); late=ref.f>.8; people=[p for p in ref.ids if p!=h]
        q,art=ref.project(fit)
        def proj(x):return (((x-art['mu'][ref.t])/art['sd'][ref.t]-art['center'])@art['basis'])@art['white']
        a,s=map(proj,data); row={}; bundle={'projection':art,'people':people,'tasks':sorted(ref.M.task.unique().tolist())}
        for rule in ['E254','clean_net','paired_net']:
            net=None if rule=='E254' else train(q,a,fit,h,rule=='paired_net')
            with torch.no_grad():
                def embed(x):
                    if net is None:return x
                    # Preserve E254 trial magnitude when averaging registration templates.
                    z=F.normalize(x); return F.normalize(z+net(z))*x.norm(dim=1,keepdim=True)
                enc=embed(q); p,v=ref.templates(enc,fit,people); cs,_=ref.score(enc,p,v,cal,people); tau=float(np.quantile(cs[0],.10)); bundle[rule]=dict(net=None if net is None else net.state_dict(),p=p,v=v,tau=tau)
                for view,query in [('clean',enc),('stress',embed(s))]:
                    ss,pred=ref.score(query,p,v,late,people); row[rule+'_'+view]=metrics(ss[0],pred,ref.y[late].cpu().numpy(),h,tau)
        torch.save(bundle,weights/f'held{h:02}.pt'); folds.append(row);print('held',h,flush=True)
    summary={arm:{m:[float(np.mean(vals)),float(max(vals) if m in ('misrank','reject_correct') else min(vals))] for m in folds[0][arm] for vals in [[r[arm][m] for r in folds]]} for arm in folds[0]}
    (H/'results.json').write_text(json.dumps(dict(folds=folds,summary=summary),indent=2)); print(json.dumps(summary))

if __name__=='__main__':main()
