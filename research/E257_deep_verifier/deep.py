"""Train-only E257: a shared residual verifier over four E254 prototype matches."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
CACHE = OUT.parents[1] / 'BrainPrint_cache'
DEVICE = 'cuda'
TASKS = ['EC', 'EO', 'Imagery', 'SSVEP']
SEED = 257


def space(x, t, y):
    mu = torch.stack([x[t == j].mean(0) for j in range(4)])
    sd = torch.stack([x[t == j].std(0).clamp_min(1e-6) for j in range(4)])
    z = (x - mu[t]) / sd[t]
    pm = z.mean(0)
    z = z - pm
    _, v = torch.linalg.eigh(z.T @ z / (len(z)-1))
    basis = v[:, -256:].flip([1]).contiguous()
    q = z @ basis
    residual = torch.cat([a - a.mean(0) for i in range(19) for a in [q[y == i]]])
    c = residual.T @ residual / (len(q)-19)
    d, v = torch.linalg.eigh(.95*c + .05*c.diag().mean()*torch.eye(256, device=DEVICE))
    w = (v * d.clamp_min(1e-6).rsqrt()) @ v.T
    return dict(mu=mu, sd=sd, pm=pm, basis=basis, w=w)


def project(x, t, mu, sd, pm, basis, w):
    return (((x-mu[t])/sd[t]-pm) @ basis) @ w


def templates(q, t, y, order):
    p = q.new_empty(4, 19, 4, 256)
    for j in range(4):
        for i in range(19):
            ix = ((t == j) & (y == i)).nonzero().flatten()
            ix = ix[order[ix].argsort(stable=True)]
            n = len(ix)
            assert n >= 4
            for k in range(4):
                p[j,i,k] = q[ix[k*n//4:(k+1)*n//4]].mean(0)
    return F.normalize(p, dim=-1)


def matches(q, t, p):
    out = q.new_empty(len(q), 19, 4)
    for j in range(4):
        m = t == j
        out[m] = (F.normalize(q[m]) @ p[j].flatten(0,1).T).reshape(-1,19,4)
    return out.sort(descending=True).values


class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,16), nn.SiLU(), nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x[...,0] + self.net(x).squeeze(-1)


def train(q,t,y,order):
    # Alternate within fit groups; a training query never contributes to its template.
    parity = torch.empty_like(y)
    for j in range(4):
        for i in range(19):
            ix = ((t==j)&(y==i)).nonzero().flatten()
            ix = ix[order[ix].argsort(stable=True)]
            parity[ix] = torch.arange(len(ix),device=DEVICE)%2
    a = q.new_empty(len(q),19,4)
    for k in (0,1):
        m = parity == k
        a[m] = matches(q[m],t[m],templates(q[~m],t[~m],y[~m],order[~m]))
    model = Verifier().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=.002)
    for _ in range(10):
        for ix in torch.randperm(len(y),device=DEVICE).split(256):
            logits = 12*(model(a[ix])-.33)
            positive = F.one_hot(y[ix],19).bool()
            loss = .5*(F.softplus(-logits[positive]).mean()+F.softplus(logits[~positive]).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def gate(s):
    v = s.topk(2).values
    return (v[:,0]+.5*(v[:,0]-v[:,1])).cpu().numpy()


def metrics(cal, ev, truth, ids):
    tau95,tau90 = np.quantile(gate(cal),[.05,.10])
    g = gate(ev); k = truth != 0; pred = ids[ev.argmax(1).cpu().numpy()]; reject = g < tau90
    correct = pred == truth
    # Highest empirical threshold attaining at least the target late known TPR.
    known_desc = np.sort(g[k])[::-1]
    def roc_rejection(percent):
        rank = (percent * len(known_desc) + 99) // 100
        return (g[~k] < known_desc[rank-1]).mean()
    return dict(auc=roc_auc_score(k,g), u95=roc_rejection(95), rej90=roc_rejection(90),
                frozen_u95=(g[~k]<tau95).mean(), frozen_rej90=reject[~k].mean(),
                known_accept90=(~reject[k]).mean(), known_accept95=(g[k]>=tau95).mean(),
                macro_f1=f1_score(truth,np.where(reject,0,pred),average='macro'),
                closed_f1=f1_score(truth[k],pred[k],average='macro'),
                reject_correct=int((reject & correct & k).sum()), misrank=int((~correct & k).sum()),
                accepted_misrank=int((~reject & ~correct & k).sum()), tau90=tau90,tau95=tau95)


def main():
    meta = pd.read_csv(CACHE/'train_metadata.csv')
    assert set(meta.session)=={'session1'} and set(meta.subject)==set(range(1,21))
    raw = np.load(CACHE/'e253_subwindow/train_feat.npy')
    assert raw.shape==(len(meta),6,855)
    x = torch.from_numpy(raw).to(DEVICE,torch.float32).mean(1)
    cv = lambda a: torch.as_tensor(np.asarray(a),device=DEVICE)
    t = cv([TASKS.index(s) for s in meta.task]); s = cv(meta.subject); order=cv(meta.trial)
    frac = cv(meta.trial/meta.groupby(['subject','task']).trial.transform('max'))
    folds=[]; (OUT/'weights').mkdir(exist_ok=True)
    for h in range(1,21):
        torch.manual_seed(SEED+h)
        ids=np.array([i for i in range(1,21) if i!=h]); keep=s!=h
        f=keep & (frac<=.6); c=keep & (frac>.6) & (frac<=.8); e=frac>.8
        y=torch.searchsorted(cv(ids),s[f]); tf=t[f]; of=order[f]
        sp=space(x[f],tf,y); q=project(x[f],tf,**sp)
        p=templates(q,tf,y,of); model=train(q,tf,y,of)
        with torch.no_grad():
            cm=matches(project(x[c],t[c],**sp),t[c],p)
            em=matches(project(x[e],t[e],**sp),t[e],p)
            truth=torch.where(keep[e],s[e],0).cpu().numpy()
            row=dict(held=h,nfit=len(q),ncal=len(cm),known=int((truth!=0).sum()),unknown=int((truth==0).sum()))
            for name,fn in [('E254',lambda a:a[...,0]),('E257',model)]:
                row[name]=metrics(fn(cm),fn(em),truth,ids)
            state=dict(**sp, prototypes=p, ids=cv(ids), tasks=TASKS, model=model.state_dict(), held=h,
                       thresholds={n:[float(row[n][a]) for a in ('tau95','tau90')] for n in ('E254','E257')})
            torch.save(state,OUT/'weights'/f'held{h:02d}.pt')
        folds.append(row); print(h,row['E257']['auc'],flush=True)
    summary={}
    for n in ('E254','E257'):
        summary[n]={}
        for key in folds[0][n]:
            if key.startswith('tau'): continue
            a=np.array([r[n][key] for r in folds]); error=key in ('reject_correct','misrank','accepted_misrank')
            worst=a.max() if error else a.min()
            summary[n][key]=dict(mean=float(a.mean()),worst=float(worst),held=[int(i+1) for i in np.flatnonzero(a==worst)])
            if error: summary[n][key]['total']=int(a.sum())
    (OUT/'results.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),seed=SEED,folds=folds,summary=summary),indent=2),encoding='utf-8')
    lines=['# E257 共享逐身份验证器：Train-only LOSO','', '20 折；按身份×任务 trial/max 切分。19 个身份前 60% 拟合，中间 20% 校准；所有身份末 20% 评估。', '固定 E254：PCA256、类内白化系数 0.95、每任务每身份 4 原型；两支共用 gate=s1+0.5(s1−s2)。E257 以排序后的四个余弦匹配分训练共享 4→16→1 残差 MLP。', 'u95/rej90 为逐折晚期经验 ROC 上已知接受率首次达到或超过 95%/90% 时的未知拒识率，不插值、同分一起接受，不将此阈值用于预测。frozen_u95/frozen_rej90 保留原中段阈值在晚期的拒识率。macro_f1 为中段冻结 90% 工作点的开放集 20 类宏 F1。拒对是已知 top1 正确但被拒，错排是门前已知 top1 错。','', '|指标（均值 / 最差）|E254|E257|','|---|---:|---:|']
    for key in summary['E254']:
        lines.append('|'+key+'|'+ '|'.join('{mean:.5f} / {worst:.5f}'.format(**summary[n][key]) for n in summary)+'|')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
