from __future__ import annotations
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text())
DEVICE = 'cuda'
TASKS = CFG['tasks']

def log(s):
    print(s, flush=True)
    with (OUT / 'run.log').open('a', encoding='utf8') as f: f.write(str(s) + '\n')

def historical_functions(path, names, env):
    tree = ast.parse(path.read_text(encoding='utf8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(tree, str(path), 'exec'), env)
    return env

base = historical_functions(ROOT / 'research/E215_strict_loso_e170_rebuild/run_e215.py', ['km', 'prototypes', 'scores', 'nearest_identity'], dict(np=np, torch=torch, DEVICE=DEVICE, TASKS=TASKS))
oldcfg = json.loads((ROOT / 'research/E219_stable_projection/config.json').read_text())
old = historical_functions(ROOT / 'research/E219_stable_projection/run_e219.py', ['load_features'], dict(np=np, loadmat=loadmat, ROOT=ROOT, CFG=oldcfg, log=log))

def tensor(x): return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
def frac(m): return m.trial.to_numpy(float) / m.groupby(['subject', 'task']).trial.transform('max').to_numpy(float).clip(1)
def fit_model(kind, arrays, tasks, y, fit):
    state = {'kind': kind, 'branches': []}
    streams = [arrays['fisher']] if kind == 'fisher' else ([arrays['e65'], arrays['logbp']] if kind == 'fusion' else [arrays['e65']])
    for raw in streams:
        x = tensor(raw); z = torch.empty_like(x); mu = []; sd = []
        groups = ['all'] if kind == 'fisher' else TASKS
        for t in groups:
            rows = np.ones(len(y), bool) if t == 'all' else tasks == t
            a = x[fit & rows]
            if kind in ['norm', 'fusion']:
                m = torch.quantile(a, .5, dim=0); s = (torch.quantile((a-m).abs(), .5, dim=0)*1.4826).clamp_min(1e-4)
            else:
                m = a.mean(0); s = a.std(0, correction=0) + (1e-5 if kind == 'fisher' else 1e-6)
            z[rows] = (x[rows]-m)/s; mu.append(m.cpu().numpy()); sd.append(s.cpu().numpy())
        W = None; pm = None
        if kind in ['baseline', 'fisher']:
            a = z[fit].double(); pm = a.mean(0)
            if kind == 'baseline':
                _, v = torch.linalg.eigh((a-pm).T @ (a-pm)); W = v[:, -96:].flip(1)
            else:
                cls = np.unique(y[fit]); means = torch.stack([a[y[fit] == c].mean(0) for c in cls]); b = means-means.mean(0); B = b.T@b/len(cls)
                within = torch.zeros_like(B)
                for c in cls:
                    q = a[y[fit] == c]; q = q-q.mean(0); within += q.T@q/len(q)
                # The historical random stability surrogate is recomputed on fit rows only.
                noise = np.random.default_rng(oldcfg['seed']).normal(size=(fit.sum(), raw.shape[1]))
                stability = tensor(np.maximum(np.mean(np.abs(noise*.01), axis=0), 1e-4)).double()
                A = torch.linalg.pinv(within/len(cls)+100*torch.diag(stability))@B
                _, v = torch.linalg.eigh((A+A.T)/2); W = v[:, -64:].flip(1); pm = torch.zeros_like(pm)
            z = ((z.double()-pm)@W).float()
        if kind == 'norm': z = (z-z.mean(1, keepdim=True))/(z.std(1, correction=0, keepdim=True)+1e-5)
        z = z.cpu().numpy(); cls = np.unique(y[fit])
        if kind == 'baseline': p, keys = base['prototypes'](z, y, tasks, fit, cls, 12)
        elif kind == 'fisher': p = np.stack([z[fit & (y==c)].mean(0) for c in cls]); keys = [('all', int(c)) for c in cls]
        else:
            ps = []; keys = []
            for t in TASKS:
                for c in cls:
                    ix = np.flatnonzero(fit & (tasks==t) & (y==c)); center = z[ix].mean(0)
                    if kind == 'norm': centers = [center]
                    else:
                        order = ix[np.argsort(np.linalg.norm(z[ix]-center, axis=1))]
                        centers = z[order[np.linspace(0, len(order)-1, min(3,len(order))).astype(int)]]
                    ps.extend(centers); keys.extend([(t,int(c))]*len(centers))
            p = np.asarray(ps)
        state['branches'].append(dict(mu=mu, sd=sd, W=None if W is None else W.cpu().numpy(), pm=None if pm is None else pm.cpu().numpy(), p=p, keys=keys))
    return state

def score_model(state, arrays, tasks):
    kind = state['kind']; streams = [arrays['fisher']] if kind=='fisher' else ([arrays['e65'],arrays['logbp']] if kind=='fusion' else [arrays['e65']]); scores=[]; preds=[]
    for raw, st in zip(streams,state['branches']):
        x=tensor(raw); z=torch.empty_like(x)
        for i,t in enumerate(['all'] if kind=='fisher' else TASKS):
            rows=np.ones(len(x),bool) if t=='all' else tasks==t; z[rows]=(x[rows]-tensor(st['mu'][i]))/tensor(st['sd'][i])
        if st['W'] is not None: z=((z.double()-torch.as_tensor(st['pm'],device=DEVICE))@torch.as_tensor(st['W'],device=DEVICE)).float()
        if kind=='norm': z=(z-z.mean(1,keepdim=True))/(z.std(1,correction=0,keepdim=True)+1e-5)
        s=np.empty(len(x)); pred=np.empty(len(x),int); keys=st['keys']; p=tensor(st['p'])
        for t in (['all'] if kind=='fisher' else TASKS):
            rows=np.flatnonzero(np.ones(len(x),bool) if t=='all' else tasks==t); cols=[i for i,k in enumerate(keys) if k[0]==t]
            d=torch.cdist(z[rows],p[cols]); v,ix=d.min(1); v=v.cpu().numpy(); ix=ix.cpu().numpy()
            s[rows]=-v if kind=='baseline' else -v**2/(1 if kind=='fisher' else z.shape[1]); pred[rows]=[keys[cols[i]][1] for i in ix]
        scores.append(s); preds.append(pred)
    return np.stack(scores,1),preds[0]

def fuse(s,tasks,ref,rtasks):
    out=np.empty(len(s))
    for t in TASKS:
        a=s[tasks==t]; r=ref[rtasks==t]; vals=[]
        for j in range(2):
            q=np.sort(r[:,j]); vals.append((np.searchsorted(q,a[:,j],'left')+np.searchsorted(q,a[:,j],'right'))/(2*len(q)))
        out[tasks==t]=.25*vals[0]+.75*vals[1]
    return out

def train(kind,arrays,tasks,y,fr,fit):
    blocks=np.minimum((fr*5).astype(int),4); oof=np.full((len(y),2 if kind=='fusion' else 1),np.nan)
    for b in np.unique(blocks[fit]):
        hold=fit & (blocks==b); inner=fit & ~hold
        assert set(y[inner])==set(y[fit])
        st=fit_model(kind,arrays,tasks,y,inner)
        oof[hold]=score_model(st,{k:v[hold] for k,v in arrays.items()},tasks[hold])[0]
    ref=oof[fit]; rt=tasks[fit]
    os=fuse(ref,rt,ref,rt) if kind=='fusion' else ref[:,0]
    tau=float(np.quantile(os,.1)); st=fit_model(kind,arrays,tasks,y,fit)
    st.update(tau=tau,ref=ref,ref_tasks=rt,fit_rows=np.flatnonzero(fit),oof_scores=os)
    return st

def evaluate(s,p,y,known,unknown,tau):
    k=s[known]; u=s[unknown]; tp=(u<tau).sum(); fp=(k<tau).sum(); fn=(u>=tau).sum()
    return dict(auc=float(((u[:,None]<k).sum()+.5*(u[:,None]==k).sum())/(len(k)*len(u))),rejection_f1=float(2*tp/(2*tp+fp+fn)),known_acceptance=float((k>=tau).mean()),unknown_rejection=float((u<tau).mean()),identity_correct_but_rejected=float(((p[known]==y[known])&(k<tau)).mean()),threshold=tau)

def main():
    assert torch.cuda.is_available(); log(torch.cuda.get_device_name())
    m=pd.read_csv(ROOT/CFG['data']['metadata']); tasks=m.task.to_numpy(); y=m.subject.to_numpy(int); fr=frac(m)
    assert set(tasks)==set(TASKS) and set(y)==set(range(1,21))
    raw=np.load(ROOT/'research/E219_stable_projection/cache/raw_features.npz', allow_pickle=True)
    assert np.array_equal(raw['subject'],y) and np.array_equal(raw['task'],tasks) and np.allclose(raw['frac'],fr)
    arrays=dict(e65=np.load(ROOT/CFG['features']['e65']),logbp=np.load(ROOT/CFG['features']['logbp']),fisher=raw['features'])
    assert all(len(a)==len(m) for a in arrays.values())
    kinds=['baseline','fisher','norm','fusion']; folds=[]
    badblocks=np.arange(len(y))*5//len(y)
    diagnostic={str(b):sorted(int(v) for v in set(y[(fr<=.6)&(badblocks==b)])-set(y[(fr<=.6)&(badblocks!=b)])) for b in range(5)}
    for held in range(1,21):
        path=OUT/'cache'/f'fold_{held:02d}.json'
        if path.exists(): folds.append(json.loads(path.read_text())); continue
        fit=(y!=held)&(fr<=.6); window=(fr>.6)&(fr<=.8); known=window&(y!=held); unknown=window&(y==held); row={}
        for kind in kinds:
            st=train(kind,arrays,tasks,y,fr,fit); ss,p=score_model(st,arrays,tasks); s=fuse(ss,tasks,st['ref'],st['ref_tasks']) if kind=='fusion' else ss[:,0]
            row[kind]=evaluate(s,p,y,known,unknown,st['tau']); torch.save(st,OUT/'models'/f'{kind}_fold_{held:02d}.pt')
            np.savez_compressed(OUT/'cache'/f'{kind}_fold_{held:02d}_scores.npz',score=s,pred=p,known=known,unknown=unknown,subject=y)
            log(f'held={held} {kind}: {row[kind]}')
        fold=dict(held_subject=held,metrics=row); folds.append(fold); path.write_text(json.dumps(fold,indent=2))
    avg={k:{metric:float(np.mean([f['metrics'][k][metric] for f in folds])) for metric in folds[0]['metrics'][k]} for k in kinds}
    best=max(kinds,key=lambda k:(avg[k]['unknown_rejection'],avg[k]['known_acceptance']))
    frozen={k:torch.load(OUT/'models'/f'{k}_final.pt', weights_only=False) if (OUT/'models'/f'{k}_final.pt').exists() else train(k,arrays,tasks,y,fr,fr<=.6) for k in kinds}
    for k,st in frozen.items(): torch.save(st,OUT/'models'/f'{k}_final.pt')
    (OUT/'train_selection.json').write_text(json.dumps(dict(candidate_summary=avg,selected_candidate=best),indent=2))
    tm=pd.read_csv(ROOT/CFG['data']['test_metadata']); tt=tm.task.to_numpy(); test=dict(e65=np.load(ROOT/CFG['features']['e65_test']),logbp=np.load(ROOT/CFG['features']['logbp_test']))
    xp=OUT/'cache'/'test_fisher.npy'
    if not xp.exists(): np.save(xp,old['load_features'](tm))
    test['fisher']=np.load(xp); audit={}
    for k,st in frozen.items():
        ss,p=score_model(st,test,tt); s=fuse(ss,tt,st['ref'],st['ref_tasks']) if k=='fusion' else ss[:,0]; pred=np.where(s>=st['tau'],p,0)
        sess=tm.session.astype(str).str.casefold(); s1=sess.isin(['session1','s1','1']); s2=sess.isin(['session2','s2','2'])
        audit[k]=dict(S1_N=int(s1.sum()),S1_rejection=float((pred[s1]==0).mean()),S2_N=int(s2.sum()),S2_acceptance=float((pred[s2]!=0).mean()),official_AUC='N/A',official_F1='N/A')
        np.savez_compressed(OUT/'cache'/f'{k}_test_scores.npz',score=s,pred=pred)
    result=dict(experiment=CFG['experiment'],device=torch.cuda.get_device_name(),fold_count=len(folds),candidate_summary=avg,selected_candidate=best,folds=folds,test_audit=audit,old_blocks_missing_enrolled_subjects=diagnostic)
    log(json.dumps(dict(candidate_summary=avg,test_audit=audit,missing_subjects=diagnostic),indent=2))
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# E221 指标与校准修正审计','','统一：高分=known；阈值=known inner OOF 的 0.1 分位数；score<阈值拒识；unknown=1 计算 AUC(-score) 和拒识 F1。完整20折，训练≤0.6，评估(0.6,0.8]。内层按每个subject/task的trial分数分块，归一化/投影/原型仅用该内层fit。冻结Train选择后才读取Test。','', '|候选|AUC|拒识F1|unknown拒识|known接受|Test S1拒识|Test S2接受|','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in avg.items(): lines.append(f"|{k}|{v['auc']:.6f}|{v['rejection_f1']:.6f}|{v['unknown_rejection']:.6f}|{v['known_acceptance']:.6f}|{audit[k]['S1_rejection']:.6f}|{audit[k]['S2_acceptance']:.6f}|")
    lines += ['',f'仅Train选择：{best}。Test无真实身份标签，官方AUC/F1不可计算；S1/S2只是会话分层行为指标，不等同官方未知/正常分类指标。','','## 已确认根因','- E220距离方向与0.9分位本来一致；仅翻转符号不能修复退化。','- E220全局连续行分块会移除完整注册身份，所谓known OOF变成未知身份分数，阈值偏宽。实际缺失身份见results.json。','- E220最终inner_oof收到候选字典而非kind字符串，融合分支未执行：原始距离阈值用于[0,1]融合分数，造成全接受。','- E220排名依赖整个查询集合，含outer-held/Test；现改成每任务固定known OOF参考ECDF，Test不改变尺度。','- E219的rejection_f1实际上是known-positive接受F1，不能与E220的unknown-positive拒识F1直接比较；其原阈值来自in-sample分数，本轮改OOF。','','## 保持方法与限制','baseline复用E215的E65任务标准化、PCA96、k12原型；fisher冻结E219 lambda100/d64公式和原始缓存特征；norm保留E220逐query标准化与k1原型；fusion冻结E220 w025、k3原型。没有搜索新方法或Test选参。Fisher随机稳定性代理改在fit行内计算，非PGD。缓存Fisher实际855维，以实际shape为准，旧报告945维说法不准确。模型与OOF/评估分数已保存。ECDF参考和分位估计同取known OOF，是训练内经验校准，非独立校准集；LOSO指标为逐折宏平均，F1受19:1样本比例影响。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); log('completed')

if __name__=='__main__': main()
