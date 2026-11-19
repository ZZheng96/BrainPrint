"""Replay E275's saved artifacts, including its shared metadata state."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

H = Path(__file__).resolve().parent
R = H.parents[1]
E = H.parent / 'E275_threshold_validation'
C = R / 'BrainPrint_cache'
G = runpy.run_path(str(R / 'final3_compat/team_hymn/src/team_hymn_geometry.py'))
SCENES = [('clean_complete','clean',None), ('stress_complete','stress',None),
          ('clean_missing_EC','clean',0), ('stress_missing_Imagery','stress',2)]


def score(a, x, tasks, missing, old_cols, single):
    out = []
    for ix in np.array_split(np.arange(len(x)), (len(x)+127)//128):
        sim = x.new_full((len(ix),23), -float('inf'))
        for k in range(4):
            use = (tasks[ix] == k) | (tasks[ix] < 0)
            if k == missing or not use.any():
                continue
            q = G['_unit'](G['_project'](a,x[ix][use],k) @ a['whiten'])
            p = a['prototypes'][k]
            v = (p.flatten(0,1) @ a['whiten']).reshape_as(p)
            if single:
                v[old_cols] = p[old_cols]
            s = (q[:,None,None,:] * G['_unit'](v)[None]).sum(-1).amax(2)
            sim[use] = torch.maximum(sim[use], s.masked_fill(~a['valid'][k], -float('inf')))
        out.append(sim.cpu().numpy())
    return np.concatenate(out)


def measure(s, truth, a):
    w = s.argmax(1)+1
    v = np.sort(s,axis=1)[:,-2:]
    margin = v[:,1] + a['beta']*(v[:,1]-v[:,0]) - np.where(w>20,float(a['tau'])-.06,float(a['tau']))
    pred = np.where(margin>=0,w,0)
    known = truth!=0
    n = int(known.sum()); u = len(w)-n
    d = {'AUC':float((rankdata(margin)[known].sum()-n*(n+1)/2)/(n*u)),
         'F1':float(np.mean([2*((truth==k)&(pred==k)).sum()/((truth==k).sum()+(pred==k).sum()) for k in np.union1d(truth,pred)])),
         'unknown_rejection':float((pred[~known]==0).mean()),'diagnostics':{}}
    for group,mask in [('old',known&(truth<=20)),('new',truth>20)]:
        correct = w[mask]==truth[mask]; reject = pred[mask]==0
        d[group+'_accuracy'] = float((correct & ~reject).mean())
        d['diagnostics'][group] = dict(n=int(mask.sum()),rank=int(correct.sum()),rank_reject=int((correct&reject).sum()),reject=int(reject.sum()))
    return d


@torch.inference_mode()
def main():
    ref = json.loads((E/'results.json').read_text(encoding='utf8'))
    paths = {'metadata':C/'train_metadata.csv', **{k:C/f'e262_wave_pair/{k}.npy' for k in ('clean','stress')}}
    for k,path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['hashes'][k]
    m = pd.read_csv(paths['metadata']); assert m.path.str.startswith('Train/').all()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = {k:torch.from_numpy(np.load(paths[k])).to(device) for k in ('clean','stress')}
    y = m.subject.to_numpy(); task = G['_task_indices'](m[['task']].to_dict('records'),len(m))
    effective = task.copy(); late = (m.trial/m.groupby(['subject','task']).trial.transform('max')>.8).to_numpy()
    results = []; max_error = 0.
    for f in ref['folds']:
        seed = f['seed']; old = np.array(f['old_ids'])-1
        a = torch.load(E/f'fits/seed{seed}_compat.pt',map_location=device,weights_only=True)
        assert all(np.array_equal(np.flatnonzero(v.cpu()),np.r_[old,20:23]) for v in a['valid'])
        truth = y.copy(); truth[np.isin(y,f['unknown_ids'])]=0
        for i,p in enumerate(f['new_ids']): truth[y==p]=21+i
        for name,feat,missing in SCENES:
            rows = np.flatnonzero(late if missing is None else late & (task!=missing))
            # E275 shallow-copies an object array then pops task from shared dicts.
            if missing is not None: effective[rows]=-1
            for method in ('repeat','single'):
                d = measure(score(a,x[feat][rows],effective[rows],missing,old,method=='single'),truth[rows],a)
                if method=='repeat':
                    errors = [abs(d[k]-f['scenarios'][name]['candidate'][k]) for k in d if k!='diagnostics']
                    max_error=max(max_error,max(errors)); assert max(errors)<1e-8,(seed,name,errors)
                results.append(dict(seed=seed,scene=name,method=method,fallback=int((effective[rows]<0).sum()),**d))
    (H/'results.json').write_text(json.dumps(dict(max_baseline_error=max_error,rows=results),indent=2),encoding='utf8')
    report(results,max_error)


def report(rows,error):
    df = pd.DataFrame(rows)
    cols=['old clean','old stress','new clean','unknown clean','四项等权','AUC','F1']
    vals={}
    for (method,seed),g in df.groupby(['method','seed']):
        g=g.set_index('scene'); c=g.loc['clean_complete']; v=[c.old_accuracy,g.loc['stress_complete','old_accuracy'],c.new_accuracy,c.unknown_rejection]
        vals[method,seed]=np.array(v+[np.mean(v),g.AUC.mean(),g.F1.mean()])
    seeds=df.seed.unique(); lines=['# E277：老模板是否需要第二次白化？','','## 四项代理及逐折比较','','两种方案均固定新人赢家阈值偏移 −0.06。四项等权为前四列均值；AUC（已知对未知，按门控余量排序）和 F1（含未知类的宏平均）为四场景等权，不是四项代理。','','|折|方案|'+'|'.join(cols)+'|','|---|---|'+'---:|'*7]
    for seed in ['三折均值',*seeds]:
        v=[np.mean([vals[m,i] for i in seeds],axis=0) if isinstance(seed,str) else vals[m,seed] for m in ('repeat','single')]
        for label,a in zip(['重复 W','老模板单 W','单 W − 重复 W'],[*v,v[1]-v[0]]):
            lines.append(f'|{seed}|{label}|'+ '|'.join(f'{z:.6f}' for z in a)+'|')
    lines+=['','## 协议与基线复现','',f'三个 E275 已注册的 seed*_compat.pt 直接推理，未拟合、注册或调参。Train 元数据与 clean/stress 缓存的 SHA-256 均匹配。三折 × 四场景逐项对照 E275 candidate 的老/新准确率、未知拒识、AUC、F1，最大绝对误差 {error:.3g}。', '', '每折 14 老 / 3 新 / 3 未知；查询为同身份同任务进度 >0.8 的晚段。老模板是已乘 W 的投影均值，新注册模板是未乘 W 的投影均值。基线对所有模板再乘 W；实验只对该折 old_ids 对应的有效列绕过此步，21–23 列新人及查询仍乘一次 W。其余有效掩码、归一化、原型最大余弦、门控和阈值相同。', '', '**场景名沿用 E275，不能全部理解为任务元数据完整。** E275 在缺失场景对浅拷贝中的字典删除 task，状态延续到后续折。为保证配对复现，两方案重放相同状态：第一折完整场景按任务打分，后两折完整场景的晚段查询均跨任务取最大值。缺失场景排除指定任务查询并禁用该任务原型。stress 仅为模拟扰动，不代表真实跨设备。','','## 排名与误拒分解','','下表按折、场景和真实身份角色计数。每格依次为 **闭集排名正确 / 排名正确但被拒 / 全部误拒 / 该组查询数**。闭集指不启用拒识时在所有有效身份中取第一名；全部误拒包含排名错误但也被拒的已知查询。前两数之差即最终正确接受数。','','|种子|场景|方案|老身份|新人|AUC|F1|','|---:|---|---|---|---|---:|---:|']
    for r in rows:
        counts=[' / '.join(str(d[k]) for k in ('rank','rank_reject','reject','n')) for d in r['diagnostics'].values()]
        lines.append('|{}|{}|{}|{}|{}|{:.6f}|{:.6f}|'.format(r['seed'],r['scene'],'重复 W' if r['method']=='repeat' else '老模板单 W',*counts,r['AUC'],r['F1']))
    lines+=['','复现命令（项目根目录）：` .\\.venv-py311\\Scripts\\python.exe -B research\\E277_single_whitening\\run.py `。仅使用 Train；逐折原始计数与指标见 `results.json`。']
    (H/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); print('\n'.join(lines[:12]))


if __name__=='__main__': main()
