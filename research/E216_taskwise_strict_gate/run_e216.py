from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
E215 = ROOT / 'research' / 'E215_strict_loso_e170_rebuild'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TASKS = CFG['tasks']; CLASSES = np.arange(1, 21)
ACC = CFG['acceptance_grid']; DIAG_ACC = CFG.get('diagnostic_acceptance_grid', [0.50, 0.60, 0.70, 0.75, 0.80])
ALL_ACC = ACC + DIAG_ACC

def log(x): print(x, flush=True)
def auc_ties(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); pos = y == 1; neg = ~pos
    if not pos.any() or not neg.any(): return float('nan')
    order = np.argsort(s, kind='mergesort'); ss = s[order]; ranks = np.empty(len(s), float); i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and ss[j] == ss[i]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0; i = j
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
def binary_f1(y, p):
    y = np.asarray(y, int); p = np.asarray(p, int); tp = np.sum((y == 1) & (p == 1))
    den = 2 * tp + np.sum((y == 0) & (p == 1)) + np.sum((y == 1) & (p == 0))
    return float(2 * tp / den) if den else 0.0

def taskwise_metric(ref, ref_tasks, sk, sk_tasks, su, su_tasks, acc):
    thresholds = {}; ka = np.zeros(len(sk), bool); rej = np.zeros(len(su), bool)
    for task in TASKS:
        r = ref[ref_tasks == task];
        if len(r) == 0: continue
        th = float(np.quantile(r, 1 - acc)); thresholds[task] = th
        ka[sk_tasks == task] = sk[sk_tasks == task] >= th
        rej[su_tasks == task] = su[su_tasks == task] < th
    f1 = binary_f1(np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[ka, rej])
    return {'acceptance': acc, 'thresholds': thresholds, 'rejection': float(rej.mean()), 'known_acceptance': float(ka.mean()), 'rejection_f1': f1, 'n_unknown': int(len(su)), 'n_known': int(len(sk)), 'rejected_unknown': int(rej.sum()), 'accepted_known': int(ka.sum())}

def fold_reference(fold_dir, meta, held):
    z = np.load(fold_dir / 'outer_model_scores.npz', allow_pickle=True)
    sk, su = z['known_scores'], z['unknown_scores']; k_ids, u_ids = z['known_row_ids'], z['unknown_row_ids']
    tasks = meta.task.to_numpy()
    refs=[]; rtasks=[]
    for p in sorted(fold_dir.glob('inner_block_*.npz')):
        b=np.load(p, allow_pickle=True); ids=b['query_row_ids']; refs.append(b['oof_scores']); rtasks.append(tasks[ids])
    return sk, tasks[k_ids], su, tasks[u_ids], np.concatenate(refs), np.concatenate(rtasks)

def evaluate(name, old):
    tasks = pd.read_csv(ROOT / CFG['data']['train_metadata']).task.to_numpy()
    tmeta = pd.read_csv(ROOT / CFG['data']['test_metadata']); test_tasks = tmeta.task.to_numpy()
    model_root = E215 / 'models' / name; folds=[]; per=[]
    for held in CLASSES:
        fd=model_root / f'held_subject_{held:02d}'
        sk, skt, su, sut, ref, reft = fold_reference(fd, pd.DataFrame({'task':tasks}), held)
        curve=[taskwise_metric(ref,reft,sk,skt,su,sut,a) for a in ALL_ACC]
        q=next(x for x in curve if x['acceptance']==.90)
        np.savez_compressed(OUT / f'{name}_held_{held:02d}_thresholds.npz', **{f'acc_{x["acceptance"]:.2f}': np.array([x['thresholds'].get(t, np.nan) for t in TASKS]) for x in curve})
        oldp=old[name]['per_subject'][held-1]
        per.append({'subject':int(held), 'fold_auc':auc_ties(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[sk,su]), 'rejection_090':q['rejection'], 'rejection_f1_090':q['rejection_f1'], 'known_acceptance_090':q['known_acceptance'], 'identity_macro_f1':oldp.get('identity_macro_f1'), 'n_unknown':len(su),'n_known':len(sk), 'thresholds_090':q['thresholds']})
        folds.append({'curve':curve,'auc':per[-1]['fold_auc']}); log(f'{name} outer {held}/20 done')
    curve=[]
    for a in ALL_ACC:
        rows=[f['curve'][i] for f in folds for i in [next(j for j,q in enumerate(f['curve']) if q['acceptance']==a)]]
        curve.append({'acceptance':a,'LOSO_rejection_micro':sum(q['rejected_unknown'] for q in rows)/sum(q['n_unknown'] for q in rows),'LOSO_rejection_macro':float(np.mean([q['rejection'] for q in rows])),'known_acceptance_micro':sum(q['accepted_known'] for q in rows)/sum(q['n_known'] for q in rows),'known_acceptance_macro':float(np.mean([q['known_acceptance'] for q in rows])),'rejection_f1_mean':float(np.mean([q['rejection_f1'] for q in rows]))})
    final_dir=model_root/'final_inner_oof'; fz=np.load(final_dir/'final_oof.npz',allow_pickle=True); final_ids=fz['final_ids']; final_ref=fz['final_ref']; final_tasks=tasks[final_ids]
    deploy=np.load(model_root/'final_deployment_model.npz',allow_pickle=True); ts=deploy['test_scores']; is1=tmeta.session.astype(str).str.casefold().isin(['session1','s1','1']).to_numpy()
    final_curve=[]
    for q in curve:
        m=taskwise_metric(final_ref,final_tasks,final_ref,final_tasks,final_ref,final_tasks,q['acceptance'])
        th=q['acceptance']; q['thresholds_final']=m['thresholds']; q['Test_S1_rejection']=float(np.mean(np.array([ts[i] < m['thresholds'].get(test_tasks[i],np.inf) for i in np.flatnonzero(is1)]))); q['Test_S2_acceptance']=float(np.mean(np.array([ts[i] >= m['thresholds'].get(test_tasks[i],-np.inf) for i in np.flatnonzero(~is1)])))
    np.savez_compressed(OUT / f'{name}_final_thresholds.npz', **{f'acc_{q["acceptance"]:.2f}':np.array([q['thresholds_final'].get(t,np.nan) for t in TASKS]) for q in curve})
    chosen=next(q for q in curve if q['acceptance']==.90)
    r={'config':name,'device':str(DEVICE),'loso_auc_fold_mean':float(np.nanmean([f['auc'] for f in folds])),'loso_auc_fold_values':[f['auc'] for f in folds],'loso_rejection_f1_090_mean':chosen['rejection_f1_mean'],'identity_macro_f1':old[name]['identity_macro_f1'],'identity_macro_f1_final_inner_oof':old[name]['identity_macro_f1_final_inner_oof'],'acceptance_curve':curve,'reference_090':chosen,'per_subject':per,'per_subject_task':'taskwise thresholds; full rows in per_subject thresholds_090','test_official':{'AUC':'N/A','F1':'N/A'},'feature_shape':old[name].get('feature_shape'),'inner_block_count':old[name].get('inner_block_count'),'inner_query_sample_range':old[name].get('inner_query_sample_range'),'strict_protocol':True,'threshold_protocol':'Each threshold uses only outer-fold inner OOF known scores by task; final uses all-20 final_inner_oof by task; Test only frozen audit','score_cache':'E215 models/outer_model_scores.npz and final deployment cache reused'}
    return r

def main():
    t0=time.time(); old=json.loads((E215/'results.json').read_text(encoding='utf-8')); results={}
    for name in old:
        if isinstance(old[name],dict) and 'config' in old[name]: results[name]=evaluate(name,old)
    results['runtime_seconds']=time.time()-t0; (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# E216 taskwise strict gate','',f'- 设备：`{DEVICE}`；复用 E215 已保存模型/分数缓存，不修改 E215。','- 阈值严格按 task 从 outer fold 的 inner OOF known 分数计算；最终部署按 all-20 final_inner_oof 计算。Test 不参与选点，仅冻结后审计。','- 官方 Test AUC/F1：N/A（Test 无公开身份标签）。','']
    lines += ['## acceptance=0.90 重点结果','']
    for n,r in results.items():
        if not isinstance(r,dict): continue
        q=r['reference_090']; lines += [f'### {n}',f'- LOSO AUC fold mean：**{r["loso_auc_fold_mean"]:.6f}**；LOSO rejection F1：**{r["loso_rejection_f1_090_mean"]:.6f}**；identity macro-F1：**{r["identity_macro_f1"]:.6f}**；final inner OOF identity macro-F1：**{r["identity_macro_f1_final_inner_oof"]:.6f}**。',f'- LOSO rejection micro/macro：{q["LOSO_rejection_micro"]:.6f}/{q["LOSO_rejection_macro"]:.6f}；known acceptance micro/macro：{q["known_acceptance_micro"]:.6f}/{q["known_acceptance_macro"]:.6f}。',f'- Test S1 rejection：**{q["Test_S1_rejection"]:.6f}**；Test S2 acceptance：**{q["Test_S2_acceptance"]:.6f}**。','']
    lines += ['## 完整曲线','','|config|acceptance|LOSO reject micro|LOSO reject macro|known accept micro|known accept macro|reject F1|S1 rejection|S2 acceptance|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in results.items():
        if isinstance(r,dict):
            for q in r['acceptance_curve']: lines.append(f'|{n}|{q["acceptance"]:.2f}|{q["LOSO_rejection_micro"]:.6f}|{q["LOSO_rejection_macro"]:.6f}|{q["known_acceptance_micro"]:.6f}|{q["known_acceptance_macro"]:.6f}|{q["rejection_f1_mean"]:.6f}|{q["Test_S1_rejection"]:.6f}|{q["Test_S2_acceptance"]:.6f}|')
    lines += ['',f'运行耗时：{time.time()-t0:.1f}s。结果详见 `results.json`；每折及最终 task threshold 另存为 npz。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
