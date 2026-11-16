from __future__ import annotations
import itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
E215 = ROOT / 'research' / 'E215_strict_loso_e170_rebuild'
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
TASKS = CFG['tasks']; CLASSES = np.arange(1, 21); GRID = CFG['acceptance_grid']

def auc_ties(y, s):
    y=np.asarray(y,int); s=np.asarray(s,float); pos=y==1; neg=~pos
    if not pos.any() or not neg.any(): return float('nan')
    order=np.argsort(s,kind='mergesort'); ss=s[order]; ranks=np.empty(len(s),float); i=0
    while i<len(s):
        j=i+1
        while j<len(s) and ss[j]==ss[i]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    return float((ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))

def f1(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int); tp=np.sum((y==1)&(p==1))
    den=2*tp+np.sum((y==0)&(p==1))+np.sum((y==1)&(p==0))
    return float(2*tp/den) if den else 0.0

def score_combo(ref, rt, sk, st, su, sut, combo):
    th={t:float(np.quantile(ref[rt==t],1-a)) for t,a in zip(TASKS,combo)}
    ka=np.array([sk[i]>=th[st[i]] for i in range(len(sk))])
    rej=np.array([su[i]<th[sut[i]] for i in range(len(su))])
    return {'acceptance_by_task':dict(zip(TASKS,combo)), 'thresholds':th,
            'rejected_unknown':int(rej.sum()),'n_unknown':len(su),'accepted_known':int(ka.sum()),'n_known':len(sk),
            'rejection':float(rej.mean()),'known_acceptance':float(ka.mean()),
            'rejection_f1':f1(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[ka,rej])}

def load_fold(model_root, meta, held):
    fd=model_root/f'held_subject_{held:02d}'; z=np.load(fd/'outer_model_scores.npz',allow_pickle=True)
    sk,su=z['known_scores'],z['unknown_scores']; ki,ui=z['known_row_ids'],z['unknown_row_ids']; tasks=meta.task.to_numpy()
    refs=[]; rts=[]
    for p in sorted(fd.glob('inner_block_*.npz')):
        b=np.load(p,allow_pickle=True); refs.append(b['oof_scores']); rts.append(tasks[b['query_row_ids']])
    return sk,tasks[ki],su,tasks[ui],np.concatenate(refs),np.concatenate(rts)

def select(records):
    eligible=[r for r in records if (r['LOSO_rejection_micro']>=CFG['loso_rejection_target'] or r['LOSO_rejection_macro']>=CFG['loso_rejection_target']) and r['known_acceptance_micro']>=CFG['known_acceptance_floor']]
    # Predetermined lexicographic order: rejection micro, rejection macro, rejection F1, known acceptance.
    chosen=max(eligible,key=lambda r:(r['LOSO_rejection_micro'],r['LOSO_rejection_macro'],r['rejection_f1_mean'],r['known_acceptance_micro'])) if eligible else None
    pareto=[]
    for r in eligible:
        dominated=False
        for q in eligible:
            if q is r: continue
            if q['LOSO_rejection_micro']>=r['LOSO_rejection_micro'] and q['known_acceptance_micro']>=r['known_acceptance_micro'] and (q['LOSO_rejection_micro']>r['LOSO_rejection_micro'] or q['known_acceptance_micro']>r['known_acceptance_micro']): dominated=True; break
        if not dominated: pareto.append(r)
    return chosen,eligible,pareto

def evaluate(name, old, tasks, test_meta):
    root=E215/'models'/name; folds=[]
    for held in CLASSES:
        sk,st,su,sut,ref,rt=load_fold(root,pd.DataFrame({'task':tasks}),held)
        rows=[score_combo(ref,rt,sk,st,su,sut,c) for c in itertools.product(GRID,repeat=len(TASKS))]
        folds.append({'rows':rows,'auc':auc_ties(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[sk,su])})
    records=[]
    for j,combo in enumerate(itertools.product(GRID,repeat=len(TASKS))):
        rs=[x['rows'][j] for x in folds]
        records.append({'acceptance_by_task':dict(zip(TASKS,combo)),
          'LOSO_rejection_micro':sum(x['rejected_unknown'] for x in rs)/sum(x['n_unknown'] for x in rs),
          'LOSO_rejection_macro':float(np.mean([x['rejection'] for x in rs])),
          'known_acceptance_micro':sum(x['accepted_known'] for x in rs)/sum(x['n_known'] for x in rs),
          'known_acceptance_macro':float(np.mean([x['known_acceptance'] for x in rs])),
          'rejection_f1_mean':float(np.mean([x['rejection_f1'] for x in rs]))})
    chosen,eligible,pareto=select(records)
    if chosen is None:
        result={'config':name,'device':'cuda' if __import__('torch').cuda.is_available() else 'cpu','selected':None,'selection_status':'no_combination_meets_loso_rejection_and_known_acceptance_constraints','loso_auc_fold_mean':float(np.nanmean([x['auc'] for x in folds])),'loso_auc_fold_values':[x['auc'] for x in folds],'identity_macro_f1':old[name].get('identity_macro_f1'),'identity_macro_f1_final_inner_oof':old[name].get('identity_macro_f1_final_inner_oof'),'official_test_auc':'N/A','official_test_f1':'N/A','candidate_count':len(records),'eligible_count':0,'pareto_candidates':[],'candidate_summary':sorted(records,key=lambda r:(r['LOSO_rejection_micro'],r['LOSO_rejection_macro'],r['rejection_f1_mean']),reverse=True)[:20],'protocol':'Selection uses only outer-fold inner OOF known scores by task and strict LOSO scores; no Test selection.'}
        return result
    final=np.load(root/'final_inner_oof'/'final_oof.npz',allow_pickle=True); final_ref=final['final_ref']; final_tasks=tasks[final['final_ids']]
    dep=np.load(root/'final_deployment_model.npz',allow_pickle=True); ts=dep['test_scores']; tt=test_meta.task.to_numpy()
    is1=test_meta.session.astype(str).str.casefold().isin(['session1','s1','1']).to_numpy()
    th={t:float(np.quantile(final_ref[final_tasks==t],1-a)) for t,a in chosen['acceptance_by_task'].items()}
    s1=np.array([ts[i]<th[tt[i]] for i in np.flatnonzero(is1)]); s2=np.array([ts[i]>=th[tt[i]] for i in np.flatnonzero(~is1)])
    chosen=dict(chosen,thresholds_final=th,Test_S1_rejection=float(s1.mean()),Test_S2_acceptance=float(s2.mean()),eligible_count=len(eligible),pareto_count=len(pareto))
    result={'config':name,'device':'cuda' if __import__('torch').cuda.is_available() else 'cpu','selected':chosen,
      'loso_auc_fold_mean':float(np.nanmean([x['auc'] for x in folds])),'loso_auc_fold_values':[x['auc'] for x in folds],
      'identity_macro_f1':old[name].get('identity_macro_f1'),'identity_macro_f1_final_inner_oof':old[name].get('identity_macro_f1_final_inner_oof'),
      'official_test_auc':'N/A','official_test_f1':'N/A','candidate_count':len(records),'eligible_count':len(eligible),
      'pareto_candidates':pareto[:100], 'candidate_summary':sorted(eligible,key=lambda r:(r['LOSO_rejection_micro'],r['LOSO_rejection_macro'],r['rejection_f1_mean']),reverse=True)[:20],
      'protocol':'Selection uses only outer-fold inner OOF known scores by task and strict LOSO scores; final thresholds use all-20 final_inner_oof; Test S1/S2 only frozen audit.'}
    np.savez_compressed(OUT/f'{name}_final_thresholds.npz',**{f'task_{t}':np.array([th[t]]) for t in TASKS})
    return result

def main():
    t=time.time(); old=json.loads((E215/'results.json').read_text(encoding='utf-8')); tasks=pd.read_csv(ROOT/CFG['data']['train_metadata']).task.to_numpy(); test=pd.read_csv(ROOT/CFG['data']['test_metadata']); out={}
    for name in CFG['configs']:
        print(f'evaluating {name}',flush=True); out[name]=evaluate(name,old,tasks,test)
    out['runtime_seconds']=time.time()-t; (OUT/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# E217 taskwise acceptance search','','- 独立实验；复用 E215 分数/模型缓存，不修改 E215/E216。','- 任务级 acceptance 候选：`[0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]`；组合数：4096/config。','- 选点只使用 outer-fold inner OOF known 分数校准阈值及严格 LOSO；筛选 LOSO rejection micro 或 macro ≥ 0.90 且 known acceptance micro ≥ 0.80。字典序：rejection micro → rejection macro → rejection F1 → known acceptance。','- all-20 final_inner_oof 仅用于冻结最终 task threshold；Test S1/S2 仅审计。官方 Test AUC/F1：N/A。','']
    for name,r in out.items():
        if not isinstance(r,dict): continue
        q=r['selected']; lines += [f'## {name}']
        if q is None:
            lines += [f'- **无组合同时满足** LOSO rejection micro 或 macro ≥ 0.90 且 known acceptance micro ≥ 0.80；4096 个候选均未入选。',f'- LOSO AUC fold mean：**{r["loso_auc_fold_mean"]:.6f}**；identity macro-F1：**{r["identity_macro_f1"]:.6f}**；final-inner-OOF identity macro-F1：**{r["identity_macro_f1_final_inner_oof"]:.6f}**；官方 Test AUC/F1：**N/A / N/A**。','']
            continue
        lines += [f'- 选中 task acceptance：`{q["acceptance_by_task"]}`；候选/合格/Pareto：{r["candidate_count"]}/{r["eligible_count"]}/{q["pareto_count"]}。',f'- LOSO rejection micro/macro：**{q["LOSO_rejection_micro"]:.6f} / {q["LOSO_rejection_macro"]:.6f}**；known acceptance micro/macro：**{q["known_acceptance_micro"]:.6f} / {q["known_acceptance_macro"]:.6f}**；rejection F1：**{q["rejection_f1_mean"]:.6f}**。',f'- LOSO AUC fold mean：**{r["loso_auc_fold_mean"]:.6f}**；identity macro-F1：**{r["identity_macro_f1"]:.6f}**；final-inner-OOF identity macro-F1：**{r["identity_macro_f1_final_inner_oof"]:.6f}**。',f'- Test S1 rejection：**{q["Test_S1_rejection"]:.6f}**；Test S2 acceptance：**{q["Test_S2_acceptance"]:.6f}**；官方 Test AUC/F1：**N/A / N/A**。','- 冻结阈值：'+', '.join(f'{t}={v:.6f}' for t,v in q['thresholds_final'].items()),'','### Pareto/候选摘要','|task acceptance|LOSO reject micro|LOSO reject macro|known accept micro|known accept macro|reject F1|','|---|---:|---:|---:|---:|---:|']
        for x in r['pareto_candidates'][:20]: lines.append('|'+','.join(f'{x["acceptance_by_task"][t]:.2f}' for t in TASKS)+f'|{x["LOSO_rejection_micro"]:.6f}|{x["LOSO_rejection_macro"]:.6f}|{x["known_acceptance_micro"]:.6f}|{x["known_acceptance_macro"]:.6f}|{x["rejection_f1_mean"]:.6f}|')
        lines.append('')
    lines.append(f'运行耗时：{time.time()-t:.2f}s；完整候选与结果见 `results.json`。')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
