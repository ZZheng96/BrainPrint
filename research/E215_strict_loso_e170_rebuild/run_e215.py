from __future__ import annotations
import json, os, sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TASKS = CFG['tasks']
CLASSES = np.arange(1, 21)
ACC = CFG['acceptance_grid']
DIAG_ACC = CFG.get('diagnostic_acceptance_grid', [0.50, 0.60, 0.70, 0.75, 0.80])
FIT_FRACTION = CFG['fit_fraction']
MODEL_DIR = OUT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

def log(x): print(x, flush=True)

def frac(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1)

def standardize_fit(tr, te, tasks, tte, fit):
    tr = tr.astype(np.float32, copy=True); te = te.astype(np.float32, copy=True)
    params = {}
    for task in TASKS:
        m = (tasks == task) & fit
        mu = tr[m].mean(0); sd = tr[m].std(0) + 1e-6
        tr[tasks == task] = (tr[tasks == task] - mu) / sd
        te[tte == task] = (te[tte == task] - mu) / sd
        params[task] = (mu, sd)
    return tr, te, params

def project(tr, te, fit, dim):
    z = tr[fit].astype(np.float64); mu = z.mean(0); centered = z - mu
    _, _, v = np.linalg.svd(centered, full_matrices=False)
    n = min(dim, z.shape[1], max(1, len(z) - 1)); basis = v[:n].T
    return ((tr.astype(np.float64) - mu) @ basis).astype(np.float32), ((te.astype(np.float64) - mu) @ basis).astype(np.float32), (mu, basis)

def make_representation(spec, raws, meta, tmeta, fit):
    parts = []; testparts = []; state = {}
    tasks = meta.task.to_numpy(); ttasks = tmeta.task.to_numpy()
    for stream in spec['streams']:
        tr, te, std = standardize_fit(*raws[stream], tasks, ttasks, fit)
        pa, pb, pstate = project(tr, te, fit, spec['pca'][stream])
        parts.append(pa); testparts.append(pb); state[stream] = {'standardization': std, 'pca': pstate}
    if len(parts) > 1:
        scales = [np.sqrt(np.mean(a[fit] ** 2)) + 1e-6 for a in parts]
        if spec.get('block_norm'):
            parts = [a / s for a, s in zip(parts, scales)]
            testparts = [b / s for b, s in zip(testparts, scales)]
        x = np.concatenate(parts, 1); xt = np.concatenate(testparts, 1)
        if spec.get('joint_pca'):
            x, xt, jstate = project(x, xt, fit, spec['joint_pca']); state['joint_pca'] = jstate
        state['block_scales'] = scales
    else:
        x, xt = parts[0], testparts[0]
    return x, xt, state

def km(x, k, seed):
    x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    if len(x) <= k: return x.cpu().numpy()
    rng = np.random.default_rng(seed); c = x[rng.choice(len(x), k, False)].clone()
    for _ in range(20):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        counts = torch.zeros(k, device=DEVICE).index_add_(0, a, torch.ones(len(x), device=DEVICE))
        nc = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], c)
        if torch.allclose(nc, c, atol=1e-6): break
        c = nc
    return c.cpu().numpy()

def prototypes(x, subjects, tasks, mask, classes, k):
    ps = []; keys = []
    for task in TASKS:
        for sub in classes:
            m = mask & (tasks == task) & (subjects == sub)
            if m.any():
                for p in km(x[m], k, int(sub) * 17): ps.append(p); keys.append((task, int(sub)))
    return np.stack(ps).astype(np.float32), keys

def scores(x, tasks, ps, keys):
    out = np.full(len(x), -np.inf, np.float32)
    kt = np.array([a for a, _ in keys])
    for task in TASKS:
        rows = np.flatnonzero(tasks == task); cols = np.flatnonzero(kt == task)
        if len(rows) and len(cols):
            q = torch.as_tensor(x[rows], device=DEVICE); p = torch.as_tensor(ps[cols], device=DEVICE)
            out[rows] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out

def nearest_identity(x, tasks, ps, keys):
    pred = np.full(len(x), -1, np.int32); kt = np.array([s for _, s in keys]); ktask = np.array([t for t, _ in keys])
    for task in TASKS:
        rows = np.flatnonzero(tasks == task); cols = np.flatnonzero(ktask == task)
        if len(rows) and len(cols):
            d = torch.cdist(torch.as_tensor(x[rows], device=DEVICE), torch.as_tensor(ps[cols], device=DEVICE)).argmin(1).cpu().numpy()
            pred[rows] = kt[cols[d]]
    return pred

def auc_ties(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); pos = y == 1; neg = ~pos
    if not pos.any() or not neg.any(): return float('nan')
    order = np.argsort(s, kind='mergesort'); sorted_s = s[order]; ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0; i = j
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))

def macro_f1(y, p, classes):
    vals = []
    for c in classes:
        yy = np.asarray(y) == c; pp = np.asarray(p) == c; tp = np.sum(yy & pp)
        den = 2 * tp + np.sum(~yy & pp) + np.sum(yy & ~pp)
        vals.append(2 * tp / den if den else 0.0)
    return float(np.mean(vals))

def binary_f1(y, p):
    y = np.asarray(y, int); p = np.asarray(p, int); tp = np.sum((y == 1) & (p == 1)); den = 2 * tp + np.sum((y == 0) & (p == 1)) + np.sum((y == 1) & (p == 0))
    return float(2 * tp / den) if den else 0.0

def fit_inner_oof(spec, raws, meta, tmeta, fr, subjects, outer_train, blocks, classes, k, fold_dir):
    tasks = meta.task.to_numpy(); outer_fit = outer_train & (fr <= FIT_FRACTION)
    query_blocks = sorted(np.unique(blocks[outer_fit]).tolist())
    oof = np.full(len(meta), np.nan, np.float32); pred = np.full(len(meta), -1, np.int32); block_rows = []
    for b in query_blocks:
        fitmask = outer_fit & (blocks != b); query = outer_fit & (blocks == b)
        if not fitmask.any() or not query.any(): continue
        x, _, state = make_representation(spec, raws, meta, tmeta, fitmask)
        ps, keys = prototypes(x, subjects, tasks, fitmask, classes, k)
        oof[query] = scores(x[query], tasks[query], ps, keys)
        pred[query] = nearest_identity(x[query], tasks[query], ps, keys)
        block_rows.append({'block': int(b), 'n_fit': int(fitmask.sum()), 'n_query': int(query.sum()), 'query_row_ids': np.flatnonzero(query).astype(np.int32)})
        np.savez_compressed(fold_dir / f'inner_block_{b}.npz', fit_row_ids=np.flatnonzero(fitmask), query_row_ids=np.flatnonzero(query), oof_scores=oof[query], prototypes=ps, prototype_tasks=np.array([t for t, _ in keys]), prototype_subjects=np.array([s for _, s in keys]), representation_train=x, state=np.array([state], dtype=object))
    return oof, pred, query_blocks, block_rows

def metric_at(sk, su, ref, acc):
    th = float(np.quantile(ref, 1 - acc)); reject = su < th; known_accept = sk >= th
    return {'acceptance': acc, 'threshold': th, 'rejection': float(reject.mean()), 'known_acceptance': float(known_accept.mean()), 'rejection_f1': binary_f1(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[~known_accept, reject]), 'n_unknown': int(len(su)), 'n_known': int(len(sk)), 'rejected_unknown': int(reject.sum()), 'accepted_known': int(known_accept.sum())}

def evaluate_config(name, spec, train_raws, test_raws, meta, tmeta, fr, subjects, blocks, fold_classes):
    tasks = meta.task.to_numpy(); empty_meta = tmeta.iloc[:0].copy(); folds = []; per = []; per_task = []; model_root = MODEL_DIR / name; model_root.mkdir(exist_ok=True)
    for held in fold_classes:
        fold_dir = model_root / f'held_subject_{held:02d}'; fold_dir.mkdir(exist_ok=True)
        outer_train = subjects != held; outer_fit = outer_train & (fr <= FIT_FRACTION); cache = fold_dir / 'outer_model_scores.npz'
        block_files = sorted(fold_dir.glob('inner_block_*.npz'))
        if cache.exists() and block_files:
            z = np.load(cache, allow_pickle=True); sk, su = z['known_scores'], z['unknown_scores']; x = z['representation_train']; ps = z['prototypes']; keys = list(zip(z['prototype_tasks'].tolist(), z['prototype_subjects'].tolist()))
            query_blocks = [int(p.stem.rsplit('_', 1)[1]) for p in block_files]; ref_parts = []; y_parts = []; p_parts = []
            for p in block_files:
                b = np.load(p, allow_pickle=True); ref_parts.append(b['oof_scores']); qids = b['query_row_ids']; ref_tasks = tasks[qids]; pps = b['prototypes']; pkeys = list(zip(b['prototype_tasks'].tolist(), b['prototype_subjects'].tolist())); ref_pred = nearest_identity(b['representation_train'][qids], ref_tasks, pps, pkeys); y_parts.append(subjects[qids]); p_parts.append(ref_pred)
            ref = np.concatenate(ref_parts) if ref_parts else np.array([], np.float32); fold_identity_f1 = macro_f1(np.concatenate(y_parts), np.concatenate(p_parts), np.unique(subjects[outer_train])) if y_parts else float('nan'); block_rows = [{'n_query': len(v)} for v in ref_parts]
        else:
            oof, pred, query_blocks, block_rows = fit_inner_oof(spec, train_raws, meta, empty_meta, fr, subjects, outer_train, blocks, np.unique(subjects[outer_train]), spec['k'], fold_dir)
            ref = oof[np.isfinite(oof)]; valid_pred = pred >= 0; fold_identity_f1 = macro_f1(subjects[valid_pred], pred[valid_pred], np.unique(subjects[outer_train]))
            x, _, state = make_representation(spec, train_raws, meta, empty_meta, outer_fit); ps, keys = prototypes(x, subjects, tasks, outer_fit, np.unique(subjects[outer_train]), spec['k']); scoremask = (fr > .6) & (fr <= .8); known = scoremask & outer_train; unknown = scoremask & (subjects == held); sk = scores(x[known], tasks[known], ps, keys); su = scores(x[unknown], tasks[unknown], ps, keys)
            np.savez_compressed(cache, fit_row_ids=np.flatnonzero(outer_fit), known_row_ids=np.flatnonzero(known), unknown_row_ids=np.flatnonzero(unknown), known_scores=sk, unknown_scores=su, prototypes=ps, prototype_tasks=np.array([t for t, _ in keys]), prototype_subjects=np.array([s for _, s in keys]), representation_train=x, state=np.array([state], dtype=object))
        curve_fold = [metric_at(sk, su, ref, a) for a in ACC + DIAG_ACC]; known_y = np.ones(len(sk)); unknown_y = np.zeros(len(su)); fold_auc = auc_ties(np.r_[known_y, unknown_y], np.r_[sk, su]); chosen = next(q for q in curve_fold if q['acceptance'] == .90)
        per.append({'subject': int(held), 'inner_blocks': query_blocks, 'inner_block_count': len(query_blocks), 'inner_query_samples': int(sum(r['n_query'] for r in block_rows)), 'fold_auc': fold_auc, 'rejection_090': chosen['rejection'], 'rejection_f1_090': chosen['rejection_f1'], 'known_acceptance_090': chosen['known_acceptance'], 'identity_macro_f1': fold_identity_f1, 'n_unknown': len(su), 'n_known': len(sk)})
        folds.append({'subject': int(held), 'known': sk, 'unknown': su, 'curve': curve_fold, 'auc': fold_auc, 'ref': ref}); log(f'{name} outer {held}/20 done; inner_blocks={query_blocks}')
    curve = []
    for acc in ACC + DIAG_ACC:
        rows = [f['curve'][next(i for i, q in enumerate(f['curve']) if q['acceptance'] == acc)] for f in folds]; n_u = sum(q['n_unknown'] for q in rows); n_k = sum(q['n_known'] for q in rows)
        curve.append({'acceptance': acc, 'LOSO_rejection_micro': float(sum(q['rejected_unknown'] for q in rows) / n_u), 'LOSO_rejection_macro': float(np.mean([q['rejection'] for q in rows])), 'known_acceptance_micro': float(sum(q['accepted_known'] for q in rows) / n_k), 'known_acceptance_macro': float(np.mean([q['known_acceptance'] for q in rows])), 'rejection_f1_mean': float(np.mean([q['rejection_f1'] for q in rows]))})
    fit = fr <= FIT_FRACTION; final_dir = model_root / 'final_inner_oof'; final_dir.mkdir(exist_ok=True); final_files = sorted(final_dir.glob('inner_block_*.npz'))
    if (final_dir / 'final_oof.npz').exists() and final_files:
        fz = np.load(final_dir / 'final_oof.npz', allow_pickle=True); final_ids = fz['final_ids']; final_ref = fz['final_ref']; final_pred = fz['final_pred']; final_blocks = [int(p.stem.rsplit('_', 1)[1]) for p in final_files]; final_rows = []
    else:
        oof, pred, final_blocks, final_rows = fit_inner_oof(spec, train_raws, meta, empty_meta, fr, subjects, np.ones(len(meta), bool), blocks, CLASSES, spec['k'], final_dir); final_ids = np.flatnonzero(np.isfinite(oof)); final_ref = oof[final_ids]; final_pred = pred[final_ids]; np.savez_compressed(final_dir / 'final_oof.npz', final_ids=final_ids, final_ref=final_ref, final_pred=final_pred)
    final_identity_f1 = macro_f1(subjects[final_ids], final_pred, CLASSES)
    test_raws = {'e65': np.load(ROOT / CFG['data']['e65_test']).astype(np.float32), 'logbp': np.load(ROOT / CFG['data']['logbp_test']).astype(np.float32)}
    all_raws = {'e65': (train_raws['e65'][0], test_raws['e65']), 'logbp': (train_raws['logbp'][0], test_raws['logbp'])}; x, xt, state = make_representation(spec, all_raws, meta, tmeta, fit); ps, keys = prototypes(x, subjects, tasks, fit, CLASSES, spec['k']); ts = scores(xt, tmeta.task.to_numpy(), ps, keys); is1 = tmeta.session.astype(str).str.casefold().isin(['session1', 's1', '1']).to_numpy()
    final_oof_row_ids = final_ids; np.savez_compressed(model_root / 'final_deployment_model.npz', fit_row_ids=np.flatnonzero(fit), test_row_ids=np.arange(len(tmeta)), test_scores=ts, prototypes=ps, prototype_tasks=np.array([t for t, _ in keys]), prototype_subjects=np.array([s for _, s in keys]), representation_train=x, representation_test=xt, state=np.array([state], dtype=object), final_oof_row_ids=final_oof_row_ids, final_oof_scores=final_ref, final_reference_model='all20_final_inner_oof')
    for q in curve:
        th = float(np.quantile(final_ref, 1 - q['acceptance'])); q['threshold_final'] = th; q['Test_S1_rejection'] = float(np.mean(ts[is1] < th)); q['Test_S2_acceptance'] = float(np.mean(ts[~is1] >= th))
    chosen = next(q for q in curve if q['acceptance'] == .90); identity_values = [p['identity_macro_f1'] for p in per]
    return {'config': name, 'device': str(DEVICE), 'loso_auc_fold_mean': float(np.nanmean([f['auc'] for f in folds])), 'loso_auc_fold_values': [f['auc'] for f in folds], 'loso_rejection_f1_090_mean': chosen['rejection_f1_mean'], 'identity_macro_f1': float(np.nanmean(identity_values)), 'identity_macro_f1_final_inner_oof': final_identity_f1, 'acceptance_curve': curve, 'reference_090': chosen, 'per_subject': per, 'per_subject_task': per_task, 'test_official': {'AUC': 'N/A', 'F1': 'N/A'}, 'feature_shape': list(x.shape), 'inner_block_count': sorted(set(f['inner_block_count'] for f in per)), 'inner_query_sample_range': [min(f['inner_query_samples'] for f in per), max(f['inner_query_samples'] for f in per)], 'strict_protocol': True, 'score_cache': 'outer_model_scores and inner_block files reused when complete', 'supersedes': 'previous results.json/report.md'}

def main():
    sys.stdout = open(OUT / 'run.log', 'w', encoding='utf-8')
    t0 = time.time(); meta = pd.read_csv(ROOT / CFG['data']['train_metadata']); tm = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    subjects = meta.subject.to_numpy(int); fr = frac(meta); blocks = np.floor(fr * CFG['blocks']).astype(int)
    train_e65 = np.load(ROOT / CFG['data']['e65_train']).astype(np.float32); train_logbp = np.load(ROOT / CFG['data']['logbp_train']).astype(np.float32)
    test_raws = {'e65': np.load(ROOT / CFG['data']['e65_test']).astype(np.float32), 'logbp': np.load(ROOT / CFG['data']['logbp_test']).astype(np.float32)}
    train_raws = {'e65': (train_e65, np.empty((0,) + train_e65.shape[1:], dtype=np.float32)), 'logbp': (train_logbp, np.empty((0,) + train_logbp.shape[1:], dtype=np.float32))}
    log(f'device={DEVICE}, train={len(meta)}, test={len(tm)}, fit={(fr <= FIT_FRACTION).sum()}, inner_blocks={sorted(np.unique(blocks[fr <= FIT_FRACTION]).tolist())}')
    old_results = OUT / 'results.json'; old_report = OUT / 'report.md'
    if old_results.exists(): shutil.copy2(old_results, OUT / 'superseded_results_previous.json')
    if old_report.exists(): shutil.copy2(old_report, OUT / 'superseded_report_previous.md')
    fold_limit = int(os.environ.get('E215_FOLDS', '20')); fold_classes = CLASSES[:fold_limit]
    results = {n: evaluate_config(n, s, train_raws, test_raws, meta, tm, fr, subjects, blocks, fold_classes) for n, s in CFG['configs'].items()}
    results['runtime_seconds'] = time.time() - t0; (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# E215 strict LOSO E170 rebuild（修正版）', '', f'- 设备：`{DEVICE}`。完整20折；outer held subject 完全不参与任何内层拟合或校准。', '- 修复：prototype 最近邻使用 `argmin`；每个非空 inner query block 排除该 block 后重新拟合 task standardization、PCA、prototype。', '- 内层实际块数/样本范围、每折模型状态、行 ID、分数和最终部署模型已保存。', '- ties 使用平均秩 AUC；LOSO AUC 报告 per-fold AUC 均值，不混合不同 outer fold 的异尺度分数。', '- 旧 `results.json/report.md` 已保存为 `superseded_results_previous.json` / `superseded_report_previous.md`；旧结果不可用于结论。', '- 官方 Test AUC/F1：N/A（Test 无公开身份标签）。', '']
    for n, r in results.items():
        if not isinstance(r, dict): continue
        q = r['reference_090']; lines += [f'## {n}', '', f'- Strict LOSO per-fold AUC mean：**{r["loso_auc_fold_mean"]:.6f}**；开放集拒识 F1（acceptance=0.90）：**{r["loso_rejection_f1_090_mean"]:.6f}**；identity macro-F1（per-subject identity_macro_f1，非官方F1）：**{r["identity_macro_f1"]:.6f}**；final inner OOF：**{r["identity_macro_f1_final_inner_oof"]:.6f}**。', f'- acceptance=0.90：LOSO rejection micro/macro={q["LOSO_rejection_micro"]:.6f}/{q["LOSO_rejection_macro"]:.6f}；known acceptance micro/macro={q["known_acceptance_micro"]:.6f}/{q["known_acceptance_macro"]:.6f}。', f'- 冻结全量 Train 后 Test S1 rejection={q["Test_S1_rejection"]:.6f}；S2 acceptance={q["Test_S2_acceptance"]:.6f}。', f'- inner query blocks：{r["inner_block_count"]}；每 outer fold query 样本数范围：{r["inner_query_sample_range"]}。', '']
    lines += ['## 固定 acceptance 结果', '', '- acceptance 全部预先固定于 Train/LOSO 参考分数；Test 仅用于冻结模型后的 S1/S2 描述，不据 Test 挑工作点。', '- LOSO rejection F1 为拒识二分类F1；identity_macro_f1 是身份识别宏F1，非官方Test F1。', '', '|config|acceptance|LOSO rejection micro|LOSO rejection macro|known acceptance micro|known acceptance macro|rejection F1|S1 rejection|S2 acceptance|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n, r in results.items():
        if not isinstance(r, dict): continue
        for q in r['acceptance_curve']:
            lines.append(f'|{n}|{q["acceptance"]:.2f}|{q["LOSO_rejection_micro"]:.6f}|{q["LOSO_rejection_macro"]:.6f}|{q["known_acceptance_micro"]:.6f}|{q["known_acceptance_macro"]:.6f}|{q["rejection_f1_mean"]:.6f}|{q.get("Test_S1_rejection", "N/A")}|{q.get("Test_S2_acceptance", "N/A")}|')
    lines += ['', '完整 per-subject/task 明细与模型/阈值引用行见 `results.json`；官方 Test AUC/F1 未完成（无公开标签）。', '实际运行：`.venv\\Scripts\\python.exe research\\E215_strict_loso_e170_rebuild\\run_e215.py`（CUDA，完整20折；先以 `E215_FOLDS=1` 验证后再运行完整实验）。', '未完成项：无法在无Test身份标签时计算官方Test AUC/F1；本报告不宣称11项全达标。', f'运行耗时：{time.time() - t0:.1f}s。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
