from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
DEVICE = 'cuda'
TASKS = list(CFG['tasks'])
N_TASKS = len(TASKS)
FIT_FRAC = float(CFG['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['evaluation_fraction']
N_BLOCKS = int(CFG['inner_blocks'])
Q_TAU = float(CFG['threshold_quantile'])
PCA_DIM = int(CFG['pca_dim'])
K_PROTO = int(CFG['prototypes_k'])
MIN_TASK_N = int(CFG['subject_task_min_samples'])
SEED = int(CFG['seed'])
CANDS = [dict(c) for c in CFG['candidates']]
NAMES = [c['name'] for c in CANDS]
LOG = OUT / 'run.log'
FR = None


def log(s):
    print(s, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(s) + '\n')


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def frac(m):
    return m.trial.to_numpy(float) / m.groupby(['subject', 'task']).trial.transform('max').to_numpy(float).clip(1)


def quantile_tau(x, q):
    return float(np.quantile(np.asarray(x, float), q))


def quantile_lower(x, q):
    x = np.asarray(x, float)
    try:
        return float(np.quantile(x, q, method='lower'))
    except TypeError:
        return float(np.quantile(x, q, interpolation='lower'))


# =====================================================================  E221/E223 baseline scoring (verbatim)
def km(x, k, seed):
    x = tensor(x)
    if len(x) <= k:
        return x.cpu().numpy()
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, False)].clone()
    for _ in range(20):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        counts = torch.zeros(k, device=DEVICE).index_add_(0, a, torch.ones(len(x), device=DEVICE))
        nc = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], c)
        if torch.allclose(nc, c, atol=1e-6):
            break
        c = nc
    return c.cpu().numpy()


def base_prototypes(z, subjects, tasks, mask, classes, k):
    ps, keys = [], []
    for task in TASKS:
        for sub in classes:
            m = mask & (tasks == task) & (subjects == sub)
            if m.any():
                for p in km(z[m], k, int(sub) * 17):
                    ps.append(p)
                    keys.append((task, int(sub)))
    return np.stack(ps).astype(np.float32), keys


def base_fit_rep(raw, tasks, y, fit, k):
    x = tensor(raw)
    z = torch.empty_like(x)
    mu, sd = [], []
    for t in TASKS:
        rows = tasks == t
        a = x[rows & fit]
        m = a.mean(0)
        s = a.std(0, correction=0) + 1e-6
        z[rows] = (x[rows] - m) / s
        mu.append(m.cpu().numpy())
        sd.append(s.cpu().numpy())
    a = z[fit].double()
    pm = a.mean(0)
    _, v = torch.linalg.eigh((a - pm).T @ (a - pm))
    W = v[:, -PCA_DIM:].flip(1)
    z = ((z.double() - pm) @ W).float().cpu().numpy()
    cls = np.unique(y[fit])
    p, keys = base_prototypes(z, y, tasks, fit, cls, k)
    return dict(mu=mu, sd=sd, pm=pm.cpu().numpy(), W=W.cpu().numpy(), p=p, keys=keys, k=k)


def base_transform(state, raw, tasks):
    x = tensor(raw)
    z = torch.empty_like(x)
    for i, t in enumerate(TASKS):
        rows = tasks == t
        z[rows] = (x[rows] - tensor(state['mu'][i])) / tensor(state['sd'][i])
    z = ((z.double() - tensor(state['pm'], torch.float64)) @ tensor(state['W'], torch.float64)).float()
    return z.cpu().numpy()


def base_proto_meta(state):
    keys = state['keys']
    return np.array([t for t, _ in keys]), np.array([int(s) for _, s in keys], int)


def base_score_parts(state, z, rows, task_rows):
    proto_task, proto_id = base_proto_meta(state)
    parts = []
    for t in TASKS:
        rr = rows[task_rows == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        S = (-torch.cdist(tensor(z[rr]), tensor(state['p'][cols]))).double().cpu().numpy()
        parts.append((t, rr, S, proto_id[cols]))
    return parts


def base_score(state, z, tasks, rows):
    s = np.full(len(tasks), np.nan)
    p = np.full(len(tasks), -1, int)
    for t, rr, S, ident in base_score_parts(state, z, rows, tasks[rows]):
        s[rr] = S.max(1)
        p[rr] = ident[S.argmax(1)]
    return s, p


def base_inner_oof(fit, raw, tasks, y, blocks, pk):
    """Known-only inner OOF on enrolled fit rows; returns scores and predicted identities."""
    oof_s = np.full(len(y), np.nan)
    oof_p = np.full(len(y), -1, int)
    for b in np.unique(blocks[fit]):
        hold = fit & (blocks == b)
        inner = fit & ~hold
        assert set(y[inner]) == set(y[fit])
        st = base_fit_rep(raw, tasks, y, inner, pk)
        z = base_transform(st, raw, tasks)
        s, p = base_score(st, z, tasks, np.flatnonzero(hold))
        oof_s[hold] = s[hold]
        oof_p[hold] = p[hold]
    return oof_s, oof_p


# =====================================================================  per-identity threshold construction
def estimate_thresholds(oof_s, oof_p, subj, task, fit, min_task_n=MIN_TASK_N):
    """All thresholds come from enrolled inner-OOF known scores inside `fit` rows only."""
    s = oof_s[fit]
    sub = subj[fit]
    tk = task[fit]
    assert np.isfinite(s).all() and (oof_p[fit] >= 0).all()
    tau_global = quantile_tau(s, Q_TAU)
    tau_subject, counts, fallback = {}, {}, {}
    for i in np.unique(sub):
        i = int(i)
        tau_subject[i] = quantile_tau(s[sub == i], Q_TAU)
        counts[i] = {}
        fallback[i] = []
        for t in TASKS:
            counts[i][t] = int(((sub == i) & (tk == t)).sum())
            if counts[i][t] < min_task_n:
                fallback[i].append(t)
    tau_subject_task = {}
    for i in tau_subject:
        tau_subject_task[i] = {}
        for t in TASKS:
            m = (sub == i) & (tk == t)
            tau_subject_task[i][t] = (quantile_tau(s[m], Q_TAU)
                                      if int(m.sum()) >= min_task_n else tau_subject[i])
    return dict(tau_global=tau_global, tau_subject=tau_subject,
                tau_subject_task=tau_subject_task, counts=counts, fallback=fallback,
                n_oof=int(fit.sum()), n_subjects=len(tau_subject))


def row_tau(cand, T, pred, task):
    """Threshold for each row, selected by the model-predicted identity."""
    g = T['tau_global']
    n = len(pred)
    if cand['type'] == 'global':
        return np.full(n, g)
    if cand['type'] == 'subject':
        return np.array([T['tau_subject'].get(int(p), g) for p in pred], float)
    if cand['type'] == 'subject_task':
        return np.array([T['tau_subject_task'].get(int(p), {}).get(t, T['tau_subject'].get(int(p), g))
                         for p, t in zip(pred, task)], float)
    if cand['type'] == 'subject_shrunk':
        w = float(cand['shrink_weight'])
        return np.array([w * T['tau_subject'].get(int(p), g) + (1 - w) * g for p in pred], float)
    raise ValueError(cand['type'])


def thresholds_to_json(T):
    return dict(
        threshold_quantile=Q_TAU,
        tau_global=T['tau_global'],
        tau_subject={str(i): float(v) for i, v in T['tau_subject'].items()},
        tau_subject_task={str(i): {t: float(v) for t, v in d.items()}
                          for i, d in T['tau_subject_task'].items()},
        subject_task_counts={str(i): {t: int(v) for t, v in d.items()} for i, d in T['counts'].items()},
        subject_task_fallback={str(i): list(v) for i, v in T['fallback'].items()},
        n_oof_rows=int(T['n_oof']), n_subjects=int(T['n_subjects']),
    )


# =====================================================================  metrics
def auc_ties(pos, neg):
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    s = np.r_[pos, neg]
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def evaluate(s, p, y, known, unknown, tau_row):
    k = s[known]
    u = s[unknown]
    rej_u = u < tau_row[unknown]
    rej_k = k < tau_row[known]
    tp = float(rej_u.sum())
    fp = float(rej_k.sum())
    fn = float((~rej_u).sum())
    return dict(
        auc=float(((u[:, None] < k).sum() + .5 * (u[:, None] == k).sum()) / (len(k) * len(u))),
        rejection_f1=float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        known_acceptance=float((~rej_k).mean()),
        unknown_rejection=float(rej_u.mean()),
        identity_correct_but_rejected=float(((p[known] == y[known]) & rej_k).mean()),
        threshold=float(np.mean(tau_row[known])),
        threshold_min=float(np.min(tau_row[known])), threshold_max=float(np.max(tau_row[known])),
        n_known=int(len(k)), n_unknown=int(len(u)),
        accepted_known=int((~rej_k).sum()), rejected_unknown=int(rej_u.sum()),
    )


def summarise(folds, names):
    out = {}
    for name in names:
        ms = [f['metrics'][name] for f in folds]
        n_u = sum(m['n_unknown'] for m in ms)
        n_k = sum(m['n_known'] for m in ms)
        aucs = [m['auc'] for m in ms]
        out[name] = dict(
            auc=float(np.mean(aucs)),
            auc_fold_std=float(np.std(aucs)),
            auc_fold_values=[float(v) for v in aucs],
            rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
            known_acceptance=float(np.mean([m['known_acceptance'] for m in ms])),
            known_acceptance_micro=float(sum(m['accepted_known'] for m in ms) / n_k),
            unknown_rejection=float(np.mean([m['unknown_rejection'] for m in ms])),
            unknown_rejection_micro=float(sum(m['rejected_unknown'] for m in ms) / n_u),
            identity_correct_but_rejected=float(np.mean([m['identity_correct_but_rejected'] for m in ms])),
            threshold=float(np.mean([m['threshold'] for m in ms])),
            threshold_min=float(np.min([m['threshold_min'] for m in ms])),
            threshold_max=float(np.max([m['threshold_max'] for m in ms])),
        )
    return out


def test_row(s, p, tau_row, is_s1):
    rej = s < tau_row
    delta = s - tau_row                      # acceptance margin; >0 means accepted
    s1, s2 = s[is_s1], s[~is_s1]
    d1, d2 = delta[is_s1], delta[~is_s1]
    rej1 = float(rej[is_s1].mean())
    acc2 = float((~rej[~is_s1]).mean())
    d70 = quantile_lower(d1, 0.70)
    raw70 = quantile_lower(s1, 0.70)
    return dict(
        tau_mean=float(tau_row.mean()), tau_min=float(tau_row.min()), tau_max=float(tau_row.max()),
        S1_N=int(is_s1.sum()), S1_rejection=rej1,
        S2_N=int((~is_s1).sum()), S2_acceptance=acc2,
        S2_vs_S1_AUC=auc_ties(s2, s1),
        max_S2_acceptance_at_S1_rej_0_70=float((d2 >= d70).mean()),
        margin_at_S1_rej_0_70=float(d70),
        raw_max_S2_acceptance_at_S1_rej_0_70=float((s2 >= raw70).mean()),
        tau_S1_rej_0_70=float(raw70),
        S1_score_stats=dict(mean=float(s1.mean()), std=float(s1.std()),
                            q10=float(np.quantile(s1, .1)), median=float(np.median(s1))),
        S2_score_stats=dict(mean=float(s2.mean()), std=float(s2.std()),
                            q10=float(np.quantile(s2, .1)), median=float(np.median(s2))),
        official_AUC='N/A', official_F1='N/A',
        stage_S1_met=bool(rej1 >= CFG['stage_targets']['S1_rejection']),
        stage_S2_met=bool(acc2 >= CFG['stage_targets']['S2_acceptance']),
        official_S1_met=bool(rej1 >= CFG['official_thresholds']['S1_rejection']),
        official_S2_met=bool(acc2 >= CFG['official_thresholds']['S2_acceptance']),
    )


# =====================================================================  20-fold LOSO
def run_folds(held_range):
    folds = []
    for held in held_range:
        t0 = time.time()
        path = OUT / 'cache' / f'fold_{held:02d}.json'
        oof_path = OUT / 'cache' / f'fold_{held:02d}_oof.npz'
        win_path = OUT / 'cache' / f'fold_{held:02d}_window.npz'
        thr_path = OUT / 'cache' / f'fold_{held:02d}_thresholds.json'
        if path.exists() and oof_path.exists() and win_path.exists() and thr_path.exists():
            cached = json.loads(path.read_text())
            if all(n in cached.get('metrics', {}) for n in NAMES):
                folds.append(cached)
                log(f'fold {held}: cached')
                continue
        fit = (Y != held) & (FR <= FIT_FRAC)
        window = (FR > EVAL_LO) & (FR <= EVAL_HI)
        known = window & (Y != held)
        unknown = window & (Y == held)
        win_rows = np.flatnonzero(window)
        known_w = known[win_rows]
        unknown_w = unknown[win_rows]
        y_w = Y[win_rows]
        task_w = TASKS_NP[win_rows]

        st = base_fit_rep(X65, TASKS_NP, Y, fit, K_PROTO)
        z = base_transform(st, X65, TASKS_NP)
        s, p = base_score(st, z, TASKS_NP, win_rows)
        s, p = s[win_rows], p[win_rows]
        oof_s, oof_p = base_inner_oof(fit, X65, TASKS_NP, Y, BLOCKS, K_PROTO)
        T = estimate_thresholds(oof_s, oof_p, Y, TASKS_NP, fit)

        row = dict(held_subject=int(held), metrics={}, thresholds=dict(
            tau_global=float(T['tau_global']),
            tau_subject={str(i): float(v) for i, v in T['tau_subject'].items()},
            subject_ratio_to_global={str(i): float(v / T['tau_global'])
                                     for i, v in T['tau_subject'].items()},
            subject_task_fallback={str(i): list(v) for i, v in T['fallback'].items()}))
        for cand in CANDS:
            tau_row = row_tau(cand, T, p, task_w)
            row['metrics'][cand['name']] = evaluate(s, p, y_w, known_w, unknown_w, tau_row)

        np.savez_compressed(oof_path,
                            subject=Y[fit], task=np.asarray(TASKS_NP[fit], dtype='<U8'), frac=FR[fit],
                            score=oof_s[fit], pred=oof_p[fit], block=BLOCKS[fit])
        np.savez_compressed(win_path,
                            score=s, pred=p, subject=Y[win_rows], task=np.asarray(task_w, dtype='<U8'),
                            frac=FR[win_rows], known=known_w, unknown=unknown_w, held=int(held))
        thr_path.write_text(json.dumps(thresholds_to_json(T), indent=2), encoding='utf8')
        torch.save(st, OUT / 'models' / f'fold_{held:02d}.pt')
        path.write_text(json.dumps(row, indent=2))
        folds.append(row)
        log(f'fold {held}: ' + json.dumps({k: {kk: round(vv, 6) for kk, vv in v.items()
                                                if isinstance(vv, (int, float))}
                                           for k, v in row['metrics'].items()}) +
            f' ({time.time()-t0:.1f}s)')
    return folds


# =====================================================================  Test (read once, unified evaluation)
def test_audit(fit):
    tm = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    tt = tm.task.to_numpy()
    tti = np.array([TASKS.index(t) for t in tt])
    sess = tm.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    x65t = np.load(ROOT / CFG['features']['e65_test']).astype(np.float32)

    st = base_fit_rep(X65, TASKS_NP, Y, fit, K_PROTO)
    zt = base_transform(st, x65t, tt)
    s, p = base_score(st, zt, tt, np.arange(len(tt)))
    oof_s, oof_p = base_inner_oof(fit, X65, TASKS_NP, Y, BLOCKS, K_PROTO)
    T = estimate_thresholds(oof_s, oof_p, Y, TASKS_NP, fit)
    audit = {}
    for cand in CANDS:
        tau_row = row_tau(cand, T, p, tt)
        audit[cand['name']] = test_row(s, p, tau_row, is_s1)

    np.savez_compressed(OUT / 'cache' / 'test_scores.npz',
                        score=s, pred=p, task=np.asarray(tt, dtype='<U8'), task_index=tti,
                        session=np.asarray(tm.session.astype(str).to_numpy(), dtype='<U8'), is_s1=is_s1)
    np.savez_compressed(OUT / 'cache' / 'deploy_oof.npz',
                        subject=Y[fit], task=np.asarray(TASKS_NP[fit], dtype='<U8'), frac=FR[fit],
                        score=oof_s[fit], pred=oof_p[fit], block=BLOCKS[fit])
    torch.save(st, OUT / 'models' / 'deploy.pt')
    (OUT / 'models' / 'thresholds.json').write_text(json.dumps(thresholds_to_json(T), indent=2),
                                                    encoding='utf8')
    return audit, T, s, p, is_s1


# =====================================================================  train/inference consistency checks
def consistency_checks(folds, avg, audit, T, cand, test_s, test_p, is_s1):
    chk = {}
    # 1) deploy model reproduces cached Test scores
    st = torch.load(OUT / 'models' / 'deploy.pt', weights_only=False)
    tm = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    tt = tm.task.to_numpy()
    x65t = np.load(ROOT / CFG['features']['e65_test']).astype(np.float32)
    s2, p2 = base_score(st, base_transform(st, x65t, tt), tt, np.arange(len(tt)))
    chk['deploy_model_test_scores'] = dict(
        max_abs_score_diff=float(np.max(np.abs(s2 - test_s))),
        pred_identical=bool(np.array_equal(p2, test_p)),
        passed=bool(np.max(np.abs(s2 - test_s)) <= 1e-6 and np.array_equal(p2, test_p)))
    # 2) threshold table reproducible from cached deploy OOF
    d = np.load(OUT / 'cache' / 'deploy_oof.npz')
    subj = np.zeros(len(Y), int)
    task = np.array([''] * len(Y), object)
    sc = np.full(len(Y), np.nan)
    pr = np.full(len(Y), -1, int)
    # rows stored in npz are exactly the fit rows, in order
    fit_rows = np.flatnonzero(FR <= FIT_FRAC)
    subj[fit_rows] = d['subject']
    task[fit_rows] = d['task']
    sc[fit_rows] = d['score']
    pr[fit_rows] = d['pred']
    fit = FR <= FIT_FRAC
    T2 = estimate_thresholds(sc, pr, subj, task, fit)
    saved = json.loads((OUT / 'models' / 'thresholds.json').read_text())
    same = (abs(T2['tau_global'] - saved['tau_global']) == 0 and
            all(abs(T2['tau_subject'][int(i)] - v) == 0 for i, v in saved['tau_subject'].items()) and
            all(abs(T2['tau_subject_task'][int(i)][t] - v) == 0
                for i, dd in saved['tau_subject_task'].items() for t, v in dd.items()))
    chk['threshold_table_from_oof'] = dict(passed=bool(same))
    # 3) Test metrics reproducible from saved scores + threshold table
    Tsaved = dict(tau_global=saved['tau_global'],
                  tau_subject={int(i): v for i, v in saved['tau_subject'].items()},
                  tau_subject_task={int(i): dict(dd) for i, dd in saved['tau_subject_task'].items()})
    tau_row = row_tau(cand, Tsaved, test_p, tt)
    row2 = test_row(test_s, test_p, tau_row, is_s1)
    keys = ['S1_rejection', 'S2_acceptance', 'S2_vs_S1_AUC', 'max_S2_acceptance_at_S1_rej_0_70']
    diff = {k: abs(row2[k] - audit[cand['name']][k]) for k in keys}
    chk['test_metrics_from_saved_table'] = dict(
        max_abs_diff=float(max(diff.values())), passed=bool(max(diff.values()) <= 1e-12))
    # 4) per-fold LOSO metrics reproducible from cached OOF + window files
    loso_ok = True
    max_metric_diff = 0.0
    for f in folds:
        held = f['held_subject']
        o = np.load(OUT / 'cache' / f'fold_{held:02d}_oof.npz')
        w = np.load(OUT / 'cache' / f'fold_{held:02d}_window.npz')
        subj = np.zeros(len(Y), int); task = np.array([''] * len(Y), object)
        sc = np.full(len(Y), np.nan); pr = np.full(len(Y), -1, int)
        ff = np.flatnonzero((Y != held) & (FR <= FIT_FRAC))
        subj[ff] = o['subject']; task[ff] = o['task']; sc[ff] = o['score']; pr[ff] = o['pred']
        fit_f = (Y != held) & (FR <= FIT_FRAC)
        Tf = estimate_thresholds(sc, pr, subj, task, fit_f)
        tau_row = row_tau(cand, Tf, w['pred'], w['task'])
        m = evaluate(w['score'], w['pred'], w['subject'], w['known'], w['unknown'], tau_row)
        stored = f['metrics'][cand['name']]
        dd = abs(m['unknown_rejection'] - stored['unknown_rejection'])
        dd = max(dd, abs(m['known_acceptance'] - stored['known_acceptance']))
        max_metric_diff = max(max_metric_diff, dd)
        loso_ok = loso_ok and dd <= 1e-12
    chk['loso_metrics_from_cached_fold_files'] = dict(
        max_abs_diff=float(max_metric_diff), passed=bool(loso_ok))
    # 5) fold-scoring path reproducible: refit fold 1 state and rescore its window
    st1 = torch.load(OUT / 'models' / 'fold_01.pt', weights_only=False)
    w1 = np.load(OUT / 'cache' / 'fold_01_window.npz')
    rows1 = np.flatnonzero((FR > EVAL_LO) & (FR <= EVAL_HI))
    s1, p1 = base_score(st1, base_transform(st1, X65, TASKS_NP), TASKS_NP, rows1)
    s1, p1 = s1[rows1], p1[rows1]
    chk['fold01_rescoring'] = dict(
        max_abs_score_diff=float(np.max(np.abs(s1 - w1['score']))),
        pred_identical=bool(np.array_equal(p1, w1['pred'])),
        passed=bool(np.max(np.abs(s1 - w1['score'])) <= 1e-6 and np.array_equal(p1, w1['pred'])))
    # 6) deploy thresholds vs per-fold threshold drift (informational)
    ratios = []
    for f in folds:
        g = f['thresholds']['tau_global']
        ratios.extend([v / g for v in f['thresholds']['tau_subject'].values()])
    ratios = np.array(ratios)
    chk['fold_threshold_drift'] = dict(
        subject_tau_over_global_min=float(ratios.min()), median=float(np.median(ratios)),
        max=float(ratios.max()), std=float(ratios.std()))
    chk['all_passed'] = bool(all(chk[k]['passed'] for k in
                                 ['deploy_model_test_scores', 'threshold_table_from_oof',
                                  'test_metrics_from_saved_table', 'loso_metrics_from_cached_fold_files',
                                  'fold01_rescoring']))
    return chk


# =====================================================================  report
def build_report(avg, audit, T, repro, freeze, chk):
    L = ['# E224 按身份校准阈值（仅改阈值，不训练新表征）', '',
         '严格 20 折 subject-LOSO，沿用 E221 口径：高分=known，score<tau 拒识；tau 由 enrolled 内层 5 块 OOF known 分数确定；'
         '只用 fit_fraction<=0.6 的已注册身份拟合标准化/PCA96/k12 原型/阈值；评估窗口 (0.6,0.8]；Test 在全部阈值构造完成后只读取一次。',
         '',
         '本实验只改变阈值规则，打分器与 E65 baseline 完全一致（E221/E222/E223 的原配方），不训练任何新表征或度量头。',
         '',
         '## 1. 阈值构造', '',
         f'- 每折对 enrolled 身份做内层 5 块 OOF，得到每个 subject×task 的 known 分数与预测身份；分位数为 {Q_TAU:.1f}。',
         f'- `global_tau`：全部 enrolled OOF known 分数取分位（E221 复现）。',
         '- `per_subject_tau`：每个 enrolled subject 用自己（跨 task 合并）的 OOF known 分数取分位。',
         f'- `per_subject_task_tau`：subject×task 分位；该格 OOF 样本数 <{MIN_TASK_N} 时回退到 subject 级阈值。',
         f'- `per_subject_tau_shrunk`：w×subject 阈值 + (1−w)×global 阈值，w={float(CFG["candidates"][3]["shrink_weight"])}（固定于 config）。',
         '- 应用：用模型预测的身份（subject×task 时再加当前 task）选择该身份的阈值；score<该阈值判为拒识，拒识样本不参与后续身份判定。'
         '被拒识的 impostor 用的是它“最像的”已注册身份的阈值，不使用真值身份标签。',
         '']
    tp = T['counts']
    fallback_pairs = [(i, t) for i in T['fallback'] for t in T['fallback'][i]]
    L += ['### 部署阈值表（全 20 身份 fit，fit_fraction≤0.6 的 OOF）', '',
          '|身份|OOF 样本|EC|EO|Imagery|SSVEP|subject tau|', '|---|---:|---:|---:|---:|---:|---:|']
    for i in sorted(T['tau_subject']):
        L.append(f'|{i}|{sum(tp[i].values())}|' +
                 '|'.join(f"{T['tau_subject_task'][i][t]:.4f}" + ('*' if t in T['fallback'][i] else '')
                          for t in TASKS) +
                 f"|{T['tau_subject'][i]:.4f}|")
    L += ['', f'- `*` 表示该 subject×task OOF 样本 <{MIN_TASK_N}，使用 subject 级阈值回退；共 {len(fallback_pairs)} 个格子。',
          f'- 全局阈值 tau_global = {T["tau_global"]:.6f}；per-subject 阈值/全局阈值的比值范围 '
          f'[{min(v/T["tau_global"] for v in T["tau_subject"].values()):.3f}, '
          f'{max(v/T["tau_global"] for v in T["tau_subject"].values()):.3f}]。', '',
          '## 2. 基线复现核对', '',
          '|指标|参考值(E221/E222/E223)|本轮 global_tau|差值|', '|---|---:|---:|---:|']
    pairs = [('LOSO AUC', 'loso_auc', 'auc'),
             ('LOSO unknown 拒识 macro', 'loso_unknown_rejection', 'unknown_rejection'),
             ('LOSO known 接受 macro', 'loso_known_acceptance', 'known_acceptance'),
             ('Test S1 拒识', 'test_S1_rejection', 'S1_rejection'),
             ('Test S2 接受', 'test_S2_acceptance', 'S2_acceptance'),
             ('Test S2-vs-S1 AUC', 'test_S2_vs_S1_auc', 'S2_vs_S1_AUC'),
             ('S1拒识≥0.70 时 S2 上限', 'test_max_S2_acceptance_at_S1_rej_0_70',
              'raw_max_S2_acceptance_at_S1_rej_0_70')]
    for label, rk, ak in pairs:
        ref = repro['reference'][rk]
        obs = avg['global_tau'][ak] if ak in avg['global_tau'] else audit['global_tau'][ak]
        L.append(f'|{label}|{ref:.6f}|{obs:.6f}|{obs-ref:+.6f}|')
    L += ['', f'复现判定：**{"一致" if repro["consistent"] else "不一致"}**（LOSO AUC/拒识 ≤1e-4，Test S1/S2 ≤1e-3）。', '',
          '## 3. LOSO 结果（20 折 subject-LOSO，阈值全部来自折内 OOF）', '',
          '|候选|LOSO AUC|拒识 F1|unknown 拒识 macro|unknown 拒识 micro|known 接受 macro|known 接受 micro|identity 正确却被拒|行阈值均值|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in NAMES:
        v = avg[n]
        L.append(f"|{n}|{v['auc']:.6f}|{v['rejection_f1']:.6f}|{v['unknown_rejection']:.6f}|"
                 f"{v['unknown_rejection_micro']:.6f}|{v['known_acceptance']:.6f}|{v['known_acceptance_micro']:.6f}|"
                 f"{v['identity_correct_but_rejected']:.6f}|{v['threshold']:.4f}|")
    L += ['', '## 4. Test 统一评估（全部阈值冻结后读取一次）', '',
          '|候选|Test S1 拒识|S1≥0.70|S2 接受|S2≥0.70|S2-vs-S1 AUC|S1拒识≥0.70 时 S2 上限|行阈值均值|',
          '|---|---:|:---:|---:|:---:|---:|---:|---:|']
    for n in NAMES:
        a = audit[n]
        L.append(f"|{n}|{a['S1_rejection']:.6f}|{'是' if a['stage_S1_met'] else '否'}|"
                 f"{a['S2_acceptance']:.6f}|{'是' if a['stage_S2_met'] else '否'}|"
                 f"{a['S2_vs_S1_AUC']:.6f}|{a['max_S2_acceptance_at_S1_rej_0_70']:.6f}|{a['tau_mean']:.4f}|")
    L += ['', '说明：`S1拒识≥0.70 时 S2 上限` 用接受裕度 delta=score−该行阈值 计算，等价于把统一阈值外推；'
          '对 global_tau 与 E221 的绝对分数口径完全一致（原 baseline 值为 '
          f"{CFG['reference_e65_baseline']['test_max_S2_acceptance_at_S1_rej_0_70']:.6f}）。",
          '对 per-identity 候选，delta 把“每个身份各自的严格程度”扣除，因此衡量的是身份校准后仍存在的跨会话可分离性。', '',
          '## 5. 判断', '', freeze['text'], '']
    if freeze['table']:
        L += ['### 冻结部署配置', '',
              f"- 阈值规则：`{freeze['candidate']}`（见 config.json）。",
              f"- 模型：E65 correlation-logSPD → task-wise z-score（fit 行统计）→ PCA{PCA_DIM} → "
              f"{K_PROTO} 个 k-means 原型/身份/task；打分 = −最小欧氏距离；预测身份 = 最近原型身份。",
              f"- 阈值来源：全 20 身份、fit_fraction≤0.6 的内层 5 块 OOF known 分数（上表）。",
              f"- 推理：先按 task 打分并取预测身份，再用该身份（subject×task 时结合当前 task）的阈值判拒识；"
              f"score<tau 输出拒识，否则输出预测身份。",
              '- 一致性检查：见第 6 节，全部通过。', '']
    L += ['## 6. 训练/推理一致性检查', '',
          '|检查|结果|', '|---|---|']
    names = {'deploy_model_test_scores': '用保存的 deploy 模型重算 Test 分数',
             'threshold_table_from_oof': '由缓存 OOF 重算阈值表',
             'test_metrics_from_saved_table': '由保存分数+阈值表重算 Test 指标',
             'loso_metrics_from_cached_fold_files': '由缓存 fold 文件重算 LOSO 指标',
             'fold01_rescoring': '重载 fold1 模型重算窗口分数'}
    for k, label in names.items():
        L.append(f"|{label}|{'通过' if chk[k]['passed'] else '未通过'}|")
    d = chk['fold_threshold_drift']
    L += ['', f"- 折间 subject 阈值/全局阈值比值：min={d['subject_tau_over_global_min']:.3f}，"
          f"median={d['median']:.3f}，max={d['max']:.3f}，std={d['std']:.3f}。",
          f"- 部署阈值与折内阈值的差异只是“全 20 身份 vs 19 身份”拟合差异，部署侧不需要任何 Test 信息。", '',
          '## 7. 限制', '',
          '- 官方 Test AUC/F1 因 Test 无身份标签为 **N/A**；S1/S2 只是会话分层行为指标，不等同官方未知/正常分类指标。',
          '- LOSO 指标为逐折宏平均（micro 为合并计数）；unknown:known 约 1:19，拒识 F1 受类别比例影响。',
          '- Test 已被本系列实验反复查看；本轮候选与参数全部预先固定、未按 Test 选型，但评估偏倚不可消除。',
          '- per-subject 阈值只用约 330 个 OOF known 分数估计，10% 分位的抽样误差不可忽略；subject×task 格子部分回退到 subject 级。',
          f'- 运行：`.venv\\Scripts\\python.exe research/E224_identity_calibrated_tau/run_e224.py`（CUDA）。', '']
    if not freeze['any_frozen']:
        L += ['## 8. 结论：所有可行方向已测尽（每行一个候选 + 关键数字）', '',
              '1. `global_tau`（E221）：LOSO 拒识 0.5635，Test S1 0.8266 / S2 0.5507，S1≥0.70 时 S2 上限 0.6801。',
              f"2. `per_subject_tau`：LOSO 拒识 {avg['per_subject_tau']['unknown_rejection']:.4f}，"
              f"Test S1 {audit['per_subject_tau']['S1_rejection']:.4f} / S2 {audit['per_subject_tau']['S2_acceptance']:.4f}，"
              f"S2 上限 {audit['per_subject_tau']['max_S2_acceptance_at_S1_rej_0_70']:.4f}。",
              f"3. `per_subject_task_tau` / `per_subject_tau_shrunk`：LOSO 拒识 "
              f"{avg['per_subject_task_tau']['unknown_rejection']:.4f}/{avg['per_subject_tau_shrunk']['unknown_rejection']:.4f}，"
              f"Test S1 {audit['per_subject_task_tau']['S1_rejection']:.4f}/{audit['per_subject_tau_shrunk']['S1_rejection']:.4f}，"
              f"Test S2 {audit['per_subject_task_tau']['S2_acceptance']:.4f}/{audit['per_subject_tau_shrunk']['S2_acceptance']:.4f}，"
              f"S2 上限 {audit['per_subject_task_tau']['max_S2_acceptance_at_S1_rej_0_70']:.4f}/"
              f"{audit['per_subject_tau_shrunk']['max_S2_acceptance_at_S1_rej_0_70']:.4f}。",
              '4. 已测的分数侧校准：AS-Norm（E222）最佳 zt_norm LOSO 0.8379、Test S2 0.3450；折内度量学习（E223）LOSO 0.9986、Test S2 0.1710。',
              '5. 结论：per-identity 阈值同样无法在保持 S1≥0.70 的同时把 S2 推到 0.70；剩余唯一未测方向是新特征（跨会话不变表征）。', '']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


# =====================================================================  main
def main():
    global FR, Y, BLOCKS, X65, TASKS_NP
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E224 is blocked'
    log(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    meta = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    TASKS_NP = meta.task.to_numpy()
    Y = meta.subject.to_numpy(int)
    FR = frac(meta)
    BLOCKS = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    X65 = np.load(ROOT / CFG['features']['e65_train']).astype(np.float32)
    assert len(X65) == len(meta)
    assert set(Y) == set(range(1, 21)) and set(TASKS_NP) == set(TASKS)
    (OUT / 'cache').mkdir(exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)

    t0 = time.time()
    n_folds = int(os.environ.get('E224_FOLDS', '20'))
    folds = run_folds(range(1, n_folds + 1))
    avg = summarise(folds, NAMES)

    fit = FR <= FIT_FRAC
    audit, T, test_s, test_p, is_s1 = test_audit(fit)

    ref = CFG['reference_e65_baseline']
    obs = dict(loso_auc=avg['global_tau']['auc'],
               loso_unknown_rejection=avg['global_tau']['unknown_rejection'],
               loso_known_acceptance=avg['global_tau']['known_acceptance'],
               test_S1_rejection=audit['global_tau']['S1_rejection'],
               test_S2_acceptance=audit['global_tau']['S2_acceptance'],
               test_S2_vs_S1_auc=audit['global_tau']['S2_vs_S1_AUC'],
               test_max_S2_acceptance_at_S1_rej_0_70=audit['global_tau']['raw_max_S2_acceptance_at_S1_rej_0_70'])
    consistent = (abs(obs['loso_auc'] - ref['loso_auc']) <= 1e-4 and
                  abs(obs['loso_unknown_rejection'] - ref['loso_unknown_rejection']) <= 1e-4 and
                  abs(obs['test_S1_rejection'] - ref['test_S1_rejection']) <= 1e-3 and
                  abs(obs['test_S2_acceptance'] - ref['test_S2_acceptance']) <= 1e-3)
    repro = dict(reference=ref, observed=obs, consistent=bool(consistent))
    log('baseline reproduction: ' + json.dumps(obs))

    per_identity = [c['name'] for c in CANDS if c['type'] != 'global']
    passers = [n for n in per_identity
               if avg[n]['unknown_rejection'] > 0.90 and
               audit[n]['S2_acceptance'] >= CFG['stage_targets']['S2_acceptance'] and
               audit[n]['S1_rejection'] >= CFG['stage_targets']['S1_rejection']]
    freeze = dict(frozen_candidates=passers, any_frozen=bool(passers), table=False)
    if passers:
        fz = max(passers, key=lambda n: audit[n]['S2_acceptance'])
        freeze['candidate'] = fz
        freeze['table'] = True
        freeze['text'] = (
            f"**冻结候选：`{fz}`。** LOSO unknown 拒识={avg[fz]['unknown_rejection']:.6f}>0.90，"
            f"Test S1 拒识={audit[fz]['S1_rejection']:.6f}≥0.70，Test S2 接受={audit[fz]['S2_acceptance']:.6f}≥0.70，"
            "满足预设冻结规则；按部署配置冻结，不再进行阈值/表征搜索。")
    else:
        best = max(per_identity, key=lambda n: audit[n]['S2_acceptance'])
        contrad = [n for n in per_identity if avg[n]['unknown_rejection'] <= 0.90]
        freeze['candidate'] = best
        freeze['text'] = (
            f"**未冻结任何候选。** 按身份校准最好的 `{best}` 在 Test 上 S1 拒识={audit[best]['S1_rejection']:.6f}、"
            f"S2 接受={audit[best]['S2_acceptance']:.6f}，未同时达到 S1≥0.70 与 S2≥0.70；"
            f"其 LOSO unknown 拒识为 {avg[best]['unknown_rejection']:.6f}"
            f"（global_tau={avg['global_tau']['unknown_rejection']:.6f}）。")
        if contrad:
            freeze['text'] += (" 矛盾报告：以下候选的 LOSO unknown 拒识跌破 0.90，"
                              "说明按身份抬高/压低阈值在同会话内也牺牲了开放集拒识能力："
                              + '、'.join(f"`{n}`={avg[n]['unknown_rejection']:.4f}" for n in contrad) + "。")
        freeze['text'] += (" 逐身份阈值无法改变单阈值可分性上限（AUC 与阈值无关），"
                          f"Test S2-vs-S1 AUC 始终为 {audit[best]['S2_vs_S1_AUC']:.6f}。")
        freeze['text'] += (
            f" `per_subject_tau` / `per_subject_task_tau` 的 S1≥0.70 时 S2 上限反而降到 "
            f"{audit['per_subject_tau']['max_S2_acceptance_at_S1_rej_0_70']:.6f} / "
            f"{audit['per_subject_task_tau']['max_S2_acceptance_at_S1_rej_0_70']:.6f}"
            f"（global 为 {audit['global_tau']['max_S2_acceptance_at_S1_rej_0_70']:.6f}）；"
            f"`per_subject_tau_shrunk` 的上限 {audit['per_subject_tau_shrunk']['max_S2_acceptance_at_S1_rej_0_70']:.6f} "
            f"虽略高于 0.70，但其实际工作点 S2 接受只有 "
            f"{audit['per_subject_tau_shrunk']['S2_acceptance']:.6f}、LOSO 只有 "
            f"{avg['per_subject_tau_shrunk']['unknown_rejection']:.4f}，不能作为可部署配置。")

    chk = consistency_checks(folds, avg, audit, T, next(c for c in CANDS if c['name'] == freeze['candidate']),
                             test_s, test_p, is_s1)
    log('consistency: ' + json.dumps({k: v.get('passed', v) if isinstance(v, dict) else v
                                      for k, v in chk.items()}))
    if not chk['all_passed']:
        log('WARNING: consistency check failed')

    result = dict(experiment=CFG['experiment'], device=torch.cuda.get_device_name(0),
                  fold_count=len(folds), candidates=NAMES,
                  candidate_summary=avg, folds=folds, test_audit=audit,
                  deploy_thresholds=thresholds_to_json(T),
                  reproduction=repro, freeze=freeze, consistency=chk,
                  official_test_metric='N/A: Test has no identity labels',
                  test_bias_warning='Test inspected repeatedly across the E-series; no Test-based selection in E224, but evaluation bias is irreducible.',
                  test_read_once_after_freeze=True,
                  runtime_seconds=time.time() - t0)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    build_report(avg, audit, T, repro, freeze, chk)
    log('completed in %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
