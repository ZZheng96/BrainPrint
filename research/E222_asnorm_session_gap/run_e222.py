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
TASKS = CFG['tasks']
DEVICE = 'cuda'
PCA_DIM = int(CFG['pca_dim'])
K = int(CFG['prototypes_k'])
Q_TAU = float(CFG['threshold_quantile'])
FIT_FRAC = float(CFG['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['evaluation_fraction']
N_BLOCKS = int(CFG['inner_blocks'])
SPECS = [dict(s) for s in CFG['candidates']]
E221 = ROOT / 'research/E221_metric_corrected'
LOG = OUT / 'run.log'
FR = None  # global trial-fraction vector


def log(s):
    print(s, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(s) + '\n')


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def frac(m):
    return m.trial.to_numpy(float) / m.groupby(['subject', 'task']).trial.transform('max').to_numpy(float).clip(1)


# ---------------------------------------------------------------- baseline representation (identical to E221 baseline)
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


def prototypes(z, subjects, tasks, mask, classes, k):
    ps, keys = [], []
    for task in TASKS:
        for sub in classes:
            m = mask & (tasks == task) & (subjects == sub)
            if m.any():
                for p in km(z[m], k, int(sub) * 17):
                    ps.append(p)
                    keys.append((task, int(sub)))
    return np.stack(ps).astype(np.float32), keys


def fit_rep(raw, tasks, y, fit, k):
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
    p, keys = prototypes(z, y, tasks, fit, cls, k)
    return dict(mu=mu, sd=sd, pm=pm.cpu().numpy(), W=W.cpu().numpy(), p=p, keys=keys, k=k)


def transform(state, raw, tasks):
    x = tensor(raw)
    z = torch.empty_like(x)
    for i, t in enumerate(TASKS):
        rows = tasks == t
        z[rows] = (x[rows] - tensor(state['mu'][i])) / tensor(state['sd'][i])
    z = ((z.double() - tensor(state['pm'], torch.float64)) @ tensor(state['W'], torch.float64)).float()
    return z.cpu().numpy()


def proto_meta(state):
    keys = state['keys']
    return np.array([t for t, _ in keys]), np.array([int(s) for _, s in keys], int)


def score_parts_state(state, z, rows, task_rows):
    """Per-task prototype score matrix S = -distance plus per-column identity labels."""
    proto_task, proto_id = proto_meta(state)
    parts = []
    for t in TASKS:
        rr = rows[task_rows == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        S = (-torch.cdist(tensor(z[rr]), tensor(state['p'][cols]))).double().cpu().numpy()
        parts.append((t, rr, S, proto_id[cols]))
    return parts


def calib_stats(fit_parts, y):
    """z-norm: per identity-model mean/std of impostor scores, estimated on outer-train fit utterances only."""
    stats = {}
    for t, rr, S, ident in fit_parts:
        yf = y[rr]
        ids = np.unique(ident)
        M = np.stack([S[:, ident == i].max(1) for i in ids], 1)
        sel = yf[:, None] != ids[None, :]
        mu = np.array([M[sel[:, j], j].mean() for j in range(len(ids))])
        sd = np.maximum(np.array([M[sel[:, j], j].std() for j in range(len(ids))]), 1e-6)
        stats[t] = (ids, mu, sd)
    return stats


def apply_norm(S, ident, mode, topk, stats_t=None):
    s1 = S.max(1)
    am = S.argmax(1)
    claimed = ident[am]
    if mode == 'raw':
        return s1
    if mode == 'asnorm':
        mask = ident[None, :] != claimed[:, None]
        C = np.where(mask, S, np.nan)
        return (s1 - np.nanmean(C, 1)) / np.maximum(np.nanstd(C, 1), 1e-6)
    if mode == 'asnorm_topk':
        mask = ident[None, :] != claimed[:, None]
        C = np.where(mask, S, -np.inf)
        k = min(int(topk), C.shape[1])
        idx = np.argpartition(-C, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(C, idx, 1)
        return (s1 - vals.mean(1)) / np.maximum(vals.std(1), 1e-6)
    if mode in ('znorm', 'zt'):
        ids, mu, sd = stats_t
        col = np.searchsorted(ids, ident)
        Z = (S - mu[col]) / sd[col]
        z1 = Z[np.arange(len(Z)), am]
        if mode == 'znorm':
            return z1
        mask = ident[None, :] != claimed[:, None]
        Zc = np.where(mask, Z, np.nan)
        return (z1 - np.nanmean(Zc, 1)) / np.maximum(np.nanstd(Zc, 1), 1e-6)
    raise ValueError(mode)


def score_candidate(state, z, tasks, spec, rows, y=None, fit_rows=None,
                    stat_z=None, stat_tasks=None, stat_rows=None):
    """Score `rows`; z-norm statistics may come from a different (train) representation via stat_z."""
    mode, topk = spec['mode'], int(spec['topk'])
    parts = score_parts_state(state, z, rows, tasks[rows])
    stats = {}
    if mode in ('znorm', 'zt'):
        sz = z if stat_z is None else stat_z
        stasks = tasks if stat_tasks is None else stat_tasks
        srows = fit_rows if stat_rows is None else stat_rows
        stats = calib_stats(score_parts_state(state, sz, srows, stasks[srows]), y)
    s = np.full(len(tasks), np.nan)
    p = np.full(len(tasks), -1, int)
    for t, rr, S, ident in parts:
        s[rr] = apply_norm(S, ident, mode, topk, stats.get(t))
        p[rr] = ident[S.argmax(1)]
    return s, p


# ---------------------------------------------------------------- metrics
def evaluate(s, p, y, known, unknown, tau):
    k = s[known]
    u = s[unknown]
    rej_u = (u < tau)
    rej_k = (k < tau)
    tp = float(rej_u.sum())
    fp = float(rej_k.sum())
    fn = float((~rej_u).sum())
    return dict(
        auc=float(((u[:, None] < k).sum() + .5 * (u[:, None] == k).sum()) / (len(k) * len(u))),
        rejection_f1=float(2 * tp / (2 * tp + fp + fn)),
        known_acceptance=float((~rej_k).mean()),
        unknown_rejection=float(rej_u.mean()),
        identity_correct_but_rejected=float(((p[known] == y[known]) & rej_k).mean()),
        threshold=float(tau),
        n_known=int(len(k)), n_unknown=int(len(u)),
        accepted_known=int((~rej_k).sum()), rejected_unknown=int(rej_u.sum()),
    )


def auc_ties(pos, neg):
    """AUC with positive=pos (higher score = more positive), tie-aware."""
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


def quantile_lower(x, q):
    try:
        return float(np.quantile(x, q, method='lower'))
    except TypeError:
        return float(np.quantile(x, q, interpolation='lower'))


# ---------------------------------------------------------------- phase 1: diagnosis with E221 cached Test scores only
def phase_diagnostic():
    tm = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    sess = tm.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    z = np.load(E221 / 'cache' / 'baseline_test_scores.npz')
    s = z['score'].astype(float)
    a, b = s[is_s1], s[~is_s1]

    def stats(v):
        return dict(n=int(len(v)), mean=float(v.mean()), std=float(v.std()),
                    min=float(v.min()), q10=float(np.quantile(v, .1)),
                    median=float(np.median(v)), q90=float(np.quantile(v, .9)), max=float(v.max()))

    def point(tau):
        return dict(tau=float(tau),
                    S1_rejection=float((a < tau).mean()),
                    S2_acceptance=float((b >= tau).mean()))

    grid_q = np.unique(np.r_[np.linspace(0.0, 1.0, 101), 0.5])
    taus = np.unique(np.quantile(s, grid_q))
    curve = [point(t) for t in taus]

    cond = []
    for rej in [CFG['stage_targets']['S1_rejection'], CFG['official_thresholds']['S1_rejection']]:
        tau = quantile_lower(a, rej)
        cond.append(dict(constraint=f'S1_rejection>={rej:.2f}', tau=float(tau),
                         achieved_S1_rejection=float((a < tau).mean()),
                         max_S2_acceptance=float((b >= tau).mean())))
    for acc in [CFG['stage_targets']['S2_acceptance'], CFG['official_thresholds']['S2_acceptance']]:
        tau = quantile_lower(b, 1 - acc)
        cond.append(dict(constraint=f'S2_acceptance>={acc:.2f}', tau=float(tau),
                         achieved_S2_acceptance=float((b >= tau).mean()),
                         max_S1_rejection=float((a < tau).mean())))

    sep_auc = auc_ties(b, a)

    unk, loso_known, tau90_folds = [], [], []
    for held in range(1, 21):
        d = np.load(E221 / 'cache' / f'baseline_fold_{held:02d}_scores.npz')
        u = d['score'][d['unknown']].astype(float)
        unk.append(u)
        loso_known.append(d['score'][d['known']].astype(float))
        tau90_folds.append(float(np.quantile(u, 0.90)))
    unk = np.concatenate(unk)
    loso_known = np.concatenate(loso_known)
    tau90 = float(np.quantile(unk, 0.90))
    p90 = point(tau90)
    p90.update(tau90_pooled=tau90, tau90_per_fold_mean=float(np.mean(tau90_folds)),
               loso_unknown_n=int(len(unk)), loso_known_n=int(len(loso_known)),
               loso_unknown_score_stats=stats(unk), loso_known_score_stats=stats(loso_known))

    diag = dict(
        source='E221_metric_corrected/cache/baseline_*_scores.npz (frozen E221 selection already spent)',
        warning='Test-only diagnosis. Not for model/threshold selection.',
        test_score_stats=dict(S1=stats(a), S2=stats(b)),
        threshold_sweep=curve,
        conditional_points=cond,
        S2_vs_S1_single_threshold_AUC=sep_auc,
        loso_unknown_rejection_090=p90,
    )
    (OUT / 'cache').mkdir(exist_ok=True)
    (OUT / 'cache' / 'diagnostic.json').write_text(json.dumps(diag, indent=2), encoding='utf8')

    lines = ['', '=== PHASE 1 DIAGNOSTIC (E221 cached baseline Test scores; not for selection) ===',
             'test score stats: ' + json.dumps(diag['test_score_stats']),
             'S2(known)-vs-S1 single-threshold AUC = %.6f' % sep_auc,
             'conditional: ' + json.dumps(cond, indent=1),
             'LOSO tau@unknown_rej=0.90 -> Test: ' + json.dumps(p90, indent=1)]
    log('\n'.join(lines))
    return diag


# ---------------------------------------------------------------- phase 2: 20-fold strict LOSO
def inner_oof(spec, fit, raw, tasks, y, blocks, pk):
    oof = np.full(len(y), np.nan)
    for b in np.unique(blocks[fit]):
        hold = fit & (blocks == b)
        inner = fit & ~hold
        assert set(y[inner]) == set(y[fit])
        st = fit_rep(raw, tasks, y, inner, pk)
        z = transform(st, raw, tasks)
        s, _ = score_candidate(st, z, tasks, spec, np.flatnonzero(hold), y=y,
                               fit_rows=np.flatnonzero(inner))
        oof[hold] = s[hold]
    return oof


def run_folds(specs, raw, tasks, y, blocks, held_range=range(1, 21), use_cache=True):
    folds = []
    for held in held_range:
        t0 = time.time()
        path = OUT / 'cache' / f'fold_{held:02d}.json'
        cached = json.loads(path.read_text()) if (use_cache and path.exists()) else None
        if cached and all(sp['name'] in cached['metrics'] for sp in specs):
            folds.append(cached)
            log(f'fold {held}: cached')
            continue
        fit = (y != held) & (FR <= FIT_FRAC)
        window = (FR > EVAL_LO) & (FR <= EVAL_HI)
        known = window & (y != held)
        unknown = window & (y == held)
        pks = sorted({sp['proto_k'] for sp in specs})
        oof = {}
        for pk in pks:
            for sp in [q for q in specs if q['proto_k'] == pk]:
                oof[sp['name']] = inner_oof(sp, fit, raw, tasks, y, blocks, pk)
        row = dict(held_subject=held, metrics={})
        states = {}
        for pk in pks:
            st = fit_rep(raw, tasks, y, fit, pk)
            states[pk] = st
            z = transform(st, raw, tasks)
            rows = np.flatnonzero(window)
            for sp in [q for q in specs if q['proto_k'] == pk]:
                s, p = score_candidate(st, z, tasks, sp, rows, y=y, fit_rows=np.flatnonzero(fit))
                tau = float(np.quantile(oof[sp['name']][fit], Q_TAU))
                m = evaluate(s, p, y, known, unknown, tau)
                row['metrics'][sp['name']] = m
                np.savez_compressed(OUT / 'cache' / f"{sp['name']}_fold_{held:02d}_scores.npz",
                                    score=s, pred=p, known=known, unknown=unknown, subject=y)
            torch.save(st, OUT / 'models' / f'fold_{held:02d}_k{pk}.pt')
        if cached:
            row['metrics'] = {**cached['metrics'], **row['metrics']}
        path.write_text(json.dumps(row, indent=2))
        folds.append(row)
        log(f'fold {held}: ' + json.dumps({k: {kk: round(vv, 6) for kk, vv in v.items()}
                                          for k, v in row['metrics'].items()}) + f' ({time.time()-t0:.1f}s)')
    return folds


def summarise(folds, names):
    out = {}
    for name in names:
        ms = [f['metrics'][name] for f in folds]
        n_u = sum(m['n_unknown'] for m in ms)
        n_k = sum(m['n_known'] for m in ms)
        out[name] = dict(
            auc=float(np.mean([m['auc'] for m in ms])),
            auc_fold_std=float(np.std([m['auc'] for m in ms])),
            rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
            known_acceptance=float(np.mean([m['known_acceptance'] for m in ms])),
            known_acceptance_micro=float(sum(m['accepted_known'] for m in ms) / n_k),
            unknown_rejection=float(np.mean([m['unknown_rejection'] for m in ms])),
            unknown_rejection_micro=float(sum(m['rejected_unknown'] for m in ms) / n_u),
            identity_correct_but_rejected=float(np.mean([m['identity_correct_but_rejected'] for m in ms])),
            threshold=float(np.mean([m['threshold'] for m in ms])),
        )
    return out


def test_audit(specs, raw, tasks, y, fit, raw_test, tm):
    tt = tm.task.to_numpy()
    sess = tm.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    blocks = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    audit = {}
    for pk in sorted({sp['proto_k'] for sp in specs}):
        st = fit_rep(raw, tasks, y, fit, pk)
        zt = transform(st, raw_test, tt)
        ztr = transform(st, raw, tasks)
        fit_rows = np.flatnonzero(fit)
        for sp in [q for q in specs if q['proto_k'] == pk]:
            oof = inner_oof(sp, fit, raw, tasks, y, blocks, pk)
            tau = float(np.quantile(oof[fit], Q_TAU))
            s, p = score_candidate(st, zt, tt, sp, np.arange(len(tt)), y=y,
                                   stat_z=ztr, stat_tasks=tasks, stat_rows=fit_rows)
            pred = np.where(s >= tau, p, 0)
            s1, s2 = s[is_s1], s[~is_s1]
            rej1 = float((pred[is_s1] == 0).mean())
            acc2 = float((pred[~is_s1] != 0).mean())
            audit[sp['name']] = dict(
                tau=tau,
                S1_N=int(is_s1.sum()), S1_rejection=rej1,
                S2_N=int((~is_s1).sum()), S2_acceptance=acc2,
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
            np.savez_compressed(OUT / 'cache' / f'{sp["name"]}_test_scores.npz', score=s, pred=pred, tau=tau)
            torch.save(st, OUT / 'models' / f'{sp["name"]}_final_k{pk}.pt')
    return audit


def build_report(diag, avg, audit, extra):
    L = ['# E222 跨会话分数尺度偏移诊断与 AS-Norm 校准', '',
         '统一口径（沿用 E221）：高分=known；tau=known inner OOF 的 0.1 分位数（known acceptance≈0.90）；score<tau 拒识；unknown 为正类计算 F1/AUC(-score)。完整 20 折 strict subject-LOSO，训练 fit_fraction<=0.6，评估 (0.6,0.8]，Test 仅在冻结后读取。', '',
         '## 1. 可行性诊断（只用 Test 分数，禁止用于选型）', '',
         '数据源：E221 缓存的 baseline Test 分数（E221 的 Train 选择已完成）。Test 已被反复查看，此处只作可行性判断，不能作为候选选择或阈值搜索依据。', '',
         '### 表 1 Test 分数分布', '', '|集合|N|mean|std|q10|median|q90|', '|---|---:|---:|---:|---:|---:|---:|']
    for k in ['S1', 'S2']:
        v = diag['test_score_stats'][k]
        L.append(f"|{k}|{v['n']}|{v['mean']:.4f}|{v['std']:.4f}|{v['q10']:.4f}|{v['median']:.4f}|{v['q90']:.4f}|")
    L += ['', '### 表 2 单阈值扫描（baseline 原始分数；tau 为绝对分数）', '', '|tau|S1 拒识率|S2 接受率|', '|---:|---:|---:|']
    for r in diag['threshold_sweep']:
        L.append(f"|{r['tau']:.4f}|{r['S1_rejection']:.4f}|{r['S2_acceptance']:.4f}|")
    L += ['', '### 表 3 约束下的条件最优（同一阈值）', '', '|约束|tau|该阈值下指标|', '|---|---:|---|']
    for c in diag['conditional_points']:
        if 'max_S2_acceptance' in c:
            L.append(f"|{c['constraint']}|{c['tau']:.4f}|S1 拒识={c['achieved_S1_rejection']:.4f}；最大 S2 接受={c['max_S2_acceptance']:.4f}|")
        else:
            L.append(f"|{c['constraint']}|{c['tau']:.4f}|S2 接受={c['achieved_S2_acceptance']:.4f}；最大 S1 拒识={c['max_S1_rejection']:.4f}|")
    d4 = diag['loso_unknown_rejection_090']
    L += ['', '### 表 4 单阈值可分性上限与 LOSO 0.90 阈值外推', '',
          f"- 该分数区分 Test S1 vs S2（正类=S2 已知）的 AUC = **{diag['S2_vs_S1_single_threshold_AUC']:.6f}**。",
          f"- LOSO unknown 90% 拒识对应 tau（20 折 unknown 分数合并取 0.90 分位）= **{d4['tau90_pooled']:.4f}**；逐折 tau 均值 = {d4['tau90_per_fold_mean']:.4f}。直接套到 Test：S1 拒识 = **{d4['S1_rejection']:.4f}**，S2 接受 = **{d4['S2_acceptance']:.4f}**。",
          f"- LOSO unknown（n={d4['loso_unknown_n']}）mean={d4['loso_unknown_score_stats']['mean']:.4f}，median={d4['loso_unknown_score_stats']['median']:.4f}；LOSO known（n={d4['loso_known_n']}）mean={d4['loso_known_score_stats']['mean']:.4f}，median={d4['loso_known_score_stats']['median']:.4f}。", '',
          '**诊断结论**：单阈值在 Test 上无法同时满足 S1 高拒识与 S2 高接受（表 3）；LOSO 尺度的 tau 直接迁移到 Test 明显偏移（表 4），说明问题主要是跨会话分数尺度不一致，而非阈值挑错。', '',
          '## 2. AS-Norm 候选（固定 config，完整 20 折）', '',
          '- 基础打分：E65 任务标准化 + PCA96 + k12 原型 top-1 距离（与 E221 baseline 完全一致）。',
          '- 原型级 cohort：每个 task 下所有注册身份的原型（LOSO 19×12；Test 20×12），排除查询 top-1 所声称身份的全部原型后计算 cohort 均值/标准差。排除的是该查询 top-1 声称身份（可部署、无需标签），不是真值身份。',
          '- asnorm_all：cohort 全量；asnorm_topk10/25/50：cohort 中最高 k 个原型分数。z-norm：每个身份模型用 outer-train fit 中其他身份语句估计均值/标准差；z+t-norm：z-norm 后再按 cohort 做 t-norm。',
          '- 每折内 cohort、标准化、PCA、原型、归一化参数只用 fit_fraction<=0.6 且非 held 身份拟合；tau 来自内层 5 块 OOF 的 known 分数，不用 in-sample。',
          '', '### 表 5 20 折 LOSO 与冻结 Test 结果', '',
          '|候选|LOSO AUC|拒识 F1|unknown 拒识 macro|unknown 拒识 micro|known 接受|identity 正确却被拒|Test S1 拒识|S1 阶段≥0.70|S1 官方≥0.95|Test S2 接受|S2 阶段≥0.70|S2 官方≥0.98|同时满足 LOSO>0.90 & S2>0.70 & S1≥0.70|',
          '|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|:---:|:---:|:---:|']
    for name, v in avg.items():
        a = audit[name]
        ok = (v['unknown_rejection'] > 0.90) and (a['S2_acceptance'] > 0.70) and (a['S1_rejection'] >= 0.70)
        L.append(
            f"|{name}|{v['auc']:.6f}|{v['rejection_f1']:.6f}|{v['unknown_rejection']:.6f}|{v['unknown_rejection_micro']:.6f}|"
            f"{v['known_acceptance']:.6f}|{v['identity_correct_but_rejected']:.6f}|{a['S1_rejection']:.6f}|"
            f"{'是' if a['stage_S1_met'] else '否'}|{'是' if a['official_S1_met'] else '否'}|{a['S2_acceptance']:.6f}|"
            f"{'是' if a['stage_S2_met'] else '否'}|{'是' if a['official_S2_met'] else '否'}|{'是' if ok else '否'}|")
    if extra:
        L += ['', '### 表 6 追加一轮（Train 侧规则触发的唯一变体）', '',
              f"- 触发规则：{CFG['extra_round']['rule']}",
              f"- 实际追加：`{extra['name']}`（{extra['reason']}）", '',
              '|候选|LOSO AUC|拒识 F1|unknown 拒识|known 接受|Test S1 拒识|Test S2 接受|', '|---|---:|---:|---:|---:|---:|---:|']
        v, a = avg[extra['name']], audit[extra['name']]
        L.append(f"|{extra['name']}|{v['auc']:.6f}|{v['rejection_f1']:.6f}|{v['unknown_rejection']:.6f}|{v['known_acceptance']:.6f}|{a['S1_rejection']:.6f}|{a['S2_acceptance']:.6f}|")
    else:
        L += ['', '### 表 6 追加一轮', '', '- 未触发：最佳 AS-Norm 变体未同时满足 Train 侧改善条件。']
    L += ['', '## 3. 结论与限制', '',
          '- AS-Norm 是否真正缩小跨会话差距，以表 5 中 Test S1/S2 与 LOSO 拒识是否同时改善为准；不凭 Test 结果选择候选。',
          '- 官方 Test AUC/F1 因 Test 无身份标签为 **N/A**；S1/S2 只是会话分层行为指标，不等同官方未知/正常分类指标。',
          '- Test 已被反复查看，任何基于 Test 的判断都存在不可消除的评估偏倚；本报告所有候选与阈值均预先固定于 config。',
          '- LOSO 指标为逐折宏平均（micro 为合并计数），unknown:known 样本约 1:19，拒识 F1 受类别比例影响。',
          '- 运行：`.venv\\Scripts\\python.exe research\\E222_asnorm_session_gap\\run_e222.py`（CUDA）。']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


def main():
    global FR
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available'
    log(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    meta = pd.read_csv(ROOT / CFG['data']['metadata'])
    tasks = meta.task.to_numpy()
    y = meta.subject.to_numpy(int)
    FR = frac(meta)
    blocks = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    raw = np.load(ROOT / CFG['features']['e65']).astype(np.float32)
    assert len(raw) == len(meta) and set(y) == set(range(1, 21)) and set(tasks) == set(TASKS)
    (OUT / 'cache').mkdir(exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)

    diag = phase_diagnostic()

    t0 = time.time()
    n_folds = int(os.environ.get('E222_FOLDS', '20'))
    folds = run_folds(SPECS, raw, tasks, y, blocks, held_range=range(1, n_folds + 1))
    avg = summarise(folds, [s['name'] for s in SPECS])

    asn = [s['name'] for s in SPECS if s['mode'].startswith('asnorm')]
    best = max(asn, key=lambda n: (avg[n]['unknown_rejection'], avg[n]['known_acceptance']))
    base = avg['baseline_raw']
    extra_info = None
    extra_spec = None
    if os.environ.get('E222_NO_EXTRA') == '1':
        log(f'extra round disabled by env (best={best})')
    elif avg[best]['unknown_rejection'] >= base['unknown_rejection'] + 0.05 and avg[best]['unknown_rejection'] < 0.90:
        bs = next(s for s in SPECS if s['name'] == best)
        extra_spec = dict(name=best + CFG['extra_round']['name_suffix'], mode=bs['mode'],
                          topk=bs['topk'], proto_k=int(CFG['extra_round']['prototypes_k']))
        extra_info = dict(name=extra_spec['name'],
                          reason=f'{best} LOSO 拒识 {avg[best]["unknown_rejection"]:.4f} ≥ baseline+0.05 且 <0.90，按规则用 k={extra_spec["proto_k"]} 原型重跑',
                          spec=extra_spec)
        log('extra round triggered: ' + json.dumps(extra_info))
        folds2 = run_folds([extra_spec], raw, tasks, y, blocks)
        for f, g in zip(folds, folds2):
            f['metrics'].update(g['metrics'])
        avg.update(summarise(folds, [extra_spec['name']]))
    else:
        log(f'extra round not triggered (best={best}, LOSO rejection={avg[best]["unknown_rejection"]:.4f})')

    tm = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    raw_test = np.load(ROOT / CFG['features']['e65_test']).astype(np.float32)
    fit = (FR <= FIT_FRAC)
    audit_specs = SPECS + ([extra_spec] if extra_spec else [])
    audit = test_audit(audit_specs, raw, tasks, y, fit, raw_test, tm)

    result = dict(experiment=CFG['experiment'], device=torch.cuda.get_device_name(0),
                  fold_count=len(folds), candidates=[s['name'] for s in audit_specs],
                  candidate_summary=avg, folds=folds, test_audit=audit,
                  extra_round=extra_info, diagnostic=diag,
                  official_test_metric='N/A: Test has no identity labels',
                  test_bias_warning='Test inspected repeatedly; no Test-based selection performed, but evaluation bias is irreducible.',
                  runtime_seconds=time.time() - t0)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    build_report(diag, avg, audit, extra_info)
    log('completed in %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
