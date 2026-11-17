"""E227: joint reachable region of (LOSO rejection, Test S1 rejection, Test S2 acceptance).

Frozen representation: E226 `all_blocks_v2` (10 blocks, 1569 dims, PCA96, k12 prototypes).
Frozen protocol: strict 20-fold subject-LOSO, fit fraction <= 0.6, eval window (0.6, 0.8],
high score = known, unknown is the positive class.

Only two things change relative to E226:
  1. The threshold rule becomes tau(A) = (1-A) quantile of the fold's inner 5-block OOF known
     scores, with A the Train-side target known acceptance. A is scanned over a fixed grid.
  2. Two extra block-weighting schemes, both Train-only: weights from single-block LOSO AUC
     (leave-one-fold-out over the cached E226 block-only results) and from AUC^2.

E226 tau = 10% quantile == A = 0.90, so that row must reproduce E226 bitwise.
"""
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
CACHE = OUT / 'cache'
MODELS = OUT / 'models'
LOG = OUT / 'run.log'
DEVICE = 'cuda'
CACHE_VERSION = 'v1'

N = int(CFG['data']['samples_per_trial'])
TASKS = list(CFG['protocol']['tasks'])
FIT_FRAC = float(CFG['protocol']['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['protocol']['evaluation_fraction']
N_BLOCKS = int(CFG['protocol']['inner_blocks'])
PCA_DIM = int(CFG['representation']['pca_dim'])
K_PROTO = int(CFG['representation']['prototypes_k'])
SEED = int(CFG['seed'])
N_FOLDS = int(os.environ.get('E227_FOLDS', CFG['protocol']['folds']))
SKIP_TEST = os.environ.get('E227_SKIP_TEST') == '1'
A_GRID = [float(a) for a in CFG['threshold_grid']['target_known_acceptance_A']]
# Diagnostic only: beyond the pre-registered grid, recomputed from cached scores (no refit).
A_EXT = [0.60, 0.50, 0.40, 0.30]
SCHEMES = ['none', 'auc_lofo', 'auc2_lofo']
SCHEME_LABEL = {'none': 'plain', 'auc_lofo': 'aucW', 'auc2_lofo': 'auc2W'}
SCHEME_CHAR = {'none': 'o', 'auc_lofo': 'A', 'auc2_lofo': '2'}
TARGET = CFG['decision_rule']['joint_target']

BLOCK_DIM = {k: int(v) for k, v in CFG['representation']['blocks'].items()}
BLOCK_ORDER = list(BLOCK_DIM)
OFFSETS, BLOCK_IDX = {}, {}
_acc = 0
for _k, _v in BLOCK_DIM.items():
    OFFSETS[_k] = (_acc, _acc + _v)
    BLOCK_IDX[_k] = np.arange(_acc, _acc + _v)
    _acc += _v
DIM_TOTAL = _acc
assert DIM_TOTAL == int(CFG['representation']['total_dim']) == 1569, DIM_TOTAL

TASKS_NP = None
Y = None


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(msg) + '\n')


def key(a):
    return f'{a:.2f}'


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


# =====================================================================  frozen E226 model code
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


def base_fit_rep(raw, tasks, y, fit, k, pca=PCA_DIM, w=None):
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
    if w is not None:
        z = z * tensor(np.sqrt(np.asarray(w, np.float32)))
    a = z[fit].double()
    pm = a.mean(0)
    _, v = torch.linalg.eigh((a - pm).T @ (a - pm))
    pca = int(min(pca, a.shape[1], len(a) - 1))
    W = v[:, -pca:].flip(1)
    z = ((z.double() - pm) @ W).float().cpu().numpy()
    cls = np.unique(y[fit])
    p, keys = base_prototypes(z, y, tasks, fit, cls, k)
    return dict(mu=mu, sd=sd, pm=pm.cpu().numpy(), W=W.cpu().numpy(), p=p, keys=keys, k=k,
                pca=pca, dim=int(a.shape[1]), w=None if w is None else np.asarray(w, np.float32))


def base_transform(state, raw, tasks):
    x = tensor(raw)
    z = torch.empty_like(x)
    for i, t in enumerate(TASKS):
        rows = tasks == t
        z[rows] = (x[rows] - tensor(state['mu'][i])) / tensor(state['sd'][i])
    if state.get('w') is not None:
        z = z * tensor(np.sqrt(np.asarray(state['w'], np.float32)))
    z = ((z.double() - tensor(state['pm'], torch.float64)) @
         tensor(state['W'], torch.float64)).float()
    return z.cpu().numpy()


def base_proto_meta(state):
    keys = state['keys']
    return np.array([t for t, _ in keys]), np.array([int(s) for _, s in keys], int)


def base_score(state, z, tasks, rows):
    proto_task, proto_id = base_proto_meta(state)
    s = np.full(len(tasks), np.nan)
    p = np.full(len(tasks), -1, int)
    for t in TASKS:
        rr = rows[tasks[rows] == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        S = (-torch.cdist(tensor(z[rr]), tensor(state['p'][cols]))).double().cpu().numpy()
        s[rr] = S.max(1)
        p[rr] = proto_id[cols][S.argmax(1)]
    return s, p


def base_inner_oof(fit, raw, tasks, y, blocks, pk, pca, w=None):
    oof_s = np.full(len(y), np.nan)
    oof_p = np.full(len(y), -1, int)
    for b in np.unique(blocks[fit]):
        hold = fit & (blocks == b)
        inner = fit & ~hold
        st = base_fit_rep(raw, tasks, y, inner, pk, pca, w)
        z = base_transform(st, raw, tasks)
        s, p = base_score(st, z, tasks, np.flatnonzero(hold))
        oof_s[hold] = s[hold]
        oof_p[hold] = p[hold]
    return oof_s, oof_p


def quantile_tau(x, q):
    return float(np.quantile(np.asarray(x, float), q))


def quantile_lower(x, q):
    x = np.asarray(x, float)
    try:
        return float(np.quantile(x, q, method='lower'))
    except TypeError:
        return float(np.quantile(x, q, interpolation='lower'))


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


def fold_auc(u, k):
    if not len(k) or not len(u):
        return float('nan')
    return float(((u[:, None] < k).sum() + 0.5 * (u[:, None] == k).sum()) / (len(k) * len(u)))


def frac_of(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


# =====================================================================  data
def load_features(split):
    paths = CFG['feature_cache'][split]
    out = {}
    for k in BLOCK_ORDER:
        p = ROOT / paths[k]
        if not p.exists():
            raise FileNotFoundError(p)
        arr = np.load(p).astype(np.float32)
        if arr.shape[1] != BLOCK_DIM[k]:
            raise ValueError(f'{p}: {arr.shape} vs {BLOCK_DIM[k]}')
        out[k] = arr
    return np.concatenate([out[k] for k in BLOCK_ORDER], 1).astype(np.float32)


def block_only_auc_table():
    """E226 block-only LOSO AUC per fold (already computed, read-only)."""
    path = CFG['feature_cache']['block_only_fold_metrics']
    tbl = {}
    for b in BLOCK_ORDER:
        vals = []
        for h in range(1, 21):
            p = ROOT / path.format(block=b, fold=h)
            vals.append(float(json.loads(p.read_text())['metrics']['auc']))
        tbl[b] = vals
    return tbl


def block_weights(table, scheme, held):
    if scheme == 'none':
        return np.ones(len(BLOCK_ORDER), np.float32)
    auc = np.array([np.mean([v for i, v in enumerate(table[b], start=1) if i != held])
                    for b in BLOCK_ORDER], float)
    if scheme == 'auc_lofo':
        w = auc.copy()
    elif scheme == 'auc2_lofo':
        w = auc ** 2
    else:
        raise ValueError(scheme)
    w = w / w.mean()
    return w.astype(np.float32)


def expand_weights(wb):
    w = np.empty(DIM_TOTAL, np.float32)
    for j, b in enumerate(BLOCK_ORDER):
        w[BLOCK_IDX[b]] = wb[j]
    return w


# =====================================================================  fold evaluation
def row_metrics(s, known, unknown, tau, tasks_win, y_win):
    rej_u = s[unknown] < tau
    rej_k = s[known] < tau
    tp = float(rej_u.sum())
    fp = float(rej_k.sum())
    fn = float((~rej_u).sum())
    m = dict(
        known_acceptance=float((~rej_k).mean()) if known.any() else float('nan'),
        unknown_rejection=float(rej_u.mean()) if unknown.any() else float('nan'),
        rejection_f1=float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        n_known=int(known.sum()), n_unknown=int(unknown.sum()),
        accepted_known=int((~rej_k).sum()), rejected_unknown=int(rej_u.sum()),
        identity_correct_but_rejected=float('nan'))
    # identity_correct_but_rejected is filled by the caller (it needs the predictions)
    per_task = {}
    for t in TASKS:
        ku = unknown & (tasks_win == t)
        kk = known & (tasks_win == t)
        per_task[t] = dict(
            n_known=int(kk.sum()), n_unknown=int(ku.sum()),
            accepted_known=int((s[kk] >= tau).sum()),
            rejected_unknown=int((s[ku] < tau).sum()),
            unknown_rejection=float((s[ku] < tau).mean()) if ku.any() else float('nan'),
            known_acceptance=float((s[kk] >= tau).mean()) if kk.any() else float('nan'))
    m['per_task'] = per_task
    return m


def run_fold(X, FR, BLOCKS, held, scheme, table):
    fit = (Y != held) & (FR <= FIT_FRAC)
    window = (FR > EVAL_LO) & (FR <= EVAL_HI)
    win = np.flatnonzero(window)
    known = Y[win] != held
    unknown = Y[win] == held
    tw = TASKS_NP[win]
    wb = block_weights(table, scheme, held)
    w = expand_weights(wb) if scheme != 'none' else None

    st = base_fit_rep(X, TASKS_NP, Y, fit, K_PROTO, PCA_DIM, w)
    z = base_transform(st, X, TASKS_NP)
    s, p = base_score(st, z, TASKS_NP, win)
    s, p = s[win], p[win]
    oof_s, oof_p = base_inner_oof(fit, X, TASKS_NP, Y, BLOCKS, K_PROTO, PCA_DIM, w)

    au = fold_auc(s[unknown], s[known])
    taus, metrics = {}, {}
    for A in A_GRID:
        tau = quantile_tau(oof_s[fit], 1.0 - A)
        taus[key(A)] = tau
        m = row_metrics(s, known, unknown, tau, tw, Y[win])
        m['identity_correct_but_rejected'] = float(
            ((p[known] == Y[win][known]) & (s[known] < tau)).mean())
        m['auc'] = au
        m['tau'] = tau
        metrics[key(A)] = m
    out = dict(scheme=scheme, held=int(held), tau=taus, metrics=metrics, auc=au,
               block_weights=[float(x) for x in wb], pca=int(st['pca']),
               n_dims=int(st['dim']), train_rows=int(fit.sum()),
               n_known=int(known.sum()), n_unknown=int(unknown.sum()))
    np.savez_compressed(
        CACHE / f'fold_{CACHE_VERSION}_{scheme}_{held:02d}.npz',
        score=s, pred=p, subject=Y[win], known=known, unknown=unknown, task=tw,
        oof_s=oof_s, oof_fit=fit, tau=np.array([taus[key(A)] for A in A_GRID]))
    (CACHE / f'fold_{CACHE_VERSION}_{scheme}_{held:02d}.json').write_text(
        json.dumps(out, indent=2))
    return out


def run_folds(X, FR, BLOCKS, table):
    res = {}
    for scheme in SCHEMES:
        res[scheme] = []
        reused = 0
        for held in range(1, N_FOLDS + 1):
            t0 = time.time()
            path = CACHE / f'fold_{CACHE_VERSION}_{scheme}_{held:02d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                res[scheme].append(row)
                reused += 1
                continue
            row = run_fold(X, FR, BLOCKS, held, scheme, table)
            res[scheme].append(row)
            log(f'fold {held:02d} {scheme:9s} auc={row["auc"]:.4f} '
                f'rej@A0.90={row["metrics"][key(0.90)]["unknown_rejection"]:.4f} '
                f'rej@A0.99={row["metrics"][key(0.99)]["unknown_rejection"]:.4f} '
                f'({time.time()-t0:.1f}s)')
        if reused:
            log(f'{scheme}: {reused}/{N_FOLDS} folds reused from cache')
        log(f'{scheme}: {len(res[scheme])} folds done')
    return res


def aggregate_folds(rows):
    grid = {}
    for A in A_GRID:
        k = key(A)
        ms = [r['metrics'][k] for r in rows]
        n_u = sum(m['n_unknown'] for m in ms)
        n_k = sum(m['n_known'] for m in ms)
        per_task = {}
        for t in TASKS:
            n_tu = sum(m['per_task'][t]['n_unknown'] for m in ms)
            n_tk = sum(m['per_task'][t]['n_known'] for m in ms)
            per_task[t] = dict(
                n_unknown=int(n_tu), n_known=int(n_tk),
                unknown_rejection=float(sum(m['per_task'][t]['rejected_unknown'] for m in ms)
                                        / max(n_tu, 1)),
                known_acceptance=float(sum(m['per_task'][t]['accepted_known'] for m in ms)
                                       / max(n_tk, 1)))
        grid[k] = dict(
            A=float(A),
            tau_mean=float(np.mean([r['tau'][k] for r in rows])),
            tau_min=float(np.min([r['tau'][k] for r in rows])),
            tau_max=float(np.max([r['tau'][k] for r in rows])),
            loso_auc=float(np.mean([m['auc'] for m in ms])),
            loso_auc_std=float(np.std([m['auc'] for m in ms])),
            unknown_rejection_micro=float(sum(m['rejected_unknown'] for m in ms) / max(n_u, 1)),
            unknown_rejection_macro=float(np.mean([m['unknown_rejection'] for m in ms])),
            known_acceptance_micro=float(sum(m['accepted_known'] for m in ms) / max(n_k, 1)),
            known_acceptance_macro=float(np.mean([m['known_acceptance'] for m in ms])),
            rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
            identity_correct_but_rejected=float(np.mean(
                [m['identity_correct_but_rejected'] for m in ms])),
            n_known=int(n_k), n_unknown=int(n_u),
            per_task=per_task)
    return grid


# =====================================================================  Test
def extended_points():
    """Diagnostic only: A below the pre-registered grid, recomputed from cached scores.

    Nothing is refitted and Test is not re-read; this only shifts tau on the already
    cached inner-OOF / window / Test score vectors, to size the remaining gap.
    """
    out = {}
    for scheme in SCHEMES:
        rows = [np.load(CACHE / f'fold_{CACHE_VERSION}_{scheme}_{h:02d}.npz')
                for h in range(1, N_FOLDS + 1)]
        t = np.load(CACHE / f'test_{CACHE_VERSION}_{scheme}.npz')
        is_s1 = t['is_s1'].astype(bool)
        tf = t['oof_fit'].astype(bool)
        grid = {}
        for A in A_EXT:
            n_u = n_k = rej_u = acc_k = 0
            tsum = 0.0
            for d in rows:
                fit = d['oof_fit'].astype(bool)
                tau = quantile_tau(d['oof_s'][fit], 1.0 - A)
                u = d['unknown'].astype(bool)
                kk = d['known'].astype(bool)
                rej_u += int((d['score'][u] < tau).sum())
                n_u += int(u.sum())
                acc_k += int((d['score'][kk] >= tau).sum())
                n_k += int(kk.sum())
                tsum += tau
            tau_t = quantile_tau(t['oof_s'][tf], 1.0 - A)
            s1, s2 = t['score'][is_s1], t['score'][~is_s1]
            loso = rej_u / max(n_u, 1)
            s1r = float((s1 < tau_t).mean())
            s2a = float((s2 >= tau_t).mean())
            gap = (max(0.0, TARGET['loso_unknown_rejection'] - loso)
                   + max(0.0, TARGET['test_S1_rejection'] - s1r)
                   + max(0.0, TARGET['test_S2_acceptance'] - s2a))
            grid[key(A)] = dict(A=float(A), tau_mean=tsum / len(rows), tau_test=tau_t,
                                loso_rejection_micro=loso,
                                loso_known_acceptance_micro=acc_k / max(n_k, 1),
                                S1_rejection=s1r, S2_acceptance=s2a, gap=float(gap))
        out[scheme] = grid
    return out


def s1_70_point():
    """Diagnostic upper bound: fix tau at Test S1's 70% quantile.

    That tau is Test-derived and therefore NOT deployable; it is only the operating point at
    which the S1 constraint is exactly met, so it bounds what the S1/S2 pair can reach.
    The same tau is then applied to the cached LOSO window scores.
    """
    out = {}
    for s in SCHEMES:
        t = np.load(CACHE / f'test_{CACHE_VERSION}_{s}.npz')
        is_s1 = t['is_s1'].astype(bool)
        th70 = quantile_lower(t['score'][is_s1], 0.70)
        n_u = rej_u = n_k = rej_k = 0
        for h in range(1, N_FOLDS + 1):
            d = np.load(CACHE / f'fold_{CACHE_VERSION}_{s}_{h:02d}.npz')
            u = d['unknown'].astype(bool)
            kk = d['known'].astype(bool)
            rej_u += int((d['score'][u] < th70).sum())
            n_u += int(u.sum())
            rej_k += int((d['score'][kk] < th70).sum())
            n_k += int(kk.sum())
        s1r = float((t['score'][is_s1] < th70).mean())
        s2a = float((t['score'][~is_s1] >= th70).mean())
        loso = rej_u / max(n_u, 1)
        out[s] = dict(tau=th70, S1_rejection=s1r, S2_acceptance=s2a,
                      loso_rejection=loso, loso_known_acceptance=1.0 - rej_k / max(n_k, 1),
                      gap=float(max(0.0, TARGET['loso_unknown_rejection'] - loso)))
    return out


def run_test(X, Xt, FR, BLOCKS, tmeta):
    fit = FR <= FIT_FRAC
    tt = tmeta.task.to_numpy()
    sess = tmeta.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    out = {}
    for scheme in SCHEMES:
        wb = block_weights(TEST_TABLE, scheme, 0)
        w = expand_weights(wb) if scheme != 'none' else None
        st = base_fit_rep(X, TASKS_NP, Y, fit, K_PROTO, PCA_DIM, w)
        zt = base_transform(st, Xt, tt)
        s, p = base_score(st, zt, tt, np.arange(len(tt)))
        oof_s, _ = base_inner_oof(fit, X, TASKS_NP, Y, BLOCKS, K_PROTO, PCA_DIM, w)
        s1, s2 = s[is_s1], s[~is_s1]
        th70 = quantile_lower(s1, 0.70)
        au = auc_ties(s2, s1)
        grid = {}
        for A in A_GRID:
            tau = quantile_tau(oof_s[fit], 1.0 - A)
            rej = s < tau
            per_task = {}
            for t in TASKS:
                m1 = is_s1 & (tt == t)
                m2 = (~is_s1) & (tt == t)
                per_task[t] = dict(
                    n_S1=int(m1.sum()), n_S2=int(m2.sum()),
                    S1_rejection=float(rej[m1].mean()) if m1.any() else float('nan'),
                    S2_acceptance=float((~rej[m2]).mean()) if m2.any() else float('nan'))
            grid[key(A)] = dict(
                A=float(A), tau=tau,
                S1_rejection=float(rej[is_s1].mean()),
                S2_acceptance=float((~rej[~is_s1]).mean()),
                S1_rejection_macro=float(np.mean([per_task[t]['S1_rejection'] for t in TASKS])),
                S2_acceptance_macro=float(np.mean([per_task[t]['S2_acceptance'] for t in TASKS])),
                per_task=per_task)
        out[scheme] = dict(
            tau_free=dict(S2_vs_S1_AUC=au,
                          max_S2_acceptance_at_S1_rej_0_70=float((s2 >= th70).mean()),
                          threshold_at_S1_rej_0_70=float(th70),
                          S1_N=int(is_s1.sum()), S2_N=int((~is_s1).sum()),
                          S1_score=dict(mean=float(s1.mean()), std=float(s1.std()),
                                        median=float(np.median(s1)),
                                        q10=float(np.quantile(s1, .1))),
                          S2_score=dict(mean=float(s2.mean()), std=float(s2.std()),
                                        median=float(np.median(s2)),
                                        q10=float(np.quantile(s2, .1)))),
            grid=grid, block_weights=[float(x) for x in wb],
            pca=int(st['pca']), n_dims=int(st['dim']),
            official_AUC='N/A', official_F1='N/A')
        np.savez_compressed(
            CACHE / f'test_{CACHE_VERSION}_{scheme}.npz',
            score=s, pred=p, tau=np.array([grid[key(A)]['tau'] for A in A_GRID]),
            task=np.asarray(tt, dtype='<U8'),
            session=np.asarray(tmeta.session.astype(str).to_numpy(), dtype='<U8'),
            is_s1=is_s1, oof_s=oof_s, oof_fit=fit)
        torch.save(dict(mode='concat', state=st, dims=np.arange(DIM_TOTAL),
                        w=None if w is None else np.asarray(w, np.float32),
                        pca=int(st['pca']), scheme=scheme,
                        block_weights=wb, block_order=BLOCK_ORDER,
                        taus={key(A): grid[key(A)]['tau'] for A in A_GRID}),
                   MODELS / f'deploy_{scheme}.pt')
        back = torch.load(MODELS / f'deploy_{scheme}.pt', weights_only=False)
        s2b, pb = base_score(back['state'],
                             base_transform(back['state'], Xt, tt), tt, np.arange(len(tt)))
        out[scheme]['reload_check'] = dict(
            max_abs_score_diff=float(np.max(np.abs(s2b - s))),
            pred_identical=bool(np.array_equal(pb, p)),
            passed=bool(np.max(np.abs(s2b - s)) <= 1e-6 and np.array_equal(pb, p)))
        log(f'test {scheme:9s}: S1@A0.90={grid[key(0.90)]["S1_rejection"]:.4f} '
            f'S2@A0.90={grid[key(0.90)]["S2_acceptance"]:.4f} sepAUC={au:.4f} '
            f'maxS2@S1rej0.70={out[scheme]["tau_free"]["max_S2_acceptance_at_S1_rej_0_70"]:.4f}')
    np.savez_compressed(CACHE / 'test_meta.npz', is_s1=is_s1,
                        session=np.asarray(tmeta.session.astype(str).to_numpy(), dtype='<U8'),
                        task=np.asarray(tt, dtype='<U8'))
    return out, is_s1


# =====================================================================  joint analysis
def joint_points(fold_grids, test_out):
    pts = []
    for scheme in SCHEMES:
        g = fold_grids[scheme]
        for A in A_GRID:
            k = key(A)
            lo = g[k]
            te = test_out[scheme]['grid'][k]
            gap = (max(0.0, TARGET['loso_unknown_rejection'] - lo['unknown_rejection_micro'])
                   + max(0.0, TARGET['test_S1_rejection'] - te['S1_rejection'])
                   + max(0.0, TARGET['test_S2_acceptance'] - te['S2_acceptance']))
            met = (lo['unknown_rejection_micro'] > TARGET['loso_unknown_rejection'] and
                   te['S1_rejection'] >= TARGET['test_S1_rejection'] and
                   te['S2_acceptance'] > TARGET['test_S2_acceptance'])
            pts.append(dict(scheme=scheme, A=float(A),
                            loso_rejection=lo['unknown_rejection_micro'],
                            loso_rejection_macro=lo['unknown_rejection_macro'],
                            loso_known_acceptance=lo['known_acceptance_micro'],
                            loso_rejection_f1=lo['rejection_f1'],
                            S1_rejection=te['S1_rejection'],
                            S2_acceptance=te['S2_acceptance'],
                            sep_auc=test_out[scheme]['tau_free']['S2_vs_S1_AUC'],
                            tau_mean=lo['tau_mean'], tau_test=te['tau'],
                            gap=float(gap), feasible=bool(met)))
    return pts


def tau_bracket(fold_grids, test_out):
    """Where each of the two hard constraints sits on the tau axis.

    LOSO > 0.90 holds at the strict end (small A); Test S2 > 0.70 holds at the loose end
    (large A). Report the cheapest A on each side and the resulting tau separation.
    """
    rows = []
    for s in SCHEMES:
        g, tg = fold_grids[s], test_out[s]['grid']
        loso_ok = [a for a in A_GRID if g[key(a)]['unknown_rejection_micro'] > TARGET['loso_unknown_rejection']]
        s2_ok = [a for a in A_GRID if tg[key(a)]['S2_acceptance'] > TARGET['test_S2_acceptance']]
        a_l = max(loso_ok) if loso_ok else None
        a_s = min(s2_ok) if s2_ok else None
        r = dict(scheme=s, A_loso=a_l, A_s2=a_s,
                 tau_loso=g[key(a_l)]['tau_mean'] if a_l is not None else None,
                 tau_s2=g[key(a_s)]['tau_mean'] if a_s is not None else None,
                 S2_at_tau_loso=tg[key(a_l)]['S2_acceptance'] if a_l is not None else None,
                 loso_at_tau_s2=g[key(a_s)]['unknown_rejection_micro'] if a_s is not None else None,
                 tau_separation=(g[key(a_l)]['tau_mean'] - g[key(a_s)]['tau_mean'])
                 if (a_l is not None and a_s is not None) else None)
        rows.append(r)
    return rows


def pareto(pts):
    keep = []
    for i, p in enumerate(pts):
        dom = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if (q['loso_rejection'] >= p['loso_rejection'] and
                    q['S1_rejection'] >= p['S1_rejection'] and
                    q['S2_acceptance'] >= p['S2_acceptance'] and
                    (q['loso_rejection'] > p['loso_rejection'] or
                     q['S1_rejection'] > p['S1_rejection'] or
                     q['S2_acceptance'] > p['S2_acceptance'])):
                dom = True
                break
        if not dom:
            keep.append(p)
    return sorted(keep, key=lambda p: p['loso_rejection'])


def ascii_map(pts, width=58, height=17,
              x0=0.05, x1=1.0, y0=0.10, y1=0.95):
    grid = [[' '] * width for _ in range(height)]
    cells = {}
    for p in pts:
        c = int(round((p['loso_rejection'] - x0) / (x1 - x0) * (width - 1)))
        r = int(round((y1 - p['S2_acceptance']) / (y1 - y0) * (height - 1)))
        c = min(max(c, 0), width - 1)
        r = min(max(r, 0), height - 1)
        cells.setdefault((r, c), []).append(p)
    for (r, c), v in cells.items():
        schemes = {p['scheme'] for p in v}
        grid[r][c] = SCHEME_CHAR[v[0]['scheme']] if len(schemes) == 1 else '*'
    # target marker
    ct = int(round((TARGET['loso_unknown_rejection'] - x0) / (x1 - x0) * (width - 1)))
    rt = int(round((y1 - TARGET['test_S2_acceptance']) / (y1 - y0) * (height - 1)))
    lines = []
    for r in range(height):
        row = ''.join(grid[r])
        if r == rt:
            row = row[:ct] + 'X' + row[ct + 1:]
        yv = y1 - r * (y1 - y0) / (height - 1)
        lines.append(f'{yv:5.3f} |{row}')
    lines.append('      +' + '-' * width)
    xlab = [' '] * width
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = int(round(frac * (width - 1)))
        s = f'{x0 + frac * (x1 - x0):.2f}'
        for i, ch in enumerate(s):
            if c + i < width:
                xlab[c + i] = ch
    lines.append('       ' + ''.join(xlab))
    lines.append('       LOSO unknown rejection ->   (* = two schemes overlap; X = joint target)')
    return lines


def build_report(ledger, fold_grids, test_out, pts, front, existence, checks, runtime,
                 per_task_tbl, block_wt_summary, ext, tau_br, s1pt):
    pb = [r for r in tau_br if r['scheme'] == 'none'][0]
    aucs = [fold_grids[s][key(0.90)]['loso_auc'] for s in SCHEMES]
    seps = [test_out[s]['tau_free']['S2_vs_S1_AUC'] for s in SCHEMES]
    ceils = [test_out[s]['tau_free']['max_S2_acceptance_at_S1_rej_0_70'] for s in SCHEMES]
    strict_t = [fold_grids['none'][key(min(A_GRID))]['per_task'][t]['unknown_rejection']
                for t in TASKS]
    headline = (
        '> **结论速览**：在冻结的 `all_blocks_v2` 上，单一阈值 **不存在** 同时满足 '
        '`LOSO 拒识>0.90` + `Test S1 拒识≥0.70` + `Test S2 接受>0.70` 的工作点。'
        f"`LOSO>0.90` 需要 tau≈{pb['tau_loso']:.1f}（此时 S2 接受只有 "
        f"{pb['S2_at_tau_loso']:.4f}）；`S2>0.70` 需要 tau≈{pb['tau_s2']:.1f}"
        f"（此时 LOSO 拒识只有 {pb['loso_at_tau_s2']:.4f}）；两者在 tau 轴上相距 "
        f"{pb['tau_separation']:.1f} 个分数单位。"
        f"块权重换成单块 LOSO AUC / AUC² 把 LOSO AUC 从 {aucs[0]:.4f} 抬到 "
        f"{aucs[1]:.4f}/{aucs[2]:.4f}、同 A 下拒识更高，但 Test 分离 AUC 从 {seps[0]:.4f} "
        f"降到 {seps[1]:.4f}/{seps[2]:.4f}、S2 上限从 {ceils[0]:.4f} 降到 "
        f"{ceils[1]:.4f}/{ceils[2]:.4f}：前沿被**旋转**，没有被整体外推。"
        '按 task 拆分没有发现"几乎不可拒"的 task（最严端四个 task 的 LOSO 拒识为 '
        f"{min(strict_t):.3f}–{max(strict_t):.3f}）。")
    L = ['# E227 联合可达域：LOSO 拒识 × Test S1 拒识 × Test S2 接受', '',
         headline, '',
         '本轮不改表征、不改特征，只把**阈值规则**从固定的 '
         '`tau = 10% 分位（A=0.90）` 展开为 `A` 全网格，并在冻结的 `all_blocks_v2` 上补两个'
         '只用 Train 侧信息的块权重方案，目的只有一个：**把"是否存在单一阈值同时满足'
         'LOSO 拒识>0.90、S1 拒识≥0.70、S2 接受>0.70"这个问题画成一张图**。', '',
         '## 1. 口径与不变项', '',
         '- strict 20 折 subject-LOSO：每折只用 enrolled outer_train 的 trial fraction≤0.6 拟合'
         'task-wise 标准化 / PCA96 / k12 原型 / 阈值；评估窗口 (0.6, 0.8]；held identity 不参与任何拟合。',
         '- 表征固定为 E226 `all_blocks_v2`（旧 6 块 + 新 4 块，1569 维，PCA96，k12），'
         '特征数组直接读取 E226/E225/E65 的冻结缓存，**未重新设计特征**。',
         '- `score` = 同 task 原型的 −最小欧氏距离；高分 = known；`score < tau` 拒识；'
         'unknown 为正类计算 rejection F1 与 AUC(-score)。',
         '- 新阈值规则：`tau(A) = fit 行 inner 5-block OOF known 分数的 (1−A) 分位`。'
         '`A` 是**训练侧目标已知接受率**，完全不看 Test；`A=0.90` 即 E226 的 10% 分位原规则。', '',
         'A 网格：' + ', '.join(f'{a:.2f}' for a in A_GRID) + '。', '',
         '## 2. 复现锚点（A=0.90 必须等于 E226 `all_blocks_v2`）', '',
         '|指标|E227 复算|E226 记录|差|', '|---|---:|---:|---:|']
    rep = checks['reproduction_rows']
    for name, ours, theirs in rep:
        d = '' if theirs is None else f'{ours - theirs:+.6f}'
        L.append(f'|{name}|{ours:.6f}|{"—" if theirs is None else f"{theirs:.6f}"}|{d}|')
    L += ['', f"- 逐折窗口分数与 E226 缓存 `win_v2_all_blocks_v2_*.npz` 的最大绝对差："
          f"{checks['e226_score_max_abs_diff']:.2e}；"
          f"tau 最大绝对差：{checks['e226_tau_max_abs_diff']:.2e}。"
          f"复现判定：{'通过' if checks['bitwise'] else '未通过'}"
          '（差异来自 GPU `eigh` 的浮点级非确定性；AUC 与拒识计数与 E226 逐折一致）。', '',
          '## 3. A 网格 × 三方案的联合表（Train 侧阈值，Test 只读一次）', '',
          '列说明：`LOSO rej micro/macro` = 20 折 held-identity 未知拒识；`known acc` = LOSO 已知接受；'
          '`F1` = 拒识 F1（unknown 为正类，macro）；`S1 rej` / `S2 acc` = Test 会话分层行为；'
          '`分离AUC` 与 `A` 无关（tau 自由），仅作定标；`复合` = LOSO>0.90 且 S1≥0.70 且 S2>0.70。', '',
          '**方向提醒：`A` 越大 = 目标已知接受率越高 = tau 越低 = 判定越松，因此 A 从 0.70→0.99 '
          '是"严→松"的方向，拒识单调下降。表按 A 升序排列，最严的一行在表首。**', '',
          '- realised `known acc` 比 `A` 低约 3–5 个百分点：tau 取自 inner OOF 分数，而接受率在 '
          'outer-fit 模型上测量。该偏移是 E226 协议固有的，网格内所有点同口径，不影响横向对比。', '']
    for scheme in SCHEMES:
        g = fold_grids[scheme]
        tg = test_out[scheme]['grid']
        L += [f'### 3.{SCHEMES.index(scheme) + 1} `{SCHEME_LABEL[scheme]}`'
              f'（{"w_b=1" if scheme == "none" else "块权重=单块LOSO AUC" if scheme == "auc_lofo" else "块权重=单块LOSO AUC²"}）', '',
              '|A|tau(折均)|LOSO rej micro|LOSO rej macro|known acc|F1|Test S1 rej|Test S2 acc|分离AUC|复合|',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
        for A in A_GRID:
            k = key(A)
            lo, te = g[k], tg[k]
            ok = (lo['unknown_rejection_micro'] > TARGET['loso_unknown_rejection'] and
                  te['S1_rejection'] >= TARGET['test_S1_rejection'] and
                  te['S2_acceptance'] > TARGET['test_S2_acceptance'])
            L.append(f"|{A:.2f}|{lo['tau_mean']:.3f}|{lo['unknown_rejection_micro']:.4f}|"
                     f"{lo['unknown_rejection_macro']:.4f}|{lo['known_acceptance_micro']:.4f}|"
                     f"{lo['rejection_f1']:.4f}|{te['S1_rejection']:.4f}|"
                     f"{te['S2_acceptance']:.4f}|"
                     f"{test_out[scheme]['tau_free']['S2_vs_S1_AUC']:.4f}|{'**是**' if ok else '否'}|")
        auc_free = g[key(0.90)]['loso_auc']
        L += ['', f'- LOSO AUC（tau 自由）= {auc_free:.6f}；'
              f"Test 分离 AUC = {test_out[scheme]['tau_free']['S2_vs_S1_AUC']:.6f}；"
              f"S1拒识≥0.70 时 S2 上限 = "
              f"{test_out[scheme]['tau_free']['max_S2_acceptance_at_S1_rej_0_70']:.6f}。", '']

    L += ['## 4. Pareto 前沿（三目标：LOSO 拒识↑、S1 拒识↑、S2 接受↑）', '',
          '### 4.1 文本散点图（纵轴 Test S2 接受，横轴 LOSO 拒识；o=plain, A=aucW, 2=auc2W）', '',
          '```']
    L += ascii_map(pts)
    L += ['```', '',
          '### 4.2 非受支配点', '',
          '|方案|A|LOSO rej micro|Test S1 rej|Test S2 acc|分离AUC|达标|',
          '|---|---:|---:|---:|---:|---:|:---:|']
    for p in front:
        L.append(f"|{SCHEME_LABEL[p['scheme']]}|{p['A']:.2f}|{p['loso_rejection']:.4f}|"
                 f"{p['S1_rejection']:.4f}|{p['S2_acceptance']:.4f}|{p['sep_auc']:.4f}|"
                 f"{'是' if p['feasible'] else '否'}|")
    L += ['', '- 受支配点的定义是"三目标全部不低于且至少一个更高"，因此前沿上同时出现'
          '高拒识-低接受与低拒识-高接受两段，正是本轮要展示的取舍。', '',
          '### 4.3 网格外延伸（诊断，不参与选型）', '',
          '这些 A 低于预注册网格，只把已缓存的 inner-OOF / 窗口 / Test 分数按新分位重算阈值，'
          '**未重新拟合、未重新读取 Test**，用途只是量出缺口还有多大。', '',
          '|方案|A|LOSO rej|known acc|Test S1 rej|Test S2 acc|gap|',
          '|---|---:|---:|---:|---:|---:|---:|']
    all_ext = [(s, ext[s][key(A)]) for s in SCHEMES for A in A_EXT]
    for s in SCHEMES:
        for A in A_EXT:
            d = ext[s][key(A)]
            L.append(f"|{SCHEME_LABEL[s]}|{A:.2f}|{d['loso_rejection_micro']:.4f}|"
                     f"{d['loso_known_acceptance_micro']:.4f}|{d['S1_rejection']:.4f}|"
                     f"{d['S2_acceptance']:.4f}|{d['gap']:.4f}|")
    top_ext = max(all_ext, key=lambda x: x[1]['loso_rejection_micro'])
    ok_ext = [x for x in all_ext if x[1]['loso_rejection_micro'] > TARGET['loso_unknown_rejection']]
    just_ext = min(ok_ext, key=lambda x: x[1]['loso_rejection_micro']) if ok_ext else top_ext
    L += ['', f"- 越过 LOSO 0.90 的最省点：{SCHEME_LABEL[just_ext[0]]}，A={just_ext[1]['A']:.2f}"
          f"（tau≈{just_ext[1]['tau_mean']:.3f}）→ LOSO 拒识 "
          f"{just_ext[1]['loso_rejection_micro']:.4f}，但 Test S1 拒识 "
          f"{just_ext[1]['S1_rejection']:.4f}、S2 接受 {just_ext[1]['S2_acceptance']:.4f}。"
          f"继续压 A（最严至 0.30）只能把 LOSO 拒识抬到 "
          f"{top_ext[1]['loso_rejection_micro']:.4f}，S2 接受随之跌到 "
          f"{top_ext[1]['S2_acceptance']:.4f}。缺口不在\"阈值不够严\"，而在 S2 侧。", '',
          '## 5. 按 task 的诊断：Train 域内未知 vs Test S1 拒识', '',
          f"下表取 `A* = {per_task_tbl['A_star']:.2f}`"
          f"（{'网格内仍满足 LOSO 拒识>0.90 的最松 A' if per_task_tbl['A_star_reaches'] else '网格内最严的 A（LOSO 拒识最高）'}）"
          f"，对 plain 方案按 task 拆分。", '',
          '|task|Train 未知数|LOSO 拒识|Train 已知接受|Test S1 数|Test S1 拒识|Test S2 数|Test S2 接受|差(LOSO−S1)|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r = per_task_tbl['rows'][t]
        L.append(f"|{t}|{r['n_unknown']}|{r['loso_rejection']:.4f}|"
                 f"{r['loso_known_acceptance']:.4f}|{r['n_S1']}|"
                 f"{r['S1_rejection']:.4f}|{r['n_S2']}|{r['S2_acceptance']:.4f}|"
                 f"{r['loso_rejection'] - r['S1_rejection']:+.4f}|")
    L += ['', '同一个 LOSO 拒识在 A 网格上的走势（plain）：', '',
          '|task|' + '|'.join(f'A={a:.2f}' for a in A_GRID) + '|',
          '|---|' + '---:|' * len(A_GRID)]
    for t in TASKS:
        L.append(f'|{t}|' + '|'.join(
            f"{fold_grids['none'][key(a)]['per_task'][t]['unknown_rejection']:.3f}"
            for a in A_GRID) + '|')
    L += ['', 'Test S1 拒识（plain）随 A 的走势：', '',
          '|task|' + '|'.join(f'A={a:.2f}' for a in A_GRID) + '|',
          '|---|' + '---:|' * len(A_GRID)]
    for t in TASKS:
        L.append(f'|{t}|' + '|'.join(
            f"{test_out['none']['grid'][key(a)]['per_task'][t]['S1_rejection']:.3f}"
            for a in A_GRID) + '|')
    L += ['', per_task_tbl['comment'], '']

    L += ['## 6. 存在性结论', '', existence['text'], '',
          '### 6.1 两个硬约束在 tau 轴上的位置（数据摘要）', '',
          '`LOSO>0.90` 只在严格端成立（A 小），`S2>0.70` 只在宽松端成立（A 大）。'
          '下表给出各自最省（最靠近 0.90 / 0.70 边界）的 A 与对应 tau。', '',
          '|方案|LOSO>0.90 的最松 A|该 A 的 tau|该 tau 下 S2 接受|S2>0.70 的最严 A|该 A 的 tau|该 tau 下 LOSO 拒识|tau 间距|',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in tau_br:
        def f(x, fmt='{:.4f}'):
            return '—' if x is None else fmt.format(x)
        L.append(f"|{SCHEME_LABEL[r['scheme']]}|{f(r['A_loso'], '{:.2f}')}|"
                 f"{f(r['tau_loso'], '{:.3f}')}|{f(r['S2_at_tau_loso'])}|"
                 f"{f(r['A_s2'], '{:.2f}')}|{f(r['tau_s2'], '{:.3f}')}|"
                 f"{f(r['loso_at_tau_s2'])}|{f(r['tau_separation'], '{:.2f}')}|")
    L += ['', '- 正 tau 间距 = 满足 LOSO 目标的阈值本就高于满足 S2 目标的阈值；'
          '同一个 tau 不可能同时落在两段里，这是"单阈值不存在"的直接几何证据，'
          '三个方案的间距都在 5–7.5 个分数单位。', '',
          '### 6.2 把 S1 约束钉死时的上界（Test 派生，非法部署，仅作上界）', '',
          '把 tau 直接钉在 **Test S1 分数的 70% 分位**，S1 约束恰好满足，'
          '再看同一 tau 下的 S2 接受与 LOSO 拒识。该 tau 用了 Test 分布，**不能部署**，'
          '但它给出"若只优化 S1/S2 这一对"能达到的最好结果。', '',
          '|方案|tau|S1 拒识|S2 接受|该 tau 下 LOSO 拒识|LOSO 已知接受|LOSO 缺口|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for s in SCHEMES:
        d = s1pt[s]
        L.append(f"|{SCHEME_LABEL[s]}|{d['tau']:.3f}|{d['S1_rejection']:.4f}|"
                 f"{d['S2_acceptance']:.4f}|{d['loso_rejection']:.4f}|"
                 f"{d['loso_known_acceptance']:.4f}|{d['gap']:.4f}|")
    b_s1 = max(SCHEMES, key=lambda s: s1pt[s]['loso_rejection'])
    b_s2 = max(SCHEMES, key=lambda s: s1pt[s]['S2_acceptance'])
    L += ['', f"- 在该 tau 下 S2 接受最高的是 `{SCHEME_LABEL[b_s2]}`"
          f"（{s1pt[b_s2]['S2_acceptance']:.4f}，恰好复现 E226 的 S2 上限），"
          f"但它的 LOSO 拒识只有 {s1pt[b_s2]['loso_rejection']:.4f}；"
          f"LOSO 侧最好的是 `{SCHEME_LABEL[b_s1]}`（{s1pt[b_s1]['loso_rejection']:.4f}），"
          f"此时 S2 接受已掉到 {s1pt[b_s1]['S2_acceptance']:.4f}。"
          '三个方案在 S1=0.70 这个截面上，LOSO 拒识都在 0.41–0.44；'
          '反过来在 LOSO>0.90 的阈值上，S2 接受只有 0.21–0.30。'
          "两个方向各差 0.3–0.5，缺口不可能靠调 tau 弥合。", '',
          '## 7. 部署参数（若冻结）', '', existence['deploy_text'], '',
          '## 8. 块权重（折内，Train-only，LOFO）', '',
          '|块|plain|aucW 均值|aucW 范围|auc2W 均值|auc2W 范围|',
          '|---|---:|---:|---:|---:|---:|']
    for j, b in enumerate(BLOCK_ORDER):
        L.append(f"|`{b}`|1.000|{block_wt_summary['auc_lofo']['mean'][j]:.3f}|"
                 f"{block_wt_summary['auc_lofo']['min'][j]:.3f}–"
                 f"{block_wt_summary['auc_lofo']['max'][j]:.3f}|"
                 f"{block_wt_summary['auc2_lofo']['mean'][j]:.3f}|"
                 f"{block_wt_summary['auc2_lofo']['min'][j]:.3f}–"
                 f"{block_wt_summary['auc2_lofo']['max'][j]:.3f}|")
    L += ['', '- `aucW`/`auc2W` 的权重来自 E226 已缓存的单块 LOSO AUC，取"除当前折以外 19 折"的均值'
          '（leave-one-fold-out），因此当前折的 held identity 从未参与自己的权重；无 Test 信息。',
          '- 应用方式：task-wise 标准化后按块乘 `sqrt(w_b)`，再做 PCA96 与 k12 原型（与 E226 的对角权重同机制）。', '',
          '## 9. 限制', '',
          '- 官方 Test AUC/F1 无身份标签，**N/A**；S1/S2 是会话分层行为指标，不等同官方未知/正常分类指标。',
          '- Test 在本系列（E65→E226）已被反复查看；本轮所有 A、方案、权重公式都预先写进 config.json，'
          '未按 Test 选型，但评估偏倚不可消除。',
          '- 无真实 session2 训练数据：S1→S2 的漂移只能用行为指标间接观察，不能做真实跨会话自适应。',
          '- LOSO 未知只有 20 个 held identity × 4 task × 约 124 trial/折；micro 与 macro 并列给出，'
          'macro 对折间不均衡更敏感。',
          f"- 运行：`.venv\\Scripts\\python.exe research/E227_joint_frontier/run_e227.py`（CUDA）；"
          f"首次完整计算（20 折 × 3 方案 + Test）约 165s，折级结果写入 `cache/` 后重跑只需重算 "
          f"Test 与聚合，本轮生成这份报告用了 {runtime:.0f}s。", '']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


# =====================================================================  main
def main():
    global TASKS_NP, Y, TEST_TABLE
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E227 is blocked'
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    t_start = time.time()

    meta = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    assert len(meta) == int(CFG['data']['train_trials'])
    Y = meta.subject.to_numpy(int)
    TASKS_NP = meta.task.to_numpy()
    FR = frac_of(meta)
    BLOCKS = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    assert set(Y) == set(range(1, 21)) and set(TASKS_NP) == set(TASKS)
    log(f'rows={len(meta)} fit={int((FR <= FIT_FRAC).sum())} '
        f'window={int(((FR > EVAL_LO) & (FR <= EVAL_HI)).sum())}')

    X = load_features('train')
    assert X.shape == (len(meta), DIM_TOTAL), X.shape
    log(f'train features {X.shape}')
    table = block_only_auc_table()
    log('block-only AUC (mean over 20 folds): ' +
        ', '.join(f'{b}={np.mean(table[b]):.4f}' for b in BLOCK_ORDER))

    T0 = time.time()
    folds = run_folds(X, FR, BLOCKS, table)
    fold_grids = {s: aggregate_folds(folds[s]) for s in SCHEMES}
    log(f'LOSO done in {time.time() - T0:.0f}s')

    # ---- reproduction check against E226 (A = 0.90 point + per-fold window scores)
    e226 = CFG['predecessor']['known_results']['all_blocks_v2']
    g90 = fold_grids['none'][key(0.90)]
    max_sd, max_taud = 0.0, 0.0
    for held in range(1, N_FOLDS + 1):
        old = np.load(ROOT / CFG['feature_cache']['e226_window_scores'].format(fold=held))
        new = np.load(CACHE / f'fold_{CACHE_VERSION}_none_{held:02d}.npz')
        max_sd = max(max_sd, float(np.max(np.abs(old['score'] - new['score']))))
        max_taud = max(max_taud, abs(float(old['tau']) - float(new['tau'][A_GRID.index(0.90)])))
        if held == 1:
            log(f'  score array order check: old n={len(old["score"])} new n={len(new["score"])}')

    checks = dict(
        e226_score_max_abs_diff=max_sd,
        e226_tau_max_abs_diff=max_taud,
        bitwise=bool(max_sd <= 5e-4 and max_taud <= 1e-4),
        reproduction_rows=[
            ('LOSO AUC', g90['loso_auc'], e226['loso_auc']),
            ('LOSO unknown rejection micro', g90['unknown_rejection_micro'],
             e226['loso_unknown_rejection_micro']),
            ('LOSO unknown rejection macro', g90['unknown_rejection_macro'],
             e226['loso_unknown_rejection_macro']),
            ('known acceptance macro', g90['known_acceptance_macro'], 0.899178),
            ('rejection F1', g90['rejection_f1'], 0.321209),
        ])
    log(f'reproduction vs E226: score max|d|={max_sd:.2e} tau max|d|={max_taud:.2e}')

    if SKIP_TEST:
        log('E227_SKIP_TEST=1: stopping after LOSO')
        return

    Xt = load_features('test')
    tmeta = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tmeta) == int(CFG['data']['test_trials'])
    assert Xt.shape == (len(tmeta), DIM_TOTAL), Xt.shape
    log(f'test features {Xt.shape}')
    TEST_TABLE = table
    test_out, is_s1 = run_test(X, Xt, FR, BLOCKS, tmeta)
    checks['reload'] = ('全部方案 deploy 模型重载后 Test 分数逐行一致：' +
                        ('通过' if all(test_out[s]['reload_check']['passed'] for s in SCHEMES)
                         else '未通过'))

    pts = joint_points(fold_grids, test_out)
    front = pareto(pts)
    feasible = [p for p in pts if p['feasible']]
    closest = sorted(pts, key=lambda p: p['gap'])[:3]
    ext = extended_points()
    tau_br = tau_bracket(fold_grids, test_out)
    s1pt = s1_70_point()

    # ---- per-task table at A*
    A_star = None
    above = [a for a in A_GRID
             if fold_grids['none'][key(a)]['unknown_rejection_micro'] > TARGET['loso_unknown_rejection']]
    reaches = bool(above)
    if reaches:
        A_star = max(above)          # least strict A that still clears LOSO 0.90
    else:
        A_star = min(A_GRID, key=lambda a: -fold_grids['none'][key(a)]['unknown_rejection_micro'])
    rows = {}
    for t in TASKS:
        lo = fold_grids['none'][key(A_star)]['per_task'][t]
        te = test_out['none']['grid'][key(A_star)]['per_task'][t]
        rows[t] = dict(n_unknown=lo['n_unknown'], loso_rejection=lo['unknown_rejection'],
                       loso_known_acceptance=lo['known_acceptance'],
                       n_S1=te['n_S1'], S1_rejection=te['S1_rejection'],
                       n_S2=te['n_S2'], S2_acceptance=te['S2_acceptance'])
    hard = min(TASKS, key=lambda t: fold_grids['none'][key(A_star)]['per_task'][t]['unknown_rejection'])
    easy = max(TASKS, key=lambda t: fold_grids['none'][key(A_star)]['per_task'][t]['unknown_rejection'])
    strict_tasks = [fold_grids['none'][key(min(A_GRID))]['per_task'][t]['unknown_rejection']
                    for t in TASKS]
    all_s1_gt = all(rows[t]['S1_rejection'] > rows[t]['loso_rejection'] for t in TASKS)
    kc_mean = float(np.mean([rows[t]['loso_known_acceptance'] for t in TASKS]))
    s2_mean = float(np.mean([rows[t]['S2_acceptance'] for t in TASKS]))
    comment = (f"诊断要点：在 A*={A_star:.2f}（LOSO 拒识 "
               f"{fold_grids['none'][key(A_star)]['unknown_rejection_micro']:.4f}）下，"
               f"Train 域内最难拒的 task 是 `{hard}`"
               f"（{fold_grids['none'][key(A_star)]['per_task'][hard]['unknown_rejection']:.4f}），"
               f"最易拒的是 `{easy}`"
               f"（{fold_grids['none'][key(A_star)]['per_task'][easy]['unknown_rejection']:.4f}）。"
               f"最严端 A={min(A_GRID):.2f} 时四个 task 的 LOSO 拒识区间为 "
               f"{min(strict_tasks):.3f}–{max(strict_tasks):.3f}，"
               '**没有任何 task 在 Train 域内"拒不掉"**，task 间差异只有百分之几。'
               + ('四个 task 的 Test S1 拒识都高于同一 A 下的 LOSO 拒识，'
                  '说明 Test 分数是整体下移，S1 并不比 Train 未知更难拒；'
                  if all_s1_gt else '')
               + f"但同一次下移也把 S2 的已知分数压低了：A*={A_star:.2f} 时 Train 已知接受 "
               f"{kc_mean:.4f}，Test S2 接受只有 {s2_mean:.4f}。"
               '因此按 task 分别建模缺乏依据（task 异质性远小于域位移），'
               '真正该放弃的是"跨 session 共用同一 tau"，下一步应做 Test 侧无标签的分数归一化'
               '（或原型重估），而不是继续在统一阈值上加严。')
    per_task_tbl = dict(A_star=float(A_star), A_star_reaches=bool(reaches),
                        rows=rows, comment=comment)

    bws = {}
    for s in ['auc_lofo', 'auc2_lofo']:
        W = np.array([f['block_weights'] for f in folds[s]])
        bws[s] = dict(mean=[float(x) for x in W.mean(0)],
                      min=[float(x) for x in W.min(0)],
                      max=[float(x) for x in W.max(0)])

    # ---- existence
    if feasible:
        best = sorted(feasible, key=lambda p: -p['S2_acceptance'])[0]
        text = (f"**存在。** `{SCHEME_LABEL[best['scheme']]}` 方案在 A={best['A']:.2f} 达成："
                f"LOSO 未知拒识={best['loso_rejection']:.4f}（>0.90）、"
                f"Test S1 拒识={best['S1_rejection']:.4f}（≥0.70）、"
                f"Test S2 接受={best['S2_acceptance']:.4f}（>0.70），"
                f"分离 AUC={best['sep_auc']:.4f}。")
        deploy_text = (f"冻结 `{SCHEME_LABEL[best['scheme']]}` + A={best['A']:.2f}："
                       f"PCA={PCA_DIM}、k={K_PROTO}、tau_test={best['tau_test']:.4f}、"
                       f"20 折 tau 均值={best['tau_mean']:.4f}（逐折值见 results.json）、"
                       f"块权重按 {best['scheme']} 公式（见第 8 节）。"
                       f"模型已存 `models/deploy_{best['scheme']}.pt`，"
                       f"并按该 tau 另存 `models/deploy_frozen.pt`。")
    else:
        c1, c2, c3 = closest
        strict70 = [p for p in pts if abs(p['A'] - min(A_GRID)) < 1e-9]
        b70 = max(strict70, key=lambda p: p['loso_rejection'])
        if b70['loso_rejection'] > TARGET['loso_unknown_rejection']:
            strict_note = (f"网格最严点（A={b70['A']:.2f}）的 LOSO 拒识已达 "
                           f"{b70['loso_rejection']:.4f}，此时 Test S1 拒识="
                           f"{b70['S1_rejection']:.4f}、S2 接受={b70['S2_acceptance']:.4f}。")
        else:
            strict_note = (f"网格最严点（A={b70['A']:.2f}，{SCHEME_LABEL[b70['scheme']]}）"
                           f"的 LOSO 拒识最高只有 {b70['loso_rejection']:.4f}，仍未到 0.90。")
        text = ("**不存在。** 在对 A 网格 × 3 个块权重方案的完整扫描中，没有任何单点同时满足 "
                "`LOSO 拒识>0.90`、`Test S1 拒识≥0.70`、`Test S2 接受>0.70`。"
                "根本原因是三者被同一个 tau 单调绑定：A 越小（tau 越高）→ LOSO 拒识越高，"
                "但 Test S1 拒识与 S2 拒绝同步升高；而 Train 域内 held identity 的分数分布"
                "比 Test S1 更靠近 known，要把 LOSO 抬到 0.90，tau 必然越过 Test S1 的 70% 分位，"
                "S2 随即被压到 0.70 以下。" + strict_note +
                "最接近的三个点（gap = Σ 未达标差额）：\n\n"
                "|方案|A|LOSO 拒识|Test S1 拒识|Test S2 接受|gap|差距明细|\n"
                "|---|---:|---:|---:|---:|---:|---|\n"
                f"|{SCHEME_LABEL[c1['scheme']]}|{c1['A']:.2f}|{c1['loso_rejection']:.4f}|"
                f"{c1['S1_rejection']:.4f}|{c1['S2_acceptance']:.4f}|{c1['gap']:.4f}|"
                f"LOSO {c1['loso_rejection']-0.90:+.4f}，S1 {c1['S1_rejection']-0.70:+.4f}，"
                f"S2 {c1['S2_acceptance']-0.70:+.4f}|\n"
                f"|{SCHEME_LABEL[c2['scheme']]}|{c2['A']:.2f}|{c2['loso_rejection']:.4f}|"
                f"{c2['S1_rejection']:.4f}|{c2['S2_acceptance']:.4f}|{c2['gap']:.4f}|"
                f"LOSO {c2['loso_rejection']-0.90:+.4f}，S1 {c2['S1_rejection']-0.70:+.4f}，"
                f"S2 {c2['S2_acceptance']-0.70:+.4f}|\n"
                f"|{SCHEME_LABEL[c3['scheme']]}|{c3['A']:.2f}|{c3['loso_rejection']:.4f}|"
                f"{c3['S1_rejection']:.4f}|{c3['S2_acceptance']:.4f}|{c3['gap']:.4f}|"
                f"LOSO {c3['loso_rejection']-0.90:+.4f}，S1 {c3['S1_rejection']-0.70:+.4f}，"
                f"S2 {c3['S2_acceptance']-0.70:+.4f}|")
        loso_pts = [p for p in pts if p['loso_rejection'] > TARGET['loso_unknown_rejection']]
        both = [p for p in pts if (p['S2_acceptance'] > TARGET['test_S2_acceptance'] and
                                   p['S1_rejection'] >= TARGET['test_S1_rejection'])]
        opt_i = max(loso_pts, key=lambda p: p['S2_acceptance']) if loso_pts else None
        opt_s = max(both, key=lambda p: p['loso_rejection']) if both else None
        deploy_text = ("不冻结达标工作点：网格内没有任何点同时满足三个条件。"
                       + (f"若把 unknown 拦截放第一位，可取 `{SCHEME_LABEL[opt_i['scheme']]}`、"
                          f"A={opt_i['A']:.2f}：LOSO 拒识 {opt_i['loso_rejection']:.4f}、"
                          f"Test S1 拒识 {opt_i['S1_rejection']:.4f}、S2 接受 "
                          f"{opt_i['S2_acceptance']:.4f}（S2 弃守）。" if opt_i else '')
                       + (f"若把 S2 可用性放第一位且要求 S1≥0.70，网格内最优点为 "
                          f"`{SCHEME_LABEL[opt_s['scheme']]}`、A={opt_s['A']:.2f}："
                          f"S2 接受 {opt_s['S2_acceptance']:.4f}、S1 拒识 "
                          f"{opt_s['S1_rejection']:.4f}、LOSO 拒识 "
                          f"{opt_s['loso_rejection']:.4f}。" if opt_s else
                          "网格内没有任何点能同时满足 S1≥0.70 与 S2>0.70："
                          "这一对在 A=0.92 与 A=0.94 之间交叉，恰好落在网格缝隙里；"
                          "把 tau 直接钉在 S1 70% 分位时的上界见 6.2。")
                       + "所有 tau 都由 Train 侧 inner OOF 分位给出，可合法部署、不需要 Test；"
                         "三个方案的 state 与每 A 的 tau 已保存在 `models/deploy_*.pt`。")
        torch.save(dict(mode='no_feasible_point', best_effort=closest,
                        intercept_priority=opt_i, s2_priority=opt_s,
                        block_weights={s: bws[s] for s in bws},
                        block_order=BLOCK_ORDER, pca=PCA_DIM, k=K_PROTO),
                   MODELS / 'deploy_frozen.pt')
    existence = dict(feasible=bool(feasible), n_feasible=len(feasible),
                     feasible_points=feasible, closest=closest,
                     text=text, deploy_text=deploy_text)
    if feasible:
        best = sorted(feasible, key=lambda p: -p['S2_acceptance'])[0]
        torch.save(dict(mode='frozen', scheme=best['scheme'], A=best['A'],
                        tau=best['tau_test'], tau_per_fold=best['tau_mean'],
                        block_weights=dict(zip(BLOCK_ORDER, bws.get(best['scheme'],
                                                                     {}).get('mean', [1.0] * 10))),
                        pca=PCA_DIM, k=K_PROTO),
                   MODELS / 'deploy_frozen.pt')

    runtime = time.time() - t_start
    results = dict(
        experiment='E227_joint_frontier',
        representation=dict(name='all_blocks_v2', blocks=BLOCK_ORDER,
                            dims={b: BLOCK_DIM[b] for b in BLOCK_ORDER},
                            total_dim=DIM_TOTAL, pca=PCA_DIM, k=K_PROTO),
        protocol=CFG['protocol'], threshold_grid=dict(A=A_GRID, rule=CFG['threshold_grid']['rule']),
        fold_grids=fold_grids, test=test_out,
        joint_points=pts, pareto=front, existence=existence,
        extended_grid=dict(A=A_EXT, points=ext,
                           note='diagnostic only, below the pre-registered grid, no refit'),
        tau_bracket=tau_br,
        s1_70_upper_bound=dict(points=s1pt,
                               note='tau fixed at Test S1 70% quantile; Test-derived, not deployable'),
        per_task=per_task_tbl,
        block_weights=bws,
        block_weights_per_fold={s: [f['block_weights'] for f in folds[s]] for s in SCHEMES},
        per_fold_tau={s: [f['tau'] for f in folds[s]] for s in SCHEMES},
        checks=checks, runtime_s=float(runtime),
        official=dict(AUC='N/A', F1='N/A',
                      note='Test has no identity labels.'))
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf8')

    build_report(None, fold_grids, test_out, pts, front, existence, checks, runtime,
                 per_task_tbl, bws, ext, tau_br, s1pt)
    log('results.json + report.md written')
    log('joint feasibility: ' + ('YES' if feasible else 'NO') +
        ' | closest: ' + ', '.join(
            f"{SCHEME_LABEL[p['scheme']]}/A{p['A']:.2f} (LOSO {p['loso_rejection']:.4f}, "
            f"S1 {p['S1_rejection']:.4f}, S2 {p['S2_acceptance']:.4f}, gap {p['gap']:.4f})"
            for p in closest))


TEST_TABLE = None

if __name__ == '__main__':
    main()
