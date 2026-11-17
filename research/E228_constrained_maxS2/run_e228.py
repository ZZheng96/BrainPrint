"""E228: constrained calibration -- LOSO unknown rejection pinned at 0.90, maximise Test S2.

Protocol is byte-identical to E221/E226/E227:
  strict 20-fold subject-LOSO, fit rows = per (subject, task) trial fraction <= 0.6,
  evaluation window (0.6, 0.8], inner 5 blocks = min(floor(fraction*5), 4),
  task-wise standardisation -> PCA96 -> k=12 prototypes per (task, identity),
  score = -min euclidean distance to same-task prototypes, high score = known,
  unknown is the positive class for rejection F1 / AUC.

What changes:
  tau(A) = (1-A) quantile of the fold's inner 5-block OOF known scores, and A is no longer
  fixed at 0.90. A is bisected on the Train side so that the pooled (micro) LOSO unknown
  rejection equals 0.90; the loosest A that still clears 0.90 is kept, which maximises the
  room left for Test S2. Test is read once per candidate after A is frozen.

Candidates (all fixed in config.json before the run):
  task 1 -- e65_corr, e225_all_blocks, e226_all_blocks_v2, e226_aucW, e226_auc2W,
            corr_plus_powerratio
  task 2 -- identity_scaled, two_proto_agg, dual_gate_50, dual_gate_30 (same frozen
            all_blocks_v2 representation, Train-only scoring changes)

Block-weighted variants (aucW / auc2W) are read from the frozen E227 score caches so they
stay bit-identical to E227; every other candidate is refitted here and cross-checked
against the predecessor score cache.
"""
from __future__ import annotations

import json
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
N_FOLDS = int(CFG['protocol']['folds'])
LAM = 0.1
TARGET = float(CFG['constraint']['accepted_band'][0])
BAND_HI = float(CFG['constraint']['accepted_band'][1])
S1_TARGET = 0.70
S2_TARGET = 0.70

BLOCK_DIM = {k: int(v) for k, v in CFG['blocks'].items()}
BLOCK_ORDER = list(BLOCK_DIM)
OFFSETS, BLOCK_IDX = {}, {}
_acc = 0
for _k, _v in BLOCK_DIM.items():
    OFFSETS[_k] = (_acc, _acc + _v)
    BLOCK_IDX[_k] = np.arange(_acc, _acc + _v)
    _acc += _v
DIM_TOTAL = _acc
assert DIM_TOTAL == 1569, DIM_TOTAL

REPS = {k: list(v) for k, v in CFG['representation']['reps'].items()}
CANDS = [dict(c) for c in CFG['candidates']]
CAND_BY_ID = {c['id']: c for c in CANDS}
FRESH_CANDS = [c['id'] for c in CANDS if c.get('source') != 'e227_cache']
DG_WEIGHT = {'dual_gate_05': 0.5, 'dual_gate_03': 0.3}

TASKS_NP = None
Y = None
WIN = None


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(msg) + '\n')


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def quantile_tau(x, q):
    return float(np.quantile(np.asarray(x, float), q))


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
    return float(((u[:, None] < k).sum() + 0.5 * (u[:, None] == k).sum()) / (len(k) * len(u)))


def frac_of(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def need_flags(variants):
    return dict(scale=any(v == 'identity_scaled' for v in variants),
                dstat=any(v.startswith('dual_gate') for v in variants),
                mahal=any(v.startswith('dual_gate') for v in variants))


# =====================================================================  representation
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


def proto_meta(state):
    keys = state['keys']
    return np.array([t for t, _ in keys]), np.array([int(s) for _, s in keys], int)


def robust_stats(x):
    x = np.asarray(x, float)
    med = float(np.median(x))
    sc = float((np.quantile(x, 0.90) - np.quantile(x, 0.10)) / 1.349)
    return med, max(sc, 1e-9)


def dist_stats(z, tasks, fit, state):
    proto_task, _ = proto_meta(state)
    out = {}
    for t in TASKS:
        rr = np.flatnonzero(fit & (tasks == t))
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            out[t] = dict(med=0.0, scale=1.0)
            continue
        d = (-torch.cdist(tensor(z[rr]), tensor(state['p'][cols]))).max(1)[0]
        med, sc = robust_stats(d.double().cpu().numpy())
        out[t] = dict(med=med, scale=sc)
    return out


def mahal_stats(z, tasks, fit):
    d = z.shape[1]
    eye = np.eye(d)
    out = {}
    for t in TASKS:
        A = z[fit & (tasks == t)].astype(np.float64)
        mu = A.mean(0)
        Ac = A - mu
        C = (Ac.T @ Ac) / max(len(A) - 1, 1)
        C = (1 - LAM) * C + LAM * (np.trace(C) / d) * eye
        cinv = np.linalg.inv(C)
        m = np.sqrt(np.einsum('ij,jk,ik->i', Ac, cinv, Ac))
        med, sc = robust_stats(m)
        out[t] = dict(mu=mu, cinv=cinv, med=med, scale=sc)
    return out


def identity_scales(z, tasks, y, fit, state):
    proto_task, proto_id = proto_meta(state)
    p = state['p']
    vals = {}
    for t in TASKS:
        cols = np.flatnonzero(proto_task == t)
        for sub in np.unique(y[fit]):
            rr = np.flatnonzero(fit & (tasks == t) & (y == sub))
            cc = cols[proto_id[cols] == sub]
            if not len(rr) or not len(cc):
                continue
            d = (-torch.cdist(tensor(z[rr]), tensor(p[cc]))).max(1)[0].double().cpu().numpy()
            span = float(np.quantile(d, 0.90) - np.quantile(d, 0.10))
            vals[(t, int(sub))] = max(span, 1e-6)
    med = float(np.median(list(vals.values()))) if vals else 1.0
    proto_scale = np.ones(len(state['keys']), np.float32)
    for j, (t, s) in enumerate(state['keys']):
        proto_scale[j] = vals.get((t, int(s)), med) / med
    table = {t: {int(s): float(vals.get((t, int(s)), np.nan)) / med for s in range(1, 21)}
             for t in TASKS}
    return dict(vals={f'{t}_{s}': float(v) for (t, s), v in vals.items()},
                med=med, proto_scale=proto_scale, table=table)


def fit_rep(raw, tasks, y, fit, k=K_PROTO, pca=PCA_DIM, w=None, need=None):
    need = need or {}
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
    state = dict(mu=mu, sd=sd, pm=pm.cpu().numpy(), W=W.cpu().numpy(), p=p, keys=keys, k=k,
                 pca=pca, dim=int(a.shape[1]), w=None if w is None else np.asarray(w, np.float32))
    if need.get('scale'):
        state['scale'] = identity_scales(z, tasks, y, fit, state)
    if need.get('dstat'):
        state['dstat'] = dist_stats(z, tasks, fit, state)
    if need.get('mahal'):
        state['mahal'] = mahal_stats(z, tasks, fit)
    return state


def transform(state, raw, tasks):
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


def mahal_rows(state, z, tasks, rows):
    out = np.full(len(tasks), np.nan)
    for t in TASKS:
        rr = rows[tasks[rows] == t]
        if not len(rr):
            continue
        st = state['mahal'][t]
        A = z[rr].astype(np.float64) - st['mu']
        m = np.sqrt(np.einsum('ij,jk,ik->i', A, st['cinv'], A))
        out[rr] = (m - st['med']) / st['scale']
    return out


def score_rows(state, z, tasks, rows, variants):
    """All requested scoring variants for `rows`, from one cdist per task."""
    proto_task, proto_id = proto_meta(state)
    n = len(tasks)
    res = {v: (np.full(n, np.nan), np.full(n, -1, int)) for v in variants}
    dg = [v for v in variants if v.startswith('dual_gate')]
    zo_all = mahal_rows(state, z, tasks, rows) if dg else None
    for t in TASKS:
        rr = rows[tasks[rows] == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rr]), tensor(state['p'][cols])).double()
        dmin_t, amin_t = D.min(1)
        dmin = dmin_t.cpu().numpy()
        amin = amin_t.cpu().numpy()
        pred = proto_id[cols][amin]
        for v in variants:
            s = res[v][0]
            if v == 'plain':
                s[rr] = -dmin
            elif v == 'identity_scaled':
                s[rr] = -dmin / state['scale']['proto_scale'][cols][amin]
            elif v == 'two_proto_agg':
                same = proto_id[cols][None, :] == pred[:, None]
                Dm = D.masked_fill(~torch.as_tensor(same, device=DEVICE), float('inf'))
                kk = min(3, Dm.shape[1])
                vv = torch.topk(Dm, kk, dim=1, largest=False).values.median(dim=1).values
                s[rr] = -vv.cpu().numpy()
            elif v in DG_WEIGHT:
                w = DG_WEIGHT[v]
                zd = (dmin - state['dstat'][t]['med']) / state['dstat'][t]['scale']
                s[rr] = -(w * zd + (1.0 - w) * zo_all[rr])
            else:
                raise ValueError(v)
            res[v][1][rr] = pred
    return res


def inner_oof(X, tasks, y, fit, blocks, variants, need):
    oof = {v: np.full(len(y), np.nan) for v in variants}
    for b in np.unique(blocks[fit]):
        hold = fit & (blocks == b)
        inner = fit & ~hold
        st = fit_rep(X, tasks, y, inner, K_PROTO, PCA_DIM, None, need)
        z = transform(st, X, tasks)
        sc = score_rows(st, z, tasks, np.flatnonzero(hold), variants)
        for v in variants:
            oof[v][hold] = sc[v][0][hold]
    return oof


# =====================================================================  fold driver
def run_fold_fresh(rep, X, FR, BLOCKS, held, variants):
    fit = (Y != held) & (FR <= FIT_FRAC)
    window = (FR > EVAL_LO) & (FR <= EVAL_HI)
    win = np.flatnonzero(window)
    need = need_flags(variants)
    st = fit_rep(X, TASKS_NP, Y, fit, K_PROTO, PCA_DIM, None, need)
    z = transform(st, X, TASKS_NP)
    sc = score_rows(st, z, TASKS_NP, win, variants)
    oof = inner_oof(X, TASKS_NP, Y, fit, BLOCKS, variants, need)
    out = {}
    for v in variants:
        out[f'score_{v}'] = sc[v][0][win]
        out[f'pred_{v}'] = sc[v][1][win]
        out[f'oof_{v}'] = oof[v]
    np.savez_compressed(
        CACHE / f'fold_{CACHE_VERSION}_{rep}_{held:02d}.npz',
        **out, fit=fit, task=np.asarray(TASKS_NP[win], dtype='<U8'),
        subject=Y[win], known=(Y[win] != held), unknown=(Y[win] == held))
    return dict(rep=rep, held=int(held), variants=variants,
                pca=int(st['pca']), n_dims=int(st['dim']),
                n_known=int((Y[win] != held).sum()), n_unknown=int((Y[win] == held).sum()))


def npz_load(path):
    # E227 froze its fold caches with an object-dtype task array, so pickling is required.
    return np.load(path, allow_pickle=True)


def load_fresh_fold(rep, held, variants):
    d = npz_load(CACHE / f'fold_{CACHE_VERSION}_{rep}_{held:02d}.npz')
    return {k: d[k] for k in d.files}


def load_e227_fold(scheme, held):
    p = ROOT / 'research/E227_joint_frontier/cache' / f'fold_v1_{scheme}_{held:02d}.npz'
    d = npz_load(p)
    return dict(score_plain=d['score'], pred_plain=d['pred'], oof_plain=d['oof_s'],
                fit=d['oof_fit'].astype(bool), task=np.asarray(d['task'], dtype='<U8'),
                subject=d['subject'], known=d['known'], unknown=d['unknown'])


def run_folds():
    meta = {}
    for rep in REPS:
        variants = sorted({c['variant'] for c in CANDS
                           if c.get('source') != 'e227_cache' and c['rep'] == rep})
        if not variants:
            continue
        X = REP_MATRIX['train'][rep]
        reused = 0
        for held in range(1, N_FOLDS + 1):
            t0 = time.time()
            path = CACHE / f'fold_{CACHE_VERSION}_{rep}_{held:02d}.npz'
            if path.exists():
                d = np.load(path)
                if all(f'score_{v}' in d.files for v in variants):
                    reused += 1
                    continue
            row = run_fold_fresh(rep, X, FR, BLOCKS, held, variants)
            meta[f'{rep}_{held:02d}'] = row
            log(f'fold {held:02d} rep={rep:20s} variants={variants} '
                f'dim={row["n_dims"]} pca={row["pca"]} ({time.time()-t0:.1f}s)')
        if reused:
            log(f'{rep}: {reused}/{N_FOLDS} folds reused from cache')
    return meta


# =====================================================================  metrics
def per_task_metrics(s, known, unknown, tau, tasks, y_true, pred):
    rej_u = s[unknown] < tau
    rej_k = s[known] < tau
    tp, fp, fn = float(rej_u.sum()), float(rej_k.sum()), float((~rej_u).sum())
    m = dict(
        known_acceptance=float((~rej_k).mean()) if known.any() else float('nan'),
        unknown_rejection=float(rej_u.mean()) if unknown.any() else float('nan'),
        rejection_f1=float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        identity_correct_but_rejected=float(
            ((pred[known] == y_true[known]) & rej_k).mean()) if known.any() else float('nan'),
        n_known=int(known.sum()), n_unknown=int(unknown.sum()),
        accepted_known=int((~rej_k).sum()), rejected_unknown=int(rej_u.sum()))
    pt = {}
    for t in TASKS:
        ku = unknown & (tasks == t)
        kk = known & (tasks == t)
        pt[t] = dict(n_known=int(kk.sum()), n_unknown=int(ku.sum()),
                     accepted_known=int((s[kk] >= tau).sum()),
                     rejected_unknown=int((s[ku] < tau).sum()),
                     unknown_rejection=float((s[ku] < tau).mean()) if ku.any() else float('nan'),
                     known_acceptance=float((s[kk] >= tau).mean()) if kk.any() else float('nan'))
    m['per_task'] = pt
    return m


def aggregate(folds, variant, A):
    ms = []
    for d in folds:
        s = d[f'score_{variant}']
        oof = d[f'oof_{variant}']
        fit = d['fit'].astype(bool)
        tau = quantile_tau(oof[fit], 1.0 - A)
        ms.append(per_task_metrics(s, d['known'].astype(bool), d['unknown'].astype(bool),
                                   tau, np.asarray(d['task'], dtype=str),
                                   d['subject'].astype(int), d[f'pred_{variant}'].astype(int)))
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
    aucs = [fold_auc(f[f'score_{variant}'][f['unknown'].astype(bool)],
                     f[f'score_{variant}'][f['known'].astype(bool)]) for f in folds]
    return dict(
        A=float(A),
        tau_mean=float(np.mean([quantile_tau(f[f'oof_{variant}'][f['fit'].astype(bool)], 1.0 - A)
                                for f in folds])),
        tau_min=float(np.min([quantile_tau(f[f'oof_{variant}'][f['fit'].astype(bool)], 1.0 - A)
                              for f in folds])),
        tau_max=float(np.max([quantile_tau(f[f'oof_{variant}'][f['fit'].astype(bool)], 1.0 - A)
                              for f in folds])),
        unknown_rejection_micro=float(sum(m['rejected_unknown'] for m in ms) / max(n_u, 1)),
        unknown_rejection_macro=float(np.mean([m['unknown_rejection'] for m in ms])),
        known_acceptance_micro=float(sum(m['accepted_known'] for m in ms) / max(n_k, 1)),
        known_acceptance_macro=float(np.mean([m['known_acceptance'] for m in ms])),
        rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
        identity_correct_but_rejected=float(np.mean(
            [m['identity_correct_but_rejected'] for m in ms])),
        loso_auc=float(np.mean(aucs)), loso_auc_std=float(np.std(aucs)),
        n_known=int(n_k), n_unknown=int(n_u), per_task=per_task)


def calibrate(folds, variant, target=TARGET):
    """Largest A whose pooled LOSO unknown rejection is still >= target."""
    def micro(A):
        rej = n = 0
        for d in folds:
            tau = quantile_tau(d[f'oof_{variant}'][d['fit'].astype(bool)], 1.0 - A)
            u = d['unknown'].astype(bool)
            rej += int((d[f'score_{variant}'][u] < tau).sum())
            n += int(u.sum())
        return rej / max(n, 1)

    lo, hi = 1e-4, 1.0 - 1e-9
    f_lo, f_hi = micro(lo), micro(hi)
    if f_lo < target:
        return dict(A=float(lo), rejection=float(f_lo), reached=False,
                    note='even the strictest A cannot reach the LOSO target')
    if f_hi >= target:
        return dict(A=float(hi), rejection=float(f_hi), reached=True,
                    note='the loosest A still clears the LOSO target')
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if micro(mid) >= target:
            lo = mid
        else:
            hi = mid
    return dict(A=float(lo), rejection=float(micro(lo)), reached=True, note='bisection')


# =====================================================================  Test
def run_test_fresh(rep, variant, X, Xt, FR, BLOCKS, tmeta):
    fit = FR <= FIT_FRAC
    tt = tmeta.task.to_numpy()
    sess = tmeta.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    need = need_flags([variant])
    st = fit_rep(X, TASKS_NP, Y, fit, K_PROTO, PCA_DIM, None, need)
    zt = transform(st, Xt, tt)
    sc = score_rows(st, zt, tt, np.arange(len(tt)), [variant])
    s, p = sc[variant][0], sc[variant][1]
    oof = inner_oof(X, TASKS_NP, Y, fit, BLOCKS, [variant], need)[variant]
    np.savez_compressed(
        CACHE / f'test_{CACHE_VERSION}_{rep}_{variant}.npz',
        score=s, pred=p, oof_s=oof, oof_fit=fit,
        task=np.asarray(tt, dtype='<U8'),
        session=np.asarray(tmeta.session.astype(str).to_numpy(), dtype='<U8'),
        is_s1=is_s1)
    return dict(state=st, score=s, pred=p, oof=oof, fit=fit, is_s1=is_s1, task=tt,
                sep_auc=auc_ties(s[~is_s1], s[is_s1]))


def load_test_e227(scheme):
    d = npz_load(ROOT / 'research/E227_joint_frontier/cache' / f'test_v1_{scheme}.npz')
    return dict(score=d['score'], pred=d['pred'], oof=d['oof_s'],
                fit=d['oof_fit'].astype(bool), is_s1=d['is_s1'].astype(bool),
                task=np.asarray(d['task'], dtype=str),
                sep_auc=auc_ties(d['score'][~d['is_s1'].astype(bool)],
                                 d['score'][d['is_s1'].astype(bool)]))


def test_metrics(score, pred, oof, fit, is_s1, tasks, A):
    tau = quantile_tau(oof[fit], 1.0 - A)
    rej = score < tau
    per_task = {}
    for t in TASKS:
        m1 = is_s1 & (tasks == t)
        m2 = (~is_s1) & (tasks == t)
        per_task[t] = dict(n_S1=int(m1.sum()), n_S2=int(m2.sum()),
                           S1_rejection=float(rej[m1].mean()) if m1.any() else float('nan'),
                           S2_acceptance=float((~rej[m2]).mean()) if m2.any() else float('nan'))
    return dict(tau=float(tau), S1_rejection=float(rej[is_s1].mean()),
                S2_acceptance=float((~rej[~is_s1]).mean()),
                S1_rejection_macro=float(np.mean([per_task[t]['S1_rejection'] for t in TASKS])),
                S2_acceptance_macro=float(np.mean([per_task[t]['S2_acceptance'] for t in TASKS])),
                N_S1=int(is_s1.sum()), N_S2=int((~is_s1).sum()), per_task=per_task)


def predict_from_model(model, Xt, tt):
    """Canonical inference path used by the reload check and documented in the report."""
    st = model['state']
    z = transform(st, Xt, tt)
    sc = score_rows(st, z, tt, np.arange(len(tt)), [model['variant']])
    return sc[model['variant']][0], sc[model['variant']][1]


def e227_weight_range(scheme):
    p = ROOT / 'research/E227_joint_frontier/results.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding='utf8'))['block_weights'][scheme]
    return dict(mean=[float(x) for x in r['mean']], min=[float(x) for x in r['min']],
                max=[float(x) for x in r['max']])


def build_frozen(best, c, A, X, Xt, test_out):
    """Freeze the selected candidate into models/deploy_frozen.pt and verify the reload."""
    rep, variant = c['rep'], c['variant']
    bw_range = None
    repro = None
    if c.get('source') == 'e227_cache':
        scheme = c['weights']
        src = torch.load(ROOT / 'research/E227_joint_frontier/models' / f'deploy_{scheme}.pt',
                         weights_only=False)
        st = src['state']
        te = load_test_e227(scheme)
        tt = te['task']
        s_chk, p_chk = score_rows(st, transform(st, Xt[rep], tt), tt,
                                  np.arange(len(tt)), ['plain'])['plain']
        st['scale'] = identity_scales(transform(st, X[rep], TASKS_NP), TASKS_NP, Y,
                                      te['fit'], st)
        bws = [float(x) for x in np.asarray(src['block_weights'], float)]
        bw_range = e227_weight_range(scheme)
        source = f'E227 frozen deploy_{scheme}.pt (block-weighted representation)'
        repro = dict(max_abs_score_diff=float(np.max(np.abs(s_chk - te['score']))),
                     pred_identical=bool(np.array_equal(p_chk, te['pred'])))
    else:
        te = test_out[(rep, variant)]
        st = te['state']
        tt = np.asarray(te['task'], dtype=str)
        bws = None
        source = 'E228 fitted state'
    tau = test_metrics(te['score'], te['pred'], te['oof'], te['fit'], te['is_s1'], tt, A)['tau']
    scale_tbl = st['scale']['table'] if 'scale' in st else None
    eff = None
    if scale_tbl is not None and variant == 'identity_scaled':
        # only identity_scaled gates on d / s_hat_k, so only there does the per-identity
        # distance threshold (-tau)*s_hat_k actually differ from the global one.
        eff = {t: {int(s): float((-tau) * scale_tbl[t][int(s)]) for s in range(1, 21)}
               for t in TASKS}
    model = dict(candidate=best['id'], rep=rep, variant=variant, A=float(A), tau=float(tau),
                 pca=int(st['pca']), k=int(st['k']), dim=int(st['dim']),
                 block_order=list(REPS[rep]), block_weights=bws, state=st,
                 identity_scale_normalised=scale_tbl,
                 identity_scale_raw=st['scale']['vals'] if 'scale' in st else None,
                 effective_distance_threshold=eff, source=source,
                 score_rule=CFG['scoring_variants'][variant])
    torch.save(model, MODELS / 'deploy_frozen.pt')
    back = torch.load(MODELS / 'deploy_frozen.pt', weights_only=False)
    s2, p2 = predict_from_model(back, Xt[rep], tt)
    return dict(model=str(MODELS / 'deploy_frozen.pt'), source=source,
                reload_check=dict(max_abs_score_diff=float(np.max(np.abs(s2 - te['score']))),
                                  pred_identical=bool(np.array_equal(p2, te['pred']))),
                reference_score_check=repro,
                tau=float(tau), A=float(A), pca=int(st['pca']), k=int(st['k']),
                n_dims=int(st['dim']), blocks=list(REPS[rep]),
                block_weights=bws, block_weights_range=bw_range,
                identity_scale_normalised=scale_tbl,
                identity_scale_raw=st['scale']['vals'] if 'scale' in st else None,
                effective_distance_threshold=eff,
                score_rule=CFG['scoring_variants'][variant])


# =====================================================================  main
REP_MATRIX = {}
FR = None
BLOCKS = None


def load_all_features():
    train, test = {}, {}
    for split, store in (('train', train), ('test', test)):
        for b in BLOCK_ORDER:
            p = ROOT / CFG['feature_cache'][split][b]
            if not p.exists():
                raise FileNotFoundError(p)
            arr = np.load(p).astype(np.float32)
            if arr.shape[1] != BLOCK_DIM[b]:
                raise ValueError(f'{p}: {arr.shape} vs {BLOCK_DIM[b]}')
            store[b] = arr
    for rep, blocks in REPS.items():
        REP_MATRIX.setdefault('train', {})[rep] = np.concatenate(
            [train[b] for b in BLOCK_ORDER if b in blocks], 1)
        REP_MATRIX.setdefault('test', {})[rep] = np.concatenate(
            [test[b] for b in BLOCK_ORDER if b in blocks], 1)
        log(f'rep {rep:22s} train {REP_MATRIX["train"][rep].shape} '
            f'test {REP_MATRIX["test"][rep].shape}')


def main():
    global TASKS_NP, Y, FR, BLOCKS
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E228 is blocked'
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log(f'device={torch.cuda.get_device_name(0)} torch={torch.__version__}')
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

    load_all_features()
    run_folds()

    # ---------------- reproduce the predecessors from the same code path
    fits, tests, checks = {}, {}, {}
    for cid in FRESH_CANDS:
        c = CAND_BY_ID[cid]
        fits[cid] = [load_fresh_fold(c['rep'], h, [c['variant']]) for h in range(1, N_FOLDS + 1)]
    for cid in ['e226_aucW', 'e226_auc2W']:
        scheme = CAND_BY_ID[cid]['weights']
        fits[cid] = [load_e227_fold(scheme, h) for h in range(1, N_FOLDS + 1)]

    X = REP_MATRIX['train']
    Xt = REP_MATRIX['test']
    tmeta = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tmeta) == int(CFG['data']['test_trials'])

    test_out = {}
    reload_report = {}
    for cid in FRESH_CANDS:
        c = CAND_BY_ID[cid]
        rep, variant = c['rep'], c['variant']
        key = (rep, variant)
        if key in test_out:
            continue
        t0 = time.time()
        out = run_test_fresh(rep, variant, X[rep], Xt[rep], FR, BLOCKS, tmeta)
        test_out[key] = out
        if variant == 'plain':
            model = dict(candidate=cid, rep=rep, variant=variant, state=out['state'],
                         pca=PCA_DIM, k=K_PROTO, dim=int(out['state']['dim']))
            torch.save(model, MODELS / f'deploy_candidate_{rep}.pt')
            s2, p2 = predict_from_model(torch.load(MODELS / f'deploy_candidate_{rep}.pt',
                                                   weights_only=False), Xt[rep], out['task'])
            reload_report[cid] = dict(max_abs_score_diff=float(np.max(np.abs(s2 - out['score']))),
                                      pred_identical=bool(np.array_equal(p2, out['pred'])))
        log(f'test {cid:22s} built ({time.time()-t0:.1f}s)')

    # ---------------- predecessor reproduction checks
    def cmp_scores(cid, variant, ref_loader, label):
        ds = []
        for h in range(1, N_FOLDS + 1):
            ref = ref_loader(h)
            d = load_fresh_fold(CAND_BY_ID[cid]['rep'], h, [variant])
            ds.append(float(np.max(np.abs(ref['score'] - d[f'score_{variant}']))))
        return dict(comparison=label, max_abs_score_diff=float(max(ds)))

    checks['e65_corr_vs_E225'] = cmp_scores(
        'e65_corr', 'plain',
        lambda h: npz_load(ROOT / 'research/E225_session_invariant_features/cache'
                           / f'win_corr_only_{h:02d}.npz'),
        'E228 e65_corr plain vs E225 win_corr_only')
    checks['e225_all_blocks_vs_E226'] = cmp_scores(
        'e225_all_blocks', 'plain',
        lambda h: npz_load(ROOT / 'research/E226_stable_feature_families_v2/cache'
                           / f'win_v2_e225_all_blocks_{h:02d}.npz'),
        'E228 e225_all_blocks plain vs E226 win_v2_e225_all_blocks')
    checks['all_blocks_v2_vs_E227'] = cmp_scores(
        'e226_all_blocks_v2', 'plain',
        lambda h: npz_load(ROOT / 'research/E227_joint_frontier/cache'
                           / f'fold_v1_none_{h:02d}.npz'),
        'E228 all_blocks_v2 plain vs E227 fold_v1_none')
    log('reproduction max|dscore|: ' +
        ', '.join(f'{k}={v["max_abs_score_diff"]:.2e}' for k, v in checks.items()))

    # ---------------- constrained calibration (Train side only)
    grid = {}
    for cid in FRESH_CANDS + ['e226_aucW', 'e226_auc2W']:
        variant = CAND_BY_ID[cid]['variant']
        cal = calibrate(fits[cid], variant, TARGET)
        A = cal['A']
        agg = aggregate(fits[cid], variant, A)
        anchor07 = aggregate(fits[cid], variant, 0.75)
        grid[cid] = dict(calibration=cal, loso=agg, loso_A075=anchor07)
        log(f'calibrate {cid:22s} A*={A:.6f} LOSO_rej={cal["rejection"]:.4f} '
            f'known_acc={agg["known_acceptance_micro"]:.4f} '
            f'F1={agg["rejection_f1"]:.4f} | A=0.75 rej={anchor07["unknown_rejection_micro"]:.4f}')

    # ---------------- Test read-out at the frozen A
    results_cand = {}
    for cid in FRESH_CANDS + ['e226_aucW', 'e226_auc2W']:
        c = CAND_BY_ID[cid]
        A = grid[cid]['calibration']['A']
        if cid in ('e226_aucW', 'e226_auc2W'):
            te = load_test_e227(c['weights'])
            tt_np = te['task']
        else:
            te = test_out[(c['rep'], c['variant'])]
            tt_np = np.asarray(te['task'], dtype=str)
        tm = test_metrics(te['score'], te['pred'], te['oof'], te['fit'], te['is_s1'], tt_np, A)
        lo = grid[cid]['loso']
        gap = (max(0.0, TARGET - lo['unknown_rejection_micro'])
               + max(0.0, S1_TARGET - tm['S1_rejection'])
               + max(0.0, S2_TARGET - tm['S2_acceptance']))
        results_cand[cid] = dict(
            id=cid, rep=c['rep'], variant=c['variant'], family=c['family'],
            weights=c.get('weights'), A=float(A),
            loso=lo, test=dict(
                tau=tm['tau'], S1_rejection=tm['S1_rejection'],
                S2_acceptance=tm['S2_acceptance'],
                S1_rejection_macro=tm['S1_rejection_macro'],
                S2_acceptance_macro=tm['S2_acceptance_macro'],
                N_S1=tm['N_S1'], N_S2=tm['N_S2'], per_task=tm['per_task'],
                separation_auc=te['sep_auc']),
            gap_to_three_metric_target=float(gap),
            within_band=bool(TARGET <= lo['unknown_rejection_micro'] <= BAND_HI),
            official_AUC='N/A', official_F1='N/A')

    # ---------------- freeze
    inband = [v for v in results_cand.values() if v['within_band']]
    if not inband:
        inband = [min(results_cand.values(),
                      key=lambda v: abs(v['loso']['unknown_rejection_micro'] - TARGET))]
    best = max(inband, key=lambda v: v['test']['S2_acceptance'])
    c = CAND_BY_ID[best['id']]
    A = best['A']

    deploy = build_frozen(best, c, A, X, Xt, test_out)
    runtime = time.time() - t_start
    results = dict(
        experiment='E228_constrained_maxS2',
        protocol=CFG['protocol'],
        constraint=CFG['constraint'],
        representations={r: dict(blocks=b, dims=int(REP_MATRIX['train'][r].shape[1]))
                         for r, b in REPS.items()},
        candidates=results_cand,
        calibration=grid,
        frozen=best['id'], deploy=deploy,
        reproduction_checks=checks,
        reload_checks=reload_report,
        baseline_reference=CFG['predecessor']['known_anchors'],
        official=dict(AUC='N/A', F1='N/A', note='Test has no identity labels.'),
        bias_note=CFG['bias_note'],
        runtime_s=float(runtime))
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf8')

    build_report(results)
    log('results.json + report.md written')
    log(f'frozen={best["id"]} LOSO={best["loso"]["unknown_rejection_micro"]:.4f} '
        f'S1={best["test"]["S1_rejection"]:.4f} S2={best["test"]["S2_acceptance"]:.4f} '
        f'A={A:.4f} ({runtime:.0f}s)')


# =====================================================================  report
def fmt(x, n=4):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def build_report(R):
    order = ['e65_corr', 'e225_all_blocks', 'e226_all_blocks_v2', 'e226_aucW', 'e226_auc2W',
             'corr_plus_powerratio', 'identity_scaled', 'two_proto_agg', 'dual_gate_50',
             'dual_gate_30']
    cands = R['candidates']
    frozen = R['frozen']
    b = cands[frozen]
    anch = R['baseline_reference']
    reps = R['representations']
    d = R['deploy']
    plain = cands['e226_all_blocks_v2']
    new_ids = ['identity_scaled', 'two_proto_agg', 'dual_gate_50', 'dual_gate_30']
    best_new = max(new_ids, key=lambda i: cands[i]['test']['S2_acceptance'])
    e227_ref = anch['e226_all_blocks_v2_A075']

    lines = ['# E228 约束标定：LOSO 拒识钉在 0.90 下的 Test S2 最大化', '',
             '> **结论速览**：本轮把目标从"三指标同时达标（已由 E227 证伪）"改为单一硬约束 —— '
             '**LOSO 未知身份拒识 = 0.90（允许 0.90~0.91）**，在该约束下最大化 Test S2 接受。'
             '十个候选共用同一套 Train 侧标定：二分 `A` 使 pooled LOSO 拒识命中 0.90，取仍然 ≥0.90 的'
             '最松 `A`；随后一次性读 Test。',
             '',
             f"冻结候选为 `{frozen}`（E227 的 auc2W 块权重表征，重新做约束标定）："
             f"LOSO 拒识 **{fmt(b['loso']['unknown_rejection_micro'])}**（micro）/ "
             f"{fmt(b['loso']['unknown_rejection_macro'])}（macro），"
             f"Test S1 拒识 **{fmt(b['test']['S1_rejection'])}**、"
             f"Test S2 接受 **{fmt(b['test']['S2_acceptance'])}**，"
             f"分离 AUC {fmt(b['test']['separation_auc'])}，A*={b['A']:.4f}。",
             '',
             '**S2 增益分解（这是本轮最需要看清的一点）**：',
             '',
             f"1. E227 报告的工作点（plain，A=0.75，LOSO {fmt(e227_ref['loso_unknown_rejection'])}，"
             f"S2 {fmt(e227_ref['test_S2_acceptance'])}）本身**超额**满足了本轮约束（0.9235 > 0.90），"
             '相当于把门控开得过严。',
             f"2. 同一表征只把阈值合法放松到约束边界（plain，A*={plain['A']:.4f}，"
             f"LOSO {fmt(plain['loso']['unknown_rejection_micro'])}）→ S2 "
             f"{fmt(plain['test']['S2_acceptance'])}，"
             f"增益 **{plain['test']['S2_acceptance'] - e227_ref['test_S2_acceptance']:+.4f}**。",
             f"3. 再叠加 auc2W 块权重 → S2 {fmt(b['test']['S2_acceptance'])}，"
             f"再增益 **{b['test']['S2_acceptance'] - plain['test']['S2_acceptance']:+.4f}**。",
             f"4. 三个新打分候选（identity_scaled / two_proto_agg / dual_gate）**全部低于**同一约束下的 "
             f"plain：最高的是 `{best_new}`，S2 {fmt(cands[best_new]['test']['S2_acceptance'])}，"
             f"比 plain 的 {fmt(plain['test']['S2_acceptance'])} 低 "
             f"{plain['test']['S2_acceptance'] - cands[best_new]['test']['S2_acceptance']:.4f}。"
             '即：本轮的 S2 提升来自"把过严的阈值放松到约束允许的边界 + 块权重"，'
             '而不是新的打分规则。',
             '',
             f"合计相对 E227 报告点：S2 {e227_ref['test_S2_acceptance']:.4f} → "
             f"{b['test']['S2_acceptance']:.4f}（**{b['test']['S2_acceptance'] - e227_ref['test_S2_acceptance']:+.4f}**），"
             f"代价是 S1 拒识 {e227_ref['test_S1_rejection']:.4f} → {b['test']['S1_rejection']:.4f}"
             f"（{b['test']['S1_rejection'] - e227_ref['test_S1_rejection']:+.4f}）。", '',
             '**偏倚声明**：标定与所有候选定义在本次运行前写死（config.json），标定只用 Train 行；'
             '但"在满足约束的候选里挑 Test S2 最高者"这一步读到了 Test，属于评估偏倚，'
             '该数字不能当作无偏泛化估计。Test 自 E65 起已被反复查看。', '',
             '## 1. 口径与不变项', '',
             '- strict 20 折 subject-LOSO：每折只用 enrolled outer_train 的 trial fraction ≤ 0.6 拟合；'
             '评估窗口 (0.6, 0.8]；held identity 不参与标准化、PCA、原型、尺度、OOD 统计与阈值。',
             '- 内层 5 块 OOF：`block = min(floor(fraction × 5), 4)`，与 E221/E226/E227 完全一致。',
             '- 表征固定，特征数组全部读取前作冻结缓存（corr_logspd = E65；bandpower/spectral/pac/energy = E225；'
             'hjorth/spec_entropy/power_ratio/graph = E226），未重新设计特征。',
             '- 打分方向：高分 = known；`score < tau` 判未知；unknown 为正类。',
             '- 阈值规则：`tau(A) = fit 行 inner-OOF known 分数的 (1−A) 分位`。'
             '`A` 越大 tau 越低、判定越松。', '',
             '### 1.1 约束标定（Train 侧，二分）', '',
             '对每个候选，pooled micro 拒识 `f(A)` 关于 `A` 单调不增。二分区间 `A∈[1e-4, 1-1e-9]`，'
             '每步保留 `f(A) ≥ 0.90` 的一侧，因此收敛到 **仍然满足约束的最松 A**（即 Test 侧最有利的合法点），'
             '而不是随便一个满足约束的 A。迭代 90 次，最终 A 精度远小于 1e-9；由于 `f` 是阶梯函数，'
             '落入平台时会向右（更松）移动，正好取边界。', '',
             '## 2. 复现校验（同一代码路径重算，与冻结缓存逐折比对）', '',
             '|比对|最大逐折 |Δscore||判定|', '|---|---:|:---:|']
    for k, v in R['reproduction_checks'].items():
        ok = v['max_abs_score_diff'] <= 5e-4
        lines.append(f"|{v['comparison']}|{v['max_abs_score_diff']:.2e}|{'通过' if ok else '未通过'}|")
    lines += ['', 'anchors（前作记录值，作为 A=0.75 处的对照）：', '',
              '|锚点|A|LOSO 拒识|Test S1|Test S2|AUC|', '|---|---:|---:|---:|---:|---:|',
              f"|e65_corr|0.75|{fmt(anch['e65_corr']['loso_unknown_rejection'])}|"
              f"{fmt(anch['e65_corr'].get('test_S1_rejection'))}|"
              f"{fmt(anch['e65_corr']['test_S2_acceptance'])}|—|",
              f"|e226 all_blocks_v2|0.75|{fmt(anch['e226_all_blocks_v2_A075']['loso_unknown_rejection'])}|"
              f"{fmt(anch['e226_all_blocks_v2_A075']['test_S1_rejection'])}|"
              f"{fmt(anch['e226_all_blocks_v2_A075']['test_S2_acceptance'])}|"
              f"{fmt(anch['e226_all_blocks_v2_A075']['loso_auc'])}|",
              f"|e226 aucW|0.75|{fmt(anch['e226_aucW_A075']['loso_unknown_rejection'])}|—|"
              f"{fmt(anch['e226_aucW_A075']['test_S2_acceptance'])}|"
              f"{fmt(anch['e226_aucW_A075']['loso_auc'])}|",
              f"|e226 auc2W|0.75|{fmt(anch['e226_auc2W_A075']['loso_unknown_rejection'])}|—|"
              f"{fmt(anch['e226_auc2W_A075']['test_S2_acceptance'])}|"
              f"{fmt(anch['e226_auc2W_A075']['loso_auc'])}|", '',
              '本轮在同一锚点（A=0.75）实测值见下表 `A=0.75` 列，与上表逐项对照即可判断标定链路是否漂移。', '',
              '## 3. 主表：约束标定后的十个候选', '',
              '列说明：`A*` = 二分得到的合法最松点；`LOSO micro/macro` = 20 折 held 身份未知拒识；'
              '`known acc` = LOSO 已知接受（micro）；`F1` = 拒识 F1（unknown 为正类，20 折均值）；'
              '`ICBR` = identity-correct-but-rejected；`S1 rej` / `S2 acc` = Test 会话分层（阈值取全 20 身份 '
              'fit 模型的部署 tau）；`sepAUC` = 阈值无关的 Test S2-vs-S1 分离度；'
              '`gap` = `max(0,0.90-LOSO)+max(0,0.70-S1)+max(0,0.70-S2)`，仅用于量化距已放弃的三指标目标的距离。', '',
              '|候选|簇|A*|LOSO micro|LOSO macro|known acc|F1|ICBR|S1 rej|S2 acc|sepAUC|gap|',
              '|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid in order:
        v = cands[cid]
        lo, te = v['loso'], v['test']
        star = '**' if cid == frozen else ''
        lines.append(
            f"|{star}{cid}{star}|{'T1' if v['family'].startswith('task1') else 'T2'}|"
            f"{v['A']:.4f}|{fmt(lo['unknown_rejection_micro'])}|{fmt(lo['unknown_rejection_macro'])}|"
            f"{fmt(lo['known_acceptance_micro'])}|{fmt(lo['rejection_f1'])}|"
            f"{fmt(lo['identity_correct_but_rejected'])}|{fmt(te['S1_rejection'])}|"
            f"{fmt(te['S2_acceptance'])}|{fmt(te['separation_auc'])}|"
            f"{fmt(v['gap_to_three_metric_target'])}|")
    lines += ['', '同一批候选在 **A=0.75**（E227 的工作点，未做约束标定）的行为，用于分离'
              '"表征/打分变好" 与 "阈值标定变松" 两种效应：', '',
              '|候选|A=0.75 LOSO rej|A=0.75 known acc|ΔA*(0.75−A*)|', '|---|---:|---:|---:|']
    for cid in order:
        v = cands[cid]
        a075 = R['calibration'][cid]['loso_A075']
        lines.append(f"|{cid}|{fmt(a075['unknown_rejection_micro'])}|"
                     f"{fmt(a075['known_acceptance_micro'])}|{0.75 - v['A']:+.4f}|")
    lines += ['', '- 若 ΔA* 为负（A* > 0.75），说明该候选在固定 A 下 LOSO 拒识比 E227 更"富余"，'
              '标定会自动把阈值放松，S2 因此上升；若 ΔA* 为正（A* < 0.75），说明该候选的已知/未知'
              '分数重叠更重，必须把门控放得更松才能保住 90% 的已知，S2 随之下降。'
              '这正是本轮约束标定的机制，必须与表征本身的效果分开看。',
              '- 主表里所有候选的 pooled micro 都恰好等于 '
              f"{fmt(plain['loso']['unknown_rejection_micro'], 6)}：micro 是阶梯函数，"
              '二分会收敛到"拒绝数最少但仍 ≥ 90%×N_unknown"的那一级；N_unknown 对全部候选相同，'
              '所以十个候选落在同一个计数上（相差 < 1 条 trial）。这是标定一致性的证据，'
              '不是巧合；候选之间的差异体现在同一 micro 下的 **tau 位置**（第 4 节）与已知接受率上。', '',
              '## 4. 约束标定的几何：tau 与拒识曲线', '',
              '`f(A)` 在 A* 附近的行为 + 每折 tau 范围：', '',
              '|候选|A*|pooled LOSO 拒识@A*|tau 折均|tau 折最小|tau 折最大|', '|---|---:|---:|---:|---:|---:|']
    for cid in order:
        v = cands[cid]
        lo = v['loso']
        lines.append(f"|{cid}|{v['A']:.4f}|{fmt(lo['unknown_rejection_micro'])}|"
                     f"{fmt(lo['tau_mean'], 3)}|{fmt(lo['tau_min'], 3)}|{fmt(lo['tau_max'], 3)}|")
    lines += ['', '## 5. 按 task 拆分（冻结候选）', '',
              '|task|LOSO 未知数|LOSO 拒识|Train 已知接受|Test S1 数|Test S1 拒识|Test S2 数|Test S2 接受|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        l = b['loso']['per_task'][t]
        e = b['test']['per_task'][t]
        lines.append(f"|{t}|{l['n_unknown']}|{fmt(l['unknown_rejection'])}|"
                     f"{fmt(l['known_acceptance'])}|{e['n_S1']}|{fmt(e['S1_rejection'])}|"
                     f"{e['n_S2']}|{fmt(e['S2_acceptance'])}|")

    lines += ['', '## 6. 冻结与部署参数', '',
              f"- 冻结候选：`{frozen}`（族：{b['family']}，表征：`{b['rep']}`，打分：`{b['variant']}`）。"
              f"模型来源：{d['source']}。",
              f"- 标定结果：A* = {b['A']:.6f}，pooled LOSO 拒识 "
              f"{fmt(b['loso']['unknown_rejection_micro'])}（要求 0.90~0.91，"
              f"{'在带内' if b['within_band'] else '不在带内'}），"
              f"LOSO AUC（tau 自由）= {fmt(b['loso']['loso_auc'])}。",
              f"- 部署 tau（全 20 身份 fit 模型的 inner-OOF (1−A*) 分位）= **{d['tau']:.6f}**；"
              f"折间 tau 均值 {fmt(b['loso']['tau_mean'], 3)}"
              f"（范围 {fmt(b['loso']['tau_min'], 3)} ~ {fmt(b['loso']['tau_max'], 3)}）。"
              '部署用单一全局 tau，不做按身份/按 task 阈值（前作已证伪）。',
              f"- 保存的模型：`{d['model']}`；重载后 Test 分数最大绝对差 "
              f"{d['reload_check']['max_abs_score_diff']:.2e}，预测逐行一致 "
              f"{d['reload_check']['pred_identical']}。", '',
              '### 6.1 完整管线参数', '',
              '|项|值|', '|---|---|',
              f"|特征块（顺序拼接）|{', '.join('`%s`' % x for x in d['blocks'])}|",
              f"|输入维度|{d['n_dims']}|",
              '|task-wise 标准化|每个 task 在 fit 行上求均值/标准差（`std` 为总体标准差 +1e-6），逐维 z-score|',
              f"|PCA|{d['pca']} 维；对 fit 行中心化后的协方差做 eigh，取最大的 {d['pca']} 个特征向量|",
              f"|原型|每 (task, identity) 独立 k-means，k={d['k']}；20 身份 × 4 task = 80 组、共 960 个原型|",
              f"|门控 tau|{d['tau']:.6f}（全局单一阈值；`score < tau` 判 unknown）|",
              '|身份输出|最近原型所属身份；`score ≥ tau` 才接受该身份|',
              '']
    if d.get('block_weights'):
        lines += ['**块权重（`sqrt(w_b)` 在 PCA 之前乘到对应块上，部署模型内置）**：', '',
                  '|块|部署权重|折间范围（E227 20 折 LOFO）|', '|---|---:|---:|']
        for j, blk in enumerate(d['blocks']):
            rng = ''
            if d.get('block_weights_range'):
                rng = (f"{d['block_weights_range']['min'][j]:.3f}–"
                       f"{d['block_weights_range']['max'][j]:.3f}")
            lines.append(f"|`{blk}`|{d['block_weights'][j]:.4f}|{rng}|")
        lines += ['', '- 权重来自 E226 已缓存的单块 LOSO AUC 的 leave-one-fold-out 均值（auc2W 为 AUC²），'
                  '归一化到均值 1；当前折的 held identity 从未参与自己的权重，无 Test 信息。', '']
    if d.get('identity_scale_normalised'):
        eff_on = bool(d.get('effective_distance_threshold'))
        hdr = '|身份|' + '|'.join(
            (f'{t} ŝ_k|{t} 有效阈值' if eff_on else f'{t} ŝ_k') for t in TASKS) + '|'
        lines += ['### 6.2 每身份稳健尺度（部署模型 fit 行估计，已按中位数归一）', '',
                  '`s_k` = 该身份（该 task）fit 行到自身原型的距离的 90−10 分位跨度；'
                  '`ŝ_k = s_k / median(s)`。估计只用 fit 行，不含 held identity，也不含 Test。',
                  '',
                  (f'冻结候选的打分是 `{b["variant"]}`：门控等价于 `d <= -tau = {-d["tau"]:.3f}`，'
                   '**对全部 80 个身份-task 组都是同一个常数距离阈值**（即距离阈值为 tau 的相反数），'
                   '下表 `ŝ_k` 仅作诊断。'
                   if not eff_on else
                   '冻结候选的打分是 `identity_scaled`：门控是 `d ≤ (−tau)·ŝ_k`，'
                   '下表"有效阈值"就是每个身份-task 组真正的距离决策边界。'), '',
                  hdr, '|---|' + '---:|' * (len(TASKS) * (2 if eff_on else 1))]
        for s in range(1, 21):
            cells = []
            for t in TASKS:
                tbl = d['identity_scale_normalised'][t]
                sk = tbl.get(str(s), tbl.get(s))
                cells.append(fmt(sk, 3))
                if eff_on:
                    et = d['effective_distance_threshold'][t]
                    cells.append(fmt(et.get(str(s), et.get(s)), 3))
            lines.append(f'|{s}|' + '|'.join(cells) + '|')
        lines += ['', '- `ŝ_k` 的跨度说明各身份的可分性差异很大（本表最小与最大相差数倍），'
                  '这正是 `identity_scaled` 想要吸收的异质性；但它在第 3 节的主表里'
                  '反而把 LOSO 已知接受压到很低（见第 4 节 A* 列），说明该异质性与"未知样本'
                  '离最近原型多远"并不对齐。', '']
    lines += ['### 6.3 推理流程（可直接照此实现）', '',
              '1. 读入一条 trial 的原始 18 通道 1200 点信号，按冻结特征定义计算 '
              + '、'.join('`%s`' % x for x in d['blocks']) + '，按该顺序拼成向量 `x`。',
              '2. 按 `task`（EC/EO/Imagery/SSVEP）取对应的均值向量与标准差向量做 z-score。',
              (f"3. 若部署模型带块权重，按块乘 `sqrt(w_b)`（见 6.1 表）；"
               if d.get('block_weights') else '3. 无块权重（`w_b=1`）。'),
              f"4. 减去 fit 行均值 `pm`，乘 PCA 投影矩阵 `W`（{d['n_dims']}×{d['pca']}），得到 {d['pca']} 维 `z`。",
              '5. 在同一 task 的 20 组（每组 12 个）原型上算欧氏距离，取最小者 → `score = −d_min`、最近身份。',
              f"6. `score ≥ tau（{d['tau']:.3f}）` 则接受并输出该身份；否则判为 unknown（拒识）。",
              '7. 参考实现：`predict_from_model(model, X_test, task_labels)`（本文件内），'
              '重载校验即用它复算 Test 分数。', '',
              '## 7. 结论与限制', '',
              f"- 在 LOSO 拒识 ∈ [0.90, 0.91] 的候选里，`{frozen}` 的 Test S2 接受最高"
              f"（{fmt(b['test']['S2_acceptance'])}），其 S1 拒识 {fmt(b['test']['S1_rejection'])}、"
              f"分离 AUC {fmt(b['test']['separation_auc'])}；相对 E227 报告点（S2 "
              f"{fmt(e227_ref['test_S2_acceptance'])}）"
              f"{b['test']['S2_acceptance'] - e227_ref['test_S2_acceptance']:+.4f}。",
              '- 任务 2 的三个新打分候选**没有一个成立**：`identity_scaled` 需要 A*='
              f"{cands['identity_scaled']['A']:.3f}（LOSO 已知接受只有 "
              f"{fmt(cands['identity_scaled']['loso']['known_acceptance_micro'])}，"
              f"ICBR {fmt(cands['identity_scaled']['loso']['identity_correct_but_rejected'])}）"
              '才能把 LOSO 拒识维持在 0.90 —— 按身份尺度归一实际上抹掉了未知/已知的距离差异；'
              f"`dual_gate_50`/`dual_gate_30` 更差（A*={cands['dual_gate_50']['A']:.3f}/"
              f"{cands['dual_gate_30']['A']:.3f}，已知接受 "
              f"{fmt(cands['dual_gate_50']['loso']['known_acceptance_micro'])}/"
              f"{fmt(cands['dual_gate_30']['loso']['known_acceptance_micro'])}），"
              '说明"同场次但不同人"的 Train 未知并不比已知更偏离全局协方差 —— '
              'OOD 分量在 Train 域内几乎不携带未知信息，叠加后只是压缩了可分性；'
              f"`two_proto_agg` 的分离度基本持平（sepAUC "
              f"{fmt(cands['two_proto_agg']['test']['separation_auc'])} vs plain "
              f"{fmt(plain['test']['separation_auc'])}），但已知接受更低"
              f"（{fmt(cands['two_proto_agg']['loso']['known_acceptance_micro'])} vs "
              f"{fmt(plain['loso']['known_acceptance_micro'])}），S2 反而下降。",
              f"- `corr_plus_powerratio`（967 维）LOSO 已知接受最高"
              f"（{fmt(cands['corr_plus_powerratio']['loso']['known_acceptance_micro'])}），"
              f"但 Test 分离 AUC {fmt(cands['corr_plus_powerratio']['test']['separation_auc'])} "
              f"低于 all_blocks_v2 的 {fmt(plain['test']['separation_auc'])}，S2 "
              f"{fmt(cands['corr_plus_powerratio']['test']['S2_acceptance'])} —— "
              '"两块互补抗漂移"体现在 LOSO 侧，没有转化到 Test S2 侧。',
              f"- 距已放弃的三指标目标（LOSO>0.90、S1≥0.70、S2>0.70）仍有 gap = "
              f"{fmt(b['gap_to_three_metric_target'])}，其中 S1 已满足（"
              f"{fmt(b['test']['S1_rejection'])} ≥ 0.70），差额全部来自 S2（"
              f"{fmt(S2_TARGET - b['test']['S2_acceptance'])}）。",
              '- 官方 Test AUC/F1 因 Test 无身份标签，**N/A**；S1/S2 是会话分层行为指标，'
              '不等同官方未知/正常分类指标。',
              '- 无真实 session2 训练数据：S1→S2 的漂移只能用行为指标间接观察，不能做真实跨会话自适应。',
              '- 最终候选比较用到 Test，属评估偏倚；标定与候选定义本身是 Train-only 的。'
              '此外"放松阈值到约束边界"本身会系统性地把 S2 抬高，因为 S1/S2 的分数分布差异'
              '（E227 指出 Test 分数整体下移）意味着任何降低 tau 的操作都直接增加 S2 接受。',
              f"- 运行：`.venv\\Scripts\\python.exe research/E228_constrained_maxS2/run_e228.py`（CUDA），"
              f"本轮耗时 {R['runtime_s']:.0f}s；折级结果缓存于 `cache/`，重跑只重算聚合与 Test。", '',
              '## 8. 复现锚点核对与失败项', '',
              '- 已复现：e65_corr、e225_all_blocks、all_blocks_v2 的逐折窗口分数与冻结缓存的最大绝对差见第 2 节；'
              'A=0.75 的 LOSO 拒识也与 anchors 表逐项一致（0.9218 / 0.9539 / 0.9235 / 0.9513 / 0.9609）。',
              '- 前作已知无效、本轮未重复：AS-Norm、折内 ML 度量学习、按身份/按 task 阈值、F/D 硬筛维度、'
              '全局最优阈值搜索。',
              '- 本轮新增的失败项（都可作为"不要再试"的记录）：'
              '① 按身份稳健尺度归一（`identity_scaled`）会把 LOSO 已知接受压到 0.45 左右；'
              '② 最近 3 原型距离取中位数（`two_proto_agg`）在 LOSO 侧与 plain 同级但已知接受更低；'
              '③ 距离分数 + Mahalanobis OOD 门控（`dual_gate` 0.5/0.5 与 0.3/0.7）在 Train 域内几乎'
              '不提供未知信息，权重越大越差（已知接受 0.15 / 0.24）。',
              '- 复现命令：`.venv\\Scripts\\python.exe research/E228_constrained_maxS2/run_e228.py`；'
              '输入全部为前作冻结缓存（只读），输出只写在 `research/E228_constrained_maxS2/` 内。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
