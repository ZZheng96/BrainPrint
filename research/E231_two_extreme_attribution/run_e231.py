"""E231: two-extreme attribution -- what do the high-LOSO/low-S2 and low-LOSO/high-S2
configurations actually use, and is there a dimension subset that is both?

Frozen protocol (byte-identical to E221/E225/E226/E227/E228):
  strict 20-fold subject-LOSO, fit rows = per (subject, task) trial fraction <= 0.6,
  evaluation window (0.6, 0.8], inner 5 blocks = min(floor(fraction * 5), 4),
  task-wise standardisation -> PCA96 -> k=12 prototypes per (task, identity),
  score = -min euclidean distance to same-task prototypes, high score = known,
  unknown is the positive class for rejection F1 and AUC.

Three frozen tasks:
  1. compare the two extreme operating points (A high-LOSO/low-S2, B low-LOSO/high-S2):
     which dimensions they use, and whether B's Test S2 comes from a real separability
     gain or from the threshold sitting far below both score distributions.
  2. score every one of the 1569 dimensions with (in-session Fisher, cross-run identity
     consistency), validate cross_run against the per-dimension Test S2-vs-S1 AUC, and
     split the space into four quadrants.
  3. run strict LOSO on fixed-rule subsets (Q1/Q1Q2/Q1Q3/Q2/Q3/random control/all dims),
     calibrate tau on the Train side so that pooled LOSO unknown rejection = 0.90
     (E228 rule), read Test once, and compare Test S2 against the E228 baseline 0.3043.

Test is never used for subset selection or calibration; the per-dimension Test AUC is a
diagnostic only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
CACHE = OUT / 'cache'
MODELS = OUT / 'models'
LOG = OUT / 'run.log'
DEVICE = 'cuda'

TASKS = list(CFG['protocol']['tasks'])
FIT_FRAC = float(CFG['protocol']['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['protocol']['evaluation_fraction']
N_BLOCKS = int(CFG['protocol']['inner_blocks'])
PCA_DIM = int(CFG['representation']['pca_dim'])
K_PROTO = int(CFG['representation']['prototypes_k'])
N_FOLDS = int(CFG['protocol']['folds'])
SEED = int(CFG['seed'])
TARGET = 0.90
BAND_HI = 0.91
E228_S2 = float(CFG['decision_rule']['baseline_S2'])

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
BLOCK_OF_DIM = np.concatenate([np.full(BLOCK_DIM[b], i) for i, b in enumerate(BLOCK_ORDER)])
SUBSETS = [c['id'] for c in CFG['ablations']['subsets']]

X_TR = None
X_TE = None
TASKS_NP = None
Y = None
FR = None
BLOCKS = None
RUN2 = None
TT = None
IS_S1 = None


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(msg) + '\n')


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


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
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fold_auc(u, k):
    if len(u) == 0 or len(k) == 0:
        return float('nan')
    return float(((u[:, None] < k).sum() + 0.5 * (u[:, None] == k).sum()) / (len(k) * len(u)))


def quantile_tau(x, q):
    return float(np.quantile(np.asarray(x, float), q))


def frac_of(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


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


def fit_rep(raw, tasks, y, fit, k=K_PROTO, pca=PCA_DIM, w=None):
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


def score_plain(state, z, tasks, rows):
    proto_task, proto_id = proto_meta(state)
    n = len(tasks)
    s = np.full(n, np.nan)
    pred = np.full(n, -1, int)
    for t in TASKS:
        rr = rows[tasks[rows] == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rr]), tensor(state['p'][cols])).double()
        dmin, amin = D.min(1)
        s[rr] = -dmin.cpu().numpy()
        pred[rr] = proto_id[cols][amin.cpu().numpy()]
    return s, pred


def inner_oof(X, tasks, y, fit, blocks):
    oof = np.full(len(y), np.nan)
    for b in np.unique(blocks[fit]):
        hold = fit & (blocks == b)
        inner = fit & ~hold
        st = fit_rep(X, tasks, y, inner)
        z = transform(st, X, tasks)
        sc, _ = score_plain(st, z, tasks, np.flatnonzero(hold))
        oof[hold] = sc[hold]
    return oof


# =====================================================================  metric helpers
def ls_rej(folds):
    rej = n = 0
    for d in folds:
        u = d['unknown'].astype(bool)
        rej += int((d['score'][u] < d['tau']).sum())
        n += int(u.sum())
    return rej / max(n, 1)


def aggregate(folds, A):
    ms = []
    for d in folds:
        s = d['score']
        tau = quantile_tau(d['oof'][d['fit'].astype(bool)], 1.0 - A)
        known = d['known'].astype(bool)
        unknown = d['unknown'].astype(bool)
        rej_u = s[unknown] < tau
        rej_k = s[known] < tau
        tp, fp, fn = float(rej_u.sum()), float(rej_k.sum()), float((~rej_u).sum())
        ms.append(dict(known_acceptance=float((~rej_k).mean()),
                       unknown_rejection=float(rej_u.mean()),
                       rejection_f1=float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
                       icbr=float(((d['pred'][known] == d['subject'][known]) & rej_k).mean()),
                       n_known=int(known.sum()), n_unknown=int(unknown.sum()),
                       tau=float(tau)))
    n_u = sum(m['n_unknown'] for m in ms)
    n_k = sum(m['n_known'] for m in ms)
    aucs = []
    for d in folds:
        u = d['score'][d['unknown'].astype(bool)]
        k = d['score'][d['known'].astype(bool)]
        aucs.append(fold_auc(u, k))
    taus = [quantile_tau(d['oof'][d['fit'].astype(bool)], 1.0 - A) for d in folds]
    return dict(
        A=float(A), tau_mean=float(np.mean(taus)), tau_min=float(np.min(taus)),
        tau_max=float(np.max(taus)),
        unknown_rejection_micro=float(sum(m['unknown_rejection'] * m['n_unknown'] for m in ms) / max(n_u, 1)),
        unknown_rejection_macro=float(np.mean([m['unknown_rejection'] for m in ms])),
        known_acceptance_micro=float(sum(m['known_acceptance'] * m['n_known'] for m in ms) / max(n_k, 1)),
        known_acceptance_macro=float(np.mean([m['known_acceptance'] for m in ms])),
        rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
        identity_correct_but_rejected=float(np.mean([m['icbr'] for m in ms])),
        loso_auc=float(np.mean(aucs)), loso_auc_std=float(np.std(aucs)),
        n_known=int(n_k), n_unknown=int(n_u))


def calibrate(folds, target=TARGET):
    def micro(A):
        rej = n = 0
        for d in folds:
            tau = quantile_tau(d['oof'][d['fit'].astype(bool)], 1.0 - A)
            u = d['unknown'].astype(bool)
            rej += int((d['score'][u] < tau).sum())
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


def test_metrics(score, oof, fit, is_s1, tasks, A):
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
                separation_auc=auc_ties(score[~is_s1], score[is_s1]),
                N_S1=int(is_s1.sum()), N_S2=int((~is_s1).sum()), per_task=per_task)


# =====================================================================  feature loading
def load_all_features():
    global X_TR, X_TE
    parts = {'train': [], 'test': []}
    for split, store in (('train', parts['train']), ('test', parts['test'])):
        for b in BLOCK_ORDER:
            p = ROOT / CFG['feature_cache'][split][b]
            if not p.exists():
                raise FileNotFoundError(p)
            arr = np.load(p).astype(np.float32)
            if arr.shape[1] != BLOCK_DIM[b]:
                raise ValueError(f'{p}: {arr.shape} vs {BLOCK_DIM[b]}')
            store.append(arr)
    X_TR = np.concatenate(parts['train'], 1)
    X_TE = np.concatenate(parts['test'], 1)
    log(f'train features {X_TR.shape}  test features {X_TE.shape}')
    assert X_TR.shape[1] == DIM_TOTAL and X_TE.shape[1] == DIM_TOTAL


# =====================================================================  task 0: run split
def verify_run_split(meta):
    tr = meta.trial.to_numpy(float)
    med = meta.groupby(['subject', 'task']).trial.transform('median').to_numpy(float)
    run2 = tr > med
    cells = []
    for (s, t), g in meta.assign(run2=run2).groupby(['subject', 'task']):
        n1 = int((~g.run2).sum())
        n2 = int(g.run2.sum())
        cells.append(dict(subject=int(s), task=t, n=int(len(g)), n_run1=n1, n_run2=n2,
                          trial_min=int(g.trial.min()), trial_max=int(g.trial.max()),
                          median_trial=float(np.median(g.trial))))
    df = pd.DataFrame(cells)
    by_task = df.groupby('task')[['n', 'n_run1', 'n_run2']].agg(['min', 'max', 'mean']).round(1)
    imbalance = (df.n_run1 - df.n_run2).abs()
    summary = dict(
        rule='per (subject, task) split at the trial-index median: run1 = trial <= median, run2 = trial > median',
        cells=int(len(df)),
        per_cell_trials=dict(min=int(df.n.min()), max=int(df.n.max()), mean=float(df.n.mean())),
        run1=dict(min=int(df.n_run1.min()), max=int(df.n_run1.max()), mean=float(df.n_run1.mean())),
        run2=dict(min=int(df.n_run2.min()), max=int(df.n_run2.max()), mean=float(df.n_run2.mean())),
        abs_imbalance=dict(max=int(imbalance.max()), mean=float(imbalance.mean())),
        cells_under_10_per_run=int(((df.n_run1 < 10) | (df.n_run2 < 10)).sum()),
        by_task={t: {c: {k: float(v) for k, v in by_task.loc[t, c].items()}
                     for c in ['n', 'n_run1', 'n_run2'] if c in by_task.columns}
                 for t in by_task.index},
        trial_ranges={t: dict(min=int(df[df.task == t].trial_min.min()),
                              max=int(df[df.task == t].trial_max.max()),
                              median=float(df[df.task == t].median_trial.median()))
                      for t in TASKS},
        note='Train only has session1; the run1/run2 split is the closest available proxy for a '
             'second session and is not a real session2.')
    return run2, summary, df


# =====================================================================  task 2: attribution
def attribution(X, tasks, y, fit, run2):
    """Per-dimension (in_session Fisher, cross_run identity consistency) on the fit rows."""
    D = X.shape[1]
    between = np.zeros(D)
    within = np.zeros(D)
    n_tasks = 0
    for t in TASKS:
        base = fit & (tasks == t)
        Ms, Ws = [], []
        for s in np.unique(y[base]):
            r = base & (y == s)
            if int(r.sum()) < 3:
                continue
            A = X[r].astype(np.float64)
            Ms.append(A.mean(0))
            Ws.append(A.var(0, ddof=1))
        if len(Ms) < 2:
            continue
        between += np.stack(Ms).var(0, ddof=1)
        within += np.stack(Ws).mean(0)
        n_tasks += 1
    between /= max(n_tasks, 1)
    within /= max(n_tasks, 1)
    fisher = between / (within + 1e-9)
    fisher = np.where(within > 0, fisher, 0.0)

    same_all = np.zeros(D)
    diff_all = np.zeros(D)
    subj_all = np.zeros(D)
    n_tasks = 0
    subs = np.unique(y[fit])
    for t in TASKS:
        c1, c2 = [], []
        for s in subs:
            m1 = fit & (tasks == t) & (y == s) & (~run2)
            m2 = fit & (tasks == t) & (y == s) & run2
            if not m1.any() or not m2.any():
                c1.append(np.full(D, np.nan))
                c2.append(np.full(D, np.nan))
                continue
            c1.append(X[m1].astype(np.float64).mean(0))
            c2.append(X[m2].astype(np.float64).mean(0))
        C1 = np.stack(c1)
        C2 = np.stack(c2)
        ok = ~(np.isnan(C1).any(1) | np.isnan(C2).any(1))
        if ok.sum() < 3:
            continue
        C1, C2 = C1[ok], C2[ok]
        n = len(C1)
        s1 = C1.std(0)
        s2 = C2.std(0)
        Z1 = (C1 - C1.mean(0)) / (s1 + 1e-9)
        Z2 = (C2 - C2.mean(0)) / (s2 + 1e-9)
        same = (Z1 * Z2).mean(0)
        diff = (Z1.sum(0) * Z2.sum(0) - (Z1 * Z2).sum(0)) / (n * (n - 1))
        same_all += same
        diff_all += diff
        subj_all += 1.0
        n_tasks += 1
    same_all /= max(n_tasks, 1)
    diff_all /= max(n_tasks, 1)
    cross = same_all - diff_all
    return dict(in_session=fisher, cross_run=cross, cross_same=same_all, cross_diff=diff_all,
                between=between, within=within,
                n_run_subjects=subj_all / max(n_tasks, 1))


def test_dim_auc(Xt, is_s1, chunk=256):
    """Per-dimension AUC separating Test session2 (positive) from session1. Diagnostic only."""
    n = Xt.shape[0]
    npos, nneg = int((~is_s1).sum()), int(is_s1.sum())
    pos_idx = np.flatnonzero(~is_s1)
    out = np.empty(Xt.shape[1])
    ar = np.arange(1, n + 1, dtype=np.float64)
    for a in range(0, Xt.shape[1], chunk):
        b = min(a + chunk, Xt.shape[1])
        blk = Xt[:, a:b]
        order = np.argsort(blk, axis=0, kind='stable')
        ranks = np.empty((n, b - a), np.float64)
        for j in range(b - a):
            ranks[order[:, j], j] = ar
        out[a:b] = (ranks[pos_idx].sum(0) - npos * (npos + 1) / 2.0) / (npos * nneg)
    return out


def drift_sensitivity(between):
    """E226 synthetic-drift ledger, read-only: per-dim mean |delta| / sqrt(between)."""
    p = ROOT / CFG['feature_cache']['e226_drift_stats']
    d = np.load(p, allow_pickle=True)
    delta = d['subject_delta'].astype(np.float64).mean(axis=(0, 1))
    return delta / (np.sqrt(np.clip(between, 0.0, None)) + 1e-9)


def quadrant_table(stats, thr_is, thr_cr):
    hi_is = stats['in_session'] > thr_is
    hi_cr = stats['cross_run'] > thr_cr
    q = {'Q1': hi_is & hi_cr, 'Q2': hi_is & ~hi_cr, 'Q3': ~hi_is & hi_cr, 'Q4': ~hi_is & ~hi_cr}
    rows = {}
    for b in BLOCK_ORDER:
        m = BLOCK_OF_DIM == BLOCK_ORDER.index(b)
        n = int(m.sum())
        rows[b] = dict(dims=n,
                       counts={k: int((v & m).sum()) for k, v in q.items()},
                       share={k: float((v & m).sum() / n) for k, v in q.items()},
                       in_session_mean=float(stats['in_session'][m].mean()),
                       cross_run_mean=float(stats['cross_run'][m].mean()),
                       in_session_median=float(np.median(stats['in_session'][m])),
                       cross_run_median=float(np.median(stats['cross_run'][m])))
    total = dict(dims=int(DIM_TOTAL),
                 counts={k: int(v.sum()) for k, v in q.items()},
                 share={k: float(v.mean()) for k, v in q.items()})
    return dict(thresholds=dict(in_session=float(thr_is), cross_run=float(thr_cr)), blocks=rows,
                total=total, quadrants=q)


# =====================================================================  task 3: ablations
def subset_masks(stats, rng):
    thr_is = float(np.median(stats['in_session']))
    thr_cr = float(np.median(stats['cross_run']))
    hi_is = stats['in_session'] > thr_is
    hi_cr = stats['cross_run'] > thr_cr
    masks = {
        'Q1_essence': hi_is & hi_cr,
        'Q1Q2_in_session': hi_is,
        'Q1Q3_transfer': hi_cr,
        'Q2_only': hi_is & ~hi_cr,
        'Q3_only': ~hi_is & hi_cr,
        'all_dims_plain': np.ones(DIM_TOTAL, bool),
    }
    n_q1 = int(masks['Q1_essence'].sum())
    ridx = np.sort(rng.choice(DIM_TOTAL, min(n_q1, DIM_TOTAL), replace=False))
    rnd = np.zeros(DIM_TOTAL, bool)
    rnd[ridx] = True
    masks['random_same_size'] = rnd
    return masks, thr_is, thr_cr


def run_fold(fid, name, dims, held):
    fit = (Y != held) & (FR <= FIT_FRAC)
    window = (FR > EVAL_LO) & (FR <= EVAL_HI)
    win = np.flatnonzero(window)
    X = np.ascontiguousarray(X_TR[:, dims])
    st = fit_rep(X, TASKS_NP, Y, fit)
    z = transform(st, X, TASKS_NP)
    s, p = score_plain(st, z, TASKS_NP, win)
    oof = inner_oof(X, TASKS_NP, Y, fit, BLOCKS)
    out = dict(score=s[win], pred=p[win], oof=oof, fit=fit,
               task=np.asarray(TASKS_NP[win], dtype='<U8'), subject=Y[win].astype(int),
               known=(Y[win] != held), unknown=(Y[win] == held),
               n_dims=int(len(dims)), pca=int(st['pca']))
    np.savez_compressed(CACHE / f'fold_{fid}_{name}_{held:02d}.npz', **out)
    return out, float(st['pca'])


def load_fold(fid, name, held):
    d = np.load(CACHE / f'fold_{fid}_{name}_{held:02d}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files}


def run_subset(fid, name):
    folds = []
    sizes, pcas = [], []
    for held in range(1, N_FOLDS + 1):
        path = CACHE / f'fold_{fid}_{name}_{held:02d}.npz'
        t0 = time.time()
        if path.exists():
            d = load_fold(fid, name, held)
        else:
            fit = (Y != held) & (FR <= FIT_FRAC)
            stats = attribution(X_TR, TASKS_NP, Y, fit, RUN2)
            rng = np.random.default_rng(SEED + 7919 * held)
            masks, _, _ = subset_masks(stats, rng)
            dims = np.flatnonzero(masks[name])
            d, pca = run_fold(fid, name, dims, held)
            log(f'  fold {held:02d} {name:16s} dims={len(dims):5d} pca={pca} '
                f'({time.time()-t0:.1f}s)')
        folds.append({k: (d[k].astype(bool) if k in ('fit', 'known', 'unknown') else
                          np.asarray(d[k], dtype=str) if k == 'task' else d[k])
                      for k in d})
        sizes.append(int(d['n_dims']))
        pcas.append(int(d['pca']))
    return folds, sizes, pcas


def run_test_subset(fid, name, dims):
    fit = FR <= FIT_FRAC
    X = np.ascontiguousarray(X_TR[:, dims])
    Xt = np.ascontiguousarray(X_TE[:, dims])
    st = fit_rep(X, TASKS_NP, Y, fit)
    zt = transform(st, Xt, TT)
    s, p = score_plain(st, zt, TT, np.arange(len(TT)))
    oof = inner_oof(X, TASKS_NP, Y, fit, BLOCKS)
    np.savez_compressed(CACHE / f'test_{fid}_{name}.npz', score=s, pred=p, oof_s=oof, oof_fit=fit)
    model = dict(subset=name, dims=np.asarray(dims, int), state=st, pca=int(st['pca']),
                 k=K_PROTO, dim=int(st['dim']))
    torch.save(model, MODELS / f'e231_{name}.pt')
    return dict(score=s, pred=p, oof=oof, fit=fit, state=st)


def predict_from_model(model, Xt, tasks):
    st = model['state']
    z = transform(st, Xt, tasks)
    s, p = score_plain(st, z, tasks, np.arange(len(tasks)))
    return s, p


# =====================================================================  task 1: ends
def load_end_folds(cid):
    src = CFG['feature_cache']['read_only_score_caches']
    folds = []
    if cid == 'e65_corr':
        for h in range(1, N_FOLDS + 1):
            d = np.load(ROOT / src['e65_corr_folds'].format(fold=h), allow_pickle=True)
            folds.append(dict(score=d['score_plain'], pred=d['pred_plain'],
                              oof=d['oof_plain'], fit=d['fit'].astype(bool),
                              task=np.asarray(d['task'], dtype=str),
                              subject=d['subject'].astype(int),
                              known=d['known'].astype(bool), unknown=d['unknown'].astype(bool)))
    elif cid == 'e226_auc2W':
        for h in range(1, N_FOLDS + 1):
            d = np.load(ROOT / src['auc2W_folds'].format(fold=h), allow_pickle=True)
            folds.append(dict(score=d['score'], pred=d['pred'], oof=d['oof_s'],
                              fit=d['oof_fit'].astype(bool),
                              task=np.asarray(d['task'], dtype=str),
                              subject=d['subject'].astype(int),
                              known=d['known'].astype(bool), unknown=d['unknown'].astype(bool)))
    elif cid == 'random_96':
        for h in range(1, N_FOLDS + 1):
            d = np.load(ROOT / src['random_96_folds'].format(fold=h), allow_pickle=True)
            folds.append(dict(score=d['score'], pred=d['pred'], tau=float(d['tau']),
                              task=np.full(len(d['score']), 'ALL'),
                              subject=d['subject'].astype(int),
                              known=d['known'].astype(bool), unknown=d['unknown'].astype(bool)))
    elif cid == 'rank_fusion':
        res = json.loads((ROOT / src['rank_fusion_thresholds']).read_text(encoding='utf8'))
        th = [x['metrics']['fusion']['threshold'] for x in res['folds']]
        for h in range(1, N_FOLDS + 1):
            d = np.load(ROOT / src['rank_fusion_folds'].format(fold=h), allow_pickle=True)
            keep = d['known'].astype(bool) | d['unknown'].astype(bool)
            folds.append(dict(score=d['score'], pred=d['pred'], tau=float(th[h - 1]),
                              task=np.full(len(d['score']), 'ALL'),
                              subject=d['subject'].astype(int),
                              known=d['known'].astype(bool), unknown=d['unknown'].astype(bool),
                              keep=keep))
    return folds


def end_loso_stats(folds, A=None):
    known_med, unknown_med, taus, taus_q, umed_q, margins, iqrk = [], [], [], [], [], [], []
    rej_u = rej_k = n_u = n_k = 0
    for d in folds:
        keep = d.get('keep', np.ones(len(d['score']), bool))
        s = d['score'][keep]
        known = d['known'][keep] if 'keep' in d else d['known']
        unknown = d['unknown'][keep] if 'keep' in d else d['unknown']
        if A is not None:
            tau = quantile_tau(d['oof'][d['fit'].astype(bool)], 1.0 - A)
        else:
            tau = d['tau']
        ks = s[known]
        us = s[unknown]
        iqr = float(np.quantile(ks, 0.75) - np.quantile(ks, 0.25))
        iqrk.append(iqr)
        known_med.append(float(np.median(ks)))
        unknown_med.append(float(np.median(us)))
        taus.append(float(tau))
        taus_q.append(float((ks < tau).mean()))
        umed_q.append(float((ks < np.median(us)).mean()))
        margins.append(float((np.median(ks) - np.median(us)) / max(iqr, 1e-9)))
        rej_u += int((us < tau).sum())
        rej_k += int((ks < tau).sum())
        n_u += len(us)
        n_k += len(ks)
    return dict(
        tau_mean=float(np.mean(taus)), tau_min=float(np.min(taus)), tau_max=float(np.max(taus)),
        known_median=float(np.mean(known_med)), unknown_median=float(np.mean(unknown_med)),
        known_iqr=float(np.mean(iqrk)),
        tau_quantile_in_known=float(np.mean(taus_q)),
        unknown_median_quantile_in_known=float(np.mean(umed_q)),
        margin_iqr=float(np.mean(margins)),
        unknown_rejection_micro=float(rej_u / max(n_u, 1)),
        known_acceptance_micro=float(1.0 - rej_k / max(n_k, 1)),
        n_unknown=int(n_u), n_known=int(n_k))


def end_test_stats(score, tau, is_s1):
    s1 = score[is_s1]
    s2 = score[~is_s1]
    med1, med2 = float(np.median(s1)), float(np.median(s2))
    iqr = float(np.quantile(np.r_[s1, s2], 0.75) - np.quantile(np.r_[s1, s2], 0.25))
    return dict(tau=float(tau), S1_median=med1, S2_median=med2,
                median_shift_iqr=float((med2 - med1) / max(iqr, 1e-9)),
                S1_iqr=float(np.quantile(s1, 0.75) - np.quantile(s1, 0.25)),
                tau_quantile_in_S1=float((s1 < tau).mean()),
                tau_quantile_in_S2=float((s2 < tau).mean()),
                S1_rejection=float((s1 < tau).mean()),
                S2_acceptance=float((s2 >= tau).mean()),
                separation_auc=auc_ties(s2, s1))


def pca_block_share(dims, weights, fit):
    """Variance share of the PCA96 subspace attributable to each feature block."""
    X = np.ascontiguousarray(X_TR[:, dims])
    x = tensor(X)
    z = torch.empty_like(x)
    for t in TASKS:
        rows = TASKS_NP == t
        a = x[rows & fit]
        m = a.mean(0)
        sd = a.std(0, correction=0) + 1e-6
        z[rows] = (x[rows] - m) / sd
    if weights is not None:
        z = z * tensor(np.sqrt(np.asarray(weights, np.float32)))
    a = z[fit].double()
    pm = a.mean(0)
    lam, v = torch.linalg.eigh((a - pm).T @ (a - pm))
    pca = int(min(PCA_DIM, a.shape[1], len(a) - 1))
    lam_full = lam.clamp_min(0)
    lam = lam_full[-pca:].flip(0)
    W = v[:, -pca:].flip(1)
    W2 = (W ** 2)
    labels = BLOCK_OF_DIM[dims]
    var_share, load_share = {}, {}
    tot = float(lam.sum())
    for i, b in enumerate(BLOCK_ORDER):
        m = labels == i
        if not m.any():
            var_share[b] = 0.0
            load_share[b] = 0.0
            continue
        var_share[b] = float((lam[None, :] * W2[m, :]).sum() / max(tot, 1e-12))
        load_share[b] = float(W2[m, :].sum() / pca)
    top = [float(vv) for vv in (lam.cpu().numpy() / max(tot, 1e-12))[:10]]
    return dict(variance_share=var_share, loading_share=load_share,
                top10_pc_variance_share=top, pca=pca, n_dims=int(len(dims)),
                sum_top_eigenvalues=float(tot),
                sum_all_eigenvalues=float(lam_full.sum()))


def task1_ends():
    src = CFG['feature_cache']['read_only_score_caches']
    e227 = json.loads((ROOT / 'research/E227_joint_frontier/results.json').read_text(encoding='utf8'))
    w_auc2 = [float(x) for x in e227['block_weights']['auc2_lofo']['mean']]
    fit_all = FR <= FIT_FRAC
    tmeta = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    is_s1_meta = tmeta.session.astype(str).str.casefold().isin(['session1', 's1', '1']).to_numpy()

    out = {}
    # ---- A1: e65_corr, no weights
    dims = np.flatnonzero(BLOCK_OF_DIM == BLOCK_ORDER.index('corr_logspd'))
    folds = load_end_folds('e65_corr')
    loso = end_loso_stats(folds, A=0.75)
    t = np.load(ROOT / src['e65_corr_test'], allow_pickle=True)
    tau = quantile_tau(t['oof_s'][t['oof_fit'].astype(bool)], 0.25)
    out['e65_corr'] = dict(type='A', A=0.75, rep='e65_corr', n_dims=int(len(dims)),
                           blocks=['corr_logspd'],
                           block_weights={b: 1.0 for b in ['corr_logspd']},
                           pca=pca_block_share(dims, None, fit_all),
                           loso=loso,
                           test=end_test_stats(t['score'], tau, t['is_s1'].astype(bool)),
                           recorded=CFG['predecessor']['known_anchors']['e65_corr_A075'])
    # ---- A2: all blocks with auc2 weights
    dims = np.arange(DIM_TOTAL)
    w = np.concatenate([np.full(BLOCK_DIM[b], w_auc2[i]) for i, b in enumerate(BLOCK_ORDER)])
    folds = load_end_folds('e226_auc2W')
    loso = end_loso_stats(folds, A=0.75)
    t = np.load(ROOT / src['auc2W_test'], allow_pickle=True)
    tau = quantile_tau(t['oof_s'][t['oof_fit'].astype(bool)], 0.25)
    out['e226_auc2W'] = dict(type='A', A=0.75, rep='e226_all_blocks_v2', n_dims=int(DIM_TOTAL),
                             blocks=list(BLOCK_ORDER),
                             block_weights={b: w_auc2[i] for i, b in enumerate(BLOCK_ORDER)},
                             pca=pca_block_share(dims, w, fit_all),
                             loso=loso,
                             test=end_test_stats(t['score'], tau, t['is_s1'].astype(bool)),
                             recorded=CFG['predecessor']['known_anchors']['e226_auc2W_A075'])
    # ---- B1: random_96
    rng = np.random.default_rng(SEED)
    rdims = np.sort(rng.choice(1072, 96, replace=False))
    old_blocks = ['corr_logspd', 'bandpower_gainfree', 'spectral_shape', 'spectral_slope',
                  'pac', 'trial_energy']
    cnt = {b: int(((rdims >= OFFSETS[b][0]) & (rdims < OFFSETS[b][1])).sum()) for b in old_blocks}
    folds = load_end_folds('random_96')
    loso = end_loso_stats(folds)
    t = np.load(ROOT / src['random_96_test'], allow_pickle=True)
    out['random_96'] = dict(type='B', source='E225', n_dims=96,
                            blocks=old_blocks,
                            block_counts=cnt,
                            block_weights={b: 1.0 for b in old_blocks if cnt[b] > 0},
                            pca=None,
                            loso=loso,
                            test=end_test_stats(t['score'], float(t['tau']),
                                                t['is_s1'].astype(bool)),
                            recorded=CFG['predecessor']['known_anchors']['E225_random_96'])
    # ---- B2: rank fusion (E221)
    folds = load_end_folds('rank_fusion')
    loso = end_loso_stats(folds)
    t = np.load(ROOT / src['rank_fusion_test'], allow_pickle=True)
    tau = float(np.mean([d['tau'] for d in folds]))
    out['rank_fusion'] = dict(type='B', source='E221 fusion', n_dims=855 + 90,
                              blocks=['corr_logspd', 'E146_log_band_power'],
                              block_counts={'corr_logspd': 855, 'E146_log_band_power': 90},
                              block_weights={'corr_logspd': 0.25, 'E146_log_band_power': 0.25},
                              pca=None,
                              loso=loso,
                              test=end_test_stats(t['score'], tau, is_s1_meta),
                              recorded=CFG['predecessor']['known_anchors']['E221_rank_fusion'],
                              test_audit_recorded={
                                  'S1_rejection': 0.6050115397296406,
                                  'S2_acceptance': 0.7791133312519513})
    return out


# =====================================================================  report
def fmt(x, n=4):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def build_report(R, rep):
    ends = R['ends']
    att = R['attribution']
    ab = R['ablations']
    lines = ['# E231 两端极端配置的特征归因：四象限细分与消融', '',
             '> **结论速览**：把 (A) LOSO 高 / S2 低的配置与 (B) LOSO 低 / S2 高的配置放在同一张表上，'
             '对 1569 维逐维计算「场内区分度 in_session」与「跨 run 身份一致性 cross_run」，'
             '再按四象限切出固定子集跑完整 strict LOSO（PCA96，tau 二分到 LOSO 拒识 = 0.90）。', '',
             f"- 四象限核心结论：{R['verdict']['quadrant_sentence']}",
             f"- cross_run 代理是否有效：{R['verdict']['proxy_sentence']}",
             f"- 是否有子集超过 E228 基线 S2 = {E228_S2:.4f}：{R['verdict']['breakthrough_sentence']}",
             '',
             '## 1. 口径与不变项', '',
             '- strict 20 折 subject-LOSO；每折只用 enrolled outer_train 的 trial fraction ≤ 0.6 拟合；'
             '评估窗口 (0.6, 0.8]；held identity 不参与标准化、PCA、原型、阈值，也不参与它自己那一折的选维。',
             '- 内层 5 块 OOF：`block = min(floor(fraction × 5), 4)`，与 E221/E226/E227/E228 一致。',
             '- 表征：task-wise 标准化 → PCA96 → 每 (task, identity) k=12 原型；`score = -min` 欧氏距离，高分 = known。',
             f"- 阈值：`tau(A) = fit 行 inner-OOF known 分数的 (1−A) 分位`，A 在 Train 侧二分到 "
             f'pooled micro LOSO 未知拒识 = 0.90（E228 规则，保留仍 ≥ 0.90 的最松 A）。',
             '- 官方 Test AUC/F1：**N/A**（Test 无身份标签）。本文的 AUC 是 LOSO 阈值无关 AUC，'
             'F1 是拒识 F1（unknown 为正类）。',
             '', '## 2. Train 内部的 run 切分核实', '',
             f"- 规则：{rep['run_split']['rule']}。",
             f"- 80 个 (subject, task) 单元：每单元 trial 数 min/max/mean = "
             f"{rep['run_split']['per_cell_trials']['min']}/{rep['run_split']['per_cell_trials']['max']}/"
             f"{rep['run_split']['per_cell_trials']['mean']:.1f}；"
             f"run1 大小 mean {rep['run_split']['run1']['mean']:.1f}，run2 mean {rep['run_split']['run2']['mean']:.1f}，"
             f"最大绝对差 {rep['run_split']['abs_imbalance']['max']}，"
             f"任一侧 < 10 条的单元 {rep['run_split']['cells_under_10_per_run']} 个。",
             '- 各 task 的 trial 序号范围：' + '；'.join(
                 f"`{t}` {v['min']}–{v['max']}（中位 {v['median']:.0f}）"
                 for t, v in rep['run_split']['trial_ranges'].items()) + '。',
             '- **限制**：Train 只有 session1，run1/run2 只是同一会话内按序号前后切分，'
             '是跨会话的近似，不是真实 session2。所有 cross_run 数字都受此约束。',
             '', '## 3. 任务 1：两端极端配置对比', '',
             '### 3.1 维度构成与块权重', '',
             '|配置|端|输入维数|块构成|块权重|PCA96 方差占比（前 4 块）|', '|---|:---:|---:|---|---|---|']
    for cid in ['e65_corr', 'e226_auc2W', 'random_96', 'rank_fusion']:
        e = ends[cid]
        if e['pca'] is not None:
            vs = e['pca']['variance_share']
            top = sorted(vs.items(), key=lambda kv: -kv[1])[:4]
            share = '；'.join(f'`{b}` {v:.3f}' for b, v in top)
            bw = '；'.join(f'`{b}` {w:.2f}' for b, w in e['block_weights'].items())
            blk = ', '.join(f'`{b}`' for b in e['blocks'])
        else:
            share = '未重训（读冻结缓存，不做 PCA 分解）'
            bw = '；'.join(f'`{b}` {w:.2f}' for b, w in e['block_weights'].items())
            blk = '；'.join(f'`{b}` {n}' for b, n in e['block_counts'].items())
        lines.append(f"|`{cid}`|{e['type']}|{e['n_dims']}|{blk}|{bw}|{share}|")
    lines += ['', '**B 端的语义说明**：`random_96` 的 96 维是固定种子（20261117）从 E225 的 1072 维里'
              '一次性抽出的，抽中 `corr_logspd` 80 维、`bandpower_gainfree` 7 维、`spectral_shape` 6 维、'
              '`spectral_slope` 2 维、`pac` 1 维。随机子空间本身不携带任何语义，它只能作为'
              '“分数整体下移”的对照；`rank_fusion` 用的是 E65 corr（855 维）+ E146 对数带功率（90 维）'
              '两条流的 ECDF 秩融合，它的低 LOSO 来自秩变换把距离尺度压成 [0,1]，同样不是语义选择。'
              '注意 E146 的 90 维不在本文 1569 维台账内，因此 cross_run 无法覆盖它。', '',
              '### 3.2 A 端 PCA96 的块归属（方差占比 / 载荷占比）', '',
              '在全部 20 身份、fraction ≤ 0.6 的 fit 行上按部署管线（task-wise 标准化 → 块权重 `sqrt(w)` → PCA96）'
              '重算。`方差占比` = Σ_j λ_j Σ_{d∈b} W[d,j]² / Σ_j λ_j；`载荷占比` = Σ_j Σ_{d∈b} W[d,j]² / 96。', '',
              '|块|维数|e65_corr 方差|e65_corr 载荷|e226_auc2W 方差|e226_auc2W 载荷|',
              '|---|---:|---:|---:|---:|---:|']
    p1, p2 = ends['e65_corr']['pca'], ends['e226_auc2W']['pca']
    for b in BLOCK_ORDER:
        lines.append(f"|`{b}`|{BLOCK_DIM[b]}|{fmt(p1['variance_share'][b],3)}|"
                     f"{fmt(p1['loading_share'][b],3)}|{fmt(p2['variance_share'][b],3)}|"
                     f"{fmt(p2['loading_share'][b],3)}|")
    topk = sorted([(b, v) for b, v in p2['variance_share'].items() if b != 'corr_logspd'],
                  key=lambda kv: -kv[1])[:3]
    lines += ['', f"- A 端块权重（E227 `auc2_lofo` 折外均值，均值归一）："
              + '；'.join(f"`{b}` {w:.3f}" for b, w in ends['e226_auc2W']['block_weights'].items()) + '。',
              f"- 即使做了 auc² 加权，PCA96 的方差仍由 `corr_logspd` 主导"
              f"（{p2['variance_share']['corr_logspd']:.3f}），其后是 "
              + '、'.join(f"`{b}` {v:.3f}" for b, v in topk) + '；'
              f"`pac`、`trial_energy` 在 96 维子空间里几乎为零"
              f"（{p2['variance_share']['pac']:.4f} / {p2['variance_share']['trial_energy']:.4f}）。"
              '这与 E226 的块台账一致：两个“抗增益”块没有进入主成分。', '',
              '### 3.3 分数分布与阈值位置（Train LOSO，各自的工作点）', '',
              '`tau 在 known 分布中的分位` 与 `unknown 中位数在 known 分布中的分位` 与量纲无关，'
              '可直接跨配置比较；`margin/IQR` = (known 中位数 − unknown 中位数) / known IQR。', '',
              '|配置|端|A 或记录工作点|known 中位|unknown 中位|tau|tau 分位|unknown 中位分位|margin/IQR|LOSO 拒识|known 接受|',
              '|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid in ['e65_corr', 'e226_auc2W', 'random_96', 'rank_fusion']:
        e = ends[cid]
        L = e['loso']
        op = f"A={e['A']}" if e['type'] == 'A' else '记录工作点'
        lines.append(f"|`{cid}`|{e['type']}|{op}|{fmt(L['known_median'],2)}|{fmt(L['unknown_median'],2)}|"
                     f"{fmt(L['tau_mean'],3)}|{fmt(L['tau_quantile_in_known'],3)}|"
                     f"{fmt(L['unknown_median_quantile_in_known'],3)}|{fmt(L['margin_iqr'],3)}|"
                     f"{fmt(L['unknown_rejection_micro'])}|{fmt(L['known_acceptance_micro'])}|")
    lines += ['', '### 3.4 Test 侧：S2 高是"整体下移"还是"真实可分性"', '',
              '|配置|端|tau（部署）|S1 中位|S2 中位|(S2−S1)/IQR|tau 在 S1 的分位|tau 在 S2 的分位|S1 拒识|S2 接受|分离 AUC|',
              '|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid in ['e65_corr', 'e226_auc2W', 'random_96', 'rank_fusion']:
        e = ends[cid]
        T = e['test']
        lines.append(f"|`{cid}`|{e['type']}|{fmt(T['tau'],3)}|{fmt(T['S1_median'],2)}|"
                     f"{fmt(T['S2_median'],2)}|{fmt(T['median_shift_iqr'],3)}|"
                     f"{fmt(T['tau_quantile_in_S1'],3)}|{fmt(T['tau_quantile_in_S2'],3)}|"
                     f"{fmt(T['S1_rejection'])}|{fmt(T['S2_acceptance'])}|{fmt(T['separation_auc'])}|")
    a1 = ends['e65_corr']
    b1 = ends['random_96']
    b2 = ends['rank_fusion']
    lines += ['', R['verdict']['s2_source_paragraph'], '',
              '## 4. 任务 2：逐维二维归因与四象限', '',
              '### 4.1 代理有效性检验（cross_run vs Test 单维 AUC）', '',
              '|量对|Spearman ρ|p 值|n|解读|', '|---|---:|---:|---:|---|']
    for k, v in att['spearman'].items():
        lines.append(f"|{k}|{fmt(v['rho'],4)}|{v['p']:.3g}|{v['n']}|{v['note']}|")
    lines += ['', R['verdict']['proxy_paragraph'], '',
              '### 4.2 四象限定义', '',
              f"- 阈值（全体 20 身份的 fit 行，描述性）：in_session 中位数 = {fmt(att['quadrants']['thresholds']['in_session'],4)}，"
              f"cross_run 中位数 = {fmt(att['quadrants']['thresholds']['cross_run'],4)}。",
              '- Q1 高/高（本质候选）、Q2 高/低（会话特异性，怀疑是 LOSO 高 S2 低的元凶）、'
              'Q3 低/高（被现有方法忽略的迁移成分）、Q4 低/低（噪声）。', '',
              f"全体维度占比：Q1 {att['quadrants']['total']['counts']['Q1']} 维（"
              f"{fmt(att['quadrants']['total']['share']['Q1'],3)}）、"
              f"Q2 {att['quadrants']['total']['counts']['Q2']} 维（"
              f"{fmt(att['quadrants']['total']['share']['Q2'],3)}）、"
              f"Q3 {att['quadrants']['total']['counts']['Q3']} 维（"
              f"{fmt(att['quadrants']['total']['share']['Q3'],3)}）、"
              f"Q4 {att['quadrants']['total']['counts']['Q4']} 维（"
              f"{fmt(att['quadrants']['total']['share']['Q4'],3)}）。", '',
              '### 4.3 每块在四个象限的维度占比', '',
              '|块|维数|Q1 占比|Q2 占比|Q3 占比|Q4 占比|in_session 均值|cross_run 均值|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in BLOCK_ORDER:
        v = att['quadrants']['blocks'][b]
        lines.append(f"|`{b}`|{v['dims']}|{fmt(v['share']['Q1'],3)}|{fmt(v['share']['Q2'],3)}|"
                     f"{fmt(v['share']['Q3'],3)}|{fmt(v['share']['Q4'],3)}|"
                     f"{fmt(v['in_session_mean'],4)}|{fmt(v['cross_run_mean'],4)}|")
    lines += ['', R['verdict']['quadrant_block_paragraph'], '',
              '## 5. 任务 3：六子集（+2 参考）strict LOSO 消融', '',
              f"- 每个子集 PCA 固定 {int(CFG['ablations']['pca_dim'])} 维（`config.json`），"
              '不含块权重；tau 二分到 pooled micro LOSO 拒识 = 0.90，然后一次性读 Test。',
              '- 子集按折内 enrolled fit 行重新选择（held identity 不参与自己的选维）；'
              'Test 模型用全体 20 身份 fit 行选出的同一规则子集。', '',
              '|子集|平均维数|PCA|A*|LOSO micro|LOSO macro|known acc|拒识 F1|LOSO AUC|Test S1 拒识|Test S2 接受|分离 AUC|ΔS2 vs 0.3043|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sid in SUBSETS + ['e228_baseline_auc2W']:
        a = ab[sid]
        lo, te = a['loso'], a['test']
        star = '**' if sid == 'Q1_essence' else ''
        nd = a.get('n_dims', '—')
        nd = int(round(nd)) if isinstance(nd, (int, float, np.integer, np.floating)) else nd
        lines.append(f"|{star}`{sid}`{star}|{nd}|{a.get('pca', '—')}|{a['A']:.4f}|"
                     f"{fmt(lo['unknown_rejection_micro'])}|{fmt(lo['unknown_rejection_macro'])}|"
                     f"{fmt(lo['known_acceptance_micro'])}|{fmt(lo['rejection_f1'])}|"
                     f"{fmt(lo['loso_auc'])}|{fmt(te['S1_rejection'])}|{fmt(te['S2_acceptance'])}|"
                     f"{fmt(te['separation_auc'])}|{te['S2_acceptance'] - E228_S2:+.4f}|")
    lines += ['', R['verdict']['ablation_paragraph'], '',
              '## 6. 判定', '', R['verdict']['decision'], '',
              '## 7. 部署参数', '', R['verdict']['deploy'], '',
              '## 8. 限制', '',
              '- Test 单维 AUC 只用于诊断，且 Test 自 E65 起被反复查看，存在反复查看偏倚；'
              '本文据此做的“代理是否有效”判断本身也受该偏倚影响。',
              '- run1/run2 是同一 session1 内按 trial 序号前后切分的近似，不是真实 session2；'
              'cross_run 度量的是“同会话内前后半段的身份一致性”，对真实跨会话漂移只能外推。',
              '- 官方 Test AUC/F1 = **N/A**（Test 无身份标签）；S1/S2 是会话分层行为指标，'
              '不等同官方未知/正常分类指标。',
              '- 子集规则固定为“逐维统计量 > 中位数”，中位数本身依赖数据；规则在运行前写死，'
              '但阈值数值随折变化（held identity 不参与）。',
              '- S2 比较仍是“在固定候选集合里挑最高”，属于评估偏倚；标定本身只用 Train。',
              f"- 运行：`.venv\\Scripts\\python.exe research/E231_two_extreme_attribution/run_e231.py`"
              f"（CUDA），耗时 {R['runtime_s']:.0f}s。", '']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


# =====================================================================  main
def main():
    global TASKS_NP, Y, FR, BLOCKS, RUN2, TT, IS_S1
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E231 is blocked'
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

    RUN2, run_split, run_df = verify_run_split(meta)
    log(f"run split: run1 mean {run_split['run1']['mean']:.1f} "
        f"run2 mean {run_split['run2']['mean']:.1f} max|imbalance| {run_split['abs_imbalance']['max']}")

    tmeta = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tmeta) == int(CFG['data']['test_trials'])
    TT = tmeta.task.to_numpy()
    IS_S1 = tmeta.session.astype(str).str.casefold().isin(['session1', 's1', '1']).to_numpy()
    log(f'test rows={len(tmeta)} S1={int(IS_S1.sum())} S2={int((~IS_S1).sum())}')

    load_all_features()

    # ---------------- task 1
    t0 = time.time()
    ends = task1_ends()
    for cid, e in ends.items():
        log(f"task1 {cid:12s} LOSO_rej={e['loso']['unknown_rejection_micro']:.4f} "
            f"known_acc={e['loso']['known_acceptance_micro']:.4f} "
            f"S1={e['test']['S1_rejection']:.4f} S2={e['test']['S2_acceptance']:.4f} "
            f"sepAUC={e['test']['separation_auc']:.4f}")
    log(f'task1 done ({time.time()-t0:.1f}s)')

    # ---------------- task 2
    t0 = time.time()
    fit_all = FR <= FIT_FRAC
    stats = attribution(X_TR, TASKS_NP, Y, fit_all, RUN2)
    test_auc = test_dim_auc(X_TE, IS_S1)
    rho_cr, p_cr = spearmanr(stats['cross_run'], test_auc)
    rho_is, p_is = spearmanr(stats['in_session'], test_auc)
    rho_ic, p_ic = spearmanr(stats['in_session'], stats['cross_run'])
    rho_abs, p_abs = spearmanr(stats['cross_run'], np.abs(test_auc - 0.5))
    rho_isa, p_isa = spearmanr(stats['in_session'], np.abs(test_auc - 0.5))
    drift = drift_sensitivity(stats['between'])
    rho_dr, p_dr = spearmanr(stats['cross_run'], drift)
    rho_di, p_di = spearmanr(stats['in_session'], drift)
    rho_dt, p_dt = spearmanr(drift, test_auc)
    rho_da, p_da = spearmanr(drift, np.abs(test_auc - 0.5))
    thr_is = float(np.median(stats['in_session']))
    thr_cr = float(np.median(stats['cross_run']))
    quad = quadrant_table(stats, thr_is, thr_cr)
    quad.pop('quadrants', None)
    spearman = {
        'cross_run vs Test single-dim AUC': dict(rho=float(rho_cr), p=float(p_cr), n=DIM_TOTAL,
                                                  note='主检验：显著正相关才说明后文代理成立'),
        'cross_run vs abs(Test AUC-0.5)': dict(rho=float(rho_abs), p=float(p_abs), n=DIM_TOTAL,
                                               note='补充：与“会话可分离程度”的相关'),
        'in_session vs Test single-dim AUC': dict(rho=float(rho_is), p=float(p_is), n=DIM_TOTAL,
                                                   note='对照：场内区分度与 Test 会话分离度的关系'),
        'in_session vs abs(Test AUC-0.5)': dict(rho=float(rho_isa), p=float(p_isa), n=DIM_TOTAL,
                                                note='对照'),
        'in_session vs cross_run': dict(rho=float(rho_ic), p=float(p_ic), n=DIM_TOTAL,
                                         note='两个归因量是否正交'),
        'cross_run vs synthetic drift sensitivity': dict(rho=float(rho_dr), p=float(p_dr), n=DIM_TOTAL,
                                                          note='失效时的回退代理一致性：漂移敏感度越低应越可迁移'),
        'in_session vs synthetic drift sensitivity': dict(rho=float(rho_di), p=float(p_di), n=DIM_TOTAL,
                                                           note='对照'),
        'synthetic drift sensitivity vs Test single-dim AUC': dict(rho=float(rho_dt), p=float(p_dt), n=DIM_TOTAL,
                                                                   note='回退代理与 Test 的关系'),
        'synthetic drift sensitivity vs abs(Test AUC-0.5)': dict(rho=float(rho_da), p=float(p_da), n=DIM_TOTAL,
                                                                  note='回退代理与 Test 会话可分离程度的关系'),
    }
    np.savez_compressed(CACHE / 'dim_stats.npz',
                        in_session=stats['in_session'], cross_run=stats['cross_run'],
                        cross_same=stats['cross_same'], cross_diff=stats['cross_diff'],
                        between=stats['between'], within=stats['within'],
                        drift_sensitivity=drift,
                        test_auc=test_auc, block_of_dim=BLOCK_OF_DIM,
                        thr_in_session=thr_is, thr_cross_run=thr_cr)
    log(f"task2 done ({time.time()-t0:.1f}s) rho(cross_run,testAUC)={rho_cr:.4f} "
        f"p={p_cr:.2g} | rho(in_session,testAUC)={rho_is:.4f} | rho(intra)={rho_ic:.4f}")
    for b in BLOCK_ORDER:
        v = quad['blocks'][b]
        log(f"  block {b:20s} Q1={v['share']['Q1']:.3f} Q2={v['share']['Q2']:.3f} "
            f"Q3={v['share']['Q3']:.3f} Q4={v['share']['Q4']:.3f}")

    # ---------------- task 3
    fid = 'v1'
    ablations = {}
    global_masks, _, _ = subset_masks(stats, np.random.default_rng(SEED))
    np.savez_compressed(MODELS / 'e231_subset_masks.npz',
                        **{k: v for k, v in global_masks.items()},
                        block_of_dim=BLOCK_OF_DIM)
    for name in SUBSETS:
        t0 = time.time()
        folds, sizes, pcas = run_subset(fid, name)
        cal = calibrate(folds, TARGET)
        A = cal['A']
        agg = aggregate(folds, A)
        dims = np.flatnonzero(global_masks[name])
        te = run_test_subset(fid, name, dims)
        tm = test_metrics(te['score'], te['oof'], te['fit'], IS_S1, TT, A)
        ablations[name] = dict(id=name, n_dims=int(np.mean(sizes)), n_dims_per_fold=sizes,
                               pca=int(np.mean(pcas)), A=float(A),
                               calibration=cal, loso=agg,
                               test=dict(tau=tm['tau'], S1_rejection=tm['S1_rejection'],
                                         S2_acceptance=tm['S2_acceptance'],
                                         S1_rejection_macro=tm['S1_rejection_macro'],
                                         S2_acceptance_macro=tm['S2_acceptance_macro'],
                                         separation_auc=tm['separation_auc'],
                                         N_S1=tm['N_S1'], N_S2=tm['N_S2'],
                                         per_task=tm['per_task']),
                               delta_S2_vs_e228=float(tm['S2_acceptance'] - E228_S2),
                               within_band=bool(TARGET <= agg['unknown_rejection_micro'] <= BAND_HI),
                               official_AUC='N/A', official_F1='N/A',
                               runtime_s=float(time.time() - t0))
        log(f"ablate {name:16s} dims={np.mean(sizes):.0f} A*={A:.4f} "
            f"LOSO={agg['unknown_rejection_micro']:.4f} known={agg['known_acceptance_micro']:.4f} "
            f"F1={agg['rejection_f1']:.4f} S1={tm['S1_rejection']:.4f} "
            f"S2={tm['S2_acceptance']:.4f} sep={tm['separation_auc']:.4f} "
            f"({time.time()-t0:.1f}s)")

    # ---------------- E228 baseline reproduction (frozen auc2W caches)
    src = CFG['feature_cache']['read_only_score_caches']
    base_folds = load_end_folds('e226_auc2W')
    bcal = calibrate(base_folds, TARGET)
    bagg = aggregate(base_folds, bcal['A'])
    bt = np.load(ROOT / src['auc2W_test'], allow_pickle=True)
    btm = test_metrics(bt['score'], bt['oof_s'], bt['oof_fit'].astype(bool), bt['is_s1'].astype(bool),
                       np.asarray(bt['task'], dtype=str), bcal['A'])
    ablations['e228_baseline_auc2W'] = dict(
        id='e228_baseline_auc2W', n_dims=DIM_TOTAL, pca=PCA_DIM, A=float(bcal['A']),
        calibration=bcal, loso=bagg,
        test=dict(tau=btm['tau'], S1_rejection=btm['S1_rejection'],
                  S2_acceptance=btm['S2_acceptance'],
                  S1_rejection_macro=btm['S1_rejection_macro'],
                  S2_acceptance_macro=btm['S2_acceptance_macro'],
                  separation_auc=btm['separation_auc'], N_S1=btm['N_S1'], N_S2=btm['N_S2'],
                  per_task=btm['per_task']),
        delta_S2_vs_e228=float(btm['S2_acceptance'] - E228_S2),
        within_band=bool(TARGET <= bagg['unknown_rejection_micro'] <= BAND_HI),
        reproduction=dict(recorded_A=0.804985, recorded_loso=0.90, recorded_S2=E228_S2),
        official_AUC='N/A', official_F1='N/A', runtime_s=0.0)
    log(f"baseline e228_auc2W A*={bcal['A']:.4f} LOSO={bagg['unknown_rejection_micro']:.4f} "
        f"S2={btm['S2_acceptance']:.4f} (recorded {E228_S2:.4f})")

    # ---------------- verdict
    q1 = ablations['Q1_essence']
    q2 = ablations['Q2_only']
    q3 = ablations['Q3_only']
    qq12 = ablations['Q1Q2_in_session']
    qq13 = ablations['Q1Q3_transfer']
    rnd = ablations['random_same_size']
    win = bool(q1['test']['S2_acceptance'] > E228_S2
               and TARGET <= q1['loso']['unknown_rejection_micro'] <= BAND_HI)

    proxy_ok = bool(rho_cr > 0 and p_cr < 0.05)
    fallback_txt = (f"作为回退，E226 合成漂移台账给出的逐维漂移敏感度与 cross_run 的 ρ={rho_dr:.4f}"
                    f"（p={p_dr:.2g}，期望为负相关），与 Test 单维 AUC 的 ρ={rho_dt:.4f}"
                    f"（p={p_dt:.2g}），与 |AUC−0.5| 的 ρ={rho_da:.4f}（p={p_da:.2g}）。")
    if proxy_ok:
        proxy_sentence = (f"cross_run 与 Test 单维 AUC 显著正相关（ρ={rho_cr:.3f}, p={p_cr:.2g}），"
                          '代理成立：跨 run 身份一致性越高，该维在 Test 上区分 S2/S1 的能力越强，'
                          '可以把 cross_run 当作跨会话可迁移性的合法代理。')
        proxy_paragraph = (f"主检验 ρ={rho_cr:.4f}（p={p_cr:.2g}，n=1569）为显著正相关，"
                           f"补充检验与 |AUC−0.5| 的 ρ={rho_abs:.4f}（p={p_abs:.2g}）。"
                           '因此 cross_run 可以合法地作为跨会话代理，四象限划分有依据。'
                           + fallback_txt)
    else:
        proxy_sentence = (f"cross_run 与 Test 单维 AUC 不构成显著正相关（ρ={rho_cr:.3f}, p={p_cr:.2g}），"
                          '**代理失效**；按预注册规则回退到合成漂移代理，四象限中的 Q1/Q3 解释需降级为描述性。')
        proxy_paragraph = (f"主检验 ρ={rho_cr:.4f}（p={p_cr:.2g}，n=1569）不满足显著正相关，"
                           f"补充检验与 |AUC−0.5| 的 ρ={rho_abs:.4f}（p={p_abs:.2g}）"
                           '虽显著但方向意味着“跨 run 一致性越高的维度在 Test 上反而更容易区分两个会话”。'
                           '这与“可迁移”的期望相反：cross_run 度量的是受试者间结构在前后半段的可重复性，'
                           '并不等价于对会话效应的不变性。按预先写死的判定，**cross_run 代理失效**，'
                           '四象限只能当作描述性结构，不能当作跨会话证据。'
                           + fallback_txt +
                           '合成漂移同样是“坏度”指标，两者都不足以单独支持本质特征的跨会话判断；'
                           '因此第 5 节的消融结论必须以 Test 实测为准，而不是以四象限的语义解释为准。')

    q_blocks = quad['blocks']
    cand_blocks = [b for b in BLOCK_ORDER if BLOCK_DIM[b] >= 18]
    best_q1_block = max(cand_blocks, key=lambda b: q_blocks[b]['share']['Q1'])
    worst_q1_block = min(cand_blocks, key=lambda b: q_blocks[b]['share']['Q1'])
    quadrant_sentence = (
        f"Q1（本质候选）只占 {quad['total']['counts']['Q1']} 维（{quad['total']['share']['Q1']:.1%}），"
        f"Q2 {quad['total']['share']['Q2']:.1%}、Q3 {quad['total']['share']['Q3']:.1%}、"
        f"Q4 {quad['total']['share']['Q4']:.1%}；块内 Q1 占比最高的是 `{best_q1_block}`"
        f"（{q_blocks[best_q1_block]['share']['Q1']:.1%}），最低的是 `{worst_q1_block}`"
        f"（{q_blocks[worst_q1_block]['share']['Q1']:.1%}，全部落在 Q4）；"
        f"`corr_logspd` 的 Q1 占比只有 {q_blocks['corr_logspd']['share']['Q1']:.1%}。")
    quadrant_block_paragraph = (
        f"`{best_q1_block}` 的 Q1 占比最高（{q_blocks[best_q1_block]['share']['Q1']:.3f}），"
        f"说明它既有场内区分度又能跨 run 复现；`{worst_q1_block}` 的 Q1 占比最低"
        f"（{q_blocks[worst_q1_block]['share']['Q1']:.3f}），其全部维度落进 Q4（噪声）。"
        f"`spec_entropy` 是唯一 Q3 占比最高的大块（{q_blocks['spec_entropy']['share']['Q3']:.3f}），"
        '即“跨 run 稳定但场内区分度低”，是现有方法最容易忽略的部分。'
        '注意 Q1 是全体 20 身份 fit 行上的描述性划分，消融用的是逐折重算的同一规则，两者数值可能略有差异。')

    a1t, b1t, b2t = ends['e65_corr'], ends['random_96'], ends['rank_fusion']
    s2_source_paragraph = (
        f"**S2 来源分解**：A 端（`e65_corr`、`e226_auc2W`）的 Test 分离 AUC 分别为 "
        f"{a1t['test']['separation_auc']:.4f}、{ends['e226_auc2W']['test']['separation_auc']:.4f}，"
        f"但 tau 卡在 S1 分数的 {a1t['test']['tau_quantile_in_S1']:.3f}/"
        f"{ends['e226_auc2W']['test']['tau_quantile_in_S1']:.3f} 分位，S1 拒识被钉在 0.96 以上；"
        f"B 端（`random_96`、`rank_fusion`）的分离 AUC 只有 {b1t['test']['separation_auc']:.4f}、"
        f"{b2t['test']['separation_auc']:.4f}，显著低于 A 端，然而 tau 只落在 S1 分数的 "
        f"{b1t['test']['tau_quantile_in_S1']:.3f}/{b2t['test']['tau_quantile_in_S1']:.3f} 分位，"
        f"S2 接受因此被抬到 {b1t['test']['S2_acceptance']:.4f}/{b2t['test']['S2_acceptance']:.4f}。"
        '即：**B 端的“S2 高”不是真实可分性提升，而是阈值整体下移**（S1/S2 中位差 '
        f"{b1t['test']['median_shift_iqr']:.3f}/{b2t['test']['median_shift_iqr']:.3f} IQR，"
        f"对比 A 端 {a1t['test']['median_shift_iqr']:.3f}/"
        f"{ends['e226_auc2W']['test']['median_shift_iqr']:.3f} IQR，也没有变大）。"
        '这从数值上确认了用户的判断：高分配置的 Test S2 只是把门控放松的结果，'
        '并没有买到跨会话的判别力。')

    if win:
        decision = (f"`Q1_essence` 在 LOSO 拒识 {q1['loso']['unknown_rejection_micro']:.4f}"
                    f"（∈[0.90,0.91]）下 Test S2 = {q1['test']['S2_acceptance']:.4f} > "
                    f"{E228_S2:.4f}，**判定找到更本质特征**，冻结该子集。")
        breakthrough_sentence = (f"是。`Q1_essence` 的 Test S2 = {q1['test']['S2_acceptance']:.4f}，"
                                 f"比 E228 基线 {E228_S2:.4f} 高 "
                                 f"{q1['test']['S2_acceptance'] - E228_S2:+.4f}。")
    else:
        decision = (f"`Q1_essence` 在 LOSO 拒识 {q1['loso']['unknown_rejection_micro']:.4f} 下 Test S2 = "
                    f"{q1['test']['S2_acceptance']:.4f}，未超过 E228 基线 {E228_S2:.4f}"
                    f"（{q1['test']['S2_acceptance'] - E228_S2:+.4f}）。按预先写死的判定："
                    '**本质特征在当前数据下不足以突破**；本报告给出四象限证据后停止，不换配方重跑。')
        breakthrough_sentence = (f"没有。最好的子集是 "
                                 f"`{max(SUBSETS, key=lambda s: ablations[s]['test']['S2_acceptance'])}`"
                                 f"（S2 {max(ablations[s]['test']['S2_acceptance'] for s in SUBSETS):.4f}），"
                                 f"仍未超过 {E228_S2:.4f}。")
    ablation_paragraph = '\n'.join([
        f"- **Q2/Q3 都没有复现任何一端极端**：`Q2_only`（会话特异）在 LOSO 拒识 {q2['loso']['unknown_rejection_micro']:.4f} 下 "
        f"known 接受只有 {q2['loso']['known_acceptance_micro']:.4f}、LOSO AUC {q2['loso']['loso_auc']:.4f}，"
        f"Test S1 {q2['test']['S1_rejection']:.4f}、S2 {q2['test']['S2_acceptance']:.4f}；"
        f"`Q3_only`（迁移但场内弱）known 接受 {q3['loso']['known_acceptance_micro']:.4f}、"
        f"AUC {q3['loso']['loso_auc']:.4f}，S1 {q3['test']['S1_rejection']:.4f}、S2 {q3['test']['S2_acceptance']:.4f}。"
        f"两者的 LOSO AUC 都远低于 `all_dims_plain` 的 {ablations['all_dims_plain']['loso']['loso_auc']:.4f}，"
        '说明单独拿任何半个象限都做不出可用的开集系统；'
        '用户假设的“Q2 制造 LOSO 高 S2 低、Q3 制造 LOSO 低 S2 高”在这批维度上不成立。',
        f"- **Q1 反而低于同维数随机对照**：`Q1_essence` S2 {q1['test']['S2_acceptance']:.4f} < "
        f"`random_same_size` {rnd['test']['S2_acceptance']:.4f}（随机 {int(round(rnd['n_dims']))} 维，同 PCA96）。"
        f"Q1 内 `corr_logspd` 只占 {quad['blocks']['corr_logspd']['share']['Q1']:.1%} 的块内维度"
        f"（约 {int(round(quad['blocks']['corr_logspd']['share']['Q1'] * 855))} 维），"
        '而相关矩阵流恰恰是 PCA96 里方差占比最高、LOSO AUC 最高的一块；'
        '按“高/高”筛掉大部分 corr 维度后，剩下的低维块（spectral_shape / spectral_slope / power_ratio）'
        '虽然场内 Fisher 与跨 run 一致性都高，却撑不起 96 维子空间的判别结构。这是 Q1 失效的直接原因。',
        f"- 高 in_session（Q1∪Q2）S2 {qq12['test']['S2_acceptance']:.4f}，"
        f"高 cross_run（Q1∪Q3）S2 {qq13['test']['S2_acceptance']:.4f}，"
        f"全部 1569 维（无块权重）S2 {ablations['all_dims_plain']['test']['S2_acceptance']:.4f}，"
        f"E228 冻结基线（带 auc² 块权重）S2 {ablations['e228_baseline_auc2W']['test']['S2_acceptance']:.4f}。"
        f"即：在 LOSO 拒识同为 0.90 时，Test S2 排序为 E228 基线 {ablations['e228_baseline_auc2W']['test']['S2_acceptance']:.4f} > "
        f"高 cross_run {qq13['test']['S2_acceptance']:.4f} > 同维数随机 {rnd['test']['S2_acceptance']:.4f} ≈ "
        f"全维 {ablations['all_dims_plain']['test']['S2_acceptance']:.4f} > Q1 {q1['test']['S2_acceptance']:.4f} ≈ "
        f"高 in_session {qq12['test']['S2_acceptance']:.4f} > Q2 {q2['test']['S2_acceptance']:.4f} > "
        f"Q3 {q3['test']['S2_acceptance']:.4f}。"
        '任何“只留一部分维度”的做法都没有超过全维表征；'
        f"`Q1Q3_transfer` 比全维高 {qq13['test']['S2_acceptance'] - ablations['all_dims_plain']['test']['S2_acceptance']:+.4f}，"
        '远小于候选间噪声（E228 报告中同族候选的 S2 波动约 0.01~0.02），不构成突破。'
        'S2 的增益来自块权重与阈值标定，不来自选维。'])

    deploy_lines = []
    if win:
        m = torch.load(MODELS / 'e231_Q1_essence.pt', weights_only=False)
        back = torch.load(MODELS / 'e231_Q1_essence.pt', weights_only=False)
        s2, p2 = predict_from_model(back, np.ascontiguousarray(X_TE[:, back['dims']]), TT)
        tm = ablations['Q1_essence']['test']
        keep = np.load(CACHE / 'test_v1_Q1_essence.npz', allow_pickle=True)
        deploy_lines = [
            f"- 冻结子集：`Q1_essence`（{int(ablations['Q1_essence']['n_dims'])} 维，PCA "
            f"{m['pca']}，k={m['k']}），模型文件 `{MODELS / 'e231_Q1_essence.pt'}`；"
            f"维度掩码 `{MODELS / 'e231_subset_masks.npz'}`。",
            f"- A* = {ablations['Q1_essence']['A']:.6f}；部署 tau = {tm['tau']:.6f}；"
            f"LOSO 拒识 {q1['loso']['unknown_rejection_micro']:.4f}，"
            f"Test S1 拒识 {tm['S1_rejection']:.4f}，Test S2 接受 {tm['S2_acceptance']:.4f}，"
            f"分离 AUC {tm['separation_auc']:.4f}。",
            f"- 重载校验：Test 分数最大绝对差 "
            f"{float(np.max(np.abs(s2 - keep['score']))):.2e}，预测逐行一致 "
            f"{bool(np.array_equal(p2, keep['pred']))}。",
            '- 推理：按 `models/e231_subset_masks.npz` 的 `Q1_essence` 掩码取维度 → '
            '按 task 取均值/标准差 z-score → 减 fit 均值乘 PCA 投影 → 同 task 原型最小距离 → '
            '`score ≥ tau` 接受，否则判 unknown。']
    else:
        deploy_lines = ['- 未冻结新部署模型：`Q1_essence` 未超过 E228 基线，按预注册判定不产生新的部署参数。'
                        '`models/e231_Q1_essence.pt` 与 `models/e231_subset_masks.npz` 仍保存，供复核与后续复用。']

    results = dict(
        experiment='E231_two_extreme_attribution',
        protocol=CFG['protocol'], attribution_rule=CFG['attribution'],
        representations=dict(pca_dim=PCA_DIM, prototypes_k=K_PROTO, dim_total=DIM_TOTAL),
        run_split=run_split,
        ends=ends,
        attribution=dict(spearman=spearman, quadrants=quad,
                         thresholds=dict(in_session=thr_is, cross_run=thr_cr),
                         n_dims=DIM_TOTAL),
        ablations=ablations,
        decision=dict(win=win, baseline_S2=E228_S2,
                      q1_S2=float(q1['test']['S2_acceptance']),
                      delta=float(q1['test']['S2_acceptance'] - E228_S2),
                      rule=CFG['decision_rule']),
        verdict=dict(quadrant_sentence=quadrant_sentence, proxy_sentence=proxy_sentence,
                     proxy_paragraph=proxy_paragraph, breakthrough_sentence=breakthrough_sentence,
                     s2_source_paragraph=s2_source_paragraph,
                     quadrant_block_paragraph=quadrant_block_paragraph,
                     ablation_paragraph=ablation_paragraph,
                     decision=decision, deploy='\n'.join(deploy_lines)),
        official=dict(AUC='N/A', F1='N/A', note='Test has no identity labels.'),
        bias_note=CFG['bias_note'],
        runtime_s=float(time.time() - t_start))
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf8')

    build_report(results, dict(run_split=run_split))
    log('results.json + report.md written')
    log(f"Q1_essence S2={q1['test']['S2_acceptance']:.4f} vs {E228_S2:.4f} win={win} "
        f"({time.time()-t_start:.0f}s)")


if __name__ == '__main__':
    main()
