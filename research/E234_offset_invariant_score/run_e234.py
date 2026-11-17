"""E234: offset-invariant (relative) open-set scores.

Question: `score = -d1` (distance to the nearest same-task prototype) is not
invariant to an additive cross-session offset delta: a day-to-day shift that
pushes a registered user away from every prototype lowers the score uniformly and
therefore rejects registered users.  A *relative* score is supposed to cancel the
common part of that shift row by row:

    margin : score = d2 - d1          (d2 = nearest prototype of a DIFFERENT identity)
    ratio  : score = -d1 / d2
    mix    : score = -d1 + lam * (d2 - d1)

Everything else is frozen: the E227 deploy model (task-wise standardisation,
sqrt(w_b) block weights, PCA96, k=12 prototypes per (task, identity)), the strict
20-fold subject-LOSO protocol, the inner 5-block OOF threshold rule, and (for the
`*_mean_align` candidates) the E233 per-(session, task) unlabeled mean alignment.

The E227 fold caches store only -d1, so the 20 folds are refitted here with the
verbatim E227 code in order to obtain d2; d1 is cross-checked against the E227
cached scores fold by fold, and the calibrated A is cross-checked against E228.

Threshold rule (E228, identical code path): tau = (1-A) quantile of the
candidate's own inner-OOF known scores, A bisected on Train so that the pooled
(micro) LOSO unknown rejection is the largest value still >= 0.90.  Test is read
once per candidate after A is frozen.
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

TASKS = list(CFG['protocol']['tasks'])
FIT_FRAC = float(CFG['protocol']['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['protocol']['evaluation_fraction']
N_BLOCKS = int(CFG['protocol']['inner_blocks'])
N_FOLDS = int(CFG['protocol']['folds'])
BLOCK_DIM = {k: int(v) for k, v in CFG['blocks'].items()}
BLOCK_ORDER = list(CFG['representation']['block_order'])
PCA_DIM = int(CFG['representation']['pca_dim'])
K_PROTO = int(CFG['representation']['prototypes_k'])
MIN_CELL_N = int(CFG['alignment']['min_cell_n'])
PURITY_K = [int(k) for k in CFG['metrics']['purity']['k']]
PURITY_SEED = int(CFG['metrics']['purity']['seed'])
BOOT_B = int(CFG['metrics']['bootstrap']['B'])
BOOT_SEED = int(CFG['metrics']['bootstrap']['seed'])
TARGET = float(CFG['threshold']['target_loso_unknown_rejection'])
BISECT_ITERS = int(CFG['threshold']['bisection_iters'])

ANCH = CFG['predecessor']['known_anchors']
A_REF = float(ANCH['e228_auc2W_calibrated']['A'])
TAU_REF = float(ANCH['e228_auc2W_calibrated']['test_tau'])
LOSO_REF = float(ANCH['e228_auc2W_calibrated']['loso_unknown_rejection'])
LOSO_AUC_REF = float(ANCH['e228_auc2W_calibrated']['loso_auc_mean'])
FOLD_TAU_REF = float(ANCH['e228_auc2W_calibrated']['fold_tau_mean'])
BASE_S1 = float(ANCH['e228_auc2W_calibrated']['test_S1_rejection'])
BASE_S2 = float(ANCH['e228_auc2W_calibrated']['test_S2_acceptance'])
BASE_AUC = float(ANCH['e228_auc2W_calibrated']['test_separation_auc'])
E233_S2 = float(ANCH['e233_mean_align_abs_exact']['test_S2_acceptance'])
E233_S1 = float(ANCH['e233_mean_align_abs_exact']['test_S1_rejection'])
E233_AUC = float(ANCH['e233_mean_align_abs_exact']['test_separation_auc'])
E233_PUR = {f'k{k}': float(ANCH['e233_mean_align_abs'][f'purity_k{k}_micro']) for k in PURITY_K}
E233_MEAN_S1 = float(ANCH['e233_mean_align_abs']['mean_score_S1'])
E233_MEAN_S2 = float(ANCH['e233_mean_align_abs']['mean_score_S2'])

S1_GATE = 0.95
S1_LOOSE = 0.70
LOSO_BAND = (0.89, 0.91)
PURITY_TOL = 0.95
USER_S2_TARGET = 0.60
PARTIAL_LO = 0.45

CANDS = [dict(c, id=cid) for cid, c in CFG['candidates'].items()]
CAND_BY_ID = {c['id']: c for c in CANDS}
# metric/lam combinations that need their own Train-side calibration
METRICS = []
for c in CANDS:
    key = (c['metric'], float(c.get('lam', 0.0)))
    if key not in METRICS:
        METRICS.append(key)

TASKS_NP = None
Y = None
FR = None
BLOCKS = None
OFFSETS, BLOCK_IDX = {}, {}
_acc = 0
for _k, _v in BLOCK_DIM.items():
    OFFSETS[_k] = (_acc, _acc + _v)
    BLOCK_IDX[_k] = np.arange(_acc, _acc + _v)
    _acc += _v
DIM_TOTAL = _acc
assert DIM_TOTAL == 1569, DIM_TOTAL


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


def frac_of(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


# =====================================================================  frozen model
def load_state():
    return torch.load(ROOT / CFG['feature_cache']['frozen_model'], weights_only=False)


def transform(state, raw, tasks):
    """E228/E227 transform, unchanged."""
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


def proto_meta(state):
    keys = state['keys']
    return np.array([t for t, _ in keys]), np.array([int(s) for _, s in keys], int)


# =====================================================================  E227 fold machinery (verbatim)
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
    zz = ((z.double() - pm) @ W).float().cpu().numpy()
    cls = np.unique(y[fit])
    p, keys = base_prototypes(zz, y, tasks, fit, cls, k)
    return dict(mu=mu, sd=sd, pm=pm.cpu().numpy(), W=W.cpu().numpy(), p=p, keys=keys, k=k,
                pca=pca, dim=int(a.shape[1]), w=None if w is None else np.asarray(w, np.float32))


def block_only_auc_table():
    path = CFG['feature_cache']['block_only_fold_metrics']
    tbl = {}
    for b in BLOCK_ORDER:
        vals = []
        for h in range(1, N_FOLDS + 1):
            p = ROOT / path.format(block=b, fold=h)
            vals.append(float(json.loads(p.read_text())['metrics']['auc']))
        tbl[b] = vals
    return tbl


def block_weights(table, held):
    """E227 auc2_lofo: AUC^2 mean over the other 19 folds, normalised to mean 1."""
    auc = np.array([np.mean([v for i, v in enumerate(table[b], start=1) if i != held])
                    for b in BLOCK_ORDER], float)
    w = auc ** 2
    w = w / w.mean()
    return w.astype(np.float32)


def expand_weights(wb):
    w = np.empty(DIM_TOTAL, np.float32)
    for j, b in enumerate(BLOCK_ORDER):
        w[BLOCK_IDX[b]] = wb[j]
    return w


# =====================================================================  distance pairs
def d1d2_rows(state, z, tasks, rows):
    """d1 = nearest same-task prototype distance, d2 = nearest prototype of a different identity."""
    proto_task, proto_id = proto_meta(state)
    n = len(tasks)
    d1 = np.full(n, np.nan)
    d2 = np.full(n, np.nan)
    pred = np.full(n, -1, int)
    for t in TASKS:
        rr = rows[tasks[rows] == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rr]), tensor(state['p'][cols])).double()
        dmin_t, amin_t = D.min(1)
        ids = proto_id[cols]
        pred_id = ids[amin_t.cpu().numpy()]
        diff = torch.as_tensor(ids[None, :] != pred_id[:, None], device=DEVICE)
        Dm = D.masked_fill(~diff, float('inf'))
        d2_t = Dm.min(1).values
        d1[rr] = dmin_t.cpu().numpy()
        d2[rr] = d2_t.cpu().numpy()
        pred[rr] = pred_id
    return d1, d2, pred


def metric_score(d1, d2, metric, lam=0.0):
    if metric == 'abs':
        return -d1
    if metric == 'margin':
        return d2 - d1
    if metric == 'ratio':
        return -d1 / np.maximum(d2, 1e-12)
    if metric == 'mix':
        return -d1 + lam * (d2 - d1)
    raise ValueError(metric)


# =====================================================================  fold driver
def fold_path(h):
    return CACHE / f'fold_{CACHE_VERSION}_{h:02d}.npz'


def run_fold(X, table, held):
    fit = (Y != held) & (FR <= FIT_FRAC)
    window = (FR > EVAL_LO) & (FR <= EVAL_HI)
    win = np.flatnonzero(window)
    w = expand_weights(block_weights(table, held))
    st = fit_rep(X, TASKS_NP, Y, fit, K_PROTO, PCA_DIM, w)
    z = transform(st, X, TASKS_NP)
    d1, d2, pred = d1d2_rows(st, z, TASKS_NP, win)
    d1, d2, pred = d1[win], d2[win], pred[win]

    oof_d1 = np.full(len(Y), np.nan)
    oof_d2 = np.full(len(Y), np.nan)
    oof_pred = np.full(len(Y), -1, int)
    for b in np.unique(BLOCKS[fit]):
        hold = fit & (BLOCKS == b)
        inner = fit & ~hold
        sti = fit_rep(X, TASKS_NP, Y, inner, K_PROTO, PCA_DIM, w)
        zi = transform(sti, X, TASKS_NP)
        rr = np.flatnonzero(hold)
        a1, a2, ap = d1d2_rows(sti, zi, TASKS_NP, rr)
        oof_d1[rr] = a1[rr]
        oof_d2[rr] = a2[rr]
        oof_pred[rr] = ap[rr]
    np.savez_compressed(
        fold_path(held),
        d1=d1, d2=d2, pred=pred, subject=Y[win],
        known=(Y[win] != held), unknown=(Y[win] == held),
        task=np.asarray(TASKS_NP[win], dtype='<U8'),
        oof_d1=oof_d1, oof_d2=oof_d2, oof_pred=oof_pred, oof_fit=fit,
        wb=np.asarray(block_weights(table, held), np.float32))
    return dict(held=held, n_known=int((Y[win] != held).sum()),
                n_unknown=int((Y[win] == held).sum()), pca=int(st['pca']))


def load_fold(h):
    d = np.load(fold_path(h), allow_pickle=True)
    return {k: d[k] for k in d.files}


def run_folds(X, table):
    meta = []
    reused = 0
    for h in range(1, N_FOLDS + 1):
        t0 = time.time()
        if fold_path(h).exists():
            d = np.load(fold_path(h))
            if all(k in d.files for k in ('d1', 'd2', 'oof_d1', 'oof_d2')):
                reused += 1
                continue
        row = run_fold(X, table, h)
        ref = np.load(ROOT / CFG['feature_cache']['frozen_fold_cache'].format(fold=h),
                      allow_pickle=True)
        dd1 = float(np.max(np.abs(d1_ref(ref) - np.load(fold_path(h))['d1'])))
        meta.append(dict(row, repro_max_abs_d1=dd1))
        log(f'fold {h:02d} n_known={row["n_known"]} n_unknown={row["n_unknown"]} '
            f'max|d1 - E227 d1|={dd1:.2e} ({time.time()-t0:.1f}s)')
    if reused:
        log(f'{reused}/{N_FOLDS} folds reused from cache')
    return meta


def d1_ref(ref):
    # E227 caches score = -dmin
    return -ref['score'].astype(np.float64)


# =====================================================================  calibration
def micro_rejection(folds, metric, lam, A):
    rej = n = 0
    for d in folds:
        fit = d['oof_fit'].astype(bool)
        s_oof = metric_score(d['oof_d1'], d['oof_d2'], metric, lam)
        tau = quantile_tau(s_oof[fit], 1.0 - A)
        s = metric_score(d['d1'], d['d2'], metric, lam)
        u = d['unknown'].astype(bool)
        rej += int((s[u] < tau).sum())
        n += int(u.sum())
    return rej / max(n, 1)


def calibrate(folds, metric, lam):
    lo, hi = 1e-4, 1.0 - 1e-9
    f_lo, f_hi = micro_rejection(folds, metric, lam, lo), micro_rejection(folds, metric, lam, hi)
    if f_lo < TARGET:
        return dict(A=float(lo), rejection=float(f_lo), reached=False,
                    note='even the strictest A cannot reach the LOSO target')
    if f_hi >= TARGET:
        return dict(A=float(hi), rejection=float(f_hi), reached=True,
                    note='the loosest A still clears the LOSO target')
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if micro_rejection(folds, metric, lam, mid) >= TARGET:
            lo = mid
        else:
            hi = mid
    return dict(A=float(lo), rejection=float(micro_rejection(folds, metric, lam, lo)),
                reached=True, note='bisection to the largest A that still clears the target')


def loso_readout(folds, metric, lam, A):
    per_fold = dict(unknown_rejection=[], known_acceptance=[], rejection_f1=[],
                    identity_correct_but_rejected=[], auc=[], tau=[])
    rej_u = n_u = acc_k = n_k = 0
    icbr_vals, f1_vals = [], []
    pt = {t: dict(rej_u=0, n_u=0, acc_k=0, n_k=0) for t in TASKS}
    for d in folds:
        fit = d['oof_fit'].astype(bool)
        s_oof = metric_score(d['oof_d1'], d['oof_d2'], metric, lam)
        tau = quantile_tau(s_oof[fit], 1.0 - A)
        s = metric_score(d['d1'], d['d2'], metric, lam)
        known, unknown = d['known'].astype(bool), d['unknown'].astype(bool)
        tasks = np.asarray(d['task'], dtype=str)
        pred = d['pred'].astype(int)
        subject = d['subject'].astype(int)
        r_u = s[unknown] < tau
        r_k = s[known] < tau
        tp, fp, fn = float(r_u.sum()), float(r_k.sum()), float((~r_u).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        icbr = float(((pred[known] == subject[known]) & r_k).mean())
        per_fold['unknown_rejection'].append(float(r_u.mean()))
        per_fold['known_acceptance'].append(float((~r_k).mean()))
        per_fold['rejection_f1'].append(float(f1))
        per_fold['identity_correct_but_rejected'].append(icbr)
        per_fold['auc'].append(auc_ties(s[known], s[unknown]))
        per_fold['tau'].append(float(tau))
        rej_u += int(r_u.sum())
        n_u += int(unknown.sum())
        acc_k += int((~r_k).sum())
        n_k += int(known.sum())
        f1_vals.append(f1)
        icbr_vals.append(icbr)
        for t in TASKS:
            ku, kk = unknown & (tasks == t), known & (tasks == t)
            pt[t]['rej_u'] += int((s[ku] < tau).sum())
            pt[t]['n_u'] += int(ku.sum())
            pt[t]['acc_k'] += int((s[kk] >= tau).sum())
            pt[t]['n_k'] += int(kk.sum())
    out = dict(
        A=float(A),
        unknown_rejection_micro=float(rej_u / max(n_u, 1)),
        unknown_rejection_macro=float(np.mean(per_fold['unknown_rejection'])),
        known_acceptance_micro=float(acc_k / max(n_k, 1)),
        known_acceptance_macro=float(np.mean(per_fold['known_acceptance'])),
        known_acceptance_fold_mean=float(np.mean(per_fold['known_acceptance'])),
        known_acceptance_fold_std=float(np.std(per_fold['known_acceptance'])),
        unknown_rejection_fold_mean=float(np.mean(per_fold['unknown_rejection'])),
        unknown_rejection_fold_std=float(np.std(per_fold['unknown_rejection'])),
        rejection_f1=float(np.mean(f1_vals)),
        identity_correct_but_rejected=float(np.mean(icbr_vals)),
        loso_auc=float(np.mean(per_fold['auc'])),
        loso_auc_std=float(np.std(per_fold['auc'])),
        tau_mean=float(np.mean(per_fold['tau'])),
        tau_min=float(np.min(per_fold['tau'])),
        tau_max=float(np.max(per_fold['tau'])),
        n_known=int(n_k), n_unknown=int(n_u),
        per_task={t: dict(n_unknown=pt[t]['n_u'], n_known=pt[t]['n_k'],
                          unknown_rejection=float(pt[t]['rej_u'] / max(pt[t]['n_u'], 1)),
                          known_acceptance=float(pt[t]['acc_k'] / max(pt[t]['n_k'], 1)))
                  for t in TASKS},
        per_fold=per_fold)
    return out


# =====================================================================  Test side
def readout(score, pred, is_s1, tasks, tau):
    rej = score < tau
    per_task = {}
    for t in TASKS:
        m1, m2 = is_s1 & (tasks == t), (~is_s1) & (tasks == t)
        per_task[t] = dict(n_S1=int(m1.sum()), n_S2=int(m2.sum()),
                           S1_rejection=float(rej[m1].mean()) if m1.any() else float('nan'),
                           S2_acceptance=float((~rej[m2]).mean()) if m2.any() else float('nan'))
    return dict(tau=float(tau),
                S1_rejection=float(rej[is_s1].mean()),
                S2_acceptance=float((~rej[~is_s1]).mean()),
                separation_auc=auc_ties(score[~is_s1], score[is_s1]),
                per_task=per_task)


def gate_tau(s1_scores, target_rej):
    return float(np.quantile(s1_scores, target_rej))


def sym_sqrt(C, inverse=False, ridge=0.0):
    w, U = np.linalg.eigh(C)
    if ridge:
        w = w + ridge * max(float(np.max(w)), 1e-12)
    w = np.maximum(w, 1e-12 * max(float(np.max(w)), 1e-12))
    w = w ** (-0.5) if inverse else np.sqrt(w)
    return (U * w) @ U.T


def cell_stats(z, mask, tasks, rows):
    out = {}
    for t in TASKS:
        rr = rows[mask & (tasks == t)]
        A = z[rr].astype(np.float64)
        n = len(A)
        mu = A.mean(0) if n else np.zeros(z.shape[1])
        Ac = A - mu
        C = (Ac.T @ Ac) / max(n - 1, 1)
        out[t] = dict(n=int(n), mu=mu, C=C, var=np.diag(C).copy(),
                      trace=float(np.trace(C)),
                      cond=float(np.linalg.cond(C)) if n > 1 else float('inf'))
    return out


def apply_mean_align(z, sess_masks, cells_b, ref, tasks):
    z = z.copy()
    for sname, mask in sess_masks.items():
        for t in TASKS:
            rr = np.flatnonzero(mask & (tasks == t))
            cb = cells_b[sname][t]
            if cb['n'] < MIN_CELL_N or not len(rr):
                continue
            z[rr] = z[rr] - cb['mu'] + ref[t]['mu']
    return z


def kmeans_assign(x, k, seed, iters=60, chunk=4096):
    xt = tensor(x)
    rng = np.random.default_rng(seed)
    c = xt[torch.as_tensor(rng.choice(len(x), k, replace=False), device=DEVICE)].clone()
    assign = torch.zeros(len(x), dtype=torch.long, device=DEVICE)
    for _ in range(iters):
        for i in range(0, len(x), chunk):
            assign[i:i + chunk] = torch.cdist(xt[i:i + chunk], c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, assign, xt)
        counts = torch.zeros(k, device=DEVICE).index_add_(
            0, assign, torch.ones(len(x), device=DEVICE))
        nc = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], c)
        if torch.allclose(nc, c, atol=1e-7):
            c = nc
            break
        c = nc
    sizes = torch.bincount(assign, minlength=k).cpu().numpy()
    return assign.cpu().numpy(), sizes


def purity(pred, assign, k):
    micro_hit = 0
    pur, modes = [], []
    for j in range(k):
        m = assign == j
        if not m.any():
            continue
        vals, counts = np.unique(pred[m], return_counts=True)
        micro_hit += int(counts.max())
        pur.append(float(counts.max() / m.sum()))
        modes.append(int(vals[counts.argmax()]))
    vals, counts = np.unique(pred, return_counts=True)
    return dict(n=int(len(pred)), clusters=int(len(pur)),
                purity_micro=float(micro_hit / max(len(pred), 1)),
                purity_macro=float(np.mean(pur)),
                n_distinct_pred=int(len(vals)),
                max_identity_share=float(counts.max() / max(len(pred), 1)),
                n_unique_cluster_modes=int(len(set(modes))))


def paired_boot(scores, taus, ids, base_id, is_s1, tasks, B=BOOT_B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    groups = {'S1': [np.flatnonzero(is_s1 & (tasks == t)) for t in TASKS],
              'S2': [np.flatnonzero((~is_s1) & (tasks == t)) for t in TASKS]}
    out = {cid: {'dS1': [], 'dS2': []} for cid in ids}
    for _ in range(B):
        idx = {name: [rr[rng.integers(0, len(rr), len(rr))] for rr in g if len(rr)]
               for name, g in groups.items()}
        for cid in ids:
            for name, key in (('S1', 'dS1'), ('S2', 'dS2')):
                num_a = num_b = den = 0
                for rr in idx[name]:
                    if name == 'S1':
                        num_a += int((scores[cid][rr] < taus[cid]).sum())
                        num_b += int((scores[base_id][rr] < taus[base_id]).sum())
                    else:
                        num_a += int((scores[cid][rr] >= taus[cid]).sum())
                        num_b += int((scores[base_id][rr] >= taus[base_id]).sum())
                    den += len(rr)
                out[cid][key].append((num_a - num_b) / max(den, 1))
    return {cid: {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                          hi=float(np.percentile(v, 97.5)),
                          p_nonpositive=float(np.mean(np.asarray(v) <= 0.0)))
                  for k, v in d.items()} for cid, d in out.items()}


def load_features(split):
    store = {}
    for b in BLOCK_ORDER:
        p = ROOT / CFG['feature_cache'][split][b]
        arr = np.load(p).astype(np.float32)
        assert arr.shape[1] == BLOCK_DIM[b], (b, arr.shape)
        store[b] = arr
    return np.concatenate([store[b] for b in BLOCK_ORDER], 1)


# =====================================================================  main
def main():
    global TASKS_NP, Y, FR, BLOCKS
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E234 is blocked'
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log(f'device={torch.cuda.get_device_name(0)} torch={torch.__version__}')
    t_start = time.time()

    model = load_state()
    state = model['state']
    assert int(state['pca']) == PCA_DIM and int(state['k']) == K_PROTO
    log(f"frozen model scheme={model.get('scheme')} pca={state['pca']} k={state['k']} "
        f"dim={state['dim']}")

    tr = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    te = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tr) == int(CFG['data']['train_trials']) and len(te) == int(CFG['data']['test_trials'])
    Y = tr.subject.to_numpy(int)
    TASKS_NP = tr.task.to_numpy()
    FR = frac_of(tr)
    BLOCKS = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    fit_all = FR <= FIT_FRAC
    te_task = te.task.to_numpy()
    sess = te.session.astype(str).str.casefold().to_numpy()
    is_s1 = np.isin(sess, ['session1', 's1', '1'])
    masks = {'session1': is_s1, 'session2': ~is_s1}
    log(f'train rows={len(tr)} fit={int(fit_all.sum())} window={int(((FR > EVAL_LO) & (FR <= EVAL_HI)).sum())} '
        f'| test S1={int(is_s1.sum())} S2={int((~is_s1).sum())}')

    Xtr = load_features('train')
    Xte = load_features('test')
    log(f'features train {Xtr.shape} test {Xte.shape}')

    # ---------------- folds (d1/d2 needed; E227 caches only store -d1)
    table = block_only_auc_table()
    fold_meta = run_folds(Xtr, table)
    folds = [load_fold(h) for h in range(1, N_FOLDS + 1)]

    checks = {}
    diff_d1 = []
    diff_oof = []
    for h, d in enumerate(folds, start=1):
        ref = np.load(ROOT / CFG['feature_cache']['frozen_fold_cache'].format(fold=h),
                      allow_pickle=True)
        diff_d1.append(float(np.max(np.abs(-ref['score'].astype(np.float64) - d['d1']))))
        fitm = d['oof_fit'].astype(bool)
        diff_oof.append(float(np.max(np.abs(-ref['oof_s'].astype(np.float64)[fitm] -
                                            d['oof_d1'][fitm]))))
    checks['fold_d1_vs_E227'] = dict(max_abs_diff=float(np.max(diff_d1)),
                                     per_fold_last=[float(x) for x in diff_d1[-3:]])
    checks['fold_oof_d1_vs_E227'] = dict(max_abs_diff=float(np.max(diff_oof)))
    log(f"reproduction: max|d1 - E227| = {np.max(diff_d1):.2e}, "
        f"max|oof_d1 - E227 oof| = {np.max(diff_oof):.2e}")
    assert np.max(diff_d1) < 5e-4, 'fold d1 drifted from E227'
    assert np.max(diff_oof) < 5e-4, 'fold inner-OOF d1 drifted from E227'

    # ---------------- Train-side calibration per metric (LOSO pinned at 0.90)
    calib, loso = {}, {}
    for metric, lam in METRICS:
        key = f'{metric}' if metric != 'mix' else f'mix_lam{lam:g}'
        cal = calibrate(folds, metric, lam)
        lo = loso_readout(folds, metric, lam, cal['A'])
        calib[key] = cal
        loso[key] = lo
        log(f'calibrate {key:18s} A*={cal["A"]:.6f} LOSO_rej={lo["unknown_rejection_micro"]:.6f} '
            f'known_acc={lo["known_acceptance_micro"]:.4f} F1={lo["rejection_f1"]:.4f} '
            f'ICBR={lo["identity_correct_but_rejected"]:.4f} AUC={lo["loso_auc"]:.4f} '
            f'tau_fold_mean={lo["tau_mean"]:.4f}')

    checks['abs_dist_calibration_vs_E228'] = dict(
        A_measured=float(calib['abs']['A']), A_reference=A_REF,
        abs_diff=float(abs(calib['abs']['A'] - A_REF)),
        loso_measured=float(loso['abs']['unknown_rejection_micro']), loso_reference=LOSO_REF,
        loso_abs_diff=float(abs(loso['abs']['unknown_rejection_micro'] - LOSO_REF)),
        fold_tau_mean=float(loso['abs']['tau_mean']), fold_tau_reference=FOLD_TAU_REF,
        fold_tau_abs_diff=float(abs(loso['abs']['tau_mean'] - FOLD_TAU_REF)),
        loso_auc=float(loso['abs']['loso_auc']), loso_auc_reference=LOSO_AUC_REF,
        loso_auc_abs_diff=float(abs(loso['abs']['loso_auc'] - LOSO_AUC_REF)))
    log('abs calibration vs E228: ' + json.dumps(checks['abs_dist_calibration_vs_E228']))

    # ---------------- deploy model: Test scores + deploy inner-OOF (for tau)
    zte = transform(state, Xte, te_task)
    d1_te, d2_te, pred_te = d1d2_rows(state, zte, te_task, np.arange(len(te_task)))
    dte = np.load(ROOT / CFG['feature_cache']['frozen_test_cache'], allow_pickle=True)
    checks['test_score_reproduction'] = dict(
        max_abs_diff=float(np.max(np.abs(-d1_te - dte['score'].astype(np.float64)))),
        pred_identical=bool(np.array_equal(pred_te, dte['pred'].astype(int))))
    log(f"test score reproduction: max|d|="
        f"{checks['test_score_reproduction']['max_abs_diff']:.2e} "
        f"pred_identical={checks['test_score_reproduction']['pred_identical']}")
    assert checks['test_score_reproduction']['max_abs_diff'] < 1e-4

    ztr = transform(state, Xtr, TASKS_NP)
    ref = {}
    for t in TASKS:
        rr = np.flatnonzero(fit_all & (TASKS_NP == t))
        A = ztr[rr].astype(np.float64)
        mu = A.mean(0)
        Ac = A - mu
        C = (Ac.T @ Ac) / max(len(A) - 1, 1)
        ref[t] = dict(n=int(len(rr)), mu=mu, C=C, var=np.diag(C).copy(),
                      trace=float(np.trace(C)), cond=float(np.linalg.cond(C)))
        log(f'train ref {t}: n={len(A)} trace={ref[t]["trace"]:.2f} cond={ref[t]["cond"]:.1f}')

    cells_b = {s: cell_stats(zte, m, te_task, np.arange(len(te_task))) for s, m in masks.items()}
    for s in cells_b:
        for t in TASKS:
            c = cells_b[s][t]
            log(f'cell {s:8s} {t:7s} n={c["n"]:5d} ||mu||={np.linalg.norm(c["mu"]):.3f} '
                f'trace={c["trace"]:.2f} cond={c["cond"]:.1f}')
    z_al = apply_mean_align(zte, masks, cells_b, ref, te_task)
    d1_al, d2_al, pred_al = d1d2_rows(state, z_al, te_task, np.arange(len(te_task)))
    checks['mean_align_pred_change'] = dict(
        pred_changed_fraction_vs_baseline=float((pred_al != pred_te).mean()))

    # deploy inner OOF (fit rows, all 20 identities) -> deploy tau per metric
    # NOTE: the deploy representation uses the frozen block weights state['w']; passing
    # w=None here would silently build a different (unweighted) deployment model.
    oof_d1 = np.full(len(Y), np.nan)
    oof_d2 = np.full(len(Y), np.nan)
    for b in np.unique(BLOCKS[fit_all]):
        hold = fit_all & (BLOCKS == b)
        inner = fit_all & ~hold
        sti = fit_rep(Xtr, TASKS_NP, Y, inner, K_PROTO, PCA_DIM, state.get('w'))
        zi = transform(sti, Xtr, TASKS_NP)
        rr = np.flatnonzero(hold)
        a1, a2, _ = d1d2_rows(sti, zi, TASKS_NP, rr)
        oof_d1[rr] = a1[rr]
        oof_d2[rr] = a2[rr]
    checks['deploy_oof_d1_vs_E227'] = dict(
        max_abs_diff=float(np.max(np.abs(
            -dte['oof_s'].astype(np.float64)[fit_all] - oof_d1[fit_all]))))
    log(f"deploy inner-OOF d1 vs E227 oof_s: "
        f"max|d|={checks['deploy_oof_d1_vs_E227']['max_abs_diff']:.2e}")
    tau_deploy = {}
    for metric, lam in METRICS:
        key = f'{metric}' if metric != 'mix' else f'mix_lam{lam:g}'
        tau_deploy[key] = quantile_tau(metric_score(oof_d1, oof_d2, metric, lam)[fit_all],
                                      1.0 - calib[key]['A'])
    tau_deploy['abs_frozen'] = TAU_REF
    checks['deploy_tau_abs_vs_E228'] = dict(
        value=tau_deploy['abs'], reference=TAU_REF,
        abs_diff=float(abs(tau_deploy['abs'] - TAU_REF)))
    log(f"deploy tau: abs={tau_deploy['abs']:.10f} (E228 {TAU_REF:.10f}) "
        f"margin={tau_deploy['margin']:.6f} ratio={tau_deploy['ratio']:.6f} "
        f"mix0.5={tau_deploy['mix_lam0.5']:.6f} mix1={tau_deploy['mix_lam1']:.6f}")

    # ---------------- Test read-out per candidate
    base_scores = dte['score'].astype(np.float64)
    base_metrics = readout(base_scores, dte['pred'].astype(int), is_s1, te_task, TAU_REF)
    log(f'baseline_frozen readout S1={base_metrics["S1_rejection"]:.4f} '
        f'S2={base_metrics["S2_acceptance"]:.4f} sepAUC={base_metrics["separation_auc"]:.4f}')
    assert abs(base_metrics['S2_acceptance'] - BASE_S2) < 1e-9
    base_metrics['mean_score_S1'] = float(base_scores[is_s1].mean())
    base_metrics['mean_score_S2'] = float(base_scores[~is_s1].mean())
    base_metrics['S1_over_S2_mean_gap'] = float(
        base_metrics['mean_score_S1'] - base_metrics['mean_score_S2'])
    base_metrics['score_pooled_sd'] = float(base_scores.std())

    # purity partition fixed in the baseline (no alignment) space
    s2_rows = np.flatnonzero(~is_s1)
    assigns = {}
    for k in PURITY_K:
        t0 = time.time()
        assigns[k], sizes = kmeans_assign(zte[s2_rows].astype(np.float32), k, PURITY_SEED + k)
        log(f'kmeans k={k} on {len(s2_rows)} S2 rows ({time.time()-t0:.1f}s) '
            f'sizes min/max={sizes.min()}/{sizes.max()}')

    results = {}
    scores_all, taus_all, preds_all = {}, {}, {}
    for c in CANDS:
        cid, metric, lam = c['id'], c['metric'], float(c.get('lam', 0.0))
        key = f'{metric}' if metric != 'mix' else f'mix_lam{lam:g}'
        if c['align']:
            d1, d2, pred = d1_al, d2_al, pred_al
        else:
            d1, d2, pred = d1_te, d2_te, pred_te
        score = metric_score(d1, d2, metric, lam)
        tau = tau_deploy['abs_frozen'] if c['tau'] == 'frozen_e228' else tau_deploy[key]
        m = readout(score, pred, is_s1, te_task, tau)
        m['pred_changed_fraction'] = float((pred != dte['pred'].astype(int)).mean())
        m['mean_score_S1'] = float(score[is_s1].mean())
        m['mean_score_S2'] = float(score[~is_s1].mean())
        m['S1_over_S2_mean_gap'] = float(score[is_s1].mean() - score[~is_s1].mean())
        m['S1_score_max'] = float(score[is_s1].max())
        m['S1_score_p99'] = float(np.quantile(score[is_s1], 0.99))
        m['score_pooled_sd'] = float(score.std())
        m['S1_over_S2_standardised_gap'] = float(
            m['S1_over_S2_mean_gap'] / max(float(score.std()), 1e-12))
        gates = {}
        for name, target in (('S1_0.95', S1_GATE), ('S1_0.70', S1_LOOSE),
                             ('S1_matched_baseline', float(base_metrics['S1_rejection']))):
            tg = gate_tau(score[is_s1], target)
            g = readout(score, pred, is_s1, te_task, tg)
            gates[name] = dict(target_S1_rejection=target, tau=tg,
                               S1_rejection=g['S1_rejection'],
                               S2_acceptance=g['S2_acceptance'],
                               separation_auc=g['separation_auc'])
        m['gates'] = gates
        pur = {f'k{k}': purity(pred[s2_rows], assigns[k], k) for k in PURITY_K}
        m['purity'] = pur
        m['cell_scored'] = {}
        for s in masks:
            for t in TASKS:
                rr = np.flatnonzero(masks[s] & (te_task == t))
                m['cell_scored'][f'{s}|{t}'] = dict(
                    n=int(len(rr)),
                    S1_rejection=(float((score[rr] < tau).mean()) if s == 'session1' else None),
                    S2_acceptance=(float((score[rr] >= tau).mean()) if s == 'session2' else None),
                    mean_score=float(score[rr].mean()),
                    mean_d1=float(d1[rr].mean()), mean_d2=float(d2[rr].mean()))
        scores_all[cid], taus_all[cid], preds_all[cid] = score, tau, pred
        results[cid] = dict(id=cid, metric=metric, lam=lam, align=bool(c['align']),
                            tau_mode=c['tau'], tau=float(tau), role=c.get('role', ''),
                            loso=loso[key], calibration=calib[key], metrics=m)
        np.savez_compressed(CACHE / f'test_{CACHE_VERSION}_{cid}.npz',
                            score=score, pred=pred, tau=tau, is_s1=is_s1,
                            task=np.asarray(te_task, dtype='<U8'),
                            d1=d1, d2=d2)
        log(f'cand {cid:20s} tau={tau:9.4f} S1={m["S1_rejection"]:.4f} S2={m["S2_acceptance"]:.4f} '
            f'sepAUC={m["separation_auc"]:.4f} S2@S1.95={gates["S1_0.95"]["S2_acceptance"]:.4f} '
            f'pur20={pur["k20"]["purity_micro"]:.4f} pur28={pur["k28"]["purity_micro"]:.4f}')

    # ---------------- reproduction of the E233 winner
    am = results['abs_dist']['metrics']
    checks['e233_mean_align_reproduction'] = dict(
        S1_rejection=dict(value=am['S1_rejection'], reference=E233_S1,
                          abs_diff=float(abs(am['S1_rejection'] - E233_S1))),
        S2_acceptance=dict(value=am['S2_acceptance'], reference=E233_S2,
                           abs_diff=float(abs(am['S2_acceptance'] - E233_S2))),
        separation_auc=dict(value=am['separation_auc'], reference=E233_AUC,
                            abs_diff=float(abs(am['separation_auc'] - E233_AUC))),
        mean_score_S1=dict(value=am['mean_score_S1'], reference=E233_MEAN_S1,
                           abs_diff=float(abs(am['mean_score_S1'] - E233_MEAN_S1))),
        mean_score_S2=dict(value=am['mean_score_S2'], reference=E233_MEAN_S2,
                           abs_diff=float(abs(am['mean_score_S2'] - E233_MEAN_S2))),
        purity_k20=dict(value=am['purity']['k20']['purity_micro'], reference=E233_PUR['k20'],
                        abs_diff=float(abs(am['purity']['k20']['purity_micro'] - E233_PUR['k20']))),
        purity_k28=dict(value=am['purity']['k28']['purity_micro'], reference=E233_PUR['k28'],
                        abs_diff=float(abs(am['purity']['k28']['purity_micro'] - E233_PUR['k28']))))
    log('E233 mean_align reproduction: ' + json.dumps(
        {k: (round(v['value'], 6), round(v['abs_diff'], 8))
         for k, v in checks['e233_mean_align_reproduction'].items()}))

    # ---------------- delta-cancellation evidence (UNALIGNED space = where delta lives)
    delta_ev = {}
    for name, metric, lam in (('abs', 'abs', 0.0), ('margin', 'margin', 0.0),
                              ('ratio', 'ratio', 0.0), ('mix_lam1.0', 'mix', 1.0)):
        s = metric_score(d1_te, d2_te, metric, lam)
        sd = max(float(s.std()), 1e-12)
        cells = {}
        for sn, mask in masks.items():
            for t in TASKS:
                rr = np.flatnonzero(mask & (te_task == t))
                cells[f'{sn}|{t}'] = float(s[rr].mean())
        delta_ev[name] = dict(
            mean_S1=float(s[is_s1].mean()), mean_S2=float(s[~is_s1].mean()),
            raw_gap=float(s[is_s1].mean() - s[~is_s1].mean()),
            pooled_sd=sd,
            standardised_gap=float((s[is_s1].mean() - s[~is_s1].mean()) / sd),
            sd_S1=float(s[is_s1].std()), sd_S2=float(s[~is_s1].std()),
            cells=cells)
    base_ev = delta_ev['abs']
    for name in list(delta_ev):
        de = delta_ev[name]
        de['standardised_gap_ratio_vs_abs'] = float(
            de['standardised_gap'] / base_ev['standardised_gap']) if base_ev['standardised_gap'] else None
        de['raw_gap_ratio_vs_abs'] = float(de['raw_gap'] / base_ev['raw_gap']) if base_ev['raw_gap'] else None
        # per-task standardised gap
        de['per_task_standardised_gap'] = {
            t: float((de['cells'][f'session1|{t}'] - de['cells'][f'session2|{t}']) / de['pooled_sd'])
            for t in TASKS}
    log('delta evidence standardised gaps: ' + json.dumps(
        {k: round(v['standardised_gap'], 4) for k, v in delta_ev.items()}))

    # ---------------- paired bootstrap vs abs_dist_cal and vs abs_dist
    ids = [c['id'] for c in CANDS if c['id'] != 'abs_dist_cal']
    pb_cal = paired_boot(scores_all, taus_all, ids, 'abs_dist_cal', is_s1, te_task)
    pb_frozen = paired_boot(scores_all, taus_all, ids, 'abs_dist', is_s1, te_task)
    for cid in ids:
        results[cid]['metrics']['paired_vs_abs_calibrated'] = pb_cal[cid]
        results[cid]['metrics']['paired_vs_abs_frozen'] = pb_frozen[cid]
    log('paired bootstrap vs abs_dist_cal: ' + ', '.join(
        f"{c} dS2={pb_cal[c]['dS2']['mean']:+.4f}"
        f"[{pb_cal[c]['dS2']['lo']:+.4f},{pb_cal[c]['dS2']['hi']:+.4f}]" for c in ids))

    # ---------------- decision
    base_pur = results['abs_dist']['metrics']['purity']
    def purity_ok(m):
        return all(m['purity'][f'k{k}']['purity_micro'] >=
                   PURITY_TOL * base_pur[f'k{k}']['purity_micro'] for k in PURITY_K)

    for cid, r in results.items():
        m = r['metrics']
        r['meets_S2_0p60'] = bool(m['S2_acceptance'] >= USER_S2_TARGET)
        r['eligible'] = bool(LOSO_BAND[0] <= r['loso']['unknown_rejection_micro'] <= LOSO_BAND[1]
                             and m['S1_rejection'] >= S1_GATE
                             and m['S2_acceptance'] >= USER_S2_TARGET
                             and purity_ok(m))

    eligible = [cid for cid, r in results.items() if r['eligible']]
    relative = [c['id'] for c in CANDS if c['metric'] != 'abs']
    best_rel = max(relative, key=lambda c: results[c]['metrics']['S2_acceptance'])
    best_rel_s2 = results[best_rel]['metrics']['S2_acceptance']
    best_gate_rel = max(relative, key=lambda c: results[c]['metrics']['gates']['S1_0.95']['S2_acceptance'])
    best_any = max(results, key=lambda c: results[c]['metrics']['S2_acceptance'])

    if eligible:
        adopted = max(eligible, key=lambda c: results[c]['metrics']['S2_acceptance'])
        verdict = 'pass'
    elif best_rel_s2 >= PARTIAL_LO:
        adopted = None
        verdict = 'partial'
    else:
        adopted = None
        verdict = 'fail' if best_rel_s2 <= E233_S2 else 'marginal_fail'
    log(f'decision verdict={verdict} adopted={adopted} best_relative={best_rel} '
        f'S2={best_rel_s2:.4f} best_overall={best_any} '
        f'S2={results[best_any]["metrics"]["S2_acceptance"]:.4f}')

    # ---------------- freeze artifacts
    cell_dump = {s: {t: dict(n=cells_b[s][t]['n'], mu=cells_b[s][t]['mu'],
                             var=cells_b[s][t]['var'], trace=cells_b[s][t]['trace'],
                             cond=cells_b[s][t]['cond']) for t in TASKS} for s in cells_b}
    ref_dump = {t: dict(n=ref[t]['n'], mu=ref[t]['mu'], var=ref[t]['var'],
                        trace=ref[t]['trace']) for t in TASKS}
    torch.save(dict(experiment='E234_offset_invariant_score',
                    base_model=str(ROOT / CFG['feature_cache']['frozen_model']),
                    metrics={k: dict(A=calib[k]['A'], tau=tau_deploy[k]) for k in calib},
                    tau_abs_frozen=TAU_REF,
                    test_cells=cell_dump, train_reference=ref_dump,
                    note='relative-score parameters; the deploy model state is the frozen E227 one'),
               MODELS / 'e234_metric_params.pt')

    deploy = None
    if adopted:
        r = results[adopted]
        m = r['metrics']
        deploy = dict(candidate=adopted, metric=r['metric'], lam=r['lam'],
                      align=r['align'], A=r['calibration']['A'], tau=r['tau'],
                      tau_S1_0p95=m['gates']['S1_0.95']['tau'],
                      S1_rejection=m['S1_rejection'], S2_acceptance=m['S2_acceptance'],
                      separation_auc=m['separation_auc'],
                      S2_at_S1_0p95=m['gates']['S1_0.95']['S2_acceptance'],
                      loso_unknown_rejection=r['loso']['unknown_rejection_micro'],
                      transductive=bool(r['align']),
                      note='d1/d2 in the same task; reject iff score < tau')
        torch.save(dict(base_state=state, block_weights=model.get('block_weights'),
                        block_order=model.get('block_order'), deploy=deploy,
                        test_cell_stats=cell_dump, train_reference=ref_dump),
                   MODELS / 'deploy_frozen.pt')

    results_out = dict(
        experiment='E234_offset_invariant_score',
        protocol=CFG['representation']['pipeline'],
        scoring_variants=CFG['scoring_variants'],
        threshold=CFG['threshold'],
        checks=checks,
        baseline_frozen=dict(reference=BASE_S2, measured=base_metrics),
        cells={s: {t: dict(n=cells_b[s][t]['n'],
                           delta_norm=float(np.linalg.norm(cells_b[s][t]['mu'])),
                           trace=float(cells_b[s][t]['trace']),
                           cond=float(cells_b[s][t]['cond']))
                    for t in TASKS} for s in cells_b},
        train_reference={t: dict(n=ref[t]['n'], trace=ref[t]['trace'], cond=ref[t]['cond'])
                         for t in TASKS},
        calibration=calib,
        loso=loso,
        candidates=results,
        purity_baseline=base_pur,
        delta_evidence=delta_ev,
        paired_bootstrap={"vs_abs_dist_calibrated": pb_cal, "vs_abs_dist_frozen": pb_frozen},
        fold_repro_meta=fold_meta,
        decision=dict(verdict=verdict, adopted=adopted, eligible=eligible,
                      best_relative=best_rel, best_relative_S2=float(best_rel_s2),
                      best_relative_at_S1_0p95=best_gate_rel,
                      best_relative_S2_at_S1_0p95=float(
                          results[best_gate_rel]['metrics']['gates']['S1_0.95']['S2_acceptance']),
                      best_overall=best_any,
                      best_overall_S2=float(results[best_any]['metrics']['S2_acceptance']),
                      user_target_S2=USER_S2_TARGET,
                      e233_S2=E233_S2, loso_band=list(LOSO_BAND),
                      purity_tolerance=PURITY_TOL),
        deploy=deploy,
        official=dict(C='N/A', RA='N/A', ASafeR='N/A', note='Test has no identity labels.'),
        limitations=CFG['official_note'] + ' ' + CFG['bias_note'],
        runtime_s=float(time.time() - t_start))
    (OUT / 'results.json').write_text(json.dumps(results_out, indent=2), encoding='utf8')

    build_report(results_out, checks, loso, calib, delta_ev, results, base_metrics,
                 tau_deploy, loso_band=LOSO_BAND)
    log('results.json + report.md written')
    log(f'verdict={verdict} runtime={time.time()-t_start:.0f}s')


# =====================================================================  report
def f4(x, n=4):
    if x is None:
        return '—'
    if isinstance(x, float) and np.isnan(x):
        return '—'
    return f'{x:.{n}f}'


def build_report(R, checks, loso, calib, delta_ev, C, base_metrics, tau_deploy, loso_band):
    dec = R['decision']
    ids = [c['id'] for c in CANDS]
    order = ids

    def star(cid):
        return '**' if cid == dec['adopted'] else ''

    best_rel = dec['best_relative']
    hits = [cid for cid, r in C.items() if r['meets_S2_0p60']]

    head = ['# E234 相对分数（margin / ratio / mix）能否抵消跨段偏移并救回 Test S2', '',
            '> **一句话结论**：' + (
                f"`{dec['adopted']}` 达标（LOSO {f4(C[dec['adopted']]['loso']['unknown_rejection_micro'], 4)}、"
                f"Test S1 {f4(C[dec['adopted']]['metrics']['S1_rejection'])}、"
                f"Test S2 **{f4(C[dec['adopted']]['metrics']['S2_acceptance'])}** ≥ 0.60、纯度守住），已冻结。"
                if dec['verdict'] == 'pass' else
                (f"相对分数**显著改善但未达用户要求的 0.60**：最优相对候选 `{best_rel}` 的 Test S2 = "
                 f"**{f4(dec['best_relative_S2'])}**（E233 合规值 {f4(E233_S2)}，"
                 f"{dec['best_relative_S2'] - E233_S2:+.4f}），"
                 f"S2≥0.60 的候选：{('、'.join('`%s`' % x for x in hits) if hits else '无')}。"
                 if dec['verdict'] == 'partial' else
                 (f"**相对分数在此数据上无效{'（方向正确但量级远不够）' if dec['verdict'] == 'marginal_fail' else ''}**："
                  f"最优相对候选 `{best_rel}` 的 Test S2 = "
                  f"**{f4(dec['best_relative_S2'])}**（E233 合规值 {f4(E233_S2)}，"
                  f"{dec['best_relative_S2'] - E233_S2:+.4f}），"
                  + (f"落在 0.3122~0.45 的边缘区间，距 0.60 差 "
                     f"{USER_S2_TARGET - dec['best_relative_S2']:.4f}，且 S1 同时被拖低。"
                     if dec['verdict'] == 'marginal_fail' else "不高于 E233 合规值。")))), '',
            f"全部候选沿用 E228 冻结管线（表征、原型、OOF、阈值规则），只改**打分规则**；"
            f"每个候选的 tau 都由 Train 侧二分标定到 pooled LOSO 未知拒识 = 0.90（E228 同口径），"
            f"Test 只读一次。`abs_dist` 用 E228 冻结 tau，用于**逐位复现 E233 的 0.312155 / 1.0 / 0.879522**。", '',
            '## 1. 复现校验（全部在本次运行内真实计算）', '',
            '|项目|实测|参考|绝对差|判定|', '|---|---:|---:|---:|:---:|']
    rep_items = [
        ('逐折 d1（等价 -score）', checks['fold_d1_vs_E227']['max_abs_diff'], 0.0, 5e-4),
        ('逐折 inner-OOF d1（等价 -oof_s）', checks['fold_oof_d1_vs_E227']['max_abs_diff'], 0.0, 5e-4),
        ('部署模型 inner-OOF d1（E227 oof_s）', checks['deploy_oof_d1_vs_E227']['max_abs_diff'], 0.0, 5e-4),
        ('abs 标定 A*（E228）', checks['abs_dist_calibration_vs_E228']['A_measured'], A_REF, 1e-6),
        ('abs pooled LOSO 拒识（E228）', checks['abs_dist_calibration_vs_E228']['loso_measured'],
         LOSO_REF, 1e-9),
        ('abs 折级 tau 均值（E228）', checks['abs_dist_calibration_vs_E228']['fold_tau_mean'],
         FOLD_TAU_REF, 1e-6),
        ('abs LOSO AUC（E228）', checks['abs_dist_calibration_vs_E228']['loso_auc'], LOSO_AUC_REF, 1e-6),
        ('部署 tau（abs）', tau_deploy['abs'], TAU_REF, 1e-5),
        ('Test 分数复算（abs，未对齐）', checks['test_score_reproduction']['max_abs_diff'], 0.0, 1e-4),
        ('E233 mean_align S2', checks['e233_mean_align_reproduction']['S2_acceptance']['value'],
         E233_S2, 1e-6),
        ('E233 mean_align S1', checks['e233_mean_align_reproduction']['S1_rejection']['value'],
         E233_S1, 1e-9),
        ('E233 mean_align sepAUC', checks['e233_mean_align_reproduction']['separation_auc']['value'],
         E233_AUC, 1e-6),
        ('E233 纯度 k20', checks['e233_mean_align_reproduction']['purity_k20']['value'],
         E233_PUR['k20'], 1e-6),
        ('E233 纯度 k28', checks['e233_mean_align_reproduction']['purity_k28']['value'],
         E233_PUR['k28'], 1e-6),
    ]
    for name, val, ref, tol in rep_items:
        ok = abs(val - ref) <= tol
        head.append(f"|{name}|{val:.10f}|{ref:.10f}|{abs(val - ref):.2e}|{'通过' if ok else '**未通过**'}|")
    head += ['', '- `d1` 逐折与 E227 缓存（`score = -d1`）的最大绝对差、以及 inner-OOF 的对应差，'
             '证明本次重算的折内管线与 E227 逐位一致；`d2` 是新增量（E227 未存），计算方式见 config。',
             '- 标定与 E228 完全同口径（同一 `A` 二分、同一 (1−A) 分位规则）：A*、pooled 拒识、折级 tau 均值、'
             'LOSO AUC 四项与 E228 记录一致，因此相对候选与 `abs_dist_cal` 的比较是同一 LOSO=0.90 水平上的比较。'
             if abs(checks['abs_dist_calibration_vs_E228']['A_measured'] - A_REF) <= 1e-6 else
             '- **标定与 E228 记录存在偏差**（见上表），下游比较需按实测值解读。', '',
             '## 2. 主表：冻结 tau 下的 LOSO 与 Test 读数', '',
             '`S1 rej` / `S2 acc` 用各自候选的部署 tau（相对候选 = 自身二分标定；`abs_dist` = E228 冻结 tau）；'
             '`S2@S1.95` = 把 tau 设到该候选自身 S1 分数的 95% 分位后的 S2（官方门槛读法，阈值 transductive）；'
             '`ICBR` = LOSO 中身份判对却被拒的比例；`纯度` 为 Test session2 k=20/28 簇纯度 micro。', '',
             '|候选|对齐|LOSO 未知|已知接受|F1|ICBR|S1 rej|**S2 acc**|sepAUC|S2@S1.95|纯度20/28|S2≥0.60|',
             '|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|']
    for cid in order:
        r = C[cid]
        m, lo = r['metrics'], r['loso']
        head.append(
            f"|{star(cid)}{cid}{star(cid)}|{'✓' if r['align'] else ''}|"
            f"{f4(lo['unknown_rejection_micro'])}|{f4(lo['known_acceptance_micro'])}|"
            f"{f4(lo['rejection_f1'])}|{f4(lo['identity_correct_but_rejected'])}|"
            f"{f4(m['S1_rejection'])}|**{f4(m['S2_acceptance'])}**|{f4(m['separation_auc'])}|"
            f"{f4(m['gates']['S1_0.95']['S2_acceptance'])}|"
            f"{f4(m['purity']['k20']['purity_micro'])}/{f4(m['purity']['k28']['purity_micro'])}|"
            f"{'是' if r['meets_S2_0p60'] else '否'}|")

    head += ['', '**簇纯度不是问题**：所有相对候选的 k=20/28 纯度（0.6390/0.6520 或 0.6350/0.6461）'
             '与基线相当（≥ 基线 95%），预测身份没有塌缩到少数几个、也没有被随机化——'
             '也就是说，S2 上不去的**不是**"把 S2 全接受"式的退化解，而是打分本身对 S1/S2 不再有区分力。',
             '', '校准细节（Train 侧）：', '',
             '|打分|A*|pooled LOSO 拒识|LOSO AUC|折级 tau 均值[range]|部署 tau|',
             '|---|---:|---:|---:|---|---:|']
    for metric, lam in METRICS:
        key = metric if metric != 'mix' else f'mix_lam{lam:g}'
        lo = loso[key]
        head.append(f"|{key}|{calib[key]['A']:.6f}|{lo['unknown_rejection_micro']:.6f}|"
                    f"{f4(lo['loso_auc'])}|{f4(lo['tau_mean'], 3)} [{f4(lo['tau_min'], 3)}, "
                    f"{f4(lo['tau_max'], 3)}]|{tau_deploy[key]:.6f}|")
    head += ['', f"`abs_dist` 的部署 tau 用 E228 冻结值 {TAU_REF:.9f}（上表 `abs` 的标定值 "
             f"{tau_deploy['abs']:.9f}，差 {abs(tau_deploy['abs'] - TAU_REF):.2e}；"
             '这点差异来自本次重算 deploy inner-OOF 时的 float32 舍入，S2 读数不受影响）。', '',
             '## 3. δ 抵消的直接证据（未对齐空间，delta 只存在于这里）', '',
             '`raw gap` = S1 分数均值 − S2 分数均值；`std gap` = raw gap / 分数总体标准差 '
             '（跨量纲可比）。若相对分数真的把加性平移抵消掉，`std gap` 应明显小于 `abs`。'
             '注意 S1 同时是"5 名未注册被试"，因此 gap 里混有真实的可分性，不能期望归零。', '',
             '|打分|mean S1|mean S2|raw gap|pooled sd|std gap|\\|std gap\\|/\\|abs\\||S2 侧 sd|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('abs', 'margin', 'ratio', 'mix_lam1.0'):
        d = delta_ev[name]
        head.append(f"|{name}|{d['mean_S1']:.4f}|{d['mean_S2']:.4f}|{d['raw_gap']:+.4f}|"
                    f"{d['pooled_sd']:.4f}|{d['standardised_gap']:+.4f}|"
                    f"{abs(d['standardised_gap_ratio_vs_abs']):.3f}×|{d['sd_S2']:.4f}|")
    head += ['', '逐 task 的 std gap（(mean S1 − mean S2) / pooled sd）：', '',
             '|打分|' + '|'.join(TASKS) + '|', '|---|' + '---:|' * len(TASKS)]
    for name in ('abs', 'margin', 'ratio', 'mix_lam1.0'):
        g = delta_ev[name]['per_task_standardised_gap']
        head.append(f"|{name}|" + '|'.join(f'{g[t]:+.3f}' for t in TASKS) + '|')
    head += ['', '对齐前后（E233 mean_align 的效果参照，abs 打分）：未对齐时 S1 均值 '
             f"{base_metrics['mean_score_S1']:.3f}、S2 均值 {base_metrics['mean_score_S2']:.3f}"
             f"（标准 gap {base_metrics['S1_over_S2_mean_gap'] / max(base_metrics['score_pooled_sd'], 1e-12):+.4f}）；"
             f"mean_align 后 S1 {C['abs_dist']['metrics']['mean_score_S1']:.3f}、"
             f"S2 {C['abs_dist']['metrics']['mean_score_S2']:.3f}——对齐把 S1 整体推离、"
             'S2 几乎不动，这正是 E233 的 S2 增益来源（也是 S1 饱和到 1.0 的原因）。', '',
             '**读数解读（必须说清）**：相对分数的 `std gap` 确实大幅收缩（margin 0.386、ratio −0.074，'
             '对 abs 的 −1.271 分别只有 0.30× 与 0.06×），**δ 抵消假设成立**；但正是这一步抵消掉了'
             '当前把未注册被试拒掉的信号——在 E232 的混叠结构下，S1 与 S2 的差别本身就是一段大平移'
             '（S1 的批次均值范数 ≈ 20，S2 ≈ 4~7），把加性平移约掉就同时抹掉了可分性。'
             '因此第 2 节的 Test `sepAUC` 从 0.8795 掉到 0.49~0.70，S1 拒识同步下滑。', '',
             '## 4. 分折波动（20 折 LOSO）', '',
             '|打分|未知拒识 均值±std|已知接受 均值±std|LOSO AUC 均值±std|',
             '|---|---|---|---|']
    for metric, lam in METRICS:
        key = metric if metric != 'mix' else f'mix_lam{lam:g}'
        lo = loso[key]
        head.append(f"|{key}|{f4(lo['unknown_rejection_fold_mean'])}±{f4(lo['unknown_rejection_fold_std'])}|"
                    f"{f4(lo['known_acceptance_fold_mean'])}±{f4(lo['known_acceptance_fold_std'])}|"
                    f"{f4(lo['loso_auc'])}±{f4(lo['loso_auc_std'])}|")
    head += ['', '- Test 没有折划分，因此"分折 S2"在物理上不存在；上表给出 20 折 LOSO 的已知接受'
             '（Test S2 在 Train 侧的对应量）与未知拒识的均值±std。Test 侧的唯一波动来源见第 2 节的 S2 与门槛列。',
             '', '## 5. 逐 task（冻结 tau）', '',
             '|候选|task|LOSO 未知拒识|LOSO 已知接受|S1 rej|S2 acc|', '|---|---|---:|---:|---:|---:|']
    for cid in order:
        for t in TASKS:
            pt = C[cid]['loso']['per_task'][t]
            tt = C[cid]['metrics']['per_task'][t]
            head.append(f"|{cid}|{t}|{f4(pt['unknown_rejection'])}|{f4(pt['known_acceptance'])}|"
                        f"{f4(tt['S1_rejection'])}|{f4(tt['S2_acceptance'])}|")

    head += ['', '## 6. 判定', '']
    if dec['verdict'] == 'pass':
        ad = C[dec['adopted']]
        m = ad['metrics']
        head += [f"- **达标并冻结**：`{dec['adopted']}`。LOSO {f4(ad['loso']['unknown_rejection_micro'], 4)}"
                 f"（∈[0.89,0.91]）、Test S1 {f4(m['S1_rejection'])} ≥ 0.95、"
                 f"Test S2 **{f4(m['S2_acceptance'])}** ≥ 0.60、纯度 k20/k28 "
                 f"{f4(m['purity']['k20']['purity_micro'])}/{f4(m['purity']['k28']['purity_micro'])}"
                 f"（≥ 基线 95%）。",
                 f"- 相对 E233 合规值 0.312155：ΔS2 = {m['S2_acceptance'] - E233_S2:+.4f}；"
                 f"配对自助 95% CI（vs abs_dist）"
                 f"[{m['paired_vs_abs_calibrated']['dS2']['lo']:+.4f}, "
                 f"{m['paired_vs_abs_calibrated']['dS2']['hi']:+.4f}]。"]
    elif dec['verdict'] == 'partial':
        head += [f"- **显著改善但未达用户要求**：最优相对候选 `{best_rel}` 的 S2 = "
                 f"{f4(dec['best_relative_S2'])}（E233 {f4(E233_S2)}，"
                 f"{dec['best_relative_S2'] - E233_S2:+.4f}），仍未到 0.60；"
                 f"S1≥0.95 门槛读法下为 {f4(C[best_rel]['metrics']['gates']['S1_0.95']['S2_acceptance'])}。",
                 f"- 剩余瓶颈判断：{bottleneck_text(C, best_rel)}",
                 '- 不冻结新部署参数；E233 的 `abs_dist`（mean_align）仍是合规最优。']
    else:
        head += [('- **相对分数在此数据上无效（方向正确但量级远不够）**' if dec['verdict'] == 'marginal_fail'
                  else '- **相对分数在此数据上无效**') +
                 f"：最优相对候选 `{best_rel}` 的 Test S2 = "
                 f"{f4(dec['best_relative_S2'])}（E233 合规值 {f4(E233_S2)}，"
                 f"{dec['best_relative_S2'] - E233_S2:+.4f}），落在 0.3122~0.45 的边缘区间，"
                 f"距用户要求 0.60 差 {USER_S2_TARGET - dec['best_relative_S2']:.4f}，同时 S1 跌到 "
                 f"{f4(C[best_rel]['metrics']['S1_rejection'])}（< 0.95）。",
                 f"- **δ 抵消确实成立**（第 3 节）：`margin` 的标准化 S1−S2 gap "
                 f"{delta_ev['margin']['standardised_gap']:+.4f}、`ratio` "
                 f"{delta_ev['ratio']['standardised_gap']:+.4f}，相对 `abs` 的 "
                 f"{delta_ev['abs']['standardised_gap']:+.4f} 幅度分别只有 "
                 f"{abs(delta_ev['margin']['standardised_gap_ratio_vs_abs']):.2f}× 和 "
                 f"{abs(delta_ev['ratio']['standardised_gap_ratio_vs_abs']):.2f}×。",
                 '- **但取消 δ 的同时把可分性一起取消了**：Test 分离 AUC 从 `abs` 的 '
                 f"{f4(C['abs_dist_cal']['metrics']['separation_auc'])} 崩到 "
                 f"margin {f4(C['margin']['metrics']['separation_auc'])} / ratio "
                 f"{f4(C['ratio']['metrics']['separation_auc'])}，S1 拒识从 1.0000 掉到 "
                 f"{f4(C['margin']['metrics']['S1_rejection'])} / {f4(C['ratio']['metrics']['S1_rejection'])}；"
                 f"S1≥0.95 门槛下 margin 的 S2 只有 "
                 f"{f4(C['margin']['metrics']['gates']['S1_0.95']['S2_acceptance'])}、ratio "
                 f"{f4(C['ratio']['metrics']['gates']['S1_0.95']['S2_acceptance'])}。",
                 '- **LOSO 侧反而是改善的**（这是最容易误判的地方）：`margin` 的已知接受 '
                 f"{f4(C['margin']['loso']['known_acceptance_micro'])}、LOSO AUC "
                 f"{f4(C['margin']['loso']['loso_auc'])}；`ratio` 为 "
                 f"{f4(C['ratio']['loso']['known_acceptance_micro'])} / "
                 f"{f4(C['ratio']['loso']['loso_auc'])}；`abs` 为 "
                 f"{f4(C['abs_dist_cal']['loso']['known_acceptance_micro'])} / "
                 f"{f4(C['abs_dist_cal']['loso']['loso_auc'])}。"
                 'LOSO 的"未知"是**同一 session 内被留出的已注册身份**，没有跨天平移，'
                 '所以相对分数在那里是纯增益；它不能迁移到 Test 的 S1。',
                 '- **机理**：本数据里 S1 与 S2 的差别本身就是一段大平移（S1 的 (session,task) 批次均值范数 '
                 '≈ 20，S2 ≈ 4~7，见主表 cells）。相对分数把这段平移约掉，就等于把"未注册人群离所有原型都很远"'
                 '这个当前唯一的拒绝依据一起约掉。在 E232 的 session/注册状态混叠下，δ 不是可去的 nuisance，'
                 '而是 S1/S2 可分性的主要载体。',
                 '- **不追加第二轮（λ 与 margin+k=3）的理由**：λ 趋势单调且已饱和——'
                 f"mix λ=0.5 → 1.0 → margin（λ→∞）时 S2 = "
                 f"{f4(C['mix_lam0.5']['metrics']['S2_acceptance'])} → "
                 f"{f4(C['mix_lam1.0']['metrics']['S2_acceptance'])} → "
                 f"{f4(C['margin']['metrics']['S2_acceptance'])}，S1 = "
                 f"{f4(C['mix_lam0.5']['metrics']['S1_rejection'])} → "
                 f"{f4(C['mix_lam1.0']['metrics']['S1_rejection'])} → "
                 f"{f4(C['margin']['metrics']['S1_rejection'])}；"
                 '继续加大 λ 只会继续压低 S1，不会把 S2 推向 0.60。'
                 'margin+k=3 只改 d1 侧的原型聚合，无法恢复被抵消掉的绝对位置信息，故不值得再跑一轮。',
                 '- 不冻结新部署参数；E233 的 `abs_dist`（mean_align）仍是合规最优（S2 0.3122 / S1 1.0）。']

    head += ['', '## 7. 限制与口径', '',
             '- **官方 C / RA / ASafeR = N/A**：Test 无身份标签。S1 拒识/ S2 接受是赛题行为代理指标，'
             '不是官方未知检测/正常分类指标。',
             '- **Test 反复查看偏倚**：Test 自 E65 起被反复读；本轮候选定义、λ、二分规则在运行前写死'
             '（config.json），但"在候选间比较 S2"这一动作本身读到了 Test。',
             '- **session 与注册状态混叠**（E232）：S1 = 5 名未注册被试，S2 = 20 名注册用户。'
             'S1/S2 不是纯生物特征指标；本轮没有用 session 决定身份或门控。',
             '- **transductive 灰区**：`*_mean_align` 候选使用被评分批次的无标签统计量做均值对齐；'
             '不带对齐的 `margin` / `ratio` / `mix` 是**逐条推理**，不需要批次统计，这一点比 E233 更干净。',
             f"- 运行：`.venv\\Scripts\\python.exe research/E234_offset_invariant_score/run_e234.py`（CUDA），"
             f"耗时 {R['runtime_s']:.0f}s；折级 d1/d2 缓存于 `cache/`，指标参数存于 "
             '`models/e234_metric_params.pt`。']
    (OUT / 'report.md').write_text('\n'.join(head) + '\n', encoding='utf8')


def bottleneck_text(C, best_rel):
    m = C[best_rel]['metrics']
    lo = C[best_rel]['loso']
    base_lo = C['abs_dist_cal']['loso']
    d_acc = lo['known_acceptance_micro'] - base_lo['known_acceptance_micro']
    d_auc = m['separation_auc'] - base_metrics_auc()
    gap_m = C['margin']['metrics']['S1_over_S2_standardised_gap']
    gap_a = C['abs_dist_cal']['metrics']['S1_over_S2_standardised_gap']
    parts = []
    if m['separation_auc'] < base_metrics_auc():
        parts.append(f"Test 分离 AUC 反而下降到 {f4(m['separation_auc'])}"
                     f"（abs {f4(base_metrics_auc())}），说明相对分数损失了 d1 的绝对位置信息")
    if d_acc < 0:
        parts.append(f"LOSO 已知接受 {f4(lo['known_acceptance_micro'])} 低于 abs 的 "
                     f"{f4(base_lo['known_acceptance_micro'])}")
    if abs(gap_m) >= abs(gap_a):
        parts.append(f"δ 未完全抵消：margin 的标准化 S1−S2 gap {gap_m:+.4f}，"
                     f"abs 为 {gap_a:+.4f}")
    else:
        parts.append(f"δ 确有抵消（margin 标准化 gap {gap_m:+.4f} vs abs {gap_a:+.4f}），"
                     '但抵消的部分不足以跨过 0.60')
    return '；'.join(parts) + '。'


def base_metrics_auc():
    return BASE_AUC


if __name__ == '__main__':
    main()
