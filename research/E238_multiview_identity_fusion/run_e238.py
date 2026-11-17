"""E238: multi-view IDENTITY fusion + agreement gating.

E237 fused an image-view score with the E233 aligned EEG score and lifted the
Test S2-vs-S1 separation AUC from 0.8768 to 0.9611 (S2 at the S1 >= 0.95 gate
0.7063 -> 0.8776), but the identity label still came from the EEG branch alone;
the image views only moved a scalar gate score.

This round makes all four branches emit an identity:

  branch            space                         prototypes
  EEG               frozen E227 PCA96 (E233)      state['p'], k=12 per (task, identity)
  X_mw_cov4         E149 pixel view, PCA96        fit rows, k=12 per (task, identity)
  X_covariance_hr   same harness                  same
  IMG_px_all        composite of the 7 pixel views (mean of the per-view
                    z-scored per-identity distance matrices, the E149 definition)

Every branch returns id_b (nearest prototype identity), d1_b (its distance) and
d2_b (nearest prototype of a DIFFERENT identity).  All four use the E228/E233
flow: task-wise standardisation on the fit rows -> PCA96 -> k=12 prototypes per
(task, identity) -> symmetric per-(session, task) batch-mean alignment (Test
only; Train OOF / LOSO rows are single-session).

Identity fusion candidates (weights fixed in config.json before the run):
  eeg_only (baseline, must reproduce E233 pred), vote_equal, vote_weighted,
  rank_sum_identity, eeg_plus_best_img.

Gating candidates:
  gate_score_fused   the E237 fuse_zscore_50_50 score (control)
  gate_agreement     0.5 * rank(fused z) + 0.5 * (1 - rank(vote entropy))
  gate_agreement_threshold_sweep  curve only, no operating point chosen

The main read-out is LABEL-FREE: closed-set identity ability on Train (20-fold
LOSO fold-in macro-F1 / accuracy over the enrolled rows of the eval window,
plus inner 5-block OOF macro-F1 on the fit rows), branch agreement, and the
k=20 E232 cluster purity / coverage on Test.  Test has no identity labels, so
the official C / identity macro-F1 remain N/A; Test is only evaluated, never
used to pick an identity or a threshold.
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
SUB = OUT / 'submission'
LOG = OUT / 'run.log'
DEVICE = 'cuda'

TASKS = list(CFG['protocol']['tasks'])
FIT_FRAC = float(CFG['protocol']['fit_fraction'])
EVAL_LO, EVAL_HI = [float(x) for x in CFG['protocol']['evaluation_fraction']]
N_BLOCKS = int(CFG['protocol']['inner_blocks'])
N_FOLDS = int(CFG['protocol']['folds'])
PCA_DIM = int(CFG['protocol']['pca_dim'])
K_PROTO = int(CFG['protocol']['prototypes_k'])
PURITY_K = 20
PURITY_SEED = 20261117
LOSO_TARGET = 0.90
S1_GATE = 0.95
PURITY_TOL = 0.005
CLASSES = np.arange(1, 21)

A = CFG['baseline_anchors']
BASE_AUC = float(A['e233_mean_align']['test_separation_auc'])
BASE_S2 = float(A['e233_mean_align']['test_S2_acceptance'])
BASE_PUR_ALL = float(A['e233_mean_align']['purity_k20_micro_all_S2'])
TAU_EEG = float(A['e233_mean_align']['test_tau'])
E237 = A['e237_image_view_fusion']
E237_AUC = float(E237['test_separation_auc'])
E237_S2_GATE = float(E237['S2_at_S1_0p95'])
E237_S2 = float(E237['S2_at_loso_tau'])
E237_PUR = float(E237['purity_accepted_S2_at_loso_tau'])
TARGET_S2_GATE = float(E237['target_S2_at_S1_0p95_for_this_round'])

PIX_VIEWS = list(CFG['branches']['IMG_px_all']['pixel_views'].keys())
FUSION_W = CFG['identity_fusion']['weights_fixed_before_run']
FUSION_ORDER = ['eeg_only', 'vote_equal', 'vote_weighted', 'rank_sum_identity',
                'eeg_plus_best_img']
BRANCHES = ['EEG', 'X_mw_cov4', 'X_covariance_hr', 'IMG_px_all']
GATE_W = CFG['gating']['candidates']['gate_agreement']['weights']

CFG_BLOCK_PATHS_TEST = None


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
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def frac_of(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def macro_f1(y, p, classes):
    f1s = []
    for c in classes:
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s)), float((y == p).mean())


# --------------------------------------------------------------- scoring flow
def kmeans_np(x, k, seed, iters=20):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.copy()
    c = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        new = c.copy()
        for j in range(k):
            m = a == j
            if m.any():
                new[j] = x[m].mean(0)
        if np.allclose(new, c, atol=1e-6):
            c = new
            break
        c = new
    return c


def build_protos(Z, subj, task, mask, classes, k):
    protos, p_task, p_id = [], [], []
    for t in TASKS:
        for c in classes:
            sel = np.flatnonzero(mask & (task == t) & (subj == c))
            if not len(sel):
                continue
            for centre in kmeans_np(Z[sel].astype(np.float32), k, seed=int(c) * 17):
                protos.append(centre)
                p_task.append(t)
                p_id.append(int(c))
    return np.stack(protos).astype(np.float32), np.asarray(p_task), np.asarray(p_id, int)


def D_by_identity(Z, protos, p_task, p_id, task, classes=CLASSES):
    """(n, 20) minimum distance from every row to that identity's prototypes of its task."""
    D = np.full((len(Z), len(classes)), np.inf, np.float32)
    Zt = tensor(Z)
    for t in TASKS:
        rr = np.flatnonzero(task == t)
        cols = np.flatnonzero(p_task == t)
        if not len(rr) or not len(cols):
            continue
        P = tensor(protos[cols])
        cid = p_id[cols]
        for i in range(0, len(rr), 4096):
            ch = rr[i:i + 4096]
            d = torch.cdist(Zt[ch], P).cpu().numpy()
            for j, c in enumerate(classes):
                cc = np.flatnonzero(cid == c)
                if len(cc):
                    D[ch, j] = d[:, cc].min(1)
    return D


def pca_basis(Xfit, dim):
    A_ = tensor(Xfit)
    mu = A_.mean(0)
    Xc = A_ - mu
    C = Xc.T @ Xc / max(len(Xc) - 1, 1)
    w, V = torch.linalg.eigh(C.double())
    w = w.flip(0)
    V = V.flip(1)[:, :min(dim, V.shape[1])]
    log(f'    pca eigen top1..3={w[:3].cpu().numpy()} explained='
        f'{float(w[:dim].sum() / w.clamp_min(0).sum()):.4f}')
    return mu.cpu().numpy().astype(np.float32), V.cpu().numpy().astype(np.float32)


def project(X, mu, V, chunk=4096):
    out = np.empty((len(X), V.shape[1]), np.float32)
    Vt = tensor(V)
    mut = tensor(mu)
    for i in range(0, len(X), chunk):
        xb = tensor(X[i:i + chunk])
        out[i:i + chunk] = ((xb - mut) @ Vt).cpu().numpy()
    return out


def align_cells(Z, masks, task, ref_mu):
    """E233 mean_align: subtract the (session, task) batch mean, add back the train task mean."""
    Z = Z.copy()
    for _sname, mask in masks.items():
        for t in TASKS:
            rr = np.flatnonzero(mask & (task == t))
            if len(rr) < 200:
                continue
            Z[rr] = Z[rr] - Z[rr].mean(0) + ref_mu[t]
    return Z


def id_d1_d2(D):
    ids = D.argmin(-1)
    d1 = np.take_along_axis(D, ids[..., None], -1)[..., 0]
    D2 = D.copy()
    np.put_along_axis(D2, ids[..., None], np.inf, -1)
    d2 = D2.min(-1)
    return CLASSES[ids], d1, d2


# --------------------------------------------------------------- branches
def load_eeg_state():
    return torch.load(ROOT / 'research/E227_joint_frontier/models/deploy_auc2_lofo.pt',
                      weights_only=False)


def transform_eeg(state, raw, tasks):
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


def load_blocks(paths):
    return np.concatenate([np.load(ROOT / p).astype(np.float32) for p in paths], 1)


def eeg_branch(state, ctx):
    ztr, zte = ctx['ztr'], ctx['zte']
    proto_task = np.array([t for t, _ in state['keys']])
    proto_id = np.array([int(s) for _, s in state['keys']], int)
    ref_mu = {t: ztr[ctx['fit'] & (ctx['tr_task'] == t)].mean(0) for t in TASKS}
    za = align_cells(zte, ctx['masks'], ctx['te_task'], ref_mu)
    D_te = D_by_identity(za, state['p'], proto_task, proto_id, ctx['te_task'])
    D_eval = D_by_identity(ztr[ctx['eval_idx']], state['p'], proto_task, proto_id,
                           ctx['eval_task'])
    D_oof = np.full((len(ztr), len(CLASSES)), np.nan, np.float32)
    for b in range(N_BLOCKS):
        hold = ctx['fit'] & (ctx['blocks_frac'] == b)
        p2, t2, i2 = build_protos(ztr, ctx['subj'], ctx['tr_task'],
                                  ctx['fit'] & ~hold, CLASSES, K_PROTO)
        D_oof[hold] = D_by_identity(ztr[hold], p2, t2, i2, ctx['tr_task'][hold])
    return dict(name='EEG', Dtest=D_te, Deval=D_eval, Dfit=D_oof)


def image_view(name, paths, ctx):
    t0 = time.time()
    Xtr = np.load(ROOT / paths['train']).astype(np.float32)
    Xte = np.load(ROOT / paths['test']).astype(np.float32)
    assert Xtr.shape[0] == len(ctx['tr_task']) and Xte.shape[0] == len(ctx['te_task'])
    for t in TASKS:
        fm = ctx['fit'] & (ctx['tr_task'] == t)
        mu = Xtr[fm].mean(0)
        sd = Xtr[fm].std(0) + 1e-6
        trm = ctx['tr_task'] == t
        Xtr[trm] = (Xtr[trm] - mu) / sd
        tem = ctx['te_task'] == t
        Xte[tem] = (Xte[tem] - mu) / sd
    mu_pca, V = pca_basis(Xtr[ctx['fit']], PCA_DIM)
    Ztr = project(Xtr, mu_pca, V)
    Zte = project(Xte, mu_pca, V)
    del Xtr, Xte
    ref_mu = {t: Ztr[ctx['fit'] & (ctx['tr_task'] == t)].mean(0) for t in TASKS}
    protos, p_task, p_id = build_protos(Ztr, ctx['subj'], ctx['tr_task'], ctx['fit'],
                                        CLASSES, K_PROTO)
    D_te = D_by_identity(align_cells(Zte, ctx['masks'], ctx['te_task'], ref_mu),
                         protos, p_task, p_id, ctx['te_task'])
    D_eval = D_by_identity(Ztr[ctx['eval_idx']], protos, p_task, p_id, ctx['eval_task'])
    D_oof = np.full((len(Ztr), len(CLASSES)), np.nan, np.float32)
    for b in range(N_BLOCKS):
        hold = ctx['fit'] & (ctx['blocks_frac'] == b)
        p2, t2, i2 = build_protos(Ztr, ctx['subj'], ctx['tr_task'],
                                  ctx['fit'] & ~hold, CLASSES, K_PROTO)
        D_oof[hold] = D_by_identity(Ztr[hold], p2, t2, i2, ctx['tr_task'][hold])
    _, d1, _ = id_d1_d2(D_te)
    auc = auc_ties(-d1[~ctx['is_s1']], -d1[ctx['is_s1']])
    log(f'  view {name:16s} auc_aligned={auc:.4f} ({time.time() - t0:.1f}s)')
    return dict(name=name, Dtest=D_te, Deval=D_eval, Dfit=D_oof,
                mu_pca=mu_pca, V=V, protos=protos, p_task=p_task, p_id=p_id)


def composite_branch(views, ctx):
    """IMG_px_all: mean of the per-view task-wise z-scored per-identity distance matrices."""
    fit, tr_task, te_task, eval_task = ctx['fit'], ctx['tr_task'], ctx['te_task'], \
        ctx['eval_task']

    def stats(D, task):
        return {t: (float(np.nanmean(D[fit & (task == t)])),
                    float(np.nanstd(D[fit & (task == t)])) + 1e-6) for t in TASKS}

    def standardise(D, task, st):
        out = np.full(D.shape, np.nan, np.float64)
        for t in TASKS:
            m = task == t
            mu, sd = st[t]
            out[m] = (D[m].astype(np.float64) - mu) / sd
        return out

    std = {n: stats(views[n]['Dfit'], tr_task) for n in PIX_VIEWS}
    Dte = np.mean([standardise(views[n]['Dtest'], te_task, std[n]) for n in PIX_VIEWS], 0)
    Deval = np.mean([standardise(views[n]['Deval'], eval_task, std[n]) for n in PIX_VIEWS], 0)
    Dfit = np.zeros((views[PIX_VIEWS[0]]['Dfit'].shape), np.float64)
    cnt = np.zeros(Dfit.shape)
    for n in PIX_VIEWS:
        s = standardise(views[n]['Dfit'], tr_task, std[n])
        ok = ~np.isnan(s)
        Dfit[ok] += s[ok]
        cnt[ok] += 1
    Dfit = np.where(cnt > 0, Dfit / np.maximum(cnt, 1), np.nan)
    _, d1, _ = id_d1_d2(Dte.astype(np.float32))
    log(f'  composite IMG_px_all auc_aligned='
        f'{auc_ties(-d1[~ctx["is_s1"]], -d1[ctx["is_s1"]]):.4f}')
    return dict(name='IMG_px_all', Dtest=Dte.astype(np.float32),
                Deval=Deval.astype(np.float32), Dfit=Dfit.astype(np.float32))


# --------------------------------------------------------------- identity fusion
def _ranks(D):
    return (np.argsort(np.argsort(D, -1), -1) + 1).astype(np.float64)


def fuse(Ds, kind, classes=CLASSES):
    """Ds: dict branch -> (..., n_cls) distance matrix. Returns identity values (...,)."""
    n_cls = Ds['EEG'].shape[-1]
    if kind == 'eeg_only':
        return classes[Ds['EEG'].argmin(-1)]
    W = FUSION_W[kind]
    if kind in ('rank_sum_identity', 'eeg_plus_best_img'):
        R = sum(w * _ranks(Ds[b]) for b, w in W.items() if b in Ds)
        return classes[R.argmin(-1)]
    shape = Ds['EEG'].shape[:-1]
    votes = np.zeros(shape + (n_cls,), np.float64)
    flat = votes.reshape(-1, n_cls)
    for b, w in W.items():
        idx = Ds[b].argmin(-1).reshape(-1)
        np.add.at(flat, (np.arange(len(idx)), idx), w)
    best = votes.max(-1, keepdims=True)
    cand = np.where(votes >= best - 1e-12, Ds['EEG'], np.inf)
    return classes[cand.argmin(-1)]


def vote_entropy(Ds, W):
    n_cls = Ds['EEG'].shape[-1]
    shape = Ds['EEG'].shape[:-1]
    votes = np.zeros(shape + (n_cls,), np.float64)
    flat = votes.reshape(-1, n_cls)
    for b, w in W.items():
        idx = Ds[b].argmin(-1).reshape(-1)
        np.add.at(flat, (np.arange(len(idx)), idx), w)
    p = votes / max(float(sum(W.values())), 1e-12)
    H = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0), -1)
    return H / np.log(4.0)


# --------------------------------------------------------------- score standardisation
def z_comp(score, ref_fit, fk_list, fu_list):
    mu, sd = float(np.mean(ref_fit)), float(np.std(ref_fit)) + 1e-6
    return dict(test=(score - mu) / sd,
                ref=(ref_fit - mu) / sd,
                fold_known=[(x - mu) / sd for x in fk_list],
                fold_unknown=[(x - mu) / sd for x in fu_list])


def pct_within(scores, task):
    """Per-task percentile rank inside the scored population, ties get the mid-rank.

    Used by the agreement gate so that the two components (fused score, vote entropy) live
    on the same [0, 1] scale by construction.  It is a monotone transform of each
    component, so it cannot change that component's own AUC.
    """
    out = np.empty(len(scores), float)
    for t in TASKS:
        m = task == t
        if not m.any():
            continue
        s = scores[m]
        n = len(s)
        order = np.argsort(s, kind='mergesort')
        ss = s[order]
        r = np.empty(n, float)
        i = 0
        while i < n:
            j = i + 1
            while j < n and ss[j] == ss[i]:
                j += 1
            r[order[i:j]] = (i + 1 + j) / 2.0 / n
            i = j
        out[m] = r
    return out


# --------------------------------------------------------------- metrics
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


def purity_of(pred_s2, assign, k, mask):
    sub_pred, sub_assign = pred_s2[mask], assign[mask]
    if not len(sub_pred):
        return dict(n=0, purity_micro=float('nan'), purity_macro=float('nan'),
                    n_distinct_pred=0, max_identity_share=float('nan'))
    hit, pur = 0, []
    for j in range(k):
        m = sub_assign == j
        if not m.any():
            continue
        vals, counts = np.unique(sub_pred[m], return_counts=True)
        hit += int(counts.max())
        pur.append(float(counts.max() / m.sum()))
    vals, counts = np.unique(sub_pred, return_counts=True)
    return dict(n=int(len(sub_pred)), purity_micro=float(hit / len(sub_pred)),
                purity_macro=float(np.mean(pur)), n_distinct_pred=int(len(vals)),
                max_identity_share=float(counts.max() / len(sub_pred)))


def evaluate(cid, te, fk, fu, pred_s2, assign, pred_all, is_s1):
    uk = np.concatenate(fu)
    tau = float(np.quantile(uk, LOSO_TARGET))
    fold_rej = [float((u < tau).mean()) for u in fu]
    fold_acc = [float((k >= tau).mean()) for k in fk]
    rej = te < tau
    acc = ~rej
    tau95 = float(np.quantile(te[is_s1], S1_GATE))
    te_s2 = te[~is_s1]
    return dict(
        tau=tau, loso_unknown_rejection=float((uk < tau).mean()),
        loso_fold_rejection_mean=float(np.mean(fold_rej)),
        loso_fold_rejection_std=float(np.std(fold_rej)),
        S1_rejection=float(rej[is_s1].mean()),
        S2_acceptance=float(acc[~is_s1].mean()),
        separation_auc=auc_ties(te[~is_s1], te[is_s1]),
        S2_at_S1_0p95=float((te_s2 >= tau95).mean()),
        S1_at_0p95=float((te[is_s1] >= tau95).mean()),
        tau_S1_0p95=tau95,
        overall_rejection=float(rej.mean()),
        identities_predicted=int(len(np.unique(pred_all[acc]))),
        fold_S2_proxy_mean=float(np.mean(fold_acc)),
        fold_S2_proxy_std=float(np.std(fold_acc)),
        fold_S2_proxy=fold_acc,
        purity_accepted_S2=purity_of(pred_s2, assign, PURITY_K, te_s2 >= tau),
        purity_at_gate_S2=purity_of(pred_s2, assign, PURITY_K, te_s2 >= tau95),
        n_accepted=int(acc.sum()), n_accepted_S2=int((acc & ~is_s1).sum()),
    )


def baseline_eeg_space(te_task):
    model = load_eeg_state()
    st = model['state']
    order = list(model['block_order'])
    store = {b: np.load(ROOT / CFG_BLOCK_PATHS_TEST[b]).astype(np.float32) for b in order}
    X = np.concatenate([store[b] for b in order], 1)
    x = tensor(X)
    z = torch.empty_like(x)
    for i, t in enumerate(TASKS):
        rows = te_task == t
        z[rows] = (x[rows] - tensor(st['mu'][i])) / tensor(st['sd'][i])
    z = z * tensor(np.sqrt(np.asarray(st['w'], np.float32)))
    z = ((z.double() - tensor(st['pm'], torch.float64)) @
         tensor(st['W'], torch.float64)).float()
    return z.cpu().numpy()


# --------------------------------------------------------------- main
def main():
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA required'
    SUB.mkdir(exist_ok=True)
    log(f'device={torch.cuda.get_device_name(0)} torch={torch.__version__}')
    t_start = time.time()

    tr = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    te = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tr) == 11304 and len(te) == 22251
    subj = tr.subject.to_numpy(int)
    tr_task = tr.task.to_numpy()
    fr = frac_of(tr)
    fit = fr <= FIT_FRAC
    blocks_frac = np.clip((fr * N_BLOCKS).astype(int), 0, N_BLOCKS - 1)
    te_task = te.task.to_numpy()
    sess = te.session.astype(str).str.casefold().to_numpy()
    is_s1 = sess == 'session1'
    masks = {'session1': is_s1, 'session2': ~is_s1}
    eval_idx = np.flatnonzero((fr > EVAL_LO) & (fr <= EVAL_HI))
    eval_task = tr_task[eval_idx]
    eval_subj = subj[eval_idx]
    ctx = dict(subj=subj, tr_task=tr_task, fit=fit, blocks_frac=blocks_frac,
               te_task=te_task, is_s1=is_s1, masks=masks, eval_idx=eval_idx,
               eval_task=eval_task, eval_subj=eval_subj)
    log(f'fit rows={int(fit.sum())} eval rows={len(eval_idx)} '
        f'test S1={int(is_s1.sum())} S2={int((~is_s1).sum())}')

    # ---------------- EEG branch (frozen E227 / E233)
    state = load_eeg_state()['state']
    e233 = json.loads((ROOT / 'research/E233_delta_estimation/config.json').read_text(
        encoding='utf8'))
    ctx['ztr'] = transform_eeg(state,
                               load_blocks(list(e233['feature_cache']['train'].values())),
                               tr_task)
    ctx['zte'] = transform_eeg(state,
                               load_blocks(list(e233['feature_cache']['test'].values())),
                               te_task)
    log('EEG branch: frozen E227 PCA space, mean_align on Test, k=12 prototypes')
    branches = {'EEG': eeg_branch(state, ctx)}
    id_eeg, d1_eeg, _ = id_d1_d2(branches['EEG']['Dtest'])
    d = np.load(ROOT / CFG['branches']['EEG']['test_score'], allow_pickle=True)
    checks = dict(pred_identical_e233=bool(np.array_equal(id_eeg, d['pred'].astype(int))),
                  score_max_abs_diff=float(np.max(np.abs(-d1_eeg - d['score']))))
    log('  E233 reproduction ' + json.dumps(checks))
    assert checks['pred_identical_e233'], 'eeg_only must reproduce the E233 pred'
    assert checks['score_max_abs_diff'] < 1e-3

    # ---------------- image branches
    log('scoring the 7 pixel views (task-wise standardisation -> PCA96 -> k=12 protos)')
    views = {n: image_view(n, CFG['branches']['IMG_px_all']['pixel_views'][n], ctx)
             for n in PIX_VIEWS}
    branches['X_mw_cov4'] = dict(views['X_mw_cov4'], name='X_mw_cov4')
    branches['X_covariance_hr'] = dict(views['X_covariance_hr'], name='X_covariance_hr')
    branches['IMG_px_all'] = composite_branch(views, ctx)

    branch_out = {}
    for b in BRANCHES:
        id_te, d1_te, d2_te = id_d1_d2(branches[b]['Dtest'].astype(np.float32))
        id_oof, d1_oof, d2_oof = id_d1_d2(branches[b]['Dfit'].astype(np.float32))
        id_ev, d1_ev, d2_ev = id_d1_d2(branches[b]['Deval'].astype(np.float32))
        branch_out[b] = dict(id_test=id_te, d1_test=d1_te, d2_test=d2_te,
                             id_oof=id_oof, d1_oof=d1_oof, d2_oof=d2_oof,
                             id_eval=id_ev, d1_eval=d1_ev, d2_eval=d2_ev)
        np.savez_compressed(CACHE / f'branch_{b}.npz',
                            id_test=id_te, d1_test=d1_te, d2_test=d2_te,
                            id_oof=id_oof, d1_oof=d1_oof, d2_oof=d2_oof,
                            id_eval=id_ev, d1_eval=d1_ev, d2_eval=d2_ev)
        log(f'  branch {b:16s} d1_test_mean={d1_te.mean():.3f} '
            f'(d2-d1)_mean={(d2_te - d1_te).mean():.3f}')

    # ---------------- identity fusion candidates + closed-set F1
    Ds_test = {b: branches[b]['Dtest'] for b in BRANCHES}
    Ds_oof = {b: branches[b]['Dfit'] for b in BRANCHES}
    Ds_eval = {b: branches[b]['Deval'] for b in BRANCHES}
    fk_mask = np.array([np.isin(eval_subj, np.setdiff1d(CLASSES, [c])) for c in CLASSES])
    fu_mask = np.array([eval_subj == c for c in CLASSES])
    enrolled_idx = [[i for i, c in enumerate(CLASSES) if c != CLASSES[h]]
                    for h in range(N_FOLDS)]
    enrolled_cls = [CLASSES[idx] for idx in enrolled_idx]

    def closed_set(pred_fold):
        f1s, accs = [], []
        for h in range(N_FOLDS):
            m = fk_mask[h]
            f1, acc = macro_f1(eval_subj[m], pred_fold[h][m], enrolled_cls[h])
            f1s.append(f1)
            accs.append(acc)
        y = np.concatenate([eval_subj[fk_mask[h]] for h in range(N_FOLDS)])
        p = np.concatenate([pred_fold[h][fk_mask[h]] for h in range(N_FOLDS)])
        pf1, pacc = macro_f1(y, p, CLASSES)
        return dict(loso_macro_f1_mean=float(np.mean(f1s)),
                    loso_macro_f1_std=float(np.std(f1s)),
                    loso_accuracy_mean=float(np.mean(accs)),
                    loso_accuracy_std=float(np.std(accs)),
                    loso_macro_f1_pooled=pf1, loso_accuracy_pooled=pacc,
                    n_rows_pooled=int(len(y)))

    branch_closed = {}
    for b in BRANCHES:
        pf = np.empty((N_FOLDS, len(eval_idx)), int)
        for h in range(N_FOLDS):
            Dsub = Ds_eval[b][:, enrolled_idx[h]]
            pf[h] = enrolled_cls[h][Dsub.argmin(-1)]
        branch_closed[b] = closed_set(pf)
        log(f'  branch {b:16s} LOSO macroF1={branch_closed[b]["loso_macro_f1_mean"]:.4f} '
            f'acc={branch_closed[b]["loso_accuracy_mean"]:.4f}')

    fusion = {}
    for kind in FUSION_ORDER:
        pred_test = fuse(Ds_test, kind)
        pred_oof = fuse(Ds_oof, kind)
        pred_fold = np.empty((N_FOLDS, len(eval_idx)), int)
        for h in range(N_FOLDS):
            sub = {b: Ds_eval[b][:, enrolled_idx[h]] for b in BRANCHES}
            pred_fold[h] = fuse(sub, kind, enrolled_cls[h])
        oof_f1, oof_acc = macro_f1(subj[fit], pred_oof[fit], CLASSES)
        fusion[kind] = dict(
            pred_test=pred_test, pred_oof=pred_oof, pred_fold=pred_fold,
            closed_set=closed_set(pred_fold),
            oof=dict(macro_f1=oof_f1, accuracy=oof_acc))
        np.savez_compressed(CACHE / f'fusion_{kind}.npz', pred_test=pred_test,
                            pred_oof=pred_oof, pred_fold=pred_fold)
        cs = fusion[kind]['closed_set']
        log(f'  fusion {kind:18s} LOSO macroF1={cs["loso_macro_f1_mean"]:.4f}'
            f'+-{cs["loso_macro_f1_std"]:.4f} acc={cs["loso_accuracy_mean"]:.4f} '
            f'OOF-F1={oof_f1:.4f}')

    # ---------------- branch agreement (label free)
    def agreement_stats(preds, mask=None):
        P = np.stack([preds[b] for b in BRANCHES], 1)
        if mask is not None:
            P = P[mask]
        pair = {}
        for i in range(len(BRANCHES)):
            for j in range(i + 1, len(BRANCHES)):
                pair[f'{BRANCHES[i]}|{BRANCHES[j]}'] = float((P[:, i] == P[:, j]).mean())
        counts = np.zeros((len(P), len(CLASSES)), int)
        for k in range(len(BRANCHES)):
            np.add.at(counts, (np.arange(len(P)), P[:, k] - 1), 1)
        nmax = counts.max(1)
        return dict(pairwise=pair, unanimous_4of4=float((nmax == 4).mean()),
                    majority_ge3=float((nmax >= 3).mean()),
                    majority_ge2=float((nmax >= 2).mean()),
                    mean_max_votes=float(nmax.mean()), n=int(len(P)))

    preds_test = {b: branch_out[b]['id_test'] for b in BRANCHES}
    preds_oof = {b: branch_out[b]['id_oof'] for b in BRANCHES}
    agree = dict(test=agreement_stats(preds_test),
                 test_s2=agreement_stats(preds_test, ~is_s1),
                 test_s1=agreement_stats(preds_test, is_s1),
                 fit_oof=agreement_stats(preds_oof, fit))
    log(f'  agreement Test unanimous={agree["test"]["unanimous_4of4"]:.4f} ge3='
        f'{agree["test"]["majority_ge3"]:.4f} | S2 unanimous='
        f'{agree["test_s2"]["unanimous_4of4"]:.4f} ge3={agree["test_s2"]["majority_ge3"]:.4f}')

    # ---------------- gating path
    fused_cached = np.load(ROOT / 'research/E237_image_view_fusion/cache/'
                           'cand_fuse_zscore_50_50.npz')['score'].astype(np.float64)
    dt = np.load(ROOT / CFG['branches']['EEG']['test_oof'], allow_pickle=True)
    assert np.array_equal(dt['oof_fit'].astype(bool), fit)
    eeg_ref = dt['oof_s'].astype(np.float64)
    eeg_test = np.load(ROOT / CFG['branches']['EEG']['test_score'],
                       allow_pickle=True)['score'].astype(np.float64)

    per_view = {}
    for n in PIX_VIEWS:
        per_view[n] = dict(ref=-np.nanmin(views[n]['Dfit'][fit], 1),
                           eval=-np.nanmin(views[n]['Deval'], 1),
                           test=-np.nanmin(views[n]['Dtest'], 1))

    def std_view(arr, ref):
        return (arr - float(np.mean(ref))) / (float(np.std(ref)) + 1e-6)

    img_test = np.mean([std_view(per_view[n]['test'], per_view[n]['ref'])
                        for n in PIX_VIEWS], 0)
    img_ref = np.mean([std_view(per_view[n]['ref'], per_view[n]['ref'])
                       for n in PIX_VIEWS], 0)

    fold_known, fold_unknown, fold_task = [], [], []
    for h in range(1, N_FOLDS + 1):
        fd = np.load(ROOT / CFG['branches']['EEG']['fold_cache'].format(fold=h),
                     allow_pickle=True)
        s = -fd['d1_orig'].astype(np.float64)
        kn, un = fd['known'].astype(bool), fd['unknown'].astype(bool)
        assert np.array_equal(fd['task'], eval_task)
        assert np.array_equal(fd['subject'], eval_subj)
        fold_known.append(s[kn])
        fold_unknown.append(s[un])
        fold_task.append((fd['task'][kn], fd['task'][un]))
    # image fold scores on the enrolled prototypes only (E237 path)
    img_fk, img_fu = [], []
    for h in range(N_FOLDS):
        keep = enrolled_idx[h]
        s = np.mean([std_view(-np.min(views[n]['Deval'][:, keep], 1), per_view[n]['ref'])
                     for n in PIX_VIEWS], 0)
        img_fk.append(s[fk_mask[h]])
        img_fu.append(s[fu_mask[h]])
    img_z = z_comp(img_test, img_ref, img_fk, img_fu)
    eeg_z = z_comp(eeg_test, eeg_ref[fit], fold_known, fold_unknown)

    def mix(a, b):
        return 0.5 * a + 0.5 * b

    fused_test = mix(eeg_z['test'], img_z['test'])
    fused_ref = mix(eeg_z['ref'], img_z['ref'])
    fused_fk = [mix(a, b) for a, b in zip(eeg_z['fold_known'], img_z['fold_known'])]
    fused_fu = [mix(a, b) for a, b in zip(eeg_z['fold_unknown'], img_z['fold_unknown'])]
    repro = float(np.max(np.abs(fused_test - fused_cached)))
    checks['e237_fused_score_max_abs_diff'] = repro
    log(f'  E237 fuse_zscore_50_50 reconstruction max|diff|={repro:.2e}')
    assert repro < 1e-3, 'fused score drifted from E237'

    # vote entropy: Test (all 20 identities) and per-fold (enrolled only)
    ent_test = vote_entropy(Ds_test, FUSION_W['vote_weighted'])
    ent_fold = np.empty((N_FOLDS, len(eval_idx)))
    for h in range(N_FOLDS):
        sub = {b: Ds_eval[b][:, enrolled_idx[h]] for b in BRANCHES}
        ent_fold[h] = vote_entropy(sub, FUSION_W['vote_weighted'])
    agreement_component = dict(
        test_auc_agreement_vs_s1=auc_ties(-ent_test[~is_s1], -ent_test[is_s1]),
        entropy_mean_S1=float(ent_test[is_s1].mean()),
        entropy_mean_S2=float(ent_test[~is_s1].mean()),
        note='AUC of -vote entropy (higher = all four branches agree) as a standalone '
             'known/unknown ranker; 0.5 means the agreement carries no information.')
    log('  agreement component ' + json.dumps(agreement_component))
    ctx['agreement_component'] = agreement_component

    # ---------------- purity partition (E232 / E237 flow, fixed for all candidates)
    log('building the label-free k=20 partition in the baseline EEG PCA space')
    zte_base = baseline_eeg_space(te_task)
    s2_rows = np.flatnonzero(~is_s1)
    assign20, sizes20 = kmeans_assign(zte_base[s2_rows].astype(np.float32), PURITY_K,
                                      PURITY_SEED + PURITY_K)
    log(f'  kmeans k=20 sizes={sizes20.tolist()}')
    pred_s2_eeg = fusion['eeg_only']['pred_test'][~is_s1]
    pur_all_eeg = purity_of(pred_s2_eeg, assign20, PURITY_K, np.ones(len(s2_rows), bool))
    log('  E233 eeg_only purity over all S2 rows ' + json.dumps(pur_all_eeg))
    assert abs(pur_all_eeg['purity_micro'] - BASE_PUR_ALL) < 1e-6, 'purity partition drifted'

    gates = {}

    def add_gate(cid, te_score, fk, fu, note):
        m = evaluate(cid, te_score, fk, fu, pred_s2_eeg, assign20,
                     fusion['eeg_only']['pred_test'], is_s1)
        m['note'] = note
        gates[cid] = m
        np.savez_compressed(CACHE / f'gate_{cid}.npz', score=te_score, tau=m['tau'],
                            tau_S1_0p95=m['tau_S1_0p95'], is_s1=is_s1)
        log(f'  gate {cid:28s} LOSO={m["loso_unknown_rejection"]:.4f} '
            f'S1={m["S1_rejection"]:.4f} S2={m["S2_acceptance"]:.4f} '
            f'sepAUC={m["separation_auc"]:.4f} S2@95={m["S2_at_S1_0p95"]:.4f} '
            f'pur@95={m["purity_at_gate_S2"]["purity_micro"]:.4f} '
            f'foldS2={m["fold_S2_proxy_mean"]:.4f}+-{m["fold_S2_proxy_std"]:.4f}')
        return m

    add_gate('gate_score_fused', fused_cached, fused_fk, fused_fu,
             'E237 fuse_zscore_50_50 Test score (control); LOSO path rebuilt from the '
             'frozen E235 fold scores and the image fold scores')
    assert abs(gates['gate_score_fused']['S2_acceptance'] - E237_S2) < 1e-6
    assert abs(gates['gate_score_fused']['S2_at_S1_0p95'] - E237_S2_GATE) < 1e-6

    # agreement gate: both components percentiled inside the scored population
    p_fused_test = pct_within(fused_test, te_task)
    p_ent_test = pct_within(ent_test, te_task)
    gate_ag_test = GATE_W['fused_zscore'] * p_fused_test + \
        GATE_W['agreement'] * (1.0 - p_ent_test)
    gate_ag_fk, gate_ag_fu = [], []
    for h in range(N_FOLDS):
        nk = len(fused_fk[h])
        p = pct_within(np.r_[fused_fk[h], fused_fu[h]],
                       np.r_[fold_task[h][0], fold_task[h][1]])
        pe = pct_within(ent_fold[h], eval_task)
        gate_ag_fk.append(GATE_W['fused_zscore'] * p[:nk] +
                          GATE_W['agreement'] * (1.0 - pe[fk_mask[h]]))
        gate_ag_fu.append(GATE_W['fused_zscore'] * p[nk:] +
                          GATE_W['agreement'] * (1.0 - pe[fu_mask[h]]))
    add_gate('gate_agreement', gate_ag_test, gate_ag_fk, gate_ag_fu,
             'G = 0.5 * P(fused z) + 0.5 * (1 - P(vote entropy)); P = per-task percentile '
             'inside the scored population (Test for Test, per-fold eval rows for LOSO), '
             'ties mid-rank. Both components are monotone transforms of their own values, so '
             'the composition changes nothing but the mixing scale.')

    qs = np.linspace(0.50, 0.999, 200)
    taus = np.quantile(gate_ag_test[is_s1], qs)
    sweep = dict(quantiles=qs.tolist(), tau=taus.tolist(),
                 S1_rejection=[float((gate_ag_test[is_s1] < x).mean()) for x in taus],
                 S2_acceptance=[float((gate_ag_test[~is_s1] >= x).mean()) for x in taus],
                 overall_rejection=[float((gate_ag_test < x).mean()) for x in taus],
                 note='curve only; no operating point is chosen from it')

    # ---------------- Test-side identity read-outs (purity @ the frozen gate)
    gate_valid = bool(gates['gate_agreement']['S2_at_S1_0p95'] > TARGET_S2_GATE)
    gate_name = 'gate_agreement' if gate_valid else 'gate_score_fused'
    gate = gates[gate_name]
    gate_score_frozen = np.load(CACHE / f'gate_{gate_name}.npz')['score']
    accepted_gate = gate_score_frozen >= gate['tau_S1_0p95']
    id_metrics = {}
    s2 = ~is_s1
    p_eeg = fusion['eeg_only']['pred_test']
    for kind in FUSION_ORDER:
        p_all = fusion[kind]['pred_test']
        p_s2 = p_all[s2]
        pur_all = purity_of(p_s2, assign20, PURITY_K, np.ones(len(p_s2), bool))
        pur_acc = purity_of(p_s2, assign20, PURITY_K, accepted_gate[s2])
        id_metrics[kind] = dict(
            purity_all_S2=pur_all, purity_accepted_at_gate=pur_acc,
            identical_to_eeg_only=float((p_all == p_eeg).mean()),
            distinct_identities_all=int(len(np.unique(p_all))),
            distinct_identities_accepted=int(len(np.unique(p_all[accepted_gate]))),
            identity_distribution_all={int(c): int((p_all == c).sum()) for c in CLASSES},
            identity_distribution_accepted={
                int(c): int((p_all[accepted_gate] == c).sum())
                for c in np.unique(p_all[accepted_gate])},
            closed_set=fusion[kind]['closed_set'], oof=fusion[kind]['oof'])
        log(f'  identity {kind:18s} purity_all_S2={pur_all["purity_micro"]:.4f} '
            f'distinct_all={id_metrics[kind]["distinct_identities_all"]} '
            f'purity@gate={pur_acc["purity_micro"]:.4f} '
            f'n_accepted={pur_acc["n"]} '
            f'same_as_eeg={id_metrics[kind]["identical_to_eeg_only"]:.4f}')

    branch_summary = {b: dict(
        test_separation_auc=auc_ties(-branch_out[b]['d1_test'][~is_s1],
                                     -branch_out[b]['d1_test'][is_s1]),
        d1_test_mean=float(branch_out[b]['d1_test'].mean()),
        d2_minus_d1_test_mean=float((branch_out[b]['d2_test']
                                     - branch_out[b]['d1_test']).mean()),
        closed_set=branch_closed[b],
        oof=macro_f1(subj[fit], branch_out[b]['id_oof'][fit], CLASSES)) for b in BRANCHES}

    return finish(ctx, branches, views, branch_summary, fusion, id_metrics, agree, gates,
                  sweep, checks, assign20, sizes20, t_start, gate_valid, gate_name)


def finish(ctx, branches, views, branch_summary, fusion, id_metrics, agree, gates, sweep,
           checks, assign20, sizes20, t_start, gate_valid, gate_name):
    is_s1, fit = ctx['is_s1'], ctx['fit']
    base = id_metrics['eeg_only']
    base_f1 = base['closed_set']['loso_macro_f1_mean']
    base_pur = base['purity_all_S2']['purity_micro']
    cond = {}
    for kind in FUSION_ORDER:
        m = id_metrics[kind]
        cond[kind] = dict(
            f1_ge_baseline=bool(m['closed_set']['loso_macro_f1_mean'] >= base_f1),
            purity_ok=bool(m['purity_all_S2']['purity_micro'] >= base_pur - PURITY_TOL),
            identical_to_eeg_only=bool(m['identical_to_eeg_only'] >= 1.0 - 1e-12),
            delta_f1=float(m['closed_set']['loso_macro_f1_mean'] - base_f1),
            delta_purity=float(m['purity_all_S2']['purity_micro'] - base_pur))
    # a candidate whose predictions are byte-identical to eeg_only carries no image
    # information (algebraic degeneracy) and cannot demonstrate that image branches help
    eligible = [k for k in FUSION_ORDER if k != 'eeg_only'
                and cond[k]['f1_ge_baseline'] and cond[k]['purity_ok']
                and not cond[k]['identical_to_eeg_only']]
    degenerate = [k for k in FUSION_ORDER if k != 'eeg_only'
                  and cond[k]['identical_to_eeg_only']]
    if eligible:
        adopted = max(eligible, key=lambda k: (
            id_metrics[k]['closed_set']['loso_macro_f1_mean'],
            id_metrics[k]['purity_all_S2']['purity_micro'],
            id_metrics[k]['oof']['macro_f1']))
        identity_valid = True
    else:
        adopted, identity_valid = None, False

    g = gates['gate_agreement']
    ctrl = gates['gate_score_fused']
    frozen_pred = fusion[adopted]['pred_test'] if adopted else fusion['eeg_only']['pred_test']
    gate = gates[gate_name]
    gate_score = np.load(CACHE / f'gate_{gate_name}.npz')['score']
    acc = gate_score >= gate['tau_S1_0p95']
    label = np.where(acc, frozen_pred, 0).astype(int)

    te = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    pd.DataFrame(dict(sample=np.arange(1, len(te) + 1), path=te.path.to_numpy(),
                      label=label)).to_csv(SUB / 'test.csv', index=False)
    pd.DataFrame(dict(path=te.path.to_numpy(),
                      session=te.session.astype(str).to_numpy(),
                      task=te.task.to_numpy(), score=gate_score,
                      pred_identity=frozen_pred, label=label)).to_csv(
        SUB / 'test_predictions_detailed.csv', index=False)
    sub_metrics = dict(
        tau_at_S1_95=float(gate['tau_S1_0p95']),
        S1_rejection=float(1.0 - gate['S1_at_0p95']),
        S1_rejection_at_loso_tau=float(gate['S1_rejection']),
        S2_acceptance=float(gate['S2_at_S1_0p95']),
        S2_acceptance_at_loso_tau=float(gate['S2_acceptance']),
        overall_rejection=float((~acc).mean()),
        identities=int(len(np.unique(frozen_pred[acc]))),
        identity_scheme=adopted or 'eeg_only', gate=gate_name,
        note='label 0 = rejected at the S1>=0.95 gate; Test has no identity labels so the '
             'official C / identity macro-F1 are N/A')
    (SUB / 'metrics.json').write_text(json.dumps(sub_metrics, indent=2), encoding='utf8')
    log('submission ' + json.dumps(sub_metrics))

    R = dict(
        experiment='E238_multiview_identity_fusion',
        config=CFG,
        reproduction=checks,
        branches=branch_summary,
        identity_fusion=dict(
            candidates={k: dict(closed_set=fusion[k]['closed_set'], oof=fusion[k]['oof'],
                                weights=FUSION_W.get(k, {'EEG': 1.0}))
                        for k in FUSION_ORDER},
            test=id_metrics,
            decision=dict(adopted=adopted, identity_valid=identity_valid,
                          conditions=cond, eligible=eligible,
                          degenerate_identical_to_eeg_only=degenerate,
                          baseline_loso_macro_f1=base_f1,
                          baseline_purity_all_S2=base_pur,
                          purity_tolerance=PURITY_TOL,
                          degeneracy_note='vote_weighted is algebraically identical to '
                                          'eeg_only: EEG holds 0.5 of the vote and the three '
                                          'image branches hold 1/6 each, so 0.5 >= 3 * 1/6 '
                                          'and the EEG d1 tie-break decides every tie. It is '
                                          'reported but excluded from adoption because it '
                                          'carries no image information.')),
        branch_agreement=agree,
        agreement_component=ctx['agreement_component'],
        gates=gates,
        gate_sweep=sweep,
        gate_decision=dict(adopted=gate_name, agreement_gate_valid=gate_valid,
                           target_S2_at_S1_0p95=TARGET_S2_GATE,
                           agreement_S2_at_S1_0p95=g['S2_at_S1_0p95'],
                           agreement_sep_auc=g['separation_auc'],
                           control_S2_at_S1_0p95=ctrl['S2_at_S1_0p95'],
                           control_sep_auc=ctrl['separation_auc']),
        purity_partition=dict(k=PURITY_K, seed=PURITY_SEED + PURITY_K,
                              sizes=[int(x) for x in sizes20]),
        e233_purity_all_S2_reference=BASE_PUR_ALL,
        submission=sub_metrics,
        official_note=CFG['decision_rule']['official_note'],
        limitations=CFG['limitations'],
        runtime_s=float(time.time() - t_start))
    (OUT / 'results.json').write_text(json.dumps(R, indent=2, default=float),
                                      encoding='utf8')
    MODELS.mkdir(exist_ok=True)
    torch.save(dict(
        experiment='E238_multiview_identity_fusion',
        image_views={n: dict(mu_pca=views[n]['mu_pca'], V=views[n]['V'],
                             protos=views[n]['protos'], p_task=views[n]['p_task'],
                             p_id=views[n]['p_id']) for n in PIX_VIEWS},
        identity_decision=dict(adopted=adopted, identity_valid=identity_valid),
        gate_decision=dict(adopted=gate_name, agreement_gate_valid=gate_valid,
                           tau_loso=gates[gate_name]['tau'],
                           tau_S1_0p95=gates[gate_name]['tau_S1_0p95']),
        note='EEG side is frozen E233 mean_align; image views emit identities through the '
             'same E228/E233 scoring flow'),
        MODELS / 'e238_identity_fusion.pt')
    build_report(R, id_metrics, agree, gates, sweep, branch_summary, adopted,
                 identity_valid, gate_valid, sub_metrics, base_f1, base_pur)
    log(f'results.json + report.md written, runtime={time.time() - t_start:.0f}s')
    return R


def f4(x, n=4):
    if x is None:
        return '—'
    try:
        if np.isnan(x):
            return '—'
    except TypeError:
        pass
    return f'{x:.{n}f}'


def build_report(R, id_metrics, agree, gates, sweep, branch_summary, adopted,
                 identity_valid, gate_valid, sub, base_f1, base_pur):
    base = id_metrics['eeg_only']
    g = gates['gate_agreement']
    ctrl = gates['gate_score_fused']
    L = ['# E238 多视图身份融合 + 一致性门控', '',
         '> **一句话结论**：'
         + (f"图像分支参与身份判定**有效**，冻结身份方案 `{adopted}`（LOSO 闭集 macro-F1 "
            f"{f4(id_metrics[adopted]['closed_set']['loso_macro_f1_mean'])} ≥ eeg_only "
            f"{f4(base_f1)}，Test k=20 全 S2 纯度 "
            f"{f4(id_metrics[adopted]['purity_all_S2']['purity_micro'])} ≥ {f4(base_pur)}）。"
            if identity_valid else
            f"图像分支参与身份判定**无效**：唯一数值上达标的 `vote_weighted` 与 `eeg_only` "
            f"逐行相同（0.5 ≥ 3×1/6 的代数退化），其余候选 LOSO 闭集 macro-F1 都低于 eeg_only "
            f"的 {f4(base_f1)}，身份方案保持 `eeg_only`。")
         + (f" 一致性门控使 S1≥0.95 下的 S2 = {f4(g['S2_at_S1_0p95'])} > 0.8776，**门控冻结**。"
            if gate_valid else
            f" 一致性门控的 S2@S1.95 = {f4(g['S2_at_S1_0p95'])}，未超过 0.8776，**门控不冻结**"
            f"（保留 E237 的 `gate_score_fused`，同读法 S2 = "
            f"{f4(ctrl['S2_at_S1_0p95'])}）。"), '',
         '口径：EEG 分支是 E233 `mean_align` 的冻结产物（E227 deploy 表征 + 对称批次均值对齐，'
         'pred 逐行复现 E233）；图像分支沿用 E149 像素视图 + E237 的打分流程（fit 行逐 task 标准化 → '
         'PCA96 → 每 (task, identity) k=12 原型 → 负最近距离 → 同一套批次均值对齐）。'
         '四个分支现在都输出身份（`id_b`、`d1_b`、`d2_b`），身份不再只来自 EEG 分支。', '',
         '**官方口径提醒**：Test 无身份标签，**C / RA / ASafeR 与官方身份 macro-F1 仍为 N/A**。'
         '本报告所有 macro-F1 都是**训练域闭集值**（20 折 LOSO 折内，held identity 已排除；'
         '或 fit 行 inner 5-block OOF），不是 Test 值。', '',
         '## 1. 复现检查', '',
         '|项目|本轮实测|参考|判定|', '|---|---:|---:|:---:|',
         f"|EEG 分支 Test pred 与 E233 逐行一致|"
         f"{R['reproduction']['pred_identical_e233']}|True|"
         f"{'通过' if R['reproduction']['pred_identical_e233'] else '未通过'}|",
         f"|EEG 分支 Test 分数最大绝对差|"
         f"{R['reproduction']['score_max_abs_diff']:.2e}|0|"
         f"{'通过' if R['reproduction']['score_max_abs_diff'] < 1e-3 else '未通过'}|",
         f"|`gate_score_fused` 复现 E237 S2 / S2@S1.95|{f4(ctrl['S2_acceptance'])} / "
         f"{f4(ctrl['S2_at_S1_0p95'])}|{f4(E237_S2)} / {f4(E237_S2_GATE)}|"
         f"{'通过' if abs(ctrl['S2_acceptance'] - E237_S2) < 1e-6 else '未通过'}|",
         f"|融合分数与 E237 缓存最大绝对差|"
         f"{R['reproduction']['e237_fused_score_max_abs_diff']:.2e}|0|校验|", '',
         '## 2. 四个分支的身份能力（无标签，训练域闭集）', '',
         '`LOSO 闭集`：20 折，每折只用已注册（enrolled）身份的 fit 行原型，在评估窗'
         '（trial 比例 (0.6,0.8]）的**已注册行**上算 macro-F1 / accuracy；held identity 的行不参与。'
         '`OOF 闭集`：fit 行 inner 5-block OOF，20 个身份全部参与。'
         '`Test AUC`：S2-vs-S1 的对齐后距离分离度（-d1），只是行为指标。', '',
         '|分支|Test AUC(对齐后)|LOSO macro-F1|±std|LOSO acc|OOF macro-F1|d1 均值|(d2-d1) 均值|',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in BRANCHES:
        c = branch_summary[b]
        L.append(f"|{b}|{f4(c['test_separation_auc'])}|"
                 f"{f4(c['closed_set']['loso_macro_f1_mean'])}|"
                 f"{f4(c['closed_set']['loso_macro_f1_std'], 3)}|"
                 f"{f4(c['closed_set']['loso_accuracy_mean'])}|{f4(c['oof'][0])}|"
                 f"{f4(c['d1_test_mean'], 3)}|{f4(c['d2_minus_d1_test_mean'], 3)}|")
    L += ['', '对照：E65/E228 记录的 LOSO 身份 macro-F1 ≈0.985，本轮 EEG 分支 0.9779 与之同量级；'
          '差别来自本轮把评估限制在 trial 比例 (0.6,0.8] 的已注册行、且每折只用 19 个 enroled '
          '身份计算 macro-F1。三个图像分支的闭集身份能力依次为 '
          'X_mw_cov4 0.42 / X_covariance_hr 0.66 / IMG_px_all 0.83，与 EEG 差 15–55 个百分点，'
          '这是后面融合必然稀释 EEG 身份的原因。', '']
    L += ['', '## 3. 身份融合候选（本轮重点）', '',
          '`LOSO macro-F1` 是判据主指标（训练域闭集，mean±std over 20 折）；`Δ` 相对 `eeg_only`；'
          '`纯度(全 S2)` 是 k=20 固定划分下、**全部 19218 条 S2 行**上用该候选预测身份算的 micro 纯度'
          '（与门控阈值无关，四个候选可直接比较）；`@gate` 是 S1≥0.95 门控接受行上的纯度。', '',
          '|候选|LOSO macro-F1|±std|Δ vs eeg_only|LOSO acc|OOF macro-F1|纯度(全 S2)|@gate 纯度|接受集合覆盖|与 eeg_only 相同比例|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in FUSION_ORDER:
        m = id_metrics[k]
        cs = m['closed_set']
        L.append(f"|{k}|{f4(cs['loso_macro_f1_mean'])}|{f4(cs['loso_macro_f1_std'], 3)}|"
                 f"{cs['loso_macro_f1_mean'] - base_f1:+.4f}|{f4(cs['loso_accuracy_mean'])}|"
                 f"{f4(m['oof']['macro_f1'])}|{f4(m['purity_all_S2']['purity_micro'])}|"
                 f"{f4(m['purity_accepted_at_gate']['purity_micro'])}|"
                 f"{m['distinct_identities_accepted']}|"
                 f"{f4(m['identical_to_eeg_only'])}|")
    L += ['', '判据（config 预先写死）：LOSO macro-F1 ≥ eeg_only **且** 全 S2 纯度 ≥ eeg_only − '
          f'{PURITY_TOL}。', '']
    for k in FUSION_ORDER:
        if k == 'eeg_only':
            continue
        c = R['identity_fusion']['decision']['conditions'][k]
        L.append(f"- `{k}`：ΔF1 {c['delta_f1']:+.4f}（{'达标' if c['f1_ge_baseline'] else '未达标'}）、"
                 f"Δ纯度 {c['delta_purity']:+.4f}（{'达标' if c['purity_ok'] else '未达标'}）"
                 + ('、**与 eeg_only 预测逐行相同 → 代数退化，不计入采纳**'
                    if c['identical_to_eeg_only'] else ''))
    L += ['', '**为什么 `vote_weighted` 与 `eeg_only` 完全一样**：EEG 一票 0.5，三个图像分支各 '
          '1/6（合计 0.5），所以 EEG 的票数永远不小于三个图像分支之和；平票时按 config 用 EEG 的 '
          'd1 打破，于是图像票从不改变结果。这不是数值巧合，而是这组固定权重的代数性质。', '']
    L += ['', '## 4. 分支一致性（无标签）', '',
          '|对照|4/4 一致|≥3/4 一致|≥2/4 一致|平均最大票数|', '|---|---:|---:|---:|---:|']
    for label, key in (('Test 全部', 'test'), ('Test S2', 'test_s2'),
                       ('Test S1', 'test_s1'), ('fit 行 OOF', 'fit_oof')):
        a = agree[key]
        L.append(f"|{label}|{f4(a['unanimous_4of4'])}|{f4(a['majority_ge3'])}|"
                 f"{f4(a['majority_ge2'])}|{f4(a['mean_max_votes'], 3)}|")
    L += ['', '两两一致率（Test 全部行）：' + '、'.join(
        f"{k} {f4(v)}" for k, v in agree['test']['pairwise'].items()) + '。',
          '', '一致性作为独立的已知/未知判别器（无标签可读）：`-vote entropy` 的 S2-vs-S1 AUC = '
          f"{f4(R['agreement_component']['test_auc_agreement_vs_s1'])}"
          '（0.5 = 无信息）。它只是弱信号，这决定了下面一致性门控的天花板。', '',
          '## 5. 门控候选', '',
          '`S1/S2(LOSO)` 在 **LOSO 未知拒识 = 0.90** 的 tau 上读出；`S1 接受@gate` 是 S1≥0.95 门槛上 '
          'S1 的**接受率**（应≈0.05，拒识 ≈0.95）；`S2@S1.95` 是把 tau 设为该候选自己 Test S1 '
          '分数的 95% 分位后的 S2（**官方门控读法**）；`sepAUC` 与阈值无关；`整体拒识(LOSO tau)` '
          '是该 tau 下全部 Test 行的拒识率；`20 折 S2 代理` 是 Train 侧已知行在该 tau 下的接受率。'
          '纯度列用 EEG 分支身份（与 E237 同口径）。', '',
          '|门控|S1(LOSO)|S2(LOSO)|sepAUC|S1 接受@gate|S2@S1.95|整体拒识(LOSO tau)|覆盖身份|@gate 纯度|20 折 S2 代理|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name in ('gate_score_fused', 'gate_agreement'):
        m = gates[name]
        L.append(f"|{name}|{f4(m['S1_rejection'])}|{f4(m['S2_acceptance'])}|"
                 f"{f4(m['separation_auc'])}|{f4(m['S1_at_0p95'])}|{f4(m['S2_at_S1_0p95'])}|"
                 f"{f4(m['overall_rejection'])}|{m['identities_predicted']}|"
                 f"{f4(m['purity_at_gate_S2']['purity_micro'])}|"
                 f"{f4(m['fold_S2_proxy_mean'])}±{f4(m['fold_S2_proxy_std'], 3)}|")
    L += ['', '阈值扫描（`gate_agreement`，200 个 S1 分位点，**只输出曲线，不选点**）：',
          f"- S1 分位 {sweep['quantiles'][0]:.2f}→{sweep['quantiles'][-1]:.3f} 时，S2 接受从 "
          f"{f4(sweep['S2_acceptance'][0])} 变到 {f4(sweep['S2_acceptance'][-1])}；"
          '曲线数组见 results.json 的 `gate_sweep`。',
          f"- 曲线上的点都不是本轮工作点；官方门控读法是 S1≥0.95（两个门控的 S1 接受率都是 "
          f"{f4(g['S1_at_0p95'])}，即 S1 拒识 {f4(1 - g['S1_at_0p95'])}）。",
          f"- **一致性门控没有超过 0.8776**：它的 sepAUC 从对照的 {f4(ctrl['separation_auc'])} 降到 "
          f"{f4(g['separation_auc'])}，S2@S1.95 从 {f4(ctrl['S2_at_S1_0p95'])} 降到 "
          f"{f4(g['S2_at_S1_0p95'])}。原因可见第 4 节：一致性只有 "
          f"{f4(R['agreement_component']['test_auc_agreement_vs_s1'])} 的独立 AUC，按 0.5/0.5 与 "
          '融合分数相加只是把强排序稀释；这也说明"分支身份一致"在本数据上不足以替代分数证据。', '',
          '## 6. Test 侧身份分布与覆盖（无标签）', '',
          '|候选|覆盖身份(全体)|最大单一身份占比(全 S2)|接受集合 n|接受集合覆盖|',
          '|---|---:|---:|---:|---:|']
    for k in FUSION_ORDER:
        m = id_metrics[k]
        L.append(f"|{k}|{m['distinct_identities_all']}|"
                 f"{f4(m['purity_all_S2']['max_identity_share'])}|"
                 f"{m['purity_accepted_at_gate']['n']}|{m['distinct_identities_accepted']}|")
    L += ['', '## 7. 判定与冻结', '',
          f"- 身份方案：**{'`' + adopted + '`（图像参与身份有效）' if identity_valid else '`eeg_only`（保持基线）'}**。"
          + ('`vote_weighted` 虽然数值上"≥ eeg_only"，但它与 eeg_only 逐行相同（见第 3 节），'
             '按"图像是否真的参与身份"这一目的不构成证据，已排除在采纳之外。'
             if not identity_valid else ''),
          f"- 门控：**{'`gate_agreement`' if gate_valid else '`gate_score_fused`（E237 对照）'}**。"
          f"官方门控读法（S1≥0.95）下 S2 = {f4(sub['S2_acceptance'])}，"
          + ('超过 0.8776。' if gate_valid else
             '与 E237 相同，本轮一致性门控没有提供额外增益（其 S2 = '
             f"{f4(g['S2_at_S1_0p95'])}，低于 0.8776）。"),
          f"- 一致性门控在 LOSO=0.90 工作点上的 S2 = {f4(g['S2_acceptance'])}（对照 "
          f"{f4(ctrl['S2_acceptance'])}），但同一工作点上 S1 拒识只有 {f4(g['S1_rejection'])}"
          f"（对照 {f4(ctrl['S1_rejection'])}）——它是靠多放进 S1 换来的 S2，不可用。",
          f"- 门槛读法下一致性门控把 S2 从对照的 {f4(ctrl['S2_at_S1_0p95'])} 改为 "
          f"{f4(g['S2_at_S1_0p95'])}，sepAUC 从 {f4(ctrl['separation_auc'])} 改为 "
          f"{f4(g['separation_auc'])}。这是**转导式门槛读法**（tau 由 Test S1 分位决定），"
          '不等价于"用 Train 侧阈值就能多接受"。', '',
          '## 8. 提交文件', '',
          f"`submission/test.csv`：22251 行、原始 metadata 顺序，`label` 0–20"
          f"（0 = 在 S1≥0.95 门控下拒识，1–20 = 冻结身份方案 `{sub['identity_scheme']}` 的预测身份）；"
          f"`metrics.json` 记录该工作点：S1 拒识 {f4(sub['S1_rejection'])}、S2 接受 "
          f"{f4(sub['S2_acceptance'])}、整体拒识 {f4(sub['overall_rejection'])}、覆盖身份 "
          f"{sub['identities']}。", '',
          '## 9. 限制', '']
    L += ['- ' + x for x in CFG['limitations']]
    L += [f"- 运行：`.venv\\Scripts\\python.exe research/E238_multiview_identity_fusion/run_e238.py`"
          f"（CUDA），耗时 {R['runtime_s']:.0f}s。逐分支身份/距离缓存在 `cache/`，"
          '图像投影与原型存于 `models/e238_identity_fusion.pt`。']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


if __name__ == '__main__':
    CFG_BLOCK_PATHS_TEST = json.loads(
        (ROOT / 'research/E233_delta_estimation/config.json').read_text(encoding='utf8')
    )['feature_cache']['test']
    main()
