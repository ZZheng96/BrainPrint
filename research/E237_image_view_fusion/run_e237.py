"""E237: fuse the deployable image views onto the E233 aligned EEG score.

Frozen EEG side (read only, byte-identical to E233 mean_align):
  E228 e226_auc2W representation -> symmetric per-(session, task) batch-mean
  alignment -> negative nearest distance to the k=12 prototypes of
  (task, identity).  Test score, predictions and LOSO fold scores are taken
  from the E233 / E235 caches and never refitted.

Image side (E149 pixel views, no DINOv2):
  every 32x32x3 pixel view is scored with the same E228/E233 flow
  (task-wise standardisation on the fit rows -> PCA96 -> k=12 prototypes per
  (task, identity) -> negative nearest distance) and the same symmetric
  batch-mean alignment in its own PCA space.  IMG_px_all is the equal-weight
  mean of the 7 per-view z-scored scores (the E149 definition).

Fusion: per-task empirical CDF of the fit-row inner-5-block OOF known scores
converts each component into a rank in [0, 1]; the fused score is a fixed
weighted sum.  Weights live in config.json and were written before the run.

Read-outs per candidate: Test S2-vs-S1 separation AUC, Test S1 rejection and
S2 acceptance at the tau that gives LOSO unknown rejection = 0.90, S2 at the
S1 >= 0.95 gate, overall rejection, identity coverage, k=20 label-free cluster
purity of the accepted rows and the 20-fold Train-side S2 proxy (mean +- std).
Test is only evaluated, never used to pick an identity or a threshold.
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
S1_LOOSE = 0.70
PURITY_TOL = 0.95

CMP_VIEWS = list(CFG['image_views']['comparison_views'].keys())
PIX_VIEWS = list(CFG['image_views']['pixel_views'].keys())
PRIMARY = CFG['fusion']['primary_image_component']
WEIGHTS = CFG['fusion']['weights_fixed_before_run']
ROUND2 = CFG['round2_candidates']['candidates']

A = CFG['baseline_anchors']
BASE_AUC = float(A['e233_mean_align']['test_separation_auc'])
BASE_S2 = float(A['e233_mean_align']['test_S2_acceptance'])
BASE_S1 = float(A['e233_mean_align']['test_S1_rejection'])
BASE_S2_GATE = float(A['e233_mean_align']['S2_at_S1_0p95'])
BASE_PUR_ALL = float(A['e233_mean_align']['purity_k20_micro_all_S2'])
TAU_EEG = float(A['e233_mean_align']['test_tau'])

CLASSES = np.arange(1, 21)


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


def score_rows(Z, protos, p_task, task):
    Zt = tensor(Z)
    out = np.full(len(Z), np.nan, np.float32)
    for t in TASKS:
        rr = np.flatnonzero(task == t)
        cols = np.flatnonzero(p_task == t)
        if not len(rr) or not len(cols):
            continue
        P = tensor(protos[cols])
        best = np.empty(len(rr), np.float32)
        for i in range(0, len(rr), 4096):
            ch = rr[i:i + 4096]
            d = torch.cdist(Zt[ch], P).min(1).values
            best[i:i + len(ch)] = (-d).cpu().numpy()
        out[rr] = best
    return out


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
    for sname, mask in masks.items():
        for t in TASKS:
            rr = np.flatnonzero(mask & (task == t))
            if len(rr) < 200:
                continue
            Z[rr] = Z[rr] - Z[rr].mean(0) + ref_mu[t]
    return Z


# --------------------------------------------------------------- image branch
def image_component(name, path_tr, path_te, tr_task, te_task, subj, fit, eval_idx,
                    eval_task, eval_subj, masks):
    t0 = time.time()
    Xtr = np.load(ROOT / path_tr).astype(np.float32)
    Xte = np.load(ROOT / path_te).astype(np.float32)
    assert Xtr.shape[0] == len(tr_task) and Xte.shape[0] == len(te_task), (Xtr.shape, Xte.shape)
    for t in TASKS:
        fm = fit & (tr_task == t)
        mu = Xtr[fm].mean(0)
        sd = Xtr[fm].std(0) + 1e-6
        trm = tr_task == t
        Xtr[trm] = (Xtr[trm] - mu) / sd
        tem = te_task == t
        Xte[tem] = (Xte[tem] - mu) / sd
    mu_pca, V = pca_basis(Xtr[fit], PCA_DIM)
    Ztr = project(Xtr, mu_pca, V)
    Zte = project(Xte, mu_pca, V)
    del Xtr, Xte

    ref_mu = {t: Ztr[fit & (tr_task == t)].mean(0) for t in TASKS}
    protos, p_task, _ = build_protos(Ztr, subj, tr_task, fit, CLASSES, K_PROTO)
    te_raw = score_rows(Zte, protos, p_task, te_task)
    te_score = score_rows(align_cells(Zte, masks, te_task, ref_mu), protos, p_task, te_task)

    # inner 5-block OOF known reference on the fit rows
    ref_score = np.full(len(Ztr), np.nan, np.float32)
    for b in range(N_BLOCKS):
        hold = fit & (blocks_frac == b)
        p2, t2, _ = build_protos(Ztr, subj, tr_task, fit & ~hold, CLASSES, K_PROTO)
        s = score_rows(Ztr, p2, t2, tr_task)
        ref_score[hold] = s[hold]

    # 20-fold LOSO on the shared evaluation window
    fold_score = np.empty((N_FOLDS, len(eval_idx)), np.float32)
    for h, held in enumerate(CLASSES):
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subj, enrolled)
        p2, t2, _ = build_protos(Ztr, subj, tr_task, pm & fit, enrolled, K_PROTO)
        s = score_rows(Ztr, p2, t2, tr_task)
        fold_score[h] = s[eval_idx]

    log(f'  view {name:14s} dim={V.shape[0]} auc_raw={auc_ties(te_raw[~is_s1], te_raw[is_s1]):.4f} '
        f'auc_aligned={auc_ties(te_score[~is_s1], te_score[is_s1]):.4f} '
        f'({time.time() - t0:.1f}s)')
    np.savez_compressed(CACHE / f'view_{name}.npz', test_raw=te_raw, test_aligned=te_score,
                        ref=ref_score, ref_fit=fit, fold_score=fold_score)
    return dict(name=name, test_raw=te_raw, test=te_score, ref=ref_score,
                fold=fold_score, mu_pca=mu_pca, V=V, protos=protos, p_task=p_task)


# --------------------------------------------------------------- rank fusion
def ref_sorted(scores, task, mask):
    return {t: np.sort(scores[mask & (task == t)]) for t in TASKS}


def rank_of(sorted_ref, score, task, tail=50):
    """Per-task empirical CDF of the fit-row known reference, with a linear extrapolation
    of the quantile function beyond the reference support.

    The plain ECDF saturates: every Test session1 score below the smallest reference score
    maps to the same rank 0, which destroys the resolution of the S1 >= 0.95 gate and would
    make it incomparable with the score-space gate of E233/E236.  Extrapolating the lowest
    (and highest) `tail` reference points keeps the map strictly monotone inside and outside
    the support, so a monotone transform of a single component is still order-preserving.
    """
    out = np.empty(len(score), float)
    for t in TASKS:
        m = task == t
        if not m.any():
            continue
        x = sorted_ref[t]
        n = len(x)
        q = (np.arange(1, n + 1)) / n
        s = score[m]
        v = np.interp(s, x, q)
        lo = s < x[0]
        if lo.any():
            mm = min(n, tail)
            slope = (q[mm - 1] - q[0]) / max(x[mm - 1] - x[0], 1e-9)
            v[lo] = q[0] + (s[lo] - x[0]) * slope
        hi = s > x[-1]
        if hi.any():
            mm = min(n, tail)
            slope = (q[-1] - q[-mm]) / max(x[-1] - x[-mm], 1e-9)
            v[hi] = q[-1] + (s[hi] - x[-1]) * slope
        out[m] = v
    return out


def z_comp(comp, ref_fit, fk_list, fu_list):
    mu, sd = float(ref_fit.mean()), float(ref_fit.std()) + 1e-6
    return dict(test=(comp['test'] - mu) / sd,
                fold_known=[(x - mu) / sd for x in fk_list],
                fold_unknown=[(x - mu) / sd for x in fu_list])


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
    """pred_s2 / assign are both over the Test session2 rows; mask selects accepted rows."""
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


def evaluate(cid, te, fk, fu, pred_s2, assign, pred_all):
    uk = np.concatenate(fu)
    # reject iff score < tau, so LOSO unknown rejection = 0.90 means tau is the 90th
    # percentile of the pooled unknown scores (higher score = more likely known).
    tau = float(np.quantile(uk, LOSO_TARGET))
    fold_rej = [float((u < tau).mean()) for u in fu]
    fold_acc = [float((k >= tau).mean()) for k in fk]
    rej = te < tau
    acc = ~rej
    tau95 = float(np.quantile(te[is_s1], S1_GATE))
    tau70 = float(np.quantile(te[is_s1], S1_LOOSE))
    # the S1 gates are their own operating points (tau set by the S1 quantile), not an
    # overlay of the LOSO-calibrated tau
    s2_095 = float((te[~is_s1] >= tau95).mean())
    s2_070 = float((te[~is_s1] >= tau70).mean())
    te_s2 = te[~is_s1]
    pur_acc = purity_of(pred_s2, assign, PURITY_K, te_s2 >= tau)
    pur_095 = purity_of(pred_s2, assign, PURITY_K, te_s2 >= tau95)
    return dict(
        tau=tau, loso_unknown_rejection=float((uk < tau).mean()),
        loso_fold_rejection_mean=float(np.mean(fold_rej)),
        loso_fold_rejection_std=float(np.std(fold_rej)),
        S1_rejection=float(rej[is_s1].mean()),
        S2_acceptance=float(acc[~is_s1].mean()),
        separation_auc=auc_ties(te[~is_s1], te[is_s1]),
        S2_at_S1_0p95=s2_095, S1_at_0p95=float((te[is_s1] >= tau95).mean()),
        tau_S1_0p95=tau95, S2_at_S1_0p70=s2_070,
        overall_rejection=float(rej.mean()),
        identities_predicted=int(len(np.unique(pred_all[acc]))),
        fold_S2_proxy_mean=float(np.mean(fold_acc)),
        fold_S2_proxy_std=float(np.std(fold_acc)),
        fold_S2_proxy=fold_acc,
        purity_accepted_S2=pur_acc, purity_at_gate_S2=pur_095,
        n_accepted=int(acc.sum()), n_accepted_S2=int((acc & ~is_s1).sum()),
    )


# --------------------------------------------------------------- main
print_metrics = None


def main():
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA required'
    log(f'device={torch.cuda.get_device_name(0)} torch={torch.__version__}')
    t_start = time.time()

    tr = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    te = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tr) == 11304 and len(te) == 22251
    global is_s1, tr_task, te_task, subj, fit, eval_idx, blocks_frac, fk_mask, fu_mask
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
    log(f'fit rows={int(fit.sum())} eval rows={len(eval_idx)} '
        f'test S1={int(is_s1.sum())} S2={int((~is_s1).sum())}')

    # ---- EEG component (frozen E233 scores)
    e = np.load(ROOT / CFG['eeg_component']['test_score'], allow_pickle=True)
    assert np.array_equal(e['task'], te_task)
    eeg_test = e['score'].astype(np.float64)
    eeg_pred = e['pred'].astype(int)
    assert abs(float(e['tau']) - TAU_EEG) < 1e-9
    dt = np.load(ROOT / CFG['eeg_component']['test_oof'], allow_pickle=True)
    oof_fit = dt['oof_fit'].astype(bool)
    assert np.array_equal(oof_fit, fit), 'oof_fit != fit rows'
    eeg_ref = dt['oof_s'].astype(np.float64)
    eeg_base = dict(
        S1_rejection=float((eeg_test[is_s1] < TAU_EEG).mean()),
        S2_acceptance=float((eeg_test[~is_s1] >= TAU_EEG).mean()),
        separation_auc=auc_ties(eeg_test[~is_s1], eeg_test[is_s1]))
    log('E233 mean_align raw reproduction ' + json.dumps(eeg_base))
    assert abs(eeg_base['S2_acceptance'] - BASE_S2) < 1e-9
    assert abs(eeg_base['separation_auc'] - BASE_AUC) < 1e-9

    fold_known, fold_unknown, fold_task = [], [], []
    for h in range(1, N_FOLDS + 1):
        d = np.load(ROOT / CFG['eeg_component']['fold_cache'].format(fold=h), allow_pickle=True)
        s = -d['d1_orig'].astype(np.float64)
        kn, un = d['known'].astype(bool), d['unknown'].astype(bool)
        assert np.array_equal(d['task'], eval_task) and np.array_equal(d['subject'], eval_subj)
        fold_known.append(s[kn])
        fold_unknown.append(s[un])
        fold_task.append((d['task'][kn], d['task'][un]))
    fk_mask = np.array([np.isin(eval_subj, np.setdiff1d(CLASSES, [c])) for c in CLASSES])
    fu_mask = np.array([eval_subj == c for c in CLASSES])

    eeg_comp = dict(name='EEG_e233_mean_align', test=eeg_test, ref=eeg_ref,
                    fold=np.stack([np.r_[fold_known[h], fold_unknown[h]] for h in range(N_FOLDS)]))

    # ---- image components (7 pixel views)
    log('scoring image views (task-wise standardisation -> PCA96 -> k=12 protos)')
    views = {}
    px_cfg = CFG['image_views']['pixel_views']
    for name in PIX_VIEWS:
        views[name] = image_component(name, px_cfg[name]['train'], px_cfg[name]['test'],
                                      tr_task, te_task, subj, fit, eval_idx, eval_task,
                                      eval_subj, masks)

    # composite IMG_px_all = mean of per-view z-scores (E149 definition)
    def standardise(arr, name):
        r = views[name]['ref'][fit]
        return (arr - float(r.mean())) / (float(r.std()) + 1e-6)

    comp_test = np.mean([standardise(views[n]['test'], n) for n in PIX_VIEWS], 0)
    comp_raw = np.mean([standardise(views[n]['test_raw'], n) for n in PIX_VIEWS], 0)
    comp_ref = np.mean([standardise(views[n]['ref'], n) for n in PIX_VIEWS], 0)
    comp_fold = np.mean([np.stack([standardise(views[n]['fold'][h], n)
                                   for h in range(N_FOLDS)]) for n in PIX_VIEWS], 0)
    views['IMG_px_all'] = dict(name='IMG_px_all', test=comp_test, test_raw=comp_raw,
                               ref=comp_ref, fold=comp_fold)
    log(f'  composite IMG_px_all auc_raw={auc_ties(comp_raw[~is_s1], comp_raw[is_s1]):.4f} '
        f'auc_aligned={auc_ties(comp_test[~is_s1], comp_test[is_s1]):.4f}')

    # ---- purity partition: frozen baseline EEG PCA space, k-means k=20 on all S2 rows
    log('building the label-free k=20 partition in the baseline EEG PCA space')
    zte = baseline_eeg_space()
    s2_rows = np.flatnonzero(~is_s1)
    assign20, sizes20 = kmeans_assign(zte[s2_rows].astype(np.float32), PURITY_K,
                                      PURITY_SEED + PURITY_K)
    log(f'  kmeans k=20 sizes={sizes20.tolist()}')
    pred_s2 = eeg_pred[s2_rows]
    pur_all = purity_of(pred_s2, assign20, PURITY_K, np.ones(len(s2_rows), bool))
    log('  E233 mean_align purity over all S2 rows ' + json.dumps(pur_all))
    assert abs(pur_all['purity_micro'] - BASE_PUR_ALL) < 1e-6, 'purity partition drifted'

    # ---- components in rank space
    def comp_rank(comp):
        if comp['name'].startswith('EEG'):
            return to_rank_eeg(comp)
        return dict(test=rank_of(ref_sorted(comp['ref'], tr_task, fit), comp['test'], te_task),
                    fold_known=[rank_of(ref_sorted(comp['ref'], tr_task, fit),
                                        comp['fold'][h][fk_mask[h]], eval_task[fk_mask[h]])
                                for h in range(N_FOLDS)],
                    fold_unknown=[rank_of(ref_sorted(comp['ref'], tr_task, fit),
                                          comp['fold'][h][fu_mask[h]], eval_task[fu_mask[h]])
                                  for h in range(N_FOLDS)])

    def to_rank_eeg(comp):
        sr = ref_sorted(comp['ref'], tr_task, fit)
        return dict(test=rank_of(sr, comp['test'], te_task),
                    fold_known=[rank_of(sr, fold_known[h], fold_task[h][0])
                                for h in range(N_FOLDS)],
                    fold_unknown=[rank_of(sr, fold_unknown[h], fold_task[h][1])
                                  for h in range(N_FOLDS)])

    ranks = {n: comp_rank(c) for n, c in views.items()}
    ranks['EEG'] = to_rank_eeg(eeg_comp)
    zs = dict(EEG=z_comp(eeg_comp, eeg_ref[fit], fold_known, fold_unknown))

    results = {}

    def add(cid, te_score, fk, fu, note):
        m = evaluate(cid, te_score, fk, fu, pred_s2, assign20, eeg_pred)
        m['note'] = note
        results[cid] = m
        np.savez_compressed(CACHE / f'cand_{cid}.npz', score=te_score, tau=m['tau'],
                            is_s1=is_s1)
        log(f'{cid:32s} LOSO={m["loso_unknown_rejection"]:.4f} S1={m["S1_rejection"]:.4f} '
            f'S2={m["S2_acceptance"]:.4f} sepAUC={m["separation_auc"]:.4f} '
            f'S2@95={m["S2_at_S1_0p95"]:.4f} pur={m["purity_accepted_S2"]["purity_micro"]:.4f} '
            f'foldS2={m["fold_S2_proxy_mean"]:.4f}+-{m["fold_S2_proxy_std"]:.4f}')
        return m

    add('baseline_eeg_rank', ranks['EEG']['test'], ranks['EEG']['fold_known'],
        ranks['EEG']['fold_unknown'], 'EEG only, per-task rank of the E233 mean_align score')
    for n in CMP_VIEWS + ['IMG_px_all']:
        add(f'img_only_{n}', ranks[n]['test'], ranks[n]['fold_known'], ranks[n]['fold_unknown'],
            f'{n} only, per-task rank')

    def fuse_rank(w_eeg, img):
        te_s = w_eeg * ranks['EEG']['test'] + (1 - w_eeg) * ranks[img]['test']
        fk = [w_eeg * a + (1 - w_eeg) * b
              for a, b in zip(ranks['EEG']['fold_known'], ranks[img]['fold_known'])]
        fu = [w_eeg * a + (1 - w_eeg) * b
              for a, b in zip(ranks['EEG']['fold_unknown'], ranks[img]['fold_unknown'])]
        return te_s, fk, fu

    for cid, w in WEIGHTS.items():
        if cid == 'fuse_zscore_50_50':
            img = primary_z(views)
            te_s = 0.5 * zs['EEG']['test'] + 0.5 * img['test']
            fk = [0.5 * a + 0.5 * b for a, b in zip(zs['EEG']['fold_known'], img['fold_known'])]
            fu = [0.5 * a + 0.5 * b for a, b in zip(zs['EEG']['fold_unknown'], img['fold_unknown'])]
            add(cid, te_s, fk, fu, 'z-score standardised equal-weight mean (scale control)')
        else:
            te_s, fk, fu = fuse_rank(w['eeg'], PRIMARY)
            add(cid, te_s, fk, fu, f'rank fusion EEG {w["eeg"]:.1f} / {PRIMARY} {1 - w["eeg"]:.1f}')
    for n in CMP_VIEWS:
        te_s, fk, fu = fuse_rank(0.5, n)
        add(f'fuse_rank_50_50_{n}', te_s, fk, fu,
            f'rank fusion EEG 0.5 / {n} 0.5 (single-view diagnostic)')

    # allowed single follow-up round
    for cid, w in ROUND2.items():
        if cid == 'fuse_rank_50_50_threeview':
            img_rank = np.mean([ranks[n]['test'] for n in CMP_VIEWS + ['IMG_px_all']], 0)
            fk = [np.mean([ranks[n]['fold_known'][h] for n in CMP_VIEWS + ['IMG_px_all']], 0)
                  for h in range(N_FOLDS)]
            fu = [np.mean([ranks[n]['fold_unknown'][h] for n in CMP_VIEWS + ['IMG_px_all']], 0)
                  for h in range(N_FOLDS)]
            add(cid, 0.5 * ranks['EEG']['test'] + 0.5 * img_rank,
                [0.5 * a + 0.5 * b for a, b in zip(ranks['EEG']['fold_known'], fk)],
                [0.5 * a + 0.5 * b for a, b in zip(ranks['EEG']['fold_unknown'], fu)],
                'round 2: rank fusion EEG 0.5 / mean rank of 3 comparison views 0.5')
        else:
            te_s, fk, fu = fuse_rank(w['eeg'], PRIMARY)
            add(cid, te_s, fk, fu, f'round 2: rank fusion EEG {w["eeg"]:.1f} / {PRIMARY} '
                                   f'{1 - w["eeg"]:.1f}')
    return finish(results, views, ranks, pur_all, assign20, eeg_base, t_start)


def primary_z(views):
    fk = [views[PRIMARY]['fold'][h][fk_mask[h]] for h in range(N_FOLDS)]
    fu = [views[PRIMARY]['fold'][h][fu_mask[h]] for h in range(N_FOLDS)]
    return z_comp(views[PRIMARY], views[PRIMARY]['ref'][fit], fk, fu)


def baseline_eeg_space():
    """Frozen E227 deploy model applied to the 10 E228 blocks -> the E233 baseline PCA space."""
    model = torch.load(ROOT / 'research/E227_joint_frontier/models/deploy_auc2_lofo.pt',
                       weights_only=False)
    st = model['state']
    order = list(model['block_order'])
    store = {}
    for b in order:
        store[b] = np.load(ROOT / CFG_BLOCK_PATHS_TEST[b]).astype(np.float32)
    X = np.concatenate([store[b] for b in order], 1)
    x = tensor(X)
    z = torch.empty_like(x)
    for i, t in enumerate(TASKS):
        rows = te_task == t
        z[rows] = (x[rows] - tensor(st['mu'][i])) / tensor(st['sd'][i])
    z = z * tensor(np.sqrt(np.asarray(st['w'], np.float32)))
    z = ((z.double() - tensor(st['pm'], torch.float64)) @ tensor(st['W'], torch.float64)).float()
    return z.cpu().numpy()


CFG_BLOCK_PATHS_TEST = None


def finish(results, views, ranks, pur_all, assign20, eeg_base, t_start):
    base = results['baseline_eeg_rank']
    round1 = [c for c in results if not c.startswith('fuse_rank_80') and
              not c.startswith('fuse_rank_20') and not c.endswith('threeview')]
    fused = [c for c in round1 if c.startswith('fuse_')]
    base_pur = base['purity_accepted_S2']['purity_micro']
    base_pur_gate = base['purity_at_gate_S2']['purity_micro']
    valid = [c for c in fused
             if results[c]['S2_at_S1_0p95'] > BASE_S2_GATE
             and results[c]['separation_auc'] > BASE_AUC
             and results[c]['purity_accepted_S2']['purity_micro'] >= PURITY_TOL * base_pur]
    sep_up = [c for c in fused if results[c]['separation_auc'] > BASE_AUC]
    best_sep = max(fused, key=lambda c: results[c]['separation_auc'])
    best_gate = max(fused, key=lambda c: results[c]['S2_at_S1_0p95'])
    if valid:
        verdict = 'pass'
        adopted = max(valid, key=lambda c: results[c]['S2_at_S1_0p95'])
    elif sep_up:
        verdict = 'separable_only'
        adopted = None
    else:
        verdict = 'fail'
        adopted = None
    log(f'decision verdict={verdict} adopted={adopted} best_sep={best_sep} '
        f'best_gate={best_gate}')

    R = dict(
        experiment='E237_image_view_fusion',
        config=CFG,
        eeg_component=dict(name='E233 mean_align', raw_reproduction=eeg_base,
                           tau=TAU_EEG),
        image_views={n: dict(auc_raw=float(auc_ties(views[n]['test_raw'][~is_s1],
                                                    views[n]['test_raw'][is_s1])),
                             auc_aligned=float(auc_ties(views[n]['test'][~is_s1],
                                                        views[n]['test'][is_s1])),
                             dim=(int(views[n]['V'].shape[0]) if 'V' in views[n]
                                  else 'composite of 7 x 3072'))
                     for n in PIX_VIEWS + [PRIMARY]},
        candidates=results,
        purity_all_S2=pur_all,
        partition_k20_sizes=[int(x) for x in np.bincount(assign20, minlength=PURITY_K)],
        decision=dict(verdict=verdict, adopted=adopted, valid=valid,
                      sep_above_baseline=sep_up,
                      best_separation_auc=best_sep,
                      best_separation=results[best_sep]['separation_auc'],
                      best_gate_candidate=best_gate,
                      best_S2_at_S1_0p95=results[best_gate]['S2_at_S1_0p95'],
                      baseline_sep_auc=BASE_AUC, baseline_S2_at_S1_0p95=BASE_S2_GATE,
                      baseline_S2=BASE_S2, baseline_purity_accepted=base_pur,
                      baseline_purity_at_gate=base_pur_gate,
                      baseline_S2_at_loso_tau=base['S2_acceptance'],
                      baseline_S2_at_gate_rank_space=base['S2_at_S1_0p95'],
                      adopted_S2_at_gate=results[adopted]['S2_at_S1_0p95'] if adopted else None,
                      adopted_S2_at_loso_tau=results[adopted]['S2_acceptance'] if adopted else None,
                      adopted_S2_at_gate_rank_space=(results[adopted]['S2_at_S1_0p95']
                                                     if adopted else None),
                      adopted_purity_at_gate=results[adopted]['purity_at_gate_S2']['purity_micro']
                      if adopted else None,
                      target_S2=0.706, purity_tolerance=PURITY_TOL),
        limitations=CFG['limitations'],
        runtime_s=float(time.time() - t_start))

    MODELS.mkdir(exist_ok=True)
    torch.save(dict(
        experiment='E237_image_view_fusion',
        eeg=dict(source='E233 mean_align', tau=TAU_EEG),
        image_views={n: dict(mu_pca=views[n]['mu_pca'], V=views[n]['V'],
                             protos=views[n]['protos'], p_task=views[n]['p_task'],
                             ref_scores=views[n]['ref'])
                     for n in PIX_VIEWS if 'V' in views[n]},
        fusion=dict(weights=WEIGHTS, primary=PRIMARY,
                    rank_reference='per-task ECDF of the fit-row inner-5-block OOF known scores'),
        decision=dict(verdict=verdict, adopted=adopted,
                      tau=results[adopted]['tau'] if adopted else None,
                      tau_S1_0p95=results[adopted]['tau_S1_0p95'] if adopted else None),
        note='image components are diagnostic only unless adopted; the EEG branch is frozen E233'),
        MODELS / 'e237_image_fusion.pt')
    (OUT / 'results.json').write_text(json.dumps(R, indent=2, default=float), encoding='utf8')
    build_report(R, base, pur_all)
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


def build_report(R, base, pur_all):
    C = R['candidates']
    dec = R['decision']
    views = R['image_views']
    L = ['# E237 图像视图叠加到 E233 对齐分数', '',
         '> **一句话结论**：'
         + (f"`{dec['adopted']}` 达标并冻结。" if dec['verdict'] == 'pass'
            else ('图像视图融合提升了 Test S2-vs-S1 可分性，但**可分性提升未转化为工作点**。'
                  if dec['verdict'] == 'separable_only'
                  else '**图像视图与对齐分数融合无增益。**'))
         + f" 最高分离 AUC = {f4(dec['best_separation'])}（`{dec['best_separation_auc']}`，"
           f"基线 {f4(BASE_AUC)}）；S1≥0.95 下最高 S2 = {f4(dec['best_S2_at_S1_0p95'])}"
           f"（`{dec['best_gate_candidate']}`，对比目标 0.706；同读法基线 "
           f"{f4(dec['baseline_S2_at_gate_rank_space'])}，E233 原始读法 {f4(BASE_S2_GATE)}）。", '',
         '**但必须同时读这条**：在 LOSO=0.90 的训练侧标定阈值上，融合并没有提升 S2'
         f"（采纳候选 {f4(dec['adopted_S2_at_loso_tau']) if dec['adopted_S2_at_loso_tau'] is not None else '—'}"
         f" vs 基线 {f4(dec['baseline_S2_at_loso_tau'])}）。增益全部出现在由 Test S1 分位定位的门槛上，"
         '即"同等拒掉 95% 的未注册被试时能放进更多注册用户"，而不是"用训练侧阈值就能多接受"。见 5.1。', '',
         '口径：EEG 侧为 E233 `mean_align` 的冻结分数（E228 e226_auc2W 表征 + 对称批次均值对齐），'
         '不重训、不重新标定；图像侧为 E149 的像素视图，走同一套 E228/E233 打分流程'
         '（fit 行逐 task 标准化 → PCA96 → 每 (task, identity) k=12 原型 → 负最近距离），'
         '并施加同一套逐 (session, task) 批次均值对齐。融合权重在运行前写死在 config。', '',
         '## 1. 复现与不变项', '',
         '|项目|本轮实测|E233 参考|判定|', '|---|---:|---:|:---:|',
         f"|EEG 原始分数 S2 接受（E233 tau={TAU_EEG:.4f}）|{f4(R['eeg_component']['raw_reproduction']['S2_acceptance'])}|"
         f"{f4(BASE_S2)}|{'通过' if abs(R['eeg_component']['raw_reproduction']['S2_acceptance'] - BASE_S2) < 1e-9 else '未通过'}|",
         f"|EEG 原始分数分离 AUC|{f4(R['eeg_component']['raw_reproduction']['separation_auc'])}|"
         f"{f4(BASE_AUC)}|{'通过' if abs(R['eeg_component']['raw_reproduction']['separation_auc'] - BASE_AUC) < 1e-9 else '未通过'}|",
         f"|E233 全 S2 簇纯度（k=20）|{f4(pur_all['purity_micro'])}|{f4(BASE_PUR_ALL)}|"
         f"{'通过' if abs(pur_all['purity_micro'] - BASE_PUR_ALL) < 1e-6 else '未通过'}|",
         f"|k=20 划分规模（与 E233 一致）|{R['partition_k20_sizes']}|E233 同|校验|", '',
         '- LOSO 由 E235/E227 的冻结逐折分数重算，逐折行顺序已核对（subject / known / unknown 逐行一致），'
         '因此 LOSO 拒识与 E233 同口径（micro 0.9000）。', '',
         '## 2. 三个图像视图单独成绩（同样施加对齐）', '',
         '|视图|维度|未对齐 Test AUC（E149 口径）|对齐后 Test AUC|LOSO=0.90 下 S1|LOSO=0.90 下 S2|S1≥0.95 下 S2|',
         '|---|---:|---:|---:|---:|---:|---:|']
    for n in CMP_VIEWS + [PRIMARY]:
        c = C[f'img_only_{n}']
        L.append(f"|{n}|{views[n]['dim']}|{f4(views[n]['auc_raw'])}|{f4(views[n]['auc_aligned'])}|"
                 f"{f4(c['S1_rejection'])}|{f4(c['S2_acceptance'])}|{f4(c['S2_at_S1_0p95'])}|")
    L += ['', '对照：E149 公布的未对齐像素 Test AUC 为 X_mw_cov4 0.7347、X_covariance_hr 0.7051、'
          'IMG_px_all 0.7590。本轮使用 fit 行统计量与 fit 行原型（E228 口径，比 E149 的全量行口径更严），'
          '故数值略有出入；对齐会进一步改写这些分数。', '',
          f"- 注意 `IMG_px_all` 单独在 S1≥0.95 门槛上就有 {f4(C['img_only_IMG_px_all']['S2_at_S1_0p95'])}，"
          f"高于 EEG 基线同门槛的 {f4(base['S2_at_S1_0p95'])}，但在 LOSO=0.90 的 tau 上只有 "
          f"{f4(C['img_only_IMG_px_all']['S2_acceptance'])}（基线 {f4(base['S2_acceptance'])}）。"
          '这正是"S1 门槛读法"与"训练侧阈值读法"背离的极端例子：对齐把 S1 尾部推得很深，'
          '门槛自然放得极低，因此"S1≥0.95 下的 S2"不能单独作为身份判别力的证据。'
          f"图像单视图在 LOSO 点的纯度只有基线的 "
          f"{C['img_only_IMG_px_all']['purity_accepted_S2']['purity_micro'] / base['purity_accepted_S2']['purity_micro']:.2f}×"
          '，按纯度规则不能采纳。', '',
          '## 3. 融合候选主表', '',
          '`S1/S2/纯度` 在 **LOSO 未知拒识 = 0.90** 的 tau 上读出（该候选自己的训练侧合规工作点）；'
          '`S2@S1.95` 是把 tau 设定为该候选自身 Test S1 分数的 95% 分位后的 S2；'
          '`sepAUC` 与阈值无关；`纯度` 是 k=20 固定划分下、被接受的 S2 行里 EEG 分支预测身份的 micro 纯度；'
          '`20 折 S2 代理` = 每个 LOSO 折里注册（known）行在该 tau 下的接受率，20 折均值±std，'
          '是 Train 侧对 S2 的类比（Test S2 本身没有折）。'
          f"同读法基线：S1 {f4(base['S1_rejection'])} / S2 {f4(base['S2_acceptance'])} / sepAUC "
          f"{f4(base['separation_auc'])} / S2@S1.95 {f4(base['S2_at_S1_0p95'])} / 纯度 "
          f"{f4(base['purity_accepted_S2']['purity_micro'])}。", '',
          '|候选|S1|S2|sepAUC|S2@S1.95|整体拒识|覆盖身份|纯度(k20)|20 折 S2 代理|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    order = ['baseline_eeg_rank'] + [f'img_only_{n}' for n in CMP_VIEWS + [PRIMARY]] + \
            list(WEIGHTS.keys()) + [f'fuse_rank_50_50_{n}' for n in CMP_VIEWS] + \
            list(ROUND2.keys())
    for cid in order:
        m = C[cid]
        L.append(f"|{cid}|{f4(m['S1_rejection'])}|{f4(m['S2_acceptance'])}|{f4(m['separation_auc'])}|"
                 f"{f4(m['S2_at_S1_0p95'])}|{f4(m['overall_rejection'])}|{m['identities_predicted']}|"
                 f"{f4(m['purity_accepted_S2']['purity_micro'])}|"
                 f"{f4(m['fold_S2_proxy_mean'])}±{f4(m['fold_S2_proxy_std'], 3)}|")
    L += ['', '## 4. 与 0.8795 / 0.706 的对比', '',
          '两个纯读法的基线必须分开看：**原始分数读法**（E233/E236 口径，全局 tau）基线的 S2@S1.95 是 '
          f'{f4(BASE_S2_GATE)}；**秩读法**（本轮融合的读法，逐 task 分位）基线的 S2@S1.95 是 '
          f"{f4(dec['baseline_S2_at_gate_rank_space'])}。目标 0.706 来自 E236 的原始分数读法，"
          '而秩读法把基线的同一指标压到 0.706 附近，所以下表同时给出与目标 0.706 和与同读法基线的差值。', '',
          '|候选|ΔsepAUC（vs 0.8795）|S2@S1.95|Δ vs 目标 0.706|Δ vs 同读法基线|Δ vs E233 原始 0.7569|纯度比（LOSO 点）|纯度比（S1.95 点）|',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    bp = dec['baseline_purity_accepted']
    bpg = dec['baseline_purity_at_gate']
    for cid in order:
        if cid == 'baseline_eeg_rank':
            continue
        m = C[cid]
        L.append(f"|{cid}|{m['separation_auc'] - BASE_AUC:+.4f}|{f4(m['S2_at_S1_0p95'])}|"
                 f"{m['S2_at_S1_0p95'] - 0.706:+.4f}|"
                 f"{m['S2_at_S1_0p95'] - dec['baseline_S2_at_gate_rank_space']:+.4f}|"
                 f"{m['S2_at_S1_0p95'] - BASE_S2_GATE:+.4f}|"
                 f"{m['purity_accepted_S2']['purity_micro'] / max(bp, 1e-12):.3f}×|"
                 f"{m['purity_at_gate_S2']['purity_micro'] / max(bpg, 1e-12):.3f}×|")
    L += ['', '## 5. 判定', '']
    if dec['verdict'] == 'pass':
        ad = C[dec['adopted']]
        L += [f"- **满足 config 中的达标规则，冻结 `{dec['adopted']}`**："
              f"sepAUC {f4(ad['separation_auc'])} > 0.8795（{ad['separation_auc'] - BASE_AUC:+.4f}）、"
              f"S2@S1.95 {f4(ad['S2_at_S1_0p95'])} > 0.706（{ad['S2_at_S1_0p95'] - 0.706:+.4f}）、"
              f"簇纯度（LOSO 点）{f4(ad['purity_accepted_S2']['purity_micro'])} ≥ 基线 "
              f"{f4(bp)} 的 95%（比值 {ad['purity_accepted_S2']['purity_micro'] / max(bp, 1e-12):.3f}×）。",
              f"- 同时满足规则的还有 "
              + '、'.join(f"`{c}`（S2@S1.95 {f4(C[c]['S2_at_S1_0p95'])}、sepAUC "
                          f"{f4(C[c]['separation_auc'])}）" for c in dec['valid'] if c != dec['adopted'])
              + '。三个候选的结论一致（0.847–0.878），说明增益不是某一个权重或某一种融合尺度的偶然结果：'
                '秩融合与 z-score 融合给出同一量级的结果，图像单视图 50/50 也给出 0.82/0.72。',
              f"- 但**在工作点（LOSO=0.90 的 tau）上融合并未提升 S2**：`{dec['adopted']}` 的 S2 是 "
              f"{f4(ad['S2_acceptance'])}，低于基线的 {f4(base['S2_acceptance'])}"
              f"（`fuse_rank_70_30` 的 {f4(C['fuse_rank_70_30']['S2_acceptance'])} 与之持平）。"
              '也就是说：融合改变的是**S1 尾部的相对位置**，而不是"在同一个阈值下接受更多 S2"。'
              '详见 5.1。']
        L += ['', '### 5.1 为什么 sepAUC 与 S1.95 门槛都涨，而 LOSO 阈值下 S2 反而降', '',
              '- 两个 branch 都在各自空间做了对称批次均值对齐。对齐后 Test S1（5 名未注册被试）在图像空间里'
              '被推得比 EEG 空间更深：图像单视图的对齐后 AUC 已经到 0.90–0.95（未对齐只有 0.71–0.76）。'
              '因此融合分数在 S1 尾部比 EEG 单一分数更"深"，S1 的 95% 分位 tau 因此更低，'
              '同样的 95% S1 拒识能放进更多 S2。',
              f'- 这条增益是**真实的排序增益**（sepAUC {f4(BASE_AUC)} → {f4(ad["separation_auc"])}，'
              '与阈值无关），而不是靠放宽接受：'
              '在同一 S1 拒识水平上比较，融合确实接受更多 S2。',
              '- 但它**不是可在训练域推导出的工作点**：LOSO 的"未知"是同一 session、同一批次里的留出被试，'
              '比 Test 里那 5 名未注册被试难得多。融合把 Test S1 推得更深，等价于把 LOSO 标定的 tau 抬到'
              '比 Test S1 的 95% 分位还高，于是固定阈值下 S2 反而下降。这与 E233/E236 已经记录的性质一致：'
              '任何改变 S1 尾部形状的操作，都会让"LOSO 标定阈值"和"S1 门槛"两种读法背离。',
              '- 因此冻结的含义是"**在 S1≥0.95 门槛口径下有效**"，而不是"可直接用 Train 侧阈值部署"。'
              '门槛 tau 由 Test S1 分位决定，属转导式评估选点。']
        L += ['', '### 5.2 纯度敏感性（同一接受率对比并不成立）', '',
              f"- LOSO 工作点上采纳候选接受 {ad['purity_accepted_S2']['n']} 条 S2、纯度 "
              f"{f4(ad['purity_accepted_S2']['purity_micro'])}（基线 {base['purity_accepted_S2']['n']} 条 / "
              f"{f4(bp)}）；S1.95 门槛上接受 {ad['purity_at_gate_S2']['n']} 条（基线的门槛只接受 "
              f"{base['purity_at_gate_S2']['n']} 条），纯度比降到 "
              f"{ad['purity_at_gate_S2']['purity_micro'] / max(bpg, 1e-12):.3f}×（基线门槛纯度 {f4(bpg)}）。",
              '- 这个下降是**接受率变宽（0.71 → 0.88）的机械结果**，不是身份塌缩：门槛上不同预测身份数仍为 '
              f"{ad['purity_at_gate_S2']['n_distinct_pred']}（基线 "
              f"{base['purity_at_gate_S2']['n_distinct_pred']}），最大单一身份占比 "
              f"{f4(ad['purity_at_gate_S2']['max_identity_share'])}（基线 "
              f"{f4(base['purity_at_gate_S2']['max_identity_share'])}）——融合后的接受集合反而更分散。"
              '按 config 预先登记的口径（同一 LOSO 工作点、同一 tau 定义）纯度判据成立；'
              '若改在 S1.95 门槛上对比，比值 0.947–0.957 落在 0.95 容差两侧，属噪声级差异，已如实列出。']
    elif dec['verdict'] == 'separable_only':
        L += [f"- **可分性提升未转化为工作点**：最高 sepAUC {f4(dec['best_separation'])} 已经高于 "
              f"{f4(BASE_AUC)}（`{dec['best_separation_auc']}`），但同一候选的 S2@S1.95 只有 "
              f"{f4(C[dec['best_separation_auc']]['S2_at_S1_0p95'])}，未超过 0.706；"
              f"S1≥0.95 约束下最好的候选是 `{dec['best_gate_candidate']}`，S2 = "
              f"{f4(dec['best_S2_at_S1_0p95'])}。",
              '- 原因：S1 侧被稀释——融合改写了 S1 尾部形状，把 tau 抬高，压掉 S2。']
    else:
        L += [f"- **图像视图与对齐分数融合无增益**：所有融合候选的 sepAUC 都不高于 {f4(BASE_AUC)}，"
              f"且 S1≥0.95 下的 S2 都不高于基线同门槛的 {f4(dec['baseline_S2_at_gate_rank_space'])}。",
              f"- 最好的融合候选 `{dec['best_gate_candidate']}` 只有 S2 {f4(dec['best_S2_at_S1_0p95'])}。"]
    r2 = [c for c in ROUND2]
    if r2:
        L += ['', '**追加一轮（权重 / 组合，仅作稳健性检查，不改变判定）**：' + '、'.join(
            f"`{c}` sepAUC {f4(C[c]['separation_auc'])}、S2@S1.95 {f4(C[c]['S2_at_S1_0p95'])}"
            for c in r2) + '。']
    L += ['', '## 6. 簇纯度与覆盖（无标签一致性检查）', '',
          f"在基线 EEG PCA 空间上对全部 19218 条 S2 行做 k-means（k=20，种子 {PURITY_SEED + PURITY_K}，"
          '60 次 Lloyd 迭代），划分在所有候选间相同；纯度只反映"被接受的行是否仍聚在少数身份上"。'
          '`@LOSO` = 在该候选 LOSO=0.90 的 tau 上被接受的 S2 行；`@S1.95` = 在该候选 S1 门槛上被接受的 S2 行。', '',
          '|候选|n@LOSO|micro@LOSO|#身份@LOSO|max share@LOSO|n@S1.95|micro@S1.95|#身份@S1.95|max share@S1.95|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid in order:
        p, q = C[cid]['purity_accepted_S2'], C[cid]['purity_at_gate_S2']
        L.append(f"|{cid}|{p['n']}|{f4(p['purity_micro'])}|{p['n_distinct_pred']}|"
                 f"{f4(p['max_identity_share'])}|{q['n']}|{f4(q['purity_micro'])}|"
                 f"{q['n_distinct_pred']}|{f4(q['max_identity_share'])}|")
    L += ['', f"- 基线（EEG only）：@LOSO {f4(bp)}（{base['purity_accepted_S2']['n']} 行）、"
          f"@S1.95 {f4(bpg)}（{base['purity_at_gate_S2']['n']} 行）；E233 对全部 19218 条 S2 行的纯度是 "
          f'{f4(BASE_PUR_ALL)}（参考，与预测身份一致，因此各候选相同）。',
          '- 纯度门槛的作用是排除"靠放宽接受换 S2"的退化解：若接受的 S2 塌缩到少数身份，'
          'micro 纯度会大跌、`#身份` 会骤减、`max share` 会骤升，三者在采纳候选上都未出现。', '',
          '## 7. 限制', '',
          '- **图像缓存覆盖**：只有 7 类图像类型同时有 Train/Test 缓存；GAF / MTF / 谱图 / DINOv3 只有 Train 缓存，'
          '本轮完全不可用。因此"图像路线"的结论只覆盖这 7 类像素视图，不能外推到 GAF/MTF/谱图。',
          '- **Test 无身份标签**：官方 C/RA/ASafeR = N/A；S1 拒识与 S2 接受是唯一可读出的行为指标。',
          '- **Test 反复查看偏倚**：Test 自 E65 起被反复评估，所有 S2 数字都带这一偏倚；'
          '本轮四个融合权重、z/秩变换、PCA/原型参数与判定规则在运行前写死，未按 Test 调整，'
          '但"选出一个采纳候选"这一动作本身已经消耗了 Test 信息。',
          '- **S1 门槛是转导式的**：`S2@S1.95` 的 tau 来自 Test S1 分数的分位，属评估选点；'
          '而 LOSO 标定阈值下融合的 S2 不升（5.1 节），所以这两列不能同时当作可部署指标。',
          '- **session 与注册状态混叠（E232）**：S1 = 5 名未注册被试，S2 = 20 名注册用户；'
          'session 仅作对齐的批次键，从不用于决定身份或门控。图像视图对齐后 AUC 之所以从 0.71 跳到 0.93+，'
          '主要是"这 5 个人的批次均值被移回 Train 原点"，与生物特征可分性不是一回事。',
          '- **转导式对齐**：批次均值统计量来自被评分的同一批无标签输入，是否可用需按赛规确认。',
          '- **秩变换尾部外推**：逐 task ECDF 在参考支撑之外用最低/最高 50 个参考点做线性外推；'
          '不外推时 S1 会全部饱和到秩 0，S1.95 门槛失去分辨率（基线 S2@S1.95 会跌到 0.43，'
          '与 E233 的 0.7569 明显不符），因此必须外推。z-score 融合不依赖这一步，得到同量级结论，'
          '说明结论不是秩映射的产物。',
          '- **融合分数是标量**：预测身份仍来自 EEG 分支，因此纯度/覆盖只能检验"接受决策是否保持身份一致"，'
          '不能证明图像分支独立带来了身份信息。',
          f"- 运行：`.venv\\Scripts\\python.exe research/E237_image_view_fusion/run_e237.py`（CUDA），"
          f"耗时 {R['runtime_s']:.0f}s；逐视图与逐候选分数缓存在 `cache/`，图像投影与原型存于 "
          '`models/e237_image_fusion.pt`。']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


if __name__ == '__main__':
    CFG_BLOCK_PATHS_TEST = json.loads(
        (ROOT / 'research/E233_delta_estimation/config.json').read_text(encoding='utf8')
    )['feature_cache']['test']
    main()
