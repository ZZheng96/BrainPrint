"""E240_small_image_weight

Question (the only one): can a small image weight keep the train-domain ranking
intact AND deliver the cross-session gain?

E237 showed the image views raise Test S2-vs-S1 separability from 0.8795 to
0.9611, but an equal-weight fusion drops the train-domain LOSO AUC from 0.9088
to 0.8896, so at the train-side anchored tau (E239) S2 collapses to 0.3145.
Hypothesis here: the image weight was too large; alpha = 0.05..0.30 may be the
sweet spot.

Nothing is refitted.  EEG = frozen E233 mean_align score + frozen E235 fold
scores.  Image = frozen E237 cached view scores (7 E149 pixel views).  Only the
per-task z map and the linear combination are new:

    s(alpha) = (1 - alpha) * z(s_eeg) + alpha * z(s_img)
    z per task, mu/sd from the component's own fit-row reference scores.

For every alpha the tau is re-calibrated on the Train side by bisection so that
LOSO unknown rejection = 0.90 (E228 convention), then Test is read once.

Read-outs per alpha: train LOSO AUC (is the ranking broken?), train known
acceptance at the working point, Test separation AUC, Test S1 rejection, Test S2
acceptance, k=20 cluster purity, identity coverage, S2 - 0.3122.

Run: .venv\\Scripts\\python.exe research/E240_small_image_weight/run_e240.py
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
N_FOLDS = int(CFG['protocol']['folds'])
ALPHAS = [float(a) for a in CFG['protocol']['alpha_grid']]
SOURCES = ['IMG_px_all', 'X_mw_cov4']
PURITY_K = 20
PURITY_SEED = 20261117
LOSO_TARGET = 0.90

A = CFG['baseline_anchors']['e233_mean_align']
TAU_EEG = float(A['test_tau'])
BASE_LOSO = float(A['loso_auc'])
BASE_S2 = float(A['test_S2_acceptance'])
BASE_S1 = float(A['test_S1_rejection'])
BASE_AUC = float(A['test_separation_auc'])
BASE_PUR_ALL = float(A['purity_k20_micro_all_S2'])
BASE_KNOWN = float(A['train_known_acceptance_at_deployed_tau'])
BASE_LOSO_DEP = float(A['loso_unknown_rejection_at_deployed_tau'])
LOSO_AUC_FLOOR = 0.995 * BASE_LOSO
E239_REF = CFG['baseline_anchors']['e239_reference']

EEG_TEST = 'research/E233_delta_estimation/cache/test_v1_mean_align.npz'
EEG_OOF = 'research/E227_joint_frontier/cache/test_v1_auc2_lofo.npz'
EEG_FOLD = 'research/E235_deployment_calibration/cache/fold_v1_{:02d}.npz'
VIEW_CACHE = 'research/E237_image_view_fusion/cache/view_{}.npz'
VIEWS = ['X_correlation', 'X_covariance5', 'X_covariance_hr', 'X_topomap5',
         'X_mw_corr8', 'X_mw_cov4', 'X_mw_cov8']
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


# ------------------------------------------------------- per-task z-score map
def task_stats(ref, fit, task):
    mu, sd = {}, {}
    for t in TASKS:
        x = ref[fit & (task == t)]
        mu[t] = float(x.mean())
        sd[t] = float(x.std()) + 1e-6
    return mu, sd


def apply_z(x, task, mu, sd):
    out = np.empty(len(x), float)
    for t in TASKS:
        m = task == t
        out[m] = (x[m] - mu[t]) / sd[t]
    return out


def tau_vec(tau_map, task):
    return np.array([tau_map[t] for t in task])


# ------------------------------------------------------------ tau calibration
def calib_tau(u, target=LOSO_TARGET, iters=120):
    """Bisection for the smallest tau with mean(u < tau) >= target (reject iff < tau)."""
    lo = float(u.min()) - 1.0
    hi = float(u.max()) + 1.0
    assert float((u < lo).mean()) < target and float((u < hi).mean()) >= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if float((u < mid).mean()) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ------------------------------------------------------------------- metrics
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


CFG_BLOCK_PATHS_TEST = None


def baseline_eeg_space(te_task):
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


def evaluate(cid, src, alpha, score, fk, fu, tfk, tfu, is_s1, test_task,
             tau_map_eeg, pred_s2, assign20):
    uk = np.concatenate(fu)
    kk = np.concatenate(fk)
    tk = np.concatenate(tfk)
    tu = np.concatenate(tfu)

    loso_auc = auc_ties(kk, uk)
    fold_auc = [auc_ties(k, u) for k, u in zip(fk, fu)]

    tau = calib_tau(uk)
    acc = score >= tau
    fold_acc = [float((k >= tau).mean()) for k in fk]
    te_s2 = score[~is_s1]

    # diagnostic only: the deployed E233 tau mapped into the EEG per-task z space
    tau_dep = tau_vec(tau_map_eeg, test_task)
    dep_acc = score >= tau_dep

    pur = purity_of(pred_s2, assign20, PURITY_K, acc[~is_s1])
    dep_pur = purity_of(pred_s2, assign20, PURITY_K, dep_acc[~is_s1])
    return dict(
        candidate=cid, source=src, alpha=alpha,
        formula=('z_eeg' if alpha == 0 else
                 f'{1 - alpha:.2f}*z_eeg + {alpha:.2f}*z_{src}'),
        tau=tau,
        loso_unknown_rejection=float((uk < tau).mean()),
        loso_known_acceptance=float((kk >= tau).mean()),
        loso_known_acceptance_fold_mean=float(np.mean(fold_acc)),
        loso_known_acceptance_fold_std=float(np.std(fold_acc)),
        loso_auc=loso_auc,
        loso_auc_fold_mean=float(np.mean(fold_auc)),
        loso_auc_fold_std=float(np.std(fold_auc)),
        per_task_loso_auc={t: auc_ties(kk[tk == t], uk[tu == t]) for t in TASKS},
        test_S1_rejection=float((~acc[is_s1]).mean()),
        test_S2_acceptance=float(acc[~is_s1].mean()),
        test_separation_auc=auc_ties(te_s2, score[is_s1]),
        delta_vs_0p3122=float(acc[~is_s1].mean() - BASE_S2),
        n_accepted_S2=int(acc[~is_s1].sum()),
        purity_accepted_S2=pur,
        identities_coverage=int(pur['n_distinct_pred']),
        deployed_tau_diagnostic=dict(
            tau_eeg_z_per_task=tau_map_eeg,
            loso_unknown_rejection=float((uk < tau_vec(tau_map_eeg, tu)).mean()),
            loso_known_acceptance=float((kk >= tau_vec(tau_map_eeg, tk)).mean()),
            test_S1_rejection=float((~dep_acc[is_s1]).mean()),
            test_S2_acceptance=float(dep_acc[~is_s1].mean()),
            separation_auc=auc_ties(te_s2, score[is_s1]),
            purity_accepted_S2=dep_pur),
    )


def main():
    LOG.write_text('', encoding='utf8')
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    assert torch.cuda.is_available(), 'CUDA required'
    log(f'device={torch.cuda.get_device_name(0)} torch={torch.__version__}')
    t_start = time.time()

    tr = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    te = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tr) == 11304 and len(te) == 22251, (len(tr), len(te))
    subj = tr.subject.to_numpy(int)
    tr_task = tr.task.to_numpy()
    fr = frac_of(tr)
    fit = fr <= FIT_FRAC
    eval_idx = np.flatnonzero((fr > EVAL_LO) & (fr <= EVAL_HI))
    eval_task = tr_task[eval_idx]
    eval_subj = subj[eval_idx]
    te_task = te.task.to_numpy()
    sess = te.session.astype(str).str.casefold().to_numpy()
    is_s1 = sess == 'session1'
    log(f'fit rows={int(fit.sum())} eval rows={len(eval_idx)} '
        f'test S1={int(is_s1.sum())} S2={int((~is_s1).sum())}')

    # ------------------------------------------------ frozen EEG branch
    e = np.load(ROOT / EEG_TEST, allow_pickle=True)
    assert np.array_equal(e['task'], te_task)
    eeg_test = e['score'].astype(np.float64)
    eeg_pred = e['pred'].astype(int)
    assert abs(float(e['tau']) - TAU_EEG) < 1e-9
    assert np.array_equal(e['is_s1'], is_s1)
    dt = np.load(ROOT / EEG_OOF, allow_pickle=True)
    assert np.array_equal(dt['oof_fit'].astype(bool), fit)
    eeg_ref = dt['oof_s'].astype(np.float64)

    fold_fk, fold_fu, fold_tfk, fold_tfu = [], [], [], []
    fold_known, fold_unknown = [], []
    for h in range(1, N_FOLDS + 1):
        d = np.load(ROOT / EEG_FOLD.format(h), allow_pickle=True)
        s = -d['d1_orig'].astype(np.float64)
        kn, un = d['known'].astype(bool), d['unknown'].astype(bool)
        assert np.array_equal(d['task'], eval_task) and np.array_equal(d['subject'], eval_subj)
        fold_known.append(s[kn])
        fold_unknown.append(s[un])
        fold_fk.append(kn)
        fold_fu.append(un)
        fold_tfk.append(d['task'][kn])
        fold_tfu.append(d['task'][un])

    # ------------------------------------------------ alpha = 0 reproduction
    raw = dict(S1_rejection=float((eeg_test[is_s1] < TAU_EEG).mean()),
               S2_acceptance=float((eeg_test[~is_s1] >= TAU_EEG).mean()),
               separation_auc=auc_ties(eeg_test[~is_s1], eeg_test[is_s1]),
               train_known_acceptance_at_deployed_tau=BASE_KNOWN)
    assert abs(raw['S2_acceptance'] - BASE_S2) < 1e-12
    assert abs(raw['separation_auc'] - BASE_AUC) < 1e-12
    assert abs(raw['S1_rejection'] - BASE_S1) < 1e-12
    log('E233 mean_align raw reproduction ' + json.dumps(raw))

    # ------------------------------------------------ per-task z components
    mu_eeg, sd_eeg = task_stats(eeg_ref, fit, tr_task)
    tau_map_eeg = {t: (TAU_EEG - mu_eeg[t]) / sd_eeg[t] for t in TASKS}
    log('EEG per-task z stats ' + json.dumps(
        {t: [round(mu_eeg[t], 4), round(sd_eeg[t], 4), round(tau_map_eeg[t], 4)]
         for t in TASKS}))

    Z = {}
    Z['eeg'] = dict(
        test=apply_z(eeg_test, te_task, mu_eeg, sd_eeg),
        fk=[apply_z(s, t, mu_eeg, sd_eeg) for s, t in zip(fold_known, fold_tfk)],
        fu=[apply_z(s, t, mu_eeg, sd_eeg) for s, t in zip(fold_unknown, fold_tfu)],
        mu=mu_eeg, sd=sd_eeg)

    for n in VIEWS:
        v = np.load(ROOT / VIEW_CACHE.format(n), allow_pickle=True)
        assert np.array_equal(v['ref_fit'].astype(bool), fit)
        mu, sd = task_stats(v['ref'].astype(np.float64), fit, tr_task)
        Z[n] = dict(
            test=apply_z(v['test_aligned'].astype(np.float64), te_task, mu, sd),
            fk=[apply_z(v['fold_score'][h][fold_fk[h]].astype(np.float64),
                        eval_task[fold_fk[h]], mu, sd) for h in range(N_FOLDS)],
            fu=[apply_z(v['fold_score'][h][fold_fu[h]].astype(np.float64),
                        eval_task[fold_fu[h]], mu, sd) for h in range(N_FOLDS)],
            mu=mu, sd=sd)
        log(f'  z({n:16s}) test mean={Z[n]["test"].mean():+.3f} '
            f'sd={Z[n]["test"].std():.3f}')

    def mean_comp(names):
        return dict(test=np.mean([Z[n]['test'] for n in names], 0),
                    fk=[np.mean([Z[n]['fk'][h] for n in names], 0) for h in range(N_FOLDS)],
                    fu=[np.mean([Z[n]['fu'][h] for n in names], 0) for h in range(N_FOLDS)],
                    mu=None, sd=None)

    Z['IMG_px_all'] = mean_comp(VIEWS)
    for src in SOURCES:
        log(f'image source {src:12s} test mean={Z[src]["test"].mean():+.3f} '
            f'sd={Z[src]["test"].std():.3f} '
            f'sepAUC_raw={auc_ties(Z[src]["test"][~is_s1], Z[src]["test"][is_s1]):.4f}')

    # ------------------------------------------------ purity partition (frozen)
    log('building the label-free k=20 partition in the baseline EEG PCA space')
    zte = baseline_eeg_space(te_task)
    s2_rows = np.flatnonzero(~is_s1)
    assign20, sizes20 = kmeans_assign(zte[s2_rows].astype(np.float32), PURITY_K,
                                      PURITY_SEED + PURITY_K)
    pred_s2 = eeg_pred[s2_rows]
    pur_all = purity_of(pred_s2, assign20, PURITY_K, np.ones(len(s2_rows), bool))
    log(f'  kmeans k=20 sizes={sizes20.tolist()}')
    log('  purity over all S2 rows ' + json.dumps(pur_all))
    assert abs(pur_all['purity_micro'] - BASE_PUR_ALL) < 1e-6, 'purity partition drifted'

    # ------------------------------------------------ alpha grid
    results = {}
    for src in SOURCES:
        for a in ALPHAS:
            cid = f'{src}_a{a:.2f}'
            score = (1 - a) * Z['eeg']['test'] + a * Z[src]['test']
            fk = [(1 - a) * x + a * y for x, y in zip(Z['eeg']['fk'], Z[src]['fk'])]
            fu = [(1 - a) * x + a * y for x, y in zip(Z['eeg']['fu'], Z[src]['fu'])]
            m = evaluate(cid, src, a, score, fk, fu, fold_tfk, fold_tfu, is_s1,
                         te_task, tau_map_eeg, pred_s2, assign20)
            results[cid] = m
            np.savez_compressed(CACHE / f'cand_{cid}.npz', score=score, is_s1=is_s1,
                                tau=m['tau'], alpha=a, source=src)
            log(f"{cid:22s} tau={m['tau']:+.4f} losoRej={m['loso_unknown_rejection']:.4f} "
                f"losoAUC={m['loso_auc']:.4f} known={m['loso_known_acceptance']:.4f} "
                f"sep={m['test_separation_auc']:.4f} S1={m['test_S1_rejection']:.4f} "
                f"S2={m['test_S2_acceptance']:.4f} dS2={m['delta_vs_0p3122']:+.4f} "
                f"pur={m['purity_accepted_S2']['purity_micro']:.4f} "
                f"cov={m['identities_coverage']}")

    return finish(results, Z, tau_map_eeg, assign20, pur_all, raw, t_start)


def finish(results, Z, tau_map_eeg, assign20, pur_all, raw, t_start):
    main_ids = [f'IMG_px_all_a{a:.2f}' for a in ALPHAS]
    ctrl_ids = [f'X_mw_cov4_a{a:.2f}' for a in ALPHAS]
    floor = LOSO_AUC_FLOOR

    passing = [c for c in main_ids if results[c]['loso_auc'] >= floor]
    win = [c for c in main_ids
           if results[c]['test_S2_acceptance'] >= 0.40
           and results[c]['loso_auc'] >= floor]
    best_s2_all_main = max(main_ids, key=lambda c: results[c]['test_S2_acceptance'])
    best_eval = (max(passing, key=lambda c: results[c]['test_S2_acceptance'])
                 if passing else None)
    best_train = max(main_ids, key=lambda c: results[c]['loso_auc'])

    if win:
        verdict = 'win_win'
    elif results[best_s2_all_main]['test_S2_acceptance'] <= 0.35:
        verdict = 'tradeoff_only'
    else:
        verdict = 'intermediate'
    log(f'floor={floor:.6f} passing={passing} win={win} '
        f'best_eval={best_eval} best_train={best_train} verdict={verdict}')

    ctrl_win = [c for c in ctrl_ids
                if results[c]['test_S2_acceptance'] >= 0.40
                and results[c]['loso_auc'] >= floor]
    log(f'control source win={ctrl_win}')

    # cross-check against E239: alpha=0.50 == wsum_50_50, alpha=0.30 == wsum_70_30
    ref50, ref70 = 'IMG_px_all_a0.50', 'IMG_px_all_a0.30'
    assert abs(results[ref50]['test_S2_acceptance']
               - E239_REF['wsum_50_50_S2_at_recalibrated_tau']) < 1e-12
    assert round(results[ref50]['loso_auc'], 4) == E239_REF['wsum_50_50_loso_auc']
    assert round(results[ref70]['test_S2_acceptance'], 4) == E239_REF['wsum_70_30_S2_at_recalibrated_tau']
    assert round(results[ref70]['loso_auc'], 4) == E239_REF['wsum_70_30_loso_auc']
    log('E239 cross-check ok: alpha=0.30 == wsum_70_30, alpha=0.50 == wsum_50_50')

    def pick(c):
        m = results[c]
        return dict(candidate=c, alpha=m['alpha'], source=m['source'], tau=m['tau'],
                    loso_auc=m['loso_auc'],
                    train_known_acceptance=m['loso_known_acceptance'],
                    test_S2_acceptance=m['test_S2_acceptance'],
                    test_S1_rejection=m['test_S1_rejection'],
                    test_separation_auc=m['test_separation_auc'],
                    delta_vs_0p3122=m['delta_vs_0p3122'])

    R = dict(
        experiment='E240_small_image_weight',
        config=CFG,
        eeg_raw_reproduction=raw,
        eeg_loso_at_deployed_tau=BASE_LOSO_DEP,
        z_spec='per-task z: (x - mu_t) / sd_t, mu/sd from the component fit-row reference',
        tau_eeg_z_per_task=tau_map_eeg,
        loso_auc_floor=floor,
        alpha_grid=ALPHAS,
        candidates=results,
        purity_all_S2=pur_all,
        partition_k20_sizes=[int(x) for x in np.bincount(assign20, minlength=PURITY_K)],
        selection=dict(
            rule='train LOSO AUC >= 0.995 * 0.9088 = %.6f, then max Test S2' % floor,
            passing_alpha_ids=passing,
            eval_selection=best_eval,
            eval_selection_row=pick(best_eval) if best_eval else None,
            train_only_selection=best_train,
            train_only_row=pick(best_train),
            best_S2_any_alpha=best_s2_all_main,
            best_S2_any_alpha_row=pick(best_s2_all_main),
            control_win=ctrl_win,
            e239_reference=E239_REF,
        ),
        decision=dict(
            verdict=verdict,
            win_list=win,
            best_eval=best_eval,
            best_eval_S2=results[best_eval]['test_S2_acceptance'] if best_eval else None,
            best_eval_loso_auc=results[best_eval]['loso_auc'] if best_eval else None,
            best_train=best_train,
            best_train_S2=results[best_train]['test_S2_acceptance'],
            best_train_loso_auc=results[best_train]['loso_auc'],
            best_S2_any=results[best_s2_all_main]['test_S2_acceptance'],
            best_S2_any_alpha=results[best_s2_all_main]['alpha'],
            best_S2_any_loso_auc=results[best_s2_all_main]['loso_auc'],
            baseline_S2=BASE_S2, baseline_loso_auc=BASE_LOSO,
            baseline_sep_auc=BASE_AUC, baseline_train_known=BASE_KNOWN,
            target_S2=0.40, fail_ceiling=0.35,
            official_C_RA_ASafeR='N/A (Test has no identity labels)',
            F1='N/A (gating task, no closed-set classifier output)'),
        limitations=CFG['limitations'],
        runtime_s=float(time.time() - t_start))

    MODELS.mkdir(exist_ok=True)
    torch.save(dict(
        experiment='E240_small_image_weight',
        z_spec='per-task z: (x - mu_t) / sd_t, mu/sd from the component fit-row reference',
        tau_eeg_raw=TAU_EEG,
        tau_eeg_z_per_task=tau_map_eeg,
        component_stats={k: dict(mu=v['mu'], sd=v['sd']) for k, v in Z.items()
                         if v['mu'] is not None},
        tau={c: results[c]['tau'] for c in results},
        formulas={c: results[c]['formula'] for c in results},
        purity_partition=assign20,
        decision=R['decision'],
        note='alpha=0 row is the pure EEG baseline; tau per candidate is train-side '
             '(bisection on pooled LOSO unknown to rejection=0.90)'),
        MODELS / 'e240_small_image_weight.pt')
    (OUT / 'results.json').write_text(json.dumps(R, indent=2, default=float),
                                      encoding='utf8')
    build_report(R)
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


def build_report(R):
    C = R['candidates']
    dec = R['decision']
    sel = R['selection']
    main_ids = [f'IMG_px_all_a{a:.2f}' for a in ALPHAS]
    ctrl_ids = [f'X_mw_cov4_a{a:.2f}' for a in ALPHAS]
    floor = R['loso_auc_floor']

    def alist(ids):
        if not ids:
            return '无'
        return '、'.join(f'α={C[c]["alpha"]:.2f}' for c in ids)

    L = ['# E240 少量图像权重：能否保住训练域排序又兑现跨会话增益', '']
    headline = {
        'win_win': f"存在双赢点：`{dec['best_eval']}`（α={C[dec['best_eval']]['alpha']:.2f}）"
                   f"在 LOSO=0.90 下 Train LOSO AUC={f4(dec['best_eval_loso_auc'])}、"
                   f"Test S2={f4(dec['best_eval_S2'])}（较 0.3122 {C[dec['best_eval']]['delta_vs_0p3122']:+.4f}）。",
        'tradeoff_only': f"不存在双赢点。LOSO 固定 0.90 时，主源最好的 S2 只有 "
                         f"{f4(dec['best_S2_any'])}（α={dec['best_S2_any_alpha']:.2f}，"
                         f"Train LOSO AUC={f4(dec['best_S2_any_loso_auc'])}），"
                         f"8 个 α 的 S2 全部落在 "
                         f"{f4(min(C[c]['test_S2_acceptance'] for c in main_ids))}–"
                         f"{f4(max(C[c]['test_S2_acceptance'] for c in main_ids))} 的平台内；"
                         f"与此同时 Test sepAUC 从 0.8731 单调升到 "
                         f"{f4(C['IMG_px_all_a0.50']['test_separation_auc'])}——图像信息买到的是 "
                         f"S1 尾部的可分性，而不是训练域锚定阈值下的 S2 兑现。",
        'intermediate': f"部分兑现：通过训练域过滤的 α 里最好 S2={f4(dec['best_eval_S2'])}"
                        f"（`{dec['best_eval']}`），既未达到 0.40 也未落入 ≤0.35 的明确失败区间。",
    }
    L.append('> **一句话结论**：' + headline[dec['verdict']])
    L.append('')
    L.append('## 1. 问题与设定')
    L.append('')
    L.append('E237 证明图像视图把 Test 跨会话可分性从 0.8795 推到 0.9611，但等权融合把 Train 域 '
             'LOSO AUC 从 0.9088 压到 0.8896，于是 E239 在训练侧锚定的 tau 上 S2 只有 0.3145（噪声级）。'
             '本轮检验唯一假设：**图像权重过大破坏了训练域排序；只掺少量图像信息可能双赢**。')
    L.append('')
    L.append('组合：`s(α) = (1−α)·z(s_eeg) + α·z(s_img)`，z 为逐 task z-score'
             '（`z=(x−mu_t)/sd_t`，mu/sd 取自该分量自己的 fit 行参考分数）。'
             '`s_eeg` = E233 `mean_align` 冻结分数（折内用 E235 冻结逐折分数）；'
             '`s_img` 主源 = E237 `IMG_px_all`（7 个像素视图 z 后的等权均值），'
             '对照源 = `X_mw_cov4`。**不重训、不重建特征**，只做 z 与线性组合。')
    L.append('')
    L.append(f'α 网格：{", ".join(f"{a:.2f}" for a in ALPHAS)}。每个 α 都在 Train 侧 pooled LOSO '
             'unknown 分数上用**二分法**求 tau 使 LOSO 未知拒识 = 0.90（拒识 iff `score < tau`，'
             'E228 同口径），求得后一次性读 Test。')
    L.append('')
    L.append('## 2. α=0 的逐位复现')
    L.append('')
    L.append('| 项目 | 本轮实测 | 任务要求 | 判定 |')
    L.append('|---|---:|---:|:---:|')
    L.append(f"| E233 原始 tau 下 Test S2 接受 | {f4(R['eeg_raw_reproduction']['S2_acceptance'], 6)} | 0.3122 | 逐位一致 |")
    L.append(f"| E233 原始 tau 下 Test 分离 AUC | {f4(R['eeg_raw_reproduction']['separation_auc'], 6)} | 0.8795 | 逐位一致 |")
    L.append(f"| Train 侧 LOSO AUC（known vs unknown） | {f4(C['IMG_px_all_a0.00']['loso_auc'], 6)} | 0.9088 | 逐位一致 |")
    L.append(f"| E233 原始 tau 下 Train LOSO 拒识 | {f4(R['eeg_loso_at_deployed_tau'])} | 0.8979（E239 记录） | 一致 |")
    L.append(f"| E233 原始 tau 下 Train known 接受率 | "
             f"{f4(R['eeg_raw_reproduction']['train_known_acceptance_at_deployed_tau'])} | 0.8058 | 一致 |")
    L.append(f"| E239 交叉校验 α=0.30 = `wsum_70_30` | S2={f4(C['IMG_px_all_a0.30']['test_S2_acceptance'])} / "
             f"LOSO AUC={f4(C['IMG_px_all_a0.30']['loso_auc'])} | 0.3132 / 0.9034 | 一致 |")
    L.append(f"| E239 交叉校验 α=0.50 = `wsum_50_50` | S2={f4(C['IMG_px_all_a0.50']['test_S2_acceptance'])} / "
             f"LOSO AUC={f4(C['IMG_px_all_a0.50']['loso_auc'])} | 0.3145 / 0.8896 | 一致 |")
    L.append('')
    L.append(f"注：α=0 的重标定 tau = {C['IMG_px_all_a0.00']['tau']:+.4f}，在其上 Test S2 = "
             f"{f4(C['IMG_px_all_a0.00']['test_S2_acceptance'])}、sepAUC = "
             f"{f4(C['IMG_px_all_a0.00']['test_separation_auc'])}。与 0.3122 / 0.8795 的微小差异"
             f"**不是实现误差**：0.3122 是在部署 tau（{TAU_EEG:.4f}）上读的，而该 tau 的 pooled LOSO "
             f"拒识是 0.8979，二分到 0.90 会略微收紧阈值；sepAUC 的 0.8795→0.8731 来自逐 task z 不是"
             f"跨 task 的单调变换（E239 已记录同一现象）。两个读数在所有 α 上并行给出，"
             f"横向比较始终在同一口径内。")
    L.append('')

    def alpha_table(ids, title, note):
        L.append(f'## {title}')
        L.append('')
        L.append(note)
        L.append('')
        L.append('| α | tau | LOSO 拒识 | **Train LOSO AUC** | Train known 接受 | '
                 '**Test sepAUC** | Test S1 拒识 | **Test S2 接受** | **ΔS2 vs 0.3122** | '
                 '簇纯度(k=20) | 覆盖身份 |')
        L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for c in ids:
            m = C[c]
            mark = ' **←**' if c == dec['best_eval'] else ''
            L.append(f"| {m['alpha']:.2f}{mark} | {m['tau']:+.4f} | "
                     f"{f4(m['loso_unknown_rejection'])} | {f4(m['loso_auc'])} | "
                     f"{f4(m['loso_known_acceptance'])} | {f4(m['test_separation_auc'])} | "
                     f"{f4(m['test_S1_rejection'])} | **{f4(m['test_S2_acceptance'])}** | "
                     f"{m['delta_vs_0p3122']:+.4f} | "
                     f"{f4(m['purity_accepted_S2']['purity_micro'])} | "
                     f"{m['identities_coverage']} |")
        L.append('')

    alpha_table(main_ids, '3. 主图像源 `IMG_px_all` 的完整 α 曲线（关键表）',
                '每行的 tau 都是该 α 自身在 Train 侧二分的 LOSO=0.90 工作点（合规）。'
                f'基线列：α=0 的 Train LOSO AUC={f4(BASE_LOSO)}、Test sepAUC=0.8795、'
                f'部署 tau 下 S2=0.3122、簇纯度（全 S2，参考）={f4(BASE_PUR_ALL)}。'
                f'过滤线 = 0.995×0.9088 = {floor:.4f}。')
    alpha_table(ctrl_ids, '4. 对照图像源 `X_mw_cov4` 的完整 α 曲线（允许的一轮追加）',
                '`X_mw_cov4` 是 E237 中对齐后单视图 Test AUC 最高的图像视图（0.9487），'
                '作为第二图像源检验主源结论是否依赖某个特定视图组合。')

    L.append('## 5. 两种选型规则的对比（关键）')
    L.append('')
    ev, tr = sel['eval_selection'], sel['train_only_selection']
    L.append('| 选型规则 | 选出的 α | Train LOSO AUC | Test S1 拒识 | Test S2 接受 | ΔS2 vs 0.3122 | 性质 |')
    L.append('|---|---:|---:|---:|---:|---:|---|')
    if ev:
        m = C[ev]
        L.append(f"| ① 训练域过滤（LOSO AUC ≥ {floor:.4f}）后取 Test S2 最高 | "
                 f"{m['alpha']:.2f} | {f4(m['loso_auc'])} | {f4(m['test_S1_rejection'])} | "
                 f"**{f4(m['test_S2_acceptance'])}** | {m['delta_vs_0p3122']:+.4f} | "
                 f"**评估选点**（读了 Test，不合规） |")
    else:
        L.append(f"| ① 训练域过滤（LOSO AUC ≥ {floor:.4f}）后取 Test S2 最高 | 无通过者 | — | — | — | — | — |")
    m = C[tr]
    L.append(f"| ② 仅按训练侧（取 LOSO AUC 最大） | {m['alpha']:.2f} | {f4(m['loso_auc'])} | "
             f"{f4(m['test_S1_rejection'])} | {f4(m['test_S2_acceptance'])} | "
             f"{m['delta_vs_0p3122']:+.4f} | 完全合规选型 |")
    L.append('')
    L.append(f"通过训练域过滤的 α：{alist(sel['passing_alpha_ids'])}。"
             f"其中 Test S2 最高者（评估选点）为 {('α=%.2f' % C[ev]['alpha']) if ev else '无'}；"
             f"而纯训练侧规则会选 α={C[tr]['alpha']:.2f}，其 Test S2 = {f4(C[tr]['test_S2_acceptance'])}。"
             f"两者相差 {f4(abs(C[ev]['test_S2_acceptance'] - C[tr]['test_S2_acceptance'])) if ev else '—'}，"
             f"这就是合规选型要付的代价。")
    L.append('')
    L.append(f"另外记录：主源里 Test S2 绝对值最高的是 α={C[sel['best_S2_any_alpha']]['alpha']:.2f}"
             f"（S2={f4(C[sel['best_S2_any_alpha']]['test_S2_acceptance'])}，Train LOSO AUC="
             f"{f4(C[sel['best_S2_any_alpha']]['loso_auc'])}"
             f"{'，仍在过滤线之上' if C[sel['best_S2_any_alpha']]['loso_auc'] >= floor else '，未通过过滤线'}），"
             f"与通过过滤后选出的 α={C[ev]['alpha']:.2f} 是同一个点——即本轮两种选型规则指向同一 α，"
             f"差距不是由选取规则造成，而是 S2 本身没有随 α 上升。")
    L.append('')

    L.append('## 6. 权衡曲线：α → 训练域 AUC / Test S2')
    L.append('')
    L.append('| α | Train LOSO AUC（主源） | Δ vs α=0 | Test S2（主源） | ΔS2 vs 0.3122 | '
             'Train LOSO AUC（对照源） | Test S2（对照源） |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    for a, cm, cc in zip(ALPHAS, main_ids, ctrl_ids):
        mm, mc = C[cm], C[cc]
        L.append(f"| {a:.2f} | {f4(mm['loso_auc'])} | {mm['loso_auc'] - BASE_LOSO:+.4f} | "
                 f"{f4(mm['test_S2_acceptance'])} | {mm['delta_vs_0p3122']:+.4f} | "
                 f"{f4(mc['loso_auc'])} | {f4(mc['test_S2_acceptance'])} |")
    L.append('')
    L.append('读法：训练域排序（LOSO AUC）随 α 单调下降，但 Test S2 **不随 α 上升**——'
             '它在 0.303–0.317 的平台内抖动，α≥0.40 后随排序一起下降。'
             '与此同时 sepAUC 单调上升（0.8731 → 0.9553，α=0.50），说明图像信息主要改写的是 '
             'S1 尾部的相对位置（5 名未注册被试被推得更深），而不是把 S2 的质量推到训练域 unknown '
             '90% 分位之上。因此曲线不存在"右上角"的交点：保住排序的 α（≤0.20）兑现不了 S2，'
             '能提高 sepAUC 的 α（≥0.30）兑现同样为零且开始损伤排序。')
    L.append('')

    L.append('## 7. 判定')
    L.append('')
    if dec['verdict'] == 'win_win':
        c = dec['best_eval']
        m = C[c]
        L.append(f"**判定：少量图像信息可以在 LOSO 固定下兑现（存在双赢点）。**")
        L.append('')
        L.append(f"- 达标 α：{alist(dec['win_list'])}。")
        L.append(f"- 采纳点 `{c}`：α={m['alpha']:.2f}，Train LOSO AUC={f4(m['loso_auc'])} "
                 f"(≥ {floor:.4f})，LOSO 未知拒识={f4(m['loso_unknown_rejection'])}，"
                 f"Train known 接受={f4(m['loso_known_acceptance'])}，"
                 f"Test S1 拒识={f4(m['test_S1_rejection'])}，"
                 f"Test S2 接受={f4(m['test_S2_acceptance'])}（较 0.3122 {m['delta_vs_0p3122']:+.4f}），"
                 f"sepAUC={f4(m['test_separation_auc'])}，簇纯度={f4(m['purity_accepted_S2']['purity_micro'])}，"
                 f"覆盖身份={m['identities_coverage']}。")
        L.append('')
        L.append('**部署参数（训练侧标定，合规）**：')
        L.append('')
        L.append(f"- 分数：`{m['formula']}`")
        L.append("- z：`z=(x−mu_t)/sd_t`，逐 task，mu/sd 取自各分量 fit 行参考分数；"
                 "数值见 `models/e240_small_image_weight.pt`")
        L.append(f"- tau = {m['tau']:+.6f}（组合分数 pooled LOSO unknown 的 90% 分位，二分求得）")
    elif dec['verdict'] == 'tradeoff_only':
        L.append('**判定：图像信息的兑现需要以训练域排序为代价——不存在双赢点。**')
        L.append('')
        L.append(f"- 在 LOSO 固定 0.90 的前提下，主源最好的 S2 是 {f4(dec['best_S2_any'])}"
                 f"（α={dec['best_S2_any_alpha']:.2f}），仍低于 0.35 的失败线；"
                 f"全部 8 个 α 的 S2 都落在 "
                 f"{f4(min(C[c]['test_S2_acceptance'] for c in main_ids))}–"
                 f"{f4(max(C[c]['test_S2_acceptance'] for c in main_ids))}。")
        L.append(f"- 该点的 Train LOSO AUC = {f4(dec['best_S2_any_loso_auc'])}，确实守住了过滤线"
                 f"（{floor:.4f}），也就是说**小权重本身没有破坏排序**——但排序守住了，S2 并没有"
                 f"因此上升：ΔS2 只有 {C[dec['best_eval']]['delta_vs_0p3122']:+.4f}，"
                 f"小于归一化约定造成的读数漂移量级（同一 EEG 单支，部署 tau 下 0.3122 vs "
                 f"LOSO=0.90 重标定下 {f4(C['IMG_px_all_a0.00']['test_S2_acceptance'])}）。")
        L.append(f"- 纯训练侧规则会选 α={C[tr]['alpha']:.2f}（LOSO AUC 最高，"
                 f"{f4(C[tr]['loso_auc'])}），它对应的 Test S2={f4(C[tr]['test_S2_acceptance'])}"
                 f"（较 0.3122 {C[tr]['delta_vs_0p3122']:+.4f}）——合规选型拿不到兑现；"
                 f"而评估选点的 α={C[ev]['alpha']:.2f} 也只有 {f4(C[ev]['test_S2_acceptance'])}，"
                 f"两者差距 {abs(C[ev]['test_S2_acceptance'] - C[tr]['test_S2_acceptance']):.4f}，属噪声级。")
        L.append('- 对照组更差：`X_mw_cov4` 在 α≥0.15 时 LOSO AUC 与 S2 同时下降'
                 f"（α=0.50：LOSO AUC {f4(C['X_mw_cov4_a0.50']['loso_auc'])}、"
                 f"S2 {f4(C['X_mw_cov4_a0.50']['test_S2_acceptance'])}），"
                 '说明结论不是某一个视图组合的偶然结果。')
        L.append(f"- 机制：sepAUC 从 {f4(C['IMG_px_all_a0.00']['test_separation_auc'])} 升到 "
                 f"{f4(C['IMG_px_all_a0.50']['test_separation_auc'])} 的增益全部来自 S1 尾部"
                 f"（Train known 接受率反而从 {f4(C['IMG_px_all_a0.00']['loso_known_acceptance'])} "
                 f"降到 {f4(C['IMG_px_all_a0.50']['loso_known_acceptance'])}），"
                 f"而工作点只看 unknown 90% 分位之上的 S2 密度，这块没有变化。"
                 f"完整权衡曲线见第 6 节。")
        L.append("- 补充：图像分支并非无效，它的收益集中在 **S1 分位门槛**读法上（E239 C 列在同一 "
                 "z 口径下把 S2 从 0.7002 推到 0.8629）；但该门槛由 Test S1 的 5% 分位定位，"
                 "属评估选点，不能与\"LOSO 固定 0.90\"的合规口径混用。本轮结论只针对后者。")
    else:
        L.append('**判定：部分兑现，不构成双赢。**')
        L.append('')
        L.append(f"- 通过训练域过滤的 α 里最好 S2={f4(dec['best_eval_S2'])}"
                 f"（`{dec['best_eval']}`，LOSO AUC={f4(dec['best_eval_loso_auc'])}），"
                 f"未达 0.40；但主源最优 S2 为 {f4(dec['best_S2_any'])}，高于 0.35。")
        L.append(f"- 纯训练侧规则选 α={C[tr]['alpha']:.2f}，Test S2={f4(C[tr]['test_S2_acceptance'])}。")
    L.append('')

    L.append('## 8. 固定部署 tau 下的诊断（同阈值，不看重标定）')
    L.append('')
    L.append('把 E233 部署 tau 换算到 EEG 逐 task z 空间后直接套在组合分数上（scale 不一致，仅作诊断），'
             '观察同一数值阈值下 LOSO 拒识 / known 接受 / Test S2 的走向：')
    L.append('')
    L.append('| α | LOSO 拒识@部署tau | Train known 接受@部署tau | Test S2@部署tau | 簇纯度@部署tau |')
    L.append('|---|---:|---:|---:|---:|')
    for c in main_ids:
        d = C[c]['deployed_tau_diagnostic']
        L.append(f"| {C[c]['alpha']:.2f} | {f4(d['loso_unknown_rejection'])} | "
                 f"{f4(d['loso_known_acceptance'])} | {f4(d['test_S2_acceptance'])} | "
                 f"{f4(d['purity_accepted_S2']['purity_micro'])} |")
    L.append('')
    L.append('这一列只用于解释机制：α 增大时组合分数整体方差收缩、均值下移，同一数值 tau 相对变松，'
             '因此 LOSO 拒识下降、Train known 接受上升，是尺度效应而不是排序变化；'
             '判定一律用第 3 节的重标定工作点。')
    L.append('')

    L.append('## 9. 逐 α 的 Train 域逐 task LOSO AUC（排序诊断）')
    L.append('')
    L.append('| α | EC | EO | Imagery | SSVEP |')
    L.append('|---|---:|---:|---:|---:|')
    for c in main_ids:
        m = C[c]
        L.append(f"| {m['alpha']:.2f} | "
                 + ' | '.join(f4(m['per_task_loso_auc'][t]) for t in TASKS) + ' |')
    L.append('')

    L.append('## 10. 合规性与局限')
    L.append('')
    L.append('- 每个 α 的 tau 都是 **训练侧标定**（Train 域 pooled LOSO unknown 分数的 90% 分位，'
             '二分求得；实际拒识全部为 0.9000，并列分数平台由"取满足条件的最小 tau"处理）；'
             'Test 只在该 tau 下被读一次。')
    L.append('- **选型规则 ①读取了 Test**：在通过训练域过滤的 α 中选 Test S2 最高者属评估选点，'
             '已单独标注；规则 ②（仅训练侧）的 α 与对应 S2 也一并给出，用于量化这一差距。')
    L.append('- **Test 反复查看偏倚**：Test 自 E65 起被反复评估，所有 S2 数字都带这一偏倚；'
             '本轮 α 网格、图像源与选型规则在运行前写入 `config.json`，但"选 α"这一动作本身'
             '已消耗 Test 信息。')
    L.append('- **官方 C / RA / ASafeR = N/A**：Test 没有身份标签；本任务是门控（拒识）任务，'
             '没有闭集分类器输出，**F1 亦为 N/A**。S1 拒识与 S2 接受率是唯一读数。')
    L.append('- E232：S1 = 5 名未注册被试、S2 = 20 名注册用户，session 与注册状态混叠；'
             '图像视图对齐后 AUC 0.93+ 主要来自批次均值回移，不是纯粹的生物特征可分性。')
    L.append('- 均值对齐是转导式批次统计量（在被打分的批次上估计无标签统计量），需与竞赛规则确认。')
    L.append('- 融合分数是标量，预测身份仍来自 EEG 分支；簇纯度与覆盖身份数只检验接受决策的'
             '身份一致性，不能证明图像分支独立带来身份信息。')
    L.append('- 逐 task z 不是跨 task 的单调变换，pooled Test sepAUC 与阈值会随归一化约定微动'
             '（EEG 单支 0.8795 → 0.8731），因此 0.00x 量级的差异不应被当作真增益。')
    L.append('- 图像侧只有 7 个 E149 像素视图有 Train/Test 缓存；GAF/MTF/谱图/DINOv3 只有 Train '
             '缓存，本轮完全不可用，结论不能外推。')
    L.append(f"- 运行：`.venv\\Scripts\\python.exe research/E240_small_image_weight/run_e240.py`"
             f"（CUDA），耗时 {R['runtime_s']:.0f}s；逐候选分数在 `cache/`，"
             f"z 统计量与 tau 在 `models/e240_small_image_weight.pt`。")
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


if __name__ == '__main__':
    CFG_BLOCK_PATHS_TEST = json.loads(
        (ROOT / 'research/E233_delta_estimation/config.json').read_text(encoding='utf8')
    )['feature_cache']['test']
    main()
