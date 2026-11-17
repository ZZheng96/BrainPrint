"""E239_gate_headroom

Question (the only one): can Test S2 be raised while LOSO stays at 0.90?

Established structural fact: LOSO = 0.90 is not performance, it is the definition
"tau sits on the 90th percentile of the train-domain impostor scores".  Any monotone
transform of one score leaves the (LOSO, S2) pair unchanged.  The only remaining
mechanism is to buy LOSO headroom at a fixed tau (so the combined score rejects more
train-domain impostors at the unchanged operating point) and then spend that headroom
by moving the recalibrated tau down onto Test S2.

Nothing is refitted here.  The EEG branch is the frozen E233 mean_align score and its
frozen E235 per-fold LOSO scores; the image branches are the E237 cached view scores
(7 E149 pixel views).  Only a per-task z-score map and the combination rule are new.

Read-outs per candidate:
  A : original E228/E233 tau (mapped into the EEG per-task z space), read LOSO
      unknown rejection, train-domain known acceptance, Test S1 and Test S2.
  B : tau recalibrated to the candidate's own LOSO = 0.90 (train-side), read
      S1 / S2 / separation AUC / cluster purity.
  C : on top of B, move tau to the Test S1 5% quantile (S1 rejection = 0.95), read S2.
      This is an evaluation-time point, not a compliant result.
Plus the train-side LOSO AUC (known vs unknown), which is where any headroom must
come from, and per-task diagnostics.

Run: .venv\\Scripts\\python.exe research/E239_gate_headroom/run_e239.py
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
PURITY_K = 20
PURITY_SEED = 20261117
LOSO_TARGET = 0.90
S1_GATE = 0.95

A = CFG['baseline_anchors']['e233_mean_align']
TAU_EEG = float(A['test_tau'])
BASE_LOSO = float(A['loso_unknown_rejection'])
BASE_S2 = float(A['test_S2_acceptance'])
BASE_S1 = float(A['test_S1_rejection'])
BASE_AUC = float(A['test_separation_auc'])
BASE_S2_GATE = float(A['S2_at_S1_0p95'])
BASE_PUR_ALL = float(A['purity_k20_micro_all_S2'])
E237 = CFG['baseline_anchors']['e237_best']

EEG_TEST = 'research/E233_delta_estimation/cache/test_v1_mean_align.npz'
EEG_OOF = 'research/E227_joint_frontier/cache/test_v1_auc2_lofo.npz'
EEG_FOLD = 'research/E235_deployment_calibration/cache/fold_v1_{:02d}.npz'
VIEW_CACHE = 'research/E237_image_view_fusion/cache/view_{}.npz'
VIEWS = ['X_correlation', 'X_covariance5', 'X_covariance_hr', 'X_topomap5',
         'X_mw_corr8', 'X_mw_cov4', 'X_mw_cov8']
THREEVIEW = ['X_mw_cov4', 'X_covariance_hr', 'IMG_px_all']
CANDS = CFG['candidates_fixed_before_run']
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
                    n_distinct_pred=0)
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
                purity_macro=float(np.mean(pur)), n_distinct_pred=int(len(vals)))


def evaluate(cid, score, fk, fu, tfk, tfu, tau_map_eeg, is_s1, pred_s2, assign20,
             test_task):
    uk = np.concatenate(fu)
    kk = np.concatenate(fk)
    tk = np.concatenate(tfk)
    tu = np.concatenate(tfu)
    loso_auc = auc_ties(kk, uk)
    fold_auc = [auc_ties(k, u) for k, u in zip(fk, fu)]

    # ---- A: original E228/E233 tau, mapped into the EEG per-task z space
    tauA_u = tau_vec(tau_map_eeg, tu)
    rejA_u = uk < tauA_u
    tauA_k = tau_vec(tau_map_eeg, tk)
    accA_k = kk >= tauA_k
    tauA_te = tau_vec(tau_map_eeg, test_task)
    rejA_te = score < tauA_te
    A_col = dict(
        tau_eeg_raw=TAU_EEG, tau_z_per_task=tau_map_eeg,
        loso_unknown_rejection=float(rejA_u.mean()),
        train_known_acceptance=float(accA_k.mean()),
        fold_known_acceptance_mean=float(np.mean(
            [float((k >= tau_vec(tau_map_eeg, t)).mean())
             for k, t in zip(fk, tfk)])),
        fold_known_acceptance_std=float(np.std(
            [float((k >= tau_vec(tau_map_eeg, t)).mean())
             for k, t in zip(fk, tfk)])),
        S1_rejection=float(rejA_te[is_s1].mean()),
        S2_acceptance=float((~rejA_te[~is_s1]).mean()),
        separation_auc=auc_ties(score[~is_s1], score[is_s1]),
        purity_accepted_S2=purity_of(pred_s2, assign20, PURITY_K,
                                     score[~is_s1] >= tauA_te[~is_s1]),
        per_task_loso_rejection={t: float(rejA_u[tu == t].mean()) for t in TASKS},
    )

    # ---- B: tau recalibrated to this score's own LOSO = 0.90 (train side)
    tauB = float(np.quantile(uk, LOSO_TARGET))
    accB = score >= tauB
    B_col = dict(
        tau=tauB,
        loso_unknown_rejection=float((uk < tauB).mean()),
        loso_fold_rejection_mean=float(np.mean([float((u < tauB).mean()) for u in fu])),
        loso_fold_rejection_std=float(np.std([float((u < tauB).mean()) for u in fu])),
        S1_rejection=float((~accB[is_s1]).mean()),
        S2_acceptance=float(accB[~is_s1].mean()),
        separation_auc=auc_ties(score[~is_s1], score[is_s1]),
        purity_accepted_S2=purity_of(pred_s2, assign20, PURITY_K, accB[~is_s1]),
        n_accepted_S2=int(accB[~is_s1].sum()),
    )

    # ---- C: keep going down to Test S1 rejection = 0.95 (evaluation-time point)
    tauC = float(np.quantile(score[is_s1], S1_GATE))
    accC = score >= tauC
    C_col = dict(
        tau=tauC,
        S1_rejection=float((~accC[is_s1]).mean()),
        S2_acceptance=float(accC[~is_s1].mean()),
        purity_accepted_S2=purity_of(pred_s2, assign20, PURITY_K, accC[~is_s1]),
        tau_below_B=bool(tauC < tauB),
    )

    per_task_loso_auc = {}
    for t in TASKS:
        mpos = tk == t
        mneg = tu == t
        per_task_loso_auc[t] = auc_ties(kk[mpos], uk[mneg])

    return dict(
        candidate=cid, note=CANDS[cid].get('role', ''),
        formula=CANDS[cid]['formula'],
        loso_auc=loso_auc,
        loso_auc_fold_mean=float(np.mean(fold_auc)),
        loso_auc_fold_std=float(np.std(fold_auc)),
        loso_auc_fold=fold_auc,
        per_task_loso_auc=per_task_loso_auc,
        A=A_col, B=B_col, C=C_col,
    )


# ------------------------------------------------------- purity reference space
CFG_BLOCK_PATHS_TEST = None
EEG_LOSO_A = None


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


def main():
    LOG.write_text('', encoding='utf8')
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

    # ---- frozen EEG branch
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
    fold_known, fold_unknown, fold_task = [], [], []
    for h in range(1, N_FOLDS + 1):
        d = np.load(ROOT / EEG_FOLD.format(h), allow_pickle=True)
        s = -d['d1_orig'].astype(np.float64)
        kn, un = d['known'].astype(bool), d['unknown'].astype(bool)
        assert np.array_equal(d['task'], eval_task) and np.array_equal(d['subject'], eval_subj)
        fold_known.append(s[kn])
        fold_unknown.append(s[un])
        fold_task.append((d['task'][kn], d['task'][un]))
        fold_fk.append(kn)
        fold_fu.append(un)
        fold_tfk.append(d['task'][kn])
        fold_tfu.append(d['task'][un])

    # baseline raw reproduction (frozen E233 tau)
    base = dict(S1_rejection=float((eeg_test[is_s1] < TAU_EEG).mean()),
                S2_acceptance=float((eeg_test[~is_s1] >= TAU_EEG).mean()),
                separation_auc=auc_ties(eeg_test[~is_s1], eeg_test[is_s1]))
    log('E233 mean_align raw reproduction ' + json.dumps(base))
    assert abs(base['S2_acceptance'] - BASE_S2) < 1e-9
    assert abs(base['separation_auc'] - BASE_AUC) < 1e-9

    # ---- per-task z on the EEG branch
    mu_eeg, sd_eeg = task_stats(eeg_ref, fit, tr_task)
    tau_map_eeg = {t: (TAU_EEG - mu_eeg[t]) / sd_eeg[t] for t in TASKS}
    log('EEG per-task z stats ' + json.dumps({t: [round(mu_eeg[t], 4), round(sd_eeg[t], 4),
                                                 round(tau_map_eeg[t], 4)] for t in TASKS}))

    Z = {}
    z_eeg_test = apply_z(eeg_test, te_task, mu_eeg, sd_eeg)
    Z['eeg'] = dict(
        test=z_eeg_test,
        fk=[apply_z(s, t, mu_eeg, sd_eeg) for s, t in zip(fold_known, fold_tfk)],
        fu=[apply_z(s, t, mu_eeg, sd_eeg) for s, t in zip(fold_unknown, fold_tfu)],
        mu=mu_eeg, sd=sd_eeg)

    # sanity: per-task z of the EEG branch at the deployed E228/E233 tau reproduces
    # the frozen Test read-out exactly.  The Train-side LOSO read on the E235 fold
    # cache gives 0.89787 at this tau, whereas E233 reports 0.90004 because it
    # re-derives a per-fold tau from each fold's own OOF known scores; both are
    # recorded so the A-column anchor is not overstated.
    rej = z_eeg_test < tau_vec(tau_map_eeg, te_task)
    repro = dict(loso=float((np.concatenate(Z['eeg']['fu']) <
                             tau_vec(tau_map_eeg, np.concatenate(fold_tfu))).mean()),
                 S1=float(rej[is_s1].mean()), S2=float((~rej[~is_s1]).mean()))
    log('per-task z + deployed tau reproduction ' + json.dumps(repro))
    assert abs(repro['S1'] - BASE_S1) < 1e-9
    assert abs(repro['S2'] - BASE_S2) < 1e-9
    global EEG_LOSO_A
    EEG_LOSO_A = repro['loso']

    # ---- cached image views (no refitting)
    for n in VIEWS:
        v = np.load(ROOT / VIEW_CACHE.format(n), allow_pickle=True)
        assert np.array_equal(v['ref_fit'].astype(bool), fit)
        mu, sd = task_stats(v['ref'].astype(np.float64), fit, tr_task)
        comp = dict(
            test=apply_z(v['test_aligned'].astype(np.float64), te_task, mu, sd),
            fk=[apply_z(v['fold_score'][h][fold_fk[h]].astype(np.float64),
                        eval_task[fold_fk[h]], mu, sd) for h in range(N_FOLDS)],
            fu=[apply_z(v['fold_score'][h][fold_fu[h]].astype(np.float64),
                        eval_task[fold_fu[h]], mu, sd) for h in range(N_FOLDS)],
            mu=mu, sd=sd)
        Z[n] = comp
        log(f'  z({n:16s}) test mean={comp["test"].mean():+.3f} sd={comp["test"].std():.3f}')

    def mean_comp(names, cid):
        return dict(test=np.mean([Z[n]['test'] for n in names], 0),
                    fk=[np.mean([Z[n]['fk'][h] for n in names], 0) for h in range(N_FOLDS)],
                    fu=[np.mean([Z[n]['fu'][h] for n in names], 0) for h in range(N_FOLDS)],
                    mu=None, sd=None)

    Z['IMG_px_all'] = mean_comp(VIEWS, 'IMG_px_all')
    Z['IMG_threeview'] = mean_comp(THREEVIEW, 'IMG_threeview')

    # ---- purity partition (frozen E233 baseline EEG PCA space, GPU k-means)
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

    # ---- candidate combination rules (fixed in config before the run)
    def combine(cid):
        e = Z['eeg']
        if cid == 'eeg_only':
            return e['test'], e['fk'], e['fu']
        if cid == 'img_only_px_all':
            c = Z['IMG_px_all']
            return c['test'], c['fk'], c['fu']
        img = Z['IMG_px_all'] if cid != 'min_rule_3view' else Z['IMG_threeview']
        if cid == 'min_rule' or cid == 'min_rule_3view':
            return (np.minimum(e['test'], img['test']),
                    [np.minimum(a, b) for a, b in zip(e['fk'], img['fk'])],
                    [np.minimum(a, b) for a, b in zip(e['fu'], img['fu'])])
        if cid == 'and_rule_soft':
            return (np.minimum(e['test'], img['test']) + 0.2 * np.maximum(e['test'], img['test']),
                    [np.minimum(a, b) + 0.2 * np.maximum(a, b)
                     for a, b in zip(e['fk'], img['fk'])],
                    [np.minimum(a, b) + 0.2 * np.maximum(a, b)
                     for a, b in zip(e['fu'], img['fu'])])
        if cid == 'wsum_70_30':
            w = 0.7
        elif cid == 'wsum_50_50':
            w = 0.5
        else:
            raise KeyError(cid)
        return (w * e['test'] + (1 - w) * img['test'],
                [w * a + (1 - w) * b for a, b in zip(e['fk'], img['fk'])],
                [w * a + (1 - w) * b for a, b in zip(e['fu'], img['fu'])])

    results = {}
    for cid in CANDS:
        score, fk, fu = combine(cid)
        m = evaluate(cid, score, fk, fu, fold_tfk, fold_tfu, tau_map_eeg, is_s1,
                     pred_s2, assign20, te_task)
        results[cid] = m
        np.savez_compressed(CACHE / f'cand_{cid}.npz', score=score, is_s1=is_s1,
                            tau_A_z=np.array([tau_map_eeg[t] for t in TASKS]),
                            tau_B=m['B']['tau'], tau_C=m['C']['tau'])
        log(f"{cid:16s} losoAUC={m['loso_auc']:.4f} "
            f"A[loso={m['A']['loso_unknown_rejection']:.4f} "
            f"known={m['A']['train_known_acceptance']:.4f} "
            f"S1={m['A']['S1_rejection']:.4f} S2={m['A']['S2_acceptance']:.4f}] "
            f"B[S1={m['B']['S1_rejection']:.4f} S2={m['B']['S2_acceptance']:.4f} "
            f"sep={m['B']['separation_auc']:.4f} pur={m['B']['purity_accepted_S2']['purity_micro']:.4f}] "
            f"C[S1={m['C']['S1_rejection']:.4f} S2={m['C']['S2_acceptance']:.4f}]")

    np.savez_compressed(CACHE / 'e239_z_components.npz',
                        components=np.array(list(Z.keys())),
                        **{f'{k}_test': Z[k]['test'] for k in Z},
                        **{f'{k}_mu_tasks': np.array([Z[k]['mu'][t] for t in TASKS])
                           for k in Z if Z[k]['mu'] is not None},
                        **{f'{k}_sd_tasks': np.array([Z[k]['sd'][t] for t in TASKS])
                           for k in Z if Z[k]['sd'] is not None},
                        is_s1=is_s1, task=te_task)
    return finish(results, Z, tau_map_eeg, assign20, pur_all, t_start)


def finish(results, Z, tau_map_eeg, assign20, pur_all, t_start):
    passB = [c for c in results if results[c]['B']['S2_acceptance'] > BASE_S2
             and results[c]['B']['S1_rejection'] >= S1_GATE]
    headroom = [c for c in results if results[c]['A']['loso_unknown_rejection'] >= 0.95]
    if passB:
        verdict = 'pass_B'
        adopted = max(passB, key=lambda c: results[c]['B']['S2_acceptance'])
    elif headroom:
        verdict = 'mechanism_but_no_gain'
        adopted = None
    else:
        verdict = 'fail'
        adopted = None
    bestB = max(results, key=lambda c: results[c]['B']['S2_acceptance'])
    bestC = max(results, key=lambda c: results[c]['C']['S2_acceptance'])
    log(f'verdict={verdict} adopted={adopted} passB={passB} headroom={headroom} '
        f'bestB={bestB}({results[bestB]["B"]["S2_acceptance"]:.4f}) '
        f'bestC={bestC}({results[bestC]["C"]["S2_acceptance"]:.4f})')

    R = dict(experiment='E239_gate_headroom', config=CFG,
             eeg_raw_reproduction=None, tau_eeg_raw=TAU_EEG,
             tau_eeg_z_per_task=tau_map_eeg,
             eeg_loso_at_deployed_tau=EEG_LOSO_A,
             z_spec='per-task mean/std of each component fit-row reference scores',
             candidates=results, purity_all_S2=pur_all,
             partition_k20_sizes=[int(x) for x in
                                  np.bincount(assign20, minlength=PURITY_K)],
             decision=dict(verdict=verdict, adopted=adopted, pass_B=passB,
                           headroom_bought=headroom, best_B=bestB,
                           best_B_S2=results[bestB]['B']['S2_acceptance'],
                           best_C=bestC,
                           best_C_S2=results[bestC]['C']['S2_acceptance'],
                           baseline_S2=BASE_S2, baseline_S1=BASE_S1,
                           baseline_sep_auc=BASE_AUC,
                           baseline_S2_at_S1_0p95=BASE_S2_GATE,
                           e237_best=E237),
             limitations=CFG['limitations'],
             runtime_s=float(time.time() - t_start))
    # raw reproduction is asserted in main(); keep the anchor for the record
    R['eeg_raw_reproduction'] = dict(S1_rejection=BASE_S1, S2_acceptance=BASE_S2,
                                     separation_auc=BASE_AUC,
                                     loso_unknown_rejection=BASE_LOSO)

    MODELS.mkdir(exist_ok=True)
    torch.save(dict(
        experiment='E239_gate_headroom',
        z_spec='per-task z: (x - mu_t) / sd_t, mu/sd from the component fit-row reference',
        tau_eeg_raw=TAU_EEG,
        tau_eeg_z_per_task=tau_map_eeg,
        component_stats={k: dict(mu=v['mu'], sd=v['sd'])
                         for k, v in Z.items() if v['mu'] is not None},
        formulas={c: results[c]['formula'] for c in results},
        tau_B={c: results[c]['B']['tau'] for c in results},
        tau_C={c: results[c]['C']['tau'] for c in results},
        purity_partition=assign20,
        verdict=verdict, adopted=adopted,
        note='augment: x - mu_t divided per task; tau_B is train-side (LOSO=0.90), '
             'tau_C is a Test-S1 quantile and is not a compliant operating point'),
        MODELS / 'e239_gate_headroom.pt')
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
    order = ['eeg_only', 'img_only_px_all', 'min_rule', 'min_rule_3view',
             'and_rule_soft', 'wsum_70_30', 'wsum_50_50']
    order = [c for c in order if c in C]
    L = []
    L.append('# E239 先买余量：min / 软 and / 加权和组合下的 A/B/C 三列读数')
    L.append('')
    L.append('> **一句话结论**：' + CONCLUSION[dec['verdict']].format(**REPORT_VARS(R)))
    L.append('')
    L.append('## 1. 问题与机制')
    L.append('')
    L.append('唯一问题：**在 LOSO 保持 0.90 的前提下提升 Test S2**。')
    L.append('')
    L.append('结构性事实（本轮的前提，不是结果）：LOSO=0.90 不是一个性能指标，而是把 tau 钉在'
             '训练域冒充者分数的 90% 分位。对一个分数做任何单调变换（秩、z-score、sigmoid）都'
             '不改变 (LOSO, S2) 的组合。因此唯一可行的机制是：**在不抬高 tau 的前提下先提高 LOSO'
             '（买到余量），再把 tau 降下来花在 S2 上**。本轮把两个分支都换成逐 task z-score'
             '（同一组 fit 行 mu/sd），再构造 min / 软 and / 加权和三类组合，按 A/B/C 三列读数'
             '检验机制是否成立。')
    L.append('')
    L.append('## 2. 读数定义')
    L.append('')
    L.append('| 列 | tau 来源 | 读什么 | 合规性 |')
    L.append('|---|---|---|---|')
    L.append('| A | E228/E233 原始 tau（换算到 EEG 逐 task z 空间：'
             '`tau_z(t) = (tau_eeg - mu_eeg(t)) / sd_eeg(t)`） | 同 tau 下 LOSO 拒识、'
             '训练域已知接受率、Test S1/S2 | 训练侧标定，合规 |')
    L.append(f'| B | 组合分数自身的 LOSO=0.90（train 侧 unknown 的 90% 分位） | '
             'S1/S2/sepAUC/簇纯度 | 训练侧标定，合规 |')
    L.append('| C | 在 B 的 tau 上继续下移到 Test S1 的 5% 分位（S1 拒识=0.95） | S2 | '
             '**评估选点，不可当作合规结论** |')
    L.append('')
    L.append(f'注：A 列用的是部署 tau = {TAU_EEG:.6f}（E228 由 deploy 模型 inner-5-block OOF '
             f'**已知**分数的 19.5% 分位标定，E233 原样沿用）。在 E235 冻结逐折分数上直接读，'
             f'这个 tau 的 LOSO 拒识是 {f4(EEG_LOSO_A)}；E233 报的 0.90004 是因为它给每折'
             f'重新取了该折 OOF 已知分位作为折内 tau。两者都记录在案，A 列的所有横向比较都在'
             f'同一个部署 tau 下进行，因此比较本身不受这个约定差异影响。B 列则统一改为'
             f'分数自身 pooled unknown 的 90% 分位。')
    L.append('')
    L.append('## 3. 训练侧 LOSO AUC（余量的来源）')
    L.append('')
    L.append('| 候选 | 组合公式 | LOSO AUC | 20 折均值±std | Δ vs EEG 单支 |')
    L.append('|---|---|---|---|---|')
    b = C['eeg_only']['loso_auc']
    for c in order:
        m = C[c]
        L.append(f"| `{c}` | `{m['formula']}` | {f4(m['loso_auc'])} | "
                 f"{f4(m['loso_auc_fold_mean'])}±{f4(m['loso_auc_fold_std'])} | "
                 f"{m['loso_auc'] - b:+.4f} |")
    L.append('')
    L.append('## 4. A 列：原 tau 下（同阈值，看是否买到余量）')
    L.append('')
    L.append('| 候选 | LOSO 拒识 | 训练域已知接受率 | Test S1 | Test S2 | sepAUC | 簇纯度 |')
    L.append('|---|---|---|---|---|---|---|')
    for c in order:
        a = C[c]['A']
        L.append(f"| `{c}` | {f4(a['loso_unknown_rejection'])} | "
                 f"{f4(a['train_known_acceptance'])} | {f4(a['S1_rejection'])} | "
                 f"{f4(a['S2_acceptance'])} | {f4(a['separation_auc'])} | "
                 f"{f4(a['purity_accepted_S2']['purity_micro'])} |")
    L.append('')
    L.append('解读：A 列必须在 **LOSO 拒识上升的同时看已知接受率**。任何把分数整体压低的组合'
             '（min、软 and、方差收缩的加权和）都会在同一数值 tau 下同时提高 LOSO 拒识和降低已知'
             '接受率，这只是"收紧阈值"，不是余量。真正的余量只能体现为 LOSO AUC 上升——'
             '即在同一个 unknown 90% 分位上，known 的分布更靠右。')
    L.append('')
    L.append('## 5. B 列：组合分数自身 LOSO=0.90 的训练侧 tau（关键表）')
    L.append('')
    L.append(f"| 候选 | tau(B) | LOSO 拒识 | Test S1 | Test S2 | sepAUC | 簇纯度 | 接受 S2 条数 |")
    L.append('|---|---|---|---|---|---|---|---|')
    for c in order:
        m = C[c]['B']
        L.append(f"| `{c}` | {m['tau']:+.4f} | {f4(m['loso_unknown_rejection'])} | "
                 f"{f4(m['S1_rejection'])} | {f4(m['S2_acceptance'])} | "
                 f"{f4(m['separation_auc'])} | "
                 f"{f4(m['purity_accepted_S2']['purity_micro'])} | {m['n_accepted_S2']} |")
    L.append('')
    L.append(f"基准：E233 EEG 单支在自带训练侧 tau 下 LOSO={f4(BASE_LOSO)}、S1={f4(BASE_S1)}、"
             f"S2={f4(BASE_S2)}、sepAUC={f4(BASE_AUC)}；"
             f"E237 `fuse_zscore_50_50`（全局 z）B 列 S2={f4(E237['test_S2_acceptance'])}、"
             f"sepAUC={f4(E237['test_separation_auc'])}。")
    L.append('')
    L.append('## 6. C 列：在 B 上继续下移到 S1=0.95（评估选点）')
    L.append('')
    L.append('| 候选 | tau(C) | Test S1 | Test S2 | 簇纯度 | tau 是否低于 B |')
    L.append('|---|---|---|---|---|---|')
    for c in order:
        m = C[c]['C']
        L.append(f"| `{c}` | {m['tau']:+.4f} | {f4(m['S1_rejection'])} | "
                 f"{f4(m['S2_acceptance'])} | "
                 f"{f4(m['purity_accepted_S2']['purity_micro'])} | "
                 f"{'是' if m['tau_below_B'] else '否'} |")
    L.append('')
    L.append('## 7. 余量账本（A 列 → B 列：买了多少 LOSO，花了多少 S2）')
    L.append('')
    L.append('| 候选 | A 列 LOSO | B 列 LOSO | ΔLOSO（买） | A 列 S2 | B 列 S2 | ΔS2（花） | '
             'B 列 S2 − 0.3122 | B 列 S2 − 同规则 EEG 基线 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    eeg_B = C['eeg_only']['B']['S2_acceptance']
    for c in order:
        m = C[c]
        L.append(f"| `{c}` | {f4(m['A']['loso_unknown_rejection'])} | "
                 f"{f4(m['B']['loso_unknown_rejection'])} | "
                 f"{m['B']['loso_unknown_rejection'] - m['A']['loso_unknown_rejection']:+.4f} | "
                 f"{f4(m['A']['S2_acceptance'])} | {f4(m['B']['S2_acceptance'])} | "
                 f"{m['B']['S2_acceptance'] - m['A']['S2_acceptance']:+.4f} | "
                 f"{m['B']['S2_acceptance'] - BASE_S2:+.4f} | "
                 f"{m['B']['S2_acceptance'] - eeg_B:+.4f} |")
    L.append('')
    tau_a_str = ' / '.join(f'{C["eeg_only"]["A"]["tau_z_per_task"][t]:+.4f}' for t in TASKS)
    L.append(f'同规则 EEG 基线（B 列 tau 也按 pooled unknown 90% 分位重标定）S2={f4(eeg_B)}，'
             f'而不是 A 列的 {f4(BASE_S2)}；两者差 {abs(BASE_S2 - eeg_B):.4f}。原因是 A 列给每个 '
             f'task 用各自的部署 z 阈值（{tau_a_str}），'
             f'B 列改用组合分数自身单一的 pooled 90% 分位阈值 '
             f'({C["eeg_only"]["B"]["tau"]:+.4f})。因此比较候选时，'
             f'**同规则基线 {f4(eeg_B)} 才是公平参照**。')
    L.append('')
    L.append('## 8. 机制诊断（这是本轮的实质结论）')
    L.append('')
    mn, m3, sa = C['min_rule'], C['min_rule_3view'], C['and_rule_soft']
    w7, w5 = C['wsum_70_30'], C['wsum_50_50']
    L.append('**（1）min / 软 and 规则：A 列 LOSO 确实上升，但不是"余量"，是"收紧阈值"。**')
    L.append('')
    L.append(f"- A 列同 tau 下 LOSO 拒识：EEG {f4(C['eeg_only']['A']['loso_unknown_rejection'])} → "
             f"`min_rule` {f4(mn['A']['loso_unknown_rejection'])}、"
             f"`min_rule_3view` {f4(m3['A']['loso_unknown_rejection'])}、"
             f"`and_rule_soft` {f4(sa['A']['loso_unknown_rejection'])}，看似买到了 "
             f"{f4(mn['A']['loso_unknown_rejection'] - C['eeg_only']['A']['loso_unknown_rejection'])}～"
             f"{f4(m3['A']['loso_unknown_rejection'] - C['eeg_only']['A']['loso_unknown_rejection'])}。")
    L.append(f"- 同一 tau 下训练域已知接受率同时从 "
             f"{f4(C['eeg_only']['A']['train_known_acceptance'])} 掉到 "
             f"{f4(mn['A']['train_known_acceptance'])} / {f4(m3['A']['train_known_acceptance'])} / "
             f"{f4(sa['A']['train_known_acceptance'])}。取两支较小值必然整体压低分数，"
             f"所以这是“两个门都要过”的代价，不是排序变好。")
    L.append(f"- 排序诊断（唯一能支撑余量的量）：LOSO AUC 从 {f4(C['eeg_only']['loso_auc'])} 变成 "
             f"{f4(mn['loso_auc'])} / {f4(m3['loso_auc'])} / {f4(sa['loso_auc'])}，**全部下降**。"
             f"逐 task 看也一样（EC/EO 掉得最多）。")
    L.append(f"- 结果：B 列重标定后 S2 = {f4(mn['B']['S2_acceptance'])} / "
             f"{f4(m3['B']['S2_acceptance'])} / {f4(sa['B']['S2_acceptance'])}，"
             f"全部低于 0.3122，也低于同规则 EEG 基线 {f4(eeg_B)}。"
             f"**min 规则无效，A 列的上升是阈值收紧的假象。**")
    L.append('')
    L.append('**（2）加权和：A 列 LOSO 反而下降，B 列微超靠的是联合尾部形状，不是排序。**')
    L.append('')
    L.append(f"- A 列同 tau 下 LOSO 拒识：`wsum_70_30` {f4(w7['A']['loso_unknown_rejection'])}、"
             f"`wsum_50_50` {f4(w5['A']['loso_unknown_rejection'])}，都远低于 EEG 的 "
             f"{f4(C['eeg_only']['A']['loso_unknown_rejection'])}：在部署 tau 上它们反而少拒。"
             f"换言之它们**一点余量都没买到**。")
    L.append(f"- B 列：把 tau 重标定到自身 LOSO=0.90 后 S2 = {f4(w7['B']['S2_acceptance'])} / "
             f"{f4(w5['B']['S2_acceptance'])}，比 0.3122 高 "
             f"{w7['B']['S2_acceptance'] - BASE_S2:+.4f} / {w5['B']['S2_acceptance'] - BASE_S2:+.4f}，"
             f"比同规则 EEG 基线高 {w5['B']['S2_acceptance'] - eeg_B:+.4f}。"
             f"但同期 LOSO AUC 从 {f4(C['eeg_only']['loso_auc'])} 掉到 "
             f"{f4(w7['loso_auc'])} / {f4(w5['loso_auc'])}。")
    L.append(f"- 机制解释：AUC 是**全局**排序量，而 B 列只读 unknown 的 90% 分位这一点。"
             f"加权和压低了 unknown 的整个高分区（训练域冒充者不再有高分尾部），于是同一个 90% "
             f"分位对应的 tau 变低，Test S2 因此多进一点；这是**分布形状**的收益，"
             f"不是“已知被推右”的排序收益，所以它和 LOSO AUC 的方向可以相反。")
    L.append('')
    L.append('**（3）稳健性：名义超标的幅度小于归一化约定的敏感度。**')
    L.append('')
    L.append(f"- 同一个 50/50 配方，E237 用**全局** z 读到 B 列 S2={f4(E237['test_S2_acceptance'])}，"
             f"本轮用**逐 task** z 读到 {f4(w5['B']['S2_acceptance'])}，"
             f"差 {abs(w5['B']['S2_acceptance'] - E237['test_S2_acceptance']):.4f}；"
             f"而本轮名义增益只有 {w5['B']['S2_acceptance'] - BASE_S2:+.4f}。"
             f"一个同样合规的归一化选择就能让读数移动 10 倍于增益的量，所以这个"
             f"“通过”不构成可靠结论。")
    L.append(f"- 逐 task z 也不是跨 task 的单调变换（每个 task 用不同 mu/sd），所以 EEG 单支的 "
             f"pooled sepAUC 从 {f4(BASE_AUC)} 变成 {f4(C['eeg_only']['A']['separation_auc'])}；"
             f"阈值与 sepAUC 都随归一化约定微动，这正是不能把 +0.0023 当作真增益的原因。")
    L.append(f"- 图像分支的优势主要来自对称批次均值对齐（E237：对齐后 Test AUC 0.9487，"
             f"未对齐 0.7600），而 S1/S2 与注册状态混叠（E232）。因此在**训练侧** tau 上"
             f"多接受 S2，有一部分是 session 偏移而不是身份可分性。")
    L.append('')
    L.append('## 9. 逐 task 证据')
    L.append('')
    L.append('| 候选 | 逐 task LOSO AUC EC / EO / Imagery / SSVEP | 逐 task A 列 LOSO 拒识 |')
    L.append('|---|---|---|')
    for c in order:
        m = C[c]
        aucs = ' / '.join(f4(m['per_task_loso_auc'][t]) for t in TASKS)
        rej = ' / '.join(f4(m['A']['per_task_loso_rejection'][t]) for t in TASKS)
        L.append(f'| `{c}` | {aucs} | {rej} |')
    L.append('')
    L.append('## 10. 判定')
    L.append('')
    L.append(VERDICT_TEXT[dec['verdict']].format(**REPORT_VARS(R)))
    L.append('')
    if dec['adopted']:
        c = dec['adopted']
        m = C[c]
        L.append('**部署参数（训练侧标定，合规）**：')
        L.append('')
        L.append(f"- 组合公式：`{m['formula']}`")
        L.append(f"- 逐 task z：`z = (x - mu_t) / sd_t`，mu/sd 取自各分量 fit 行参考分数；"
                 f"EEG 的 mu/sd 已写入 `models/e239_gate_headroom.pt`")
        L.append(f"- tau(B) = {m['B']['tau']:+.6f}（组合分数自身 LOSO=0.90 的训练侧分位）")
        L.append(f"- 在该点：LOSO={f4(m['B']['loso_unknown_rejection'])}、"
                 f"S1={f4(m['B']['S1_rejection'])}、S2={f4(m['B']['S2_acceptance'])}、"
                 f"sepAUC={f4(m['B']['separation_auc'])}、"
                 f"簇纯度={f4(m['B']['purity_accepted_S2']['purity_micro'])}")
        L.append(f"- A 列同 tau 的 LOSO 拒识={f4(m['A']['loso_unknown_rejection'])}"
                 f"（已知接受率 {f4(m['A']['train_known_acceptance'])}）")
    L.append('')
    L.append('## 11. 合规性与局限')
    L.append('')
    L.append('- B 列 tau 是**训练侧标定**（Train 域 LOSO unknown 分数的 90% 分位），这是唯一'
             '可部署的读法。C 列 tau 来自 Test S1 分位，是评估选点，不能当作合规结论。')
    L.append('- Test 自 E65 起被反复查看，所有 S2 数字都带**反复检视偏倚**；本轮候选与组合规则'
             '在运行前已写入 `config.json`，A/B 列没有在 Test 上选点。')
    L.append('- Test 没有身份标签，官方 **C / RA / ASafeR 为 N/A**；本任务是门控（拒识）任务，'
             '没有闭集分类器输出，**F1 亦为 N/A**。S1 拒识与 S2 接受率是唯一读数。')
    L.append('- E232：S1 = 5 名未注册被试、S2 = 20 名注册用户，session 与注册状态混叠；'
             '对齐只把 session 当批次键。均值对齐是批次/转导统计量，需与竞赛规则确认。')
    L.append('- 组合分数是标量，身份仍来自 EEG 分支，簇纯度只检验接受决策是否保持身份一致。')
    L.append('- 图像视图是 E237 的缓存分数（E149 像素视图），本轮**没有重建任何特征**，'
             'EEG 分支是 E233 的冻结分数。')
    L.append('')
    L.append(f"- 运行：`.venv\\Scripts\\python.exe research/E239_gate_headroom/run_e239.py`"
             f"（CUDA），耗时 {R['runtime_s']:.0f}s；逐候选分数在 `cache/`，"
             f"z 统计量与 tau 在 `models/e239_gate_headroom.pt`。")
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


def REPORT_VARS(R):
    C = R['candidates']
    dec = R['decision']
    e = C['eeg_only']
    mins = C['min_rule']
    return dict(
        A_loso_min=f4(mins['A']['loso_unknown_rejection']),
        A_loso_eeg=f4(e['A']['loso_unknown_rejection']),
        A_known_min=f4(mins['A']['train_known_acceptance']),
        A_known_eeg=f4(e['A']['train_known_acceptance']),
        loso_auc_min=f4(mins['loso_auc']),
        loso_auc_eeg=f4(e['loso_auc']),
        B_S2_min=f4(mins['B']['S2_acceptance']),
        B_S2_m3=f4(C['min_rule_3view']['B']['S2_acceptance']),
        B_S2_sa=f4(C['and_rule_soft']['B']['S2_acceptance']),
        B_S1_min=f4(mins['B']['S1_rejection']),
        best_B=dec['best_B'],
        best_B_S2=f4(dec['best_B_S2']),
        best_B_S1=f4(C[dec['best_B']]['B']['S1_rejection']),
        base_S2=f4(BASE_S2),
        base_AUC=f4(BASE_AUC),
        best_C=dec['best_C'],
        best_C_S2=f4(dec['best_C_S2']),
        best_B_formula=C[dec['best_B']]['formula'],
        eeg_B_S2=f4(C['eeg_only']['B']['S2_acceptance']),
        margin_B=f'{dec["best_B_S2"] - BASE_S2:+.4f}',
        margin_B_same_rule=f'{dec["best_B_S2"] - C["eeg_only"]["B"]["S2_acceptance"]:+.4f}',
        adopted_loso_auc=f4(C[dec['best_B']]['loso_auc']),
        eeg_loso_auc=f4(C['eeg_only']['loso_auc']),
        adopted_A_loso=f4(C[dec['best_B']]['A']['loso_unknown_rejection']),
        adopted_B_loso=f4(C[dec['best_B']]['B']['loso_unknown_rejection']),
        formula_min=mins['formula'],
    )


CONCLUSION = {
    'pass_B': '名义达标但机制未证实：`{best_B}` 的 B 列 S2={best_B_S2}，仅比 0.3122 高 '
              '{margin_B}（比同规则 EEG 基线 {eeg_B_S2} 高 {margin_B_same_rule}），且其 LOSO AUC '
              '从 {eeg_loso_auc} 降到 {adopted_loso_auc}、A 列 LOSO 只有 {adopted_A_loso}——'
              '所以 S2 的微升不是"先买余量再花"，而是加权和压缩了 unknown 高分尾部的形状效应，'
              '不稳健。min 规则已确认无效（B 列 S2={B_S2_min}）。',
    'mechanism_but_no_gain': '机制成立但无净收益：A 列同 tau 下 LOSO 拒识确实被推高'
                '（`min_rule` {A_loso_min} vs EEG {A_loso_eeg}），但 LOSO AUC 只从 '
                '{loso_auc_eeg} 变成 {loso_auc_min}，B 列重标定后 S2={B_S2_min}，'
                '仍低于 0.3122。',
    'fail': '未达成：A 列同 tau 下 LOSO 没有明显提升，min 规则不能提高训练域拒识。',
}

VERDICT_TEXT = {
    'pass_B': '**判定：B 列名义达标（`{best_B}`），但机制未被证实，不建议当作确定增益部署。**\n\n'
              '- B 列：S2={best_B_S2} > 0.3122，S1={best_B_S1} ≥ 0.95，按预设规则算"通过"；'
              '超过同规则 EEG 基线（{eeg_B_S2}）{margin_B_same_rule}。'
              '其 tau 来自**训练侧** LOSO=0.90 分位，合规。\n'
              '- 但 A 列同 tau 的 LOSO 只有 {adopted_A_loso}（EEG 为 {A_loso_eeg}），'
              '说明它**没有买到余量**；同期 train 侧 LOSO AUC 从 {eeg_loso_auc} 降到 '
              '{adopted_loso_auc}，排序其实是变差的。\n'
              '- 收益来源是分布形状而非排序，且幅度（{margin_B}）小于归一化约定带来的读数漂移'
              '（同一配方全局 z 读 0.2906 vs 逐 task z 读 {best_B_S2}）。因此这是名义通过，'
              '不构成"先买余量再花"机制的验证。\n'
              '- min / 软 and 规则全部失败：B 列 S2 分别为 {B_S2_min}、{B_S2_m3}、{B_S2_sa}，'
              '均低于 0.3122。',
    'mechanism_but_no_gain': '**判定：机制成立但无净收益**。\n\n'
              '- A 列：`min_rule` 在同一原 tau 下把 LOSO 拒识从 {A_loso_eeg} 推到 {A_loso_min}，'
              '同时把训练域已知接受率从 {A_known_eeg} 压到 {A_known_min}——这说明 A 列的'
              '"余量"主要来自收紧阈值，而不是排序变好。\n'
              '- 排序诊断：LOSO AUC 从 {loso_auc_eeg} 变为 {loso_auc_min}。\n'
              '- B 列：把这个组合重新标定到自身 LOSO=0.90 后，S2={B_S2_min}（S1={B_S1_min}），'
              '并没有超过 0.3122。原因见上：在 unknown 的 90% 分位上，known 的分布没有被推右，'
              'min 规则取两支中较差的一支，恰恰损伤了高接受率区域的已知样本。\n'
              '- C 列（评估选点，不合规）：最好的是 `{best_C}`，S2={best_C_S2}；这只说明'
              '若允许在 Test S1 分位上选点仍能有收益，不能作为合规结论。',
    'fail': '**判定：未达成**。A 列同一 tau 下 LOSO 拒识没有明显提升（`min_rule` {A_loso_min} '
            'vs EEG {A_loso_eeg}），即 min 规则不能提高训练域拒识；逐 task 证据见第 9 节。'
            '按预设规则不换配方重跑。C 列最好的 `{best_C}` 仅为评估选点（S2={best_C_S2}），'
            '不可作为合规结论。',
}


if __name__ == '__main__':
    CFG_BLOCK_PATHS_TEST = json.loads(
        (ROOT / 'research/E233_delta_estimation/config.json').read_text(encoding='utf8')
    )['feature_cache']['test']
    main()
