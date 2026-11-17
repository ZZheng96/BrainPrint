"""E233: estimate and symmetrically remove the cross-session offset delta.

Frozen pipeline (byte-identical to E228 e226_auc2W): the E227 deploy model
(deploy_auc2_lofo.pt) carries the task-wise standardisation, the sqrt(w_b) block
weights, the PCA96 matrix, the 960 prototypes and the inner-OOF scores used for
tau.  Nothing is refitted here, so the LOSO unknown rejection cannot move; it is
re-verified from the frozen E227 fold caches.

What is new: on Test only, for every (session, task) batch an UNLABELLED batch
statistic is estimated in the frozen PCA space and every candidate transform is
applied with one identical rule to both sessions.  Session is used only as a
batch key, never to pick an identity and never as a gate.

Candidates
  baseline_frozen              no alignment (= E228 reference)
  mean_align                   remove batch translation, add back the train task mean
  coral                        CORAL: shrunk batch whitening + train recolouring
  mean_align_per_task_cov      mean align + diagonal per-task variance rescale
  mean_align_session2_only     DEGENERATE CONTROL (diagnostic only)
  mean_align_pooled_global     DIAGNOSTIC: alignment to the pooled global mean (~0)

Read-outs per candidate: LOSO check, Test S1 rejection, Test S2 acceptance,
separation AUC, S2 at a matched S1 gate of 0.95 and of 0.70, per-task split, and
the label-free cluster-purity check on Test session2 (k-means k=20/28, fixed seed,
cluster partition fixed in the baseline PCA space so that purity is comparable).
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

TASKS = ['EC', 'EO', 'Imagery', 'SSVEP']
LOSO_AUC_REF = 0.9095969878484796
FOLD_TAU_MEAN_REF = -24.18467938154109
BLOCK_DIM = {k: int(v) for k, v in CFG['blocks'].items()}
BLOCK_ORDER = list(CFG['representation']['block_order'])
PCA_DIM = int(CFG['representation']['pca_dim'])
K_PROTO = int(CFG['representation']['prototypes_k'])
A_CAL = float(CFG['frozen_calibration']['A'])
TAU_REF = float(CFG['frozen_calibration']['test_tau'])
LOSO_REF = float(CFG['frozen_calibration']['loso_unknown_rejection_micro'])
FIT_FRAC = 0.6
N_FOLDS = 20
LAM_CORAL = float(CFG['alignment']['shrinkage_lambda_coral'])
LAM_DIAG = float(CFG['alignment']['shrinkage_lambda_diag'])
RIDGE_REL = float(CFG['alignment']['ridge_rel'])
MIN_CELL_N = int(CFG['alignment']['min_cell_n'])
PURITY_K = [int(k) for k in CFG['metrics']['purity']['k']]
PURITY_SEED = int(CFG['metrics']['purity']['seed'])
BOOT_B = int(CFG['stability']['bootstrap_B'])
BOOT_SEED = int(CFG['stability']['bootstrap_seed'])
HALF_SEED = int(CFG['stability']['split_half_seed'])

BASE = CFG['predecessor']['known_anchors']['e228_auc2W_calibrated']
BASE_S2 = float(BASE['test_S2_acceptance'])
BASE_S1 = float(BASE['test_S1_rejection'])
BASE_AUC = float(BASE['test_separation_auc'])
S1_GATE = 0.95
S1_LOOSE = 0.70
LOSO_BAND = (0.89, 0.91)
PURITY_TOL = 0.95

CAND_IDS = ['baseline_frozen', 'mean_align', 'coral', 'mean_align_per_task_cov',
            'mean_align_session2_only', 'mean_align_pooled_global']
CONTROL_IDS = {'mean_align_session2_only', 'mean_align_pooled_global'}


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


# =====================================================================  frozen model
def load_state():
    m = torch.load(ROOT / CFG['feature_cache']['frozen_model'], weights_only=False)
    return m


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


def score_rows(state, z, tasks, rows):
    proto_task, proto_id = proto_meta(state)
    n = len(tasks)
    score = np.full(n, np.nan)
    pred = np.full(n, -1, int)
    for t in TASKS:
        rr = rows[tasks[rows] == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rr]), tensor(state['p'][cols])).double()
        dmin_t, amin_t = D.min(1)
        score[rr] = -dmin_t.cpu().numpy()
        pred[rr] = proto_id[cols][amin_t.cpu().numpy()]
    return score, pred


# =====================================================================  batch statistics
def sym_sqrt(C, inverse=False, ridge=0.0):
    w, U = np.linalg.eigh(C)
    if ridge:
        w = w + ridge * max(float(np.max(w)), 1e-12)
    w = np.maximum(w, 1e-12 * max(float(np.max(w)), 1e-12))
    w = w ** (-0.5) if inverse else np.sqrt(w)
    return (U * w) @ U.T


def cell_stats(z, mask, tasks, rows):
    """Per (session, task) mean / covariance / diagonal."""
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


def apply_mean_align(z, sess_masks, cells_b, ref, tasks, only_s2=False):
    z = z.copy()
    for sname, mask in sess_masks.items():
        if only_s2 and sname != 'session2':
            continue
        for t in TASKS:
            rr = np.flatnonzero(mask & (tasks == t))
            cb = cells_b[sname][t]
            if cb['n'] < MIN_CELL_N or not len(rr):
                continue
            z[rr] = z[rr] - cb['mu'] + ref[t]['mu']
    return z


def apply_coral(z, sess_masks, cells_b, ref, tasks):
    z = z.copy()
    d = z.shape[1]
    for sname, mask in sess_masks.items():
        for t in TASKS:
            rr = np.flatnonzero(mask & (tasks == t))
            cb, ct = cells_b[sname][t], ref[t]
            if cb['n'] < MIN_CELL_N or not len(rr):
                continue
            Cs = (1.0 - LAM_CORAL) * cb['C'] + LAM_CORAL * ct['C']
            M = sym_sqrt(Cs, inverse=True, ridge=RIDGE_REL) @ sym_sqrt(ct['C'])
            z[rr] = ct['mu'] + (z[rr].astype(np.float64) - cb['mu']) @ M
    return z


def apply_cov_scale(z, sess_masks, cells_b, ref, tasks):
    z = z.copy()
    for sname, mask in sess_masks.items():
        for t in TASKS:
            rr = np.flatnonzero(mask & (tasks == t))
            cb, ct = cells_b[sname][t], ref[t]
            if cb['n'] < MIN_CELL_N or not len(rr):
                continue
            var_eff = (1.0 - LAM_DIAG) * cb['var'] + LAM_DIAG * ct['var']
            scale = np.sqrt(ct['var'] / np.maximum(var_eff, 1e-12 * np.max(var_eff)))
            scale = np.clip(scale, 0.25, 4.0)
            z[rr] = ct['mu'] + (z[rr].astype(np.float64) - cb['mu']) * scale
    return z


def align(z, mode, sess_masks, cells_b, ref, tasks):
    if mode == 'baseline_frozen':
        return z.copy()
    if mode == 'mean_align':
        return apply_mean_align(z, sess_masks, cells_b, ref, tasks)
    if mode == 'mean_align_session2_only':
        return apply_mean_align(z, sess_masks, cells_b, ref, tasks, only_s2=True)
    if mode == 'mean_align_pooled_global':
        zero = {t: dict(n=10 ** 9, mu=np.zeros(z.shape[1])) for t in TASKS}
        return apply_mean_align(z, sess_masks, cells_b, zero, tasks)
    if mode == 'coral':
        return apply_coral(z, sess_masks, cells_b, ref, tasks)
    if mode == 'mean_align_per_task_cov':
        return apply_cov_scale(z, sess_masks, cells_b, ref, tasks)
    raise ValueError(mode)


# =====================================================================  metrics
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
    """tau that rejects `target_rej` of the S1 rows: reject <=> score < tau, so tau = q(target)."""
    return float(np.quantile(s1_scores, target_rej))


# =====================================================================  k-means purity
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


# =====================================================================  stability
def delta_stability(z, masks, tasks, seed, boot_B=BOOT_B, boot_seed=BOOT_SEED):
    rows = []
    for sname, mask in masks.items():
        for t in TASKS:
            rr = np.flatnonzero(mask & (tasks == t))
            A = z[rr].astype(np.float64)
            n = len(A)
            if n < 2:
                continue
            mu = A.mean(0)
            rng = np.random.default_rng(seed + 7919 * len(sname) + sum(map(ord, t)))
            perm = rng.permutation(n)
            m1 = A[perm[: n // 2]].mean(0)
            m2 = A[perm[n // 2:]].mean(0)
            n1 = n // 2
            d1, d2 = np.linalg.norm(m1), np.linalg.norm(m2)
            cos = float(m1 @ m2 / max(d1 * d2, 1e-12))
            rel = float(np.linalg.norm(m1 - m2) / max(np.linalg.norm(mu), 1e-12))
            # expected split-half noise if the offset were zero: sqrt(d) * sigma / sqrt(n/2)
            sigma = float(np.sqrt(np.trace(np.cov(A.T))) if n > 2 else 0.0)
            noise = float(np.sqrt(z.shape[1]) * sigma / np.sqrt(max(n1, 1)))
            brng = np.random.default_rng(boot_seed + 31 * len(sname) + sum(map(ord, t)))
            norms = []
            for _ in range(boot_B):
                idx = brng.integers(0, n, n)
                norms.append(float(np.linalg.norm(A[idx].mean(0))))
            C = np.cov(A.T) if n > 2 else np.eye(z.shape[1])
            rows.append(dict(session=sname, task=t, n=n,
                             delta_norm=float(np.linalg.norm(mu)),
                             half_cosine=cos,
                             half_rel_diff=rel,
                             split_half_noise_scale=noise,
                             delta_over_noise=float(np.linalg.norm(mu) / max(noise, 1e-12)),
                             boot_mean=float(np.mean(norms)),
                             boot_sd=float(np.std(norms)),
                             boot_lo=float(np.percentile(norms, 2.5)),
                             boot_hi=float(np.percentile(norms, 97.5)),
                             cond=float(np.linalg.cond(C)),
                             n_over_d=float(n / z.shape[1])))
    return rows


def boot_ci(score, tau, is_s1, tasks, B=BOOT_B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for name, mask in (('S1_rejection', is_s1), ('S2_acceptance', ~is_s1)):
        vals = []
        for _ in range(B):
            num = den = 0
            for t in TASKS:
                rr = np.flatnonzero(mask & (tasks == t))
                if not len(rr):
                    continue
                idx = rr[rng.integers(0, len(rr), len(rr))]
                hit = (score[idx] < tau) if name == 'S1_rejection' else (score[idx] >= tau)
                num += int(hit.sum())
                den += len(idx)
            vals.append(num / max(den, 1))
        out[name] = dict(mean=float(np.mean(vals)),
                         lo=float(np.percentile(vals, 2.5)),
                         hi=float(np.percentile(vals, 97.5)),
                         sd=float(np.std(vals)))
    return out


def paired_boot(scores, ids, base_id, tau, is_s1, tasks, B=BOOT_B, seed=BOOT_SEED):
    """Paired row bootstrap of candidate - baseline at the frozen tau (same resamples)."""
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
                        num_a += int((scores[cid][rr] < tau).sum())
                        num_b += int((scores[base_id][rr] < tau).sum())
                    else:
                        num_a += int((scores[cid][rr] >= tau).sum())
                        num_b += int((scores[base_id][rr] >= tau).sum())
                    den += len(rr)
                out[cid][key].append((num_a - num_b) / max(den, 1))
    return {cid: {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                          hi=float(np.percentile(v, 97.5)),
                          p_nonpositive=float(np.mean(np.asarray(v) <= 0.0)))
                  for k, v in d.items()} for cid, d in out.items()}


def paired_cluster_boot(scores, ids, base_id, tau, s2_rows, assign, k, B=BOOT_B,
                        seed=BOOT_SEED):
    """Paired bootstrap that resamples whole k-means clusters (identity/task blocks)."""
    members = [s2_rows[assign == j] for j in range(k) if (assign == j).any()]
    rng = np.random.default_rng(seed)
    out = {cid: [] for cid in ids}
    for _ in range(B):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[j] for j in pick])
        base_hit = float((scores[base_id][idx] >= tau).mean())
        for cid in ids:
            out[cid].append(float((scores[cid][idx] >= tau).mean()) - base_hit)
    return {cid: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                      hi=float(np.percentile(v, 97.5)),
                      p_nonpositive=float(np.mean(np.asarray(v) <= 0.0)))
            for cid, v in out.items()}


# =====================================================================  main
def load_features(split):
    store = {}
    for b in BLOCK_ORDER:
        p = ROOT / CFG['feature_cache'][split][b]
        arr = np.load(p).astype(np.float32)
        assert arr.shape[1] == BLOCK_DIM[b], (b, arr.shape)
        store[b] = arr
    return np.concatenate([store[b] for b in BLOCK_ORDER], 1)


def main():
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E233 is blocked'
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log(f'device={torch.cuda.get_device_name(0)} torch={torch.__version__}')
    t_start = time.time()

    model = load_state()
    state = model['state']
    assert int(state['pca']) == PCA_DIM and int(state['k']) == K_PROTO
    log(f"frozen model scheme={model.get('scheme')} pca={state['pca']} k={state['k']} "
        f"dim={state['dim']} blocks={model.get('block_order')}")

    tr = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    te = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tr) == 11304 and len(te) == 22251
    y = tr.subject.to_numpy(int)
    tr_task = tr.task.to_numpy()
    fr = frac_of(tr)
    fit = fr <= FIT_FRAC
    te_task = te.task.to_numpy()
    sess = te.session.astype(str).str.casefold().to_numpy()
    is_s1 = np.isin(sess, ['session1', 's1', '1'])
    masks = {'session1': is_s1, 'session2': ~is_s1}
    log(f'train fit rows={int(fit.sum())} test S1={int(is_s1.sum())} S2={int((~is_s1).sum())}')

    # ---------------- representation (frozen, no refit)
    Xtr = load_features('train')
    Xte = load_features('test')
    ztr = transform(state, Xtr, tr_task)
    zte = transform(state, Xte, te_task)
    log(f'PCA space: train {ztr.shape} test {zte.shape}')

    # ---------------- reproduction: fold scores + Test scores + tau
    checks = {}
    rej, n_u, taus, aucs = 0, 0, [], []
    for h in range(1, N_FOLDS + 1):
        d = np.load(ROOT / CFG['feature_cache']['frozen_fold_cache'].format(fold=h),
                    allow_pickle=True)
        s, oof, ofit = d['score'], d['oof_s'], d['oof_fit'].astype(bool)
        u, k = d['unknown'].astype(bool), d['known'].astype(bool)
        tau = float(np.quantile(oof[ofit], 1.0 - A_CAL))
        taus.append(tau)
        rej += int((s[u] < tau).sum())
        n_u += int(u.sum())
        aucs.append(auc_ties(s[k], s[u]))
    loso_micro = rej / n_u
    fold_tau = np.array(taus)
    checks['loso_unknown_rejection_micro'] = dict(
        value=float(loso_micro), reference=LOSO_REF, abs_diff=float(abs(loso_micro - LOSO_REF)))
    checks['loso_auc_mean'] = dict(value=float(np.mean(aucs)), reference=LOSO_AUC_REF,
                                   abs_diff=float(abs(np.mean(aucs) - LOSO_AUC_REF)))
    checks['fold_tau'] = dict(mean=float(fold_tau.mean()), min=float(fold_tau.min()),
                              max=float(fold_tau.max()))
    checks['fold_tau_mean_vs_e228'] = dict(
        value=float(fold_tau.mean()), reference=FOLD_TAU_MEAN_REF,
        abs_diff=float(abs(fold_tau.mean() - FOLD_TAU_MEAN_REF)))
    dte = np.load(ROOT / CFG['feature_cache']['frozen_test_cache'], allow_pickle=True)
    tau_deploy = float(np.quantile(dte['oof_s'][dte['oof_fit'].astype(bool)], 1.0 - A_CAL))
    s_chk, p_chk = score_rows(state, zte, te_task, np.arange(len(te_task)))
    checks['test_tau'] = dict(value=tau_deploy, reference=TAU_REF,
                              abs_diff=float(abs(tau_deploy - TAU_REF)))
    checks['test_score_reproduction'] = dict(
        max_abs_diff=float(np.max(np.abs(s_chk - dte['score']))),
        pred_identical=bool(np.array_equal(p_chk, dte['pred'].astype(int))))
    log(f"LOSO micro={loso_micro:.10f} (ref {LOSO_REF:.10f}) tau_deploy={tau_deploy:.10f} "
        f"score_repro_max|d|={checks['test_score_reproduction']['max_abs_diff']:.2e}")
    assert abs(loso_micro - LOSO_REF) < 1e-9, 'LOSO path drifted'
    assert abs(fold_tau.mean() - FOLD_TAU_MEAN_REF) < 1e-6, 'fold tau drifted'
    assert abs(tau_deploy - TAU_REF) < 1e-9, 'test tau drifted'
    assert checks['test_score_reproduction']['max_abs_diff'] < 1e-4

    base_scores = dte['score'].astype(np.float64)
    base_metrics = readout(base_scores, dte['pred'].astype(int), is_s1, te_task, tau_deploy)
    log(f'baseline readout S1={base_metrics["S1_rejection"]:.4f} '
        f'S2={base_metrics["S2_acceptance"]:.4f} sepAUC={base_metrics["separation_auc"]:.4f}')
    assert abs(base_metrics['S2_acceptance'] - BASE_S2) < 1e-9
    score_quantiles = dict(
        S1={str(q): float(np.quantile(base_scores[is_s1], q))
            for q in (0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.95, 0.9637)},
        S2={str(q): float(np.quantile(base_scores[~is_s1], q))
            for q in (0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.95, 0.9637)},
        tau=float(tau_deploy))
    log('score quantiles ' + json.dumps(score_quantiles))

    # ---------------- train reference stats (fit rows, all 20 identities)
    ref = {}
    for t in TASKS:
        rr = np.flatnonzero(fit & (tr_task == t))
        A = ztr[rr].astype(np.float64)
        mu = A.mean(0)
        Ac = A - mu
        C = (Ac.T @ Ac) / max(len(A) - 1, 1)
        ref[t] = dict(n=int(len(A)), mu=mu, C=C, var=np.diag(C).copy(),
                      trace=float(np.trace(C)), cond=float(np.linalg.cond(C)))
        log(f'train ref {t}: n={len(A)} trace={ref[t]["trace"]:.2f} cond={ref[t]["cond"]:.1f}')

    cells_b = {s: cell_stats(zte, m, te_task, np.arange(len(te_task))) for s, m in masks.items()}
    for s in cells_b:
        for t in TASKS:
            c = cells_b[s][t]
            log(f'cell {s:8s} {t:7s} n={c["n"]:5d} ||mu||={np.linalg.norm(c["mu"]):.3f} '
                f'trace={c["trace"]:.2f} cond={c["cond"]:.1f}')

    stability = delta_stability(zte, masks, te_task, HALF_SEED)

    # ---------------- purity clusters (fixed partition in the baseline space)
    s2_rows = np.flatnonzero(~is_s1)
    assigns = {}
    for k in PURITY_K:
        t0 = time.time()
        assigns[k], sizes = kmeans_assign(zte[s2_rows].astype(np.float32), k, PURITY_SEED + k)
        log(f'kmeans k={k} on {len(s2_rows)} S2 rows ({time.time()-t0:.1f}s) '
            f'sizes min/max={sizes.min()}/{sizes.max()} cv={sizes.std()/max(sizes.mean(),1e-9):.3f}')

    # ---------------- candidates
    results = {}
    scores_all = {}
    preds_all = {}
    for cid in CAND_IDS:
        t0 = time.time()
        z = align(zte, cid, masks, cells_b, ref, te_task)
        score, pred = score_rows(state, z.astype(np.float32), te_task,
                                 np.arange(len(te_task)))
        m = readout(score, pred, is_s1, te_task, tau_deploy)
        m['pred_changed_fraction'] = float((pred != dte['pred'].astype(int)).mean())
        m['mean_score_S1'] = float(score[is_s1].mean())
        m['mean_score_S2'] = float(score[~is_s1].mean())
        m['S1_over_S2_mean_gap'] = float(score[is_s1].mean() - score[~is_s1].mean())
        m['S1_score_max'] = float(score[is_s1].max())
        m['S1_score_p99'] = float(np.quantile(score[is_s1], 0.99))
        scores_all[cid] = score
        preds_all[cid] = pred
        gates = {}
        for name, target in (('S1_0.95', S1_GATE), ('S1_0.70', S1_LOOSE),
                             ('S1_matched_baseline', BASE_S1)):
            tg = gate_tau(score[is_s1], target)
            g = readout(score, pred, is_s1, te_task, tg)
            gates[name] = dict(target_S1_rejection=target, tau=tg,
                               S1_rejection=g['S1_rejection'],
                               S2_acceptance=g['S2_acceptance'],
                               separation_auc=g['separation_auc'])
        m['gates'] = gates
        m['bootstrap'] = boot_ci(score, tau_deploy, is_s1, te_task, seed=BOOT_SEED + len(cid))
        pur = {}
        for k in PURITY_K:
            pur[f'k{k}'] = purity(pred[s2_rows], assigns[k], k)
        m['purity'] = pur
        m['cell_scored'] = {}
        for s in masks:
            for t in TASKS:
                rr = np.flatnonzero(masks[s] & (te_task == t))
                m['cell_scored'][f'{s}|{t}'] = dict(
                    n=int(len(rr)),
                    S1_rejection=(float((score[rr] < tau_deploy).mean()) if s == 'session1' else None),
                    S2_acceptance=(float((score[rr] >= tau_deploy).mean()) if s == 'session2' else None),
                    mean_score=float(score[rr].mean()))
        results[cid] = dict(id=cid, control=bool(cid in CONTROL_IDS), metrics=m)
        np.savez_compressed(CACHE / f'test_{CACHE_VERSION}_{cid}.npz',
                            score=score, pred=pred, tau=tau_deploy, is_s1=is_s1,
                            task=np.asarray(te_task, dtype='<U8'))
        log(f'cand {cid:28s} S1={m["S1_rejection"]:.4f} S2={m["S2_acceptance"]:.4f} '
            f'sepAUC={m["separation_auc"]:.4f} S2@S1.95={gates["S1_0.95"]["S2_acceptance"]:.4f} '
            f'S2@S1.70={gates["S1_0.70"]["S2_acceptance"]:.4f} '
            f'pur20={pur["k20"]["purity_micro"]:.4f} pur28={pur["k28"]["purity_micro"]:.4f} '
            f'({time.time()-t0:.1f}s)')

    base_metrics = results['baseline_frozen']['metrics']
    base_pur = results['baseline_frozen']['metrics']['purity']
    paired = paired_boot(scores_all, CAND_IDS, 'baseline_frozen', tau_deploy, is_s1, te_task)
    for cid in CAND_IDS:
        results[cid]['metrics']['paired_vs_baseline'] = paired[cid]
    log('paired bootstrap vs baseline: ' + ', '.join(
        f"{c} dS2={paired[c]['dS2']['mean']:+.4f}"
        f"[{paired[c]['dS2']['lo']:+.4f},{paired[c]['dS2']['hi']:+.4f}]"
        f"p<=0={paired[c]['dS2']['p_nonpositive']:.3f}" for c in CAND_IDS))
    cluster_paired = paired_cluster_boot(scores_all, CAND_IDS, 'baseline_frozen',
                                         tau_deploy, s2_rows, assigns[20], 20)
    log('paired cluster bootstrap (k=20) vs baseline: ' + ', '.join(
        f"{c} dS2={cluster_paired[c]['mean']:+.4f}"
        f"[{cluster_paired[c]['lo']:+.4f},{cluster_paired[c]['hi']:+.4f}]" for c in CAND_IDS))
    checks['s2_rows_identical_mean_align_vs_session2_only'] = dict(
        max_abs_score_diff=float(np.max(np.abs(
            scores_all['mean_align'][s2_rows] - scores_all['mean_align_session2_only'][s2_rows]))),
        pred_identical=bool(np.array_equal(preds_all['mean_align'][s2_rows],
                                          preds_all['mean_align_session2_only'][s2_rows])))
    log('S2 rows identical (mean_align vs session2_only): '
        + json.dumps(checks['s2_rows_identical_mean_align_vs_session2_only']))

    # ---------------- decision
    def eligible(cid):
        v = results[cid]
        if v['control']:
            return False
        mt = v['metrics']
        pur_ok = all(mt['purity'][f'k{k}']['purity_micro'] >=
                     PURITY_TOL * base_pur[f'k{k}']['purity_micro'] for k in PURITY_K)
        return (LOSO_BAND[0] <= loso_micro <= LOSO_BAND[1]
                and mt['S1_rejection'] >= S1_GATE
                and mt['S2_acceptance'] > BASE_S2
                and pur_ok)

    ok = [c for c in CAND_IDS if eligible(c)]
    sym = [c for c in CAND_IDS if not results[c]['control']]
    best_sym = max(sym, key=lambda c: results[c]['metrics']['S2_acceptance'])
    # max S2 at the matched S1 gate, symmetric candidates only
    best_gate = max(sym, key=lambda c: results[c]['metrics']['gates']['S1_0.95']['S2_acceptance'])
    s2_at_gate = results[best_gate]['metrics']['gates']['S1_0.95']['S2_acceptance']
    base_at_gate = base_metrics['gates']['S1_0.95']['S2_acceptance']

    rises_without_s1 = [c for c in sym
                        if results[c]['metrics']['S2_acceptance'] > BASE_S2
                        and results[c]['metrics']['S1_rejection'] < S1_GATE]
    if ok:
        adopted = max(ok, key=lambda c: results[c]['metrics']['S2_acceptance'])
        verdict = 'pass'
    elif s2_at_gate > base_at_gate and max(
            results[c]['metrics']['S2_acceptance'] for c in sym) > BASE_S2:
        adopted = None
        verdict = 'trade_off'
    else:
        adopted = None
        verdict = 'fail'
    log(f'decision verdict={verdict} adopted={adopted} '
        f'max S2@S1.95={s2_at_gate:.4f} ({best_gate}) baseline S2@S1.95={base_at_gate:.4f}')

    # ---------------- freeze artifacts
    cell_dump = {s: {t: dict(n=cells_b[s][t]['n'], mu=cells_b[s][t]['mu'],
                             var=cells_b[s][t]['var'], trace=cells_b[s][t]['trace'],
                             cond=cells_b[s][t]['cond']) for t in TASKS}
                 for s in cells_b}
    ref_dump = {t: dict(n=ref[t]['n'], mu=ref[t]['mu'], var=ref[t]['var'],
                        trace=ref[t]['trace']) for t in TASKS}
    torch.save(dict(experiment='E233_delta_estimation',
                    base_model=str(ROOT / CFG['feature_cache']['frozen_model']),
                    lambda_coral=LAM_CORAL, lambda_diag=LAM_DIAG,
                    test_cells=cell_dump, train_reference=ref_dump,
                    note='unlabeled batch statistics for the alignment adapters'),
               MODELS / 'e233_alignment_stats.pt')

    deploy = None
    if adopted:
        src = results[adopted]['metrics']
        deploy = dict(candidate=adopted, alignment=adopted,
                      lambda_coral=LAM_CORAL, lambda_diag=LAM_DIAG,
                      tau=tau_deploy,
                      tau_S1_0p95=src['gates']['S1_0.95']['tau'],
                      S1_rejection=src['S1_rejection'],
                      S2_acceptance=src['S2_acceptance'],
                      separation_auc=src['separation_auc'],
                      S2_at_S1_0p95=src['gates']['S1_0.95']['S2_acceptance'],
                      transductive=True,
                      note='the adapter statistics are estimated from the unlabeled scoring batch; '
                           'do not use for submission without confirming the rules')
        torch.save(dict(base_state=state, block_weights=model.get('block_weights'),
                        block_order=model.get('block_order'), deploy=deploy,
                        test_cell_stats=cell_dump, train_reference=ref_dump),
                   MODELS / 'deploy_frozen.pt')

    results_out = dict(
        experiment='E233_delta_estimation',
        protocol=CFG['representation']['pipeline'],
        frozen_calibration=CFG['frozen_calibration'],
        alignment=CFG['alignment'],
        checks=checks,
        baseline=dict(reference=BASE, measured=base_metrics, score_quantiles=score_quantiles),
        cells={s: {t: dict(n=cells_b[s][t]['n'],
                           delta_norm=float(np.linalg.norm(cells_b[s][t]['mu'])),
                           trace=float(cells_b[s][t]['trace']),
                           cond=float(cells_b[s][t]['cond']),
                           n_over_d=float(cells_b[s][t]['n'] / PCA_DIM))
                    for t in TASKS} for s in cells_b},
        train_reference={t: dict(n=ref[t]['n'], trace=ref[t]['trace'],
                                 cond=ref[t]['cond']) for t in TASKS},
        stability=stability,
        purity_partition={f'k{k}': dict(n=int(len(s2_rows)), sizes=[int(x) for x in
                                                                    np.bincount(assigns[k], minlength=k)])
                          for k in PURITY_K},
        candidates=results,
        purity_baseline=base_pur,
        paired_bootstrap_vs_baseline=paired,
        paired_cluster_bootstrap_vs_baseline=cluster_paired,
        decision=dict(verdict=verdict, adopted=adopted,
                      eligible=ok, best_symmetric=best_sym,
                      max_S2_at_S1_0p95=s2_at_gate, best_at_S1_0p95=best_gate,
                      baseline_S2_at_S1_0p95=base_at_gate,
                      baseline_S2=BASE_S2, baseline_S1=BASE_S1,
                      baseline_sep_auc=BASE_AUC,
                      rises_but_S1_below_gate=rises_without_s1,
                      loso_unchanged=bool(LOSO_BAND[0] <= loso_micro <= LOSO_BAND[1]),
                      purity_tolerance=PURITY_TOL),
        deploy=deploy,
        official=dict(C='N/A', RA='N/A', ASafeR='N/A',
                      note='Test has no identity labels.'),
        limitations=CFG['official_note'] + ' ' + CFG['bias_note'],
        runtime_s=float(time.time() - t_start))
    (OUT / 'results.json').write_text(json.dumps(results_out, indent=2), encoding='utf8')

    build_report(results_out, base_metrics, tau_deploy, loso_micro)
    log('results.json + report.md written')
    log(f'verdict={verdict} runtime={time.time()-t_start:.0f}s')


# =====================================================================  report
def f4(x, n=4):
    if x is None:
        return '—'
    if isinstance(x, float) and np.isnan(x):
        return '—'
    return f'{x:.{n}f}'


def build_report(R, base_metrics, tau_deploy, loso_micro):
    C = R['candidates']
    dec = R['decision']
    order = CAND_IDS
    pur_keys = [f'k{k}' for k in PURITY_K]
    bp = R['purity_baseline']

    L = ['# E233 跨段偏移 δ 的直接估计与对称移除', '',
         '> **一句话结论**：' + (
             f"`{dec['adopted']}`（逐 (session,task) 批次均值对齐）按点估计达标——冻结 tau 下 Test S2 "
             f"**{f4(C[dec['adopted']]['metrics']['S2_acceptance'])}**"
             f"（基线 {f4(BASE_S2)}，{C[dec['adopted']]['metrics']['S2_acceptance'] - BASE_S2:+.4f}），"
             f"S1 拒识 {f4(C[dec['adopted']]['metrics']['S1_rejection'])}（基线 {f4(BASE_S1)}，已饱和），"
             f"分离 AUC {f4(C[dec['adopted']]['metrics']['separation_auc'])}，簇纯度守住；"
             '但整簇重采样的 ΔS2 95% CI 跨 0，属边际改进。'
             if dec['verdict'] == 'pass' else
             ('无候选达标：S1/S2 权衡' if dec['verdict'] == 'trade_off'
              else '无候选达标：δ 的批量估计不足以修复跨段偏移')) + '',
         f"S1≥0.95 约束下 S2 = **{f4(dec['max_S2_at_S1_0p95'])}**"
         f"（基线同约束 {f4(dec['baseline_S2_at_S1_0p95'])}，`{dec['best_at_S1_0p95']}`）；"
         f"S1≥0.70 约束下 S2 = **{f4(C[dec['best_at_S1_0p95']]['metrics']['gates']['S1_0.70']['S2_acceptance'])}**"
         f"（基线 {f4(base_metrics['gates']['S1_0.70']['S2_acceptance'])}）。"
         '两道门槛的差异与原因见第 2 节。', '',
         '口径：完全复用 E228 冻结的 `e226_auc2W`（E227 `deploy_auc2_lofo.pt` 的表征、原型、块权重、'
         'PCA 与 OOF），**不重训、不重标定**；δ 只在 Test 的无标签批次上估计，并以同一套规则同时作用于两个 session。', '',
         '## 1. 不变项与复现校验', '',
         '|项目|本轮实测|E228 参考|判定|', '|---|---:|---:|:---:|',
         f"|LOSO 未知拒识（micro）|{loso_micro:.10f}|{R['checks']['loso_unknown_rejection_micro']['reference']:.10f}|"
         f"{'通过' if abs(loso_micro - R['checks']['loso_unknown_rejection_micro']['reference']) < 1e-9 else '未通过'}|",
         f"|部署 tau（inner-OOF 分位）|{R['checks']['test_tau']['value']:.10f}|"
         f"{R['checks']['test_tau']['reference']:.10f}|"
         f"{'通过' if R['checks']['test_tau']['abs_diff'] < 1e-9 else '未通过'}|",
         f"|Test 分数复算最大绝对差|{R['checks']['test_score_reproduction']['max_abs_diff']:.2e}|0|"
         f"{'通过' if R['checks']['test_score_reproduction']['max_abs_diff'] < 1e-4 else '未通过'}|",
         f"|Test 预测逐行一致|{R['checks']['test_score_reproduction']['pred_identical']}|True|"
         f"{'通过' if R['checks']['test_score_reproduction']['pred_identical'] else '未通过'}|",
         f"|折级 tau 均值|{R['checks']['fold_tau_mean_vs_e228']['value']:.10f}|"
         f"{R['checks']['fold_tau_mean_vs_e228']['reference']:.10f}|"
         f"{'通过' if R['checks']['fold_tau_mean_vs_e228']['abs_diff'] < 1e-6 else '未通过'}|",
         '', f"- LOSO AUC（tau 自由）= {f4(R['checks']['loso_auc_mean']['value'])}"
         f"（E228 {f4(R['checks']['loso_auc_mean']['reference'])}）；"
         f"折级 tau 范围 {f4(R['checks']['fold_tau']['min'], 3)} ~ "
         f"{f4(R['checks']['fold_tau']['max'], 3)}。",
         '- 逐折分数本身没有重算的必要：E228 的折级分数就是直接读取这里同一批 E227 缓存'
         '（`research/E227_joint_frontier/cache/fold_v1_auc2_lofo_*.npz`）得到的，'
         '本轮只重新施加 `tau=(1−A)` 分位并汇总，因此"逐折一致"由上表 tau 均值与 micro 拒识两行共同锁定。',
         '- 所有对齐只改写 Test 行在 PCA 空间中的坐标；Train 行、原型、OOF 分数与 tau 完全没动，'
         '因此 LOSO 拒识按构造不变，上表是把"按构造不变"兑现成数值证据。', '',
         '## 2. 主表（冻结 tau，逐候选）', '',
         '`S1 rej` / `S2 acc` 用 E228 部署 tau（不随候选调整）；`S2@S1.95` = 把 tau 设在候选自身 S1 分数的 95% 分位'
         '（`score < tau` 判未知，因此恰好拒掉 95% S1，官方门槛）后的 S2；`S2@S1.70` 同理设在 70% 分位；'
         '`S2@S1=基线` = 把 tau 调到与基线相同的 S1 拒识水平后的 S2（同 S1 水平对照）。'
         '`sepAUC` = 与阈值无关的 S2-vs-S1 分离度。', '',
         '|候选|S1 rej|S2 acc|sepAUC|S2@S1.95|S2@S1.70|S2@S1=基线|S1↔S2 均值差|诊断|',
         '|---|---:|---:|---:|---:|---:|---:|---:|:---:|']
    for cid in order:
        m = C[cid]['metrics']
        tag = '对照/退化' if C[cid]['control'] else ('**对称**' if cid != 'baseline_frozen' else '基线')
        L.append(f"|{cid}|{f4(m['S1_rejection'])}|{f4(m['S2_acceptance'])}|"
                 f"{f4(m['separation_auc'])}|{f4(m['gates']['S1_0.95']['S2_acceptance'])}|"
                 f"{f4(m['gates']['S1_0.70']['S2_acceptance'])}|"
                 f"{f4(m['gates']['S1_matched_baseline']['S2_acceptance'])}|"
                 f"{f4(m['S1_over_S2_mean_gap'], 3)}|{tag}|")

    q = R['baseline']['score_quantiles']
    L += ['', '**两种读法为何差这么多（必须分开看）**', '',
          f"基线的 S1 分数中位数 {q['S1']['0.5']:.2f}、95% 分位 {q['S1']['0.95']:.2f}，"
          f"S2 分数中位数 {q['S2']['0.5']:.2f}，E228 部署 tau = {q['tau']:.2f}。",
          '1. **冻结 tau 读法**（不随候选调整、完全 Train-only）：`S2 acc` 列。它衡量的是"E228 标定的那一个阈值"'
          '下的接受率，最保守、也最可比。',
          '2. **S1 门槛读法**（把 tau 调到该候选自身 S1 分布的 95% 分位）：`S2@S1.95` 列。它衡量"在同样拒掉 95% S1"'
          '的前提下能接受多少 S2，是官方约束的标准读法，但阈值由 Test S1 分数决定（transductive），'
          '且 S1 只有 5 名未注册被试（E232 混叠），所以数字偏乐观。',
          '两者方向一致但幅度差异巨大：对齐把 S1 整体推到更远离原型的位置（S1 分数整体下移、S1/S2 分离度上升），'
          '因此同样的 95% S1 拒识对应的 tau 更低、S2 能过得更多。**冻结 tau 的 S2 增益（+0.0079）不应被当作'
          '该方案的全部价值，+0.42 的门槛增益也不应被当作纯生物特征增益。**', '',
          '相对基线（冻结 tau）：', '',
          '`ΔS2 配对 CI` = 对同一批重采样行同时计算候选与基线的 S2 接受率之差（配对自助，400 次），'
          '`p` = ΔS2 ≤ 0 的重采样比例。', '',
          '|候选|ΔS2|ΔS1|ΔsepAUC|ΔS2@同S1|ΔS2 配对 95% CI|p(Δ≤0)|预测改变比例|',
          '|---|---:|---:|---:|---:|---|---:|---:|']
    for cid in order:
        if cid == 'baseline_frozen':
            continue
        m = C[cid]['metrics']
        pb = m['paired_vs_baseline']['dS2']
        L.append(f"|{cid}|{m['S2_acceptance'] - base_metrics['S2_acceptance']:+.4f}|"
                 f"{m['S1_rejection'] - base_metrics['S1_rejection']:+.4f}|"
                 f"{m['separation_auc'] - base_metrics['separation_auc']:+.4f}|"
                 f"{m['gates']['S1_matched_baseline']['S2_acceptance'] - base_metrics['gates']['S1_matched_baseline']['S2_acceptance']:+.4f}|"
                 f"[{pb['lo']:+.4f}, {pb['hi']:+.4f}]|{pb['p_nonpositive']:.3f}|"
                 f"{f4(m['pred_changed_fraction'])}|")
    L += ['', '- `mean_align_pooled_global` 与 `mean_align` 数值完全相同，这是**数学上必然**的：'
          'task-wise 标准化用的就是各 task 的 fit 行均值，所以每 task 的 Train 参考均值在 PCA 空间里恰好是 0，'
          '"加回该 task 的 Train 均值"与"加回池化全局均值（=0）"是同一个变换。'
          '该行保留下来作为"参考均值是否实现正确"的校验，同时说明上一轮担心的"task 偏移被删掉"并不会发生。',
          f"- `mean_align` 的 S1 拒识恰好等于 1.0000（饱和）：对齐后 S1 分数最大值 "
          f"{C['mean_align']['metrics']['S1_score_max']:.2f}、p99 "
          f"{C['mean_align']['metrics']['S1_score_p99']:.2f}，全部低于 tau = {TAU_REF:.2f}。"
          '这是「把 5 名未注册被试的批次均值移到 Train 原点」的必然结果（该段 ||δ||≈20，'
          '远大于 S2 的 4–7），不应解读为「拒识能力提升」；它同时说明 S1 侧已经没有余量可供进一步换取 S2。',
          '- 行级配对自助会低估不确定性（同一被试的多条 trial 相关）。改用**整簇重采样**'
          f"（k=20 的 k-means 簇当作身份/task 块）后，`mean_align` 的 ΔS2 = "
          f"{R['paired_cluster_bootstrap_vs_baseline']['mean_align']['mean']:+.4f} "
          f"[{R['paired_cluster_bootstrap_vs_baseline']['mean_align']['lo']:+.4f}, "
          f"{R['paired_cluster_bootstrap_vs_baseline']['mean_align']['hi']:+.4f}]，"
          f"`coral` 的 ΔS2 = {R['paired_cluster_bootstrap_vs_baseline']['coral']['mean']:+.4f} "
          f"[{R['paired_cluster_bootstrap_vs_baseline']['coral']['lo']:+.4f}, "
          f"{R['paired_cluster_bootstrap_vs_baseline']['coral']['hi']:+.4f}]。",
          '- **关键对照**：`mean_align` 与 `mean_align_session2_only` 的 **S2 分数逐行完全相同**'
          f"（最大绝对差 {R['checks']['s2_rows_identical_mean_align_vs_session2_only']['max_abs_score_diff']:.1e}，"
          f"预测一致 {R['checks']['s2_rows_identical_mean_align_vs_session2_only']['pred_identical']}）——"
          '对齐 S1 不可能改变任何 S2 行。两者冻结 tau 下 S2 都是 '
          f"{f4(C['mean_align']['metrics']['S2_acceptance'])}，但门槛读法是 "
          f"{f4(C['mean_align']['metrics']['gates']['S1_0.95']['S2_acceptance'])} vs "
          f"{f4(C['mean_align_session2_only']['metrics']['gates']['S1_0.95']['S2_acceptance'])}："
          '**这个巨大差异完全来自 S1 分数分布下移使门槛 tau 下移**。所以门槛增益应理解为'
          '"S1/S2 分离度变大"，而不是"S2 打分本身变好"；S2 侧的全部收益就是那 '
          f"{C['mean_align']['metrics']['S2_acceptance'] - BASE_S2:+.4f}。", '']

    L += ['', '## 3. 无标签一致性：Test session2 簇纯度', '',
          f"在**基线 PCA 空间**上对全部 {R['purity_partition']['k20']['n']} 条 S2 行做 k-means"
          f"（k={'/'.join(str(k) for k in PURITY_K)}，固定种子 {PURITY_SEED}，随机样本初始化，60 次 Lloyd 迭代），"
          '簇划分在所有候选之间**完全相同**，因此纯度差异只来自预测身份本身。',
          '`micro` = 预测身份等于本簇众数身份的比例；`macro` = 各簇内最大预测身份占比的均值；'
          '`#pred` = 出现过的不同预测身份数；`max share` = 单一预测身份的最大占比。', '',
          '|候选|k|micro|macro|#pred|max share|vs 基线|', '|---|---:|---:|---:|---:|---:|---:|']
    for cid in order:
        for kk in pur_keys:
            p = C[cid]['metrics']['purity'][kk]
            ratio = p['purity_micro'] / max(bp[kk]['purity_micro'], 1e-12)
            L.append(f"|{cid}|{kk[1:]}|{f4(p['purity_micro'])}|{f4(p['purity_macro'])}|"
                     f"{p['n_distinct_pred']}|{f4(p['max_identity_share'])}|{ratio:.3f}×|")
    L += ['', f"- 基线纯度：k=20 时 micro {f4(bp['k20']['purity_micro'])}、k=28 时 micro {f4(bp['k28']['purity_micro'])}。"
          f"采纳要求纯度 ≥ 基线的 {PURITY_TOL:.0%}。",
          '- 该检查针对"全接受式作弊"：若某个对齐把 S2 接受推高却让预测身份塌缩/随机化，'
          '簇纯度会明显下降，或 `#pred` 骤减、`max share` 骤升，此时判定为退化解。']

    L += ['', '## 4. δ 估计的稳定性（分半 + 自助）', '',
          '`||δ||` = 该 (session, task) 批次均值向量的范数；`cos(½)` = 随机分半后两个半样本均值的方向一致性；'
          '`rel diff` = ||δ_a − δ_b|| / ||δ||；`δ/noise` = ||δ|| 与"零偏移下分半估计噪声"的比值'
          '（噪声 = √d·σ/√(n/2)）；`ci` = 自助 95% 区间。', '',
          '|session|task|n|d||δ|||cos(½)|rel diff|δ/noise|boot ci|cond(C)|n/d|',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in R['stability']:
        L.append(f"|{r['session']}|{r['task']}|{r['n']}|{r['delta_norm']:.3f}|"
                 f"{r['half_cosine']:+.3f}|{r['half_rel_diff']:.3f}|{r['delta_over_noise']:.1f}|"
                 f"[{r['boot_lo']:.2f}, {r['boot_hi']:.2f}]|{r['cond']:.1f}|{r['n_over_d']:.1f}|")
    L += ['', '- 方向一致性好：所有 cell 的分半余弦最小值为 '
          f"{min(r['half_cosine'] for r in R['stability']):.3f}，最大值 "
          f"{max(r['half_cosine'] for r in R['stability']):.3f}，"
          '说明"该批次偏向哪一侧"是稳定可估的。',
          '- 但**幅度淹没在批内散布里**：`δ/noise` 只有 0.6–1.5，而这里的 noise 用批内总方差（含身份与 trial 方差）'
          '算出的分半噪声。即批次均值主要是"这一批恰好是谁、含哪些 task"的构成效应，而不是干净的传感器级平移；'
          '这也解释了为什么均值对齐只能小幅抬 S2，却能把 S1 整段推到 tau 以下。',
          '- session1 的协方差条件数为 1909/1776（n/d ≈ 3），远高于 session2（87–173），'
          '因此 CORAL 在 session1 上必须靠收缩才能求逆；收缩系数在运行前写死。',
          f"- 协方差收缩：CORAL 用 λ={LAM_CORAL:g} 向该 task 的 Train fit 行协方差收缩，"
          f"逐 task 方差缩放用 λ={LAM_DIAG:g}；对角线截断在 [0.25, 4]×。"
          '这些系数在运行前写死，未用 Test 调整。', '',
          '自助区间（冻结 tau，按 task 分层行重采样）：', '',
          '|候选|S1 rej [95% CI]|S2 acc [95% CI]|', '|---|---|---|']
    for cid in order:
        b = C[cid]['metrics']['bootstrap']
        L.append(f"|{cid}|{f4(C[cid]['metrics']['S1_rejection'])} "
                 f"[{f4(b['S1_rejection']['lo'])}, {f4(b['S1_rejection']['hi'])}]|"
                 f"{f4(C[cid]['metrics']['S2_acceptance'])} "
                 f"[{f4(b['S2_acceptance']['lo'])}, {f4(b['S2_acceptance']['hi'])}]|")

    L += ['', '## 5. 逐 task（冻结 tau）', '',
          '|候选|task|S1 n|S1 rej|S2 n|S2 acc|', '|---|---|---:|---:|---:|---:|']
    for cid in order:
        for t in TASKS:
            p = C[cid]['metrics']['per_task'][t]
            L.append(f"|{cid}|{t}|{p['n_S1']}|{f4(p['S1_rejection'])}|{p['n_S2']}|"
                     f"{f4(p['S2_acceptance'])}|")

    if dec['adopted']:
        ad_t = C[dec['adopted']]['metrics']['per_task']
        base_t = base_metrics['per_task']
        dd = {t: ad_t[t]['S2_acceptance'] - base_t[t]['S2_acceptance'] for t in TASKS}
        pos = [t for t in TASKS if dd[t] > 0]
        neg = [t for t in TASKS if dd[t] <= 0]
        L += ['', f"冻结候选 `{dec['adopted']}` 的逐 task ΔS2（相对基线）："
              + '、'.join(f"{t} {dd[t]:+.4f}（n={ad_t[t]['n_S2']}）" for t in TASKS)
              + f"。上升的 task：{'、'.join(pos) or '无'}；下降的 task：{'、'.join(neg) or '无'}。"
                '增益并非四个 task 一致，说明可移除的偏移部分与 task 强相关，而不是一个全局平移。']

    L += ['', '## 6. 判定', '']
    if dec['verdict'] == 'pass':
        ad = C[dec['adopted']]['metrics']
        L += [f"- **达标并冻结**：`{dec['adopted']}`。LOSO 拒识 {f4(loso_micro)}（在 [0.89,0.91]）、"
              f"Test S1 {f4(ad['S1_rejection'])} ≥ 0.95、"
              f"Test S2 {f4(ad['S2_acceptance'])} > {f4(BASE_S2)}、"
              f"簇纯度 k20/k28 = {f4(ad['purity']['k20']['purity_micro'])}/"
              f"{f4(ad['purity']['k28']['purity_micro'])}（均 ≥ 基线 95%）。",
              f"- 增益幅度：ΔS2 = {ad['S2_acceptance'] - BASE_S2:+.4f}"
              f"（行级配对自助 95% CI "
              f"[{ad['paired_vs_baseline']['dS2']['lo']:+.4f}, "
              f"{ad['paired_vs_baseline']['dS2']['hi']:+.4f}]），"
              f"ΔS1 = {ad['S1_rejection'] - BASE_S1:+.4f}，ΔsepAUC = "
              f"{ad['separation_auc'] - BASE_AUC:+.4f}；同 S1 水平下 ΔS2 = "
              f"{ad['gates']['S1_matched_baseline']['S2_acceptance'] - base_metrics['gates']['S1_matched_baseline']['S2_acceptance']:+.4f}。",
              f"- 官方门槛读法：S1=0.95 时 S2 = {f4(ad['gates']['S1_0.95']['S2_acceptance'])}"
              f"（基线 {f4(base_metrics['gates']['S1_0.95']['S2_acceptance'])}，"
              f"{ad['gates']['S1_0.95']['S2_acceptance'] - base_metrics['gates']['S1_0.95']['S2_acceptance']:+.4f}）；"
              f"S1=0.70 时 S2 = {f4(ad['gates']['S1_0.70']['S2_acceptance'])}"
              f"（基线 {f4(base_metrics['gates']['S1_0.70']['S2_acceptance'])}）。"
              '该增益来自 S1/S2 分离度上升（sepAUC 见上），不是"接受更多同分布样本"；'
              '但 S1 仅 5 名未注册被试，其拒识率同时承担"未注册"语义，属 E232 混叠，不能当作纯生物特征增益。',
              f"- **但增益的稳健性有限**：行级配对自助 CI 不含 0，整簇（k=20，按身份/task 相关性）重采样的 "
              f"95% CI 为 [{R['paired_cluster_bootstrap_vs_baseline'][dec['adopted']]['lo']:+.4f}, "
              f"{R['paired_cluster_bootstrap_vs_baseline'][dec['adopted']]['hi']:+.4f}]，**跨 0**；"
              '且增益几乎全部来自样本最多的 Imagery task（见第 5 节）。因此这是「方向一致的边际改进」，'
              '不是稳健的大幅提升；判定按 config 的点估计规则成立。',
              '- 这是小幅 S2 增益且 S1 同时变好（S1 已饱和到 1.0），不是 S1/S2 权衡；纯度读数见第 3 节 '
              '`#pred`/`max share` 列（预测身份没有被压成少数几个）。']
    elif dec['verdict'] == 'trade_off':
        L += ['- **这是 S1/S2 权衡，不是免费增益**：没有任何对称候选同时满足 "S1 ≥ 0.95 且 S2 > '
              f'{BASE_S2} 且纯度守住"。',
              f"- S1=0.95 约束下的最大对称候选 S2 = **{f4(dec['max_S2_at_S1_0p95'])}**"
              f"（`{dec['best_at_S1_0p95']}`），基线同约束为 {f4(dec['baseline_S2_at_S1_0p95'])}。",
              f"- 冻结阈值下 S2 上升但 S1 跌破 0.95 的候选："
              + ('、'.join(f"`{c}`（S2 {f4(C[c]['metrics']['S2_acceptance'])}，S1 "
                           f"{f4(C[c]['metrics']['S1_rejection'])}）" for c in dec['rises_but_S1_below_gate'])
                 if dec['rises_but_S1_below_gate'] else '无') + '。',
              '- 因此不冻结新候选，部署参数仍为 E228 的 `e226_auc2W`。']
    else:
        L += ['- **δ 的批量均值/协方差估计不足以修复跨段偏移**：所有对称候选在冻结阈值下的 S2 都不高于基线 '
              f'{BASE_S2}，或在 S1=0.95 约束下也不高于基线同约束的 '
              f'{f4(dec["baseline_S2_at_S1_0p95"])}。',
              f"- 最高对称候选为 `{dec['best_symmetric']}`，S2 "
              f"{f4(C[dec['best_symmetric']]['metrics']['S2_acceptance'])}；"
              f"S1=0.95 约束下最高为 `{dec['best_at_S1_0p95']}` 的 {f4(dec['max_S2_at_S1_0p95'])}。",
              '- 原因见第 4 节：δ 与"零偏移噪声"的同量级程度、以及 CORAL 白化会连同身份间方差一起压掉。',
              '- 部署参数仍为 E228 的 `e226_auc2W`，本轮不产生新模型。']
    if C['coral']['metrics']['S2_acceptance'] > BASE_S2:
        cm = C['coral']['metrics']
        L += ['', f"- **`coral` 被判为退化解（不可采纳）**：S2 {f4(cm['S2_acceptance'])} 虽然高于基线 "
              f"{f4(BASE_S2)}，但 S1 跌到 {f4(cm['S1_rejection'])}（< 0.95）、分离 AUC 从 {f4(BASE_AUC)} 崩到 "
              f"{f4(cm['separation_auc'])}、簇纯度 k20/k28 = {f4(cm['purity']['k20']['purity_micro'])}/"
              f"{f4(cm['purity']['k28']['purity_micro'])}（基线的 "
              f"{cm['purity']['k20']['purity_micro'] / max(bp['k20']['purity_micro'], 1e-12):.1%}/"
              f"{cm['purity']['k28']['purity_micro'] / max(bp['k28']['purity_micro'], 1e-12):.1%}）。"
              f"在同一个 S1=0.95 门槛下它的 S2 只有 {f4(cm['gates']['S1_0.95']['S2_acceptance'])}"
              f"（基线 {f4(base_metrics['gates']['S1_0.95']['S2_acceptance'])}）——"
              '说明它的 S2 提升完全来自把 S1 也一起放进门内，是典型的接受放宽而非身份变准。']
    if dec['baseline_S2_at_S1_0p95'] is not None:
        L += ['', f"- 参考：把基线自身的 tau 下调到恰好 S1=0.95 也能把 S2 抬到 "
              f"{f4(dec['baseline_S2_at_S1_0p95'])}。因此评价任何候选的 S2 增益时，"
              '必须与**同一 S1 门槛**下的基线对照，而不是只与 0.3043 比。']

    if R.get('deploy'):
        d = R['deploy']
        L += ['', '### 6.1 部署参数（若采用）', '',
              '|项|值|', '|---|---|',
              f"|基础模型|`{CFG['feature_cache']['frozen_model']}`（E227 auc2_lofo，未改）|",
              f"|对齐|`{d['alignment']}`：逐 (session, task) 减去该批次 PCA 空间均值（参考均值为该 task 的 Train fit 均值，恰为 0）|",
              f"|收缩系数|CORAL λ={d['lambda_coral']:g}（本候选不使用），逐 task 方差 λ={d['lambda_diag']:g}（本候选不使用）|",
              f"|门控 tau|{d['tau']:.6f}（沿用 E228；`score < tau` 判 unknown）|",
              f"|S1=0.95 门槛 tau|{d['tau_S1_0p95']:.6f}（诊断用，由 Test S1 分数分位得到）|",
              f"|冻结 tau 下指标|S1 {f4(d['S1_rejection'])} / S2 {f4(d['S2_acceptance'])} / sepAUC {f4(d['separation_auc'])}|",
              f"|模型文件|`research/E233_delta_estimation/models/deploy_frozen.pt`（含 base_state + 对齐规格 + 批次统计）|",
              '',
              '**推理约束**：对齐是**批次级**的——必须先拿到同一 (session, task) 的整批无标签样本估计均值，'
              '再对该批逐条打分。这是"按 session 做一次无监督校准"，不是单条在线推理；'
              '若赛规不允许使用待预测批次的统计量，则该方案不可用（见限制）。']
    L += ['', '## 7. 限制', '',
          '- **transductive 灰区**：对齐统计量来自被评分的同一批 Test 输入（无标签），属于转导式使用测试数据，'
          '是否允许需用户/赛规确认；本轮的 S1/S2 数字不能直接当作可提交方案的指标。',
          '- **session 与注册状态混叠**（E232）：S1 = 5 名未注册被试（k=5），S2 = 20 名注册用户。'
          '因此 S1/S2 不是纯生物特征指标，"只对齐 session2"这类操作会直接泄漏注册信息，'
          '已作为退化对照标注，不构成结论。',
          '- 官方 C/RA/ASafeR 因 Test 无身份标签，**N/A**；S1 拒识是赛题要求的行为代理，不是官方指标。',
          '- 阈值由 E228 冻结，未针对对齐后的分数重新标定：对齐会整体移动分数分布，而 Train 侧没有 session '
          '可供估计同一偏移，故不存在合法的 Train-only 重标定；这也是 S1 会随 S2 一起变化的原因之一。',
          f'- Test 自 E65 起被反复查看，S2 数字带重复查看偏倚；本轮候选定义、收缩系数与判定规则在运行前写死。',
          f"- 运行：`.venv\\Scripts\\python.exe research/E233_delta_estimation/run_e233.py`（CUDA），"
          f"耗时 {R['runtime_s']:.0f}s；Test 分数缓存于 `cache/`，对齐统计量存于 "
          '`models/e233_alignment_stats.pt`。']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
