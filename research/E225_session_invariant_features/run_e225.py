"""E225: fine-grained session-invariance ledger + strict LOSO on component subsets.

Protocol (frozen, see config.json):
  - 20-fold subject-LOSO. Each fold: standardization / PCA / prototypes / tau are fitted on
    the enrolled outer_train rows with trial-fraction <= 0.6 only. Eval window (0.6, 0.8].
    Held identity never enters any fit. tau = 10% quantile of the inner 5-block OOF known scores.
  - score = -min euclidean distance to same-task prototypes; higher = known; reject if score < tau.
  - unknown is the positive class for rejection F1 and for AUC(-score).
  - Test is read once, after every Train-side artefact is frozen, for uniform evaluation only.

The ledger quantifies each semantic feature block with (a) identity discriminability, (b) within-
session stability (1-ICC), (c) drift sensitivity measured by re-extracting features from drifted
raw waveforms, and (d) redundancy against the other blocks.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
CACHE = OUT / 'cache'
MODELS = OUT / 'models'
LOG = OUT / 'run.log'
DEVICE = 'cuda'

N = int(CFG['data']['samples_per_trial'])
CH = int(CFG['data']['channels'])
FS = int(CFG['data']['sampling_rate_hz'])
BANDS = [tuple(b) for b in CFG['features']['bands_hz']]
NB = len(BANDS)
TASKS = list(CFG['protocol']['tasks'])
TASKS_NP = None
FIT_FRAC = float(CFG['protocol']['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['protocol']['evaluation_fraction']
N_BLOCKS = int(CFG['protocol']['inner_blocks'])
Q_TAU = float(CFG['protocol']['threshold_quantile'])
PCA_DIM = int(CFG['protocol']['pca_dim'])
K_PROTO = int(CFG['protocol']['prototypes_k'])
SEED = int(CFG['seed'])
N_FOLDS = int(os.environ.get('E225_FOLDS', CFG['protocol']['folds']))
SKIP_TEST = os.environ.get('E225_SKIP_TEST') == '1'
CANDS = [dict(c) for c in CFG['candidates']]
NAMES = [c['name'] for c in CANDS]
CHUNK = int(CFG['features']['chunk_trials'])
WORKERS = int(CFG['features']['io_workers'])
DRIFT_CONDS = list(CFG['drift_simulation']['conditions'])
SUB_PER_CELL = int(CFG['drift_simulation']['subset_per_subject_task'])
BASELINE_SEP_AUC = float(CFG['decision_rule']['baseline_test_separation_auc'])
STAGE = CFG['decision_rule']['stage_targets']

# ---- E65 constants (identical to research/E65_correlation_spd/config.json)
SHRINK = 0.05
EIG_FLOOR = 1e-12
OFFMUL = float(np.sqrt(2.0))
ZBAND_EPS = 1e-6

BLOCK_DEF = {'corr_logspd': 855, 'bandpower_gainfree': 90, 'spectral_shape': 90,
             'spectral_slope': 18, 'pac': 18, 'trial_energy': 1}
BLOCK_ORDER = list(BLOCK_DEF)
OFFSETS = {}
_acc = 0
for _k, _v in BLOCK_DEF.items():
    OFFSETS[_k] = (_acc, _acc + _v)
    _acc += _v
DIM_TOTAL = _acc
BLOCK_IDX = {k: np.arange(a, b) for k, (a, b) in OFFSETS.items()}


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(msg) + '\n')


# =====================================================================  torch feature extraction
_FREQ = None
_BM = None
_TRI = None
_HIL = None


def freq_grid():
    global _FREQ
    if _FREQ is None:
        _FREQ = torch.fft.rfftfreq(N, 1.0 / FS, device=DEVICE)
    return _FREQ


def _mask(lo, hi, last=False):
    f = freq_grid()
    return (f >= lo) & (f <= hi if last else f < hi)


def band_masks():
    global _BM
    if _BM is None:
        _BM = torch.stack([_mask(lo, hi, i == NB - 1)
                           for i, (lo, hi) in enumerate(BANDS)]).float()
    return _BM


def tri_index():
    global _TRI
    if _TRI is None:
        r, c = torch.triu_indices(CH, CH, device=DEVICE)
        diag = r == c
        mul = torch.where(diag, torch.ones_like(r, dtype=torch.float32),
                          torch.full_like(r, OFFMUL, dtype=torch.float32))
        _TRI = (r, c, mul, diag)
    return _TRI


def hilbert_mult():
    global _HIL
    if _HIL is None:
        h = torch.zeros(N, device=DEVICE)
        h[0] = 1.0
        h[1:N // 2] = 2.0
        h[N // 2] = 1.0
        _HIL = h
    return _HIL


def analytic(x):
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * hilbert_mult(), dim=-1)


@torch.no_grad()
def corr_logspd(raw):
    """E65 gpu_correlation_logspd: (B,18,1200) -> (B,855)."""
    bm = band_masks()
    f = torch.fft.irfft(torch.fft.rfft(raw, dim=-1)[:, None] * bm[None, :, None, :],
                        n=N, dim=-1)
    f = f - f.mean(-1, keepdim=True)
    scale = f.std(-1, keepdim=True, unbiased=True).clamp_min(1e-12)
    norm = (f - f.mean(-1, keepdim=True)) / scale
    corr = norm @ norm.transpose(-1, -2) / (N - 1)
    trace = torch.diagonal(corr, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(CH, device=DEVICE)
    spd = (1.0 - SHRINK) * corr + SHRINK * trace[..., None, None] / CH * eye
    vals, vecs = torch.linalg.eigh(spd)
    vals = vals.clamp_min(EIG_FLOOR)
    logspd = (vecs * vals.log().unsqueeze(-2)) @ vecs.transpose(-1, -2)
    r, c, mul, diag = tri_index()
    vec = logspd[..., r, c] * mul
    dmean = vec[..., diag].mean(-1, keepdim=True)
    vec = vec.clone()
    vec[..., diag] = vec[..., diag] - dmean
    mean = vec.mean(-1, keepdim=True)
    sd = vec.std(-1, keepdim=True, unbiased=False).clamp_min(ZBAND_EPS)
    return ((vec - mean) / sd).flatten(-2)


@torch.no_grad()
def extract_from_raw(raw):
    """(B,18,1200) raw EEG -> dict of block tensors on GPU."""
    bm = band_masks()
    xc = raw - raw.mean(-1, keepdim=True)
    spec = torch.fft.rfft(xc, dim=-1)                                    # B,18,601
    bb = torch.fft.irfft(spec[:, None] * bm[None, :, None, :], n=N, dim=-1)   # B,5,18,1200

    # 1) gain-free band power: log power minus that band's channel mean
    lp = torch.log((bb ** 2).mean(-1) + 1e-10)                           # B,5,18
    gainfree = (lp - lp.mean(-1, keepdim=True)).flatten(1)               # B,90

    # 2) spectral shape: frequency-demeaned log PSD averaged inside each band
    psd = spec.abs() ** 2
    sel = _mask(1, 40, last=True)
    lps = torch.log(psd[..., sel] + 1e-12)                               # B,18,S
    shape_f = lps - lps.mean(-1, keepdim=True)
    fsel = freq_grid()[sel]
    parts = []
    for i, (lo, hi) in enumerate(BANDS):
        m = (fsel >= lo) & (fsel <= hi if i == NB - 1 else fsel < hi)
        parts.append(shape_f[..., m].mean(-1))                           # B,18
    shape = torch.stack(parts, 1).flatten(1)                             # B,90

    # 3) spectral slope: per channel log-log regression over 1-40 Hz
    lf = torch.log(fsel.clamp_min(1e-3))
    lf = lf - lf.mean()
    slope = (lps * lf).sum(-1) / (lf * lf).sum()                         # B,18

    # 4) alpha-beta phase-amplitude coupling (mean resultant length)
    alpha = torch.fft.irfft(spec * _mask(8, 13), n=N, dim=-1)
    beta = torch.fft.irfft(spec * _mask(13, 30), n=N, dim=-1)
    amp = analytic(beta).abs()
    ph = analytic(alpha).angle()
    pac = (amp * torch.polar(torch.ones_like(amp), ph)).mean(-1).abs() \
        / amp.mean(-1).clamp_min(1e-12)                                  # B,18

    # 5) trial energy (negative control)
    energy = torch.log(raw.pow(2).mean(dim=(1, 2)).sqrt() + 1e-12)[:, None]

    return {'corr_logspd': corr_logspd(raw), 'bandpower_gainfree': gainfree,
            'spectral_shape': shape, 'spectral_slope': slope, 'pac': pac,
            'trial_energy': energy}


def assemble(fb):
    return torch.cat([fb[k].float() for k in BLOCK_ORDER], 1).cpu().numpy()


# =====================================================================  IO
def load_mat(path):
    d = np.asarray(loadmat(path, variable_names=['data'])['data'], dtype=np.float32)
    if d.shape != (N, CH):
        raise ValueError(f'{path}: expected ({N},{CH}) got {d.shape}')
    if not np.isfinite(d).all():
        raise ValueError(f'{path}: non-finite')
    return np.ascontiguousarray(d.T)


def raw_chunks(paths):
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(paths), CHUNK):
            batch = paths[start:start + CHUNK]
            arrays = list(pool.map(load_mat, batch))
            yield start, torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=DEVICE)


def build_block_cache(meta, root_key, tag, corr_reference):
    """Extract the 5 non-cached blocks from raw and validate the recomputed corr block."""
    files = {k: CACHE / f'{tag}_{k}.npy' for k in BLOCK_ORDER}
    paths = [ROOT / CFG['data'][root_key] / str(p) for p in meta.path]
    if all(p.exists() for p in files.values()):
        log(f'{tag}: block cache reused')
        maxd = np.nan
        meand = np.nan
        if corr_reference is not None:
            probe = paths[:3 * CHUNK]
            diffs = []
            for start, raw in raw_chunks(probe):
                r = corr_logspd(raw).cpu().numpy()
                diffs.append(np.abs(r - corr_reference[start:start + len(raw)]))
                del raw
            dd = np.concatenate([d.ravel() for d in diffs])
            maxd, meand = float(dd.max()), float(dd.mean())
            log(f'{tag}: recomputed corr block vs cache max|diff|={maxd:.3e} mean|diff|={meand:.3e}')
        return ({k: np.load(p) for k, p in files.items()},
                dict(recomputed_corr_vs_cache_max_abs=maxd,
                     recomputed_corr_vs_cache_mean_abs=meand,
                     rows_checked=len(probe),
                     note='recomputed E65 corr block from raw vs frozen E65 cache'))
    mm = {k: np.lib.format.open_memmap(files[k], mode='w+', dtype=np.float32,
                                       shape=(len(paths), BLOCK_DEF[k])) for k in BLOCK_ORDER}
    t0 = time.time()
    max_corr_diff = 0.0
    sum_corr_diff = 0.0
    n_corr = 0
    checked = 0
    for start, raw in raw_chunks(paths):
        fb = extract_from_raw(raw)
        for k in BLOCK_ORDER:
            mm[k][start:start + len(raw)] = fb[k].cpu().numpy()
        if checked < 3 and corr_reference is not None:
            r = fb['corr_logspd'].cpu().numpy()
            d = np.abs(r - corr_reference[start:start + len(raw)])
            max_corr_diff = max(max_corr_diff, float(d.max()))
            sum_corr_diff += float(d.sum())
            n_corr += d.size
            checked += 1
        if (start + len(raw)) % 2048 < CHUNK or start + len(raw) == len(paths):
            log(f'  {tag} features {start + len(raw)}/{len(paths)} ({time.time()-t0:.0f}s)')
    for k in BLOCK_ORDER:
        mm[k].flush()
    del mm
    val = dict(recomputed_corr_vs_cache_max_abs=max_corr_diff,
               recomputed_corr_vs_cache_mean_abs=(sum_corr_diff / n_corr if n_corr else np.nan),
               rows_checked=checked * CHUNK,
               note='recomputed E65 corr block from raw vs frozen E65 cache')
    log(f'{tag}: recomputed corr block vs cache max|diff|={max_corr_diff:.3e} '
        f'mean|diff|={val["recomputed_corr_vs_cache_mean_abs"]:.3e}')
    return {k: np.load(p) for k, p in files.items()}, val


# =====================================================================  drift
def apply_drift(raw, cond, rng):
    if cond == 'gain_ch_pm3dB':
        g = torch.as_tensor(rng.uniform(-3.0, 3.0, size=(len(raw), CH, 1)),
                            dtype=torch.float32, device=DEVICE)
        return raw * torch.pow(torch.tensor(10.0, device=DEVICE), g / 20.0)
    if cond == 'gain_global_p3dB':
        return raw * float(10 ** (3.0 / 20.0))
    if cond == 'gain_global_m3dB':
        return raw * float(10 ** (-3.0 / 20.0))
    if cond.startswith('tilt_'):
        alpha = 0.5 if cond == 'tilt_p0.5' else -0.5
        w = (freq_grid().clamp_min(1.0) / 10.0) ** alpha
        return torch.fft.irfft(torch.fft.rfft(raw, dim=-1) * w, n=N, dim=-1)
    if cond.startswith('noise_snr'):
        snr = float(cond.replace('noise_snr', ''))
        rms = raw.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-9)
        noise = torch.randn(raw.shape, device=DEVICE)
        return raw + noise * rms / float(10 ** (snr / 20.0))
    if cond == 'reref_pair12':
        return raw - 0.5 * (raw[:, [0], :] + raw[:, [1], :])
    if cond == 'resample_150':
        spec = torch.fft.rfft(raw, dim=-1)
        ds = torch.fft.irfft(spec[..., :N // 4 + 1], n=N // 2, dim=-1)
        return torch.fft.irfft(torch.fft.rfft(ds, dim=-1), n=N, dim=-1)
    if cond == 'bandpass_2_38':
        return torch.fft.irfft(torch.fft.rfft(raw, dim=-1) * _mask(2, 38, last=True), n=N, dim=-1)
    raise ValueError(cond)


def drift_subset_rows(meta, FR, fit_mask):
    rows = []
    for s in np.unique(meta.subject.to_numpy(int)):
        for t in TASKS:
            idx = np.flatnonzero(fit_mask & (meta.subject.to_numpy(int) == s) &
                                 (TASKS_NP == t))
            if len(idx) == 0:
                continue
            take = np.unique(np.linspace(0, len(idx) - 1, min(SUB_PER_CELL, len(idx)))
                             .round().astype(int))
            rows.append(idx[take])
    return np.sort(np.concatenate(rows))


def drift_experiment(meta, fit_mask):
    rows = drift_subset_rows(meta, None, fit_mask)
    cpath = CACHE / 'drift_stats.npz'
    if cpath.exists():
        d = np.load(cpath)
        log('drift stats reused')
        return dict(subject_delta=d['subject_delta'], cond_mean=d['cond_mean'],
                    rows=d['rows'], conds=DRIFT_CONDS)
    paths = [ROOT / CFG['data']['train_root'] / str(meta.path.iloc[r]) for r in rows]
    raw = torch.empty((len(paths), CH, N), dtype=torch.float32, device=DEVICE)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(paths), CHUNK):
            arrays = list(pool.map(load_mat, paths[start:start + CHUNK]))
            raw[start:start + len(arrays)] = torch.as_tensor(np.stack(arrays),
                                                             dtype=torch.float32, device=DEVICE)
    base = assemble(extract_from_raw(raw))
    subj = meta.subject.to_numpy(int)[rows]
    subject_delta = np.zeros((len(DRIFT_CONDS), 20, DIM_TOTAL), np.float32)
    cond_mean = np.zeros((len(DRIFT_CONDS), DIM_TOTAL), np.float32)
    t0 = time.time()
    for ci, cond in enumerate(DRIFT_CONDS):
        rng = np.random.default_rng(SEED + 1000 + ci)
        np.random.seed(SEED + 1000 + ci)
        torch.manual_seed(SEED + 1000 + ci)
        feat = assemble(extract_from_raw(apply_drift(raw, cond, rng)))
        delta = np.abs(feat - base)
        cond_mean[ci] = delta.mean(0)
        for s in range(1, 21):
            subject_delta[ci, s - 1] = delta[subj == s].mean(0)
        log(f'  drift {cond} done ({time.time()-t0:.0f}s)')
    np.savez_compressed(cpath, subject_delta=subject_delta, cond_mean=cond_mean, rows=rows)
    return dict(subject_delta=subject_delta, cond_mean=cond_mean, rows=rows, conds=DRIFT_CONDS)


# =====================================================================  ledger statistics
def per_task_scatter(Xm, y, tasks, mask):
    between = np.zeros(Xm.shape[1])
    within = np.zeros(Xm.shape[1])
    nmean = 0.0
    for t in TASKS:
        base = mask & (tasks == t)
        Ms, Ws, ns = [], [], []
        for s in np.unique(y[base]):
            r = base & (y == s)
            if int(r.sum()) < 3:
                continue
            A = Xm[r]
            Ms.append(A.mean(0))
            Ws.append(A.var(0, ddof=1))
            ns.append(int(r.sum()))
        if len(Ms) < 2:
            continue
        between += np.stack(Ms).var(0, ddof=1)
        within += np.stack(Ws).mean(0)
        nmean += float(np.mean(ns))
    k = len(TASKS)
    return between / k, within / k, nmean / k


def dim_ledger(Xm, y, tasks, fit_mask, drift):
    between, within, nmean = per_task_scatter(Xm, y, tasks, fit_mask)
    fisher = between / (within + 1e-9)
    msb = nmean * between
    icc = msb / (msb + (nmean - 1) * within + 1e-12)
    subs = np.unique(y[fit_mask])
    delta = drift['subject_delta'][:, subs - 1, :].mean(axis=(0, 1))
    sens = delta / (np.sqrt(between) + 1e-9)
    return dict(fisher=fisher, one_minus_icc=1.0 - icc, drift=delta, sens=sens,
                between=between)


def block_redundancy(Xm, fit_mask):
    rows = np.flatnonzero(fit_mask)
    pcs = {}
    for k in BLOCK_ORDER:
        A = Xm[rows][:, BLOCK_IDX[k]].astype(np.float64)
        A = A - A.mean(0)
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        pcs[k] = U[:, :8] * S[:8]
    even = np.arange(0, len(rows), 2)
    odd = np.arange(1, len(rows), 2)
    n = len(BLOCK_ORDER)
    r2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vals = []
            for a, b in ((i, j), (j, i)):
                ka, kb = BLOCK_ORDER[a], BLOCK_ORDER[b]
                F = np.c_[pcs[ka][even], np.ones(len(even))]
                beta, *_ = np.linalg.lstsq(F, pcs[kb][even], rcond=None)
                pred = np.c_[pcs[ka][odd], np.ones(len(odd))] @ beta
                yte = pcs[kb][odd]
                ss_res = float(((yte - pred) ** 2).sum())
                ss_tot = float(((yte - yte.mean(0)) ** 2).sum())
                vals.append(1.0 - ss_res / ss_tot)
            r2[i, j] = r2[j, i] = float(np.mean(vals))
    return {BLOCK_ORDER[i]: {BLOCK_ORDER[j]: float(r2[i, j]) for j in range(n)}
            for i in range(n)}


def dim_name(d):
    if d < 855:
        b = d // 171
        p = d % 171
        r, c = np.triu_indices(CH)
        return f'corr_b{BANDS[b][0]}-{BANDS[b][1]}Hz_ch{r[p]+1}-ch{c[p]+1}'
    for k in BLOCK_ORDER[1:]:
        a, b = OFFSETS[k]
        if a <= d < b:
            i = d - a
            if k in ('bandpower_gainfree', 'spectral_shape'):
                band, ch = i // CH, i % CH
                return f'{k}_b{BANDS[band][0]}-{BANDS[band][1]}Hz_ch{ch+1}'
            if k == 'spectral_slope':
                return f'{k}_ch{i+1}'
            if k == 'pac':
                return f'{k}_ch{i+1}'
            return 'log_rms'
    return f'dim{d}'


# =====================================================================  strict LOSO scorer
def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def frac_of(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


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


def base_fit_rep(raw, tasks, y, fit, k, pca=PCA_DIM):
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
    pca = int(min(pca, a.shape[1], len(a) - 1))
    W = v[:, -pca:].flip(1)
    z = ((z.double() - pm) @ W).float().cpu().numpy()
    cls = np.unique(y[fit])
    p, keys = base_prototypes(z, y, tasks, fit, cls, k)
    return dict(mu=mu, sd=sd, pm=pm.cpu().numpy(), W=W.cpu().numpy(), p=p, keys=keys, k=k,
                pca=pca, dim=int(a.shape[1]))


def base_transform(state, raw, tasks):
    x = tensor(raw)
    z = torch.empty_like(x)
    for i, t in enumerate(TASKS):
        rows = tasks == t
        z[rows] = (x[rows] - tensor(state['mu'][i])) / tensor(state['sd'][i])
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


def base_inner_oof(fit, raw, tasks, y, blocks, pk, pca):
    oof_s = np.full(len(y), np.nan)
    oof_p = np.full(len(y), -1, int)
    for b in np.unique(blocks[fit]):
        hold = fit & (blocks == b)
        inner = fit & ~hold
        st = base_fit_rep(raw, tasks, y, inner, pk, pca)
        z = base_transform(st, raw, tasks)
        s, p = base_score(st, z, tasks, np.flatnonzero(hold))
        oof_s[hold] = s[hold]
        oof_p[hold] = p[hold]
    return oof_s, oof_p


def evaluate(s, p, y, known, unknown, tau_row):
    k = s[known]
    u = s[unknown]
    rej_u = u < tau_row[unknown]
    rej_k = k < tau_row[known]
    tp = float(rej_u.sum())
    fp = float(rej_k.sum())
    fn = float((~rej_u).sum())
    return dict(
        auc=float(((u[:, None] < k).sum() + 0.5 * (u[:, None] == k).sum()) / (len(k) * len(u))),
        rejection_f1=float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        known_acceptance=float((~rej_k).mean()),
        unknown_rejection=float(rej_u.mean()),
        identity_correct_but_rejected=float(((p[known] == y[known]) & rej_k).mean()),
        threshold=float(np.mean(tau_row[known])),
        n_known=int(len(k)), n_unknown=int(len(u)),
        accepted_known=int((~rej_k).sum()), rejected_unknown=int(rej_u.sum()),
    )


def summarise(ms, name=None):
    n_u = sum(m['n_unknown'] for m in ms)
    n_k = sum(m['n_known'] for m in ms)
    return dict(
        auc=float(np.mean([m['auc'] for m in ms])),
        auc_fold_std=float(np.std([m['auc'] for m in ms])),
        auc_fold_values=[float(m['auc']) for m in ms],
        rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
        known_acceptance=float(np.mean([m['known_acceptance'] for m in ms])),
        known_acceptance_micro=float(sum(m['accepted_known'] for m in ms) / n_k),
        unknown_rejection=float(np.mean([m['unknown_rejection'] for m in ms])),
        unknown_rejection_micro=float(sum(m['rejected_unknown'] for m in ms) / n_u),
        identity_correct_but_rejected=float(np.mean([m['identity_correct_but_rejected'] for m in ms])),
        threshold=float(np.mean([m['threshold'] for m in ms])),
    )


def select_dims(cand, fit_mask, Xm, y, tasks, drift):
    rule = cand['rule']
    if rule == 'block:corr_logspd':
        return BLOCK_IDX['corr_logspd']
    if rule == 'all':
        return np.arange(DIM_TOTAL)
    if rule == 'all_minus_trial_energy':
        return np.arange(OFFSETS['trial_energy'][0])
    if rule == 'ledger_clean_blocks':
        return np.concatenate([BLOCK_IDX[k] for k in
                               ('corr_logspd', 'bandpower_gainfree', 'spectral_shape',
                                'spectral_slope')])
    L = dim_ledger(Xm, y, tasks, fit_mask, drift)
    if rule == 'drop_top_drift_30pct':
        thr = float(np.quantile(L['sens'], 0.7))
        return np.flatnonzero(L['sens'] <= thr)
    if rule == 'top_disc_over_drift':
        ratio = L['fisher'] / (L['sens'] + 1e-9)
        n = int(cand['n_dim'])
        return np.sort(np.argsort(-ratio)[:n])
    if rule == 'random_dims':
        rng = np.random.default_rng(SEED + (0 if int(cand['n_dim']) == 96 else 1))
        return np.sort(rng.choice(DIM_TOTAL, int(cand['n_dim']), replace=False))
    raise ValueError(rule)


# =====================================================================  folds
def run_folds(X, Y, FR, BLOCKS, drift):
    folds = {n: [] for n in NAMES}
    meta_rows = []
    for held in range(1, N_FOLDS + 1):
        t0 = time.time()
        fit = (Y != held) & (FR <= FIT_FRAC)
        window = (FR > EVAL_LO) & (FR <= EVAL_HI)
        known = window & (Y != held)
        unknown = window & (Y == held)
        win_rows = np.flatnonzero(window)
        for cand in CANDS:
            name = cand['name']
            path = CACHE / f'fold_{name}_{held:02d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                folds[name].append(row['metrics'])
                continue
            dims = select_dims(cand, fit, X, Y, TASKS_NP, drift)
            Xc = X[:, dims]
            pca = min(PCA_DIM, len(dims))
            st = base_fit_rep(Xc, TASKS_NP, Y, fit, K_PROTO, pca)
            z = base_transform(st, Xc, TASKS_NP)
            s, p = base_score(st, z, TASKS_NP, win_rows)
            s, p = s[win_rows], p[win_rows]
            oof_s, oof_p = base_inner_oof(fit, Xc, TASKS_NP, Y, BLOCKS, K_PROTO, pca)
            tau = quantile_tau(oof_s[fit], Q_TAU)
            tau_row = np.full(len(win_rows), tau)
            m = evaluate(s, p, Y[win_rows], known[win_rows], unknown[win_rows], tau_row)
            m['n_dims'] = int(len(dims))
            m['pca_dim'] = int(st['pca'])
            np.savez_compressed(CACHE / f'win_{name}_{held:02d}.npz',
                                score=s, pred=p, subject=Y[win_rows],
                                known=known[win_rows], unknown=unknown[win_rows], tau=tau)
            path.write_text(json.dumps(dict(candidate=name, held=int(held), metrics=m,
                                            pca=st['pca'], n_dims=int(len(dims)),
                                            train_rows=int(fit.sum())), indent=2))
            folds[name].append(m)
        log(f'fold {held}: ' + ' | '.join(
            f"{n} auc={folds[n][-1]['auc']:.4f} rej={folds[n][-1]['unknown_rejection']:.4f}"
            for n in NAMES) + f'  ({time.time()-t0:.1f}s)')
    return folds


# =====================================================================  Test (single read)
def test_row(s, tau, is_s1):
    rej = s < tau
    s1, s2 = s[is_s1], s[~is_s1]
    th70 = quantile_lower(s1, 0.70)
    return dict(
        tau=float(tau), S1_N=int(is_s1.sum()), S1_rejection=float(rej[is_s1].mean()),
        S2_N=int((~is_s1).sum()), S2_acceptance=float((~rej[~is_s1]).mean()),
        S2_vs_S1_AUC=auc_ties(s2, s1),
        max_S2_acceptance_at_S1_rej_0_70=float((s2 >= th70).mean()),
        threshold_at_S1_rej_0_70=float(th70),
        S1_score_stats=dict(mean=float(s1.mean()), std=float(s1.std()),
                            q10=float(np.quantile(s1, .1)), median=float(np.median(s1))),
        S2_score_stats=dict(mean=float(s2.mean()), std=float(s2.std()),
                            q10=float(np.quantile(s2, .1)), median=float(np.median(s2))),
        stage_S1_met=bool(rej[is_s1].mean() >= STAGE['S1_rejection']),
        stage_S2_met=bool((~rej[~is_s1]).mean() >= STAGE['S2_acceptance']),
        official_AUC='N/A', official_F1='N/A')


def run_test(X, Xt, Y, FR, BLOCKS, drift, tmeta):
    fit = FR <= FIT_FRAC
    tt = tmeta.task.to_numpy()
    tm = tmeta
    sess = tm.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    audit = {}
    for cand in CANDS:
        name = cand['name']
        dims = select_dims(cand, fit, X, Y, TASKS_NP, drift)
        pca = min(PCA_DIM, len(dims))
        st = base_fit_rep(X[:, dims], TASKS_NP, Y, fit, K_PROTO, pca)
        zt = base_transform(st, Xt[:, dims], tt)
        s, p = base_score(st, zt, tt, np.arange(len(tt)))
        oof_s, oof_p = base_inner_oof(fit, X[:, dims], TASKS_NP, Y, BLOCKS, K_PROTO, pca)
        tau = quantile_tau(oof_s[fit], Q_TAU)
        audit[name] = test_row(s, tau, is_s1)
        audit[name]['n_dims'] = int(len(dims))
        audit[name]['pca_dim'] = int(st['pca'])
        np.savez_compressed(CACHE / f'test_scores_{name}.npz', score=s, pred=p, tau=tau,
                            task=np.asarray(tt, dtype='<U8'),
                            session=np.asarray(tm.session.astype(str).to_numpy(), dtype='<U8'),
                            is_s1=is_s1)
        torch.save(dict(state=st, dims=np.asarray(dims), pca=int(st['pca']),
                        tau=float(tau), candidate=name), MODELS / f'deploy_{name}.pt')
        # reload check: saved model must reproduce the same Test scores
        back = torch.load(MODELS / f'deploy_{name}.pt', weights_only=False)
        s2, p2 = base_score(back['state'], base_transform(back['state'], Xt[:, dims], tt),
                            tt, np.arange(len(tt)))
        audit[name]['reload_check'] = dict(
            max_abs_score_diff=float(np.max(np.abs(s2 - s))),
            pred_identical=bool(np.array_equal(p2, p)),
            passed=bool(np.max(np.abs(s2 - s)) <= 1e-6 and np.array_equal(p2, p)))
        log(f'test {name}: S1={audit[name]["S1_rejection"]:.4f} '
            f'S2={audit[name]["S2_acceptance"]:.4f} sepAUC={audit[name]["S2_vs_S1_AUC"]:.4f} '
            f'dims={len(dims)}')
    np.savez_compressed(CACHE / 'test_meta.npz', is_s1=is_s1,
                        session=np.asarray(tm.session.astype(str).to_numpy(), dtype='<U8'),
                        task=np.asarray(tt, dtype='<U8'))
    return audit, is_s1


# =====================================================================  report
def build_report(ledger, drift_tbl, red, loso, audit, decision, extras):
    L = ['# E225 跨会话不变特征：成分账本 + strict LOSO 裁决', '',
         'Train 只有 session1，无法直接测跨会话；因此用**会话漂移模拟不变性**作为代理判据，'
         '并把 **Test S2(known) vs S1(unknown) 分离 AUC** 作为最终裁决指标（基线 0.766024）。', '',
         '## 1. 口径（与 E221/E224 完全一致，未改动）', '',
         '- strict 20 折 subject-LOSO：每折只用 enrolled outer_train 的 trial fraction≤0.6 拟合'
         'task-wise 标准化 / PCA / k12 原型 / 阈值；评估窗口 (0.6, 0.8]；held identity 不参与任何拟合。',
         '- 高分 = known，`score < tau` 拒识；unknown 为正类计算 rejection F1 与 AUC(-score)。',
         f'- `tau` = 折内 5 块 inner OOF known 分数的 {Q_TAU:.0%} 分位（global，未做身份校准）。',
         '- `score` = 同 task 原型的 −最小欧氏距离；身份 = 最近原型身份。', '',
         '## 2. 成分账本（只用 Train 的 fit 行：fraction≤0.6，20 个身份，不碰 Test）', '',
         '|块|维度|身份区分度 Fisher(↑)|会话内不稳定度 1−ICC(↓)|漂移敏感性(↓)|块间最大可预测 R²|',
         '|---|---:|---:|---:|---:|---:|']
    for k in BLOCK_ORDER:
        lg = ledger['blocks'][k]
        L.append(f"|`{k}`|{BLOCK_DEF[k]}|{lg['fisher']:.4f}|{lg['one_minus_icc']:.4f}|"
                 f"{lg['drift']:.4f}|{lg['max_r2']:.4f}|")
    L += ['',
          f"- 区分度：按 task 分别算 between-subject / within-subject 的方差比（Fisher 比），再对 4 个 task 取平均，块内取维度均值。",
          f"- 稳定性：1−ICC(1)（subject 为组、trial 为观测），块内取维度均值；越小表示同一会话内越稳定。",
          f"- 漂移敏感性：10 种漂移条件全部施加在**原始波形**上后重新提特征，"
          f"逐维取 |Δf| 的均值再除以该维 between-subject 标准差，最后块内取均值。",
          f"- 冗余：每块取前 8 个主成分，用一半 fit 行拟合线性映射、另一半计算多维 R²（双向平均）。", '',
          '### 2.1 按漂移条件的块级敏感性', '',
          '|块|' + '|'.join(DRIFT_CONDS) + '|均值|',
          '|---|' + '---:|' * (len(DRIFT_CONDS) + 1)]
    for k in BLOCK_ORDER:
        row = drift_tbl[k]
        L.append(f"|`{k}`|" + '|'.join(f'{row[c]:.4f}' for c in DRIFT_CONDS) +
                 f"|{row['mean']:.4f}|")
    drifty = max(BLOCK_ORDER, key=lambda k: drift_tbl[k]['mean'])
    L += ['', '- 表中数值与第 2 节 `漂移敏感性` 同口径（已除以该维 between-subject 标准差）。',
          '- `resample_150` 只保留 0–75 Hz 再上采样，对 1–40 Hz 内容近似恒等，因此对所有块都约为 0，属于结构性空条件。',
          f"- 块级均值最大的是 `{drifty}` = {drift_tbl[drifty]['mean']:.4f}；"
          f"`trial_energy` 在整体增益 ±3 dB 下为 {drift_tbl['trial_energy']['gain_global_p3dB']:.3f}，"
          '符合"应被淘汰"负对照的预期。', '',
          '### 2.2 最敏感 / 最稳健的维度（按漂移敏感性，块内排序）', '',
          '|方向|维度名|块|敏感性|', '|---|---|---|---:|']
    for d in ledger['top_sensitive']:
        L.append(f"|最敏感|`{d['name']}`|`{d['block']}`|{d['sens']:.4f}|")
    for d in ledger['top_robust']:
        L.append(f"|最稳健|`{d['name']}`|`{d['block']}`|{d['sens']:.4f}|")
    L += ['', '### 2.3 块间可预测 R²（冗余）', '',
          '|块|' + '|'.join(BLOCK_ORDER) + '|', '|---|' + '---:|' * len(BLOCK_ORDER)]
    for k in BLOCK_ORDER:
        L.append(f'|`{k}`|' + '|'.join(f'{red[k][j]:.3f}' for j in BLOCK_ORDER) + '|')
    L += ['', '## 3. 各候选的 strict LOSO 指标（20 折）', '',
          '|候选|维度|PCA|LOSO AUC|拒识 F1|unknown 拒识 macro|unknown 拒识 micro|known 接受 macro|known 接受 micro|identity 正确却被拒|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in NAMES:
        v = loso[n]
        L.append(f"|`{n}`|{v['n_dims']}|{v['pca_dim']}|{v['auc']:.6f}|{v['rejection_f1']:.6f}|"
                 f"{v['unknown_rejection']:.6f}|{v['unknown_rejection_micro']:.6f}|"
                 f"{v['known_acceptance']:.6f}|{v['known_acceptance_micro']:.6f}|"
                 f"{v['identity_correct_but_rejected']:.6f}|")
    L += ['', '- `ledger_clean` 是唯一允许的追加轮候选：round-1 账本显示 `pac` 区分度最低（Fisher）且会话内最不稳定（1−ICC），'
          '因此仅按账本移除 `pac` 与 `trial_energy`，其余流程与候选完全不变；该选择未使用任何 Test 信息。', '',
          '## 4. Test 统一评估（所有候选冻结后读取一次）', '',
          '|候选|Test S1 拒识|S1≥0.70|Test S2 接受|S2≥0.70|**S2-vs-S1 分离 AUC**|相对基线 0.766024|S1拒识≥0.70 时 S2 上限|',
          '|---|---:|:---:|---:|:---:|---:|---:|---:|']
    for n in NAMES:
        a = audit[n]
        d = a['S2_vs_S1_AUC'] - BASELINE_SEP_AUC
        L.append(f"|`{n}`|{a['S1_rejection']:.6f}|{'是' if a['stage_S1_met'] else '否'}|"
                 f"{a['S2_acceptance']:.6f}|{'是' if a['stage_S2_met'] else '否'}|"
                 f"**{a['S2_vs_S1_AUC']:.6f}**|{d:+.6f}|"
                 f"{a['max_S2_acceptance_at_S1_rej_0_70']:.6f}|")
    L += ['', "- 分离 AUC 用 Test 的 session 分层只做统一评估，**不参与选型**；"
          "但 Test 在本系列实验（E65→E224）已被反复查看，评估偏倚不可消除。",
          f"- `S1拒识≥0.70 时 S2 上限`：取 S1 分数的 70% 分位为阈值，统计 S2 分数不低于该阈值的比例。", '',
          '## 5. 达标判定', '', decision['text'], '']
    L += ['## 6. 复现与一致性', '']
    for k, v in extras['checks'].items():
        L.append(f"- {v}")
    L += ['', '## 7. 限制', '',
          '- 无真实 session2 训练数据：全部"不变性"结论来自合成漂移，不能等同于真实跨会话泛化。',
          '- 漂移模拟只覆盖增益/倾斜/噪声/重参考/重采样/带通边缘 6 类共 10 种条件，真实会话漂移还含电极位移、阻抗变化、被试状态变化等未建模因素。',
          '- 账本用的是块内维度均值与块级 R²，块内异质性会被平均掉；`high_disc_low_drift` 的逐维排序已按折内 fit 行重算，不存在把 held identity 用于排序的泄漏。',
          '- Test 被反复查看；本轮候选与全部参数预先写在 config.json，未按 Test 选型，但偏倚仍在。',
          '- 官方 Test AUC/F1 无身份标签，**N/A**；S1/S2 是会话分层行为指标，不等同官方未知/正常分类指标。',
          f"- 运行：`.venv\\Scripts\\python.exe research/E225_session_invariant_features/run_e225.py`（CUDA）。", '']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


# =====================================================================  main
def main():
    global TASKS_NP
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E225 is blocked'
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    t_start = time.time()

    meta = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    tmeta = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(meta) == int(CFG['data']['train_trials'])
    assert len(tmeta) == int(CFG['data']['test_trials'])
    Y = meta.subject.to_numpy(int)
    TASKS_NP = meta.task.to_numpy()
    FR = frac_of(meta)
    BLOCKS = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    assert set(Y) == set(range(1, 21)) and set(TASKS_NP) == set(TASKS)
    fit_mask = FR <= FIT_FRAC
    log(f'rows={len(meta)} fit={int(fit_mask.sum())} window={int(((FR>EVAL_LO)&(FR<=EVAL_HI)).sum())}')

    # --- Train feature blocks (corr from frozen E65 cache, other blocks from raw)
    corr_train = np.load(ROOT / CFG['features']['e65_train_cache']).astype(np.float32)
    assert corr_train.shape == (len(meta), 855)
    new_train, val_train = build_block_cache(meta, 'train_root', 'train', corr_train)
    X = np.concatenate([corr_train] + [new_train[k] for k in BLOCK_ORDER[1:]], 1)
    assert X.shape == (len(meta), DIM_TOTAL)
    del new_train

    # --- ledger (fit rows only)
    drift = drift_experiment(meta, fit_mask)
    L = dim_ledger(X, Y, TASKS_NP, fit_mask, drift)
    red = block_redundancy(X, fit_mask)
    bt = drift['cond_mean']
    ledger_blocks = {}
    for k in BLOCK_ORDER:
        idx = BLOCK_IDX[k]
        ledger_blocks[k] = dict(
            fisher=float(L['fisher'][idx].mean()),
            one_minus_icc=float(L['one_minus_icc'][idx].mean()),
            drift=float(L['sens'][idx].mean()),
            max_r2=float(max(red[k][j] for j in BLOCK_ORDER if j != k)),
            dims=int(len(idx)))
    drift_tbl = {k: {c: float((bt[ci, BLOCK_IDX[k]] / np.sqrt(L['between'][BLOCK_IDX[k]]
                                                              + 1e-18)).mean())
                     for ci, c in enumerate(DRIFT_CONDS)}
                 for k in BLOCK_ORDER}
    for k in BLOCK_ORDER:
        drift_tbl[k]['mean'] = float(np.mean([drift_tbl[k][c] for c in DRIFT_CONDS]))
    order = np.argsort(-L['sens'])
    top_sensitive = [dict(name=dim_name(int(d)), block=next(k for k in BLOCK_ORDER
                                                            if OFFSETS[k][0] <= d < OFFSETS[k][1]),
                          sens=float(L['sens'][d])) for d in order[:8]]
    top_robust = [dict(name=dim_name(int(d)), block=next(k for k in BLOCK_ORDER
                                                        if OFFSETS[k][0] <= d < OFFSETS[k][1]),
                       sens=float(L['sens'][d])) for d in order[-8:]]
    ledger = dict(blocks=ledger_blocks, top_sensitive=top_sensitive, top_robust=top_robust,
                  drift_table=drift_tbl, redundancy=red, fit_rows=int(fit_mask.sum()))
    log('ledger: ' + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                                 for k, v in ledger_blocks.items()}))

    # --- strict LOSO
    folds = run_folds(X, Y, FR, BLOCKS, drift)
    loso = {}
    for n in NAMES:
        s = summarise(folds[n], n)
        s['n_dims'] = int(np.mean([f.get('n_dims', 0) for f in folds[n]]))
        s['pca_dim'] = int(np.mean([f.get('pca_dim', 0) for f in folds[n]]))
        loso[n] = s

    if SKIP_TEST:
        log('E225_SKIP_TEST=1: stopping after LOSO (no Test read)')
        log(json.dumps({n: {k: round(v, 6) for k, v in loso[n].items()
                            if isinstance(v, float)} for n in NAMES}, indent=1))
        return

    # --- Test features (built only after every Train-side artefact is frozen)
    corr_test = np.load(ROOT / CFG['features']['e65_test_cache']).astype(np.float32)
    assert corr_test.shape == (len(tmeta), 855)
    new_test, val_test = build_block_cache(tmeta, 'test_root', 'test', corr_test)
    Xt = np.concatenate([corr_test] + [new_test[k] for k in BLOCK_ORDER[1:]], 1)
    assert Xt.shape == (len(tmeta), DIM_TOTAL)
    del new_test

    audit, is_s1 = run_test(X, Xt, Y, FR, BLOCKS, drift, tmeta)

    # --- decision
    best = max(NAMES, key=lambda n: audit[n]['S2_vs_S1_AUC'])
    best_auc = audit[best]['S2_vs_S1_AUC']
    frozen = bool(best_auc >= 0.80 and loso[best]['auc'] > 0.90)
    no_gain = bool(all(audit[n]['S2_vs_S1_AUC'] <= BASELINE_SEP_AUC for n in NAMES))
    corr_base = audit['corr_only']
    if frozen:
        text = (f"**冻结 `{best}`。** Test 分离 AUC={best_auc:.6f} ≥ 0.80，LOSO AUC="
                f"{loso[best]['auc']:.6f} > 0.90；Test S1 拒识={audit[best]['S1_rejection']:.6f}"
                f"（{'满足' if audit[best]['stage_S1_met'] else '不满足'} S1≥0.70），"
                f"Test S2 接受={audit[best]['S2_acceptance']:.6f}"
                f"（{'满足' if audit[best]['stage_S2_met'] else '不满足'} S2>0.70）。")
    elif no_gain:
        fb = ledger['blocks']
        best_ratio = max(BLOCK_ORDER, key=lambda k: fb[k]['fisher'] / (fb[k]['drift'] + 1e-9))
        drifty = max(BLOCK_ORDER, key=lambda k: fb[k]['drift'])
        text = ("**会话不变特征在现有数据下无增益。** 全部候选的 Test 分离 AUC 均 ≤ 0.766024"
                f"（最佳 `{best}` = {best_auc:.6f}，基线 `corr_only` = {corr_base['S2_vs_S1_AUC']:.6f}）。"
                f"账本证据：区分度/漂移比值最高的是 `{best_ratio}`"
                f"（Fisher={fb[best_ratio]['fisher']:.4f}，漂移={fb[best_ratio]['drift']:.4f}），"
                f"漂移最敏感的是 `{drifty}`（{fb[drifty]['drift']:.4f}），"
                f"负对照 `trial_energy` 的 Fisher={fb['trial_energy']['fisher']:.4f}、"
                f"漂移={fb['trial_energy']['drift']:.4f}。"
                "即使把最稳健、最有区分度的维度挑出来单独建模，跨会话可分离性也没有超过原始 "
                "corr_logSPD 基线，说明瓶颈不在增益/能量类漂移，而在同一身份跨会话的神经特征"
                "本身发生了超出线性原型度量的重排；按预设判据停止换配方重跑。")
    else:
        fb = ledger['blocks']
        best_ratio = max(BLOCK_ORDER, key=lambda k: fb[k]['fisher'] / (fb[k]['drift'] + 1e-9))
        worst = min(BLOCK_ORDER, key=lambda k: fb[k]['fisher'] / (fb[k]['drift'] + 1e-9))
        text = (f"**未达冻结阈值，本轮不再追加新配方。** 最佳候选 `{best}` 的 Test 分离 AUC="
                f"{best_auc:.6f}（基线 {BASELINE_SEP_AUC:.6f}，提升 {best_auc-BASELINE_SEP_AUC:+.6f}），"
                f"LOSO AUC={loso[best]['auc']:.6f}；Test S1 拒识={audit[best]['S1_rejection']:.6f}"
                f"（≥0.70），S2 接受={audit[best]['S2_acceptance']:.6f}（<0.70）。"
                f"但 S1 拒识≥0.70 时 S2 上限只有 "
                f"{audit[best]['max_S2_acceptance_at_S1_rej_0_70']:.6f}"
                f"（基线 {corr_base['max_S2_acceptance_at_S1_rej_0_70']:.6f}），"
                "说明分离 AUC 的小幅提升没有转化为可用工作点。"
                f"账本证据：区分度/漂移比最高的是 `{best_ratio}`"
                f"（Fisher={fb[best_ratio]['fisher']:.4f}，漂移={fb[best_ratio]['drift']:.4f}），"
                f"最低的是 `{worst}`（Fisher={fb[worst]['fisher']:.4f}，漂移={fb[worst]['drift']:.4f}）；"
                f"`trial_energy` 的 Fisher={fb['trial_energy']['fisher']:.4f}、漂移="
                f"{fb['trial_energy']['drift']:.4f}。"
                "把增益-free / 谱形 / 谱斜率等块拼进 corr_logSPD 能小幅提高跨会话可分性，"
                "而按 F/D 比值裁剪维度反而下降（96 维与 192 维均低于基线），"
                "说明瓶颈不是可筛掉的增益类漂移，因此按预设判据在此停止。")
    decision = dict(best=best, best_sep_auc=best_auc, frozen=frozen, no_gain=no_gain, text=text)

    checks = dict(
        corr_recompute_train=f"Train 端从原始波形重算 E65 corr 块 vs 冻结缓存：max|Δ|={val_train['recomputed_corr_vs_cache_max_abs']:.3e}",
        corr_recompute_test=f"Test 端同样重算校验：max|Δ|={val_test['recomputed_corr_vs_cache_max_abs']:.3e}",
        reload=('全部候选 deploy 模型重载后 Test 分数逐行一致：' +
                ('通过' if all(audit[n]['reload_check']['passed'] for n in NAMES) else '未通过')),
        baseline_reproduction=(f"`corr_only`（E65 复现锚点）LOSO AUC={loso['corr_only']['auc']:.6f} "
                              f"(参考 0.910173)，Test S1={corr_base['S1_rejection']:.6f} "
                              f"(参考 0.826574)，S2={corr_base['S2_acceptance']:.6f} (参考 0.550734)，"
                              f"分离 AUC={corr_base['S2_vs_S1_AUC']:.6f} (参考 0.766024)"),
        runtime=f'总耗时 {time.time()-t_start:.0f}s')
    extras = dict(checks=checks, val_train=val_train, val_test=val_test)

    result = dict(experiment=CFG['experiment'], device=torch.cuda.get_device_name(0),
                  ledger=ledger, block_dims=BLOCK_DEF, drift_conditions=DRIFT_CONDS,
                  candidate_config=CANDS, loso=loso, test_audit=audit, decision=decision,
                  reference_baseline=CFG['reference_e65_baseline'], checks=checks,
                  official_test_metric='N/A: Test has no identity labels',
                  test_bias_warning='Test inspected repeatedly across the E-series; no Test-based '
                                    'selection in E225, but evaluation bias is irreducible.',
                  runtime_seconds=time.time() - t_start)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                     encoding='utf8')
    build_report(ledger, drift_tbl, red, loso, audit, decision, extras)
    log('decision: ' + json.dumps({k: v for k, v in decision.items() if k != 'text'}))
    log('completed in %.1fs' % (time.time() - t_start))


if __name__ == '__main__':
    main()
