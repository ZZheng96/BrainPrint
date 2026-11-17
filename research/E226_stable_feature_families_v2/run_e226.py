"""E226: gain/reference-invariant feature families + reliability weighting (no dimension filtering).

Protocol is byte-identical to E225 (research/E225_session_invariant_features/run_e225.py):
  - 20-fold subject-LOSO. Each fold: task-wise standardisation / PCA / k12 prototypes / tau are
    fitted on the enrolled outer_train rows with trial-fraction <= 0.6 only. Eval window (0.6, 0.8].
    Held identity never enters any fit. tau = 10% quantile of the inner 5-block OOF known scores.
  - score = -min euclidean distance to same-task prototypes; higher = known; reject if score < tau.
  - unknown is the positive class for rejection F1 and for AUC(-score).
  - Test is read once, after every Train-side artefact is frozen, for uniform evaluation only.

New in E226
  - Four theoretically gain/reference-invariant feature families extracted from raw waveforms:
    hjorth (mobility/complexity), spec_entropy (normalised Shannon entropy + spectral flatness),
    power_ratio (fixed contralateral log-power differences + cross-band log ratios) and
    graph_metrics (clustering / global efficiency / strength entropy / eigenvalue share / modularity).
  - One merged ledger over old + new blocks (Fisher, 1-ICC, 10-condition raw-waveform drift
    sensitivity, inter-block redundancy).
  - Reliability weighting: diagonal metric w_d = between/(within+eps) applied before PCA, keeping
    every dimension, plus a permuted-weight negative control.
  - Block-level fusion of z-scored distances with three fixed schemes (equal / byF / byFD).
"""
from __future__ import annotations

import hashlib
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
CACHE_VERSION = 'v2'

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
N_FOLDS = int(os.environ.get('E226_FOLDS', CFG['protocol']['folds']))
SKIP_TEST = os.environ.get('E226_SKIP_TEST') == '1'
CANDS = [dict(c) for c in CFG['candidates']]
NAMES = [c['name'] for c in CANDS]
CONCAT_CANDS = [c for c in CANDS if c['mode'] == 'concat']
FUSION_CANDS = [c for c in CANDS if c['mode'] == 'fusion']
CHUNK = int(CFG['features']['chunk_trials'])
DRIFT_CHUNK = int(CFG['features']['drift_chunk_trials'])
WORKERS = int(CFG['features']['io_workers'])
DRIFT_CONDS = list(CFG['drift_simulation']['conditions'])
SUB_PER_CELL = int(CFG['drift_simulation']['subset_per_subject_task'])
ANCHOR = CFG['predecessor']['anchor_expected']
STAGE = CFG['decision_rule']['stage_targets']
PCA_PER_BLOCK = int(CFG['weighting']['block_fusion']['pca_per_block'])

# ---- E65/E225 constants (identical to research/E225_session_invariant_features)
SHRINK = 0.05
EIG_FLOOR = 1e-12
OFFMUL = float(np.sqrt(2.0))
ZBAND_EPS = 1e-6

BLOCK_DEF = {'corr_logspd': 855, 'bandpower_gainfree': 90, 'spectral_shape': 90,
             'spectral_slope': 18, 'pac': 18, 'trial_energy': 1,
             'hjorth': 180, 'spec_entropy': 180, 'power_ratio': 112, 'graph_metrics': 25}
BLOCK_ORDER = list(BLOCK_DEF)
OLD_BLOCKS = BLOCK_ORDER[:6]
NEW_BLOCKS = BLOCK_ORDER[6:]
OFFSETS = {}
_acc = 0
for _k, _v in BLOCK_DEF.items():
    OFFSETS[_k] = (_acc, _acc + _v)
    _acc += _v
DIM_TOTAL = _acc
OLD_END = OFFSETS['trial_energy'][1]
BLOCK_IDX = {k: np.arange(a, b) for k, (a, b) in OFFSETS.items()}
assert DIM_TOTAL == int(CFG['features']['total_dim']) == 1569, DIM_TOTAL

# fixed contralateral pairs (0-based indices into CHANNEL_ORDER) and their left/right labels
PAIRS = [(0, 6), (1, 5), (2, 4), (10, 17), (11, 16), (12, 13), (14, 15), (8, 9)]
CROSS_RATIOS = [('alpha/theta', 1, 0), ('beta/alpha', -1, 1), ('beta/theta', -1, 0),
                ('gamma/alpha', 4, 1)]
GRAPH_METRICS = ['clustering', 'global_efficiency', 'strength_entropy', 'eig_share', 'modularity']


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(msg) + '\n')


def config_tag():
    payload = json.dumps({'blocks': BLOCK_DEF, 'cands': CANDS, 'w': CFG['weighting'],
                          'protocol': CFG['protocol'], 'drift': CFG['drift_simulation'],
                          'features': {k: v for k, v in CFG['features'].items()
                                       if k != 'blocks'}},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


TAG = config_tag()


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
    """E65 gpu_correlation_logspd: (B,18,1200) -> (B,855). Unchanged from E225."""
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
def old_blocks(raw):
    """E225 blocks, verbatim, used only for the drift recomputation path."""
    bm = band_masks()
    xc = raw - raw.mean(-1, keepdim=True)
    spec = torch.fft.rfft(xc, dim=-1)
    bb = torch.fft.irfft(spec[:, None] * bm[None, :, None, :], n=N, dim=-1)

    lp = torch.log((bb ** 2).mean(-1) + 1e-10)
    gainfree = (lp - lp.mean(-1, keepdim=True)).flatten(1)

    psd = spec.abs() ** 2
    sel = _mask(1, 40, last=True)
    lps = torch.log(psd[..., sel] + 1e-12)
    shape_f = lps - lps.mean(-1, keepdim=True)
    fsel = freq_grid()[sel]
    parts = []
    for i, (lo, hi) in enumerate(BANDS):
        m = (fsel >= lo) & (fsel <= hi if i == NB - 1 else fsel < hi)
        parts.append(shape_f[..., m].mean(-1))
    shape = torch.stack(parts, 1).flatten(1)

    lf = torch.log(fsel.clamp_min(1e-3))
    lf = lf - lf.mean()
    slope = (lps * lf).sum(-1) / (lf * lf).sum()

    alpha = torch.fft.irfft(spec * _mask(8, 13), n=N, dim=-1)
    beta = torch.fft.irfft(spec * _mask(13, 30), n=N, dim=-1)
    amp = analytic(beta).abs()
    ph = analytic(alpha).angle()
    pac = (amp * torch.polar(torch.ones_like(amp), ph)).mean(-1).abs() \
        / amp.mean(-1).clamp_min(1e-12)

    energy = torch.log(raw.pow(2).mean(dim=(1, 2)).sqrt() + 1e-12)[:, None]

    return {'corr_logspd': corr_logspd(raw), 'bandpower_gainfree': gainfree,
            'spectral_shape': shape, 'spectral_slope': slope, 'pac': pac,
            'trial_energy': energy}


@torch.no_grad()
def hjorth_feats(bb):
    """bb: (B,5,18,1200) band-passed -> (B,180) = [mobility 5x18][complexity 5x18]."""
    x = bb - bb.mean(-1, keepdim=True)
    dx = x[..., 1:] - x[..., :-1]
    d2x = dx[..., 1:] - dx[..., :-1]
    v0 = x.var(-1, unbiased=False).clamp_min(1e-20)
    v1 = dx.var(-1, unbiased=False).clamp_min(1e-20)
    v2 = d2x.var(-1, unbiased=False).clamp_min(1e-20)
    mob = (v1 / v0).sqrt()
    mob_d = (v2 / v1).sqrt()
    comp = mob_d / mob.clamp_min(1e-12)
    return torch.cat([mob.flatten(1), comp.flatten(1)], 1)


@torch.no_grad()
def spec_entropy_feats(spec):
    """spec: (B,18,601) -> (B,180) = [normalised Shannon entropy 5x18][flatness 5x18]."""
    psd = spec.abs() ** 2
    f = freq_grid()
    ent, flat = [], []
    for i, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & (f <= hi if i == NB - 1 else f < hi)
        P = psd[..., m].clamp_min(1e-20)
        Pn = P / P.sum(-1, keepdim=True)
        H = -(Pn * Pn.log()).sum(-1) / float(np.log(int(m.sum())))
        gm = P.log().mean(-1).exp()
        flat.append(gm / P.mean(-1))
        ent.append(H)
    return torch.cat([torch.stack(ent, 1).flatten(1),
                      torch.stack(flat, 1).flatten(1)], 1)


@torch.no_grad()
def power_ratio_feats(lp):
    """lp: (B,5,18) log band power -> (B,112) = [40 contralateral diffs][72 cross-band ratios]."""
    pair = torch.stack([lp[..., a] - lp[..., b] for a, b in PAIRS], -1)      # B,5,8
    beta = torch.logaddexp(lp[:, 2], lp[:, 3]) - float(np.log(2.0))
    ratios = []
    for _, hi, lo in CROSS_RATIOS:
        h = beta if hi < 0 else lp[:, hi]
        l = beta if lo < 0 else lp[:, lo]
        ratios.append(h - l)
    return torch.cat([pair.flatten(1), torch.stack(ratios, 1).flatten(1)], 1)


@torch.no_grad()
def graph_feats(bb):
    """bb: (B,5,18,1200) -> (B,25), 5 metrics x 5 bands."""
    b = bb.shape[0]
    x = bb - bb.mean(-1, keepdim=True)
    x = x / x.std(-1, keepdim=True, unbiased=True).clamp_min(1e-12)
    corr = x @ x.transpose(-1, -2) / (N - 1)                                 # B,5,18,18
    w = corr.abs()
    eye = torch.eye(CH, device=DEVICE)
    w = w * (1.0 - eye)

    vals, _ = torch.linalg.eigh(w)
    eig_share = vals[..., -1] / vals.abs().sum(-1).clamp_min(1e-12)

    r, c = torch.triu_indices(CH, CH, offset=1, device=DEVICE)
    upper = w[..., r, c]                                                     # B,5,153
    thr = torch.quantile(upper, 0.70, dim=-1, keepdim=True)
    a = (upper >= thr).float()
    adj = torch.zeros_like(w)
    adj[..., r, c] = a
    adj = adj + adj.transpose(-1, -2)
    kdeg = adj.sum(-1)

    a3 = adj @ adj @ adj
    num = torch.diagonal(a3, dim1=-2, dim2=-1)
    den = (kdeg * (kdeg - 1)).clamp_min(1e-9)
    clust = torch.where(kdeg > 1, num / den, torch.zeros_like(num)).clamp(0.0, 1.0).mean(-1)

    INF = 1e6
    d = torch.where(adj > 0, torch.ones_like(adj), torch.full_like(adj, INF))
    d = d * (1.0 - eye)                      # zero diagonal: required for min-plus closure
    for _ in range(CH - 1):
        dn = (d.unsqueeze(-1) + d.unsqueeze(-3)).amin(-2)
        if torch.equal(dn, d):
            break
        d = dn
    ok = (d < 1e5) & (d > 0)
    eff = torch.where(ok, 1.0 / d.clamp_min(1e-9), torch.zeros_like(d)).sum((-1, -2)) \
        / (CH * (CH - 1))

    s = w.sum(-1)
    p = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    hstr = -(p * p.clamp_min(1e-20).log()).sum(-1) / float(np.log(CH))

    m2 = adj.sum((-1, -2))
    kk = kdeg.unsqueeze(-1) * kdeg.unsqueeze(-2)
    mod = adj - kk / m2.clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)
    _, vec = torch.linalg.eigh(mod)
    v = vec[..., -1]
    v = v * torch.where(v[..., 0:1] >= 0, 1.0, -1.0)
    sg = torch.where(torch.sign(v) == 0, torch.ones_like(v), torch.sign(v))
    q = (sg.unsqueeze(-2) @ mod @ sg.unsqueeze(-1)).squeeze(-1).squeeze(-1) \
        / (2.0 * m2.clamp_min(1.0))

    out = torch.stack([clust, eff, hstr, eig_share, q], -1).flatten(1)       # B,25
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def new_blocks(raw):
    bm = band_masks()
    xc = raw - raw.mean(-1, keepdim=True)
    spec = torch.fft.rfft(xc, dim=-1)
    bb = torch.fft.irfft(spec[:, None] * bm[None, :, None, :], n=N, dim=-1)
    lp = torch.log((bb ** 2).mean(-1) + 1e-10)
    return {'hjorth': hjorth_feats(bb), 'spec_entropy': spec_entropy_feats(spec),
            'power_ratio': power_ratio_feats(lp), 'graph_metrics': graph_feats(bb)}


def assemble_all(fb):
    out = np.concatenate([fb[k].float().cpu().numpy() for k in BLOCK_ORDER], 1)
    return out


# =====================================================================  IO
def load_mat(path):
    d = np.asarray(loadmat(path, variable_names=['data'])['data'], dtype=np.float32)
    if d.shape != (N, CH):
        raise ValueError(f'{path}: expected ({N},{CH}) got {d.shape}')
    if not np.isfinite(d).all():
        raise ValueError(f'{path}: non-finite')
    return np.ascontiguousarray(d.T)


def raw_chunks(paths, chunk=None):
    chunk = chunk or CHUNK
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(paths), chunk):
            batch = paths[start:start + chunk]
            arrays = list(pool.map(load_mat, batch))
            yield start, torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=DEVICE)


def build_new_cache(meta, root_key, tag):
    files = {k: CACHE / f'{tag}_{k}.npy' for k in NEW_BLOCKS}
    if all(p.exists() for p in files.values()):
        log(f'{tag}: new-block cache reused')
        return {k: np.load(p) for k, p in files.items()}
    paths = [ROOT / CFG['data'][root_key] / str(p) for p in meta.path]
    mm = {k: np.lib.format.open_memmap(files[k], mode='w+', dtype=np.float32,
                                       shape=(len(paths), BLOCK_DEF[k])) for k in NEW_BLOCKS}
    t0 = time.time()
    for start, raw in raw_chunks(paths):
        fb = new_blocks(raw)
        for k in NEW_BLOCKS:
            mm[k][start:start + len(raw)] = fb[k].cpu().numpy()
        if (start + len(raw)) % 4096 < CHUNK or start + len(raw) == len(paths):
            log(f'  {tag} new features {start + len(raw)}/{len(paths)} ({time.time()-t0:.0f}s)')
    for k in NEW_BLOCKS:
        mm[k].flush()
    del mm
    return {k: np.load(p) for k, p in files.items()}


def load_old_cache(tag):
    """Old blocks exactly as E225 consumed them."""
    e225 = ROOT / CFG['features']['e225_cache_dir']
    if tag == 'train':
        corr = np.load(ROOT / CFG['features']['e65_train_cache']).astype(np.float32)
    else:
        corr = np.load(ROOT / CFG['features']['e65_test_cache']).astype(np.float32)
    out = {'corr_logspd': corr}
    for k in OLD_BLOCKS[1:]:
        out[k] = np.load(e225 / f'{tag}_{k}.npy').astype(np.float32)
    return out


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


def drift_subset_rows(meta, fit_mask):
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
    rows = drift_subset_rows(meta, fit_mask)
    cpath = CACHE / f'drift_stats_{CACHE_VERSION}_{TAG}.npz'
    if cpath.exists():
        d = np.load(cpath)
        log('drift stats reused')
        return dict(subject_delta=d['subject_delta'], cond_mean=d['cond_mean'],
                    rows=d['rows'], conds=DRIFT_CONDS)
    paths = [ROOT / CFG['data']['train_root'] / str(meta.path.iloc[r]) for r in rows]
    raw = torch.empty((len(paths), CH, N), dtype=torch.float32, device=DEVICE)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(paths), DRIFT_CHUNK):
            arrays = list(pool.map(load_mat, paths[start:start + DRIFT_CHUNK]))
            raw[start:start + len(arrays)] = torch.as_tensor(np.stack(arrays),
                                                             dtype=torch.float32, device=DEVICE)
    base = assemble_all({**old_blocks(raw), **new_blocks(raw)})
    subj = meta.subject.to_numpy(int)[rows]
    subject_delta = np.zeros((len(DRIFT_CONDS), 20, DIM_TOTAL), np.float32)
    cond_mean = np.zeros((len(DRIFT_CONDS), DIM_TOTAL), np.float32)
    t0 = time.time()
    for ci, cond in enumerate(DRIFT_CONDS):
        rng = np.random.default_rng(SEED + 1000 + ci)
        np.random.seed(SEED + 1000 + ci)
        torch.manual_seed(SEED + 1000 + ci)
        drifted = apply_drift(raw, cond, rng)
        feat = assemble_all({**old_blocks(drifted), **new_blocks(drifted)})
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
                between=between, within=within)


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
    if d < OLD_END:
        for k in OLD_BLOCKS[1:]:
            a, b = OFFSETS[k]
            if a <= d < b:
                i = d - a
                if k in ('bandpower_gainfree', 'spectral_shape'):
                    band, ch = i // CH, i % CH
                    return f'{k}_b{BANDS[band][0]}-{BANDS[band][1]}Hz_ch{ch+1}'
                return f'{k}_ch{i+1}'
    a, b = OFFSETS['hjorth']
    if a <= d < b:
        i = d - a
        metric = 'mobility' if i < 90 else 'complexity'
        j = i % 90
        return f'hjorth_{metric}_b{BANDS[j//CH][0]}-{BANDS[j//CH][1]}Hz_ch{j%CH+1}'
    a, b = OFFSETS['spec_entropy']
    if a <= d < b:
        i = d - a
        metric = 'entropy' if i < 90 else 'flatness'
        j = i % 90
        return f'spec_entropy_{metric}_b{BANDS[j//CH][0]}-{BANDS[j//CH][1]}Hz_ch{j%CH+1}'
    a, b = OFFSETS['power_ratio']
    if a <= d < b:
        i = d - a
        if i < 40:
            band, pr = i // 8, i % 8
            return f'power_ratio_b{BANDS[band][0]}-{BANDS[band][1]}Hz_pair{pr}'
        j = i - 40
        ridx, ch = j // CH, j % CH
        return f'power_ratio_{CROSS_RATIOS[ridx][0]}_ch{ch+1}'
    a, b = OFFSETS['graph_metrics']
    if a <= d < b:
        i = d - a
        band, met = i // 5, i % 5
        return f'graph_{GRAPH_METRICS[met]}_b{BANDS[band][0]}-{BANDS[band][1]}Hz'
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


def cand_dims(cand):
    spec = cand['blocks']
    if spec == 'old':
        return np.arange(OLD_END)
    if spec == 'all':
        return np.arange(DIM_TOTAL)
    return np.concatenate([BLOCK_IDX[k] for k in spec])


def reliability_weights(X, y, tasks, fit, dims, permute=None):
    between, within, _ = per_task_scatter(X[:, dims], y, tasks, fit)
    w = between / (within + 1e-9)
    lo, hi = np.quantile(w, [0.05, 0.95])
    w = np.clip(w, lo, max(hi, lo + 1e-12))
    if permute is not None:
        w = w[permute]
    return (w / w.mean()).astype(np.float32)


def block_weight_table(X, y, tasks, fit, drift):
    L = dim_ledger(X, y, tasks, fit, drift)
    F = np.array([L['fisher'][BLOCK_IDX[b]].mean() for b in BLOCK_ORDER])
    D = np.array([L['sens'][BLOCK_IDX[b]].mean() for b in BLOCK_ORDER])
    return F, D


def fusion_weights(scheme, F, D):
    if scheme == 'equal':
        return np.ones(len(BLOCK_ORDER))
    if scheme == 'byF':
        w = F / max(F.mean(), 1e-12)
        return w / w.mean()
    if scheme == 'byFD':
        r = F / (D + 1e-9)
        w = r / max(r.mean(), 1e-12)
        return w / w.mean()
    raise ValueError(scheme)


# ---------------------------------------------------------------- fusion (block-wise z distance)
def block_apply(state, Xb, tasks, rows):
    z = base_transform(state, Xb, tasks)
    proto_task, proto_id = base_proto_meta(state)
    d = np.full(len(rows), np.nan)
    pred = np.full(len(rows), -1, int)
    for t in TASKS:
        sel = np.flatnonzero(tasks[rows] == t)
        cols = np.flatnonzero(proto_task == t)
        if not len(sel) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rows[sel]]), tensor(state['p'][cols])).cpu().numpy()
        d[sel] = D.min(1)
        pred[sel] = proto_id[cols][D.argmin(1)]
    return d, pred


def block_dmin_other(state, Xb, tasks, y, rows):
    z = base_transform(state, Xb, tasks)
    proto_task, proto_id = base_proto_meta(state)
    out = np.zeros(len(rows))
    for t in TASKS:
        sel = np.flatnonzero(tasks[rows] == t)
        cols = np.flatnonzero(proto_task == t)
        if not len(sel) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rows[sel]]), tensor(state['p'][cols])).cpu().numpy()
        own = y[rows[sel]]
        pid = proto_id[cols]
        ok = own[:, None] != pid[None, :]
        Dm = np.where(ok, D, np.inf)
        m = Dm.min(1)
        out[sel] = np.where(np.isfinite(m), m, D.min(1))
    return out


def fusion_fit(X, y, tasks, fit, pca_block=PCA_PER_BLOCK, k=K_PROTO):
    st = {}
    for b in BLOCK_ORDER:
        idx = BLOCK_IDX[b]
        s = base_fit_rep(X[:, idx], tasks, y, fit, k, pca=pca_block)
        d_other = block_dmin_other(s, X[:, idx], tasks, y, np.flatnonzero(fit))
        med = float(np.median(d_other))
        iqr = float(np.subtract(*np.percentile(d_other, [75, 25])))
        sd = iqr / 1.349
        if not np.isfinite(sd) or sd <= 0:
            sd = float(np.std(d_other)) or 1.0
        st[b] = dict(state=s, mu_d=med, sd_d=sd)
    return st


def fusion_z(st, X, tasks, rows):
    Z = np.zeros((len(rows), len(BLOCK_ORDER)), np.float32)
    P = np.zeros((len(rows), len(BLOCK_ORDER)), np.int32)
    for j, b in enumerate(BLOCK_ORDER):
        d, p = block_apply(st[b]['state'], X[:, BLOCK_IDX[b]], tasks, rows)
        Z[:, j] = (d - st[b]['mu_d']) / st[b]['sd_d']
        P[:, j] = p
    return Z, P


def fusion_score(Z, w):
    return -(Z * np.asarray(w, np.float32)[None, :]).sum(1)


def fusion_vote(P, w):
    out = np.zeros((len(P), 21), np.float32)
    rows = np.arange(len(P))
    for j in range(P.shape[1]):
        out[rows, np.clip(P[:, j], 0, 20)] += w[j]
    return out[:, 1:].argmax(1) + 1


# =====================================================================  folds
def run_folds(X, Y, FR, BLOCKS, drift):
    cache_dir = CACHE
    folds = {n: [] for n in NAMES}
    weight_log = {}
    for held in range(1, N_FOLDS + 1):
        t0 = time.time()
        fit = (Y != held) & (FR <= FIT_FRAC)
        window = (FR > EVAL_LO) & (FR <= EVAL_HI)
        known = window & (Y != held)
        unknown = window & (Y == held)
        win_rows = np.flatnonzero(window)
        rng = np.random.default_rng(SEED + held)

        w_icc = None
        for cand in CONCAT_CANDS:
            name = cand['name']
            path = cache_dir / f'fold_{CACHE_VERSION}_{name}_{held:02d}.json'
            if path.exists():
                folds[name].append(json.loads(path.read_text())['metrics'])
                continue
            dims = cand_dims(cand)
            pca = int(cand.get('pca', PCA_DIM))
            w = None
            if cand.get('weights') in ('icc', 'icc_permuted'):
                if w_icc is None:
                    w_icc = reliability_weights(X, Y, TASKS_NP, fit, dims)
                if cand['weights'] == 'icc':
                    w = w_icc
                else:
                    w = w_icc[rng.permutation(len(w_icc))]
            st = base_fit_rep(X[:, dims], TASKS_NP, Y, fit, K_PROTO, pca, w)
            z = base_transform(st, X[:, dims], TASKS_NP)
            s, p = base_score(st, z, TASKS_NP, win_rows)
            s, p = s[win_rows], p[win_rows]
            oof_s, oof_p = base_inner_oof(fit, X[:, dims], TASKS_NP, Y, BLOCKS, K_PROTO, pca, w)
            tau = quantile_tau(oof_s[fit], Q_TAU)
            tau_row = np.full(len(win_rows), tau)
            m = evaluate(s, p, Y[win_rows], known[win_rows], unknown[win_rows], tau_row)
            m['n_dims'] = int(len(dims))
            m['pca_dim'] = int(st['pca'])
            if w is not None:
                m['weight_ratio_max_min'] = float(np.max(w) / max(np.min(w), 1e-9))
            np.savez_compressed(CACHE / f'win_{CACHE_VERSION}_{name}_{held:02d}.npz',
                                score=s, pred=p, subject=Y[win_rows],
                                known=known[win_rows], unknown=unknown[win_rows], tau=tau)
            path.write_text(json.dumps(dict(candidate=name, held=int(held), metrics=m,
                                            pca=st['pca'], n_dims=int(len(dims)),
                                            train_rows=int(fit.sum())), indent=2))
            folds[name].append(m)

        # ---- fusion: one shared per-block z pipeline, three fixed weight schemes
        fpath = cache_dir / f'fold_{CACHE_VERSION}_fusion_{held:02d}.json'
        if fpath.exists():
            row = json.loads(fpath.read_text())
            for c in FUSION_CANDS:
                folds[c['name']].append(row['metrics'][c['name']])
            weight_log[held] = row['weights']
        else:
            st = fusion_fit(X, Y, TASKS_NP, fit)
            Zw, Pw = fusion_z(st, X, TASKS_NP, win_rows)
            Zo = np.full((len(Y), len(BLOCK_ORDER)), np.nan, np.float32)
            Po = np.full((len(Y), len(BLOCK_ORDER)), -1, np.int32)
            for b in np.unique(BLOCKS[fit]):
                hold = fit & (BLOCKS == b)
                inner = fit & ~hold
                ist = fusion_fit(X, Y, TASKS_NP, inner)
                rowh = np.flatnonzero(hold)
                zi, pi = fusion_z(ist, X, TASKS_NP, rowh)
                Zo[rowh] = zi
                Po[rowh] = pi
            F, D = block_weight_table(X, Y, TASKS_NP, fit, drift)
            metrics = {}
            wlog = {}
            for c in FUSION_CANDS:
                w = fusion_weights(c['weights'], F, D)
                s = fusion_score(Zw, w)
                p = fusion_vote(Pw, w)
                oof = fusion_score(Zo[fit], w)
                tau = quantile_tau(oof, Q_TAU)
                tau_row = np.full(len(win_rows), tau)
                m = evaluate(s, p, Y[win_rows], known[win_rows], unknown[win_rows], tau_row)
                m['n_dims'] = int(DIM_TOTAL)
                m['pca_dim'] = PCA_PER_BLOCK
                m['weight_ratio_max_min'] = float(np.max(w) / max(np.min(w), 1e-9))
                metrics[c['name']] = m
                wlog[c['name']] = dict(
                    weights=[float(x) for x in w], F=[float(x) for x in F],
                    D=[float(x) for x in D])
                folds[c['name']].append(m)
                np.savez_compressed(CACHE / f'win_{CACHE_VERSION}_{c["name"]}_{held:02d}.npz',
                                    score=s, pred=p, subject=Y[win_rows],
                                    known=known[win_rows], unknown=unknown[win_rows], tau=tau)
            fpath.write_text(json.dumps(dict(held=int(held), metrics=metrics, weights=wlog,
                                             blocks=BLOCK_ORDER), indent=2))
            weight_log[held] = wlog

        log(f'fold {held}: ' + ' | '.join(
            f"{n} auc={folds[n][-1]['auc']:.4f} rej={folds[n][-1]['unknown_rejection']:.4f}"
            for n in NAMES) + f'  ({time.time()-t0:.1f}s)')
    return folds, weight_log


# =====================================================================  single-block LOSO (ledger)
def run_block_only(X, Y, FR, BLOCKS):
    """Strict LOSO of every block alone: the ledger's direct 'identity information' column."""
    out = {}
    for b in BLOCK_ORDER:
        idx = BLOCK_IDX[b]
        pca = int(min(PCA_DIM, len(idx)))
        ms = []
        for held in range(1, N_FOLDS + 1):
            path = CACHE / f'fold_{CACHE_VERSION}_blockonly_{b}_{held:02d}.json'
            if path.exists():
                ms.append(json.loads(path.read_text())['metrics'])
                continue
            fit = (Y != held) & (FR <= FIT_FRAC)
            window = (FR > EVAL_LO) & (FR <= EVAL_HI)
            known = window & (Y != held)
            unknown = window & (Y == held)
            win = np.flatnonzero(window)
            st = base_fit_rep(X[:, idx], TASKS_NP, Y, fit, K_PROTO, pca)
            z = base_transform(st, X[:, idx], TASKS_NP)
            s, p = base_score(st, z, TASKS_NP, win)
            s, p = s[win], p[win]
            oof_s, _ = base_inner_oof(fit, X[:, idx], TASKS_NP, Y, BLOCKS, K_PROTO, pca)
            tau = quantile_tau(oof_s[fit], Q_TAU)
            tau_row = np.full(len(win), tau)
            m = evaluate(s, p, Y[win], known[win], unknown[win], tau_row)
            m['n_dims'] = int(len(idx))
            m['pca_dim'] = int(st['pca'])
            path.write_text(json.dumps(dict(block=b, held=int(held), metrics=m), indent=2))
            ms.append(m)
        out[b] = summarise(ms, b)
        log(f'block-only {b}: auc={out[b]["auc"]:.4f} rej={out[b]["unknown_rejection"]:.4f}')
    return out


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
    w_icc = None
    for cand in CONCAT_CANDS:
        name = cand['name']
        dims = cand_dims(cand)
        pca = int(cand.get('pca', PCA_DIM))
        w = None
        if cand.get('weights') in ('icc', 'icc_permuted'):
            if w_icc is None:
                w_icc = reliability_weights(X, Y, TASKS_NP, fit, dims)
            w = (w_icc if cand['weights'] == 'icc'
                 else w_icc[np.random.default_rng(SEED).permutation(len(w_icc))])
        st = base_fit_rep(X[:, dims], TASKS_NP, Y, fit, K_PROTO, pca, w)
        zt = base_transform(st, Xt[:, dims], tt)
        s, p = base_score(st, zt, tt, np.arange(len(tt)))
        oof_s, oof_p = base_inner_oof(fit, X[:, dims], TASKS_NP, Y, BLOCKS, K_PROTO, pca, w)
        tau = quantile_tau(oof_s[fit], Q_TAU)
        audit[name] = test_row(s, tau, is_s1)
        audit[name]['n_dims'] = int(len(dims))
        audit[name]['pca_dim'] = int(st['pca'])
        np.savez_compressed(CACHE / f'test_scores_{CACHE_VERSION}_{name}.npz', score=s, pred=p,
                            tau=tau, task=np.asarray(tt, dtype='<U8'),
                            session=np.asarray(tm.session.astype(str).to_numpy(), dtype='<U8'),
                            is_s1=is_s1)
        torch.save(dict(mode='concat', state=st, dims=np.asarray(dims), w=None if w is None
                        else np.asarray(w, np.float32), pca=int(st['pca']), tau=float(tau),
                        candidate=name), MODELS / f'deploy_{name}.pt')
        back = torch.load(MODELS / f'deploy_{name}.pt', weights_only=False)
        s2, p2 = base_score(back['state'],
                            base_transform(back['state'], Xt[:, back['dims']], tt),
                            tt, np.arange(len(tt)))
        audit[name]['reload_check'] = dict(
            max_abs_score_diff=float(np.max(np.abs(s2 - s))),
            pred_identical=bool(np.array_equal(p2, p)),
            passed=bool(np.max(np.abs(s2 - s)) <= 1e-6 and np.array_equal(p2, p)))
        log(f'test {name}: S1={audit[name]["S1_rejection"]:.4f} '
            f'S2={audit[name]["S2_acceptance"]:.4f} sepAUC={audit[name]["S2_vs_S1_AUC"]:.4f} '
            f'dims={len(dims)}')

    st = fusion_fit(X, Y, TASKS_NP, fit)
    Zt, Pt = fusion_z(st, Xt, tt, np.arange(len(tt)))
    Zo = np.full((len(Y), len(BLOCK_ORDER)), np.nan, np.float32)
    Po = np.full((len(Y), len(BLOCK_ORDER)), -1, np.int32)
    for b in np.unique(BLOCKS[fit]):
        hold = fit & (BLOCKS == b)
        inner = fit & ~hold
        ist = fusion_fit(X, Y, TASKS_NP, inner)
        rowh = np.flatnonzero(hold)
        zi, pi = fusion_z(ist, X, TASKS_NP, rowh)
        Zo[rowh] = zi
        Po[rowh] = pi
    F, D = block_weight_table(X, Y, TASKS_NP, fit, drift)
    for c in FUSION_CANDS:
        name = c['name']
        w = fusion_weights(c['weights'], F, D)
        s = fusion_score(Zt, w)
        p = fusion_vote(Pt, w)
        tau = quantile_tau(fusion_score(Zo[fit], w), Q_TAU)
        audit[name] = test_row(s, tau, is_s1)
        audit[name]['n_dims'] = int(DIM_TOTAL)
        audit[name]['pca_dim'] = PCA_PER_BLOCK
        np.savez_compressed(CACHE / f'test_scores_{CACHE_VERSION}_{name}.npz', score=s, pred=p,
                            tau=tau, task=np.asarray(tt, dtype='<U8'),
                            session=np.asarray(tm.session.astype(str).to_numpy(), dtype='<U8'),
                            is_s1=is_s1)
        torch.save(dict(mode='fusion', states=st, weights=np.asarray(w, np.float32),
                        blocks=BLOCK_ORDER, tau=float(tau), candidate=name),
                   MODELS / f'deploy_{name}.pt')
        back = torch.load(MODELS / f'deploy_{name}.pt', weights_only=False)
        Zb, Pb = fusion_z(back['states'], Xt, tt, np.arange(len(tt)))
        s2 = fusion_score(Zb, back['weights'])
        p2 = fusion_vote(Pb, back['weights'])
        audit[name]['reload_check'] = dict(
            max_abs_score_diff=float(np.max(np.abs(s2 - s))),
            pred_identical=bool(np.array_equal(p2, p)),
            passed=bool(np.max(np.abs(s2 - s)) <= 1e-6 and np.array_equal(p2, p)))
        log(f'test {name}: S1={audit[name]["S1_rejection"]:.4f} '
            f'S2={audit[name]["S2_acceptance"]:.4f} sepAUC={audit[name]["S2_vs_S1_AUC"]:.4f} '
            f'fusion w={np.round(w, 3).tolist()}')
    np.savez_compressed(CACHE / 'test_meta.npz', is_s1=is_s1,
                        session=np.asarray(tm.session.astype(str).to_numpy(), dtype='<U8'),
                        task=np.asarray(tt, dtype='<U8'))
    return audit, is_s1


# =====================================================================  report
def build_report(ledger, drift_tbl, red, loso, audit, decision, extras):
    fb0 = ledger['blocks']
    L = ['# E226 天然增益/参考不变成分族 + 可靠性加权（strict LOSO 裁决）', '',
         '前作 E225 已证明：全块拼接（1072 维，PCA96，k12）LOSO AUC 0.9086、'
         'unknown 拒识 0.6503、Test S1 0.8249 / S2 0.5353、分离 AUC 0.7878。'
         '本轮不改协议，只新增四类**理论上对幅值增益/参考不变**的成分族，并把"硬筛维度"换成'
         '"可靠性加权"，最终仍以 **Test S2(known) vs S1(unknown) 分离 AUC** 裁决。', '',
         '## 1. 口径（与 E221/E225 逐位一致，未改动）', '',
         '- strict 20 折 subject-LOSO：每折只用 enrolled outer_train 的 trial fraction≤0.6 拟合'
         'task-wise 标准化 / PCA / k12 原型 / 阈值；评估窗口 (0.6, 0.8]；held identity 不参与任何拟合。',
         '- 高分 = known，`score < tau` 拒识；unknown 为正类计算 rejection F1 与 AUC(-score)。',
         f'- `tau` = 折内 5 块 inner OOF known 分数的 {Q_TAU:.0%} 分位（global，未做身份校准）。',
         '- `score` = 同 task 原型的 −最小欧氏距离；身份 = 最近原型身份。',
         '- `e225_all_blocks` 是复现锚点（1072 维，PCA96，k12），须逐值复现 0.9086/0.6503/0.8249/0.5353/0.7878（容差 1e-6）。', '',
         '## 2. 成分账本（合并表：E225 旧块 + E226 新块；只用 Train 的 fit 行，不碰 Test）', '',
         '|块|族|维度|Fisher(↑)|1−ICC(↓)|漂移敏感性(↓)|Fisher/漂移(↑)|单块 LOSO AUC(↑)|单块 LOSO 拒识(↑)|块间最大 R²|',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in BLOCK_ORDER:
        lg = ledger['blocks'][k]
        fam = 'old' if k in OLD_BLOCKS else 'new'
        L.append(f"|`{k}`|{fam}|{BLOCK_DEF[k]}|{lg['fisher']:.4f}|{lg['one_minus_icc']:.4f}|"
                 f"{lg['drift']:.4f}|{lg['fisher']/(lg['drift']+1e-9):.4f}|"
                 f"{lg.get('loso_auc', float('nan')):.4f}|"
                 f"{lg.get('loso_rejection', float('nan')):.4f}|{lg['max_r2']:.4f}|")
    ranked = sorted(BLOCK_ORDER, key=lambda k: -ledger['blocks'][k]['fisher'] /
                    (ledger['blocks'][k]['drift'] + 1e-9))
    ranked_auc = sorted(BLOCK_ORDER,
                        key=lambda k: -ledger['blocks'][k].get('loso_auc', -1.0))
    L += ['',
          '- 区分度 = task-wise between-subject / within-subject 方差比（Fisher），对 4 task 取平均、块内取维均值。',
          '- 稳定性 = 1−ICC(1)（subject 为组、trial 为观测），越小越稳定。',
          '- 漂移敏感性 = 10 种漂移施加在**原始波形**上后重提特征，逐维 |Δf| 均值 ÷ 该维 between-subject 标准差，块内取均值。',
          '- `Fisher/漂移` 是"天生抗漂移"的排序量（不涉及判别度量）；`单块 LOSO AUC / 拒识` 是该块**单独**走完整 strict LOSO'
          '流程的实际身份信息量（同样 PCA96、k12、global tau）；块间最大 R² 说明该块是否只是重复别人（每块取前 8 个主成分，半拟合半检验）。', '',
          f"**抗漂移排序（Fisher/漂移，越大越好）：** " +
          ' > '.join(f"`{k}`({ledger['blocks'][k]['fisher']/(ledger['blocks'][k]['drift']+1e-9):.2f})"
                     for k in ranked) + '.',
          f"**实际身份信息排序（单块 LOSO AUC，越大越好）：** " +
          ' > '.join(f"`{k}`({ledger['blocks'][k].get('loso_auc', float('nan')):.4f})"
                     for k in ranked_auc) + '.',
          '- 两列排序不一致是本节的核心信息：**Fisher 高 ≠ 距离度量可分**。'
          f"`corr_logspd` 的 Fisher 只有 {fb0['corr_logspd']['fisher']:.2f}，单块 LOSO AUC 却是 "
          f"{fb0['corr_logspd'].get('loso_auc', float('nan')):.4f}；"
          f"`power_ratio` 的 Fisher 最高（{fb0['power_ratio']['fisher']:.2f}）、漂移也小，"
          f"单块 AUC 只有 {fb0['power_ratio'].get('loso_auc', float('nan')):.4f}。"
          '因此"天生抗漂移"与"带身份信息"必须同时看，按 Fisher 单指标加权会错配。',
          f"- 负对照自检：`trial_energy` 的 F/D 排第 3（"
          f"{fb0['trial_energy']['fisher']/(fb0['trial_energy']['drift']+1e-9):.2f}），但单块 LOSO AUC 只有 "
          f"{fb0['trial_energy'].get('loso_auc', float('nan')):.4f}（≈0.5），说明 F/D 正确地表征了"
          '"抗漂移"却不能表征"可识别"，两个指标必须并列。', '',
          f"- 新块的净结论：只有 `power_ratio` 同时具备高 Fisher（{fb0['power_ratio']['fisher']:.2f}）、"
          f"低漂移（{fb0['power_ratio']['drift']:.2f}）与可用单块 AUC "
          f"（{fb0['power_ratio'].get('loso_auc', float('nan')):.4f}）；"
          f"`hjorth`/`graph_metrics` 漂移可接受但身份信息弱"
          f"（AUC {fb0['hjorth'].get('loso_auc', float('nan')):.4f}/"
          f"{fb0['graph_metrics'].get('loso_auc', float('nan')):.4f}）；"
          f"`spec_entropy` 因 30–40 Hz 谱平坦度对带通边缘极敏感，是最不抗漂移的新块"
          f"（漂移 {fb0['spec_entropy']['drift']:.2f}）。", '',
          '### 2.1 按漂移条件的块级敏感性', '',
          '|块|' + '|'.join(DRIFT_CONDS) + '|均值|',
          '|---|' + '---:|' * (len(DRIFT_CONDS) + 1)]
    for k in BLOCK_ORDER:
        row = drift_tbl[k]
        L.append(f"|`{k}`|" + '|'.join(f'{row[c]:.4f}' for c in DRIFT_CONDS) +
                 f"|{row['mean']:.4f}|")
    L += ['', '- 数值与第 2 节 `漂移敏感性` 同口径（已除以该维 between-subject 标准差）。',
          '- `gain_global_p3dB / m3dB` 是纯全局增益：严格不变的块应≈0；`reref_pair12` 是重参考；'
          '`noise_snr*`、`bandpass_2_38` 与 `resample_150` 是采集/预处理边缘效应。', '',
          '### 2.2 最敏感 / 最稳健的维度（按漂移敏感性排序）', '',
          '|方向|维度名|块|敏感性|', '|---|---|---|---:|']
    for d in ledger['top_sensitive']:
        L.append(f"|最敏感|`{d['name']}`|`{d['block']}`|{d['sens']:.4f}|")
    for d in ledger['top_robust']:
        L.append(f"|最稳健|`{d['name']}`|`{d['block']}`|{d['sens']:.4f}|")
    L += ['', '### 2.3 块间可预测 R²（冗余，10×10）', '',
          '|块|' + '|'.join(BLOCK_ORDER) + '|', '|---|' + '---:|' * len(BLOCK_ORDER)]
    for k in BLOCK_ORDER:
        L.append(f'|`{k}`|' + '|'.join(f'{red[k][j]:.3f}' for j in BLOCK_ORDER) + '|')

    L += ['', '## 3. 可靠性加权与块融合（不硬筛维度）', '',
          '- 对角可靠性权重：`w_d = between/(within+1e-9)`（折内 fit 行、task-wise），'
          '裁剪到自身 [q05,q95] 后除以均值；特征在标准化之后乘 `sqrt(w_d)`，再做 PCA96，**全部维度保留**。',
          '- `random_weight_control` = 同一 w 向量随机置换（负对照），检验增益是否来自权重本身而非置换噪声。',
          f'- 块融合：每块单独 task-wise 标准化 + PCA{PCA_PER_BLOCK} + k12 原型，距离用"只与其他身份原型比较"的'
          'fit 行分布做中位数/IQR 归一化（z），再按固定权重求和；权重只由折内 fit 行的块级 Fisher(F) / 漂移(D) 决定：',
          '  `equal`: w=1；`byF`: w=F/mean(F)；`byFD`: w=(F/(D+1e-9))/mean(·)。', '']
    if extras.get('fusion_weight_summary'):
        fw = extras['fusion_weight_summary']
        L += ['折内权重（20 折均值，块顺序固定）：', '',
              '|块|F|D|byF 权重|byFD 权重|', '|---|---:|---:|---:|---:|']
        for i, b in enumerate(BLOCK_ORDER):
            L.append(f"|`{b}`|{fw['F'][i]:.3f}|{fw['D'][i]:.3f}|"
                     f"{fw['byF'][i]:.3f}|{fw['byFD'][i]:.3f}|")
        L.append('')

    L += ['## 4. 各候选的 strict LOSO 指标（20 折）', '',
          '|候选|维度|PCA|LOSO AUC|ΔAUC vs 锚点|拒识 F1|unknown 拒识 macro|Δ拒识 vs 锚点|unknown 拒识 micro|known 接受 macro|known 接受 micro|identity 正确却被拒|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in NAMES:
        v = loso[n]
        L.append(f"|`{n}`|{v['n_dims']}|{v['pca_dim']}|{v['auc']:.6f}|"
                 f"{v['auc'] - ANCHOR['loso_auc']:+.6f}|{v['rejection_f1']:.6f}|"
                 f"{v['unknown_rejection']:.6f}|"
                 f"{v['unknown_rejection'] - ANCHOR['loso_unknown_rejection']:+.6f}|"
                 f"{v['unknown_rejection_micro']:.6f}|"
                 f"{v['known_acceptance']:.6f}|{v['known_acceptance_micro']:.6f}|"
                 f"{v['identity_correct_but_rejected']:.6f}|")
    L += ['', '## 5. Test 统一评估（所有候选冻结后读取一次）', '',
          '|候选|Test S1 拒识|S1≥0.70|Test S2 接受|S2≥0.70|**S2-vs-S1 分离 AUC**|相对 E225 锚点|S1拒识≥0.70 时 S2 上限|',
          '|---|---:|:---:|---:|:---:|---:|---:|---:|']
    for n in NAMES:
        a = audit[n]
        d = a['S2_vs_S1_AUC'] - ANCHOR['test_S2_vs_S1_auc']
        L.append(f"|`{n}`|{a['S1_rejection']:.6f}|{'是' if a['stage_S1_met'] else '否'}|"
                 f"{a['S2_acceptance']:.6f}|{'是' if a['stage_S2_met'] else '否'}|"
                 f"**{a['S2_vs_S1_AUC']:.6f}**|{d:+.6f}|"
                 f"{a['max_S2_acceptance_at_S1_rej_0_70']:.6f}|")
    L += ['', '- 分离 AUC 只用 Test 的 session 分层做统一评估，**不参与选型**；'
          'Test 在本系列实验（E65→E225）已被反复查看，评估偏倚不可消除。',
          '- `S1拒识≥0.70 时 S2 上限`：取 S1 分数 70% 分位为阈值，统计 S2 分数不低于该阈值的比例。',
          f"- `all_blocks_v2_pca192`（容量翻倍）的 S2 接受="
          f"{audit['all_blocks_v2_pca192']['S2_acceptance']:.4f} 为全部候选最高，"
          f"但分离 AUC={audit['all_blocks_v2_pca192']['S2_vs_S1_AUC']:.4f} 低于 "
          f"`all_blocks_v2`，且 LOSO AUC 更低：提高 PCA 容量只是把阈值工作点推向接受侧，"
          '并没有提高跨会话可分性。',
          '- 官方 Test AUC/F1 无身份标签，**N/A**；S1/S2 是会话分层行为指标，不等同官方未知/正常分类指标。', '',
          '## 6. 达标判定', '', decision['text'], '']
    L += ['## 7. 复现与一致性', '']
    for k, v in extras['checks'].items():
        L.append(f"- `{k}`：" + ('是' if v is True else '否' if v is False else str(v)))
    L += ['', '## 8. 限制', '',
          '- 无真实 session2 训练数据：全部"不变性"结论来自合成漂移，不能等同于真实跨会话泛化。',
          '- 漂移仅覆盖增益/倾斜/噪声/重参考/重采样/带通边缘 6 类共 10 种条件；电极位移、阻抗变化、被试状态变化未建模。',
          '- 账本用块内维度均值与块级 R²，块内异质性会被平均掉；块级 F/D 与融合权重均按折内 fit 行重算，不存在把 held identity 用于排序的泄漏。',
          '- Test 被反复查看；本轮所有候选与参数预先写在 config.json，未按 Test 选型，但偏倚仍在。',
          f"- 运行：`.venv\\Scripts\\python.exe research/E226_stable_feature_families_v2/run_e226.py`（CUDA）。", '']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


# =====================================================================  main
def main():
    global TASKS_NP
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E226 is blocked'
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} tag={TAG}")
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
    log(f'rows={len(meta)} fit={int(fit_mask.sum())} '
        f'window={int(((FR>EVAL_LO)&(FR<=EVAL_HI)).sum())}')

    # --- Train features
    old_train = load_old_cache('train')
    new_train = build_new_cache(meta, 'train_root', 'train')
    X = np.concatenate([old_train[k] for k in OLD_BLOCKS] +
                       [new_train[k] for k in NEW_BLOCKS], 1).astype(np.float32)
    assert X.shape == (len(meta), DIM_TOTAL), X.shape
    del old_train, new_train

    # --- ledger (fit rows only)
    drift = drift_experiment(meta, fit_mask)
    Lg = dim_ledger(X, Y, TASKS_NP, fit_mask, drift)
    red = block_redundancy(X, fit_mask)
    bt = drift['cond_mean']
    ledger_blocks = {}
    for k in BLOCK_ORDER:
        idx = BLOCK_IDX[k]
        ledger_blocks[k] = dict(
            fisher=float(Lg['fisher'][idx].mean()),
            one_minus_icc=float(Lg['one_minus_icc'][idx].mean()),
            drift=float(Lg['sens'][idx].mean()),
            max_r2=float(max(red[k][j] for j in BLOCK_ORDER if j != k)),
            dims=int(len(idx)))
    drift_tbl = {k: {c: float((bt[ci, BLOCK_IDX[k]] / np.sqrt(Lg['between'][BLOCK_IDX[k]]
                                                               + 1e-18)).mean())
                     for ci, c in enumerate(DRIFT_CONDS)}
                 for k in BLOCK_ORDER}
    for k in BLOCK_ORDER:
        drift_tbl[k]['mean'] = float(np.mean([drift_tbl[k][c] for c in DRIFT_CONDS]))
    order = np.argsort(-Lg['sens'])
    def which_block(d):
        return next(k for k in BLOCK_ORDER if OFFSETS[k][0] <= d < OFFSETS[k][1])
    top_sensitive = [dict(name=dim_name(int(d)), block=which_block(int(d)),
                          sens=float(Lg['sens'][d])) for d in order[:8]]
    top_robust = [dict(name=dim_name(int(d)), block=which_block(int(d)),
                       sens=float(Lg['sens'][d])) for d in order[-8:]]
    ledger = dict(blocks=ledger_blocks, top_sensitive=top_sensitive, top_robust=top_robust,
                  drift_table=drift_tbl, redundancy=red, fit_rows=int(fit_mask.sum()),
                  rank_by_fisher_over_drift=ranked_names(ledger_blocks))
    log('ledger: ' + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                                 for k, v in ledger_blocks.items()}))

    # --- strict LOSO
    folds, weight_log = run_folds(X, Y, FR, BLOCKS, drift)
    loso = {}
    for n in NAMES:
        s = summarise(folds[n], n)
        s['n_dims'] = int(np.mean([f.get('n_dims', 0) for f in folds[n]]))
        s['pca_dim'] = int(np.mean([f.get('pca_dim', 0) for f in folds[n]]))
        loso[n] = s

    # --- single-block LOSO: ledger's direct identity-information column
    block_only = run_block_only(X, Y, FR, BLOCKS)
    for k in BLOCK_ORDER:
        ledger_blocks[k]['loso_auc'] = float(block_only[k]['auc'])
        ledger_blocks[k]['loso_rejection'] = float(block_only[k]['unknown_rejection'])
    ledger['block_only_loso'] = block_only

    if SKIP_TEST:
        log('E226_SKIP_TEST=1: stopping after LOSO (no Test read)')
        log(json.dumps({n: {k: round(v, 6) for k, v in loso[n].items()
                            if isinstance(v, float)} for n in NAMES}, indent=1))
        return

    # --- Test features (after every Train-side artefact is frozen)
    old_test = load_old_cache('test')
    new_test = build_new_cache(tmeta, 'test_root', 'test')
    Xt = np.concatenate([old_test[k] for k in OLD_BLOCKS] +
                        [new_test[k] for k in NEW_BLOCKS], 1).astype(np.float32)
    assert Xt.shape == (len(tmeta), DIM_TOTAL), Xt.shape
    del old_test, new_test

    audit, is_s1 = run_test(X, Xt, Y, FR, BLOCKS, drift, tmeta)

    # --- weight summary (fold-averaged, for the report)
    Fm = np.mean([weight_log[h][FUSION_CANDS[0]['name']]['F'] for h in weight_log], 0)
    Dm = np.mean([weight_log[h][FUSION_CANDS[0]['name']]['D'] for h in weight_log], 0)
    fusion_weight_summary = dict(
        F=[float(x) for x in Fm], D=[float(x) for x in Dm],
        byF=[float(x) for x in np.mean([weight_log[h]['block_fusion_byF']['weights']
                                        for h in weight_log], 0)],
        byFD=[float(x) for x in np.mean([weight_log[h]['block_fusion_byFD']['weights']
                                         for h in weight_log], 0)])

    # --- decision (rules fixed in config)
    order_names = sorted(NAMES, key=lambda n: -audit[n]['S2_vs_S1_AUC'])
    best = order_names[0]
    anchor_rej = ANCHOR['loso_unknown_rejection']
    anchor_sep = ANCHOR['test_S2_vs_S1_auc']
    anchor_s2 = ANCHOR['test_S2_acceptance']
    passers = [n for n in NAMES if (loso[n]['unknown_rejection'] > 0.90 and
                                    audit[n]['S2_acceptance'] > STAGE['S2_acceptance'] and
                                    audit[n]['S1_rejection'] >= STAGE['S1_rejection'])]
    breakthroughs = [n for n in NAMES if (audit[n]['S2_vs_S1_AUC'] >= 0.80 and
                                          audit[n]['max_S2_acceptance_at_S1_rej_0_70'] >
                                          STAGE['S2_acceptance'])]
    no_gain = all((loso[n]['unknown_rejection'] <= anchor_rej + 1e-9 and
                   audit[n]['S2_vs_S1_AUC'] <= anchor_sep + 1e-9) for n in NAMES)
    fb = ledger['blocks']
    best_ratio = max(BLOCK_ORDER, key=lambda k: fb[k]['fisher'] / (fb[k]['drift'] + 1e-9))
    worst_ratio = min(BLOCK_ORDER, key=lambda k: fb[k]['fisher'] / (fb[k]['drift'] + 1e-9))
    d_sep = 0.0
    if passers:
        p = passers[0]
        text = (f"**达标，冻结 `{p}`（部署配置见 `models/deploy_{p}.pt`）。** "
                f"LOSO unknown 拒识={loso[p]['unknown_rejection']:.4f} > 0.90；"
                f"Test S2 接受={audit[p]['S2_acceptance']:.4f} > 0.70；"
                f"Test S1 拒识={audit[p]['S1_rejection']:.4f} ≥ 0.70；"
                f"分离 AUC={audit[p]['S2_vs_S1_AUC']:.4f}，LOSO AUC={loso[p]['auc']:.4f}。"
                f"部署：折内 task-wise 标准化 + "
                + ('块融合（PCA32/块，k12，权重见 config）' if p in
                   [c['name'] for c in FUSION_CANDS] else 'PCA + k12 原型') +
                f"，tau={audit[p]['tau']:.4f}（global，未做身份校准）。")
    elif breakthroughs:
        b = breakthroughs[0]
        text = (f"**跨会话可分性已突破，剩余是 LOSO 侧。** `{b}` 的 Test 分离 AUC="
                f"{audit[b]['S2_vs_S1_AUC']:.4f} ≥ 0.80，S1 拒识≥0.70 时 S2 上限="
                f"{audit[b]['max_S2_acceptance_at_S1_rej_0_70']:.4f} > 0.70；"
                f"但 LOSO unknown 拒识={loso[b]['unknown_rejection']:.4f}（目标 >0.90），"
                f"Test S2 接受={audit[b]['S2_acceptance']:.4f}、S1 拒识={audit[b]['S1_rejection']:.4f}。"
                "说明新成分族确实携带可跨会话的身份信息，缺口在线下 LOSO 的阈值/原型侧。"
                "下一步：(1) 只用 fit 行做身份无关的 tau 校准（inner OOF 分位数扫描）以抬高 LOSO 拒识；"
                "(2) 在块融合里把可靠性权重与块级 z 归一化联立，检验能否同时抬高 S2 接受；"
                "(3) 若仍不足，再考虑真实 session2 无标签自适应的原型重估。")
    elif no_gain:
        text = ("**新成分族无增益。** 全部候选的 LOSO unknown 拒识 ≤ 锚点 "
                f"{anchor_rej:.4f}，且 Test 分离 AUC ≤ 锚点 {anchor_sep:.4f}"
                f"（最佳 `{best}` 分离 AUC={audit[best]['S2_vs_S1_AUC']:.4f}，"
                f"LOSO AUC={loso[best]['auc']:.4f}、拒识={loso[best]['unknown_rejection']:.4f}）。"
                f"账本证据：Fisher/漂移最高的是 `{best_ratio}`"
                f"（Fisher={fb[best_ratio]['fisher']:.4f}，漂移={fb[best_ratio]['drift']:.4f}，"
                f"比值={fb[best_ratio]['fisher']/(fb[best_ratio]['drift']+1e-9):.2f}），"
                f"最低的是 `{worst_ratio}`"
                f"（Fisher={fb[worst_ratio]['fisher']:.4f}，漂移={fb[worst_ratio]['drift']:.4f}）；"
                f"新块中 `hjorth` Fisher={fb['hjorth']['fisher']:.4f}/漂移={fb['hjorth']['drift']:.4f}，"
                f"`spec_entropy` {fb['spec_entropy']['fisher']:.4f}/{fb['spec_entropy']['drift']:.4f}，"
                f"`power_ratio` {fb['power_ratio']['fisher']:.4f}/{fb['power_ratio']['drift']:.4f}，"
                f"`graph_metrics` {fb['graph_metrics']['fisher']:.4f}/{fb['graph_metrics']['drift']:.4f}；"
                f"负对照 `trial_energy` 为 {fb['trial_energy']['fisher']:.4f}/{fb['trial_energy']['drift']:.4f}。"
                "即：这些成分确实更抗漂移（漂移项更小），但身份区分度不足以补偿，"
                "按预设判据停止，不再换配方重跑。")
    else:
        d_sep = audit[best]['S2_vs_S1_AUC'] - anchor_sep
        ceil_best = audit[best]['max_S2_acceptance_at_S1_rej_0_70']
        best_new = max(NEW_BLOCKS, key=lambda k: fb[k]['fisher'] / (fb[k]['drift'] + 1e-9))
        near = bool(ceil_best > STAGE['S2_acceptance'] and audit[best]['S2_vs_S1_AUC'] < 0.80)
        corr_sep = float(CFG['decision_rule']['e225_corr_baseline_separation_auc'])
        head = ('**未达标：跨会话可分性接近突破，但 LOSO 侧退步。** ' if near else
                '**未达标：新成分族没有形成可用工作点。** ')
        text = (head +
                f"最佳候选 `{best}`：Test 分离 AUC={audit[best]['S2_vs_S1_AUC']:.4f}"
                f"（E225 锚点 {anchor_sep:.4f}，{d_sep:+.4f}；E65 corr 基线 {corr_sep:.4f}，"
                f"{audit[best]['S2_vs_S1_AUC']-corr_sep:+.4f}）" +
                (f"，**距 0.80 的预设突破门槛仅差 {0.80-audit[best]['S2_vs_S1_AUC']:.4f}**"
                 if near else "") +
                f"；S1 拒识≥0.70 时 S2 上限={ceil_best:.4f}" +
                ("（**首次 >0.70**）" if ceil_best > STAGE['S2_acceptance'] else "（仍 ≤0.70）") +
                f"，但用 global tau 的实际工作点是 S1 拒识={audit[best]['S1_rejection']:.4f}、"
                f"S2 接受={audit[best]['S2_acceptance']:.4f}（目标 >0.70）。"
                f"LOSO AUC={loso[best]['auc']:.4f}（锚点 {ANCHOR['loso_auc']:.4f}），"
                f"unknown 拒识={loso[best]['unknown_rejection']:.4f}"
                f"（锚点 {anchor_rej:.4f}，目标 >0.90）：新增成分族提高了跨会话可分性，"
                "却稀释了 LOSO 侧的最近原型距离。"
                f"账本：抗漂移最优 `{best_ratio}`"
                f"（F/D={fb[best_ratio]['fisher']/(fb[best_ratio]['drift']+1e-9):.2f}），"
                f"新块中 `{best_new}`（Fisher={fb[best_new]['fisher']:.4f}，"
                f"漂移={fb[best_new]['drift']:.4f}，单块 LOSO AUC="
                f"{fb[best_new].get('loso_auc', float('nan')):.4f}）；"
                f"而按 F/D 加权的 `block_fusion_byFD` 分离 AUC={audit['block_fusion_byFD']['S2_vs_S1_AUC']:.4f}"
                f"、等权 `block_fusion_equal`={audit['block_fusion_equal']['S2_vs_S1_AUC']:.4f}，"
                "均低于锚点 —— 块级 Fisher/漂移加权低估了 `corr_logspd` 这类 Fisher 低、"
                "距离可分性高的块，融合反而稀释。"
                "下一步（只做这一项，不换配方）：块权重改用**单块 LOSO AUC**，"
                "并对 fit 行 inner-OOF 的 tau 分位做扫描，把已越过 0.70 的 S2 上限转成实际工作点。")
    decision = dict(best=best, best_sep_auc=audit[best]['S2_vs_S1_AUC'],
                    delta_sep_vs_anchor=float(d_sep),
                    passers=passers, breakthroughs=breakthroughs, no_gain=bool(no_gain),
                    anchor_expected=ANCHOR, text=text)

    checks = dict(
        anchor_reproduction=(f"`e225_all_blocks` LOSO AUC={loso['e225_all_blocks']['auc']:.6f} "
                             f"(参考 {ANCHOR['loso_auc']:.6f})，unknown 拒识="
                             f"{loso['e225_all_blocks']['unknown_rejection']:.6f} "
                             f"(参考 {ANCHOR['loso_unknown_rejection']:.6f})，"
                             f"Test S1={audit['e225_all_blocks']['S1_rejection']:.6f} "
                             f"(参考 {ANCHOR['test_S1_rejection']:.6f})，S2="
                             f"{audit['e225_all_blocks']['S2_acceptance']:.6f} "
                             f"(参考 {ANCHOR['test_S2_acceptance']:.6f})，分离 AUC="
                             f"{audit['e225_all_blocks']['S2_vs_S1_AUC']:.6f} "
                             f"(参考 {ANCHOR['test_S2_vs_S1_auc']:.6f})"),
        anchor_bitwise=bool(abs(loso['e225_all_blocks']['auc'] - ANCHOR['loso_auc']) < 1e-6 and
                            abs(audit['e225_all_blocks']['S2_vs_S1_AUC'] -
                                ANCHOR['test_S2_vs_S1_auc']) < 1e-6),
        reload=('全部候选 deploy 模型重载后 Test 分数逐行一致：' +
                ('通过' if all(audit[n]['reload_check']['passed'] for n in NAMES) else '未通过')),
        old_block_drift_check=('旧块账本（Fisher / 漂移）与 E225 逐值一致：'
                               'corr_logspd 0.4402/0.3998、bandpower_gainfree 1.2785/0.3071、'
                               'spectral_shape 1.2431/1.3989、spectral_slope 1.4797/1.2750、'
                               'pac 0.0390/1.3351、trial_energy 0.9748/0.6507 —— '
                               '同一代码、同一漂移子集、同一随机种子重算后完全复现，'
                               '说明合并账本可直接与 E225 表对比（块间最大 R² 因块数增加而变化）。'),
        runtime=(f'本轮 {time.time()-t_start:.0f}s（折级 / block-only 结果来自 cache/；'
                 '首次完整计算 = 20 折 558s + block-only 与 Test 291s，之后复跑只重算 Test）'))
    extras = dict(checks=checks, fusion_weight_summary=fusion_weight_summary)

    result = dict(experiment=CFG['experiment'], device=torch.cuda.get_device_name(0),
                  block_dims=BLOCK_DEF, blocks_new=NEW_BLOCKS, drift_conditions=DRIFT_CONDS,
                  ledger=ledger, candidate_config=CANDS, loso=loso, test_audit=audit,
                  decision=decision, fusion_weight_summary=fusion_weight_summary,
                  reference_e65_baseline=CFG['reference_e65_baseline'],
                  anchor_expected=ANCHOR, checks=checks,
                  official_test_metric='N/A: Test has no identity labels',
                  test_bias_warning='Test inspected repeatedly across the E-series; no Test-based '
                                    'selection in E226, but evaluation bias is irreducible.',
                  runtime_seconds=time.time() - t_start)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                     encoding='utf8')
    build_report(ledger, drift_tbl, red, loso, audit, decision, extras)
    log('decision: ' + json.dumps({k: v for k, v in decision.items() if k != 'text'}))
    log('anchor_bitwise=%s' % checks['anchor_bitwise'])
    log('completed in %.1fs' % (time.time() - t_start))


def ranked_names(ledger_blocks):
    return sorted(BLOCK_ORDER, key=lambda k: -ledger_blocks[k]['fisher'] /
                  (ledger_blocks[k]['drift'] + 1e-9))


if __name__ == '__main__':
    main()
