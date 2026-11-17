"""E235: deployment-condition threshold calibration.

Question.  E233 (abs_dist + symmetric per-(session, task) mean alignment) is the
compliant optimum: LOSO unknown rejection 0.9000, Test S1 1.0000, Test S2 0.3122,
separation AUC 0.8795.  Dropping the same score's threshold until S1 rejection is
0.95 gives Test S2 = 0.7569, but that threshold position cannot be derived from
the training domain: the Train LOSO "held-out impostor" sits in the SAME session
and the SAME batch as the enrolled users, so it is much harder than the five
genuinely unregistered Test subjects.

This experiment therefore moves the CALIBRATION TARGET, not the score:
 1. estimate the cross-day offset delta in the frozen e226_auc2W PCA space
    (A: transductive, delta = mean(Test session2) - mean(Train), per task;
     B: train-only, delta = mean displacement of a declared synthetic drift
        profile applied to raw waveforms and re-extracted through all 10 blocks);
 2. translate the held-out Train impostors of every LOSO fold by delta in that
    fold's own PCA space (z_shifted = z_impostor + delta_task);
 3. bisect A on the shifted impostors so that pooled LOSO unknown rejection is
    0.90, and read tau = (1-A) quantile of the deploy inner-OOF KNOWN scores;
 4. read the resulting tau on the frozen E233 mean-align Test scores.

Nothing in the representation, the prototypes, the alignment or the Test scores
is refitted.  The only free quantity is where the threshold sits.

The decisive diagnostic is the cosine between delta and the identity
between-class-scatter direction: if delta points along the direction that
separates identities, then translating the impostors by delta is moving identity
information and the resulting S2 gain is an artefact, not a deployment effect.

Read-outs per candidate: tau, unshifted LOSO rejection, shifted LOSO rejection,
Test S1 rejection, Test S2 acceptance, separation AUC, k=20/28 cluster purity,
per-fold known acceptance (the Train-side analogue of Test S2) and per-task S1/S2.
"""
from __future__ import annotations

import json
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
MIN_CELL_N = int(CFG['scoring']['min_cell_n'])
PURITY_K = [int(k) for k in CFG['metrics']['purity']['k']]
PURITY_SEED = int(CFG['metrics']['purity']['seed'])
TARGET = float(CFG['threshold']['target_loso_unknown_rejection'])
BISECT_ITERS = int(CFG['threshold']['bisection_iters'])
N = int(CFG['data']['samples_per_trial'])
CH = int(CFG['data']['channels'])
FS = int(CFG['data']['sampling_rate_hz'])
BANDS = [tuple(b) for b in CFG['features']['bands_hz']]
NB = len(BANDS)
CHUNK = int(CFG['features']['chunk_trials'])
WORKERS = int(CFG['features']['io_workers'])
SUB_PER_CELL = int(CFG['delta']['drift_subset']['per_subject_task'])
HALF_REPEATS = int(CFG['delta']['stability']['split_half_repeats'])
HALF_SEED = int(CFG['delta']['stability']['split_half_seed'])
PROFILES = CFG['delta']['profiles']
COMPONENT_PROFILES = CFG['delta']['component_profiles']
ALL_PROFILES = dict(PROFILES)
ALL_PROFILES.update(COMPONENT_PROFILES)
PRIMARY_PROFILE = 'P1_gain_tilt_noise_reref'

ANCH = CFG['predecessor']['known_anchors']
A_REF = float(ANCH['e228_auc2W_calibrated']['A'])
TAU_REF = float(ANCH['e228_auc2W_calibrated']['test_tau'])
LOSO_REF = float(ANCH['e228_auc2W_calibrated']['loso_unknown_rejection'])
LOSO_AUC_REF = float(ANCH['e228_auc2W_calibrated']['loso_auc_mean'])
FOLD_TAU_REF = float(ANCH['e228_auc2W_calibrated']['fold_tau_mean'])
BASE_S1 = float(ANCH['e228_auc2W_calibrated']['test_S1_rejection'])
BASE_S2 = float(ANCH['e228_auc2W_calibrated']['test_S2_acceptance'])
E233_S2 = float(ANCH['e233_mean_align_abs_exact']['test_S2_acceptance'])
E233_S1 = float(ANCH['e233_mean_align_abs_exact']['test_S1_rejection'])
E233_AUC = float(ANCH['e233_mean_align_abs_exact']['test_separation_auc'])
E233_S2_AT_GATE = float(ANCH['e233_mean_align_abs']['S2_at_S1_0p95'])
E233_PUR = {f'k{k}': float(ANCH['e233_mean_align_abs'][f'purity_k{k}_micro']) for k in PURITY_K}
E233_MEAN_S1 = float(ANCH['e233_mean_align_abs']['mean_score_S1'])
E233_MEAN_S2 = float(ANCH['e233_mean_align_abs']['mean_score_S2'])

S1_GATE = 0.95
USER_S2_TARGET = 0.60
PURITY_TOL = 0.95
COS_LIMIT = 0.70

CANDS = [dict(v, id=cid) for cid, v in CFG['candidates'].items()]

TASKS_NP = None
Y = None
FR = None
BLOCKS = None
TE_TASK = None
IS_S1 = None
S2 = None
OFFSETS, BLOCK_IDX = {}, {}
_acc = 0
for _k, _v in BLOCK_DIM.items():
    OFFSETS[_k] = (_acc, _acc + _v)
    BLOCK_IDX[_k] = np.arange(_acc, _acc + _v)
    _acc += _v
DIM_TOTAL = _acc
assert DIM_TOTAL == 1569, DIM_TOTAL

Xtr = Xte = None
XDRIFT_CLEAN = None
XDRIFT = {}
SUB_TASK = None
SUB_ROWS = None

# ---- E65/E225/E226 feature constants (identical to research/E225, E226)
SHRINK = 0.05
EIG_FLOOR = 1e-12
OFFMUL = float(np.sqrt(2.0))
ZBAND_EPS = 1e-6
PAIRS = [(0, 6), (1, 5), (2, 4), (10, 17), (11, 16), (12, 13), (14, 15), (8, 9)]
CROSS_RATIOS = [('alpha/theta', 1, 0), ('beta/alpha', -1, 1), ('beta/theta', -1, 0),
                ('gamma/alpha', 4, 1)]


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


def cosine(a, b):
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float('nan')


# =====================================================  frozen model / transform
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
    auc = np.array([np.mean([v for i, v in enumerate(table[b], start=1) if i != held])
                    for b in BLOCK_ORDER], float)
    w = auc ** 2
    return (w / w.mean()).astype(np.float32)


def expand_weights(wb):
    w = np.empty(DIM_TOTAL, np.float32)
    for j, b in enumerate(BLOCK_ORDER):
        w[BLOCK_IDX[b]] = wb[j]
    return w


def d1_rows(state, z, tasks, rows):
    """Nearest same-task prototype distance (score = -d1)."""
    proto_task, proto_id = proto_meta(state)
    n = len(tasks)
    d1 = np.full(n, np.nan)
    pred = np.full(n, -1, int)
    for t in TASKS:
        rr = rows[tasks[rows] == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rr]), tensor(state['p'][cols])).double()
        dmin, amin = D.min(1)
        d1[rr] = dmin.cpu().numpy()
        pred[rr] = proto_id[cols][amin.cpu().numpy()]
    return d1, pred


def shifted_d1(state, z, tasks, rows, delta, scale=1.0, imp_rows=None):
    """d1 after translating `imp_rows` (held-out impostors) by scale * delta[task]."""
    if delta is None or imp_rows is None or not len(imp_rows):
        return d1_rows(state, z, tasks, rows)
    zmod = z.copy()
    for t in TASKS:
        rr = imp_rows[tasks[imp_rows] == t]
        if len(rr):
            zmod[rr] = zmod[rr] + scale * np.asarray(delta[t], np.float32)
    return d1_rows(state, zmod, tasks, rows)


# =====================================================  feature extraction (raw)
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
def hjorth_feats(bb):
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
    pair = torch.stack([lp[..., a] - lp[..., b] for a, b in PAIRS], -1)
    beta = torch.logaddexp(lp[:, 2], lp[:, 3]) - float(np.log(2.0))
    ratios = []
    for _, hi, lo in CROSS_RATIOS:
        h = beta if hi < 0 else lp[:, hi]
        lo_t = beta if lo < 0 else lp[:, lo]
        ratios.append(h - lo_t)
    return torch.cat([pair.flatten(1), torch.stack(ratios, 1).flatten(1)], 1)


@torch.no_grad()
def graph_feats(bb):
    x = bb - bb.mean(-1, keepdim=True)
    x = x / x.std(-1, keepdim=True, unbiased=True).clamp_min(1e-12)
    corr = x @ x.transpose(-1, -2) / (N - 1)
    w = corr.abs()
    eye = torch.eye(CH, device=DEVICE)
    w = w * (1.0 - eye)

    vals, _ = torch.linalg.eigh(w)
    eig_share = vals[..., -1] / vals.abs().sum(-1).clamp_min(1e-12)

    r, c = torch.triu_indices(CH, CH, offset=1, device=DEVICE)
    upper = w[..., r, c]
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
    d = d * (1.0 - eye)
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

    out = torch.stack([clust, eff, hstr, eig_share, q], -1).flatten(1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def all_blocks(raw):
    """All 10 blocks, exactly as E225/E226 define them."""
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
    beta_t = torch.fft.irfft(spec * _mask(13, 30), n=N, dim=-1)
    amp = analytic(beta_t).abs()
    ph = analytic(alpha).angle()
    pac = (amp * torch.polar(torch.ones_like(amp), ph)).mean(-1).abs() \
        / amp.mean(-1).clamp_min(1e-12)

    energy = torch.log(raw.pow(2).mean(dim=(1, 2)).sqrt() + 1e-12)[:, None]

    return {'corr_logspd': corr_logspd(raw), 'bandpower_gainfree': gainfree,
            'spectral_shape': shape, 'spectral_slope': slope, 'pac': pac,
            'trial_energy': energy, 'hjorth': hjorth_feats(bb),
            'spec_entropy': spec_entropy_feats(spec), 'power_ratio': power_ratio_feats(lp),
            'graph_metrics': graph_feats(bb)}


def load_mat(path):
    d = np.asarray(loadmat(path, variable_names=['data'])['data'], dtype=np.float32)
    if d.shape != (N, CH):
        raise ValueError(f'{path}: expected ({N},{CH}) got {d.shape}')
    if not np.isfinite(d).all():
        raise ValueError(f'{path}: non-finite')
    return np.ascontiguousarray(d.T)


def apply_profile(raw, prof, seed, noise_seed):
    """Declared synthetic session drift, applied on the RAW waveform."""
    x = raw
    gspec = prof.get('per_channel_gain_db')
    g = None
    if gspec == 'alternating':
        g = np.array([3.0 if c % 2 == 0 else -3.0 for c in range(CH)], np.float32)
    elif isinstance(gspec, str) and gspec.startswith('random'):
        rng = np.random.default_rng(seed)
        g = rng.uniform(-3.0, 3.0, CH).astype(np.float32)
    elif isinstance(gspec, (int, float)):
        g = np.full(CH, float(gspec), np.float32)
    if g is not None:
        x = x * tensor(10.0 ** (g / 20.0))[None, :, None]
    if 'tilt_alpha' in prof:
        alpha = float(prof['tilt_alpha'])
        w = (freq_grid().clamp_min(1.0) / 10.0) ** alpha
        x = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * w, n=N, dim=-1)
    if prof.get('reref') == 'pair12':
        x = x - 0.5 * (x[:, [0], :] + x[:, [1], :])
    if 'noise_snr_db' in prof:
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(int(noise_seed))
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-9)
        noise = torch.randn(x.shape, device=DEVICE, generator=rng)
        x = x + noise * rms / float(10 ** (float(prof['noise_snr_db']) / 20.0))
    return x


def extract_features(paths, prof, tag):
    """Extract all 10 blocks for `paths`, optionally after a declared drift profile."""
    out = np.zeros((len(paths), DIM_TOTAL), np.float32)
    t0 = time.time()
    seed = HALF_SEED + sum(map(ord, tag))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(paths), CHUNK):
            batch = paths[start:start + CHUNK]
            arrays = list(pool.map(load_mat, batch))
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=DEVICE)
            if prof is not None:
                raw = apply_profile(raw, prof, seed, seed + 7919 * (start + 1))
            fb = all_blocks(raw)
            out[start:start + len(raw)] = np.concatenate(
                [fb[k].float().cpu().numpy() for k in BLOCK_ORDER], 1)
            if (start + len(raw)) % 512 < CHUNK or start + len(raw) == len(paths):
                log(f'  {tag} features {start + len(raw)}/{len(paths)} ({time.time()-t0:.0f}s)')
    return out


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


def load_or_extract_drift(tr, fit_all):
    """Clean re-extraction + one extraction per declared drift profile (cached)."""
    global SUB_ROWS, SUB_TASK, XDRIFT_CLEAN, XDRIFT
    SUB_ROWS = drift_subset_rows(tr, fit_all)
    SUB_TASK = TASKS_NP[SUB_ROWS]
    paths = [ROOT / CFG['data']['train_root'] / str(p) for p in tr.path.iloc[SUB_ROWS]]
    f_clean = CACHE / f'drift_{CACHE_VERSION}_clean.npy'
    if f_clean.exists():
        log(f'drift subset reused ({len(SUB_ROWS)} rows)')
        XDRIFT_CLEAN = np.load(f_clean).astype(np.float32)
    else:
        log(f'drift subset: {len(SUB_ROWS)} rows (re-extracting CLEAN features from raw)')
        XDRIFT_CLEAN = extract_features(paths, None, 'clean')
        np.save(f_clean, XDRIFT_CLEAN)
    for name in ALL_PROFILES:
        f = CACHE / f'drift_{CACHE_VERSION}_{name}.npy'
        if f.exists():
            XDRIFT[name] = np.load(f).astype(np.float32)
        else:
            log(f'drift profile {name}: {ALL_PROFILES[name]}')
            XDRIFT[name] = extract_features(paths, ALL_PROFILES[name], name)
            np.save(f, XDRIFT[name])
    return dict(subset_rows=[int(x) for x in SUB_ROWS], n=len(SUB_ROWS),
                per_task={t: int((SUB_TASK == t).sum()) for t in TASKS})


def feature_reproduction_check():
    """Re-extracted CLEAN features vs the frozen cached rows (same scale check)."""
    out = {}
    for b in BLOCK_ORDER:
        cached = (np.load(ROOT / CFG['feature_cache']['train'][b]).astype(np.float32))[SUB_ROWS]
        rec = XDRIFT_CLEAN[:, BLOCK_IDX[b]]
        d = np.abs(rec - cached)
        out[b] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
    out['corr_logspd_note'] = ('the corr block is recomputed from raw instead of read from the '
                               'frozen cache; a small difference is expected because the E65 cache '
                               'is the reference implementation of the same formula')
    return out


# =====================================================  delta estimation
def delta_transductive(ztr, zte):
    d = {}
    for t in TASKS:
        a = zte[(S2) & (TE_TASK == t)].astype(np.float64).mean(0)
        b = ztr[(TASKS_NP == t)].astype(np.float64).mean(0)
        d[t] = (a - b).astype(np.float32)
    return d


def delta_transductive_fitref(ztr, zte, fit):
    d = {}
    for t in TASKS:
        a = zte[(S2) & (TE_TASK == t)].astype(np.float64).mean(0)
        b = ztr[fit & (TASKS_NP == t)].astype(np.float64).mean(0)
        d[t] = (a - b).astype(np.float32)
    return d


def delta_synthetic(state, profile):
    zc = transform(state, XDRIFT_CLEAN, SUB_TASK).astype(np.float64)
    zp = transform(state, XDRIFT[profile], SUB_TASK).astype(np.float64)
    return {t: (zp[SUB_TASK == t].mean(0) - zc[SUB_TASK == t].mean(0)).astype(np.float32)
            for t in TASKS}


def split_half_stability(zte, delta_full):
    """Re-estimate delta_A from two random halves of the session2 rows of each task."""
    out = {}
    for t in TASKS:
        rows = np.flatnonzero(S2 & (TE_TASK == t))
        cos, rel, n1, n2 = [], [], [], []
        for r in range(HALF_REPEATS):
            rng = np.random.default_rng(HALF_SEED + 7919 * r + sum(map(ord, t)))
            perm = rng.permutation(len(rows))
            h1, h2 = rows[perm[: len(rows) // 2]], rows[perm[len(rows) // 2:]]
            # delta from each half, relative to the same (all-Train) reference
            ref = delta_full[t] - zte[(S2) & (TE_TASK == t)].astype(np.float64).mean(0)
            d1 = zte[h1].astype(np.float64).mean(0) + ref
            d2 = zte[h2].astype(np.float64).mean(0) + ref
            cos.append(cosine(d1, d2))
            rel.append(float(np.linalg.norm(d1 - d2) /
                             max(np.linalg.norm(delta_full[t]), 1e-12)))
            n1.append(float(np.linalg.norm(d1)))
            n2.append(float(np.linalg.norm(d2)))
        out[t] = dict(n=int(len(rows)),
                      half_cosine_mean=float(np.mean(cos)), half_cosine_min=float(np.min(cos)),
                      half_rel_diff_mean=float(np.mean(rel)),
                      half_norm_mean=float(np.mean(n1)),
                      delta_norm=float(np.linalg.norm(delta_full[t])))
    return out


def identity_directions(ztr, fit):
    """Top between-class-scatter direction per task, on Train fit rows only."""
    out = {}
    for t in TASKS:
        rows = fit & (TASKS_NP == t)
        ids = np.unique(Y[rows])
        Z = ztr[rows].astype(np.float64)
        mu = Z.mean(0)
        Sb = np.zeros((Z.shape[1], Z.shape[1]))
        for c in ids:
            m = Y[rows] == c
            mc = Z[m].mean(0)
            Sb += float(m.sum()) * np.outer(mc - mu, mc - mu)
        w, V = np.linalg.eigh(Sb)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        V = V[:, :5].copy()
        for j in range(V.shape[1]):                      # deterministic orientation
            if V[np.argmax(np.abs(V[:, j])), j] < 0:
                V[:, j] = -V[:, j]
        out[t] = dict(v1=V[:, 0].astype(np.float64), top5=V,
                      eig_share=[float(x) for x in (w[:5] / w.sum())],
                      n=int(rows.sum()), n_class=int(len(ids)))
    return out


def delta_identity_diagnostic(delta, ident):
    """|cos| is reported because an eigenvector's sign is not identifiable."""
    out = {}
    for t in TASKS:
        d = delta[t].astype(np.float64)
        v1 = ident[t]['v1']
        nd = float(d @ d)
        top5 = ident[t]['top5']
        out[t] = dict(delta_norm=float(np.linalg.norm(d)),
                      cos_identity_v1=abs(cosine(d, v1)),
                      share_top5_identity_subspace=float(np.sum((top5.T @ d) ** 2) / max(nd, 1e-12)),
                      top1_eig_share=ident[t]['eig_share'][0])
    return out


def pooled_identity_cosine(delta, ident):
    d = np.concatenate([delta[t].astype(np.float64) for t in TASKS])
    v = np.concatenate([ident[t]['v1'] for t in TASKS])
    return abs(cosine(d, v))


def impostor_outward_cosine(delta, folds):
    """cos(delta, mean unit vector from the nearest prototype to the impostor)."""
    out = {}
    for i, t in enumerate(TASKS):
        acc = [d['unit_out'][i] for d in folds
               if np.all(np.isfinite(d['unit_out'][i]))]
        if not acc:
            out[t] = float('nan')
            continue
        out[t] = cosine(delta[t].astype(np.float64), np.mean(acc, 0))
    return out


# =====================================================  folds
def fold_path(h):
    return CACHE / f'fold_{CACHE_VERSION}_{h:02d}.npz'


def run_fold(h, table):
    fit = (Y != h) & (FR <= FIT_FRAC)
    window = (FR > EVAL_LO) & (FR <= EVAL_HI)
    win = np.flatnonzero(window)
    w = expand_weights(block_weights(table, h))
    st = fit_rep(Xtr, TASKS_NP, Y, fit, K_PROTO, PCA_DIM, w)
    z = transform(st, Xtr, TASKS_NP)
    imp = win[Y[win] == h]

    # delta expressed in THIS fold's PCA space
    zte_h = transform(st, Xte, TE_TASK).astype(np.float64)
    dA = {t: (zte_h[S2 & (TE_TASK == t)].mean(0) - z[fit & (TASKS_NP == t)].mean(0))
          for t in TASKS}
    zc = transform(st, XDRIFT_CLEAN, SUB_TASK).astype(np.float64)
    zp = transform(st, XDRIFT[PRIMARY_PROFILE], SUB_TASK).astype(np.float64)
    dB = {t: (zp[SUB_TASK == t].mean(0) - zc[SUB_TASK == t].mean(0)) for t in TASKS}

    d1 = {}
    for name, delta, scale in (('orig', None, 0.0), ('A', dA, 1.0), ('B', dB, 1.0),
                              ('Ahalf', dA, 0.5)):
        dd, pred = shifted_d1(st, z, TASKS_NP, win, delta, scale, imp)
        d1[name] = dd[win]

    # mean outward direction of the impostors (per task) for the cosine diagnostic
    proto_task, proto_id = proto_meta(st)
    unit_out = {}
    for t in TASKS:
        rr = imp[TASKS_NP[imp] == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        D = torch.cdist(tensor(z[rr]), tensor(st['p'][cols])).double()
        dmin, amin = D.min(1)
        P = st['p'][cols][amin.cpu().numpy()]
        u = (z[rr] - P)
        u = u / np.linalg.norm(u, axis=1, keepdims=True).clip(min=1e-12)
        unit_out[t] = u.mean(0)

    # inner 5-block OOF known scores (tau source), verbatim E227/E234
    oof_d1 = np.full(len(Y), np.nan)
    for b in np.unique(BLOCKS[fit]):
        hold = fit & (BLOCKS == b)
        inner = fit & ~hold
        sti = fit_rep(Xtr, TASKS_NP, Y, inner, K_PROTO, PCA_DIM, w)
        zi = transform(sti, Xtr, TASKS_NP)
        rr = np.flatnonzero(hold)
        a1, _ = d1_rows(sti, zi, TASKS_NP, rr)
        oof_d1[rr] = a1[rr]

    np.savez_compressed(
        fold_path(h),
        d1_orig=d1['orig'], d1_A=d1['A'], d1_B=d1['B'], d1_Ahalf=d1['Ahalf'],
        pred=d1_rows(st, z, TASKS_NP, win)[1], subject=Y[win],
        known=(Y[win] != h), unknown=(Y[win] == h),
        task=np.asarray(TASKS_NP[win], dtype='<U8'),
        oof_d1=oof_d1, oof_fit=fit,
        delta_A=np.stack([dA[t] for t in TASKS]), delta_B=np.stack([dB[t] for t in TASKS]),
        unit_out=np.stack([unit_out.get(t, np.full(PCA_DIM, np.nan)) for t in TASKS]),
        wb=np.asarray(block_weights(table, h), np.float32))
    ref = np.load(ROOT / CFG['feature_cache']['frozen_fold_cache'].format(fold=h),
                  allow_pickle=True)
    dd = float(np.max(np.abs(-ref['score'].astype(np.float64) - d1['orig'])))
    return dict(held=h, max_abs_d1_vs_e227=dd,
                n_known=int((Y[win] != h).sum()), n_unknown=int((Y[win] == h).sum()),
                delta_A_norm={t: float(np.linalg.norm(dA[t])) for t in TASKS},
                delta_B_norm={t: float(np.linalg.norm(dB[t])) for t in TASKS})


def load_fold(h):
    d = np.load(fold_path(h), allow_pickle=True)
    return {k: d[k] for k in d.files}


def run_folds(table):
    meta, reused = [], 0
    for h in range(1, N_FOLDS + 1):
        if fold_path(h).exists():
            d = np.load(fold_path(h))
            if all(k in d.files for k in ('d1_orig', 'd1_A', 'd1_B', 'd1_Ahalf', 'oof_d1')):
                reused += 1
                continue
        t0 = time.time()
        row = run_fold(h, table)
        meta.append(row)
        log(f'fold {h:02d} n_known={row["n_known"]} n_unknown={row["n_unknown"]} '
            f'max|d1 - E227 d1|={row["max_abs_d1_vs_e227"]:.2e} ({time.time()-t0:.1f}s)')
    if reused:
        log(f'{reused}/{N_FOLDS} folds reused from cache')
    return meta


# =====================================================  calibration
def micro_rejection(folds, key, A):
    rej = n = 0
    for d in folds:
        fit = d['oof_fit'].astype(bool)
        tau = -float(np.quantile(d['oof_d1'][fit], A))
        s = -d[key]
        u = d['unknown'].astype(bool)
        rej += int((s[u] < tau).sum())
        n += int(u.sum())
    return rej / max(n, 1)


def calibrate(folds, key):
    lo, hi = 1e-4, 1.0 - 1e-9
    f_lo, f_hi = micro_rejection(folds, key, lo), micro_rejection(folds, key, hi)
    if f_lo < TARGET:
        return dict(A=float(lo), rejection=float(f_lo), reached=False,
                    note='even the strictest A cannot reach the LOSO target')
    if f_hi >= TARGET:
        return dict(A=float(hi), rejection=float(f_hi), reached=True,
                    note='the loosest A still clears the LOSO target')
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if micro_rejection(folds, key, mid) >= TARGET:
            lo = mid
        else:
            hi = mid
    return dict(A=float(lo), rejection=float(micro_rejection(folds, key, lo)),
                reached=True, note='bisection to the largest A that still clears the target')


def loso_readout(folds, key, A):
    per_fold = dict(unknown_rejection=[], known_acceptance=[], auc=[], tau=[])
    rej_u = n_u = acc_k = n_k = 0
    pt = {t: dict(rej_u=0, n_u=0, acc_k=0, n_k=0) for t in TASKS}
    for d in folds:
        fit = d['oof_fit'].astype(bool)
        tau = -float(np.quantile(d['oof_d1'][fit], A))
        s = -d[key]
        known, unknown = d['known'].astype(bool), d['unknown'].astype(bool)
        tasks = np.asarray(d['task'], dtype=str)
        r_u = s[unknown] < tau
        r_k = s[known] < tau
        per_fold['unknown_rejection'].append(float(r_u.mean()))
        per_fold['known_acceptance'].append(float((~r_k).mean()))
        per_fold['auc'].append(auc_ties(s[known], s[unknown]))
        per_fold['tau'].append(float(tau))
        rej_u += int(r_u.sum())
        n_u += int(unknown.sum())
        acc_k += int((~r_k).sum())
        n_k += int(known.sum())
        for t in TASKS:
            ku, kk = unknown & (tasks == t), known & (tasks == t)
            pt[t]['rej_u'] += int((s[ku] < tau).sum())
            pt[t]['n_u'] += int(ku.sum())
            pt[t]['acc_k'] += int((s[kk] >= tau).sum())
            pt[t]['n_k'] += int(kk.sum())
    return dict(
        A=float(A),
        unknown_rejection_micro=float(rej_u / max(n_u, 1)),
        unknown_rejection_macro=float(np.mean(per_fold['unknown_rejection'])),
        unknown_rejection_fold_mean=float(np.mean(per_fold['unknown_rejection'])),
        unknown_rejection_fold_std=float(np.std(per_fold['unknown_rejection'])),
        known_acceptance_micro=float(acc_k / max(n_k, 1)),
        known_acceptance_fold_mean=float(np.mean(per_fold['known_acceptance'])),
        known_acceptance_fold_std=float(np.std(per_fold['known_acceptance'])),
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


# =====================================================  test side
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


def cell_stats(z, mask, tasks, rows):
    out = {}
    for t in TASKS:
        rr = rows[mask & (tasks == t)]
        A = z[rr].astype(np.float64)
        n = len(A)
        mu = A.mean(0) if n else np.zeros(z.shape[1])
        out[t] = dict(n=int(n), mu=mu)
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


def load_features(split):
    store = {}
    for b in BLOCK_ORDER:
        p = ROOT / CFG['feature_cache'][split][b]
        arr = np.load(p).astype(np.float32)
        assert arr.shape[1] == BLOCK_DIM[b], (b, arr.shape)
        store[b] = arr
    return np.concatenate([store[b] for b in BLOCK_ORDER], 1)


# =====================================================  main
def main():
    global TASKS_NP, Y, FR, BLOCKS, TE_TASK, IS_S1, S2, Xtr, Xte
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available: E235 is blocked'
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
    TE_TASK = te.task.to_numpy()
    sess = te.session.astype(str).str.casefold().to_numpy()
    IS_S1 = np.isin(sess, ['session1', 's1', '1'])
    S2 = ~IS_S1
    masks = {'session1': IS_S1, 'session2': S2}
    log(f'train rows={len(tr)} fit={int(fit_all.sum())} '
        f'window={int(((FR > EVAL_LO) & (FR <= EVAL_HI)).sum())} '
        f'| test S1={int(IS_S1.sum())} S2={int(S2.sum())}')

    Xtr = load_features('train')
    Xte = load_features('test')
    log(f'features train {Xtr.shape} test {Xte.shape}')

    # ---------------- frozen Test scores (unaligned + E233 mean align)
    ztr = transform(state, Xtr, TASKS_NP)
    zte = transform(state, Xte, TE_TASK)
    d1_raw, pred_raw = d1_rows(state, zte, TE_TASK, np.arange(len(TE_TASK)))
    dte = np.load(ROOT / CFG['feature_cache']['frozen_test_cache'], allow_pickle=True)
    checks = {}
    checks['test_score_reproduction'] = dict(
        max_abs_diff=float(np.max(np.abs(-d1_raw - dte['score'].astype(np.float64)))),
        pred_identical=bool(np.array_equal(pred_raw, dte['pred'].astype(int))))
    log(f"test score reproduction: max|d|="
        f"{checks['test_score_reproduction']['max_abs_diff']:.2e} "
        f"pred_identical={checks['test_score_reproduction']['pred_identical']}")
    assert checks['test_score_reproduction']['max_abs_diff'] < 1e-4

    ref = {t: dict(n=int((fit_all & (TASKS_NP == t)).sum()),
                   mu=ztr[fit_all & (TASKS_NP == t)].astype(np.float64).mean(0))
           for t in TASKS}
    cells_b = {s: cell_stats(zte, m, TE_TASK, np.arange(len(TE_TASK))) for s, m in masks.items()}
    z_al = apply_mean_align(zte, masks, cells_b, ref, TE_TASK)
    d1_al, pred_al = d1_rows(state, z_al, TE_TASK, np.arange(len(TE_TASK)))
    score_al, score_raw = -d1_al, -d1_raw

    # purity partition, fixed in the unaligned PCA space (identical across candidates)
    s2_rows = np.flatnonzero(S2)
    assigns = {}
    for k in PURITY_K:
        t0 = time.time()
        assigns[k], sizes = kmeans_assign(zte[s2_rows].astype(np.float32), k, PURITY_SEED + k)
        log(f'kmeans k={k} on {len(s2_rows)} S2 rows ({time.time()-t0:.1f}s) '
            f'sizes min/max={sizes.min()}/{sizes.max()}')

    # ---------------- delta estimation
    delta_A = delta_transductive(ztr, zte)
    delta_A_fit = delta_transductive_fitref(ztr, zte, fit_all)
    drift_meta = load_or_extract_drift(tr, fit_all)
    checks['clean_reextraction_vs_cache'] = feature_reproduction_check()
    log('clean re-extraction vs cache: ' + json.dumps(
        {k: (round(v['max_abs'], 6), round(v['mean_abs'], 6))
         for k, v in checks['clean_reextraction_vs_cache'].items() if isinstance(v, dict)}))
    delta_B = {name: delta_synthetic(state, name) for name in PROFILES}
    delta_B_comp = {name: delta_synthetic(state, name) for name in COMPONENT_PROFILES}
    delta_A_primary = delta_A
    delta_B_primary = delta_B[PRIMARY_PROFILE]

    ident = identity_directions(ztr, fit_all)
    diag_A = delta_identity_diagnostic(delta_A, ident)
    diag_B = delta_identity_diagnostic(delta_B_primary, ident)
    cos_pool_A = pooled_identity_cosine(delta_A, ident)
    cos_pool_B = pooled_identity_cosine(delta_B_primary, ident)
    half_A = split_half_stability(zte, delta_A)
    log(f'delta norms A=' + json.dumps({t: round(float(np.linalg.norm(delta_A[t])), 3) for t in TASKS})
        + ' B=' + json.dumps({t: round(float(np.linalg.norm(delta_B_primary[t])), 3) for t in TASKS}))
    log(f'cos(delta,identity v1) A=' + json.dumps({t: round(diag_A[t]['cos_identity_v1'], 4) for t in TASKS})
        + f' pooled={cos_pool_A:.4f} | B=' + json.dumps(
            {t: round(diag_B[t]['cos_identity_v1'], 4) for t in TASKS}) + f' pooled={cos_pool_B:.4f}')

    # ---------------- LOSO folds
    table = block_only_auc_table()
    fold_meta = run_folds(table)
    folds = [load_fold(h) for h in range(1, N_FOLDS + 1)]

    # verification of the fold reproduction (per fold, from the cache)
    diff = [float(np.max(np.abs(-np.load(ROOT / CFG['feature_cache']['frozen_fold_cache']
                                        .format(fold=h), allow_pickle=True)['score']
                              .astype(np.float64) - d['d1_orig'])))
            for h, d in enumerate(folds, start=1)]
    checks['fold_d1_vs_E227'] = dict(max_abs_diff=float(np.max(diff)),
                                     per_fold_last=[float(x) for x in diff[-3:]])
    log(f"fold reproduction: max|d1 - E227| = {np.max(diff):.2e}")
    assert np.max(diff) < 5e-4, 'fold d1 drifted from E227'

    # order-statistic variant: one scalar shift of the impostor scores
    drops = [(d['d1_A'][d['unknown'].astype(bool)] - d['d1_orig'][d['unknown'].astype(bool)])
             for d in folds]
    delta_order = float(np.concatenate(drops).mean())
    for d in folds:
        d['d1_order'] = d['d1_orig'] + delta_order
    log(f'order-statistic shift Delta_order = {delta_order:+.6f} '
        f'(score_shift = score - Delta_order)')

    # effective per-task score displacement of the shifted impostors
    eff = {}
    for t in TASKS:
        vals_A, vals_B = [], []
        for d in folds:
            u = d['unknown'].astype(bool) & (np.asarray(d['task'], dtype=str) == t)
            if u.any():
                vals_A.append(d['d1_A'][u] - d['d1_orig'][u])
                vals_B.append(d['d1_B'][u] - d['d1_orig'][u])
        eff[t] = dict(delta_eff_A=float(np.concatenate(vals_A).mean()) if vals_A else float('nan'),
                      delta_eff_B=float(np.concatenate(vals_B).mean()) if vals_B else float('nan'))
    log('effective score shift per task (score_shift - score_orig): ' + json.dumps(
        {t: (round(eff[t]['delta_eff_A'], 3), round(eff[t]['delta_eff_B'], 3)) for t in TASKS}))

    # ---------------- deploy inner-OOF (tau source), verbatim E234
    oof_d1 = np.full(len(Y), np.nan)
    for b in np.unique(BLOCKS[fit_all]):
        hold = fit_all & (BLOCKS == b)
        inner = fit_all & ~hold
        sti = fit_rep(Xtr, TASKS_NP, Y, inner, K_PROTO, PCA_DIM, state.get('w'))
        zi = transform(sti, Xtr, TASKS_NP)
        rr = np.flatnonzero(hold)
        a1, _ = d1_rows(sti, zi, TASKS_NP, rr)
        oof_d1[rr] = a1[rr]
    checks['deploy_oof_d1_vs_E227'] = dict(
        max_abs_diff=float(np.max(np.abs(
            -dte['oof_s'].astype(np.float64)[fit_all] - oof_d1[fit_all]))))
    log(f"deploy inner-OOF d1 vs E227 oof_s: "
        f"max|d|={checks['deploy_oof_d1_vs_E227']['max_abs_diff']:.2e}")

    def deploy_tau(A):
        return -float(np.quantile(oof_d1[fit_all], A))

    # ---------------- calibration per candidate
    keys = dict(tau_orig='d1_orig', tau_shift_A='d1_A', tau_shift_B='d1_B',
                tau_shift_A_half='d1_Ahalf', tau_shift_order_stats='d1_order')
    calib, loso_orig, loso_shifted, taus = {}, {}, {}, {}
    for c in CANDS:
        cid, key = c['id'], keys[c['id']]
        cal = calibrate(folds, key)
        calib[cid] = cal
        taus[cid] = deploy_tau(cal['A'])
        loso_orig[cid] = loso_readout(folds, 'd1_orig', cal['A'])
        loso_shifted[cid] = loso_readout(folds, key, cal['A'])
        log(f'calibrate {cid:22s} A*={cal["A"]:.9f} shifted_LOSO={cal["rejection"]:.6f} '
            f'tau={taus[cid]:.6f} | unshifted_LOSO={loso_orig[cid]["unknown_rejection_micro"]:.6f} '
            f'known_acc={loso_orig[cid]["known_acceptance_micro"]:.4f}')

    checks['abs_dist_calibration_vs_E228'] = dict(
        A_measured=float(calib['tau_orig']['A']), A_reference=A_REF,
        abs_diff=float(abs(calib['tau_orig']['A'] - A_REF)),
        loso_measured=float(loso_orig['tau_orig']['unknown_rejection_micro']),
        loso_reference=LOSO_REF,
        loso_abs_diff=float(abs(loso_orig['tau_orig']['unknown_rejection_micro'] - LOSO_REF)),
        tau_measured=float(taus['tau_orig']), tau_reference=TAU_REF,
        tau_abs_diff=float(abs(taus['tau_orig'] - TAU_REF)),
        fold_tau_mean=float(loso_orig['tau_orig']['tau_mean']), fold_tau_reference=FOLD_TAU_REF,
        fold_tau_abs_diff=float(abs(loso_orig['tau_orig']['tau_mean'] - FOLD_TAU_REF)),
        loso_auc=float(loso_orig['tau_orig']['loso_auc']), loso_auc_reference=LOSO_AUC_REF,
        loso_auc_abs_diff=float(abs(loso_orig['tau_orig']['loso_auc'] - LOSO_AUC_REF)))
    log('tau_orig calibration vs E228: ' + json.dumps(checks['abs_dist_calibration_vs_E228']))

    # ---------------- Test read-out per candidate
    results = {}
    for c in CANDS:
        cid = c['id']
        tau = taus[cid]
        m = readout(score_al, pred_al, IS_S1, TE_TASK, tau)
        m['tau_delta_vs_orig'] = float(tau - taus['tau_orig'])
        m['mean_score_S1'] = float(score_al[IS_S1].mean())
        m['mean_score_S2'] = float(score_al[S2].mean())
        m['S1_score_min'] = float(score_al[IS_S1].min())
        m['S1_score_max'] = float(score_al[IS_S1].max())
        m['S1_score_p95'] = float(np.quantile(score_al[IS_S1], 0.95))
        m['score_pooled_sd'] = float(score_al.std())
        g = readout(score_al, pred_al, IS_S1, TE_TASK, float(np.quantile(score_al[IS_S1], S1_GATE)))
        m['gate_S1_0p95'] = dict(tau=g['tau'], S1_rejection=g['S1_rejection'],
                                 S2_acceptance=g['S2_acceptance'],
                                 separation_auc=g['separation_auc'])
        m['purity'] = {f'k{k}': purity(pred_al[s2_rows], assigns[k], k) for k in PURITY_K}
        results[cid] = dict(id=cid, delta=c['delta'], role=c.get('role', ''), tau=float(tau),
                            calibration=calib[cid], loso_shifted=loso_shifted[cid],
                            loso_original=loso_orig[cid], metrics=m)

    # reproduction of the E233 compliant point
    am = results['tau_orig']['metrics']
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
    log('E233 reproduction: ' + json.dumps(
        {k: (round(v['value'], 6), round(v['abs_diff'], 8))
         for k, v in checks['e233_mean_align_reproduction'].items()}))
    assert abs(am['S2_acceptance'] - E233_S2) < 1e-6

    # delta diagnostics that need the folds
    out_cos = impostor_outward_cosine(delta_A, folds)
    fold_delta_norms = {t: [float(np.linalg.norm(d['delta_A'][i])) for d in folds
                            for i, tt in enumerate(TASKS) if tt == t] for t in TASKS}

    # ---------------- decision
    base_pur = results['tau_orig']['metrics']['purity']

    def purity_ok(m):
        return all(m['purity'][f'k{k}']['purity_micro'] >=
                   PURITY_TOL * base_pur[f'k{k}']['purity_micro'] for k in PURITY_K)

    cos_of = {'tau_orig': 0.0, 'tau_shift_A': cos_pool_A, 'tau_shift_B': cos_pool_B,
              'tau_shift_A_half': cos_pool_A, 'tau_shift_order_stats': cos_pool_A}
    for cid, r in results.items():
        m = r['metrics']
        r['cos_delta_identity'] = float(cos_of[cid])
        r['meets_gate_S1_and_S2'] = bool(m['S1_rejection'] >= S1_GATE
                                         and m['S2_acceptance'] >= USER_S2_TARGET)
        r['purity_ok'] = bool(purity_ok(m))
        r['eligible'] = bool(r['meets_gate_S1_and_S2'] and r['purity_ok']
                             and abs(r['cos_delta_identity']) < COS_LIMIT)

    eligible = [cid for cid, r in results.items() if r['eligible']]
    hits = [cid for cid, r in results.items() if r['meets_gate_S1_and_S2']]
    improve = [cid for cid, r in results.items()
               if r['metrics']['S2_acceptance'] > E233_S2 + 1e-6]
    s1_ok = [cid for cid, r in results.items() if r['metrics']['S1_rejection'] >= S1_GATE]
    best = max(results, key=lambda c: results[c]['metrics']['S2_acceptance'])
    best_s1_ok = max(s1_ok, key=lambda c: results[c]['metrics']['S2_acceptance']) if s1_ok else None
    if eligible:
        adopted = max(eligible, key=lambda c: results[c]['metrics']['S2_acceptance'])
        verdict = 'pass'
    elif hits:
        adopted = None
        leak = [c for c in hits if abs(results[c]['cos_delta_identity']) >= COS_LIMIT]
        verdict = 'identity_leak' if leak else 'trade_off'
    elif improve:
        adopted = None
        verdict = 'trade_off'
    else:
        adopted = None
        verdict = 'fail'
    log(f'decision verdict={verdict} adopted={adopted} best={best} '
        f'S2={results[best]["metrics"]["S2_acceptance"]:.4f} '
        f'S1={results[best]["metrics"]["S1_rejection"]:.4f}')

    # the operating curve is fixed by the frozen score: report the point that satisfies S1 >= 0.95
    gate_tau = float(results['tau_orig']['metrics']['gate_S1_0p95']['tau'])
    gate_s2 = float(results['tau_orig']['metrics']['gate_S1_0p95']['S2_acceptance'])
    log(f'S1=0.95 operating point: tau={gate_tau:.4f} S2={gate_s2:.4f} '
        f'(requires dtau={gate_tau - taus["tau_orig"]:+.4f} vs tau_orig)')

    # ---------------- artifacts
    tau_table = {cid: dict(tau=float(taus[cid]), A=float(calib[cid]['A']),
                           delta=c['delta'], role=c.get('role', '')) for c in CANDS}
    torch.save(dict(experiment='E235_deployment_calibration',
                    base_model=str(ROOT / CFG['feature_cache']['frozen_model']),
                    scoring=CFG['scoring'], taus=tau_table,
                    delta_A={t: delta_A[t] for t in TASKS},
                    delta_B={name: delta_B[name] for name in PROFILES},
                    delta_A_fitref={t: delta_A_fit[t] for t in TASKS},
                    identity_direction={t: dict(v1=ident[t]['v1'], top5=ident[t]['top5'],
                                                eig_share=ident[t]['eig_share']) for t in TASKS},
                    diagnostics=dict(cos_pool_A=cos_pool_A, cos_pool_B=cos_pool_B,
                                     diag_A=diag_A, diag_B=diag_B,
                                     delta_order=delta_order),
                    test_reference=ref,
                    verdict=verdict, adopted=adopted,
                    note='threshold table only; the deploy model state is the frozen E227 one'),
               MODELS / 'e235_deployment_calibration.pt')

    results_out = dict(
        experiment='E235_deployment_calibration',
        protocol=CFG['representation']['pipeline'],
        scoring=CFG['scoring'],
        delta_config=CFG['delta'],
        threshold=CFG['threshold'],
        checks=checks,
        delta=dict(
            A={t: dict(norm=float(np.linalg.norm(delta_A[t])),
                       cos_identity_v1=diag_A[t]['cos_identity_v1'],
                       share_top5_identity_subspace=diag_A[t]['share_top5_identity_subspace'],
                       split_half=half_A[t]) for t in TASKS},
            A_fitref_norm={t: float(np.linalg.norm(delta_A_fit[t])) for t in TASKS},
            B={name: {t: dict(norm=float(np.linalg.norm(delta_B[name][t])),
                              cos_identity_v1=diag_B[t]['cos_identity_v1'] if
                              name == PRIMARY_PROFILE else None) for t in TASKS}
               for name in PROFILES},
            B_components={name: {t: float(np.linalg.norm(delta_B_comp[name][t]))
                                 for t in TASKS} for name in COMPONENT_PROFILES},
            cos_pool_A=cos_pool_A, cos_pool_B=cos_pool_B,
            impostor_outward_cosine_A=out_cos,
            fold_delta_A_norm_mean={t: float(np.mean(fold_delta_norms[t])) for t in TASKS},
            effective_score_shift=eff,
            order_statistic_shift=delta_order,
            identity_direction=dict(
                eigen_share_top5={t: ident[t]['eig_share'] for t in TASKS},
                n_fit={t: ident[t]['n'] for t in TASKS})),
        drift_subset=drift_meta,
        candidates=results,
        purity_baseline=base_pur,
        fold_repro_meta=fold_meta,
        decision=dict(verdict=verdict, adopted=adopted, eligible=eligible,
                      meets_gate_S1_and_S2=hits, improves_over_e233=improve,
                      best=best,
                      best_S2=float(results[best]['metrics']['S2_acceptance']),
                      best_S1=float(results[best]['metrics']['S1_rejection']),
                      best_with_S1_ok=best_s1_ok,
                      best_S2_with_S1_ok=float(results[best_s1_ok]['metrics']['S2_acceptance'])
                      if best_s1_ok else None,
                      s1_0p95_gate_tau=gate_tau, s1_0p95_gate_S2=gate_s2,
                      s1_0p95_gate_dtau=float(gate_tau - taus['tau_orig']),
                      cos_pool_A=cos_pool_A, cos_pool_B=cos_pool_B,
                      cos_limit=COS_LIMIT, s1_gate=S1_GATE,
                      user_target_S2=USER_S2_TARGET, e233_S2=E233_S2,
                      e233_S2_at_S1_0p95=E233_S2_AT_GATE,
                      purity_tolerance=PURITY_TOL),
        deploy=None,
        official=dict(C='N/A', RA='N/A', ASafeR='N/A', note='Test has no identity labels.'),
        limitations=CFG['official_note'] + ' ' + CFG['bias_note'],
        runtime_s=float(time.time() - t_start))
    if adopted:
        results_out['deploy'] = dict(candidate=adopted,
                                     tau=taus[adopted],
                                     scoring='abs_dist + E233 mean_align',
                                     S1_rejection=results[adopted]['metrics']['S1_rejection'],
                                     S2_acceptance=results[adopted]['metrics']['S2_acceptance'],
                                     separation_auc=results[adopted]['metrics']['separation_auc'],
                                     loso_shifted=results[adopted]['loso_shifted']
                                     ['unknown_rejection_micro'],
                                     transductive=bool(results[adopted]['delta'] != 'none'),
                                     note='reject iff score < tau in the mean-aligned PCA space')
    (OUT / 'results.json').write_text(json.dumps(results_out, indent=2), encoding='utf8')
    build_report(results_out)
    log('results.json + report.md written')
    log(f'verdict={verdict} runtime={time.time()-t_start:.0f}s')


# =====================================================  report
def f4(x, n=4):
    if x is None:
        return '—'
    if isinstance(x, float) and np.isnan(x):
        return '—'
    return f'{x:.{n}f}'


def build_report(R):
    C = R['candidates']
    dec = R['decision']
    D = R['delta']
    chk = R['checks']
    order = [c['id'] for c in CANDS]
    base = C['tau_orig']['metrics']
    tau0 = C['tau_orig']['tau']

    best = dec['best']
    L = ['# E235 部署条件阈值标定：把冒充者平移到部署域再定阈值', '',
         '> **一句话结论**：' + (
             f"`{dec['adopted']}` 达标并冻结：Test S1 {f4(C[dec['adopted']]['metrics']['S1_rejection'])} ≥ 0.95、"
             f"Test S2 **{f4(C[dec['adopted']]['metrics']['S2_acceptance'])}** ≥ 0.60，"
             f"且 cos(δ, 身份方向) = {dec['cos_pool_A']:+.3f} < 0.70。"
             if dec['verdict'] == 'pass' else
             (f"**数值上到了 0.60，但 δ 与身份判别方向的余弦 = {dec['cos_pool_A']:+.3f}（≥ 0.70），"
              f"该阈值位置实质是在搬运身份信息，判定不可采纳，未冻结。**"
              if dec['verdict'] == 'identity_leak' else
              (f"**S2 抬得动，但抬不到 0.60**：`tau_shift_A` 用 transductive δ 把 Test S2 从 "
               f"{f4(E233_S2)} 抬到 **{f4(C['tau_shift_A']['metrics']['S2_acceptance'])}**"
               f"（S1 仍 {f4(C['tau_shift_A']['metrics']['S1_rejection'])}）；`tau_shift_B` 用导联级 δ 把 S2 推到 "
               f"{f4(C['tau_shift_B']['metrics']['S2_acceptance'])}，代价是 S1 掉到 "
               f"{f4(C['tau_shift_B']['metrics']['S1_rejection'])}。"
               f"要同时满足 S1=0.95 需要 Δτ = {dec['s1_0p95_gate_dtau']:+.2f}"
               f"（对应 S2 = {f4(dec['s1_0p95_gate_S2'])}），而 δ_A 只给出 "
               f"{C['tau_shift_A']['metrics']['tau_delta_vs_orig']:+.2f}"
               f"（标量版 {C['tau_shift_order_stats']['metrics']['tau_delta_vs_orig']:+.2f}）的位移。"
               if dec['verdict'] == 'trade_off' else
               f"**δ 标定没有移动工作点**：最优候选 `{best}` 的 Test S2 = {f4(dec['best_S2'])}，"
               f"不高于 E233 合规值 {f4(E233_S2)}。"))), '',
         f"做法：冻结 E233 的表征、原型、均值对齐与 Test 分数，**只改阈值的标定目标**——"
         f"把每折留出身份冒充者按跨天偏移 δ 平移到「部署条件」（`z + δ_task`），再把 τ 二分到对这些"
         f"已平移冒充者的 LOSO 拒识 = 0.90。对照 `tau_orig` 在未平移冒充者上做同一件事。", '',
         '## 1. 复现校验（本次运行内真实计算）', '',
         '|项目|实测|参考|绝对差|判定|', '|---|---:|---:|---:|:---:|']
    rep = [
        ('Test 分数复算（未对齐，max abs）', chk['test_score_reproduction']['max_abs_diff'], 0.0, 1e-4),
        ('逐折 d1 vs E227（max abs）', chk['fold_d1_vs_E227']['max_abs_diff'], 0.0, 5e-4),
        ('部署模型 inner-OOF d1 vs E227 oof_s', chk['deploy_oof_d1_vs_E227']['max_abs_diff'], 0.0, 5e-4),
        ('A*（tau_orig）', chk['abs_dist_calibration_vs_E228']['A_measured'], A_REF, 1e-6),
        ('pooled LOSO 拒识（tau_orig）', chk['abs_dist_calibration_vs_E228']['loso_measured'], LOSO_REF, 1e-9),
        ('部署 tau（tau_orig）', chk['abs_dist_calibration_vs_E228']['tau_measured'], TAU_REF, 1e-5),
        ('折级 tau 均值（tau_orig）', chk['abs_dist_calibration_vs_E228']['fold_tau_mean'],
         FOLD_TAU_REF, 1e-6),
        ('LOSO AUC（tau_orig）', chk['abs_dist_calibration_vs_E228']['loso_auc'], LOSO_AUC_REF, 1e-6),
        ('E233 S2', chk['e233_mean_align_reproduction']['S2_acceptance']['value'], E233_S2, 1e-6),
        ('E233 S1', chk['e233_mean_align_reproduction']['S1_rejection']['value'], E233_S1, 1e-9),
        ('E233 sepAUC', chk['e233_mean_align_reproduction']['separation_auc']['value'], E233_AUC, 1e-6),
        ('E233 纯度 k20', chk['e233_mean_align_reproduction']['purity_k20']['value'], E233_PUR['k20'], 1e-6),
        ('E233 纯度 k28', chk['e233_mean_align_reproduction']['purity_k28']['value'], E233_PUR['k28'], 1e-6),
    ]
    for name, val, refv, tol in rep:
        ok = abs(val - refv) <= tol
        L.append(f"|{name}|{val:.10f}|{refv:.10f}|{abs(val - refv):.2e}|{'通过' if ok else '**未通过**'}|")
    L += ['', '`tau_orig` 与 E228 同口径（同一 A 二分、同一 (1−A) 分位规则），因此它的 Test 读数就是 E233 的合规点'
          f"（S2 {f4(base['S2_acceptance'])} / S1 {f4(base['S1_rejection'])} / sepAUC "
          f"{f4(base['separation_auc'])}）。", '',
          '## 2. δ 的估计（task 级范数与稳定性）', '',
          '`δ_A` = 变体 A（**transductive**）：Test session2 全样本均值 − Train 全样本均值；'
          '`δ_A(fitref)` = 参考改用 Train fit 行（与均值对齐的参考一致）；'
          '`δ_B` = 变体 B（**纯 Train-only**）：在原始波形上施加声明好的漂移剖面后重提 10 个 block，再取均值差。'
          '`cos(½)` = 用 session2 的两个随机半样本各自重估 δ 的方向余弦（50 次均值 / 最小值）；'
          '`rel diff` = ‖δ_a − δ_b‖ / ‖δ‖。', '',
          '|task|‖δ_A‖|‖δ_A(fitref)‖|‖δ_B(P1)‖|‖δ_B(P2)‖|‖δ_B(P3)‖|cos(½) 均值[最小]|rel diff|',
          '|---|---:|---:|---:|---:|---:|---|---:|']
    for t in TASKS:
        h = D['A'][t]['split_half']
        L.append(f"|{t}|{f4(D['A'][t]['norm'], 3)}|{f4(D['A_fitref_norm'][t], 3)}|"
                 f"{f4(D['B']['P1_gain_tilt_noise_reref'][t]['norm'], 3)}|"
                 f"{f4(D['B']['P2_tilt_pos_reref'][t]['norm'], 3)}|"
                 f"{f4(D['B']['P3_tilt_neg_gain'][t]['norm'], 3)}|"
                 f"{h['half_cosine_mean']:+.3f}[{h['half_cosine_min']:+.3f}]|"
                 f"{h['half_rel_diff_mean']:.3f}|")
    nA = float(np.mean([D['A'][t]['norm'] for t in TASKS]))
    nB = float(np.mean([D['B']['P1_gain_tilt_noise_reref'][t]['norm'] for t in TASKS]))
    comp = D['B_components']
    L += ['', '合成漂移的单项分解（每项单独施加后的 ‖δ‖ 均值，诊断用，不构成候选）：', '',
          '|分量|' + '|'.join(TASKS) + '|均值|', '|---|' + '---:|' * (len(TASKS) + 1)]
    for name in comp:
        vals = [comp[name][t] for t in TASKS]
        L.append(f"|{name}|" + '|'.join(f'{v:.3f}' for v in vals) + f"|{np.mean(vals):.3f}|")
    L += ['', f"- δ_A 的方向是**稳定的**（分半余弦均值普遍接近 1），但它主要是「这一批恰好是谁 + 这一天的批次平移」的合成量，"
          f"与 E233 第 4 节的结论一致。",
          (f"- **量级：δ_B(P1) 远大于 δ_A**（{nB:.2f} vs {nA:.2f}）。上表的单项分解表明四项 nuisance "
           f"任何一项单独施加就已经产生 11–26 的位移，而**真实**测得的跨天偏移只有 {nA:.2f}——"
           f"即这套声明好的合成漂移在冻结 PCA 空间里被放大了数倍，没有任何一项能复现观测到的偏移量级。"
           f"原因是这些特征经 task-wise 标准化后残余敏感度被放大（例如 `raw − ½(ch0+ch1)` 让通道 0/1 完全反相关，"
           f"logSPD 被推到极值），所以合成剖面**无法给出可信的偏移量级**，只能当作上界式扰动。"
           if nB > 2 * nA else
           f"- **量级**：‖δ_A‖ 平均 {nA:.2f}，合成漂移 ‖δ_B(P1)‖ 平均 {nB:.2f}；"
           "合成的增益/倾斜/噪声/重参考大部分被这套特征按设计吸收（gain-free 带功率、沿频率去均值的谱形状、"
           "通道标准化相关系数），Train-only 的漂移模型复现不出测试域的跨天偏移量级。"),
          "- 因此按原计划，`tau_shift_B` 只能当作「纯 Train-only 的上界式扰动」来读：它的阈值位移方向可信、量级不可信。",
          f"- 干净波形重提特征 vs 冻结缓存的最大绝对差见 `results.json:checks.clean_reextraction_vs_cache`"
          f"（最大 {max(v['max_abs'] for v in chk['clean_reextraction_vs_cache'].values() if isinstance(v, dict)):.3e}），"
          "说明 δ_B 不是由特征实现差异造成的。", '',
          '## 3. δ 是否在搬运身份信息（本轮的判据）', '',
          '身份方向 = Train fit 行上类间散度矩阵 S_b 的最大特征向量 v1（逐 task）。'
          '`share top5` = δ 的能量落在 v1..v5 子空间内的比例。'
          '`cos(outward)` = δ 与「冒充者 − 最近原型」单位方向均值的余弦（正值表示 δ 把冒充者推得更远）。'
          '余弦一律取**绝对值**：特征向量的符号不可辨识。', '',
          '|task|cos(δ_A, v1)|share top5|cos(δ_B, v1)|cos(δ_A, outward)|‖δ_A‖|',
          '|---|---:|---:|---:|---:|---:|']
    for t in TASKS:
        L.append(f"|{t}|**{D['A'][t]['cos_identity_v1']:+.4f}**|"
                 f"{D['A'][t]['share_top5_identity_subspace']:.4f}|"
                 f"{D['B']['P1_gain_tilt_noise_reref'][t]['cos_identity_v1']:+.4f}|"
                 f"{D['impostor_outward_cosine_A'][t]:+.4f}|{f4(D['A'][t]['norm'], 3)}|")
    L += ['', f"- **池化余弦（拼接各 task 的 δ 与 v1）**：δ_A = **{D['cos_pool_A']:+.4f}**，"
          f"δ_B = {D['cos_pool_B']:+.4f}；判定阈值为 |cos| ≥ {COS_LIMIT:.2f}。",
          (f"- δ_A 与身份方向**高度重合**：把冒充者沿 δ 平移，等于把「身份」这一维度整体挪动，"
           f"因此得到的阈值下移并不是「补偿跨天漂移」，而是把一个与身份可分的量当成了 nuisance。"
           if abs(D['cos_pool_A']) >= COS_LIMIT else
           f"- δ_A 与身份方向的余弦低于 {COS_LIMIT:.2f}，δ 不主要表现为身份方向上的位移；"
           f"但它与冒充者外移方向的余弦（见上表）决定分数被平移了多少。"),
          f"- 参考量级：cos(δ_A, outward) 的符号决定冒充者被推远（正）还是拉近（负），"
          f"对应的有效分数平移见第 4 节 `Δ_eff`。", '',
          '## 4. 主表：每个 τ 候选的读数', '',
          '`τ` = 该候选的部署阈值（`score < τ` 判未知）；`shifted LOSO` = 标定目标本身（平移后的冒充者拒识，'
          '必然 ≈0.90）；`原始 LOSO` = 同一个 τ 在未平移冒充者上的拒识；**Test S1/S2** 在 E233 均值对齐的分数上读取；'
          '`纯度` 与 τ 无关（它评的是预测身份），列出以示不是退化解。', '',
          '|候选|δ|τ|Δτ vs τ_orig|shifted LOSO|原始 LOSO|Test S1 rej|**Test S2 acc**|sepAUC|纯度20/28|S1≥.95 & S2≥.60|',
          '|---|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|']
    for cid in order:
        r, m = C[cid], C[cid]['metrics']
        L.append(f"|{'**' if cid == dec['adopted'] else ''}{cid}{'**' if cid == dec['adopted'] else ''}|"
                 f"{r['delta']}|{r['tau']:.4f}|{m['tau_delta_vs_orig']:+.4f}|"
                 f"{f4(r['loso_shifted']['unknown_rejection_micro'])}|"
                 f"{f4(r['loso_original']['unknown_rejection_micro'])}|"
                 f"{f4(m['S1_rejection'])}|**{f4(m['S2_acceptance'])}**|{f4(m['separation_auc'])}|"
                 f"{f4(m['purity']['k20']['purity_micro'])}/{f4(m['purity']['k28']['purity_micro'])}|"
                 f"{'是' if r['meets_gate_S1_and_S2'] else '否'}|")
    L += ['', f"**有效分数平移**（`Δ_eff` = 平移前后冒充者 d1 之差的 pooled 均值；正 = 冒充者被推远、分数下降）："
          + '、'.join(f"{t} A {D['effective_score_shift'][t]['delta_eff_A']:+.3f} / "
                      f"B {D['effective_score_shift'][t]['delta_eff_B']:+.3f}" for t in TASKS)
          + f"。`tau_shift_order_stats` 用一个标量 Δ_order = {D['order_statistic_shift']:+.4f} 取代逐 task 几何平移，"
          f"与 `tau_shift_A` 的 τ 差 {C['tau_shift_order_stats']['tau'] - C['tau_shift_A']['tau']:+.6f}，"
          '说明两种写法一致（差只来自逐 task/逐行的几何结构）。', '',
          f"**S1=0.95 的操作点由冻结分数决定、与候选无关**：τ = {dec['s1_0p95_gate_tau']:.4f}"
          f"（S1 分数的 95% 分位，`score < τ` 判未知），S2 = **{f4(dec['s1_0p95_gate_S2'])}**；"
          f"相对 τ_orig 需要 Δτ = {dec['s1_0p95_gate_dtau']:+.4f}。S1 分数的最小/最大为 "
          f"{C['tau_orig']['metrics']['S1_score_min']:.2f} / {C['tau_orig']['metrics']['S1_score_max']:.2f}，"
          f"分数总体标准差 {C['tau_orig']['metrics']['score_pooled_sd']:.2f}，"
          f"所以 τ 从 τ_orig 继续下移 {C['tau_orig']['tau'] - C['tau_orig']['metrics']['S1_score_max']:.2f} "
          "都不会损失 S1——阈值位移的量级才是本轮的瓶颈。", '']
    L += ['**机理（与判定分支无关，必须说清）**：', '',
          f"1. 本轮唯一的自由度是阈值位置。`tau_shift_A` 的 τ 比 `tau_orig` 低 "
          f"{base['tau'] - C['tau_shift_A']['tau']:.4f}，`tau_shift_B` 低 "
          f"{base['tau'] - C['tau_shift_B']['tau']:.4f}，`tau_shift_A_half` 低 "
          f"{base['tau'] - C['tau_shift_A_half']['tau']:.4f}；对应 Test S2 "
          f"{f4(base['S2_acceptance'])} → {f4(C['tau_shift_A']['metrics']['S2_acceptance'])} / "
          f"{f4(C['tau_shift_B']['metrics']['S2_acceptance'])} / "
          f"{f4(C['tau_shift_A_half']['metrics']['S2_acceptance'])}，Test S1 "
          f"{f4(base['S1_rejection'])} → {f4(C['tau_shift_A']['metrics']['S1_rejection'])} / "
          f"{f4(C['tau_shift_B']['metrics']['S1_rejection'])} / "
          f"{f4(C['tau_shift_A_half']['metrics']['S1_rejection'])}。",
          "2. **均值对齐会把同批次的加性偏移精确约掉**：本管线对 S2 打分前执行 "
          "`z − mean(batch) + mean_train`。若把平移后的冒充者也按同一条规则对齐，"
          "`z + δ − δ = z`，分数逐位回到未平移的值，标定出的 τ 与 `tau_orig` 完全相同、"
          f"S2 回到 {f4(base['S2_acceptance'])}。因此本轮的 S2 变化**全部来自阈值下移这一动作**，"
          "分数本身没有变得更有区分力（sepAUC 在所有候选中完全相同，因为它是阈值无关量）。",
          f"3. δ_A 的池化身份余弦 {D['cos_pool_A']:+.4f} < {COS_LIMIT:.2f}，"
          "说明 δ_A **不沿身份主轴**，user 规则里「沿身份方向搬运」这条作弊路径不成立；"
          "但这并不推翻第 2 条：它依然是一个无法从 Train 推出的、由 Test 无标签均值决定的阈值位置。",
          f"4. δ_B 是纯 Train-only 对照，量级 {nB:.2f} ≫ δ_A {nA:.2f}：四项合成扰动每项单独就已 11–26，"
          "远超观测到的跨天偏移，说明该剖面在冻结空间里被放大了数倍、只能当「上界式扰动」读——"
          f"它把 S2 推到 {f4(C['tau_shift_B']['metrics']['S2_acceptance'])}，同时把 S1 打到 "
          f"{f4(C['tau_shift_B']['metrics']['S1_rejection'])}，正好演示了「阈值下移换 S2」的代价。", '']

    L += ['## 5. 20 折稳定性与逐 task 读数', '',
          'Test 没有折划分，因此「分折 S2」在物理上不存在；下表给出每个 τ 下 20 折 LOSO 的'
          '**已知接受**（Test S2 在 Train 侧的对应量）与**未知拒识**的均值 ± std。', '',
          '|候选|LOSO 已知接受（20 折）|LOSO 未知拒识（原始）|LOSO 未知拒识（平移）|LOSO AUC|',
          '|---|---|---|---|---:|']
    for cid in order:
        lo, ls = C[cid]['loso_original'], C[cid]['loso_shifted']
        L.append(f"|{cid}|{lo['known_acceptance_fold_mean']:.4f} ± {lo['known_acceptance_fold_std']:.4f}|"
                 f"{lo['unknown_rejection_fold_mean']:.4f} ± {lo['unknown_rejection_fold_std']:.4f}|"
                 f"{ls['unknown_rejection_fold_mean']:.4f} ± {ls['unknown_rejection_fold_std']:.4f}|"
                 f"{f4(lo['loso_auc'])}|")
    L += ['', '逐 task（Test，冻结 τ）：', '',
          '|候选|task|S1 n|S1 rej|S2 n|S2 acc|LOSO 未知拒识|LOSO 已知接受|',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for cid in order:
        for t in TASKS:
            mt = C[cid]['metrics']['per_task'][t]
            lt = C[cid]['loso_original']['per_task'][t]
            L.append(f"|{cid}|{t}|{mt['n_S1']}|{f4(mt['S1_rejection'])}|{mt['n_S2']}|"
                     f"{f4(mt['S2_acceptance'])}|{f4(lt['unknown_rejection'])}|"
                     f"{f4(lt['known_acceptance'])}|")

    L += ['', '## 6. 判定', '']
    if dec['verdict'] == 'pass':
        m = C[dec['adopted']]['metrics']
        L += [f"- **达标并冻结** `{dec['adopted']}`：Test S1 {f4(m['S1_rejection'])} ≥ 0.95、"
              f"Test S2 **{f4(m['S2_acceptance'])}** ≥ 0.60、纯度 "
              f"{f4(m['purity']['k20']['purity_micro'])}/{f4(m['purity']['k28']['purity_micro'])} ≥ 基线 95%、"
              f"cos(δ, 身份方向) {dec['cos_pool_A']:+.3f} < 0.70。",
              f"- 部署参数：τ = {C[dec['adopted']]['tau']:.6f}，打分 = abs_dist + E233 均值对齐；"
              f"见 `models/e235_deployment_calibration.pt`。"]
    elif dec['verdict'] == 'identity_leak':
        L += [f"- **判定：该方法实质是在搬运身份信息，不可采纳，不冻结。** "
              f"依据：cos(δ_A, 身份判别方向 v1) 池化 = **{D['cos_pool_A']:+.4f}** ≥ {COS_LIMIT:.2f}；"
              f"且「平移后重新对齐 = 不平移」使标定目标恒等于 `tau_orig`（数学恒等），"
              f"说明 S2 的全部增益来自阈值下移，而非更好的分数。",
              f"- 数值记录（供对照，不作为部署参数）：`tau_shift_A` τ = {C['tau_shift_A']['tau']:.4f}，"
              f"Test S1 {f4(C['tau_shift_A']['metrics']['S1_rejection'])}、S2 "
              f"**{f4(C['tau_shift_A']['metrics']['S2_acceptance'])}**、sepAUC "
              f"{f4(C['tau_shift_A']['metrics']['separation_auc'])}；`tau_shift_A_half` S1 "
              f"{f4(C['tau_shift_A_half']['metrics']['S1_rejection'])}、S2 "
              f"{f4(C['tau_shift_A_half']['metrics']['S2_acceptance'])}。",
              f"- 合规最优仍是 E233：S1 {f4(E233_S1)} / S2 {f4(E233_S2)} / sepAUC {f4(E233_AUC)}。"
              f"若只想要 S1=0.95 下的最大 S2，E233 自身在 S1=0.95 门槛读法是 {f4(E233_S2_AT_GATE)}，"
              f"不需要引入 δ。"]
    elif dec['verdict'] == 'trade_off':
        L += [f"- **这是 S1/S2 权衡，不是免费达标，也没有到 0.60**。"
              f"五个候选的 Test 分数逐位相同（sepAUC 全是 {f4(C['tau_orig']['metrics']['separation_auc'])}），"
              f"它们只是同一条 ROC 上的五个点；本轮改变的是选点，不是曲线。",
              f"- 能把 S1 保持在 1.0 的最好候选是 `{dec['best_with_S1_ok']}`："
              f"S2 **{f4(dec['best_S2_with_S1_ok'])}**（E233 {f4(E233_S2)}，"
              f"{(dec['best_S2_with_S1_ok'] or 0) - E233_S2:+.4f}），S1 "
              f"{f4(C[dec['best_with_S1_ok']]['metrics']['S1_rejection'])}；"
              f"它仍未到用户要求的 0.60。注意 S1 分数上界只有 "
              f"{C['tau_orig']['metrics']['S1_score_max']:.2f}，所以 τ 从 {C['tau_orig']['tau']:.2f} 一路降到 "
              f"{C['tau_orig']['metrics']['S1_score_max']:.2f} 都不会损失任何 S1 拒识——这部分 S2 增益是"
              "「S1 已饱和到 1.0」换来的余量，不是分数变准。",
              f"- 要满足 S1=0.95，τ 必须降到 {dec['s1_0p95_gate_tau']:.4f}"
              f"（Δτ = {dec['s1_0p95_gate_dtau']:+.4f}），那里的 S2 = **{f4(dec['s1_0p95_gate_S2'])}**；"
              f"而 δ_A 只把 τ 推动 {C['tau_shift_A']['metrics']['tau_delta_vs_orig']:+.4f}，"
              f"标量版本 {C['tau_shift_order_stats']['metrics']['tau_delta_vs_orig']:+.4f}，"
              f"差约 {abs(dec['s1_0p95_gate_dtau'] / C['tau_shift_order_stats']['metrics']['tau_delta_vs_orig']):.1f} 倍。"
              "也就是说，**部署条件标定确实把阈值推向了正确方向、但量级远远不够**；"
              "S2≥0.60 需要的那个 τ 位置依旧只能由 Test 的 S1 分数分位（transductive）或人为设定给出。",
              f"- `tau_shift_B`（S2 {f4(C['tau_shift_B']['metrics']['S2_acceptance'])}，S1 "
              f"{f4(C['tau_shift_B']['metrics']['S1_rejection'])}）证明「用大 δ 换大 S2」方向可行，"
              f"但它的 δ 量级 {nB:.2f} 远大于真实偏移 {nA:.2f}（四项合成扰动每项单独就已 11–26，见第 2 节分解），"
              f"把 LOSO 原始拒识打到 {f4(C['tau_shift_B']['loso_original']['unknown_rejection_micro'])}，"
              "属于明确的作弊式阈值平移，不可采纳。",
              '- 不冻结新阈值；E233 的 `tau_orig` 仍是合规最优（S1 1.0 / S2 0.3122）。']
    else:
        L += [f"- **δ 标定没有移动工作点**：最优候选 `{best}` 的 Test S2 = {f4(dec['best_S2'])}，"
              f"不高于 E233 合规值 {f4(E233_S2)}。", '- 部署参数仍为 E233；本轮不产生新模型。']

    L += ['', '## 7. 限制与口径', '',
          '- **transductive 灰区**：变体 A 的 δ 来自被评分批次的无标签 session2 均值，属于转导式使用测试数据；'
          '`tau_shift_A` / `tau_shift_A_half` / `tau_shift_order_stats` 都继承这一灰区，需赛规确认。'
          '变体 B 是纯 Train-only，但它给不出可用的阈值位置。',
          '- **session 与注册状态混叠**（E232）：Test S1 = 5 名未注册被试，S2 = 20 名注册用户。'
          'S1/S2 不是纯生物特征指标；本轮没有用 session 决定身份或门控，但 δ_A 本身读到 session2 的构成，'
          f"因此它必然携带注册信息。不过本轮 cos(δ_A, 身份方向) 只有 {D['cos_pool_A']:.4f}（< {COS_LIMIT:.2f}），"
          '说明它并没有**沿身份主轴**搬运；真正的风险落在 transductive 灰区与「阈值换 S2」这两点上，'
          '而这两点在本轮都直接表现为「S2 上不去 / S1 掉下来」。',
          '- **官方 C / RA / ASafeR = N/A**：Test 无身份标签，S1 拒识 / S2 接受是行为代理指标。',
          '- **Test 反复查看偏倚**：Test 自 E65 起被反复读；本轮候选、漂移剖面、bisection 规则全部在运行前写死在 '
          '`config.json`，但「在候选间比较 S2」这一动作本身读到了 Test。',
          '- 阈值标定用的 inner-OOF 已知分数**未平移**（注册用户在打分管线里被均值对齐，批次偏移被消掉），'
          '这一点在 `config.threshold.shifted_rule` 里写死；若把已知侧也平移，标定就自相矛盾。',
          '- **本轮不改进分数**：五个候选的 Test 分数逐位相同，标定只能在同一条 ROC 上选点。'
          '因此「S1≥0.95 且 S2≥0.60」若在冻结分数上不可达，任何 δ 标定都不可能让它可达；'
          '要在同一 S1 下真正提高 S2，需要换表征/原型（E234 已证相对分数无效），不在本轮范围。',
          '- **S1=0.95 门槛读法本身是 transductive**：它需要 Test S1 的 95% 分位，'
          '这也是为什么 0.7569 这个数字在报告中只作为「曲线上的点」而不是「可达的部署指标」出现。',
          f"- 运行：`.venv\\Scripts\\python.exe research/E235_deployment_calibration/run_e235.py`（CUDA），"
          f"耗时 {R['runtime_s']:.0f}s；逐折平移分数与 δ 存于 `cache/`，阈值表与 δ 存于 "
          '`models/e235_deployment_calibration.pt`。']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
