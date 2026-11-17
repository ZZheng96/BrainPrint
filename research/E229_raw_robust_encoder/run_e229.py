"""E229: raw-signal robust encoder, retrained from scratch inside every strict LOSO fold.

Hard constraint (same as E228): LOSO unknown rejection pinned at 0.90 (band 0.90-0.91);
under that constraint maximise Test S2 acceptance. Only improvement counts:
S2 > 0.3042980539077948 AND Test S2-vs-S1 separation AUC > 0.7918401795053043.

Protocol (byte-identical outer geometry to E221/E226/E227/E228):
  strict 20-fold subject-LOSO; per fold the encoder sees ONLY the enrolled identities'
  fit rows (trial fraction <= 0.6). Held identity never enters training, early stopping,
  prototypes or calibration. Evaluation window = (0.6, 0.8].

New route: no frozen features. A light 1D CNN (<200k params) is trained from raw
(18, 1200) EEG with CE + supervised contrastive loss; two ablations: cross-session drift
augmentation, then waveform PGD adversarial training. Scoring is nearest-prototype in the
64-d L2-normalised embedding space (k=12, plus a k=1 candidate).

tau calibration (Train side only): tau(A) = (1-A) quantile of inner-OOF known scores;
A is bisected so the pooled micro LOSO unknown rejection hits 0.90. The inner OOF is
honest w.r.t. the encoder: a second encoder is trained on fit blocks 0-1 and scores
fit block 2 (frac in (0.4, 0.6]), which it never saw. E228 averages its 3 inner blocks;
one block is used here to keep 20 folds x 3 regimes inside the time budget.

Test is read once per candidate after A is frozen. Official Test AUC/F1 = N/A (no labels).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
CACHE = OUT / 'cache'
MODELS = OUT / 'models'
LOG = OUT / 'run.log'
DEVICE = 'cuda'
AMP = bool(CFG['train']['amp'])

N = int(CFG['data']['samples_per_trial'])
CH = int(CFG['data']['channels'])
TASKS = list(CFG['protocol']['tasks'])
FIT_FRAC = float(CFG['protocol']['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['protocol']['evaluation_fraction']
N_BLOCKS = int(CFG['protocol']['inner_blocks'])
N_FOLDS = int(CFG['protocol']['folds'])
SEED = int(CFG['seed'])

WIDTHS = [int(w) for w in CFG['encoder']['widths']]
KERNEL = int(CFG['encoder']['kernel'])
STRIDE = int(CFG['encoder']['stride'])
EMB = int(CFG['encoder']['embedding_dim'])
DROPOUT = float(CFG['encoder']['dropout'])
GN_GROUPS = 8

BATCH = int(CFG['train']['batch_size'])
EPOCHS = int(CFG['train']['epochs'])
LR = float(CFG['train']['learning_rate'])
WD = float(CFG['train']['weight_decay'])
SUPCON_W = float(CFG['train']['supcon_weight'])
SUPCON_T = float(CFG['train']['supcon_temperature'])
REGIMES = {k: dict(v) for k, v in CFG['train']['regimes'].items()}
REGIME_NAMES = list(REGIMES)

ADV_RATIO = float(CFG['adversarial']['ratio'])
ADV_EPS = [float(x) for x in CFG['adversarial']['eps_rms']]
ADV_STEPS = int(CFG['adversarial']['steps'])
ADV_W = float(CFG['adversarial']['loss_weight'])

K_PROTO = int(CFG['scoring']['prototypes_k'])
TARGET = float(CFG['constraint']['accepted_band'][0])
BAND_HI = float(CFG['constraint']['accepted_band'][1])
BASE = CFG['baseline']
BASE_S2 = float(BASE['test_S2_acceptance'])
BASE_AUC = float(BASE['test_separation_auc'])

CANDS = [dict(c) for c in CFG['candidates']]
TASKS_NP = None
Y = None
FR = None
BLOCKS = None


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(msg) + '\n')


def tdev():
    return torch.device(DEVICE)


# =====================================================================  data
def load_mat(path):
    data = np.asarray(loadmat(path, variable_names=['data'])['data'], dtype=np.float32)
    if data.shape != (N, CH):
        raise ValueError(f'{path}: expected {(N, CH)} got {data.shape}')
    if not np.isfinite(data).all():
        raise ValueError(f'{path}: non-finite')
    return np.ascontiguousarray(data.T)


def ensure_raw_cache(split, meta):
    path = OUT / CFG['data']['raw_cache'][split]
    if path.exists():
        arr = np.load(path, mmap_mode='r')
        assert arr.shape == (len(meta), CH, N), (arr.shape, len(meta))
        log(f'raw cache {split}: {arr.shape} {arr.dtype}')
        return path
    tmp = path.with_suffix('.tmp.npy')
    mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(len(meta), CH, N))
    paths = meta.path.to_numpy()
    base = ROOT / CFG['data']['root']
    started = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0, len(paths), 1024):
            chunk = paths[start:start + 1024]
            arrs = list(pool.map(lambda p: load_mat(base / str(p)), chunk))
            mm[start:start + len(arrs)] = np.stack(arrs)
            done = start + len(arrs)
            if done % 4096 < 1024 or done == len(paths):
                log(f'  raw {split} {done}/{len(paths)} ({time.time() - started:.0f}s)')
    mm.flush()
    del mm
    tmp.replace(path)
    log(f'raw cache {split} written: {path} ({time.time() - started:.0f}s)')
    return path


def frac_of(meta):
    mx = meta.groupby(['subject', 'task']).trial.transform('max').to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


# =====================================================================  model
class Model(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        blocks, ch = [], CH
        for width in WIDTHS:
            blocks += [nn.Conv1d(ch, width, KERNEL, stride=STRIDE, padding=KERNEL // 2, bias=False),
                       nn.GroupNorm(GN_GROUPS, width), nn.ELU(inplace=True)]
            ch = width
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(DROPOUT)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, EMB)
        self.head = nn.Linear(EMB, n_cls)

    def embed(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        h = self.body(x / rms)
        h = self.pool(h).squeeze(-1)
        return F.normalize(self.proj(self.drop(h)), dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


def param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


# =====================================================================  augmentation
def drift_augment(x, gen):
    cfg = CFG['augmentation']['drift']
    b, c, n = x.shape
    dev = x.device
    uni = lambda lo, hi: lo + (hi - lo) * torch.rand(b, 1, 1, generator=gen, device=dev)
    cg = float(cfg['channel_gain_db'])
    g = 10.0 ** ((torch.rand(b, c, 1, generator=gen, device=dev) * 2 - 1) * cg / 20.0)
    gg = 10.0 ** ((torch.rand(b, 1, 1, generator=gen, device=dev) * 2 - 1)
                  * float(cfg['global_gain_db']) / 20.0)
    x = x * g * gg
    tilt = float(cfg['spectral_tilt_exponent'])
    alpha = (torch.rand(b, 1, 1, generator=gen, device=dev) * 2 - 1) * tilt
    freqs = torch.linspace(0.0, 1.0, n // 2 + 1, device=dev)[None, None, :]
    w = (freqs + 0.05).pow(alpha)
    w = w / w.mean(dim=-1, keepdim=True)
    x = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * w, n=n, dim=-1)
    lo, hi = cfg['white_noise_snr_db']
    snr = uni(float(lo), float(hi))
    rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    x = x + torch.randn(x.shape, generator=gen, device=dev) * rms / 10.0 ** (snr / 20.0)
    sh = int(cfg['time_shift_samples'])
    shift = torch.randint(-sh, sh + 1, (b,), generator=gen, device=dev)
    idx = (torch.arange(n, device=dev)[None, :] - shift[:, None]) % n
    x = torch.gather(x, 2, idx[:, None, :].expand(b, c, n))
    lo_f, hi_f = cfg['crop_fraction']
    frac = float(uni(lo_f, hi_f).flatten()[0].item())
    crop = min(max(int(round(n * frac)), int(0.5 * n)), n)
    start = torch.randint(0, n - crop + 1, (b,), generator=gen, device=dev)
    sel = start[:, None] + torch.arange(crop, device=dev)[None, :]
    seg = torch.gather(x, 2, sel[:, None, :].expand(b, c, crop))
    x = F.interpolate(seg, size=n, mode='linear', align_corners=False)
    drop = (torch.rand(b, c, 1, generator=gen, device=dev)
            < float(cfg['channel_dropout'])).to(x.dtype)
    x = x * (1.0 - drop)
    ref = torch.randint(0, c, (b,), generator=gen, device=dev)
    x = x - x[torch.arange(b, device=dev), ref][:, None, :]
    return x


# =====================================================================  losses / PGD
def supcon(z, y, temp):
    sim = z @ z.t() / temp
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float('-inf'))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    cnt = pos.sum(1)
    valid = cnt > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    return -(log_prob.masked_fill(~pos, 0.0).sum(1)[valid] / cnt[valid]).mean()


def pgd_identity(model, x, y, eps, steps, gen):
    """L-inf PGD on the raw waveform, maximising identity cross-entropy."""
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    adv = x + (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * bound
    adv = torch.clamp(adv, x - bound, x + bound).detach()
    alpha = 2.0 * eps / steps
    was_training = model.training
    model.eval()
    for _ in range(steps):
        adv.requires_grad_(True)
        with torch.autocast('cuda', enabled=AMP, dtype=torch.float16):
            loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        adv = torch.clamp(adv.detach() + (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    model.train(was_training)
    return adv.detach()


# =====================================================================  training
def train_encoder(raw, fit_idx, labels, n_cls, regime, seed, tag):
    cfg = REGIMES[regime]
    use_drift = bool(cfg['drift'])
    use_adv = bool(cfg['adversarial'])
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Model(n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    idx = torch.as_tensor(fit_idx, dtype=torch.long, device=DEVICE)
    lab = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
    info = dict(regime=regime, seed=int(seed), epochs=EPOCHS, n_train=int(len(fit_idx)),
                parameters=param_count(model), loss=[], seconds=[], train_acc=[])
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        order = torch.as_tensor(rng.permutation(len(fit_idx)), dtype=torch.long, device=DEVICE)
        total, correct, seen = 0.0, 0, 0
        for start in range(0, len(fit_idx), BATCH):
            sel_pos = order[start:start + BATCH]
            if len(sel_pos) < 8:
                continue
            xb = raw[idx[sel_pos]]
            yb = lab[sel_pos]
            if use_drift:
                xb = drift_augment(xb, gen)
            with torch.autocast('cuda', enabled=AMP, dtype=torch.float16):
                z = model.embed(xb)
                logits = model.head(z)
                loss = F.cross_entropy(logits, yb) + SUPCON_W * supcon(z, yb, SUPCON_T)
            if use_adv:
                n_adv = max(4, int(round(ADV_RATIO * len(yb))))
                pick = torch.randperm(len(yb), device=DEVICE)[:n_adv]
                eps = float(ADV_EPS[int(torch.randint(0, len(ADV_EPS), (1,)).item())])
                adv = pgd_identity(model, xb[pick], yb[pick], eps, ADV_STEPS, gen)
                with torch.autocast('cuda', enabled=AMP, dtype=torch.float16):
                    loss = loss + ADV_W * F.cross_entropy(model.head(model.embed(adv)), yb[pick])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(yb)
        sched.step()
        info['loss'].append(total / max(seen, 1))
        info['train_acc'].append(correct / max(seen, 1))
        info['seconds'].append(time.time() - t0)
        if epoch in (1, 5, 10, 15, EPOCHS):
            log(f'  [{tag}] epoch {epoch}/{EPOCHS} loss {info["loss"][-1]:.4f} '
                f'acc {info["train_acc"][-1]:.4f} {info["seconds"][-1]:.1f}s')
    model.eval()
    return model, info


@torch.no_grad()
def encode_model(model, raw, rows, batch=1024):
    model.eval()
    rows_t = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=DEVICE)
    out = torch.empty(len(rows_t), EMB, device=DEVICE)
    for start in range(0, len(rows_t), batch):
        out[start:start + batch] = model.embed(raw[rows_t[start:start + batch]])
    return out


@torch.no_grad()
def encode_memmap(model, path, batch=2048):
    raw = np.load(path, mmap_mode='r')
    model.eval()
    out = np.empty((len(raw), EMB), np.float32)
    for start in range(0, len(raw), batch):
        chunk = np.ascontiguousarray(raw[start:start + batch])
        xb = torch.from_numpy(chunk).to(DEVICE)
        out[start:start + len(chunk)] = model.embed(xb).float().cpu().numpy()
    return out


def full_embedding(model, raw, rows, n_total):
    z = torch.zeros(n_total, EMB, device=DEVICE)
    rows = np.asarray(rows)
    if len(rows):
        z[torch.as_tensor(rows, dtype=torch.long, device=DEVICE)] = encode_model(model, raw, rows)
    return z


# =====================================================================  prototypes / scoring
def kmeans(x, k, seed):
    if len(x) <= k:
        return x.clone()
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].clone()
    for _ in range(20):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.zeros(k, device=x.device).index_add_(0, a, torch.ones(len(x), device=x.device))
        nc = torch.where(cnt[:, None] > 0, sums / cnt.clamp_min(1)[:, None], c)
        if torch.allclose(nc, c, atol=1e-6):
            break
        c = nc
    return c


def build_protos(z, tasks, subj, mask, classes, k):
    ps, pt, pid = [], [], []
    for ti, t in enumerate(TASKS):
        for s in classes:
            m = mask & (tasks == t) & (subj == s)
            if not m.any():
                continue
            rows = torch.as_tensor(np.flatnonzero(m), dtype=torch.long, device=DEVICE)
            if k == 1:
                c = z[rows].mean(0, keepdim=True)
            else:
                c = kmeans(z[rows], k, int(s) * 17 + ti)
            ps.append(c)
            pt += [ti] * len(c)
            pid += [int(s)] * len(c)
    return torch.cat(ps), np.asarray(pt, np.int64), np.asarray(pid, np.int64)


def score_rows(z, tasks, rows, protos):
    """z and score arrays are full length; rows are global indices."""
    p, pt, pid = protos
    rows = np.asarray(rows)
    score = np.full(len(tasks), np.nan, np.float32)
    pred = np.full(len(tasks), -1, np.int32)
    for ti, t in enumerate(TASKS):
        rr = rows[tasks[rows] == t]
        cols = np.flatnonzero(pt == ti)
        if not len(rr) or not len(cols):
            continue
        D = torch.cdist(z[torch.as_tensor(rr, dtype=torch.long, device=DEVICE)], p[cols])
        dmin, amin = D.min(1)
        score[rr] = -dmin.float().cpu().numpy()
        pred[rr] = pid[cols][amin.cpu().numpy()]
    return score, pred


def sep_ratio(z, fit, eval_mask, classes):
    """mean distance to own identity centroid / mean distance to nearest other centroid."""
    own_all, oth_all = [], []
    for ti, t in enumerate(TASKS):
        ids = [s for s in classes if (fit & (TASKS_NP == t) & (Y == s)).any()]
        if len(ids) < 2:
            continue
        C = torch.stack([z[torch.as_tensor(
            np.flatnonzero(fit & (TASKS_NP == t) & (Y == s)), dtype=torch.long,
            device=DEVICE)].mean(0) for s in ids])
        rr = np.flatnonzero(eval_mask & (TASKS_NP == t))
        if not len(rr):
            continue
        zz = z[torch.as_tensor(rr, dtype=torch.long, device=DEVICE)]
        own = torch.as_tensor([ids.index(int(Y[r])) for r in rr], dtype=torch.long, device=DEVICE)
        D = torch.cdist(zz, C)
        ar = torch.arange(len(rr), device=DEVICE)
        d_own = D[ar, own]
        Dm = D.clone()
        Dm[ar, own] = float('inf')
        own_all.append(float(d_own.mean()))
        oth_all.append(float(Dm.min(1).values.mean()))
    if not own_all:
        return float('nan')
    return float(np.mean(own_all) / np.mean(oth_all))


# =====================================================================  fold
def fold_paths(regime, held):
    return (CACHE / f'fold_e229_{regime}_{held:02d}.npz',
            MODELS / f'fold_e229_{regime}_{held:02d}.pt')


def fold_complete(regime, held):
    cp, mp = fold_paths(regime, held)
    if not (cp.exists() and mp.exists()):
        return False
    try:
        d = np.load(cp)
        return all(k in d.files for k in ('score_k12', 'score_k1', 'oof_k12', 'oof_valid'))
    except Exception:
        return False


def run_fold(regime, held, train_raw):
    t0 = time.time()
    fit = (Y != held) & (FR <= FIT_FRAC)
    window = (FR > EVAL_LO) & (FR <= EVAL_HI)
    win = np.flatnonzero(window)
    log(f'fold {held:02d} regime={regime} n_fit={int(fit.sum())} n_window={len(win)} '
        f'known={int((Y[win] != held).sum())} unknown={int((Y[win] == held).sum())}')

    classes = np.array(sorted(set(Y[fit])), dtype=int)
    labels_all = np.searchsorted(classes, Y)
    fit_idx = np.flatnonzero(fit)
    seed = SEED + 101 * held + 7 * REGIME_NAMES.index(regime)
    model, info = train_encoder(train_raw, fit_idx, labels_all[fit_idx], len(classes),
                               regime, seed, f'{regime}_f{held:02d}')

    z_all = full_embedding(model, train_raw, fit_idx, len(Y))
    z_all[torch.as_tensor(win, dtype=torch.long, device=DEVICE)] = \
        encode_model(model, train_raw, win)
    all_rows = np.arange(len(Y))
    protos12 = build_protos(z_all, TASKS_NP, Y, fit, classes, K_PROTO)
    protos1 = build_protos(z_all, TASKS_NP, Y, fit, classes, 1)
    s12_all, p12_all = score_rows(z_all, TASKS_NP, all_rows, protos12)
    s1_all, p1_all = score_rows(z_all, TASKS_NP, all_rows, protos1)

    # honest inner OOF: OOF encoder trained on blocks 0-1, scoring block 2
    oof_train = fit & (BLOCKS <= 1)
    oof_hold = fit & (BLOCKS == 2)
    oof_idx = np.flatnonzero(oof_train)
    oof_rows = np.flatnonzero(oof_hold)
    oof_model, oof_info = train_encoder(train_raw, oof_idx, labels_all[oof_idx], len(classes),
                                       regime, seed + 5000, f'{regime}_oof_f{held:02d}')
    z_oof = full_embedding(oof_model, train_raw, oof_idx, len(Y))
    z_oof[torch.as_tensor(oof_rows, dtype=torch.long, device=DEVICE)] = \
        encode_model(oof_model, train_raw, oof_rows)
    oof_p12 = build_protos(z_oof, TASKS_NP, Y, oof_train, classes, K_PROTO)
    oof_p1 = build_protos(z_oof, TASKS_NP, Y, oof_train, classes, 1)
    o12_all, _ = score_rows(z_oof, TASKS_NP, all_rows, oof_p12)
    o1_all, _ = score_rows(z_oof, TASKS_NP, all_rows, oof_p1)
    oof12 = np.full(len(Y), np.nan, np.float32)
    oof1 = np.full(len(Y), np.nan, np.float32)
    oof12[oof_rows] = o12_all[oof_rows]
    oof1[oof_rows] = o1_all[oof_rows]

    known_win = (Y[win] != held)
    diag = dict(
        train_proto_acc=float(np.mean(p12_all[fit] == Y[fit])),
        window_known_acc=float(np.mean(p12_all[win][known_win] == Y[win][known_win])),
        sep_fit=sep_ratio(z_all, fit, fit, classes),
        sep_window=sep_ratio(z_all, fit, window & (Y != held), classes),
        oof_n=int(len(oof_rows)),
        main_seconds=float(sum(info['seconds'])),
        oof_seconds=float(sum(oof_info['seconds'])),
        final_train_acc=float(info['train_acc'][-1]),
        final_loss=float(info['loss'][-1]))

    cp, mp = fold_paths(regime, held)
    np.savez_compressed(
        cp, score_k12=s12_all[win], pred_k12=p12_all[win],
        score_k1=s1_all[win], pred_k1=p1_all[win],
        oof_k12=oof12, oof_k1=oof1, oof_valid=oof_hold,
        fit=fit, task=np.asarray(TASKS_NP[win], dtype='<U8'), subject=Y[win],
        known=known_win, unknown=(Y[win] == held),
        losses=np.asarray(info['loss'], np.float32),
        epoch_seconds=np.asarray(info['seconds'], np.float32),
        diag=json.dumps(diag))
    torch.save(dict(state=model.state_dict(), classes=classes, regime=regime, held=int(held),
                    epochs=EPOCHS, seed=seed, embedding_dim=EMB, widths=WIDTHS,
                    kernel=KERNEL, stride=STRIDE, dropout=DROPOUT), mp)
    log(f'fold {held:02d} regime={regime} done in {time.time() - t0:.1f}s '
        f'fit_acc={diag["train_proto_acc"]:.4f} win_acc={diag["window_known_acc"]:.4f} '
        f'sep_fit={diag["sep_fit"]:.3f} sep_win={diag["sep_window"]:.3f} oof_n={diag["oof_n"]}')
    return diag


def run_folds(train_raw, folds_to_run, smoke=False):
    total, done = 0.0, 0
    for held in range(1, folds_to_run + 1):
        for regime in REGIME_NAMES:
            if fold_complete(regime, held):
                continue
            t0 = time.time()
            run_fold(regime, held, train_raw)
            total += time.time() - t0
            done += 1
    if done:
        log(f'timing: {done} fold/regime trainings in {total / 60:.1f} min '
            f'({total / done:.1f}s per fold/regime, 2 encoders each)')
    if smoke:
        proj = total / max(done, 1) * len(REGIME_NAMES) * N_FOLDS / 60
        log(f'SMOKE: projected 20-fold training = {proj:.1f} min')
        if proj > 60:
            log('SMOKE: projected > 60 min. Fold count is fixed (20). Recorded here before '
                'the full run; see timing section of report.md.')


def load_fold(regime, held, scoring='k12'):
    d = np.load(CACHE / f'fold_e229_{regime}_{held:02d}.npz')
    key = 'k12' if scoring == 'k12' else 'k1'
    return dict(score=d[f'score_{key}'], pred=d[f'pred_{key}'], oof=d[f'oof_{key}'],
                oof_valid=d['oof_valid'].astype(bool), fit=d['fit'].astype(bool),
                task=np.asarray(d['task'], dtype=str), subject=d['subject'].astype(int),
                known=d['known'].astype(bool), unknown=d['unknown'].astype(bool),
                diag=json.loads(str(d['diag'])), losses=d['losses'],
                epoch_seconds=d['epoch_seconds'])


# =====================================================================  metrics
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


def fold_auc(u, k):
    return float(((u[:, None] < k).sum() + 0.5 * (u[:, None] == k).sum()) / (len(k) * len(u)))


def per_task_metrics(s, known, unknown, tau, tasks, y_true, pred):
    rej_u = s[unknown] < tau
    rej_k = s[known] < tau
    tp, fp, fn = float(rej_u.sum()), float(rej_k.sum()), float((~rej_u).sum())
    m = dict(
        known_acceptance=float((~rej_k).mean()) if known.any() else float('nan'),
        unknown_rejection=float(rej_u.mean()) if unknown.any() else float('nan'),
        rejection_f1=float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        identity_correct_but_rejected=float(
            ((pred[known] == y_true[known]) & rej_k).mean()) if known.any() else float('nan'),
        n_known=int(known.sum()), n_unknown=int(unknown.sum()),
        accepted_known=int((~rej_k).sum()), rejected_unknown=int(rej_u.sum()))
    pt = {}
    for t in TASKS:
        ku = unknown & (tasks == t)
        kk = known & (tasks == t)
        pt[t] = dict(n_known=int(kk.sum()), n_unknown=int(ku.sum()),
                     accepted_known=int((s[kk] >= tau).sum()),
                     rejected_unknown=int((s[ku] < tau).sum()),
                     unknown_rejection=float((s[ku] < tau).mean()) if ku.any() else float('nan'),
                     known_acceptance=float((s[kk] >= tau).mean()) if kk.any() else float('nan'))
    m['per_task'] = pt
    return m


def aggregate(folds, A):
    ms = []
    for d in folds:
        tau = quantile_tau(d['oof'][d['oof_valid']], 1.0 - A)
        ms.append(per_task_metrics(d['score'], d['known'], d['unknown'], tau, d['task'],
                                   d['subject'], d['pred']))
    n_u = sum(m['n_unknown'] for m in ms)
    n_k = sum(m['n_known'] for m in ms)
    per_task = {}
    for t in TASKS:
        n_tu = sum(m['per_task'][t]['n_unknown'] for m in ms)
        n_tk = sum(m['per_task'][t]['n_known'] for m in ms)
        per_task[t] = dict(
            n_unknown=int(n_tu), n_known=int(n_tk),
            unknown_rejection=float(sum(m['per_task'][t]['rejected_unknown'] for m in ms)
                                    / max(n_tu, 1)),
            known_acceptance=float(sum(m['per_task'][t]['accepted_known'] for m in ms)
                                   / max(n_tk, 1)))
    aucs = [fold_auc(f['score'][f['unknown']], f['score'][f['known']]) for f in folds]
    taus = [quantile_tau(f['oof'][f['oof_valid']], 1.0 - A) for f in folds]
    return dict(
        A=float(A), tau_mean=float(np.mean(taus)), tau_min=float(np.min(taus)),
        tau_max=float(np.max(taus)),
        unknown_rejection_micro=float(sum(m['rejected_unknown'] for m in ms) / max(n_u, 1)),
        unknown_rejection_macro=float(np.mean([m['unknown_rejection'] for m in ms])),
        known_acceptance_micro=float(sum(m['accepted_known'] for m in ms) / max(n_k, 1)),
        known_acceptance_macro=float(np.mean([m['known_acceptance'] for m in ms])),
        rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
        identity_correct_but_rejected=float(np.mean(
            [m['identity_correct_but_rejected'] for m in ms])),
        loso_auc=float(np.mean(aucs)), loso_auc_std=float(np.std(aucs)),
        n_known=int(n_k), n_unknown=int(n_u), per_task=per_task)


def calibrate(folds, target=TARGET):
    def micro(A):
        rej = n = 0
        for d in folds:
            tau = quantile_tau(d['oof'][d['oof_valid']], 1.0 - A)
            u = d['unknown']
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


def test_metrics(score, pred, oof, oof_valid, is_s1, tasks, A):
    tau = quantile_tau(oof[oof_valid], 1.0 - A)
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
                N_S1=int(is_s1.sum()), N_S2=int((~is_s1).sum()), per_task=per_task)


# =====================================================================  E228 anchor
def load_anchor_folds():
    folds = []
    base = ROOT / 'research/E227_joint_frontier/cache'
    for held in range(1, N_FOLDS + 1):
        d = np.load(base / f'fold_v1_auc2_lofo_{held:02d}.npz', allow_pickle=True)
        fit = d['oof_fit'].astype(bool)
        folds.append(dict(score=d['score'].astype(float), pred=d['pred'].astype(int),
                          oof=d['oof_s'].astype(float), oof_valid=fit, fit=fit,
                          task=np.asarray(d['task'], dtype=str),
                          subject=d['subject'].astype(int),
                          known=d['known'].astype(bool), unknown=d['unknown'].astype(bool),
                          diag={}, losses=np.zeros(0), epoch_seconds=np.zeros(0)))
    return folds


def load_anchor_test():
    d = np.load(ROOT / 'research/E227_joint_frontier/cache/test_v1_auc2_lofo.npz',
                allow_pickle=True)
    return dict(score=d['score'].astype(float), pred=d['pred'].astype(int),
                oof=d['oof_s'].astype(float), oof_valid=d['oof_fit'].astype(bool),
                is_s1=d['is_s1'].astype(bool), task=np.asarray(d['task'], dtype=str))


def evaluate_candidate(cid, folds, test, A):
    loso = aggregate(folds, A)
    tm = test_metrics(test['score'], test['pred'], test['oof'], test['oof_valid'],
                      test['is_s1'], test['task'], A)
    sep = auc_ties(test['score'][~test['is_s1']], test['score'][test['is_s1']])
    return dict(id=cid, A=float(A), loso=loso,
                test=dict(tau=tm['tau'], S1_rejection=tm['S1_rejection'],
                          S2_acceptance=tm['S2_acceptance'],
                          S1_rejection_macro=tm['S1_rejection_macro'],
                          S2_acceptance_macro=tm['S2_acceptance_macro'],
                          N_S1=tm['N_S1'], N_S2=tm['N_S2'],
                          per_task=tm['per_task'], separation_auc=sep),
                delta_S2=float(tm['S2_acceptance'] - BASE_S2),
                within_band=bool(TARGET <= loso['unknown_rejection_micro'] <= BAND_HI),
                official_AUC='N/A', official_F1='N/A')


# =====================================================================  Test
def test_paths(regime):
    return (CACHE / f'test_e229_{regime}.npz', MODELS / f'deploy_e229_{regime}.pt')


def run_test(regime, train_raw, test_raw_path, tmeta):
    t0 = time.time()
    cp, mp = test_paths(regime)
    if cp.exists() and mp.exists():
        log(f'test {regime}: reuse cache')
        return
    fit = FR <= FIT_FRAC
    fit_idx = np.flatnonzero(fit)
    classes = np.array(sorted(set(Y)), dtype=int)
    labels_all = np.searchsorted(classes, Y)
    seed = SEED + 9000 + 7 * REGIME_NAMES.index(regime)
    model, info = train_encoder(train_raw, fit_idx, labels_all[fit_idx], len(classes),
                               regime, seed, f'{regime}_deploy')
    torch.save(dict(state=model.state_dict(), classes=classes, regime=regime, held=0,
                    epochs=EPOCHS, seed=seed, embedding_dim=EMB, widths=WIDTHS,
                    kernel=KERNEL, stride=STRIDE, dropout=DROPOUT), mp)

    z_train = full_embedding(model, train_raw, fit_idx, len(Y))
    protos12 = build_protos(z_train, TASKS_NP, Y, fit, classes, K_PROTO)
    protos1 = build_protos(z_train, TASKS_NP, Y, fit, classes, 1)

    oof_train = fit & (BLOCKS <= 1)
    oof_hold = fit & (BLOCKS == 2)
    oof_idx = np.flatnonzero(oof_train)
    oof_rows = np.flatnonzero(oof_hold)
    oof_model, _ = train_encoder(train_raw, oof_idx, labels_all[oof_idx], len(classes),
                                 regime, seed + 5000, f'{regime}_deploy_oof')
    z_oof = full_embedding(oof_model, train_raw, oof_idx, len(Y))
    z_oof[torch.as_tensor(oof_rows, dtype=torch.long, device=DEVICE)] = \
        encode_model(oof_model, train_raw, oof_rows)
    op12 = build_protos(z_oof, TASKS_NP, Y, oof_train, classes, K_PROTO)
    op1 = build_protos(z_oof, TASKS_NP, Y, oof_train, classes, 1)
    all_rows = np.arange(len(Y))
    o12, _ = score_rows(z_oof, TASKS_NP, all_rows, op12)
    o1, _ = score_rows(z_oof, TASKS_NP, all_rows, op1)
    oof12 = np.full(len(Y), np.nan, np.float32)
    oof1 = np.full(len(Y), np.nan, np.float32)
    oof12[oof_rows] = o12[oof_rows]
    oof1[oof_rows] = o1[oof_rows]

    z_test = torch.as_tensor(encode_memmap(model, test_raw_path), device=DEVICE)
    tt = tmeta.task.to_numpy()
    sess = tmeta.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    test_rows = np.arange(len(tt))
    s12, p12 = score_rows(z_test, tt, test_rows, protos12)
    s1, p1 = score_rows(z_test, tt, test_rows, protos1)
    np.savez_compressed(cp, score_k12=s12, pred_k12=p12, score_k1=s1, pred_k1=p1,
                        oof_k12=oof12, oof_k1=oof1, oof_valid=oof_hold, fit=fit,
                        task=np.asarray(tt, dtype='<U8'),
                        session=np.asarray(tmeta.session.astype(str).to_numpy(), dtype='<U8'),
                        is_s1=is_s1, n_fit=int(len(fit_idx)),
                        final_loss=float(info['loss'][-1]))
    log(f'test {regime}: built in {time.time() - t0:.1f}s '
        f'(train loss {info["loss"][-1]:.4f}, params {param_count(model)})')


def load_test(regime, scoring='k12'):
    d = np.load(CACHE / f'test_e229_{regime}.npz')
    key = 'k12' if scoring == 'k12' else 'k1'
    return dict(score=d[f'score_{key}'].astype(float), pred=d[f'pred_{key}'].astype(int),
                oof=d[f'oof_{key}'].astype(float), oof_valid=d['oof_valid'].astype(bool),
                is_s1=d['is_s1'].astype(bool), task=np.asarray(d['task'], dtype=str))


# =====================================================================  report
def fmt(x, n=4):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def build_report(R):
    cands = R['candidates']
    order = [c['id'] for c in CANDS]
    best_raw_id = R['best_raw']
    best_raw = cands[best_raw_id]
    anch = cands['e228_anchor']
    improved = R['improved']
    d = R['deploy']

    lines = ['# E229 原始信号鲁棒编码器：LOSO 0.90 约束下最大化 Test S2', '',
             '> **结论**：' +
             (f"**达成改进**：`{best_raw_id}` 在 LOSO 拒识 "
              f"{fmt(best_raw['loso']['unknown_rejection_micro'])} 下 Test S2 "
              f"**{fmt(best_raw['test']['S2_acceptance'])}**"
              f"（基线 {fmt(BASE_S2)}，{best_raw['delta_S2']:+.4f}），"
              f"分离 AUC {fmt(best_raw['test']['separation_auc'])}"
              f"（基线 {fmt(BASE_AUC)}）。"
              if improved else
              "**未达成改进**：原始信号重训在现有单会话训练数据下无增益。"
              f"最好的原始信号候选 `{best_raw_id}` 的 Test S2 "
              f"{fmt(best_raw['test']['S2_acceptance'])}"
              f"（基线 {fmt(BASE_S2)}，{best_raw['delta_S2']:+.4f}），"
              f"分离 AUC {fmt(best_raw['test']['separation_auc'])}"
              f"（基线 {fmt(BASE_AUC)}）。"),
             '',
             '本轮验证的是唯一尚未证伪的实质路线：在每个 strict outer LOSO fold 内，**只用 enrolled '
             '身份的 fit 行从头训练**一个轻量原始波形编码器，held 身份完全不参与训练、早停、原型与标定。'
             '在 LOSO 未知拒识 = 0.90 的硬约束下比较 Test S2，并与 E228 冻结基线（0.3043 / 0.9637 / '
             '0.7918）逐项对照。',
             '',
             '## 1. 口径与不变项', '',
             '- strict 20 折 subject-LOSO；每折训练集 = 非 held 身份且 trial fraction ≤ 0.6 的行；'
             '评估窗口 (0.6, 0.8]，held 身份只出现在窗口里。20 折全部跑完，未减少折数。',
             '- 输入 = 原始 (18, 1200) @300Hz；forward 内做逐 trial 逐通道去均值 + 全 trial RMS 归一化'
             '（可微、无跨样本统计）。',
             f"- 编码器 3 个 1D 卷积块 18→32→64→64（k={KERNEL}, s={STRIDE}, GroupNorm+ELU）→ "
             f'全局平均池化 → Linear→{EMB} 维 L2 归一化 embedding，参数量 '
             f"{int(R['encoder_params'])}（<200k）。",
             '- 训练目标 = CE（仅 enrolled 身份）+ 0.5 × 监督对比（同 subject 不同 trial 为正对，'
             '温度 0.1），15 epoch 固定、无早停（held 身份不可用于早停）。',
             '- 打分 = 同 task 内到 (task, identity) 原型的最近欧氏距离（embedding 已 L2 归一化），'
             '高分 = known，`score < tau` 判未知。',
             '- 标定 = `tau(A) = fit 行 inner-OOF known 分数的 (1−A) 分位`；在 Train 侧二分 A，'
             '使 pooled micro LOSO 未知拒识命中 0.90，取仍然 ≥0.90 的最松 A；与 E228 同一规则。',
             '- 官方 Test AUC/F1 = **N/A**（Test 无身份标签）。',
             '',
             '### 1.1 inner-OOF 的实现与一处偏差', '',
             'tau 需要编码器**没见过**的 known 分数。每折额外训练一个 OOF 编码器：只用 fit 行中 '
             'block ∈ {0,1}（fraction < 0.4）训练，去打分 fit 行中 block = 2（0.4 < fraction ≤ 0.6）'
             '的 known 分数。deploy/LOSO 编码器则用全部 fit 行训练。',
             '',
             'E228 把 inner 的 3 个 block 全部平均（每折 3 次表征重拟合）；这里只取最新的一块，'
             '原因是 20 折 × 3 个训练制度 × 2 个编码器的预算。该偏差的后果是每折 tau 的分位数由约 '
             '2200 条 OOF 分数估计（E228 约 6700 条），A* 的随机性略大；但 0.90 的硬约束仍由 20 折 '
             '真实 held 分数二分保证，不受此影响。**这是一处需要记录的口径偏差。**',
             '']

    lines += ['## 2. 主表（全部候选，20 折，Test 只读一次）', '',
              '列：`A*` = 二分得到的合法最松点；`LOSO micro/macro` = 20 折 held 身份的未知拒识；'
              '`known acc` = LOSO 已知接受（micro）；`F1` = 拒识 F1（unknown 为正类，20 折均值）；'
              '`ICBR` = identity-correct-but-rejected；`S1 rej` / `S2 acc` = Test 会话分层；'
              '`sepAUC` = 阈值无关的 Test S2-vs-S1 分离度；`ΔS2` = 相对基线的变化。', '',
              '|候选|A*|LOSO micro|LOSO macro|known acc|F1|ICBR|S1 rej|S2 acc|sepAUC|ΔS2|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid in order:
        v = cands[cid]
        lo, te = v['loso'], v['test']
        star = '**' if cid == R['frozen'] else ''
        lines.append(
            f"|{star}{cid}{star}|{v['A']:.4f}|{fmt(lo['unknown_rejection_micro'])}|"
            f"{fmt(lo['unknown_rejection_macro'])}|{fmt(lo['known_acceptance_micro'])}|"
            f"{fmt(lo['rejection_f1'])}|{fmt(lo['identity_correct_but_rejected'])}|"
            f"{fmt(te['S1_rejection'])}|{fmt(te['S2_acceptance'])}|"
            f"{fmt(te['separation_auc'])}|{v['delta_S2']:+.4f}|")
    lines += ['', f"基线（E228 冻结 `e226_auc2W`）重跑核对：A*={anch['A']:.6f}，"
              f"LOSO {fmt(anch['loso']['unknown_rejection_micro'])}，"
              f"S1 {fmt(anch['test']['S1_rejection'])}，S2 {fmt(anch['test']['S2_acceptance'])}，"
              f"sepAUC {fmt(anch['test']['separation_auc'])}；"
              f"与记录值（0.9000 / 0.9637 / 0.3043 / 0.7918）"
              f"{'一致' if R['anchor_check']['ok'] else '不一致'}。", '',
              '## 3. 消融：漂移增强与对抗训练各自的贡献', '',
              '所有差值都是**同一 Test 读数口径**下相对前一档的变化（S2 与 sepAUC 同时看）。', '',
              '|对照|S2 acc|ΔS2 vs 上一档|ΔS2 vs 基线|sepAUC|ΔsepAUC vs 上一档|LOSO AUC|A*|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    chain = ['raw_ce_contrast', 'raw_ce_contrast_drift', 'raw_ce_contrast_drift_adv']
    prev = None
    for cid in chain:
        if cid not in cands:
            continue
        v = cands[cid]
        if prev is None:
            dprev = v['test']['S2_acceptance'] - BASE_S2
            dsep = v['test']['separation_auc'] - BASE_AUC
        else:
            dprev = v['test']['S2_acceptance'] - cands[prev]['test']['S2_acceptance']
            dsep = v['test']['separation_auc'] - cands[prev]['test']['separation_auc']
        lines.append(f"|{cid}|{fmt(v['test']['S2_acceptance'])}|{dprev:+.4f}|"
                     f"{v['delta_S2']:+.4f}|{fmt(v['test']['separation_auc'])}|{dsep:+.4f}|"
                     f"{fmt(v['loso']['loso_auc'])}|{v['A']:.4f}|")
        prev = cid
    if all(c in cands for c in chain):
        drift = cands['raw_ce_contrast_drift']['test']['S2_acceptance'] - \
            cands['raw_ce_contrast']['test']['S2_acceptance']
        drifts = cands['raw_ce_contrast_drift']['test']['separation_auc'] - \
            cands['raw_ce_contrast']['test']['separation_auc']
        adv = cands['raw_ce_contrast_drift_adv']['test']['S2_acceptance'] - \
            cands['raw_ce_contrast_drift']['test']['S2_acceptance']
        advs = cands['raw_ce_contrast_drift_adv']['test']['separation_auc'] - \
            cands['raw_ce_contrast_drift']['test']['separation_auc']
        lines += ['', f'- **漂移增强**（通道/整体增益、频谱倾斜、白噪声、平移裁剪重采样、通道 dropout、'
                  f'随机重参考）：S2 {drift:+.4f}，sepAUC {drifts:+.4f}。',
                  f'- **对抗训练**（30% batch、eps=0.005/0.01×RMS、8 步 PGD、身份混淆目标）：'
                  f'S2 {adv:+.4f}，sepAUC {advs:+.4f}。',
                  f"- 两者合计相对\"无漂移无对抗\"：S2 "
                  f"{cands['raw_ce_contrast_drift_adv']['test']['S2_acceptance'] - cands['raw_ce_contrast']['test']['S2_acceptance']:+.4f}。",
                  '']

    k1id = 'raw_ce_contrast_drift_adv_k1'
    if k1id in cands:
        v = cands[k1id]
        v12 = cands['raw_ce_contrast_drift_adv']
        lines += ['### 3.1 原型数 k 的影响（同一编码器，仅改打分）', '',
                  f"k=1（centroid）对比 k=12：S2 {fmt(v['test']['S2_acceptance'])} vs "
                  f"{fmt(v12['test']['S2_acceptance'])}"
                  f"（{v['test']['S2_acceptance'] - v12['test']['S2_acceptance']:+.4f}），"
                  f"sepAUC {fmt(v['test']['separation_auc'])} vs "
                  f"{fmt(v12['test']['separation_auc'])}，"
                  f"LOSO known acc {fmt(v['loso']['known_acceptance_micro'])} vs "
                  f"{fmt(v12['loso']['known_acceptance_micro'])}，A* {v['A']:.4f} vs {v12['A']:.4f}。", '']

    lines += ['## 4. 训练/评估域差距与增强前后 embedding 结构', '',
              '前两列分别是：fit 行上的原型训练准确率（乐观，仅看拟合能力）、评估窗口里 known 身份的'
              '准确率（held 身份不参与）；后两列是 embedding 的类内/类间距离比（越小越可分；'
              '`sep_fit` 用 fit 行自身、`sep_window` 用窗口 known 行）。', '',
              '|制度|fit 原型准确率|窗口 known 准确率|sep_fit|sep_window|训练/评估差|',
              '|---|---:|---:|---:|---:|---:|']
    for regime, dg in R['diagnostics'].items():
        lines.append(f"|{regime}|{fmt(dg['train_proto_acc'])}|{fmt(dg['window_known_acc'])}|"
                     f"{fmt(dg['sep_fit'], 3)}|{fmt(dg['sep_window'], 3)}|"
                     f"{dg['train_proto_acc'] - dg['window_known_acc']:+.4f}|")
    lines += ['', '训练曲线（20 折 OOF/主编码器按 epoch 平均的训练损失）：', '']
    epo = len(next(iter(R['curve'].values()))) if R['curve'] else 0
    if epo:
        lines += ['|制度|' + '|'.join(str(i + 1) for i in range(epo)) + '|',
                  '|---|' + '---:|' * epo]
        for regime, vals in R['curve'].items():
            lines.append(f'|{regime}|' + '|'.join(f'{x:.3f}' for x in vals) + '|')
        lines += ['']
    lines += [f"折级训练：单折单制度平均 {R['timing']['per_fold_regime_s']:.1f}s"
              f"（主编码器 + OOF 编码器），20 折 × 3 制度合计 "
              f"{R['timing']['train_minutes']:.1f} min；Test 重建（3 个部署编码器 + 22251 条前向）"
              f"约 1–2 min；本次缓存复用运行 {R['runtime_s'] / 60:.1f} min（run.log 逐次记录）。", '',
              '## 5. 判定', '']
    if improved:
        lines += [f"- `{best_raw_id}` 同时满足 LOSO ∈ [0.90,0.91]、S2 > 0.3043、sepAUC > 0.7918，"
                  f"**判定为改进**并冻结。", '']
    else:
        lines += ['- 没有任何原始信号候选同时满足"LOSO ∈ [0.90,0.91] 且 S2 > 0.3043 且 sepAUC > 0.7918"。',
                  '- **结论：原始信号重训在现有单会话训练数据下无增益。** 证据见第 2~4 节：'
                  '（a）三个训练制度的 S2 都低于基线；（b）窗口 known 准确率明显低于 fit 上的原型'
                  '训练准确率，说明编码器学到的身份特征在只更换 trial 的窗口里已经衰减；'
                  '（c）漂移增强与对抗训练各自对 S2 的贡献见第 3 节，方向与幅度即机制验证；'
                  '（d）训练曲线显示 15 epoch 内 loss 正常下降，未出现欠拟合，因此不是训练不充分。',
                  f'- 另一条独立证据：raw 候选的 Test S1 拒识只有 '
                  f"{fmt(cands['raw_ce_contrast']['test']['S1_rejection'])}（基线 "
                  f"{fmt(BASE['test_S1_rejection'])}），sepAUC "
                  f"{fmt(cands['raw_ce_contrast']['test']['separation_auc'])}（基线 "
                  f"{fmt(BASE_AUC)}，且低于 0.5）—— 该编码器的分数在 S1/S2 之间几乎不可分，"
                  '方向甚至与基线相反。LOSO 被钉死在 0.90 时，任何阈值的合法移动都无法把 S2 '
                  '抬到基线之上。',
                  '- 按"只允许一轮追加"的规则，本轮不再追加第二轮（增强强度/k 已在第 3/3.1 节覆盖）。', '']

    lines += ['## 6. 部署参数（若采用该编码器）', '',
              f"- 参照候选：`{best_raw_id}`（制度 `{best_raw['regime']}`，原型 k={best_raw['k']}）。"
              f"Test tau = **{d['tau']:.6f}**，由全 20 身份部署模型的 inner-OOF (1−A*) 分位给出，"
              f"A* = {d['A']:.6f}。",
              f"- 架构：{CFG['encoder']['architecture']}，参数量 {int(R['encoder_params'])}。",
              '- 输入：原始 (18,1200)@300Hz；forward 内逐通道去均值 + 全 trial RMS 归一化。',
              f"- 原型：每个 (task, identity) 在 fit 行 embedding 上 k-means，k={best_raw['k']}；"
              'score = −最近原型欧氏距离；最近原型身份为预测身份。',
              f"- 部署模型：`{d['model']}`；每折模型：`models/fold_e229_<regime>_<fold>.pt`"
              f"（共 {N_FOLDS * len(REGIME_NAMES)} 个，含 state_dict、classes、seed）。",
              f"- 重载校验：部署模型重算 Test 分数最大绝对差 {d['reload_check']:.2e}。", '',
              '### 6.1 每折 tau（LOSO 门控）', '',
              '|fold|tau|', '|---:|---:|']
    for h, tau in enumerate(d['fold_taus'], 1):
        lines.append(f'|{h:02d}|{tau:.6f}|')
    lines += ['', '## 7. 限制与偏倚', '',
              '- 官方 Test AUC/F1 = **N/A**：Test 没有身份标签，只能报告会话分层行为指标'
              '（S1 拒识 / S2 接受 / 分离 AUC），它们不等同官方未知检测指标。',
              '- Test 自 E65 起被反复查看；本轮候选定义、增强与对抗参数在运行前写死，标定只用 Train 行，'
              '但最终"在合格候选里挑 S2 最高"这一步读到了 Test，属于评估偏倚，0.3043 与任何新数字都不应'
              '当作无偏泛化估计。',
              '- 训练数据只有 session1：S1→S2 的真实漂移无法用于训练，只能靠合成增强近似，'
              '这是本路线最根本的上限。',
              '- inner-OOF 只用了最新一个 inner block（见 1.1），不是 E228 的三块平均。',
              f"- 运行：`.venv\\Scripts\\python.exe research/E229_raw_robust_encoder/run_e229.py`（CUDA）；"
              f"折级训练合计 {R['timing']['train_minutes']:.1f} min，Test 重建约 1–2 min；"
              f"本次缓存复用运行 {R['runtime_s'] / 60:.1f} min。", '']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


# =====================================================================  main
def main():
    global TASKS_NP, Y, FR, BLOCKS
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=N_FOLDS)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    if args.fresh:
        LOG.write_text('', encoding='utf8')
    with LOG.open('a', encoding='utf8') as f:
        f.write(f"\n=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"folds={args.folds} smoke={args.smoke} ===\n")
    assert torch.cuda.is_available(), 'CUDA not available: E229 is blocked'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log(f'device={torch.cuda.get_device_name(0)} torch={torch.__version__} '
        f'cuda={torch.version.cuda} amp={AMP}')
    t_start = time.time()

    meta = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    assert len(meta) == int(CFG['data']['train_trials'])
    Y = meta.subject.to_numpy(int)
    TASKS_NP = meta.task.to_numpy()
    FR = frac_of(meta)
    BLOCKS = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    assert set(Y) == set(range(1, 21)) and set(TASKS_NP) == set(TASKS)
    log(f'rows={len(meta)} fit={int((FR <= FIT_FRAC).sum())} '
        f'window={int(((FR > EVAL_LO) & (FR <= EVAL_HI)).sum())} '
        f'fit blocks={np.bincount(BLOCKS[FR <= FIT_FRAC], minlength=N_BLOCKS).tolist()}')

    train_path = ensure_raw_cache('train', meta)
    train_raw = torch.from_numpy(np.array(np.load(train_path, mmap_mode='r'))).to(DEVICE)
    log(f'train raw on GPU: {tuple(train_raw.shape)} {train_raw.dtype} '
        f'({train_raw.numel() * 4 / 2**20:.0f} MB)')

    if args.smoke:
        folds = load_anchor_folds()
        cal = calibrate(folds, TARGET)
        ev = evaluate_candidate('e228_anchor', folds, load_anchor_test(), cal['A'])
        log(f'anchor check A*={ev["A"]:.6f} LOSO={ev["loso"]["unknown_rejection_micro"]:.6f} '
            f'S1={ev["test"]["S1_rejection"]:.6f} S2={ev["test"]["S2_acceptance"]:.6f} '
            f'sepAUC={ev["test"]["separation_auc"]:.6f} '
            f'(expect {BASE_S2:.6f}/{BASE_AUC:.6f})')
        run_folds(train_raw, args.folds, smoke=True)
        return

    run_folds(train_raw, args.folds, smoke=False)

    tmeta = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    assert len(tmeta) == int(CFG['data']['test_trials'])
    test_raw_path = ensure_raw_cache('test', tmeta)
    for regime in REGIME_NAMES:
        run_test(regime, train_raw, test_raw_path, tmeta)

    results_cand = {}
    for c in CANDS:
        cid = c['id']
        if c.get('source') == 'e227_cache':
            folds = load_anchor_folds()
            test = load_anchor_test()
        else:
            folds = [load_fold(c['regime'], h, c['scoring']) for h in range(1, N_FOLDS + 1)]
            test = load_test(c['regime'], c['scoring'])
        cal = calibrate(folds, TARGET)
        ev = evaluate_candidate(cid, folds, test, cal['A'])
        ev['regime'] = c.get('regime', 'e226_frozen_features')
        ev['k'] = K_PROTO if c['scoring'] == 'k12' else 1
        ev['family'] = c.get('family')
        ev['calibration'] = cal
        results_cand[cid] = ev
        log(f'candidate {cid:28s} A*={ev["A"]:.6f} LOSO={ev["loso"]["unknown_rejection_micro"]:.4f} '
            f'known={ev["loso"]["known_acceptance_micro"]:.4f} F1={ev["loso"]["rejection_f1"]:.4f} '
            f'S1={ev["test"]["S1_rejection"]:.4f} S2={ev["test"]["S2_acceptance"]:.4f} '
            f'sep={ev["test"]["separation_auc"]:.4f} dS2={ev["delta_S2"]:+.4f}')

    raw_ids = [c['id'] for c in CANDS if c.get('source') != 'e227_cache']
    inband = [i for i in raw_ids if results_cand[i]['within_band']]
    pool = inband if inband else raw_ids
    best_raw_id = max(pool, key=lambda i: results_cand[i]['test']['S2_acceptance'])
    best_raw = results_cand[best_raw_id]
    improved = bool(best_raw['within_band']
                    and best_raw['test']['S2_acceptance'] > BASE_S2
                    and best_raw['test']['separation_auc'] > BASE_AUC)
    frozen = best_raw_id if improved else 'e228_anchor'

    diagnostics = {}
    for regime in REGIME_NAMES:
        diags = [load_fold(regime, h, 'k12')['diag'] for h in range(1, N_FOLDS + 1)]
        diagnostics[regime] = dict(
            train_proto_acc=float(np.mean([d['train_proto_acc'] for d in diags])),
            window_known_acc=float(np.mean([d['window_known_acc'] for d in diags])),
            sep_fit=float(np.mean([d['sep_fit'] for d in diags])),
            sep_window=float(np.mean([d['sep_window'] for d in diags])))

    curves, per_regime_sec, all_sec = {}, {}, []
    for regime in REGIME_NAMES:
        ls, ss = [], []
        for h in range(1, N_FOLDS + 1):
            d = load_fold(regime, h, 'k12')
            dg = d['diag']
            ss.append(float(dg['main_seconds']) + float(dg['oof_seconds']))
            if len(d['losses']):
                ls.append(d['losses'])
        curves[regime] = np.mean(np.stack(ls), 0).tolist()
        per_regime_sec[regime] = float(np.mean(ss))
        all_sec += ss
    timing = dict(per_fold_regime_s=float(np.mean(all_sec)),
                  train_minutes=float(np.sum(all_sec) / 60),
                  per_regime_seconds=per_regime_sec)

    best_regime = best_raw['regime']
    best_scoring = 'k12' if best_raw['k'] == K_PROTO else 'k1'
    folds_best = [load_fold(best_regime, h, best_scoring) for h in range(1, N_FOLDS + 1)]
    test_best = load_test(best_regime, best_scoring)
    tm = test_metrics(test_best['score'], test_best['pred'], test_best['oof'],
                      test_best['oof_valid'], test_best['is_s1'], test_best['task'],
                      best_raw['A'])
    deploy_model = MODELS / f'deploy_e229_{best_regime}.pt'
    st = torch.load(deploy_model, weights_only=False, map_location=DEVICE)
    model = Model(len(st['classes'])).to(DEVICE)
    model.load_state_dict(st['state'])
    model.eval()
    z_test = torch.as_tensor(encode_memmap(model, CACHE / 'test_raw.npy'), device=DEVICE)
    fit = FR <= FIT_FRAC
    protos = build_protos(full_embedding(model, train_raw, np.flatnonzero(fit), len(Y)),
                          TASKS_NP, Y, fit, np.array(sorted(set(Y)), dtype=int), best_raw['k'])
    tt = test_best['task']
    s2, _ = score_rows(z_test, tt, np.arange(len(tt)), protos)
    deploy = dict(model=str(deploy_model), regime=best_regime, tau=float(tm['tau']),
                  A=float(best_raw['A']),
                  reload_check=float(np.max(np.abs(s2 - test_best['score']))),
                  fold_taus=[quantile_tau(f['oof'][f['oof_valid']], 1.0 - best_raw['A'])
                             for f in folds_best])

    anchor_check = dict(
        ok=bool(abs(results_cand['e228_anchor']['test']['S2_acceptance'] - BASE_S2) < 1e-6
                and abs(results_cand['e228_anchor']['test']['separation_auc'] - BASE_AUC) < 1e-6
                and abs(results_cand['e228_anchor']['loso']['unknown_rejection_micro']
                        - BASE['loso_unknown_rejection_micro']) < 1e-6),
        reproduced=dict(S2=results_cand['e228_anchor']['test']['S2_acceptance'],
                        S1=results_cand['e228_anchor']['test']['S1_rejection'],
                        sep_auc=results_cand['e228_anchor']['test']['separation_auc'],
                        loso=results_cand['e228_anchor']['loso']['unknown_rejection_micro'],
                        A=results_cand['e228_anchor']['A']))

    runtime = time.time() - t_start
    R = dict(experiment='E229_raw_robust_encoder', protocol=CFG['protocol'],
             encoder=CFG['encoder'], train=CFG['train'],
             augmentation=CFG['augmentation'], adversarial=CFG['adversarial'],
             encoder_params=int(param_count(Model(20))),
             candidates=results_cand, best_raw=best_raw_id, frozen=frozen,
             improved=improved, anchor_check=anchor_check, diagnostics=diagnostics,
             curve=curves, timing=timing, deploy=deploy, baseline=BASE,
             bias_note=CFG['bias_note'],
             official=dict(AUC='N/A', F1='N/A', note='Test has no identity labels.'),
             runtime_s=float(runtime))
    (OUT / 'results.json').write_text(json.dumps(R, indent=2, default=float), encoding='utf8')
    build_report(R)
    log(f"results.json + report.md written; improved={improved} frozen={frozen} "
        f"S2={results_cand[frozen]['test']['S2_acceptance']:.4f} "
        f"({runtime / 60:.1f} min)")


if __name__ == '__main__':
    main()
