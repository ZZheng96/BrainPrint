"""E249 gate backfill #7 RA / #8 ASafeR: self-built white-box PGD on C2 (alpha=0.5).

Mirrors E195's attack protocol: subset = Train window 0.6 < frac <= 0.8, cap 400 per task,
seed 20261115; SNR {30,20} dB; 20 steps; goals A/B/C/D; radius = per-trial L2 norm of the
channel-demeaned waveform / 10^(SNR/20); #7 = min RA, #8 = min ASafeR over conditions.

Scorer = E249 deployment model (20 ids, Early fit) with C2 drift-aware whitening; the same
P1 FAR10 threshold as the audit. This is NOT the E191 branch-A model used by E195, so the
numbers are comparable in protocol only.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import loadmat

import run_e249 as E

R, M, DEV = E.R, E.R.M, E.DEV
ROOT, OUT = E.ROOT, E.OUT
SNRS, STEPS, SEED, CAP = (30, 20), 20, 20261115, 400
TAU_KEY = "C2_a0.5"


# ---------------------------------------------------------------- differentiable features
def sub_feature(x, cfg):
    length = x.shape[-1]
    channels = int(cfg["channels"])
    bands = cfg["bands_hz"]
    freq = torch.fft.rfftfreq(length, d=1.0 / float(cfg["sampling_rate_hz"]), device=x.device)
    masks = []
    for i, (low, high) in enumerate(bands):
        upper = freq <= high if i == len(bands) - 1 else freq < high
        masks.append((freq >= low) & upper)
    mask = torch.stack(masks).to(x.dtype)
    spectrum = torch.fft.rfft(x, dim=-1)
    filtered = torch.fft.irfft(spectrum[:, None] * mask[None, :, None, :], n=length, dim=-1)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    ddof = int(cfg["channel_std_ddof"])
    scale = filtered.std(dim=-1, keepdim=True, unbiased=ddof == 1).clamp_min(
        float(cfg["channel_std_epsilon"]))
    normalised = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    corr = normalised @ normalised.transpose(-1, -2) / (length - ddof)
    trace = torch.diagonal(corr, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(channels, dtype=x.dtype, device=x.device)
    shrink = float(cfg["covariance_shrinkage"])
    spd = (1.0 - shrink) * corr + shrink * trace[..., None, None] / channels * eye
    values, vectors = torch.linalg.eigh(spd)
    values = values.clamp_min(float(cfg["eigenvalue_floor"]))
    logspd = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    rows, cols = torch.triu_indices(channels, channels, device=x.device)
    vec = logspd[..., rows, cols]
    diag = rows == cols
    vec = torch.where(diag[None, None], vec, vec * float(cfg["off_diagonal_multiplier"]))
    vec = vec.reshape(len(vec), len(bands), -1)
    dvals = vec[..., diag]
    vec = vec.index_copy(-1, torch.nonzero(diag).flatten(), dvals - dvals.mean(dim=-1, keepdim=True))
    mean = vec.mean(dim=-1, keepdim=True)
    sd = vec.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(cfg["zband_epsilon"]))
    return ((vec - mean) / sd).flatten(-2)


def features(x, cfg):
    n_sub, length = int(cfg["n_subwindows"]), int(cfg["subwindow_length"])
    parts = [sub_feature(x[..., w * length:(w + 1) * length], cfg) for w in range(n_sub)]
    return torch.stack(parts, dim=1).mean(dim=1)


class Scorer(torch.nn.Module):
    def __init__(self, art, fitted, T):
        super().__init__()
        self.cfg = art["feature"]
        s = fitted["space"]
        self.mu, self.sd, self.pm = s.mu, s.sd, s.pm
        self.pb = s.pb
        self.T = T
        self.protos = F.normalize(fitted["protos"] @ T, dim=2)  # (4,20,96)
        self.valid = fitted["valid"]

    def forward(self, x, task_idx):
        f = features(x, self.cfg)
        z = torch.empty_like(f)
        for t in range(4):
            m = task_idx == t
            if m.any():
                z[m] = (f[m] - self.mu[t]) / self.sd[t]
        p = (z - self.pm) @ self.pb
        pw = F.normalize(p @ self.T, dim=1)
        mat = torch.einsum("nd,tsd->nts", pw, self.protos)
        mat[:, :, ~self.valid] = -1e9
        logits = mat[torch.arange(len(x), device=x.device), task_idx]
        return logits, logits.max(1).values


# ---------------------------------------------------------------- data
def load_mat(path):
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    assert data.shape == (1200, 18), (path, data.shape)
    return np.ascontiguousarray(data.T)


meta = pd.read_csv(R.META_PATH)
feats = torch.as_tensor(np.load(R.FEATURE_PATH).astype(np.float32), device=DEV)
R.validate(meta, feats)
_, frac = R.temporal_blocks(meta)
subj, tasks = meta.subject.to_numpy(), R.M._task_indices([{"task": t} for t in meta.task], len(meta))
art = M.load_model(str(ROOT / "final" / "team_hymn"), str(DEV))

# feature-parity check against the submitted extractor
idx = np.arange(0, 256)
x_chk = torch.stack([torch.as_tensor(load_mat(ROOT / "Brainprint_Competition" / meta.path.iloc[i]))
                     for i in idx]).to(DEV)
d = float((features(x_chk, art["feature"]) - M._features(art, x_chk)).abs().max())
assert d < 2e-5, f"feature mismatch {d}"

window = (frac > 0.6) & (frac <= 0.8)
rng = np.random.default_rng(SEED)
sel = []
for t in range(4):
    rows = np.flatnonzero(window & (tasks == t))
    sel.append(rows if len(rows) <= CAP else np.sort(rng.choice(rows, size=CAP, replace=False)))
subset = np.sort(np.concatenate(sel))
truth = subj[subset]

plan = E.outer_plan(subj, frac, 0)
fitted, Sw, Ds, _ = E.fit_all(feats, tasks, subj, plan["known"], plan["fit"], plan["d1"], plan["d2"])
T = E.transforms(fitted, Sw, Ds)[(TAU_KEY, "all")]
tau = json.loads((OUT / "results" / "test_audit.json").read_text(encoding="utf-8"))["thresholds_P1_FAR10"][TAU_KEY]
scorer = Scorer(art, fitted, T).to(DEV).eval()
for p in scorer.parameters():
    p.requires_grad_(False)


def metrics(logits, gate, truth_sub):
    pred = np.where(gate >= tau, logits.argmax(1) + 1, 0)
    return {"RA": float(np.mean(pred == truth_sub)),
            "ASafeR": float(np.mean((pred == truth_sub) | (pred == 0))),
            "accept_rate": float(np.mean(gate >= tau)),
            "closed_set_accuracy": float(np.mean(logits.argmax(1) + 1 == truth_sub)),
            "wrong_identity_rate": float(np.mean((pred != 0) & (pred != truth_sub)))}


def pgd(x, task_idx, true_idx, target_idx, goal, snr):
    gen = torch.Generator(device=DEV).manual_seed(SEED + 1000 * int(snr) + ord(goal))
    xc = x - x.mean(dim=-1, keepdim=True)
    radius = xc.reshape(len(x), -1).norm(dim=1).clamp_min(1e-12) / (10.0 ** (float(snr) / 20.0))
    delta = torch.randn(x.shape, generator=gen, device=DEV)
    delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta = delta * radius.view(-1, 1, 1) * torch.rand(len(delta), generator=gen, device=DEV).view(-1, 1, 1)
    alpha = 2.0 * radius / STEPS
    for _ in range(STEPS):
        delta = delta.detach().requires_grad_(True)
        logits, gate = scorer(x + delta, task_idx)
        if goal == "D":
            objective = gate
        else:
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(F.one_hot(true_idx, 20).bool(), -1e9)
            if goal == "A":
                margin = wrong.max(dim=1).values
            elif goal == "B":
                margin = torch.logsumexp(wrong, dim=1)
            else:
                margin = logits.gather(1, target_idx[:, None]).squeeze(1)
            objective = (margin - true_logit) / 10.0
        grad = torch.autograd.grad(-objective.sum(), delta)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        direction = grad / grad.reshape(len(grad), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() - alpha.view(-1, 1, 1) * direction
        norm = delta.reshape(len(delta), -1).norm(dim=1)
        delta = delta * torch.minimum(torch.ones_like(norm), radius / norm).view(-1, 1, 1)
    with torch.no_grad():
        logits, gate = scorer(x + delta, task_idx)
    return logits.cpu().numpy(), gate.cpu().numpy()


conditions, clean_all = [], []
buf = {(snr, goal): [] for snr in SNRS for goal in "ABCD"}
t0 = time.time()
CHUNK = 128
for start in range(0, len(subset), CHUNK):
    rows = subset[start:start + CHUNK]
    x = torch.stack([torch.as_tensor(load_mat(ROOT / "Brainprint_Competition" / meta.path.iloc[i]))
                     for i in rows]).to(DEV)
    ti = torch.as_tensor(tasks[rows], dtype=torch.long, device=DEV)
    tri = torch.as_tensor(truth[start:start + CHUNK] - 1, dtype=torch.long, device=DEV)
    tgi = (tri + 1) % 20
    with torch.no_grad():
        lg, gt = scorer(x, ti)
    clean_all.append((lg.cpu().numpy(), gt.cpu().numpy()))
    for snr in SNRS:
        for goal in "ABCD":
            lg, gt = pgd(x, ti, tri, tgi, goal, snr)
            buf[(snr, goal)].append((lg, gt))
    print(f"chunk {start // CHUNK + 1}/{(len(subset) + CHUNK - 1) // CHUNK} "
          f"elapsed {time.time() - t0:.0f}s", flush=True)

clean = metrics(np.concatenate([c[0] for c in clean_all]), np.concatenate([c[1] for c in clean_all]), truth)
conditions = []
for (snr, goal), parts in buf.items():
    m = metrics(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), truth)
    m.update({"snr": snr, "goal": goal})
    conditions.append(m)
worst_ra = min(conditions, key=lambda r: r["RA"])
worst_safe = min(conditions, key=lambda r: r["ASafeR"])
out = {"candidate": TAU_KEY, "tau": tau, "n_trials": int(len(subset)),
       "per_task": {str(t): int((tasks[subset] == t).sum()) for t in range(4)},
       "clean": clean, "conditions": conditions,
       "RA": worst_ra["RA"], "ASafeR": worst_safe["ASafeR"],
       "worst_RA": worst_ra, "worst_ASafeR": worst_safe,
       "protocol": {"subset": "Train 0.6<frac<=0.8, cap 400/task, seed 20261115",
                    "snr_db": list(SNRS), "steps": STEPS, "goals": "A,B,C,D",
                    "radius": "per-trial L2 of demeaned waveform / 10^(SNR/20)",
                    "note": "self-built white-box PGD on the E249 C2 deployment model, not E191 branch-A"},
       "seconds": time.time() - t0}
(OUT / "results" / "gates_attack.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("n_trials", "clean", "RA", "ASafeR", "worst_RA", "worst_ASafeR", "seconds")}, indent=2))
