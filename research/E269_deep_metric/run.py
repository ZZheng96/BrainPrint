"""E269: fixed deep metric encoder with strict Train-only LOSO/enrollment evaluation.

No Test data or test metadata is read. The recipe is frozen before late-window metrics
are aggregated. Run --validate first, then run without arguments for all 40 folds.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "BrainPrint_cache"
WEIGHTS = HERE / "weights"
LOG = HERE / "run.log"
DEVICE = torch.device("cuda")
EMB = 128

# Frozen before any late-window result is inspected. Do not tune from late scores.
CFG = {
    "experiment": "E269_deep_metric",
    "fraction": "trial/max(trial) per subject x task",
    "fit_max": 0.6,
    "cal_min_exclusive": 0.6,
    "cal_max": 0.8,
    "late_min_exclusive": 0.8,
    "tasks": ["EC", "EO", "Imagery", "SSVEP"],
    "model": {
        "stem": "depthwise k65/s2 + pointwise 18->64",
        "blocks": [[17, 2], [9, 2]],
        "width": 64,
        "gn_groups": 8,
        "pool": "mean+std+max",
        "embedding": 128,
        "scale": 24.0,
        "margin": 0.15,
    },
    "train": {
        "epochs": 20,
        "batch_size": 128,
        "optimizer": "AdamW",
        "lr": 0.002,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "amp": True,
        "supcon_weight": 0.1,
        "supcon_temperature": 0.1,
        "consistency_weight": 0.1,
        "seed_loso": 269000,
        "seed_registration_offset": 10000,
    },
    "train_augmentation": {
        "channel_gain": 0.10,
        "channel_dropout": 0.05,
        "noise_snr_db": [15.0, 25.0],
        "time_shift": 60,
        "crop_fraction": [0.9, 1.0],
        "spectral_tilt": 0.35,
        "partial_common_average": 0.10,
    },
    "late_stress": {
        "seed": 269777,
        "channel_gain": 0.20,
        "channel_dropout": 0.10,
        "noise_snr_db": [12.0, 20.0],
        "time_shift": 80,
        "spectral_tilt": 0.70,
        "partial_common_average": 0.25,
        "batch_size": 256,
    },
    "score": {
        "prototypes": 4,
        "prototype_rule": "four equal trial-ordered blocks, mean then L2 normalize",
        "base": "s1 + 0.5*(s1-s2)",
        "tau_quantile": 0.10,
        "new_per_task": 5,
        "new_total": 20,
    },
}

Y = None
T = None
TRIAL = None
FRAC = None
TASKS = None


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def tens(a):
    return torch.as_tensor(a, device=DEVICE)


def load_data():
    global Y, T, TRIAL, FRAC, TASKS
    meta = pd.read_csv(CACHE / "train_metadata.csv")
    assert len(meta) == 11304
    TASKS = sorted(meta.task.unique().tolist())
    assert TASKS == CFG["tasks"]
    task_np = meta.task.map({v: i for i, v in enumerate(TASKS)}).to_numpy(np.int64)
    frac = meta.trial.to_numpy(np.float64) / np.maximum(
        meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(np.float64), 1.0
    )
    Y = tens(meta.subject.to_numpy(np.int64))
    T = tens(task_np)
    TRIAL = tens(meta.trial.to_numpy(np.int64))
    FRAC = tens(frac)
    assert set(meta.subject.unique()) == set(range(1, 21))

    raw_path = ROOT / "research/E229_raw_robust_encoder/cache/train_raw.npy"
    mm = np.load(raw_path, mmap_mode="r")
    assert mm.shape == (len(meta), 18, 1200) and mm.dtype == np.float32
    raw = torch.empty(mm.shape, dtype=torch.float32, device=DEVICE)
    for start in range(0, len(mm), 256):
        stop = min(start + 256, len(mm))
        x = torch.from_numpy(np.ascontiguousarray(mm[start:stop])).to(DEVICE)
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        raw[start:stop] = x
    assert bool(torch.isfinite(raw).all())

    clean = torch.from_numpy(np.load(CACHE / "e262_wave_pair/clean.npy")).to(DEVICE)
    stress = torch.from_numpy(np.load(CACHE / "e262_wave_pair/stress.npy")).to(DEVICE)
    assert clean.shape == stress.shape == (len(meta), 855)
    assert bool(torch.isfinite(clean).all()) and bool(torch.isfinite(stress).all())
    log(f"Train loaded on GPU: raw={tuple(raw.shape)}, E254={tuple(clean.shape)}, rows={len(meta)}")
    return raw, clean, stress


class DepthPoint(nn.Module):
    def __init__(self, channels, out_channels, kernel, stride):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel, stride=stride,
                            padding=kernel // 2, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(CFG["model"]["gn_groups"], out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class MetricNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = DepthPoint(18, 64, 65, 2)
        self.blocks = nn.Sequential(*[DepthPoint(64, 64, k, s) for k, s in CFG["model"]["blocks"]])
        self.proj = nn.Linear(64 * 3, EMB)
        self.head = nn.Parameter(torch.empty(n_classes, EMB))
        nn.init.xavier_uniform_(self.head)

    def embed(self, x):
        h = self.blocks(self.stem(x))
        pooled = torch.cat([
            h.mean(dim=-1),
            h.var(dim=-1, unbiased=False).clamp_min(1e-8).sqrt(),
            h.amax(dim=-1),
        ], dim=1)
        return F.normalize(self.proj(pooled).float(), dim=-1)

def arc_cos_logits(weight, z, labels):
    """CosFace logits; labels select the fixed margin target columns."""
    with torch.autocast(device_type="cuda", enabled=False):
        cosine = F.linear(z.float(), F.normalize(weight.float(), dim=-1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = F.one_hot(labels, cosine.shape[1]).to(cosine.dtype)
        return (cosine - one_hot * float(CFG["model"]["margin"])) * float(CFG["model"]["scale"])


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def crop_resample(x, gen, cfg):
    b, c, n = x.shape
    lo, hi = cfg["crop_fraction"]
    fraction = lo + (hi - lo) * torch.rand(b, generator=gen, device=x.device)
    length = (n * fraction).round().long().clamp(2, n)
    max_start = (n - length + 1).float()
    start = (torch.rand(b, generator=gen, device=x.device) * max_start).floor().long()
    step = (length - 1).float() / float(n - 1)
    pos = start[:, None].float() + torch.arange(n, device=x.device)[None, :] * step[:, None]
    pos = pos.clamp(0, n - 1)
    left = pos.floor().long()
    right = (left + 1).clamp_max(n - 1)
    weight = (pos - left.float())[:, None, :]
    a = torch.gather(x, 2, left[:, None, :].expand(b, c, n))
    z = torch.gather(x, 2, right[:, None, :].expand(b, c, n))
    return a + weight * (z - a)


def perturb(x, gen, cfg, training):
    """GPU-only waveform perturbation on already standardized samples."""
    b, c, n = x.shape
    gain = float(cfg["channel_gain"])
    x = x * (1.0 + gain * (2.0 * torch.rand(b, c, 1, generator=gen, device=x.device) - 1.0))
    drop = torch.rand(b, c, 1, generator=gen, device=x.device) < float(cfg["channel_dropout"])
    x = x.masked_fill(drop, 0.0)
    shift_max = int(cfg["time_shift"])
    shift = torch.randint(-shift_max, shift_max + 1, (b,), generator=gen, device=x.device)
    index = (torch.arange(n, device=x.device)[None, :] - shift[:, None]) % n
    x = torch.gather(x, 2, index[:, None, :].expand(b, c, n))
    if training:
        x = crop_resample(x, gen, cfg)
    alpha = (2.0 * torch.rand(b, 1, 1, generator=gen, device=x.device) - 1.0) * float(cfg["spectral_tilt"])
    freq = torch.linspace(0.0, 1.0, n // 2 + 1, device=x.device)[None, None, :]
    weight = (freq + 0.05).pow(alpha)
    weight = weight / weight.mean(dim=-1, keepdim=True)
    x = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * weight, n=n, dim=-1)
    x = x - float(cfg["partial_common_average"]) * x.mean(dim=1, keepdim=True)
    snr_lo, snr_hi = cfg["noise_snr_db"]
    snr = snr_lo + (snr_hi - snr_lo) * torch.rand(b, c, 1, generator=gen, device=x.device)
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    noise = torch.randn(x.shape, generator=gen, device=x.device)
    x = x + noise * rms / torch.pow(10.0, snr / 20.0)
    return x


def supcon(z, labels):
    temp = float(CFG["train"]["supcon_temperature"])
    sim = z @ z.T / temp
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -torch.inf)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    count = positive.sum(1)
    valid = count > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    return -(log_prob.masked_fill(~positive, 0.0).sum(1)[valid] / count[valid]).mean()


def train_model(raw, fit, classes, seed, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    model = MetricNet(len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["train"]["lr"],
                            weight_decay=CFG["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["train"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=CFG["train"]["amp"])
    rows = torch.where(fit)[0]
    class_t = tens(sorted(classes))
    labels = torch.searchsorted(class_t, Y[rows])
    history = []
    epochs = CFG["train"]["epochs"]
    batch_size = CFG["train"]["batch_size"]
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(rows), generator=gen, device=DEVICE)
        loss_sum = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(rows), batch_size):
            pick = order[start:start + batch_size]
            clean = raw[rows[pick]]
            labels_b = labels[pick]
            gen_aug = gen
            aug = perturb(clean, gen_aug, CFG["train_augmentation"], True)
            both = torch.cat([clean, aug], dim=0)
            labels_2 = labels_b.repeat(2)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG["train"]["amp"]):
                z = model.embed(both)
                logits = arc_cos_logits(model.head, z, labels_2)
                zc, za = z.chunk(2, dim=0)
                ce = F.cross_entropy(logits, labels_2)
                contrast = supcon(z, labels_2)
                consistency = (1.0 - (zc * za).sum(dim=1)).mean()
                loss = (ce + CFG["train"]["supcon_weight"] * contrast
                        + CFG["train"]["consistency_weight"] * consistency)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            n = len(labels_2)
            loss_sum += float(loss.detach()) * n
            correct += int((logits.detach().argmax(1) == labels_2).sum())
            seen += n
        sched.step()
        item = {
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "seconds": time.time() - started,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(item)
        if epoch in (1, 5, 10, 15, epochs):
            log(f"{tag} epoch {epoch}/{epochs} loss={item['loss']:.5f} "
                f"acc={item['train_accuracy']:.4f} seconds={item['seconds']:.2f}")
    model.eval()
    return model, history


@torch.inference_mode()
def encode(model, raw, rows, stress=False):
    out = torch.empty(len(rows), EMB, dtype=torch.float32, device=DEVICE)
    gen = None
    batch_size = CFG["late_stress"]["batch_size"] if stress else 512
    if stress:
        gen = torch.Generator(device=DEVICE).manual_seed(CFG["late_stress"]["seed"])
    for start in range(0, len(rows), batch_size):
        x = raw[rows[start:start + batch_size]]
        if stress:
            x = perturb(x, gen, CFG["late_stress"], False)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG["train"]["amp"]):
            out[start:start + len(x)] = model.embed(x)
    assert bool(torch.isfinite(out).all())
    return out


def fold_spec(kind, fold):
    ids = list(range(1, 21))
    if kind == "loso":
        held = fold
        old = [p for p in ids if p != held]
        new, unknown = [], [held]
        seed = CFG["train"]["seed_loso"] + fold
    else:
        order = ids[fold:] + ids[:fold]
        old, new, unknown = order[:14], order[14:17], order[17:]
        seed = CFG["train"]["seed_loso"] + CFG["train"]["seed_registration_offset"] + fold
    old_mask = torch.isin(Y, tens(old))
    fit = old_mask & (FRAC <= CFG["fit_max"])
    enroll = fit.clone()
    for person in new:
        for task in range(4):
            ix = torch.where((Y == person) & (T == task) & (FRAC <= CFG["fit_max"]))[0]
            ix = ix[TRIAL[ix].argsort()]
            assert len(ix) >= CFG["score"]["new_per_task"]
            enroll[ix[:CFG["score"]["new_per_task"]]] = True
        assert int((enroll & (Y == person)).sum()) == CFG["score"]["new_total"]
    cal = old_mask & (FRAC > CFG["cal_min_exclusive"]) & (FRAC <= CFG["cal_max"])
    late = FRAC > CFG["late_min_exclusive"]
    assert not bool((enroll & cal).any()) and not bool((enroll & late).any()) and not bool((cal & late).any())
    assert not bool((fit & torch.isin(Y, tens(new + unknown))).any())
    assert not bool((cal & torch.isin(Y, tens(new + unknown))).any())
    assert bool((late & torch.isin(Y, tens(old + new + unknown))).sum() == late.sum())
    for person in old:
        for task in range(4):
            assert int((fit & (Y == person) & (T == task)).sum()) >= 4
    return dict(kind=kind, fold=fold, old=old, new=new, unknown=unknown,
                people=old + new, fit=fit, enroll=enroll, cal=cal, late=late, seed=seed)


def templates(z, enroll, people):
    dim = z.shape[1]
    p = z.new_zeros(4, len(people), 4, dim)
    valid = torch.zeros(4, len(people), 4, dtype=torch.bool, device=z.device)
    for task in range(4):
        for j, person in enumerate(people):
            ix = torch.where(enroll & (Y == person) & (T == task))[0]
            ix = ix[TRIAL[ix].argsort()]
            assert len(ix) >= 4
            for k in range(4):
                a, b = k * len(ix) // 4, (k + 1) * len(ix) // 4
                assert a < b
                p[task, j, k] = p[task, j, k] + z[ix[a:b]].mean(dim=0)
                valid[task, j, k] = True
    return F.normalize(p, dim=-1), valid


def score(z, p, valid, rows, people):
    q = F.normalize(z[rows].float(), dim=-1)
    tasks = T[rows]
    similarities = q.new_empty(len(q), len(people))
    for task in range(4):
        mask = tasks == task
        sim = q[mask] @ p[task].flatten(0, 1).T
        sim = sim.view(-1, len(people), 4).masked_fill(~valid[task], -torch.inf)
        similarities[mask] = sim.amax(dim=-1)
    top, index = similarities.topk(2, dim=1)
    base = top[:, 0] + 0.5 * (top[:, 0] - top[:, 1])
    pred = tens(people)[index[:, 0]]
    assert bool(torch.isfinite(base).all())
    return base, pred


def macro_f1(truth, pred):
    values = []
    for person in np.union1d(truth, pred):
        tp = int(((truth == person) & (pred == person)).sum())
        denom = int((truth == person).sum() + (pred == person).sum())
        values.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(values)) if values else float("nan")


def auc(known, scores):
    positive = scores[known]
    negative = scores[~known]
    n = len(positive)
    ranks = rankdata(np.concatenate([positive, negative]))[:n]
    return float((ranks.sum() - n * (n + 1) / 2) / (n * len(negative)))


def mean_or_nan(x):
    return float(np.mean(x)) if len(x) else float("nan")


def late_metrics(base, pred, spec, tau):
    late = spec["late"]
    truth = Y[late].cpu().numpy()
    scores = base.cpu().numpy()
    predictions = pred.cpu().numpy()
    known = ~np.isin(truth, spec["unknown"])
    unknown = ~known
    old = np.isin(truth, spec["old"])
    new = np.isin(truth, spec["new"])
    accept = scores >= tau
    correct = predictions == truth
    out = {
        "tau": float(tau),
        "AUROC": auc(known, scores),
        "U95": float((scores[unknown] < np.quantile(scores[known], 0.05)).mean()),
        "R90": float((scores[unknown] < np.quantile(scores[known], 0.10)).mean()),
        "closed_macro_F1": macro_f1(truth[known], predictions[known]),
        "closed_accuracy": float(correct[known].mean()),
        "old_e2e": mean_or_nan((correct & accept)[old]),
        "known_accept": float(accept[known].mean()),
        "unknown_reject": float((~accept)[unknown].mean()),
        "accepted_misrank": mean_or_nan((~correct & accept)[known]),
        "n_known": int(known.sum()),
        "n_unknown": int(unknown.sum()),
        "n_old": int(old.sum()),
        "n_new": int(new.sum()),
    }
    if spec["new"]:
        out["new_e2e"] = mean_or_nan((correct & accept)[new])
        out["new_closed_accuracy"] = float(correct[new].mean())
    return out


def evaluate(z_clean, z_stress, spec):
    p, valid = templates(z_clean, spec["enroll"], spec["people"])
    cal_score, _ = score(z_clean, p, valid, spec["cal"], spec["people"])
    tau = float(np.quantile(cal_score.cpu().numpy(), CFG["score"]["tau_quantile"]))
    clean_score, clean_pred = score(z_clean, p, valid, spec["late"], spec["people"])
    stress_score, stress_pred = score(z_stress, p, valid, spec["late"], spec["people"])
    result = {
        "clean": late_metrics(clean_score, clean_pred, spec, tau),
        "stress": late_metrics(stress_score, stress_pred, spec, tau),
    }
    return result, p, valid, tau


def project_e254(clean, fit):
    """Research E254 projection, fitted only on this fold's old early rows."""
    mu = torch.stack([clean[fit & (T == i)].mean(0) for i in range(4)])
    sd = torch.stack([clean[fit & (T == i)].std(0).clamp_min(1e-6) for i in range(4)])
    z = (clean - mu[T]) / sd[T]
    center = z[fit].mean(0)
    z = z - center
    _, basis = torch.linalg.eigh(z[fit].T @ z[fit])
    basis = basis[:, -256:]
    q = z @ basis
    keys = Y[fit].unique()
    residual = torch.cat([q[fit & (Y == person)] - q[fit & (Y == person)].mean(0) for person in keys])
    covariance = residual.T @ residual / (len(residual) - len(keys))
    eigenvalue, eigenvector = torch.linalg.eigh(
        0.95 * covariance + 0.05 * covariance.diagonal().mean() * torch.eye(256, device=DEVICE)
    )
    white = (eigenvector * eigenvalue.clamp_min(1e-6).rsqrt()) @ eigenvector.T
    art = dict(mu=mu, sd=sd, center=center, basis=basis, white=white)
    return q @ white, art


def apply_e254(x, art):
    return (((x - art["mu"][T]) / art["sd"][T] - art["center"]) @ art["basis"]) @ art["white"]


def cpu_tensor_dict(d):
    return {k: v.detach().cpu() for k, v in d.items()}


def checkpoint_path(kind, fold):
    return WEIGHTS / (f"loso_h{fold:02d}.pt" if kind == "loso" else f"registration_h{fold:02d}.pt")


def run_fold(raw, clean_feature, stress_feature, kind, fold):
    path = checkpoint_path(kind, fold)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("complete") and saved.get("config") == CFG:
            log(f"{kind} fold={fold:02d} complete checkpoint reused")
            return saved
        raise RuntimeError(f"Incomplete or config-mismatched checkpoint: {path}")
    started = time.time()
    spec = fold_spec(kind, fold)
    tag = f"{kind}_h{fold:02d}"
    log(f"{tag} start old={len(spec['old'])} new={len(spec['new'])} unknown={len(spec['unknown'])} "
        f"fit={int(spec['fit'].sum())} enroll={int(spec['enroll'].sum())} cal={int(spec['cal'].sum())} "
        f"late={int(spec['late'].sum())} seed={spec['seed']}")

    q, projection = project_e254(clean_feature, spec["fit"])
    qs = apply_e254(stress_feature, projection)
    baseline_result, bp, bv, btau = evaluate(q, qs, spec)
    del qs

    classes = sorted(spec["old"])
    model, history = train_model(raw, spec["fit"], classes, spec["seed"], tag)
    needed = torch.where(spec["enroll"] | spec["cal"] | spec["late"])[0]
    z = torch.zeros(len(Y), EMB, dtype=torch.float32, device=DEVICE)
    z[needed] = encode(model, raw, needed, stress=False)
    late_rows = torch.where(spec["late"])[0]
    zs = torch.zeros_like(z)
    # Global late-row order and a reset fixed seed make stress identical in every fold.
    zs[late_rows] = encode(model, raw, late_rows, stress=True)
    new_result, p, valid, tau = evaluate(z, zs, spec)

    saved = {
        "complete": True,
        "experiment": CFG["experiment"],
        "kind": kind,
        "fold": fold,
        "seed": spec["seed"],
        "config": copy.deepcopy(CFG),
        "classes": sorted(spec["old"]),
        "people": spec["people"],
        "old": spec["old"],
        "new": spec["new"],
        "unknown": spec["unknown"],
        "tasks": TASKS,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "parameters": parameter_count(model),
        "prototypes": p.detach().cpu(),
        "valid": valid.detach().cpu(),
        "tau": tau,
        "training_log": history,
        "train_seconds": float(sum(x["seconds"] for x in history)),
        "fold_seconds": float(time.time() - started),
        "new_result": new_result,
        "E254_result": baseline_result,
        "E254": {
            "projection": cpu_tensor_dict(projection),
            "prototypes": bp.detach().cpu(),
            "valid": bv.detach().cpu(),
            "tau": btau,
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)
    log(f"{tag} complete seconds={saved['fold_seconds']:.1f} params={saved['parameters']} "
        f"final_loss={history[-1]['loss']:.5f} tau={tau:.6f}")
    del model, z, zs, q, projection
    torch.cuda.empty_cache()
    return saved


def validate(raw, clean_feature, stress_feature):
    log("VALIDATE: checking all 40 fold masks and enrollment counts")
    for fold in range(1, 21):
        fold_spec("loso", fold)
    for fold in range(20):
        spec = fold_spec("registration", fold)
        mapped = torch.searchsorted(tens(sorted(spec["old"])), Y[spec["fit"]])
        assert int(mapped.min()) >= 0 and int(mapped.max()) < len(spec["old"])
        for person in spec["new"]:
            assert int((spec["enroll"] & (Y == person)).sum()) == 20
            for task in range(4):
                assert int((spec["enroll"] & (Y == person) & (T == task)).sum()) == 5
    spec = fold_spec("loso", 1)
    torch.manual_seed(269999)
    gen = torch.Generator(device=DEVICE).manual_seed(269999)
    model = MetricNet(len(spec["old"])).to(DEVICE)
    assert parameter_count(model) < 200000
    rows = torch.where(spec["fit"])[0][:128]
    labels = torch.searchsorted(tens(spec["old"]), Y[rows])
    x = raw[rows]
    xa = perturb(x, gen, CFG["train_augmentation"], True)
    z = model.embed(torch.cat([x, xa]))
    labels2 = labels.repeat(2)
    logits = arc_cos_logits(model.head, z, labels2)
    loss = F.cross_entropy(logits, labels2) + 0.1 * supcon(z, labels2)
    loss = loss + 0.1 * (1.0 - (z.chunk(2)[0] * z.chunk(2)[1]).sum(1)).mean()
    loss.backward()
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(logits).all())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    model.eval()
    needed = torch.where(spec["enroll"] | spec["cal"])[0]
    zfull = torch.zeros(len(Y), EMB, device=DEVICE)
    zfull[needed] = encode(model, raw, needed)
    p, valid = templates(zfull, spec["enroll"], spec["people"])
    cal_score, _ = score(zfull, p, valid, spec["cal"], spec["people"])
    assert bool(torch.isfinite(cal_score).all())

    late_rows = torch.where(spec["late"])[0]
    a = encode(model, raw, late_rows[:256], stress=True)
    b = encode(model, raw, late_rows[:256], stress=True)
    assert torch.equal(a, b)
    q, art = project_e254(clean_feature, spec["fit"])
    qs = apply_e254(stress_feature, art)
    assert bool(torch.isfinite(q).all()) and bool(torch.isfinite(qs).all())
    bp, bv = templates(q, spec["enroll"], spec["people"])
    cs, _ = score(q, bp, bv, spec["cal"], spec["people"])
    ss, _ = score(qs, bp, bv, spec["late"], spec["people"])
    assert bool(torch.isfinite(cs).all()) and bool(torch.isfinite(ss).all())
    log(f"VALIDATE PASS: 40 masks, new=20/person (5/task), forward/backward, finite scores, "
        f"deterministic global-order stress, params={parameter_count(model)}")


def metric_summary(rows):
    keys = [k for k, v in rows[0].items() if isinstance(v, float) and math.isfinite(v)]
    out = {}
    for key in keys:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        worst = float(values.max()) if key == "accepted_misrank" else float(values.min())
        out[key] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "worst": worst,
        }
    return out


def paired_summary(new_rows, base_rows):
    out = {}
    for key in new_rows[0]:
        if key not in base_rows[0] or not isinstance(new_rows[0][key], float):
            continue
        delta = np.asarray([n[key] - b[key] for n, b in zip(new_rows, base_rows)], dtype=np.float64)
        if not np.isfinite(delta).all():
            continue
        improvement = -delta if key == "accepted_misrank" else delta
        out[key] = {
            "mean": float(delta.mean()),
            "min": float(delta.min()),
            "max": float(delta.max()),
            "improved_folds": int((improvement > 0).sum()),
            "harmed_folds": int((improvement < 0).sum()),
        }
    return out


def scenario_rows(saved, method):
    key = "new_result" if method == "new" else "E254_result"
    rows = []
    for s in saved:
        r = s[key]
        values = {
            "old_time": r["clean"]["old_e2e"],
            "old_device": r["stress"]["old_e2e"],
            "new_time": r["clean"]["new_e2e"],
            "URR": r["clean"]["unknown_reject"],
        }
        values["proxy_score"] = float(np.mean(list(values.values())))
        rows.append(values)
    return rows


def fmt(x):
    return f"{float(x):.4f}"


def summary_of(summary, kind, method, view):
    return summary[method][kind][view]


def build_report(results):
    s = results["summary"]
    new_l_c = summary_of(s, "loso", "new", "clean")
    new_l_s = summary_of(s, "loso", "new", "stress")
    new_r_c = summary_of(s, "registration", "new", "clean")
    new_r_s = summary_of(s, "registration", "new", "stress")
    decision = results["promotion"]
    lines = [
        "# E269 深度度量编码器：严格 Train-only LOSO 与新用户注册模拟",
        "",
        f"> **官方 Test AUC = N/A；官方 Test F1 = N/A。** 本实验未读取或加载任何 Test 数据、test_metadata 或隐藏标签。",
        f"> **当前新模型 LOSO clean AUROC = {fmt(new_l_c['AUROC']['mean'])}，闭集 macro-F1 = {fmt(new_l_c['closed_macro_F1']['mean'])}；"
        f"stress AUROC = {fmt(new_l_s['AUROC']['mean'])}，闭集 macro-F1 = {fmt(new_l_s['closed_macro_F1']['mean'])}。**",
        f"> **注册模拟 clean AUROC = {fmt(new_r_c['AUROC']['mean'])}，注册身份闭集 macro-F1 = {fmt(new_r_c['closed_macro_F1']['mean'])}；"
        f"stress AUROC = {fmt(new_r_s['AUROC']['mean'])}，闭集 macro-F1 = {fmt(new_r_s['closed_macro_F1']['mean'])}。**",
        f"> **晋级结论：{'建议后续替换候选（仍须独立复核）' if decision['promote'] else '不晋级，不建议替换'}。"
        f"四场景等权 proxy 配对差 = {decision['proxy_mean_delta']:+.4f}；本轮未修改 final/final2。**",
        "",
        "## 1. 完成度与固定配方",
        "",
        f"- 实际完成 **{results['completed']['loso']}/20 个 LOSO 折 + {results['completed']['registration']}/20 个注册模拟折**；无早停、无按晚段选折。",
        "- 每折仅用老身份 `f<=0.6` 训练；老用户原型使用老身份全部早 60%；新用户每人每任务最早 5 条、共 20 条注册，且均满足 `f<=0.6`。",
        "- 阈值仅取 14/19 个老身份中段 `(0.6,0.8]` clean base 的第 10 百分位；所有相关身份 `f>0.8` 只评估。查询不进入自身模板。",
        "- 每身份×任务按 trial 顺序四等分，各自均值后 L2 归一化；同任务余弦，身份取四原型最大，`base=s1+0.5*(s1-s2)`。",
        "- 输入一次性在 GPU 上对每个样本每通道独立去均值、除以 std（clamp=1e-6）；网络没有额外输入 RMS/方差归一化。",
        "- 网络为 depthwise temporal k65/s2 + pointwise 到 64 宽，两个 depthwise/pointwise block，GroupNorm+ELU，mean/std/max 池化，Linear→128 并 L2 normalize；CosFace scale=24、margin=0.15。",
        "- CE + 固定 0.1 supervised contrastive（temp=0.1）+ 固定 0.1 clean/aug 余弦一致性；AdamW lr=0.002、wd=1e-4、batch=128、20 epochs、cosine、AMP。",
        f"- 20 类时总参数量 **{results['parameters_20class']:,}**（每折实际头大小随老身份数变化，均 <200k）。LOSO seed=269000+h；注册 seed=279000+h。",
        "- Train-only 增强包括通道增益±10%、通道 dropout=0.05、SNR15–25dB 白噪声、循环时移±60、0.9–1 随机 crop 后线性重采样、谱倾斜±0.35、部分公共平均参考0.1；全部在标准化后的 GPU 波形上执行。",
        "- 技术修复：首次运行在 registration_h07 发现循环移位后的 old 列表非升序，初版 `searchsorted` 标签映射因此越界。修复仅为 `classes=sorted(old)`；此前已完成的 h00–h06 的 old 列表本就升序，数值不受影响。模型、epoch、增强、阈值和评估配方均未改变。",
        "",
        "## 2. 四场景主代理（注册模拟，20 折）",
        "",
        "均值为 20 折宏平均；最差折对错排以外指标取最小。`old_device` 使用合成 stress，**不是实际跨设备成绩**。",
        "",
        "|场景|E254 均值 / 最差折|新模型均值 / 最差折|配对差均值|",
        "|---|---:|---:|---:|",
    ]
    for scenario in ["old_time", "old_device", "new_time", "URR", "proxy_score"]:
        b = s["scenarios"]["E254_research"][scenario]
        n = s["scenarios"]["new"][scenario]
        d = s["scenario_paired_delta"][scenario]["mean"]
        lines.append(f"|{scenario}|{fmt(b['mean'])} / {fmt(b['worst'])}|{fmt(n['mean'])} / {fmt(n['worst'])}|{d:+.4f}|")
    lines += [
        "",
        "等权 `proxy_score=(old_time+old_device+new_time+URR)/4`。晋级判定预先固定为四个场景均值均不下降且 proxy 均值实际提高；"
        f"本次 `not_harms={decision['not_harms']}`，`actual_improvement={decision['actual_improvement']}`。",
        "",
        "## 3. LOSO：新模型与 E254 研究基线配对对照",
        "",
        "F1 为晚段已知老身份的闭集 macro-F1；old_e2e 要求身份正确且接受。U95/R90 是晚段已知接受率 95%/90% 时的未知拒识诊断，不是部署阈值。accepted_misrank 的最差折取最大。",
        "",
        "|方法|视图|AUROC 均值 / 最差|F1 均值 / 最差|old_e2e 均值 / 最差|known_accept|unknown_reject|accepted_misrank|U95|R90|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["E254_research", "new"]:
        for view in ["clean", "stress"]:
            r = s[method]["loso"][view]
            lines.append(
                f"|{method}|{view}|{fmt(r['AUROC']['mean'])} / {fmt(r['AUROC']['worst'])}|"
                f"{fmt(r['closed_macro_F1']['mean'])} / {fmt(r['closed_macro_F1']['worst'])}|"
                f"{fmt(r['old_e2e']['mean'])} / {fmt(r['old_e2e']['worst'])}|{fmt(r['known_accept']['mean'])}|"
                f"{fmt(r['unknown_reject']['mean'])}|{fmt(r['accepted_misrank']['mean'])}|"
                f"{fmt(r['U95']['mean'])}|{fmt(r['R90']['mean'])}|"
            )
    for view in ["clean", "stress"]:
        d = s["paired_delta"]["loso"][view]
        lines += ["", f"**{view} 配对差（新模型−E254）**：AUROC {d['AUROC']['mean']:+.4f}，"
                  f"F1 {d['closed_macro_F1']['mean']:+.4f}，old_e2e {d['old_e2e']['mean']:+.4f}，"
                  f"unknown_reject {d['unknown_reject']['mean']:+.4f}，accepted_misrank {d['accepted_misrank']['mean']:+.4f}。"]
    lines += [
        "",
        "## 4. 注册模拟详细结果",
        "",
        "|方法|视图|AUROC|闭集 F1|old_e2e|new_e2e|new closed acc|known_accept|unknown_reject|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["E254_research", "new"]:
        for view in ["clean", "stress"]:
            r = s[method]["registration"][view]
            lines.append(
                f"|{method}|{view}|{fmt(r['AUROC']['mean'])}|{fmt(r['closed_macro_F1']['mean'])}|"
                f"{fmt(r['old_e2e']['mean'])}|{fmt(r['new_e2e']['mean'])}|{fmt(r['new_closed_accuracy']['mean'])}|"
                f"{fmt(r['known_accept']['mean'])}|{fmt(r['unknown_reject']['mean'])}|"
            )
    d = s["paired_delta"]["registration"]
    lines += [
        "",
        f"clean 配对差：AUROC {d['clean']['AUROC']['mean']:+.4f}，F1 {d['clean']['closed_macro_F1']['mean']:+.4f}，"
        f"old_e2e {d['clean']['old_e2e']['mean']:+.4f}，new_e2e {d['clean']['new_e2e']['mean']:+.4f}，"
        f"URR {d['clean']['unknown_reject']['mean']:+.4f}。",
        f"stress 配对差：AUROC {d['stress']['AUROC']['mean']:+.4f}，F1 {d['stress']['closed_macro_F1']['mean']:+.4f}，"
        f"old_e2e {d['stress']['old_e2e']['mean']:+.4f}，new_e2e {d['stress']['new_e2e']['mean']:+.4f}，"
        f"URR {d['stress']['unknown_reject']['mean']:+.4f}。",
        "",
        "## 5. 权重、运行与复现",
        "",
        "- 新模型及折级原型、valid、tau、classes/people、tasks、固定配置和逐 epoch 训练日志均保存在 "
        "`research/E269_deep_metric/weights/loso_h01.pt`–`loso_h20.pt` 与 `registration_h00.pt`–`registration_h19.pt`。",
        "- 同一权重文件还保存 E254 的 fold-fit projection、原型、valid、tau 与逐折结果，便于严格配对复核。",
        f"- 40 折 fold 总耗时 **{results['timing']['fold_seconds_sum']/60:.1f} min**；其中新模型训练 **{results['timing']['train_seconds_sum']/60:.1f} min**；"
        f"最终汇总进程墙钟 **{results['timing']['runtime_seconds']/60:.1f} min**。",
        "- 复跑：`.venv\\Scripts\\python.exe research/E269_deep_metric/run.py --validate`，然后 `.venv\\Scripts\\python.exe research/E269_deep_metric/run.py`。完整且配置一致的折自动断点复用。",
        "- 逐折训练及复用记录见 `research/E269_deep_metric/run.log`；全部折级均值/min/max/配对差见 `RESULTS.json`。",
        "",
        "## 6. 严格边界与局限",
        "",
        "- 本实验只使用 Train 元数据与 Train 缓存；官方 Test AUC/F1 始终为 N/A，不能从本报告推断隐藏测试成绩。",
        "- E254 是按 E259 投影规则在每个 fold 仅用 old fit 重新拟合的研究基线，clean/stress 分别使用 E262 的 clean.npy/stress.npy；**不是现包 final2 的精确双白化实现**。",
        "- late stress 在标准化波形上以固定 seed 按全局晚段行顺序即时生成：通道增益±20%、dropout=0.1、SNR12–20dB、谱倾斜±0.7、部分公共平均参考0.25、时移±80。波形不保存，所有折一致。",
        "- **合成 stress 不等同真实跨设备**；时间块来自同一可见 session，也不等同真实跨时段。40 折共享试次，不是 40 个独立数据集。",
        "- 新用户 backbone 只用 14 个 old 的早 60% 训练；new 注册只建模板，不更新 backbone。校准不看 new 或 unknown，晚段不参与调架构、epoch、增强、阈值或选折。",
        "- 晋级仅表示值得后续独立复核，本轮没有修改 final/final2，也没有生成部署模型。",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(saved_loso, saved_registration, runtime_seconds):
    fold_loso = []
    fold_registration = []
    for s in saved_loso:
        fold_loso.append({
            "fold": s["fold"], "seed": s["seed"], "classes": s["classes"], "people": s["people"],
            "new": s["new_result"], "E254_research": s["E254_result"],
            "paired_delta_new_minus_E254": {
                view: {k: s["new_result"][view][k] - s["E254_result"][view][k]
                       for k in s["new_result"][view] if isinstance(s["new_result"][view][k], float)}
                for view in ["clean", "stress"]
            },
            "train_seconds": s["train_seconds"], "fold_seconds": s["fold_seconds"],
            "training_log": s["training_log"],
        })
    for s in saved_registration:
        fold_registration.append({
            "fold": s["fold"], "seed": s["seed"], "old": s["old"], "new_people": s["new"], "unknown": s["unknown"],
            "new": s["new_result"], "E254_research": s["E254_result"],
            "paired_delta_new_minus_E254": {
                view: {k: s["new_result"][view][k] - s["E254_result"][view][k]
                       for k in s["new_result"][view] if isinstance(s["new_result"][view][k], float)}
                for view in ["clean", "stress"]
            },
            "train_seconds": s["train_seconds"], "fold_seconds": s["fold_seconds"],
            "training_log": s["training_log"],
        })
    summary = {"new": {}, "E254_research": {}, "paired_delta": {}}
    for kind, saved in [("loso", saved_loso), ("registration", saved_registration)]:
        summary["new"][kind] = {}
        summary["E254_research"][kind] = {}
        summary["paired_delta"][kind] = {}
        for view in ["clean", "stress"]:
            nr = [s["new_result"][view] for s in saved]
            br = [s["E254_result"][view] for s in saved]
            summary["new"][kind][view] = metric_summary(nr)
            summary["E254_research"][kind][view] = metric_summary(br)
            summary["paired_delta"][kind][view] = paired_summary(nr, br)
    new_scenarios = scenario_rows(saved_registration, "new")
    base_scenarios = scenario_rows(saved_registration, "E254")
    summary["scenarios"] = {
        "new": {k: metric_summary([{k: r[k]} for r in new_scenarios])[k] for k in new_scenarios[0]},
        "E254_research": {k: metric_summary([{k: r[k]} for r in base_scenarios])[k] for k in base_scenarios[0]},
    }
    summary["scenario_paired_delta"] = paired_summary(new_scenarios, base_scenarios)
    four = ["old_time", "old_device", "new_time", "URR"]
    not_harms = all(summary["scenario_paired_delta"][k]["mean"] >= 0.0 for k in four)
    actual_improvement = summary["scenario_paired_delta"]["proxy_score"]["mean"] > 0.0
    promotion = {
        "rule": "all four scenario mean paired deltas >= 0 and proxy_score mean delta > 0",
        "not_harms": bool(not_harms),
        "actual_improvement": bool(actual_improvement),
        "promote": bool(not_harms and actual_improvement),
        "proxy_mean_delta": summary["scenario_paired_delta"]["proxy_score"]["mean"],
        "note": "Research recommendation only; final/final2 were not modified.",
    }
    results = {
        "experiment": CFG["experiment"],
        "official": {"Test_AUC": "N/A", "Test_F1": "N/A", "reason": "No Test data or labels were read."},
        "config": copy.deepcopy(CFG),
        "completed": {"loso": len(saved_loso), "registration": len(saved_registration), "total": len(saved_loso) + len(saved_registration)},
        "technical_fixes": [
            "registration_h07 exposed unsorted cyclic old IDs; CE class mapping now uses sorted(old). "
            "Earlier completed registration folds already had sorted old IDs, so their numbers are unchanged. "
            "No model, epoch, augmentation, threshold, or evaluation recipe changed."
        ],
        "parameters_20class": parameter_count(MetricNet(20)),
        "folds": {"loso": fold_loso, "registration": fold_registration},
        "summary": summary,
        "promotion": promotion,
        "timing": {
            "fold_seconds_sum": float(sum(s["fold_seconds"] for s in saved_loso + saved_registration)),
            "train_seconds_sum": float(sum(s["train_seconds"] for s in saved_loso + saved_registration)),
            "runtime_seconds": float(runtime_seconds),
        },
        "limitations": [
            "Synthetic stress is not actual cross-device data.",
            "Late time fractions are within the visible session, not actual longitudinal sessions.",
            "E254 is the research projection baseline, not final2's exact double-whitening implementation.",
            "Folds share trials and are not independent replications.",
        ],
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--fresh-log", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    WEIGHTS.mkdir(exist_ok=True)
    if args.fresh_log:
        LOG.write_text("", encoding="utf-8")
    started = time.time()
    log(f"E269 start torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)} "
        f"validate={args.validate}")
    raw, clean_feature, stress_feature = load_data()
    if args.validate:
        validate(raw, clean_feature, stress_feature)
        return
    saved_loso = [run_fold(raw, clean_feature, stress_feature, "loso", fold) for fold in range(1, 21)]
    saved_registration = [run_fold(raw, clean_feature, stress_feature, "registration", fold) for fold in range(20)]
    assert len(saved_loso) == len(saved_registration) == 20
    results = aggregate(saved_loso, saved_registration, time.time() - started)
    (HERE / "RESULTS.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    build_report(results)
    sc = results["summary"]["scenarios"]["new"]
    log(f"ALL 40 FOLDS COMPLETE: LOSO clean AUC={results['summary']['new']['loso']['clean']['AUROC']['mean']:.6f} "
        f"F1={results['summary']['new']['loso']['clean']['closed_macro_F1']['mean']:.6f}; "
        f"proxy={sc['proxy_score']['mean']:.6f}; promote={results['promotion']['promote']}; "
        f"runtime={(time.time()-started)/60:.1f} min")
    log("RESULTS.json and REPORT.md written; no final/final2 files touched")


if __name__ == "__main__":
    main()
