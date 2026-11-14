"""E108: LOSO search for an S1-boosting signal on top of a fixed S2 operating point.

User's framing: some candidates already reach a high S2 acceptance. The problem
is S1 rejection. So we fix the S2 operating point and ask, under leave-one-
identity-out, which additional signal can reject more held-out identities
without losing enrolled acceptance.

Protocol (Train-only, no project Test):
  For each held-out identity h:
    - enrol the other 19, fit prototypes on the first 60% of each task block
    - known pool  = enrolled identities, 0.6-0.8 block
    - unknown pool = identity h, 0.6-0.8 block
    - fix the base threshold at the quantile that gives known acceptance 0.85
      (the S2 operating point)
    - then measure how much each auxiliary signal can raise unknown rejection
      while keeping known acceptance >= 0.85

Auxiliary signals tested (all computed from Train data only):
  S1. per-identity distance ratio (already the base)
  S2. absolute nearest-prototype distance
  S3. within-task z-scored nearest distance
  S4. embedding norm / energy of the raw trial
  S5. spectral high-band ratio (20-40 Hz power share)
  S6. per-identity calibrated ratio (ratio divided by that identity's own
      fit-block ratio distribution)
  S7. AND-gate of base ratio with S6
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e54 as E54
import run_e65
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / "BrainPrint_cache/train_metadata.csv"
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"
E95_CKPT = ROOT / "research" / "E95_e54_final20_candidate" / "model_checkpoint.pt"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def build_task_prototypes(feat, subjects, tasks, fit_mask, classes):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(feat[sel], dtype=np.float32).mean(0))
            keys.append((task, int(c)))
    return np.stack(protos), keys


def per_task_distances(feat, protos, keys, tasks):
    """Return nearest distance, second distance, predicted identity, per task."""
    d1 = np.full(len(feat), np.nan, dtype=np.float32)
    d2 = np.full(len(feat), np.nan, dtype=np.float32)
    pred = np.zeros(len(feat), dtype=np.int64)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        cls = np.asarray([keys[i][1] for i in cols])
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1[chunk] = d[np.arange(len(d)), order[:, 0]]
            d2[chunk] = d[np.arange(len(d)), order[:, 1]]
            pred[chunk] = cls[order[:, 0]]
    return d1, d2, pred


def embed_raw(model, raw, batch=256):
    out = []
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def high_band_ratio(raw):
    """Share of 20-40 Hz power in the 1-40 Hz band, per trial."""
    spec = torch.fft.rfft(raw, dim=-1)
    power = spec.abs().square()
    freq = torch.fft.rfftfreq(raw.shape[-1], d=1.0 / 300.0, device=raw.device)
    total = power[..., (freq >= 1) & (freq <= 40)].sum(-1)
    high = power[..., (freq >= 20) & (freq <= 40)].sum(-1)
    return (high / total.clamp_min(1e-12)).mean(1).cpu().numpy()


def evaluate_fold(held, feat, raw_emb, hb, subjects, tasks, f):
    enrolled = np.setdiff1d(CLASSES, [held])
    fit = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
    known_idx = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
    unknown_idx = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
    if len(unknown_idx) == 0:
        return None

    protos, keys = build_task_prototypes(feat, subjects, tasks, fit, enrolled)
    d1, d2, pred = per_task_distances(feat, protos, keys, tasks)
    ratio = d2 / np.maximum(d1, 1e-9)

    # Per-identity calibration: divide ratio by the identity's own fit-block median.
    ident_med = {}
    for c in enrolled:
        sel = fit & (subjects == c) & np.isfinite(ratio)
        if sel.sum() > 0:
            ident_med[int(c)] = float(np.median(ratio[sel]))
    cal_ratio = np.array([ratio[i] / ident_med.get(int(pred[i]), 1.0) for i in range(len(ratio))],
                         dtype=np.float32)

    # Within-task z-score of nearest distance.
    z_d1 = np.full(len(d1), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        sel = fit & (tasks == task) & np.isfinite(d1)
        if sel.sum() < 10:
            continue
        mu, sd = float(np.mean(d1[sel])), float(np.std(d1[sel]) + 1e-9)
        m = tasks == task
        z_d1[m] = (d1[m] - mu) / sd

    signals = {
        "base_ratio": ratio,
        "neg_d1": -d1,
        "neg_z_d1": -z_d1,
        "neg_hb": -hb,
        "cal_ratio": cal_ratio,
    }
    k_idx, u_idx = known_idx, unknown_idx
    rows = {}
    for name, sig in signals.items():
        k, u = sig[k_idx], sig[u_idx]
        if not (np.isfinite(k).all() and np.isfinite(u).all()):
            continue
        auc = float(roc_auc_score(np.r_[np.ones(len(k)), np.zeros(len(u))], np.r_[k, u]))
        # Fix the S2 operating point: known acceptance 0.85.
        th = float(np.quantile(k, 1 - CFG["targets"]["known_acceptance"]))
        ka = float(np.mean(k >= th)); ur = float(np.mean(u < th))
        rows[name] = {"AUC": auc, "known_acceptance": ka, "unknown_rejection": ur}

    # AND-gate: base ratio AND calibrated ratio must both pass.
    th_b = float(np.quantile(ratio[k_idx], 1 - CFG["targets"]["known_acceptance"]))
    th_c = float(np.quantile(cal_ratio[k_idx], 1 - CFG["targets"]["known_acceptance"]))
    ka = float(np.mean((ratio[k_idx] >= th_b) & (cal_ratio[k_idx] >= th_c)))
    ur = float(np.mean(~((ratio[u_idx] >= th_b) & (cal_ratio[u_idx] >= th_c))))
    rows["and_base_cal"] = {"AUC": None, "known_acceptance": ka, "unknown_rejection": ur}

    # OR-gate on the rejection side: reject if either signal is low.
    ka_or = float(np.mean((ratio[k_idx] >= th_b) | (cal_ratio[k_idx] >= th_c)))
    ur_or = float(np.mean(~((ratio[u_idx] >= th_b) | (cal_ratio[u_idx] >= th_c))))
    rows["or_base_cal"] = {"AUC": None, "known_acceptance": ka_or, "unknown_rejection": ur_or}

    return {"held_out": int(held), "N_known": int(len(k_idx)), "N_unknown": int(len(u_idx)),
            "signals": rows}


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E108 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != 11304 or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    model.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    model.eval()
    raw_emb = embed_raw(model, raw)
    hb = high_band_ratio(raw)

    folds = []
    for held in CLASSES:
        r = evaluate_fold(held, feat, raw_emb, hb, subjects, tasks, f)
        if r is not None:
            folds.append(r)
            print(f"E108 held={held} done", flush=True)

    names = sorted({n for r in folds for n in r["signals"]})
    summary = {}
    for n in names:
        vals = [r["signals"][n] for r in folds if n in r["signals"]]
        if not vals:
            continue
        aucs = [v["AUC"] for v in vals if v["AUC"] is not None]
        summary[n] = {
            "AUC_mean": float(np.mean(aucs)) if aucs else None,
            "known_acceptance_mean": float(np.mean([v["known_acceptance"] for v in vals])),
            "unknown_rejection_mean": float(np.mean([v["unknown_rejection"] for v in vals])),
            "folds_meeting_both": int(sum(1 for v in vals
                                          if v["known_acceptance"] >= CFG["targets"]["known_acceptance"]
                                          and v["unknown_rejection"] >= CFG["targets"]["unknown_rejection"])),
            "n_folds": len(vals),
        }
    result = {"experiment": CFG["experiment"], "test_accessed": False,
              "protocol": "LOSO; S2 operating point fixed at known acceptance 0.85",
              "targets": CFG["targets"], "summary": summary, "folds": folds}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E108 LOSO search for an S1-boosting signal", "",
             "固定 S2 工作点（known acceptance = 0.85），在留一身份协议下比较各辅助信号能否提高 unknown rejection。",
             "未访问项目 Test。", "",
             f"目标：known acceptance ≥ {CFG['targets']['known_acceptance']}，"
             f"unknown rejection ≥ {CFG['targets']['unknown_rejection']}。", "",
             "| 信号 | AUC 均值 | known acc 均值 | unknown rej 均值 | 同时达标的折数 |",
             "|---|---:|---:|---:|---:|"]
    for n, s in summary.items():
        a = f"{s['AUC_mean']:.4f}" if s["AUC_mean"] is not None else "—"
        lines.append(f"| {n} | {a} | {s['known_acceptance_mean']:.4f} | "
                     f"{s['unknown_rejection_mean']:.4f} | {s['folds_meeting_both']}/{s['n_folds']} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
