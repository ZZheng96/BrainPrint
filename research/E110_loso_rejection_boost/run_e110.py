"""E110: raise LOSO held-out rejection on top of the E109 baselines.

E109 established that LOSO held-out rejection tracks Test S1 rejection for the
models trained on project identities (E95 0.157 vs 0.154, E97 0.159 vs 0.101,
E98 0.484 vs 0.452). So LOSO is a usable Train-only proxy for S1, and we can
search for improvements without touching Test.

This experiment keeps the S2 operating point fixed (known acceptance 0.85) and
asks which representation / gate raises held-out rejection the most.

Baselines (from E109, recomputed here for an apples-to-apples comparison):
  B1  E95 raw cosine            max cosine over the frozen E95 prototypes
  B2  E97 multiview             mean embedding over three 600-sample windows
  B3  E98 dual gate             E95 raw cosine AND E65 correlation-logSPD cosine
  B4  E102 SPD ratio            task-conditioned distance ratio on E65 logSPD

New signals (all Train-only, no Test):
  N1  SPD ratio + raw cosine    mean of the two rank-normalized scores
  N2  SPD ratio + raw ratio     mean of the two rank-normalized scores
  N3  multi-band SPD ratio      distance ratio on per-band logSPD blocks
  N4  task-agnostic SPD ratio   distance ratio over all tasks pooled
  N5  prototype-margin ratio    (d2-d1)/d1 on SPD, a scale-free variant
  N6  N1 AND N2                 both fused scores must pass
  N7  N1 OR  N2                 either fused score passes

Why these: E108 showed every single auxiliary signal is weaker than the SPD
ratio alone, but it never tried fusing the SPD ratio with the *raw* identity
branch, which is the branch that carries the high-S2 behaviour. N1/N2 test
whether the two branches are complementary in the LOSO setting.

Protocol: Train/session1 only. The project Test is never read.
"""
from __future__ import annotations
import json
import sys
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
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
E95_CKPT = ROOT / "research" / "E95_e54_final20_candidate" / "model_checkpoint.pt"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


# ------------------------------------------------------------------ scoring
def raw_scores(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.scores(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def window_batch(x, windows):
    chunks = []
    for a, b in windows:
        crop = x[..., a:b]
        chunks.append(torch.nn.functional.interpolate(crop, size=1200, mode="linear",
                                                      align_corners=False))
    return torch.stack(chunks, dim=1)


def multiview_scores(model, raw, windows, batch=128):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            xb = window_batch(raw[st:st + batch], windows)
            embs = [model.embed(xb[:, j]) for j in range(xb.shape[1])]
            emb = torch.nn.functional.normalize(torch.stack(embs, 1).mean(1), dim=1)
            out.append((emb @ model.normalized_prototypes().T).cpu().numpy())
    return np.concatenate(out, axis=0)


def spd_cosine(proto, feat, batch=1024):
    p = torch.as_tensor(proto, dtype=torch.float32, device=DEVICE)
    out = []
    for st in range(0, len(feat), batch):
        chunk = feat[st:st + batch]
        if isinstance(chunk, torch.Tensor):
            q = chunk.detach().to(device=DEVICE, dtype=torch.float32)
        else:
            q = torch.as_tensor(np.array(chunk, dtype=np.float32), device=DEVICE)
        out.append((-torch.cdist(q, p)).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_spd_prototypes(feat, subjects, fit_mask, classes):
    rows = np.flatnonzero(fit_mask)
    present = np.unique(subjects[rows])
    keep = [c for c in classes if c in present]
    proto = np.stack([np.asarray(feat[rows[subjects[rows] == c]], dtype=np.float32).mean(0)
                      for c in keep])
    return np.asarray(keep), proto


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


def distance_ratio(feat, protos, keys, tasks, task_agnostic=False):
    """d2/d1 within the sample's own task (or pooled over all tasks)."""
    ratio = np.full(len(feat), np.nan, dtype=np.float32)
    margin = np.full(len(feat), np.nan, dtype=np.float32)
    pred = np.zeros(len(feat), dtype=np.int64)
    groups = [None] if task_agnostic else list(CFG["tasks"])
    for task in groups:
        idx = np.arange(len(feat)) if task is None else np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        cols = list(range(len(keys))) if task is None else \
            [i for i, (t, _) in enumerate(keys) if t == task]
        if len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        cls = np.asarray([keys[i][1] for i in cols])
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
            margin[chunk] = (d2 - d1) / np.maximum(d1, 1e-9)
            pred[chunk] = cls[order[:, 0]]
    return ratio, margin, pred


def rank_normalize(values, reference):
    """Map values to [0,1] using the reference distribution's ranks."""
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    pos = np.searchsorted(ref, values, side="right") / len(ref)
    return pos.astype(np.float32)


# ------------------------------------------------------------------ metrics
def operating_point(known_sig, unknown_sig, target_accept):
    th = float(np.quantile(known_sig, 1 - target_accept))
    ka = float(np.mean(known_sig >= th))
    ur = float(np.mean(unknown_sig < th))
    auc = float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                              np.r_[known_sig, unknown_sig]))
    return {"AUC": auc, "known_acceptance": ka, "unknown_rejection": ur, "threshold": th}


def gate_point(known_a, known_b, unknown_a, unknown_b, target_accept, mode):
    ta = float(np.quantile(known_a, 1 - target_accept))
    tb = float(np.quantile(known_b, 1 - target_accept))
    if mode == "and":
        ka = float(np.mean((known_a >= ta) & (known_b >= tb)))
        ur = float(np.mean(~((unknown_a >= ta) & (unknown_b >= tb))))
    else:
        ka = float(np.mean((known_a >= ta) | (known_b >= tb)))
        ur = float(np.mean(~((unknown_a >= ta) | (unknown_b >= tb))))
    return {"AUC": None, "known_acceptance": ka, "unknown_rejection": ur,
            "threshold": [ta, tb]}


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E110 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    e95 = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    e95.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    e95.eval()
    s95 = raw_scores(e95, raw)
    s97 = multiview_scores(e95, raw, CFG["windows"])

    # Per-band logSPD blocks for the multi-band ratio (N3).
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    band_blocks = []
    with torch.no_grad():
        for st in range(0, len(raw), 256):
            band_blocks.append(run_e65.gpu_correlation_logspd(raw[st:st + 256], e36, e65cfg).cpu().numpy())
    feat_all = np.concatenate(band_blocks, axis=0)
    if not np.allclose(feat_all, np.asarray(feat), atol=1e-4):
        raise ValueError("Recomputed logSPD does not match the cached Train features")
    n_bands = len(e36["features"]["bands_hz"])
    per_band = 855 // n_bands
    if per_band * n_bands != 855:
        raise ValueError("Unexpected logSPD width")

    target = CFG["targets"]["known_acceptance"]
    folds = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fit = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        known_idx = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        unknown_idx = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(unknown_idx) == 0:
            continue
        rows = {}

        # ---- baselines
        rows["B1_E95_raw_cosine"] = operating_point(s95[known_idx].max(1), s95[unknown_idx].max(1), target)
        rows["B2_E97_multiview"] = operating_point(s97[known_idx].max(1), s97[unknown_idx].max(1), target)

        proto_classes, proto = build_spd_prototypes(feat, subjects, fit, enrolled)
        spd = spd_cosine(proto, feat)
        kr, ur_ = s95[known_idx].max(1), s95[unknown_idx].max(1)
        ks, us = spd[known_idx].max(1), spd[unknown_idx].max(1)
        rows["B3_E98_dual_gate"] = gate_point(kr, ks, ur_, us, target, "and")

        protos, keys = build_task_prototypes(feat, subjects, tasks, fit, enrolled)
        ratio, margin, _ = distance_ratio(feat, protos, keys, tasks)
        rows["B4_E102_spd_ratio"] = operating_point(ratio[known_idx], ratio[unknown_idx], target)

        # ---- N1/N2: fuse the SPD ratio with the raw identity branch
        # Rank-normalize on the pooled known+unknown pool of this fold so the two
        # branches become comparable without using any Test information.
        pool = np.r_[known_idx, unknown_idx]
        r_ratio = rank_normalize(ratio[pool], ratio[pool])
        r_cos = rank_normalize(s95[pool].max(1), s95[pool].max(1))
        r_rawratio = rank_normalize(
            (np.sort(s95[pool], axis=1)[:, -1] - np.sort(s95[pool], axis=1)[:, -2])
            / np.maximum(np.sort(s95[pool], axis=1)[:, -1], 1e-9),
            (np.sort(s95[pool], axis=1)[:, -1] - np.sort(s95[pool], axis=1)[:, -2])
            / np.maximum(np.sort(s95[pool], axis=1)[:, -1], 1e-9))
        nk = len(known_idx)
        fused_cos = 0.5 * (r_ratio + r_cos)
        fused_raw = 0.5 * (r_ratio + r_rawratio)
        rows["N1_spd_ratio_plus_raw_cosine"] = operating_point(fused_cos[:nk], fused_cos[nk:], target)
        rows["N2_spd_ratio_plus_raw_ratio"] = operating_point(fused_raw[:nk], fused_raw[nk:], target)
        rows["N6_N1_and_N2"] = gate_point(fused_cos[:nk], fused_raw[:nk],
                                          fused_cos[nk:], fused_raw[nk:], target, "and")
        rows["N7_N1_or_N2"] = gate_point(fused_cos[:nk], fused_raw[:nk],
                                         fused_cos[nk:], fused_raw[nk:], target, "or")

        # ---- N3: multi-band SPD ratio, averaged over the five bands
        band_ratios = []
        for b in range(n_bands):
            block = np.asarray(feat[:, b * per_band:(b + 1) * per_band], dtype=np.float32)
            bp, bk = build_task_prototypes(block, subjects, tasks, fit, enrolled)
            br, _, _ = distance_ratio(block, bp, bk, tasks)
            band_ratios.append(br)
        band_ratio = np.nanmean(np.stack(band_ratios), axis=0)
        rows["N3_multiband_spd_ratio"] = operating_point(band_ratio[known_idx], band_ratio[unknown_idx], target)

        # ---- N4: task-agnostic SPD ratio (all tasks pooled into one prototype set)
        ap, ak = build_task_prototypes(feat, subjects, tasks, fit, enrolled)
        agnostic, _, _ = distance_ratio(feat, ap, ak, tasks, task_agnostic=True)
        rows["N4_task_agnostic_spd_ratio"] = operating_point(agnostic[known_idx], agnostic[unknown_idx], target)

        # ---- N5: scale-free prototype margin (d2-d1)/d1
        rows["N5_spd_margin_ratio"] = operating_point(margin[known_idx], margin[unknown_idx], target)

        folds.append({"held_out": int(held), "N_known": int(len(known_idx)),
                      "N_unknown": int(len(unknown_idx)), "signals": rows})
        print(f"E110 held={held} done", flush=True)

    names = sorted({n for r in folds for n in r["signals"]})
    summary = {}
    for n in names:
        vals = [r["signals"][n] for r in folds if n in r["signals"]]
        aucs = [v["AUC"] for v in vals if v["AUC"] is not None]
        summary[n] = {
            "AUC_mean": float(np.mean(aucs)) if aucs else None,
            "known_acceptance_mean": float(np.mean([v["known_acceptance"] for v in vals])),
            "unknown_rejection_mean": float(np.mean([v["unknown_rejection"] for v in vals])),
            "unknown_rejection_median": float(np.median([v["unknown_rejection"] for v in vals])),
            "folds_rejection_ge_0_80": int(sum(1 for v in vals if v["unknown_rejection"] >= 0.80)),
            "folds_rejection_ge_0_50": int(sum(1 for v in vals if v["unknown_rejection"] >= 0.50)),
            "n_folds": len(vals),
        }

    result = {"experiment": CFG["experiment"], "test_accessed": False,
              "protocol": "LOSO; 19 enrolled, 1 held out; threshold fixed at known acceptance 0.85",
              "targets": CFG["targets"], "summary": summary, "folds": folds}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    base = summary["B4_E102_spd_ratio"]["unknown_rejection_mean"]
    lines = ["# E110 Raising LOSO held-out rejection", "",
             "在 E109 确认 LOSO 拒识可作为 Test S1 代理之后，固定 S2 工作点"
             "（known acceptance = 0.85），搜索能提高留出者拒识率的表征与门控。"
             "未访问项目 Test。", "",
             "## 结果", "",
             "| 信号 | LOSO AUC | known acc | 留出者拒识均值 | 中位数 | ≥0.80 折数 | ≥0.50 折数 | 相对 B4 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, s in summary.items():
        a = f"{s['AUC_mean']:.4f}" if s["AUC_mean"] is not None else "—"
        delta = s["unknown_rejection_mean"] - base
        lines.append(f"| {n} | {a} | {s['known_acceptance_mean']:.4f} | "
                     f"{s['unknown_rejection_mean']:.4f} | {s['unknown_rejection_median']:.4f} | "
                     f"{s['folds_rejection_ge_0_80']}/{s['n_folds']} | "
                     f"{s['folds_rejection_ge_0_50']}/{s['n_folds']} | {delta:+.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
