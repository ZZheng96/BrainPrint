"""E109: LOSO evaluation of the high-S2-acceptance candidates.

Motivation. E108 applied leave-one-identity-out (LOSO) only to the E102
distance-ratio family, which is the *low* S2 / high S1 candidate. The user's
question is the opposite one: for the candidates that already reach a high Test
S2 acceptance (E95, E97, E98, E104), is the LOSO held-out rejection also low?

If LOSO rejection tracks Test S1 rejection, LOSO becomes a usable Train-only
proxy for S1 and we can screen S1 improvements without touching Test.

Protocol (Train/session1 only, no project Test):
  For each held-out identity h:
    - enrol the other 19 identities
    - fit block   = enrolled identities, trial fraction <= 0.6
    - known pool  = enrolled identities, 0.6 < fraction <= 0.8
    - unknown pool= identity h,        0.6 < fraction <= 0.8
    - fix the threshold at the quantile that gives known acceptance 0.85
      (the same S2 operating point used in E108)
    - report AUC, known acceptance, held-out rejection

Candidates:
  E95_raw_cosine        frozen E95 EEGNet, max cosine over 20 prototypes
  E97_multiview         frozen E95 EEGNet, mean embedding over 3 windows
  E98_dual_gate         E95 raw cosine AND E65 correlation-logSPD cosine
  E104_session_invariant frozen E104 E88 encoder, task-conditioned distance ratio

Important caveat, stated up front: the E95/E97/E98 checkpoints were trained on
all 20 identities, so the held-out identity is *not* unseen by the encoder. This
is the same limitation E108 had for the E102 family (E102 prototypes were
rebuilt per fold, but the E65 features are hand-crafted and identity-agnostic,
so E108 was clean). For E95/E97/E98 the encoder has seen the held-out identity,
which can only make LOSO rejection look *better* than it truly is. We therefore
also report a stricter variant where the held-out identity's prototype column is
removed from the score matrix, which is the closest we can get to "unseen" with
a frozen 20-class checkpoint.
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
sys.path.insert(0, str(ROOT / "research" / "E88_spatial_channel_projection"))
sys.path.insert(0, str(ROOT / "research" / "E73_external_multisession"))
import run_e54 as E54
import run_e65
from run_e36 import load_config as load_e36_config
from e88_model import E88Spatial18Embedding

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
E95_CKPT = ROOT / "research" / "E95_e54_final20_candidate" / "model_checkpoint.pt"
E104_CKPT = ROOT / "research" / "E104_session_invariant_pretrain" / "encoder.pt"
E104_CFG = ROOT / "research" / "E104_session_invariant_pretrain" / "config.json"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


# ---------------------------------------------------------------- E95 / E97
def e95_scores(model, raw, batch=256):
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


def e97_scores(model, raw, windows, batch=128):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            xb = window_batch(raw[st:st + batch], windows)
            embs = [model.embed(xb[:, j]) for j in range(xb.shape[1])]
            emb = torch.nn.functional.normalize(torch.stack(embs, 1).mean(1), dim=1)
            out.append((emb @ model.normalized_prototypes().T).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------- E98
def spd_scores(proto, feat, batch=1024):
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


# ---------------------------------------------------------------- E104
def e104_encoder():
    cfg = json.loads(E104_CFG.read_text(encoding="utf-8"))
    cfg["data"]["channels_18"] = 18
    model = E88Spatial18Embedding(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E104_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    model.eval()
    return model, cfg


def e104_projection(cfg):
    p = cfg["channel_projection"]
    src = p["source_channels_22"]; tgt = p["target_channels_18"]
    sx = np.asarray([p["source_xy"][n] for n in src], dtype=np.float32)
    tx = np.asarray([p["target_xy"][n] for n in tgt], dtype=np.float32)
    d = np.sqrt(((tx[:, None, :] - sx[None, :, :]) ** 2).sum(-1))
    k = min(int(p["k"]), len(src))
    w = np.zeros_like(d)
    for j in range(len(tgt)):
        nn = np.argsort(d[j])[:k]
        if d[j, nn[0]] < 1e-8:
            w[j, nn[0]] = 1.0
        else:
            w[j, nn] = 1.0 / np.maximum(d[j, nn], 1e-8) ** float(p["power"])
            w[j] /= w[j].sum()
    return torch.as_tensor(w, dtype=torch.float32)


def e104_embed(model, raw, batch=256):
    out = []
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_task_prototypes(emb, subjects, tasks, fit_mask, classes):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(emb[sel], dtype=np.float32).mean(0))
            keys.append((task, int(c)))
    return np.stack(protos), keys


def distance_ratio(emb, protos, keys, tasks):
    ratio = np.full(len(emb), np.nan, dtype=np.float32)
    pred = np.zeros(len(emb), dtype=np.int64)
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
            q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
            pred[chunk] = cls[order[:, 0]]
    return ratio, pred


# ---------------------------------------------------------------- shared
def operating_point(known_sig, unknown_sig, target_accept):
    """Fix the threshold so known acceptance equals the S2 operating point."""
    th = float(np.quantile(known_sig, 1 - target_accept))
    ka = float(np.mean(known_sig >= th))
    ur = float(np.mean(unknown_sig < th))
    auc = float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                              np.r_[known_sig, unknown_sig]))
    return {"AUC": auc, "known_acceptance": ka, "unknown_rejection": ur, "threshold": th}


def masked_max_cosine(scores, held_col):
    """Max cosine with the held-out identity's prototype column removed."""
    s = scores.copy()
    s[:, held_col] = -np.inf
    return s.max(1)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E109 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)

    # E95 / E97 share the frozen E95 encoder.
    e95 = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    e95.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    e95.eval()
    s95 = e95_scores(e95, raw)
    s97 = e97_scores(e95, raw, CFG["candidates"]["E97_multiview"]["windows"])

    # E104 encoder. Its 22->18 IDW projection is defined on the external 22-channel
    # montage, so it cannot be applied to the project's native 18-channel Train data.
    # The project Train trials are already 18-channel, so they are fed directly.
    e104, e104cfg = e104_encoder()
    emb104 = e104_embed(e104, raw)

    folds = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fit = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        known_idx = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        unknown_idx = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(unknown_idx) == 0:
            continue
        held_col = int(np.flatnonzero(CLASSES == held)[0])
        target = CFG["targets"]["known_acceptance"]
        rows = {}

        # E95: max cosine, both with and without the held-out prototype column.
        rows["E95_raw_cosine"] = operating_point(s95[known_idx].max(1), s95[unknown_idx].max(1), target)
        rows["E95_raw_cosine_masked"] = operating_point(
            masked_max_cosine(s95[known_idx], held_col),
            masked_max_cosine(s95[unknown_idx], held_col), target)

        # E97: multiview mean embedding.
        rows["E97_multiview"] = operating_point(s97[known_idx].max(1), s97[unknown_idx].max(1), target)
        rows["E97_multiview_masked"] = operating_point(
            masked_max_cosine(s97[known_idx], held_col),
            masked_max_cosine(s97[unknown_idx], held_col), target)

        # E98: AND gate of E95 raw cosine and E65 SPD cosine.
        proto_classes, proto = build_spd_prototypes(feat, subjects, fit, enrolled)
        spd = spd_scores(proto, feat)
        kr, ur_ = s95[known_idx].max(1), s95[unknown_idx].max(1)
        ks, us = spd[known_idx].max(1), spd[unknown_idx].max(1)
        tr = float(np.quantile(kr, 1 - target)); ts = float(np.quantile(ks, 1 - target))
        ka = float(np.mean((kr >= tr) & (ks >= ts)))
        ur = float(np.mean(~((ur_ >= tr) & (us >= ts))))
        rows["E98_dual_gate"] = {"AUC": None, "known_acceptance": ka,
                                 "unknown_rejection": ur, "threshold": [tr, ts]}

        # E104: task-conditioned distance ratio on the frozen E88 encoder.
        protos, keys = build_task_prototypes(emb104, subjects, tasks, fit, enrolled)
        ratio, _ = distance_ratio(emb104, protos, keys, tasks)
        rows["E104_session_invariant"] = operating_point(ratio[known_idx], ratio[unknown_idx], target)

        folds.append({"held_out": int(held), "N_known": int(len(known_idx)),
                      "N_unknown": int(len(unknown_idx)), "signals": rows})
        print(f"E109 held={held} done", flush=True)

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

    # Cross-reference table: LOSO rejection vs the already-audited Test S1 rejection.
    lines = ["# E109 LOSO on the high-S2-acceptance candidates", "",
             "对 Test S2 接受率高的方案补做 LOSO（19 注册 / 1 留出），"
             "阈值固定使注册者接受率 = 0.85。未访问项目 Test。", "",
             "## LOSO 结果", "",
             "| 信号 | LOSO AUC | known acc | 留出者拒识均值 | 中位数 | ≥0.80 折数 | ≥0.50 折数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for n, s in summary.items():
        a = f"{s['AUC_mean']:.4f}" if s["AUC_mean"] is not None else "—"
        lines.append(f"| {n} | {a} | {s['known_acceptance_mean']:.4f} | "
                     f"{s['unknown_rejection_mean']:.4f} | {s['unknown_rejection_median']:.4f} | "
                     f"{s['folds_rejection_ge_0_80']}/{s['n_folds']} | "
                     f"{s['folds_rejection_ge_0_50']}/{s['n_folds']} |")
    lines += ["", "## LOSO 与 Test S1 对照", "",
              "| 方案 | Test S2 接受率 | Test S1 拒识率 | LOSO 留出者拒识均值 |",
              "|---|---:|---:|---:|"]
    for name, spec in CFG["candidates"].items():
        s = summary.get(name)
        loso = f"{s['unknown_rejection_mean']:.4f}" if s else "—"
        lines.append(f"| {name} | {spec['test_s2_acceptance']:.4f} | "
                     f"{spec['test_s1_rejection']:.4f} | {loso} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
