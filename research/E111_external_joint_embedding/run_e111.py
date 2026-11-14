"""E111: external identity-discriminative + session-invariant embedding, LOSO evaluated.

Motivation. E108 and E110 together tested 12 auxiliary signals on top of the
task-conditioned correlation-logSPD distance ratio and none beat it (LOSO
held-out rejection 0.831). The conclusion was that the *representation* has to
change, not the gate.

E104 already tried external cross-session SupCon, but it used the E88 spatial
encoder and only a cross-session contrastive loss, which does not force
different identities apart. Its LOSO rejection was 0.181.

This experiment changes two things:
  1. Use the E54 EEGNet encoder (the same backbone that reaches high S2
     acceptance) instead of the E88 spatial encoder.
  2. Train with a *joint* objective on external BCIC IV-2a:
        L = SupCon(cross-session positives) + lambda * SupCon(identity positives)
     The first term removes session, the second keeps identity discriminative.
     A pure cross-session loss can collapse identities together, which is
     exactly what would destroy open-set rejection.

Evaluation is the E109/E110 LOSO protocol on project Train/session1:
  19 identities enrolled, 1 held out as unknown, threshold fixed at known
  acceptance 0.85, report held-out rejection. The project Test is never read.

The external encoder is frozen before any project Train data is touched, so the
LOSO number is a clean Train-only estimate.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(ROOT / "research" / "E73_external_multisession"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e54 as E54
import run_e65
from run_e36 import load_config as load_e36_config
from external_data import load_external_data

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
E73_DIR = ROOT / "research" / "E73_external_multisession"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


# ------------------------------------------------------------ external data
def load_external():
    """BCIC IV-2a, 22 channels, both sessions, all runs."""
    base = json.loads((E73_DIR / "config.json").read_text(encoding="utf-8"))
    base["data"]["feature_key"] = "data"
    base["data"]["channels"] = 22
    return load_external_data(base, DEVICE)


def project22to18(raw, cfg):
    """IDW projection from the external 22-channel montage to the project 18."""
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
    return torch.as_tensor(w, dtype=torch.float32, device=raw.device)


# ------------------------------------------------------------ losses
def supcon(z, labels, temperature, mask=None):
    """Supervised contrastive loss over the given positive mask."""
    z = F.normalize(z.float(), dim=1)
    logits = z @ z.T / float(temperature)
    if mask is None:
        mask = labels[:, None] == labels[None, :]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    mask = mask & ~eye
    if not bool(mask.any()):
        return torch.zeros((), device=z.device)
    logits = logits.masked_fill(eye, -torch.inf)
    lp = logits - torch.logsumexp(logits, 1, keepdim=True)
    denom = mask.sum(1).clamp_min(1)
    return -(torch.where(mask, lp, torch.zeros_like(lp)).sum(1) / denom).mean()


# ------------------------------------------------------------ pretraining
def pretrain():
    ckpt = OUT / "encoder.pt"
    if ckpt.is_file():
        e54cfg = E54.load_config()
        model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
        saved = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        return model, {"reused": True, "history": saved.get("history", [])}

    data = load_external()
    proj = project22to18(data.raw, CFG)
    e54cfg = E54.load_config()
    model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    # The 20-class prototypes are unused here; only the encoder is trained.
    for p in (model.prototypes, model.radius_raw):
        p.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG["training"]["learning_rate"],
                            weight_decay=CFG["training"]["weight_decay"])
    rng = np.random.default_rng(CFG["seed"])
    subjects = data.subject
    sessions = data.session

    groups = []
    for s in np.unique(subjects):
        for sess in (0, 1):
            idx = np.flatnonzero((subjects == s) & (sessions == sess))
            if len(idx):
                groups.append(idx)
    k = CFG["training"]["trials_per_group"]
    history = []
    for epoch in range(1, CFG["training"]["epochs"] + 1):
        perms = [rng.permutation(g) for g in groups]
        steps = min(len(p) for p in perms) // k
        total = total_cross = total_id = 0.0
        model.train()
        for step in range(steps):
            ix = np.concatenate([p[step * k:(step + 1) * k] for p in perms])
            sel = torch.as_tensor(ix, device=DEVICE)
            sub = torch.as_tensor(subjects[ix], device=DEVICE)
            ses = torch.as_tensor(sessions[ix], device=DEVICE)
            with torch.no_grad():
                x18 = torch.einsum("ij,bjt->bit", proj, data.raw[sel])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model.embed(x18)
                cross_mask = (sub[:, None] == sub[None, :]) & (ses[:, None] != ses[None, :])
                id_mask = sub[:, None] == sub[None, :]
                l_cross = supcon(z, sub, CFG["training"]["temperature"], cross_mask)
                l_id = supcon(z, sub, CFG["training"]["temperature"], id_mask)
                loss = l_cross + float(CFG["training"]["identity_weight"]) * l_id
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           CFG["training"]["gradient_clip_norm"])
            opt.step()
            total += float(loss) * len(ix)
            total_cross += float(l_cross) * len(ix)
            total_id += float(l_id) * len(ix)
        n = steps * len(groups) * k
        row = {"epoch": epoch, "loss": total / n, "cross": total_cross / n, "identity": total_id / n}
        history.append(row)
        print(f"E111 pretrain epoch={epoch} loss={row['loss']:.4f} "
              f"cross={row['cross']:.4f} id={row['identity']:.4f}", flush=True)

    model.eval()
    torch.save({"state_dict": {k_: v.detach().cpu() for k_, v in model.state_dict().items()},
                "history": history, "config": CFG}, ckpt)
    return model, {"reused": False, "history": history}


# ------------------------------------------------------------ scoring
def embed(model, raw, batch=256):
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


def operating_point(known_sig, unknown_sig, target_accept):
    th = float(np.quantile(known_sig, 1 - target_accept))
    ka = float(np.mean(known_sig >= th))
    ur = float(np.mean(unknown_sig < th))
    auc = float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                              np.r_[known_sig, unknown_sig]))
    return {"AUC": auc, "known_acceptance": ka, "unknown_rejection": ur, "threshold": th}


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E111 requires CUDA")
    model, train_info = pretrain()

    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    emb = embed(model, raw)

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

        # E111: task-conditioned distance ratio on the external-pretrained embedding.
        protos, keys = build_task_prototypes(emb, subjects, tasks, fit, enrolled)
        ratio, _ = distance_ratio(emb, protos, keys, tasks)
        rows["E111_external_joint_ratio"] = operating_point(ratio[known_idx], ratio[unknown_idx], target)

        # Reference: the same ratio on the hand-crafted E65 logSPD (E102 baseline).
        sp_protos, sp_keys = build_task_prototypes(feat, subjects, tasks, fit, enrolled)
        sp_ratio, _ = distance_ratio(feat, sp_protos, sp_keys, tasks)
        rows["B4_E102_spd_ratio"] = operating_point(sp_ratio[known_idx], sp_ratio[unknown_idx], target)

        folds.append({"held_out": int(held), "N_known": int(len(known_idx)),
                      "N_unknown": int(len(unknown_idx)), "signals": rows})
        print(f"E111 held={held} done", flush=True)

    names = sorted({n for r in folds for n in r["signals"]})
    summary = {}
    for n in names:
        vals = [r["signals"][n] for r in folds if n in r["signals"]]
        summary[n] = {
            "AUC_mean": float(np.mean([v["AUC"] for v in vals])),
            "known_acceptance_mean": float(np.mean([v["known_acceptance"] for v in vals])),
            "unknown_rejection_mean": float(np.mean([v["unknown_rejection"] for v in vals])),
            "unknown_rejection_median": float(np.median([v["unknown_rejection"] for v in vals])),
            "folds_rejection_ge_0_80": int(sum(1 for v in vals if v["unknown_rejection"] >= 0.80)),
            "folds_rejection_ge_0_50": int(sum(1 for v in vals if v["unknown_rejection"] >= 0.50)),
            "n_folds": len(vals),
        }

    result = {"experiment": CFG["experiment"], "test_accessed": False,
              "protocol": "LOSO; 19 enrolled, 1 held out; threshold fixed at known acceptance 0.85",
              "pretraining": {"reused": train_info["reused"],
                              "final": train_info["history"][-1] if train_info["history"] else None},
              "targets": CFG["targets"], "summary": summary, "folds": folds}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    base = summary["B4_E102_spd_ratio"]["unknown_rejection_mean"]
    lines = ["# E111 External identity+session-invariant embedding (LOSO)", "",
             "外部 BCIC IV-2a 双 Session 上训练 `SupCon(跨 Session) + λ·SupCon(同身份)`，"
             "用 E54 EEGNet 骨干，冻结后在项目 Train/session1 上做 LOSO。"
             "未访问项目 Test。", "",
             "## 结果", "",
             "| 信号 | LOSO AUC | known acc | 留出者拒识均值 | 中位数 | ≥0.80 折数 | ≥0.50 折数 | 相对 B4 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, s in summary.items():
        delta = s["unknown_rejection_mean"] - base
        lines.append(f"| {n} | {s['AUC_mean']:.4f} | {s['known_acceptance_mean']:.4f} | "
                     f"{s['unknown_rejection_mean']:.4f} | {s['unknown_rejection_median']:.4f} | "
                     f"{s['folds_rejection_ge_0_80']}/{s['n_folds']} | "
                     f"{s['folds_rejection_ge_0_50']}/{s['n_folds']} | {delta:+.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
