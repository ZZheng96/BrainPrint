"""E112: explicit open-set classifier trained under the LOSO protocol.

Motivation. E108 + E110 showed that no post-hoc gate on the frozen E102
distance ratio beats LOSO held-out rejection 0.831. E111 tests a new
representation. This experiment tests the other direction: instead of a
post-hoc threshold, train a classifier that is *explicitly* optimised to reject
unseen identities.

Design (Train/session1 only, no project Test):
  For each held-out identity h (20 folds):
    - enrolled = the other 19 identities
    - fit block   = enrolled, trial fraction <= 0.6
    - known pool  = enrolled, 0.6 < fraction <= 0.8
    - unknown pool= identity h, 0.6 < fraction <= 0.8
    - train a small MLP on the E65 correlation-logSPD features with a
      *leave-one-enrolled-identity-out* open-set objective:
          L = CE(known) + lambda * BCE(reject | pseudo-unknown)
      where the pseudo-unknowns are the enrolled identities that are held out
      from the classifier's own training split (an inner LOSO), so the model
      never sees identity h during training.
    - score = P(known) from the reject head
    - fix the threshold at known acceptance 0.85, report held-out rejection

This is the "explicit open-set classifier" direction. It differs from E102 in
that the reject boundary is learned, not derived from a distance ratio.

Reference baselines recomputed in the same folds:
  B4  E102 task-conditioned SPD distance ratio
  B1  E95 raw cosine (frozen, high-S2 branch)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
import run_e54 as E54

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
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


def distance_ratio(feat, protos, keys, tasks):
    ratio = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
    return ratio


def raw_scores(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.scores(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


class OpenSetHead(nn.Module):
    """MLP with an identity head and a known/unknown head."""

    def __init__(self, in_dim, n_classes, hidden):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ELU(),
        )
        self.identity = nn.Linear(hidden, n_classes)
        self.known = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.identity(h), self.known(h).squeeze(1)


def train_open_set(x_fit, y_fit, x_pseudo, epochs, seed):
    """Train on enrolled identities; pseudo-unknowns are inner-held-out identities."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = int(y_fit.max()) + 1
    model = OpenSetHead(x_fit.shape[1], n_classes, CFG["model"]["hidden"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["training"]["learning_rate"],
                            weight_decay=CFG["training"]["weight_decay"])
    xf = torch.as_tensor(x_fit, dtype=torch.float32, device=DEVICE)
    yf = torch.as_tensor(y_fit, dtype=torch.long, device=DEVICE)
    xp = torch.as_tensor(x_pseudo, dtype=torch.float32, device=DEVICE)
    batch = CFG["training"]["batch_size"]
    lam = CFG["training"]["unknown_weight"]
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(xf), device=DEVICE)
        for st in range(0, len(xf), batch):
            sel = order[st:st + batch]
            if len(sel) < 2:
                continue
            logits, known_logit = model(xf[sel])
            loss = F.cross_entropy(logits, yf[sel])
            loss = loss + F.binary_cross_entropy_with_logits(
                known_logit, torch.ones(len(sel), device=DEVICE))
            # Pseudo-unknowns: reject head should output 0.
            psel = torch.randint(0, len(xp), (batch,), device=DEVICE)
            _, p_logit = model(xp[psel])
            loss = loss + lam * F.binary_cross_entropy_with_logits(
                p_logit, torch.zeros(len(psel), device=DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def known_probability(model, x):
    out = []
    with torch.no_grad():
        for st in range(0, len(x), 2048):
            xb = torch.as_tensor(x[st:st + 2048], dtype=torch.float32, device=DEVICE)
            _, logit = model(xb)
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def operating_point(known_sig, unknown_sig, target_accept):
    th = float(np.quantile(known_sig, 1 - target_accept))
    ka = float(np.mean(known_sig >= th))
    ur = float(np.mean(unknown_sig < th))
    auc = float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                              np.r_[known_sig, unknown_sig]))
    return {"AUC": auc, "known_acceptance": ka, "unknown_rejection": ur, "threshold": th}


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E112 requires CUDA")
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

        # ---- B4: E102 task-conditioned SPD distance ratio
        protos, keys = build_task_prototypes(feat, subjects, tasks, fit, enrolled)
        ratio = distance_ratio(feat, protos, keys, tasks)
        rows["B4_E102_spd_ratio"] = operating_point(ratio[known_idx], ratio[unknown_idx], target)

        # ---- B1: E95 raw cosine
        rows["B1_E95_raw_cosine"] = operating_point(s95[known_idx].max(1),
                                                    s95[unknown_idx].max(1), target)

        # ---- E112: explicit open-set head
        # Inner LOSO: hold out 4 enrolled identities as pseudo-unknowns for the
        # reject head, train on the remaining 15.
        rng = np.random.default_rng(CFG["seed"] + int(held))
        inner_unknown = np.sort(rng.choice(enrolled, size=CFG["training"]["inner_unknown"], replace=False))
        inner_known = np.setdiff1d(enrolled, inner_unknown)
        fit_inner = np.isin(subjects, inner_known) & (f <= CFG["calibration"]["fit_fraction"])
        pseudo_idx = np.flatnonzero(np.isin(subjects, inner_unknown) & (f <= CFG["calibration"]["fit_fraction"]))
        fit_rows = np.flatnonzero(fit_inner)
        if len(fit_rows) == 0 or len(pseudo_idx) == 0:
            continue
        class_to_idx = {int(c): i for i, c in enumerate(inner_known)}
        y_fit = np.asarray([class_to_idx[int(s)] for s in subjects[fit_rows]], dtype=np.int64)
        model = train_open_set(np.asarray(feat[fit_rows], dtype=np.float32), y_fit,
                               np.asarray(feat[pseudo_idx], dtype=np.float32),
                               CFG["training"]["epochs"], CFG["seed"] + int(held))
        prob = known_probability(model, np.asarray(feat, dtype=np.float32))
        rows["E112_open_set_head"] = operating_point(prob[known_idx], prob[unknown_idx], target)

        folds.append({"held_out": int(held), "N_known": int(len(known_idx)),
                      "N_unknown": int(len(unknown_idx)),
                      "inner_unknown": inner_unknown.tolist(), "signals": rows})
        print(f"E112 held={held} done", flush=True)

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
              "targets": CFG["targets"], "summary": summary, "folds": folds}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    base = summary["B4_E102_spd_ratio"]["unknown_rejection_mean"]
    lines = ["# E112 Explicit open-set classifier under LOSO", "",
             "在 E65 correlation-logSPD 特征上训练带拒识头的 MLP，"
             "伪未知来自内层留身份（4 个注册身份），外层留出身份从不参与训练。"
             "阈值固定使注册者接受率 = 0.85。未访问项目 Test。", "",
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
