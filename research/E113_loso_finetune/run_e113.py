"""E113: LOSO fine-tuning of the E95 encoder with the held-out identity excluded.

Motivation. E111 (external pretraining) and E112 (explicit open-set head) both
failed badly: 0.192 and 0.003 held-out rejection versus the E102 baseline 0.831.
Both changed the *representation* or the *decision rule* while throwing away the
one thing that works: the task-conditioned correlation-logSPD geometry.

E109 showed the E95 encoder has essentially no open-set ability (LOSO AUC 0.504)
because it was trained on all 20 identities with a closed-set objective. This
experiment asks a narrower question:

    If the E95 encoder is fine-tuned with the held-out identity genuinely
    excluded, does its embedding become open-set capable?

Design (Train/session1 only, no project Test):
  For each held-out identity h (20 folds):
    - enrolled = the other 19 identities
    - start from the frozen E95 checkpoint
    - fine-tune the encoder + rebuild prototypes on the enrolled identities only,
      using the E54 embedding loss (CE + ARPL) on the fit block (fraction <= 0.6)
    - known pool  = enrolled, 0.6 < fraction <= 0.8
    - unknown pool= identity h, 0.6 < fraction <= 0.8
    - score = max cosine to the rebuilt prototypes
    - threshold fixed at known acceptance 0.85, report held-out rejection

This is the honest version of "train an open-set model": identity h never
appears in the fine-tuning data, so the held-out rejection is a real estimate.

Reference baselines recomputed in the same folds:
  B4  E102 task-conditioned SPD distance ratio (0.831)
  B1  frozen E95 raw cosine (0.157)
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


def embed(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def finetune_fold(raw, subjects, fit_rows, enrolled, seed):
    """Fine-tune the E95 encoder on the enrolled identities only."""
    e54cfg = E54.load_config()
    model = E54.EEGNetEmbedding(len(enrolled), e54cfg).to(DEVICE)
    saved = torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"]
    # The E95 checkpoint has 20 prototypes; keep only the enrolled columns.
    keep = [int(np.flatnonzero(CLASSES == c)[0]) for c in enrolled]
    saved = dict(saved)
    saved["prototypes"] = saved["prototypes"][keep]
    saved["radius_raw"] = saved["radius_raw"][keep]
    model.load_state_dict(saved)

    torch.manual_seed(seed)
    np.random.seed(seed)
    class_to_idx = {int(c): i for i, c in enumerate(enrolled)}
    fit_index = torch.as_tensor(fit_rows, device=DEVICE)
    fit_raw = raw[fit_index]
    fit_labels = torch.as_tensor([class_to_idx[int(s)] for s in subjects[fit_rows]],
                                 dtype=torch.long, device=DEVICE)
    E54.initialize_prototypes(model, fit_raw, fit_labels)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["training"]["learning_rate"],
                            weight_decay=CFG["training"]["weight_decay"])
    batch = CFG["training"]["batch_size"]
    temperature = float(e54cfg["model"]["cosine_temperature"])
    arpl_weight = float(e54cfg["model"]["arpl_weight"])
    margin = float(e54cfg["model"]["arpl_margin"])
    rng = np.random.default_rng(seed)
    for epoch in range(1, CFG["training"]["epochs"] + 1):
        model.train()
        order = rng.permutation(len(fit_rows))
        for st in range(0, len(order), batch):
            sel = torch.as_tensor(order[st:st + batch], device=DEVICE)
            emb = model.embed(fit_raw[sel])
            loss = E54.embedding_loss(model, emb, fit_labels[sel], temperature,
                                      arpl_weight, margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           CFG["training"]["gradient_clip_norm"])
            opt.step()
    model.eval()
    return model


def operating_point(known_sig, unknown_sig, target_accept):
    th = float(np.quantile(known_sig, 1 - target_accept))
    ka = float(np.mean(known_sig >= th))
    ur = float(np.mean(unknown_sig < th))
    auc = float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                              np.r_[known_sig, unknown_sig]))
    return {"AUC": auc, "known_acceptance": ka, "unknown_rejection": ur, "threshold": th}


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E113 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)

    # Frozen E95 baseline scores.
    frozen = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    frozen.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    frozen.eval()
    with torch.no_grad():
        s_frozen = []
        for st in range(0, len(raw), 256):
            s_frozen.append(frozen.scores(raw[st:st + 256]).cpu().numpy())
    s_frozen = np.concatenate(s_frozen, axis=0)

    target = CFG["targets"]["known_acceptance"]
    folds = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fit = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        fit_rows = np.flatnonzero(fit)
        known_idx = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        unknown_idx = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(unknown_idx) == 0:
            continue
        rows = {}

        # B4: E102 task-conditioned SPD distance ratio
        protos, keys = build_task_prototypes(feat, subjects, tasks, fit, enrolled)
        ratio = distance_ratio(feat, protos, keys, tasks)
        rows["B4_E102_spd_ratio"] = operating_point(ratio[known_idx], ratio[unknown_idx], target)

        # B1: frozen E95 raw cosine
        rows["B1_E95_raw_cosine"] = operating_point(s_frozen[known_idx].max(1),
                                                    s_frozen[unknown_idx].max(1), target)

        # E113: fine-tuned encoder, prototypes rebuilt on enrolled only
        model = finetune_fold(raw, subjects, fit_rows, enrolled, CFG["seed"] + int(held))
        emb = embed(model, raw)
        ft_protos, ft_keys = build_task_prototypes(emb, subjects, tasks, fit, enrolled)
        ft_ratio = distance_ratio(emb, ft_protos, ft_keys, tasks)
        rows["E113_finetuned_ratio"] = operating_point(ft_ratio[known_idx], ft_ratio[unknown_idx], target)

        # E113b: fine-tuned encoder, plain max cosine to the rebuilt prototypes
        cos = np.full(len(emb), np.nan, dtype=np.float32)
        for task in CFG["tasks"]:
            idx = np.flatnonzero(tasks == task)
            cols = [i for i, (t, _) in enumerate(ft_keys) if t == task]
            if len(idx) == 0 or len(cols) < 2:
                continue
            p = torch.as_tensor(ft_protos[cols], dtype=torch.float32, device=DEVICE)
            for st in range(0, len(idx), 1024):
                chunk = idx[st:st + 1024]
                q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
                cos[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
        rows["E113b_finetuned_cosine"] = operating_point(cos[known_idx], cos[unknown_idx], target)

        folds.append({"held_out": int(held), "N_known": int(len(known_idx)),
                      "N_unknown": int(len(unknown_idx)), "signals": rows})
        print(f"E113 held={held} done", flush=True)

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
    lines = ["# E113 LOSO fine-tuning of the E95 encoder", "",
             "每折从冻结 E95 出发，只用 19 个注册身份微调编码器并重建原型，"
             "留出身份完全不参与训练。阈值固定使注册者接受率 = 0.85。未访问项目 Test。", "",
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
