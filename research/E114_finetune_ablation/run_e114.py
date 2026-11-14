"""E114: strengthen the E113b LOSO fine-tuning recipe.

E113b (fine-tune the E95 encoder per LOSO fold with the held-out identity
excluded, then score with max cosine to rebuilt task-conditioned prototypes)
reached held-out rejection 0.8428, the first signal to beat the E102 distance
ratio (0.8310). But only 14/20 folds cleared 0.80, so the recipe is unstable.

This experiment ablates and strengthens that recipe. Everything is Train-only;
the project Test is never read.

Variants (all share the E113b pipeline: per-fold fine-tune on the 19 enrolled
identities, rebuild task-conditioned prototypes, score with max cosine,
threshold fixed at known acceptance 0.85):
  V0  E113b reference           8 epochs, lr 2e-4
  V1  longer fine-tune          16 epochs, lr 2e-4
  V2  higher lr                 8 epochs,  lr 5e-4
  V3  multi-prototype           8 epochs,  lr 2e-4, 3 prototypes per (task, id)
  V4  multi-prototype + longer  16 epochs, lr 2e-4, 3 prototypes per (task, id)
  V5  multi-prototype + higher  8 epochs,  lr 5e-4, 3 prototypes per (task, id)

Multi-prototype: instead of one centroid per (task, identity), keep K centroids
from a deterministic k-means over the fit block. A held-out identity that
happens to resemble one sub-mode of an enrolled identity is then still far from
the other sub-modes, which should raise rejection without hurting acceptance.
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


def kmeans(x, k, iters=20, seed=0):
    """Deterministic k-means; returns k centroids."""
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return np.asarray(x, dtype=np.float32)
    centers = x[rng.choice(len(x), size=k, replace=False)].astype(np.float32)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        new = np.stack([x[assign == j].mean(0) if (assign == j).any() else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers, atol=1e-6):
            break
        centers = new
    return centers


def build_task_prototypes(emb, subjects, tasks, fit_mask, classes, k=1):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            block = np.asarray(emb[sel], dtype=np.float32)
            if k == 1:
                protos.append(block.mean(0))
                keys.append((task, int(c)))
            else:
                for j, center in enumerate(kmeans(block, k, seed=int(c) * 17 + j)):
                    protos.append(center)
                    keys.append((task, int(c)))
    return np.stack(protos), keys


def max_cosine(emb, protos, keys, tasks):
    """Max cosine to the prototypes of the sample's own task."""
    out = np.full(len(emb), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        p = torch.nn.functional.normalize(p, dim=1)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
            q = torch.nn.functional.normalize(q, dim=1)
            out[chunk] = (q @ p.T).max(1).values.cpu().numpy()
    return out


def embed(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def finetune_fold(raw, subjects, fit_rows, enrolled, seed, epochs, lr):
    e54cfg = E54.load_config()
    model = E54.EEGNetEmbedding(len(enrolled), e54cfg).to(DEVICE)
    saved = dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    keep = [int(np.flatnonzero(CLASSES == c)[0]) for c in enrolled]
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

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=CFG["training"]["weight_decay"])
    batch = CFG["training"]["batch_size"]
    temperature = float(e54cfg["model"]["cosine_temperature"])
    arpl_weight = float(e54cfg["model"]["arpl_weight"])
    margin = float(e54cfg["model"]["arpl_margin"])
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
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
        raise RuntimeError("E114 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)

    target = CFG["targets"]["known_acceptance"]
    variants = CFG["variants"]
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

        # B4 reference: E102 task-conditioned SPD distance ratio.
        sp_protos, sp_keys = build_task_prototypes(feat, subjects, tasks, fit, enrolled, k=1)
        sp_cos = max_cosine(feat, sp_protos, sp_keys, tasks)
        rows["B4_E102_spd_cosine"] = operating_point(sp_cos[known_idx], sp_cos[unknown_idx], target)

        # Cache fine-tuned embeddings per (epochs, lr) so variants share work.
        cache = {}
        for name, spec in variants.items():
            key = (spec["epochs"], spec["lr"])
            if key not in cache:
                model = finetune_fold(raw, subjects, fit_rows, enrolled,
                                      CFG["seed"] + int(held), spec["epochs"], spec["lr"])
                cache[key] = embed(model, raw)
            emb = cache[key]
            protos, keys = build_task_prototypes(emb, subjects, tasks, fit, enrolled,
                                                 k=spec["prototypes"])
            cos = max_cosine(emb, protos, keys, tasks)
            rows[name] = operating_point(cos[known_idx], cos[unknown_idx], target)

        folds.append({"held_out": int(held), "N_known": int(len(known_idx)),
                      "N_unknown": int(len(unknown_idx)), "signals": rows})
        print(f"E114 held={held} done", flush=True)

    names = sorted({n for r in folds for n in r["signals"]})
    summary = {}
    for n in names:
        vals = [r["signals"][n] for r in folds if n in r["signals"]]
        summary[n] = {
            "AUC_mean": float(np.mean([v["AUC"] for v in vals])),
            "known_acceptance_mean": float(np.mean([v["known_acceptance"] for v in vals])),
            "unknown_rejection_mean": float(np.mean([v["unknown_rejection"] for v in vals])),
            "unknown_rejection_median": float(np.median([v["unknown_rejection"] for v in vals])),
            "unknown_rejection_min": float(np.min([v["unknown_rejection"] for v in vals])),
            "folds_rejection_ge_0_80": int(sum(1 for v in vals if v["unknown_rejection"] >= 0.80)),
            "folds_rejection_ge_0_50": int(sum(1 for v in vals if v["unknown_rejection"] >= 0.50)),
            "n_folds": len(vals),
        }

    result = {"experiment": CFG["experiment"], "test_accessed": False,
              "protocol": "LOSO; 19 enrolled, 1 held out; threshold fixed at known acceptance 0.85",
              "targets": CFG["targets"], "variants": variants,
              "summary": summary, "folds": folds}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    base = summary["B4_E102_spd_cosine"]["unknown_rejection_mean"]
    lines = ["# E114 Strengthening the E113b LOSO fine-tuning recipe", "",
             "在 E113b（留身份微调 + cosine）基础上做消融：微调轮数、学习率、多原型。"
             "阈值固定使注册者接受率 = 0.85。未访问项目 Test。", "",
             "## 结果", "",
             "| 变体 | LOSO AUC | known acc | 留出者拒识均值 | 中位数 | 最差折 | ≥0.80 折数 | ≥0.50 折数 | 相对 B4 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, s in summary.items():
        delta = s["unknown_rejection_mean"] - base
        lines.append(f"| {n} | {s['AUC_mean']:.4f} | {s['known_acceptance_mean']:.4f} | "
                     f"{s['unknown_rejection_mean']:.4f} | {s['unknown_rejection_median']:.4f} | "
                     f"{s['unknown_rejection_min']:.4f} | "
                     f"{s['folds_rejection_ge_0_80']}/{s['n_folds']} | "
                     f"{s['folds_rejection_ge_0_50']}/{s['n_folds']} | {delta:+.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
