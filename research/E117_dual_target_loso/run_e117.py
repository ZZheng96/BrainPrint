"""E117: LOSO search for a candidate that meets BOTH targets simultaneously.

The user's requirement is explicit: high S2 acceptance AND high LOSO rejection,
both must pass. Previous experiments fixed the S2 operating point at known
acceptance 0.85 and only reported rejection, which hides whether a candidate
could actually reach a higher acceptance.

This experiment reports the full LOSO operating curve for each candidate:
for every threshold we record (known acceptance, unknown rejection), then ask
whether any threshold satisfies
    known acceptance >= 0.90  AND  unknown rejection >= 0.80
and also the stricter 0.95 / 0.80 and 0.90 / 0.85 combinations.

Candidates (all Train/session1 only, no project Test):
  C1  E95 raw cosine (frozen)                     high-S2 branch
  C2  E102 task-conditioned SPD distance ratio    high-S1 branch
  C3  E113b fine-tuned encoder + cosine           best single signal so far
  C4  C3 with 3 prototypes per (task, identity)   E114 V3
  C5  C3 with 5 prototypes per (task, identity)
  C6  C3 with 8 prototypes per (task, identity)
  C7  C3 + C2 rank fusion (mean)
  C8  C3 + C1 rank fusion (mean)

The multi-prototype count is the one lever that helped in E114, so we extend it.
The fusion candidates test whether the high-S2 branch can lift acceptance
without destroying the fine-tuned branch's rejection.
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
                for center in kmeans(block, k, seed=int(c) * 17):
                    protos.append(center)
                    keys.append((task, int(c)))
    return np.stack(protos), keys


def max_cosine(emb, protos, keys, tasks):
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


def distance_ratio(feat, protos, keys, tasks):
    ratio = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
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

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["training"]["learning_rate"],
                            weight_decay=CFG["training"]["weight_decay"])
    batch = CFG["training"]["batch_size"]
    temperature = float(e54cfg["model"]["cosine_temperature"])
    arpl_weight = float(e54cfg["model"]["arpl_weight"])
    margin = float(e54cfg["model"]["arpl_margin"])
    rng = np.random.default_rng(seed)
    for _ in range(CFG["training"]["epochs"]):
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


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def operating_curve(known_sig, unknown_sig):
    """Full curve plus the best joint point under several target pairs."""
    auc = float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                              np.r_[known_sig, unknown_sig]))
    grid = np.unique(np.quantile(np.r_[known_sig, unknown_sig], np.linspace(0, 1, 401)))
    ka = np.asarray([np.mean(known_sig >= t) for t in grid])
    ur = np.asarray([np.mean(unknown_sig < t) for t in grid])

    def best_at(min_ka, min_ur):
        m = (ka >= min_ka) & (ur >= min_ur)
        if not m.any():
            return {"feasible": False, "known_acceptance": None, "unknown_rejection": None,
                    "threshold": None}
        # Among feasible points prefer the highest known acceptance.
        i = int(np.flatnonzero(m)[np.argmax(ka[m])])
        return {"feasible": True, "known_acceptance": float(ka[i]),
                "unknown_rejection": float(ur[i]), "threshold": float(grid[i])}

    joint = float(np.max(np.minimum(ka, ur)))
    return {
        "AUC": auc,
        "best_joint_min": joint,
        "at_ka_0.85_ur_0.80": best_at(0.85, 0.80),
        "at_ka_0.90_ur_0.80": best_at(0.90, 0.80),
        "at_ka_0.90_ur_0.85": best_at(0.90, 0.85),
        "at_ka_0.95_ur_0.80": best_at(0.95, 0.80),
        "at_ka_0.95_ur_0.85": best_at(0.95, 0.85),
        "at_ka_0.95_ur_0.95": best_at(0.95, 0.95),
    }


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E117 requires CUDA")
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
    with torch.no_grad():
        s95 = []
        for st in range(0, len(raw), 256):
            s95.append(e95.scores(raw[st:st + 256]).cpu().numpy())
    s95 = np.concatenate(s95, axis=0)

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

        # C1: frozen E95 raw cosine
        rows["C1_E95_raw_cosine"] = operating_curve(s95[known_idx].max(1),
                                                    s95[unknown_idx].max(1))

        # C2: E102 task-conditioned SPD distance ratio
        sp_protos, sp_keys = build_task_prototypes(feat, subjects, tasks, fit, enrolled, k=1)
        sp_ratio = distance_ratio(feat, sp_protos, sp_keys, tasks)
        rows["C2_E102_spd_ratio"] = operating_curve(sp_ratio[known_idx], sp_ratio[unknown_idx])

        # C3-C6: fine-tuned encoder with 1/3/5/8 prototypes
        model = finetune_fold(raw, subjects, fit_rows, enrolled, CFG["seed"] + int(held))
        emb = embed(model, raw)
        ft_scores = {}
        for k in CFG["prototype_counts"]:
            protos, keys = build_task_prototypes(emb, subjects, tasks, fit, enrolled, k=k)
            cos = max_cosine(emb, protos, keys, tasks)
            ft_scores[k] = cos
            name = "C3_E113b_cosine" if k == 1 else f"C{2 + CFG['prototype_counts'].index(k)}_V3_k{k}"
            rows[name] = operating_curve(cos[known_idx], cos[unknown_idx])

        # C7/C8: rank fusion of the k=3 fine-tuned score with the high-S2 branches
        base = ft_scores[3]
        pool = np.r_[known_idx, unknown_idx]
        r_base = rank_normalize(base[pool], base[pool])
        r_spd = rank_normalize(sp_ratio[pool], sp_ratio[pool])
        r_raw = rank_normalize(s95[pool].max(1), s95[pool].max(1))
        nk = len(known_idx)
        fused_spd = 0.5 * (r_base + r_spd)
        fused_raw = 0.5 * (r_base + r_raw)
        rows["C7_V3k3_plus_spd_ratio"] = operating_curve(fused_spd[:nk], fused_spd[nk:])
        rows["C8_V3k3_plus_raw_cosine"] = operating_curve(fused_raw[:nk], fused_raw[nk:])

        folds.append({"held_out": int(held), "N_known": int(len(known_idx)),
                      "N_unknown": int(len(unknown_idx)), "signals": rows})
        print(f"E117 held={held} done", flush=True)

    names = sorted({n for r in folds for n in r["signals"]})
    summary = {}
    for n in names:
        vals = [r["signals"][n] for r in folds if n in r["signals"]]
        entry = {"AUC_mean": float(np.mean([v["AUC"] for v in vals])),
                 "best_joint_min_mean": float(np.mean([v["best_joint_min"] for v in vals])),
                 "n_folds": len(vals)}
        for key in ("at_ka_0.85_ur_0.80", "at_ka_0.90_ur_0.80", "at_ka_0.90_ur_0.85",
                    "at_ka_0.95_ur_0.80", "at_ka_0.95_ur_0.85", "at_ka_0.95_ur_0.95"):
            feasible = [v[key] for v in vals if v[key]["feasible"]]
            entry[key] = {
                "folds_feasible": len(feasible),
                "known_acceptance_mean": float(np.mean([x["known_acceptance"] for x in feasible]))
                if feasible else None,
                "unknown_rejection_mean": float(np.mean([x["unknown_rejection"] for x in feasible]))
                if feasible else None,
            }
        summary[n] = entry

    result = {"experiment": CFG["experiment"], "test_accessed": False,
              "protocol": "LOSO; 19 enrolled, 1 held out; full operating curve per fold",
              "targets": CFG["targets"], "summary": summary, "folds": folds}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E117 LOSO search for a candidate meeting BOTH targets", "",
             "对每个候选扫描完整工作曲线，统计有多少折能同时满足 known acceptance 与 "
             "unknown rejection 的目标组合。未访问项目 Test。", "",
             "## 双达标折数", "",
             "| 候选 | AUC | 最佳均衡点 | ka≥0.85 & ur≥0.80 | ka≥0.90 & ur≥0.80 | ka≥0.90 & ur≥0.85 | ka≥0.95 & ur≥0.80 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for n, s in summary.items():
        def cell(key):
            v = s[key]
            if v["folds_feasible"] == 0:
                return "0"
            return (f"{v['folds_feasible']}/{s['n_folds']} "
                    f"({v['known_acceptance_mean']:.2f}/{v['unknown_rejection_mean']:.2f})")
        lines.append(f"| {n} | {s['AUC_mean']:.4f} | {s['best_joint_min_mean']:.4f} | "
                     f"{cell('at_ka_0.85_ur_0.80')} | {cell('at_ka_0.90_ur_0.80')} | "
                     f"{cell('at_ka_0.90_ur_0.85')} | {cell('at_ka_0.95_ur_0.80')} |")
    lines += ["", "括号内为可行折上的 (known acceptance 均值 / unknown rejection 均值)。",
              "", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
