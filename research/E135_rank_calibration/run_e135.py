"""E135: remove the Train-to-Test score shift by per-task rank calibration.

E134 produced a structural finding: for every space tested,
LOSO rejection + Test S2 acceptance stays near 1.0-1.2, so the mean caps at
about 0.55-0.61. The reason is not the ranking quality (LOSO AUC 0.86, Test AUC
0.77) but a *shift*: Test scores sit systematically below Train scores, so any
Train-calibrated threshold accepts too few Test samples.

This experiment removes that shift without labels by rank-calibrating the Test
scores inside each task, so the acceptance rate on the Test batch matches the
Train-chosen rate. Which samples are accepted is still decided purely by the EEG
score, so S1 (lower scores) is rejected more often than S2.

Variants:
  RAW        Train-calibrated threshold, no calibration of the Test batch
  TASK_RANK  per-task empirical CDF of the Test batch, then the Train quantile
  K3         3-prototype variant, whose LOSO rejection is higher (0.83)

Reported: LOSO rejection, Test S1 rejection, Test S2 acceptance, their mean.
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
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
TEST_CACHE = ROOT / "research" / "E121_pca_sweep" / "test_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)
    return mu, V[:, order][:, : min(dim, x.shape[1])]


def kmeans(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return np.asarray(x, dtype=np.float32)
    c = x[rng.choice(len(x), size=k, replace=False)].astype(np.float32)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        n = np.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(n, c, atol=1e-6):
            break
        c = n
    return c


def build_prototypes(feat, subjects, tasks, fit_mask, classes, k=1):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            block = np.asarray(feat[sel], dtype=np.float32)
            if k == 1:
                protos.append(block.mean(0))
                keys.append((task, int(c)))
            else:
                for centre in kmeans(block, k, seed=int(c) * 17):
                    protos.append(centre)
                    keys.append((task, int(c)))
    return np.stack(protos), keys


def neg_d1(feat, protos, keys, tasks):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def task_rank(values, tasks):
    """Empirical CDF inside each task, computed on the given batch only."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        if len(idx) < 10:
            continue
        v = values[idx]
        order = np.argsort(v)
        ranks = np.empty(len(v), dtype=np.float32)
        ranks[order] = np.arange(len(v), dtype=np.float32) / max(len(v) - 1, 1)
        out[idx] = ranks
    return out


def train_rank_reference(values, tasks, fit_rows):
    """Map Train scores to per-task empirical CDFs; returns the reference values."""
    return task_rank(values, tasks) * 0 + 0.0, None


def run():
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["calibration"]["fit_fraction"]
    known_rows = (f > .6) & (f <= .8)

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = np.load(TEST_CACHE)

    mu, V = pca_basis(np.asarray(feat[fit_rows], dtype=np.float64), CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    results = {}
    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    for k in CFG["prototype_counts"]:
        protos, keys = build_prototypes(tr_p, subjects, tasks, fit_rows, CLASSES, k=k)
        tr_sc = neg_d1(tr_p, protos, keys, tasks)
        te_sc = neg_d1(te_p, protos, keys, test_tasks)

        # LOSO
        rejs = []
        for held in CLASSES:
            enrolled = np.setdiff1d(CLASSES, [held])
            fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
            ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
            ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
            if len(ui) == 0:
                continue
            p, kk = build_prototypes(tr_p, subjects, tasks, fitm, enrolled, k=k)
            s = neg_d1(tr_p, p, kk, tasks)
            th = float(np.quantile(s[ki], 1 - CFG["targets"]["known_acceptance"]))
            rejs.append(float(np.mean(s[ui] < th)))
        loso = float(np.mean(rejs))

        # Test with a Train-calibrated threshold, raw and rank-calibrated.
        a = CFG["targets"]["known_acceptance"]
        th_train = float(np.quantile(tr_sc[known_rows], 1 - a))
        lab_raw = (te_sc >= th_train).astype(int)

        tr_rank = task_rank(tr_sc, tasks)
        te_rank = task_rank(te_sc, test_tasks)
        th_rank = float(np.quantile(tr_rank[known_rows], 1 - a))
        lab_rank = (te_rank >= th_rank).astype(int)

        for label, lab in (("RAW", lab_raw), ("TASK_RANK", lab_rank)):
            s1 = float(np.mean(lab[is_s1] == 0))
            s2 = float(np.mean(lab[~is_s1] == 1))
            results[f"k{k}_{label}"] = {
                "LOSO_rejection": loso,
                "Test_S1_rejection": s1,
                "Test_S2_acceptance": s2,
                "overall_acceptance": float(np.mean(lab == 1)),
                "mean_objective": 0.5 * (loso + s2),
                "diagnostic_AUC": float(roc_auc_score(
                    np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
                    np.r_[te_sc[is_s1], te_sc[~is_s1]])),
            }
            print(json.dumps({f"k{k}_{label}": results[f"k{k}_{label}"]}, ensure_ascii=False),
                  flush=True)
            if len(order) == len(lab):
                pd.DataFrame({"Sample": order.Sample, "Label": lab}).to_csv(
                    OUT / f"predictions_k{k}_{label}.csv", index=False)

    best = max(results.items(), key=lambda kv: kv[1]["mean_objective"])
    out = {"experiment": CFG["experiment"],
           "objective": "(LOSO rejection + Test S2 acceptance)/2",
           "target": CFG["targets"]["mean_objective"],
           "best": {"name": best[0], **best[1]}, "variants": results}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E135 Removing the Train-to-Test score shift by per-task rank calibration", "",
             "E134 发现 LOSO 拒识 + Test S2 接受 ≈ 1.0–1.2，两者几乎完全负相关。"
             "原因不是排序质量（LOSO AUC 0.86、Test AUC 0.77），而是 Test 分数整体下移。"
             "本实验在每个任务内用 Test 批次自身的经验 CDF 校准分数（不使用标签）。", "",
             f"**最优：{best[0]}，平均值 {best[1]['mean_objective']:.4f}**", "",
             "| 配置 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 整体接受率 | 平均值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, v in results.items():
        lines.append(f"| {name} | {v['LOSO_rejection']:.4f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | {v['overall_acceptance']:.4f} | "
                     f"{v['mean_objective']:.4f} |")
    lines += ["", "**合规说明**：TASK_RANK 使用 Test 批次的分数分布做校准（无标签），属转导式校准。",
              "", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
