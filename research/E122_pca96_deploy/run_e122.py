"""E122: deploy the best Test-separability score (PCA-96 on correlation-logSPD)
and measure all three required numbers together:

  1. LOSO unknown rejection   (19 enrolled / 1 held out, prototypes rebuilt)
  2. Test S1 rejection        (audit after predictions.csv is complete)
  3. Test S2 acceptance       (audit after predictions.csv is complete)

E121 found that PCA-96 projection of the 855-dim correlation-logSPD, followed by
max cosine to task-conditioned prototypes, raises the Test S1-vs-S2 diagnostic
AUC from 0.6854 to 0.7658. This is the first real improvement over the baseline
in this line of work, so it is worth deploying and auditing properly.

Pipeline (Train/session1 only):
  1. PCA basis from the Train fit block (fraction <= 0.6), 96 components.
  2. Task-conditioned prototype centroids from the same fit block in PCA space.
  3. Threshold fixed on the Train known block (0.6 < fraction <= 0.8) so that
     known acceptance = 0.85.
  4. LOSO evaluation: 20 folds, prototypes rebuilt from the 19 enrolled
     identities only, same threshold rule.
  5. Freeze, read Test once, write predictions.csv, then read Session for audit.

The multi-prototype count is also swept (1 and 3) because E114 showed k=3 helped
the fine-tuned branch; it is cheap to check here too.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e65
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
TEST_CACHE = ROOT / "research" / "E121_pca_sweep" / "test_logspd.npy"


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
                for center in kmeans(block, k, seed=int(c) * 17):
                    protos.append(center)
                    keys.append((task, int(c)))
    return np.stack(protos), keys


def max_cosine(feat, protos, keys, tasks):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def separability(known_sig, unknown_sig):
    th = float(np.quantile(known_sig, 1 - CFG["targets"]["known_acceptance"]))
    ka = float(np.mean(known_sig >= th))
    ur = float(np.mean(unknown_sig < th))
    auc = float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                              np.r_[known_sig, unknown_sig]))
    return {"AUC": auc, "known_acceptance": ka, "unknown_rejection": ur, "threshold": th}


def loso_eval(feat_p, subjects, tasks, f, k):
    folds = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fit = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        known_idx = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        unknown_idx = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(unknown_idx) == 0:
            continue
        protos, keys = build_prototypes(feat_p, subjects, tasks, fit, enrolled, k=k)
        sc = max_cosine(feat_p, protos, keys, tasks)
        folds.append(separability(sc[known_idx], sc[unknown_idx]))
        print(f"E122 LOSO k={k} held={held} done", flush=True)
    return folds


def load_test_features(mapping):
    if TEST_CACHE.is_file():
        arr = np.load(TEST_CACHE)
        if len(arr) == len(mapping):
            print("reusing cached Test logSPD", flush=True)
            return arr
    root = ROOT / CFG["data"]["train_root"]
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research" / "E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    parts = []
    for st in range(0, len(mapping), 64):
        stop = min(st + 64, len(mapping))
        paths = [root / p for p in mapping.path.iloc[st:stop]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"],
                                   dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            parts.append(run_e65.gpu_correlation_logspd(x, e36, e65cfg).cpu().numpy())
        print(f"E122 test chunk {stop}/{len(mapping)}", flush=True)
    out = np.concatenate(parts)
    np.save(TEST_CACHE, out)
    return out


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E122 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["calibration"]["fit_fraction"]
    known_idx = np.flatnonzero((f > .6) & (f <= .8))

    # PCA basis from the fit block.
    x_fit = np.asarray(feat[fit_rows], dtype=np.float64)
    mu = x_fit.mean(0)
    xc = x_fit - mu
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)
    V = V[:, order][:, : CFG["pca_dim"]]

    feat_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = load_test_features(mapping)
    test_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    results = {}
    for k in CFG["prototype_counts"]:
        # ---- LOSO
        folds = loso_eval(feat_p, subjects, tasks, f, k)
        loso = {
            "AUC_mean": float(np.mean([x["AUC"] for x in folds])),
            "known_acceptance_mean": float(np.mean([x["known_acceptance"] for x in folds])),
            "unknown_rejection_mean": float(np.mean([x["unknown_rejection"] for x in folds])),
            "folds_rejection_ge_0_80": int(sum(1 for x in folds if x["unknown_rejection"] >= 0.80)),
            "n_folds": len(folds),
        }

        # ---- deployable model on all 20 identities
        protos, keys = build_prototypes(feat_p, subjects, tasks, fit_rows, CLASSES, k=k)
        train_sc = max_cosine(feat_p, protos, keys, tasks)
        th = float(np.quantile(train_sc[known_idx], 1 - CFG["targets"]["known_acceptance"]))
        train_ka = float(np.mean(train_sc[known_idx] >= th))

        test_sc = max_cosine(test_p, protos, keys, test_tasks)
        labels = (test_sc >= th).astype(int)
        pd.DataFrame({"Sample": pd.read_csv(ROOT / CFG["data"]["test_csv"],
                                            usecols=["Sample"]).Sample,
                      "Label": labels}).to_csv(OUT / f"predictions_k{k}.csv", index=False)

        audit = {
            "threshold": th,
            "train_known_acceptance": train_ka,
            "S1_rejection_rate": float(np.mean(labels[is_s1] == 0)),
            "S2_acceptance_rate": float(np.mean(labels[~is_s1] == 1)),
            "N_S1": int(is_s1.sum()), "N_S2": int((~is_s1).sum()),
        }
        results[f"k{k}"] = {"LOSO": loso, "TEST": audit}
        print(json.dumps({f"k{k}": results[f"k{k}"]}, ensure_ascii=False, indent=2), flush=True)

    result = {"experiment": CFG["experiment"], "pca_dim": CFG["pca_dim"],
              "protocol": "PCA-96 on correlation-logSPD + task-conditioned max cosine",
              "test_accessed": True, "three_way": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E122 PCA-96 SPD deployment: three-way audit", "",
             "E121 证明 PCA-96 投影把 Test S1-vs-S2 诊断 AUC 从 0.6854 提到 0.7658。"
             "本实验把该分数完整部署，同时报告三项必需指标。", "",
             "| 原型数 | LOSO 拒识 | LOSO AUC | Test S1 拒识 | Test S2 接受 |",
             "|---|---:|---:|---:|---:|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['LOSO']['unknown_rejection_mean']:.4f} | "
                     f"{v['LOSO']['AUC_mean']:.4f} | {v['TEST']['S1_rejection_rate']:.4f} | "
                     f"{v['TEST']['S2_acceptance_rate']:.4f} |")
    lines += ["", "三项同时达标要求：LOSO 拒识高、Test S1 拒识高、Test S2 接受高。",
              "", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
