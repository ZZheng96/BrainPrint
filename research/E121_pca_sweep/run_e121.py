"""E121: sweep the PCA dimensionality for the WCCN-whitened SPD space.

E120 found that PCA-64 followed by WCCN whitening raises the Test S1-vs-S2
diagnostic AUC from 0.6854 to 0.7614, the first real gain over the baseline.
This experiment sweeps the retained dimension and a few post-processing
variants, all fitted on the Train fit block only.

Variants per dimension d:
  p{d}_wccn_max     PCA(d) -> WCCN -> max cosine to task-conditioned prototypes
  p{d}_max          PCA(d) -> max cosine (no whitening)
  p{d}_wccn_z_max   PCA(d) -> WCCN -> per-task z-score -> max cosine

Plus:
  p{d}_ratio        PCA(d) -> WCCN -> distance ratio
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
TEST_CACHE = OUT / "test_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def build_prototypes(feat, subjects, tasks, fit_mask, classes):
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


def score_matrix(feat, protos, keys, tasks):
    n = len(feat)
    out = np.full((n, len(keys)), -np.inf, dtype=np.float32)
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
            out[np.ix_(chunk, cols)] = -d
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


def wccn_matrix(feat, subjects, fit_rows, shrinkage=0.05):
    dim = feat.shape[1]
    Sw = np.zeros((dim, dim), dtype=np.float64)
    total = 0
    for c in np.unique(subjects[fit_rows]):
        sel = fit_rows & (subjects == c)
        x = np.asarray(feat[sel], dtype=np.float64)
        if len(x) < 2:
            continue
        xc = x - x.mean(0)
        Sw += xc.T @ xc
        total += len(x)
    Sw /= max(total, 1)
    mu = float(np.trace(Sw) / dim)
    Sw = (1 - shrinkage) * Sw + shrinkage * mu * np.eye(dim)
    w, V = np.linalg.eigh(Sw)
    w = np.clip(w, 1e-6, None)
    return (V / np.sqrt(w)) @ V.T


def task_standardize(train, test, train_tasks, test_tasks):
    tr = np.asarray(train, dtype=np.float32).copy()
    te = np.asarray(test, dtype=np.float32).copy()
    for task in CFG["tasks"]:
        m = train_tasks == task
        if m.sum() < 10:
            continue
        mu = tr[m].mean(0)
        sd = tr[m].std(0) + 1e-6
        tr[m] = (tr[m] - mu) / sd
        mt = test_tasks == task
        if mt.any():
            te[mt] = (te[mt] - mu) / sd
    return tr, te


def separability(score, is_s1):
    s1 = score[is_s1]
    s2 = score[~is_s1]
    auc = float(roc_auc_score(np.r_[np.zeros(len(s1)), np.ones(len(s2))], np.r_[s1, s2]))
    grid = np.unique(np.quantile(score, np.linspace(0, 1, 401)))
    ka = np.asarray([np.mean(s1 < t) for t in grid])
    ur = np.asarray([np.mean(s2 >= t) for t in grid])
    m80 = ka >= 0.80
    return {
        "diagnostic_AUC": auc,
        "best_S2_accept_at_S1_rej_0.80": float(ur[m80].max()) if m80.any() else 0.0,
        "best_joint_min": float(np.max(np.minimum(ka, ur))),
    }


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
        print(f"E121 test chunk {stop}/{len(mapping)}", flush=True)
    out = np.concatenate(parts)
    np.save(TEST_CACHE, out)
    return out


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E121 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["calibration"]["fit_fraction"]

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = load_test_features(mapping)

    # PCA basis from the Train fit block.
    x_fit = np.asarray(feat[fit_rows], dtype=np.float64)
    mu_pca = x_fit.mean(0)
    xc = x_fit - mu_pca
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]

    results = {}
    for dim in CFG["pca_dims"]:
        dim = min(int(dim), V.shape[1])
        Vd = V[:, :dim]
        tr_p = ((np.asarray(feat, dtype=np.float64) - mu_pca) @ Vd).astype(np.float32)
        te_p = ((np.asarray(test_feat, dtype=np.float64) - mu_pca) @ Vd).astype(np.float32)

        # plain PCA
        protos, keys = build_prototypes(tr_p, subjects, tasks, fit_rows, CLASSES)
        sc = score_matrix(te_p, protos, keys, test_tasks).max(1)
        results[f"p{dim}_max"] = separability(np.asarray(sc, dtype=np.float64), is_s1)

        # PCA + WCCN
        Wp = wccn_matrix(tr_p, subjects, fit_rows)
        tr_pw = (tr_p.astype(np.float64) @ Wp).astype(np.float32)
        te_pw = (te_p.astype(np.float64) @ Wp).astype(np.float32)
        protos, keys = build_prototypes(tr_pw, subjects, tasks, fit_rows, CLASSES)
        sc = score_matrix(te_pw, protos, keys, test_tasks).max(1)
        results[f"p{dim}_wccn_max"] = separability(np.asarray(sc, dtype=np.float64), is_s1)

        # PCA + WCCN + per-task z
        tr_z, te_z = task_standardize(tr_pw, te_pw, tasks, test_tasks)
        protos, keys = build_prototypes(tr_z, subjects, tasks, fit_rows, CLASSES)
        sc = score_matrix(te_z, protos, keys, test_tasks).max(1)
        results[f"p{dim}_wccn_z_max"] = separability(np.asarray(sc, dtype=np.float64), is_s1)

        # PCA + WCCN + ratio
        protos, keys = build_prototypes(tr_pw, subjects, tasks, fit_rows, CLASSES)
        sc = distance_ratio(te_pw, protos, keys, test_tasks)
        results[f"p{dim}_wccn_ratio"] = separability(np.asarray(sc, dtype=np.float64), is_s1)

        print(f"E121 dim={dim} done", flush=True)

    best = max(results.items(), key=lambda kv: kv[1]["diagnostic_AUC"])
    result = {"experiment": CFG["experiment"], "diagnostic_only": True,
              "warning": "Session is used only to measure separability, never to build a score.",
              "N_S1": int(is_s1.sum()), "N_S2": int((~is_s1).sum()),
              "best": {"name": best[0], **best[1]}, "scores": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E121 PCA dimension sweep for the WCCN-whitened SPD space (diagnostic)", "",
             "E120 发现 PCA-64 + WCCN 把 Test S1-vs-S2 AUC 从 0.6854 提到 0.7614。"
             "本实验扫描保留维数与后处理组合，全部只用 Train 拟合块。", "",
             f"**最优：{best[0]}，诊断 AUC {best[1]['diagnostic_AUC']:.4f}，"
             f"S1≥0.80 时 S2 接受 {best[1]['best_S2_accept_at_S1_rej_0.80']:.4f}。**", "",
             "| 变体 | 诊断 AUC | S1≥0.80 时 S2 接受 | 最佳均衡点 |",
             "|---|---:|---:|---:|"]
    for name in sorted(results):
        s = results[name]
        lines.append(f"| {name} | {s['diagnostic_AUC']:.4f} | "
                     f"{s['best_S2_accept_at_S1_rej_0.80']:.4f} | {s['best_joint_min']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
