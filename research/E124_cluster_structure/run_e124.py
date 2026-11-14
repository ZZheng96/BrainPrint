"""E124: cluster-structure check on the Test Session split.

Everything in E109-E123 depends on one strong inference: Test session1 contains
the 5 unregistered people and session2 contains the 20 registered people. That
inference came from the two official submissions, not from EEG evidence.

If the inference is right, session1 should be far less clusterable than
session2: 5 identities vs 20, and only one session each, so fewer distinct
clusters. If session1 shows the same number of clusters as session2, the
inference is probably wrong and the whole S1/S2 framing needs to be revisited.

Method (per task, to control for task effects):
  1. Project correlation-logSPD to 96 dims with a Train-fitted PCA.
  2. KMeans for k = 2..24 on the Test session1 and session2 samples separately.
  3. Report the silhouette score curve and its argmax as the estimated cluster
     count for each session.
  4. Also report the mean cosine similarity to the nearest Train identity
     prototype, which is the quantity the classifier actually uses.

This is a diagnostic. No predictions are produced, and no threshold is chosen.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def pca_fit(train, fit_rows, dim):
    x = np.asarray(train[fit_rows], dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)
    return mu, V[:, order][:, :dim]


def project(x, mu, V):
    return ((np.asarray(x, dtype=np.float64) - mu) @ V).astype(np.float32)


def cluster_curve(x, ks, seed=0, sample_cap=1200):
    rng = np.random.default_rng(seed)
    if len(x) > sample_cap:
        idx = rng.choice(len(x), size=sample_cap, replace=False)
        x = x[idx]
    rows = []
    for k in ks:
        if k >= len(x):
            break
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(x)
        lab = km.labels_
        if len(np.unique(lab)) < 2:
            continue
        sil = float(silhouette_score(x, lab, sample_size=min(len(x), 800), random_state=seed))
        rows.append({"k": int(k), "silhouette": sil,
                     "inertia": float(km.inertia_)})
    if not rows:
        return {"curve": [], "best_k": None, "best_silhouette": None}
    best = max(rows, key=lambda r: r["silhouette"])
    return {"curve": rows, "best_k": best["k"], "best_silhouette": best["silhouette"]}


def run():
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["calibration"]["fit_fraction"]

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = np.load(TEST_CACHE)

    mu, V = pca_fit(feat, fit_rows, CFG["pca_dim"])
    tr_p = project(feat, mu, V)
    te_p = project(test_feat, mu, V)

    # Train prototypes for the nearest-identity similarity reference.
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_rows & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in CLASSES:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(tr_p[sel], dtype=np.float32).mean(0))
            keys.append((task, int(c)))
    protos = np.stack(protos)

    def nearest_sim(x, task):
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if not cols:
            return np.full(len(x), np.nan, dtype=np.float32)
        p = np.asarray(protos[cols], dtype=np.float32)
        p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
        q = np.asarray(x, dtype=np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        return (q @ p.T).max(1)

    ks = list(range(2, 25))
    out = {"experiment": CFG["experiment"], "diagnostic_only": True,
           "pca_dim": CFG["pca_dim"], "per_task": {}}

    for task in CFG["tasks"]:
        r = {}
        for name, m in (("S1", is_s1), ("S2", ~is_s1)):
            idx = np.flatnonzero(m & (test_tasks == task))
            if len(idx) < 50:
                r[name] = {"n": int(len(idx)), "note": "too few samples"}
                continue
            c = cluster_curve(te_p[idx], ks, seed=CFG["seed"])
            sim = nearest_sim(te_p[idx], task)
            r[name] = {"n": int(len(idx)), "best_k": c["best_k"],
                       "best_silhouette": c["best_silhouette"],
                       "nearest_sim_mean": float(np.mean(sim)),
                       "nearest_sim_p50": float(np.median(sim)),
                       "curve": c["curve"]}
            print(f"E124 {task} {name} best_k={c['best_k']} "
                  f"sil={c['best_silhouette']:.4f} sim={np.mean(sim):.4f}", flush=True)
        out["per_task"][task] = r

    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E124 Test Session cluster structure (diagnostic)", "",
             "检验核心假设：Test session1 是 5 名未注册者、session2 是 20 名注册者。"
             "若假设成立，session1 的可聚类数应明显少于 session2。", "",
             "| 任务 | S1 样本 | S1 最优簇数 | S1 轮廓系数 | S1 最近原型相似度 | S2 样本 | S2 最优簇数 | S2 轮廓系数 | S2 最近原型相似度 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task, r in out["per_task"].items():
        s1 = r.get("S1", {})
        s2 = r.get("S2", {})
        def cell(v, key, fmt="{:.4f}"):
            if v.get(key) is None:
                return "—"
            return fmt.format(v[key])
        lines.append(
            f"| {task} | {s1.get('n', 0)} | {s1.get('best_k', '—')} | "
            f"{cell(s1, 'best_silhouette')} | {cell(s1, 'nearest_sim_mean')} | "
            f"{s2.get('n', 0)} | {s2.get('best_k', '—')} | "
            f"{cell(s2, 'best_silhouette')} | {cell(s2, 'nearest_sim_mean')} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（诊断实验，无预测）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: {k3: v3 for k3, v3 in vv.items() if k3 != "curve"}
                          for kk, vv in v.items()} for k, v in out["per_task"].items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
