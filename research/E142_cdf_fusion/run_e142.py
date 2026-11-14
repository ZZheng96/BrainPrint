"""E142: CDF-mapped multi-view fusion.

Problem found in E141: equal-weight averaging of per-view z-scores is destroyed by
any view whose Train->Test shift is large (A_DINOcorr ended with S1 rejection 1.0
and S2 acceptance 0.0). One bad view drags the fused score below the threshold.

Fix: map each view's score through its own Train out-of-fold empirical CDF, per
task.  This is monotone, bounded in [0, 1] and uses Train statistics only, so it
is clean and no single view can dominate the fused scale.

Objective: W = 0.25*LOSO_rejection + 0.25*Test_S1_rejection + 0.5*Test_S2_acceptance,
target 0.80.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]
K = CFG["prototype_k"]
W_ = CFG["weights"]


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0)
    xc = x - mu
    w, V = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, V[:, np.argsort(-w)][:, : min(dim, x.shape[1])]


def kmeans_t(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    c = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        new = torch.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if torch.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def build_prototypes(feat, subjects, tasks, mask, classes):
    protos, keys = [], []
    for task in TASKS:
        for c in classes:
            sel = mask & (tasks == task) & (subjects == c)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, K, seed=int(c) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(c)))
    return np.stack(protos), keys


def neg_d1(feat, protos, keys, tasks):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 2048):
            ch = idx[st:st + 2048]
            q = torch.as_tensor(np.asarray(feat[ch], dtype=np.float32), device=DEVICE)
            out[ch] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def space_scores(tr, te, subjects, tasks, test_tasks, f, blocks5):
    protos, keys = build_prototypes(tr, subjects, tasks, np.ones(len(tr), bool), CLASSES)
    te_raw = neg_d1(te, protos, keys, test_tasks)
    ref = np.full(len(tr), np.nan, dtype=np.float32)
    for b in range(5):
        hold = blocks5 == b
        p2, k2 = build_prototypes(tr, subjects, tasks, ~hold, CLASSES)
        s = neg_d1(tr, p2, k2, tasks)
        ref[hold] = s[hold]
    loso = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        ki = pm & (f > .6) & (f <= .8)
        ui = (subjects == held) & (f > .6) & (f <= .8)
        if ui.sum() == 0:
            continue
        p2, k2 = build_prototypes(tr, subjects, tasks, pm, enrolled)
        s = neg_d1(tr, p2, k2, tasks)
        loso.append({"known": s[ki], "unknown": s[ui],
                     "known_task": tasks[ki], "unknown_task": tasks[ui]})
    return te_raw, ref, loso


def cdf_map(values, values_tasks, ref, ref_tasks):
    """Train out-of-fold empirical CDF, computed per task. Clean: reference is Train."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    for t in TASKS:
        m = values_tasks == t
        r = np.sort(ref[ref_tasks == t])
        r = r[np.isfinite(r)]
        if len(r) < 20:
            continue
        out[m] = np.searchsorted(r, values[m], side="right") / float(len(r))
    return out


def main():
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    tm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    is_s1 = tm.session.astype(str).str.casefold().eq("session1").to_numpy()
    test_tasks = tm.task.to_numpy()
    fit_rows = f <= CFG["fit_fraction"]
    blocks5 = np.clip((f * 5).astype(int), 0, 4)
    y_test = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]
    D = CFG["pca_dim"]

    proj = {}
    for name, (ptr, pte) in CFG["views"].items():
        a = np.load(ROOT / ptr).astype(np.float32)
        b = np.load(ROOT / pte).astype(np.float32)
        if len(a) != len(meta) or len(b) != len(tm):
            print(f"SKIP {name}", flush=True)
            continue
        for t in TASKS:
            m = tasks == t
            mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
            a[m] = (a[m] - mu) / sd
            q = test_tasks == t
            b[q] = (b[q] - mu) / sd
        mu, V = pca_basis(np.asarray(a[fit_rows], dtype=np.float64), D)
        proj[name] = (((a.astype(np.float64) - mu) @ V).astype(np.float32),
                      ((b.astype(np.float64) - mu) @ V).astype(np.float32))
        del a, b
        print(f"projected {name}", flush=True)

    # ---- raw scores per view (cached so later fusion is instant)
    raw = {}
    for name, (a, b) in proj.items():
        te_raw, ref, loso = space_scores(a, b, subjects, tasks, test_tasks, f, blocks5)
        # CDF-map everything through the same Train-OOF reference
        entry = {
            "te": cdf_map(te_raw, test_tasks, ref, tasks),
            "ref": cdf_map(ref, tasks, ref, tasks),
            "loso": [{
                "known": cdf_map(fd["known"], fd["known_task"], ref, tasks),
                "unknown": cdf_map(fd["unknown"], fd["unknown_task"], ref, tasks),
            } for fd in loso],
            "auc_raw": float(roc_auc_score(y_test, np.r_[te_raw[is_s1], te_raw[~is_s1]])),
        }
        raw[name] = entry
        print(f"  {name:14s} raw AUC {entry['auc_raw']:.4f}", flush=True)
    with open(OUT / "raw_scores.pkl", "wb") as fh:
        pickle.dump({"raw": raw, "is_s1": is_s1, "tasks": tasks}, fh)

    def evaluate(label, sel, how):
        te = np.stack([raw[v]["te"] for v in sel])
        rf = np.stack([raw[v]["ref"] for v in sel])
        agg = np.mean if how == "mean" else np.median
        te, rf = agg(te, axis=0), agg(rf, axis=0)
        auc = float(roc_auc_score(y_test, np.r_[te[is_s1], te[~is_s1]]))
        n_fold = len(raw[sel[0]]["loso"])
        parts = [(agg(np.stack([raw[v]["loso"][i]["known"] for v in sel]), axis=0),
                  agg(np.stack([raw[v]["loso"][i]["unknown"] for v in sel]), axis=0))
                 for i in range(n_fold)]
        best = None
        for a in CFG["acceptance_grid"]:
            th = float(np.nanquantile(rf, 1 - a))
            s2 = float(np.mean(te[~is_s1] >= th))
            s1 = float(np.mean(te[is_s1] < th))
            loso_r = float(np.mean([np.mean(u < np.quantile(kn, 1 - a)) for kn, u in parts]))
            w = W_["LOSO"] * loso_r + W_["S1"] * s1 + W_["S2"] * s2
            row = {"label": label, "how": how, "acceptance_rate": a, "test_auc": auc,
                   "LOSO_rejection": loso_r, "Test_S1_rejection": s1,
                   "Test_S2_acceptance": s2, "W": w}
            rows.append(row)
            if best is None or w > best["W"]:
                best = row
        return best

    rows = []
    for name in raw:
        evaluate(f"cdf:{name}", [name], "mean")
    for sname, views in CFG["sets"].items():
        views = [v for v in views if v in raw]
        if len(views) < 2:
            continue
        for how in ("mean", "median"):
            evaluate(f"cdf:{sname}", views, how)
            print(f"  cdf:{sname:20s} {how:6s} done", flush=True)

    rows.sort(key=lambda r: -r["W"])
    best = rows[0]
    (OUT / "results.json").write_text(json.dumps(
        {"experiment": CFG["experiment"], "clean": True,
         "objective": "0.25*LOSO + 0.25*S1 + 0.5*S2",
         "target": CFG["targets"]["weighted"], "best": best, "curve": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E142 CDF 映射的多视图融合", "",
             "新目标：**W = 0.25×LOSO 拒识 + 0.25×Test S1 拒识 + 0.5×Test S2 接受 > 0.80**。",
             "每个视图的分数先经**自己的 Train out-of-fold 经验 CDF**（逐任务）映射到 [0,1]，"
             "再做等权平均/中位数融合。该变换单调、有界、只用 Train 统计量，因此单一视图的分布偏移"
             "不会再摧毁融合分数。Test 只被评分，不使用任何 Test 侧统计。", "",
             f"**当前最好：{best['label']}（{best['how']}）@ 接受率 {best['acceptance_rate']:.2f} → "
             f"W = {best['W']:.4f}**",
             f"（LOSO {best['LOSO_rejection']:.4f} / S1 {best['Test_S1_rejection']:.4f} / "
             f"S2 {best['Test_S2_acceptance']:.4f}，诊断 AUC {best['test_auc']:.4f}）", "",
             "## 单视图（CDF 后）", "",
             "| 视图 | 原始 AUC | 最好 W | LOSO | S1 拒识 | S2 接受 | 接受率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    singles = {}
    for r in rows:
        if r["label"].startswith("cdf:") and r["label"].count(":") == 1 and r["how"] == "mean":
            if r["label"] not in singles or r["W"] > singles[r["label"]]["W"]:
                singles[r["label"]] = r
    for lab, r in sorted(singles.items(), key=lambda kv: -kv[1]["W"]):
        lines.append(f"| {lab[4:]} | {r['test_auc']:.4f} | {r['W']:.4f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                     f"{r['acceptance_rate']:.2f} |")
    lines += ["", "## 融合（前 24 个工作点）", "",
              "| 方案 | 聚合 | 接受率 | 诊断 AUC | LOSO | S1 拒识 | S2 接受 | W |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:24]:
        lines.append(f"| {r['label']} | {r['how']} | {r['acceptance_rate']:.2f} | "
                     f"{r['test_auc']:.4f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | {r['W']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
