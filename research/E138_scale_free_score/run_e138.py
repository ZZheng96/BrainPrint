"""E138: scale-free score forms to remove the Train->Test score shift.

Diagnosis (E137): the Train out-of-fold reference is heavily inflated because an
out-of-fold trial is scored against prototypes built from ~80% of its own
subject's same-session sibling trials.  A Test trial never has that advantage,
so every Train-calibrated threshold is far too strict on Test.

Here we test score forms whose scale is derived from Train-side quantities only:

  d1            raw nearest-prototype distance (baseline)
  d1_over_mean  d1 / mean_c d_c                       per-sample distance scale
  d1_over_rms   d1 / sqrt(mean_c d_c^2)               per-sample RMS scale
  d1_over_sig   d1 / sigma_c, sigma_c = mean distance of identity c's fit trials
                to its own prototype                    per-identity scatter
  d1_over_both  (d1 / sigma_c) / mean_c (d_c / sigma_c)
  margin_d1     (d2 - d1) / d1

All normalisers are fitted on Train only.  Report, for every form: Test AUC,
the residual Train->Test shift (in Train-OOF standard deviations), and the full
three-way objective over the acceptance grid.
"""
from __future__ import annotations
import json
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


def build_prototypes(feat, subjects, tasks, mask, classes, k):
    protos, keys = [], []
    for task in TASKS:
        for c in classes:
            sel = mask & (tasks == task) & (subjects == c)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, k, seed=int(c) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(c)))
    return np.stack(protos), keys


def distance_matrix(feat, protos, keys, tasks):
    """Full distance matrix restricted to same-task prototype columns (others = inf)."""
    n, m = len(feat), len(protos)
    d = np.full((n, m), np.inf, dtype=np.float32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = np.array([i for i, (t, _) in enumerate(keys) if t == task], dtype=int)
        if len(idx) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 2048):
            ch = idx[st:st + 2048]
            q = torch.as_tensor(np.asarray(feat[ch], dtype=np.float32), device=DEVICE)
            d[np.ix_(ch, cols)] = torch.cdist(q, p).cpu().numpy()
    return d


def proto_scatter(d_all, keys, subjects, tasks, fit_mask):
    """sigma_c: mean distance from identity c's own fit trials to its nearest own prototype."""
    sig = {}
    nearest = np.argmin(d_all, axis=1)
    own = d_all[np.arange(len(d_all)), nearest]
    for task in TASKS:
        for c in CLASSES:
            # trials whose nearest prototype belongs to this identity
            sel = fit_mask & (subjects == c) & (tasks == task) & np.isin(nearest, index_of(keys, task, c))
            if sel.sum() >= 5:
                sig[(task, int(c))] = float(np.mean(own[sel]))
    fallback = float(np.mean(list(sig.values()))) if sig else 1.0
    for task in TASKS:
        for c in CLASSES:
            sig.setdefault((task, int(c)), fallback)
    return sig


def index_of(keys, task, c):
    return [i for i, (t, cc) in enumerate(keys) if t == task and cc == int(c)]


def forms(d, keys, tasks, sigma):
    """Compute the score forms from a distance matrix."""
    n, m = d.shape
    d1 = d.min(1)
    order = np.argsort(d, axis=1)
    d2 = np.take_along_axis(d, order[:, 1:2], axis=1)[:, 0]
    mean_d = np.where(np.isfinite(d), d, 0.0).sum(1) / np.isfinite(d).sum(1)
    rms_d = np.sqrt((np.where(np.isfinite(d), d, 0.0) ** 2).sum(1) / np.isfinite(d).sum(1))

    sig_vec = np.empty(m, dtype=np.float32)
    for j, (t, c) in enumerate(keys):
        sig_vec[j] = sigma[(t, int(c))]
    dsig = d / sig_vec[None, :]
    d1s = dsig.min(1)
    mean_dsig = np.where(np.isfinite(dsig), dsig, 0.0).sum(1) / np.isfinite(dsig).sum(1)

    out = {
        "d1": -d1,
        "d1_over_mean": -(d1 / np.maximum(mean_d, 1e-6)),
        "d1_over_rms": -(d1 / np.maximum(rms_d, 1e-6)),
        "d1_over_sig": -d1s,
        "d1_over_both": -(d1s / np.maximum(mean_dsig, 1e-6)),
        "margin_d1": -((d2 - d1) / np.maximum(d1, 1e-6)),
    }
    return out


def main():
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    tm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    is_s1 = tm.session.astype(str).str.casefold().eq("session1").to_numpy()
    test_tasks = tm.task.to_numpy()

    feat = np.load(ROOT / CFG["data"]["train_features"])
    test_feat = np.load(ROOT / CFG["data"]["test_features"])
    fit_rows = f <= CFG["fit_fraction"]
    blocks5 = np.clip((f * 5).astype(int), 0, 4)

    def standardize(a, b):
        a = a.astype(np.float32).copy(); b = b.astype(np.float32).copy()
        for t in TASKS:
            m = tasks == t
            mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
            a[m] = (a[m] - mu) / sd
            q = test_tasks == t
            b[q] = (b[q] - mu) / sd
        return a, b

    feat, test_feat = standardize(feat, test_feat)
    mu, V = pca_basis(np.asarray(feat[fit_rows], dtype=np.float64), CFG["pca_dim"])
    trp = ((feat.astype(np.float64) - mu) @ V).astype(np.float32)
    tep = ((test_feat.astype(np.float64) - mu) @ V).astype(np.float32)
    k = CFG["prototype_k"]
    print("space ready", flush=True)

    # ---- deployed prototypes: whole Train
    allm = np.ones(len(trp), bool)
    protos, keys = build_prototypes(trp, subjects, tasks, allm, CLASSES, k)
    sigma = proto_scatter(distance_matrix(trp, protos, keys, tasks), keys,
                          subjects, tasks, fit_rows)
    d_te = distance_matrix(tep, protos, keys, test_tasks)
    te_forms = forms(d_te, keys, tasks, sigma)
    print("test distances ready", flush=True)

    # ---- Train out-of-fold reference
    ref_forms = {nm: np.full(len(trp), np.nan, np.float32) for nm in te_forms}
    for b in range(5):
        hold = blocks5 == b
        p2, k2 = build_prototypes(trp, subjects, tasks, ~hold, CLASSES, k)
        s2 = proto_scatter(distance_matrix(trp, p2, k2, tasks), k2, subjects, tasks, fit_rows & ~hold)
        fm = forms(distance_matrix(trp[hold], p2, k2, tasks[hold]), k2, tasks[hold], s2)
        for nm in ref_forms:
            ref_forms[nm][hold] = fm[nm]
    print("oof reference ready", flush=True)

    # ---- LOSO folds
    loso = {nm: [] for nm in te_forms}
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        ki = pm & (f > .6) & (f <= .8)
        ui = (subjects == held) & (f > .6) & (f <= .8)
        if ui.sum() == 0:
            continue
        p2, k2 = build_prototypes(trp, subjects, tasks, pm, enrolled, k)
        s2 = proto_scatter(distance_matrix(trp, p2, k2, tasks), k2, subjects, tasks, fit_rows & pm)
        fk = forms(distance_matrix(trp[ki], p2, k2, tasks[ki]), k2, tasks[ki], s2)
        fu = forms(distance_matrix(trp[ui], p2, k2, tasks[ui]), k2, tasks[ui], s2)
        for nm in loso:
            loso[nm].append((fk[nm], fu[nm], tasks[ki], tasks[ui]))
    print("loso ready", flush=True)

    rows = []
    for nm in te_forms:
        te, ref = te_forms[nm], ref_forms[nm]
        auc = float(roc_auc_score(np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))],
                                  np.r_[te[is_s1], te[~is_s1]]))
        sd_ref = float(np.nanstd(ref))
        shift = float(np.nanmedian(te) - np.nanmedian(ref)) / (sd_ref + 1e-9)
        for a in CFG["acceptance_grid"]:
            th = float(np.nanquantile(ref, 1 - a))
            s2a = float(np.mean(te[~is_s1] >= th))
            s1r = float(np.mean(te[is_s1] < th))
            rejs = []
            for fk, fu, kt, ut in loso[nm]:
                thf = float(np.quantile(fk, 1 - a))
                rejs.append(float(np.mean(fu < thf)))
            rows.append({"form": nm, "acceptance_rate": a, "test_auc": auc,
                         "median_shift_sd": shift,
                         "LOSO_rejection": float(np.mean(rejs)),
                         "Test_S1_rejection": s1r, "Test_S2_acceptance": s2a,
                         "three_way_mean": (float(np.mean(rejs)) + s1r + s2a) / 3.0})

    rows.sort(key=lambda r: -r["three_way_mean"])
    best = rows[0]
    (OUT / "results.json").write_text(json.dumps(
        {"experiment": CFG["experiment"], "clean": True, "best": best, "curve": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E138 Scale-free score forms", "",
             "目标：消除 Train 与 Test 之间的分数尺度差，使 Train 校准的阈值能在 Test 上落在预期工作点，"
             "同时保持可分性。三项均值 (LOSO + Test S1 拒识 + Test S2 接受)/3 目标 > 0.85。", "",
             "## 各种分数形态", "",
             "| 形态 | Test AUC | 中位偏移（Train OOF 标准差） | 最好三项均值 | LOSO | S1 拒识 | S2 接受 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    seen = set()
    for r in rows:
        if r["form"] in seen:
            continue
        seen.add(r["form"])
        lines.append(f"| {r['form']} | {r['test_auc']:.4f} | {r['median_shift_sd']:+.3f} | "
                     f"{r['three_way_mean']:.4f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} |")
    lines += ["", "## 全部工作点（按三项均值排序，前 20）", "",
              "| 形态 | 接受率 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in rows[:20]:
        lines.append(f"| {r['form']} | {r['acceptance_rate']:.2f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                     f"{r['three_way_mean']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
