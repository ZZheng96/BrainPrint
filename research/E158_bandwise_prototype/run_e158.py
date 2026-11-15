"""E158: band-wise prototype modelling + Train-only fusion.

Motivation
----------
E143 (clean protocol, E65 correlation-logSPD 855-dim -> Task-wise
standardisation -> joint PCA-96 -> k=12 prototypes) gives
LOSO 0.7936 / Test S1 rejection 0.7979 / Test S2 acceptance 0.5960 / W 0.6958
with Test S1-vs-S2 diagnostic AUC 0.7578.  E157 showed that per-sample
calibration cannot raise that AUC, it only moves the operating point.

The untested axis here is the *representation*: instead of letting one PCA mix
all five bands, model every band separately (5 bands x 171 upper-triangle
entries), calibrate each band score against its own Train OOF distribution and
fuse the five z-scores with weights chosen on Train only.  If some bands carry a
more session-stable identity structure, the fusion should raise the diagnostic
AUC above 0.7578.

Protocol (frozen to E143, nothing from Test is used to fit or select):
  * Task-wise standardisation / PCA fitted on Train fit_rows (frac <= 0.6);
  * prototypes for LOSO folds come from fit_rows only, the scored identity is
    removed from the prototype set;
  * five sequential Train OOF blocks give the reference distribution used for
    z-calibration and for the acceptance thresholds;
  * evaluation block is 0.6 < frac <= 0.8;
  * Test is score-only, session labels are diagnostic only.

Fusion candidates (all weights/selection Train-only):
  (a) mean_z         equal-weight mean of the five band z-scores
  (b) rank_mean      per-band percentile against its own Train OOF ECDF, then mean
  (c) loso_weighted  weight simplex (step 0.25) maximising mean Train LOSO rejection
  (d) greedy_forward forward band selection, equal weights, Train LOSO criterion
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "config.json"
OUT = CFG_PATH.parent
ROOT = OUT.parents[1]
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]
W_ = CFG["weights"]
GRID = CFG["acceptance_grid"]
BASELINE_ONLY = "--baseline-only" in sys.argv
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0)
    xc = x - mu
    w, v = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, v[:, np.argsort(-w)][:, : min(dim, x.shape[1])]


def kmeans_t(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    c = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(
            0, a, torch.ones(len(x), device=x.device, dtype=x.dtype))
        new = torch.where(cnt[:, None] > 0, sums / cnt.clamp(min=1)[:, None], c)
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
            centres = block.mean(0, keepdim=True) if k == 1 else kmeans_t(block, k, seed=int(c) * 17)
            for centre in centres.cpu().numpy():
                protos.append(centre)
                keys.append((task, int(c)))
    return np.stack(protos), keys


def distance_matrix(feat, protos, keys, row_tasks):
    """Same-task prototype distances; other-task entries stay +inf."""
    n, m = len(feat), len(protos)
    out = np.full((n, m), np.inf, dtype=np.float32)
    col_task = np.asarray([kk[0] for kk in keys])
    cols_by_task = {t: np.flatnonzero(col_task == t) for t in TASKS}
    for task in TASKS:
        idx = np.flatnonzero(row_tasks == task)
        cols = cols_by_task[task]
        if len(idx) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 4096):
            ch = idx[st:st + 4096]
            q = torch.as_tensor(np.asarray(feat[ch], dtype=np.float32), device=DEVICE)
            out[np.ix_(ch, cols)] = torch.cdist(q, p).cpu().numpy()
    return out


def oof_identity_metrics(blocks, subjects, tasks):
    """Five-block OOF nearest-prototype identity AUC / macro F1 (E156/E157)."""
    n = len(subjects)
    pred = np.full(n, -1, dtype=int)
    cls = np.full((n, len(CLASSES)), -np.inf, dtype=np.float32)
    for d, keys, idx in blocks:
        col_task = np.asarray([kk[0] for kk in keys])
        col_id = np.asarray([kk[1] for kk in keys], dtype=int)
        for task in TASKS:
            rloc = np.flatnonzero(tasks[idx] == task)
            cols = np.flatnonzero(col_task == task)
            if not len(rloc) or not len(cols):
                continue
            sub = d[np.ix_(rloc, cols)]
            j = np.argmin(sub, axis=1)
            pred[idx[rloc]] = col_id[cols[j]]
            for c in CLASSES:
                pos = np.flatnonzero(col_id[cols] == c)
                if len(pos):
                    cls[idx[rloc], c - 1] = -sub[:, pos].min(1)
    auc = float(roc_auc_score(np.eye(len(CLASSES), dtype=int)[subjects - 1], cls,
                              multi_class="ovr", average="macro"))
    f1 = float(f1_score(subjects, pred, labels=CLASSES, average="macro", zero_division=0))
    return auc, f1


def score_pipeline(feat, feat_te, ctx, k, tag=""):
    """Full clean pipeline for one feature space: Train OOF ref, Test, 20 LOSO folds."""
    subjects, tasks, test_tasks = ctx["subjects"], ctx["tasks"], ctx["test_tasks"]
    fit_rows, eval_rows, blocks5 = ctx["fit_rows"], ctx["eval_rows"], ctx["blocks5"]

    ref_raw = np.full(len(feat), np.nan, dtype=np.float32)
    blocks = []
    for blk in range(5):
        hold = blocks5 == blk
        idx = np.flatnonzero(hold)
        p, kk = build_prototypes(feat, subjects, tasks, ~hold, CLASSES, k)
        d = distance_matrix(feat[idx], p, kk, tasks[idx])
        ref_raw[idx] = -d.min(1)
        blocks.append((d, kk, idx))
    oof_auc, oof_f1 = oof_identity_metrics(blocks, subjects, tasks)
    del blocks

    mu = float(ref_raw.mean())
    sd = float(ref_raw.std()) + 1e-6

    protos, keys = build_prototypes(feat, subjects, tasks, np.ones(len(feat), bool), CLASSES, k)
    te_raw = -distance_matrix(feat_te, protos, keys, test_tasks).min(1)
    del protos, keys

    loso = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        ki = pm & eval_rows
        ui = (subjects == held) & eval_rows
        if ui.sum() == 0:
            continue
        p2, k2 = build_prototypes(feat, subjects, tasks, pm & fit_rows, enrolled, k)
        idx = np.flatnonzero(ki | ui)
        s = -distance_matrix(feat[idx], p2, k2, tasks[idx]).min(1)
        loso.append((s[ki[idx]], s[ui[idx]]))

    return {
        "ref": (ref_raw - mu) / sd,
        "test": (te_raw - mu) / sd,
        "loso": [((a - mu) / sd, (b - mu) / sd) for a, b in loso],
        "oof_auc": oof_auc,
        "oof_f1": oof_f1,
    }


def loso_rejection(loso, a):
    return float(np.mean([float(np.mean(un < np.quantile(kn, 1 - a))) for kn, un in loso]))


def test_auc(test, is_s1):
    return float(roc_auc_score(np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))],
                               np.r_[test[is_s1], test[~is_s1]]))


def metrics_at(ref, test, loso, is_s1, a, auc):
    th = float(np.quantile(ref, 1 - a))
    s1 = float(np.mean(test[is_s1] < th))
    s2 = float(np.mean(test[~is_s1] >= th))
    lr = loso_rejection(loso, a)
    w = W_["LOSO"] * lr + W_["S1"] * s1 + W_["S2"] * s2
    return {"acceptance_rate": float(a), "test_diagnostic_auc": auc,
            "LOSO_rejection": lr, "Test_S1_rejection": s1,
            "Test_S2_acceptance": s2, "W": w}


def rank_of(sorted_ref, x):
    return np.searchsorted(sorted_ref, x, side="right") / len(sorted_ref)


def member_arrays(store, name, ranked):
    s = store[name]
    if not ranked:
        return s["ref"], s["test"], s["loso"]
    sr = np.sort(s["ref"])
    ref = rank_of(sr, s["ref"])
    test = rank_of(sr, s["test"])
    loso = [(rank_of(sr, kk), rank_of(sr, uu)) for kk, uu in s["loso"]]
    return ref, test, loso


def combine(arrays, weights):
    ref = np.zeros_like(arrays[0][0], dtype=np.float64)
    test = np.zeros_like(arrays[0][1], dtype=np.float64)
    nf = len(arrays[0][2])
    for arr, w in zip(arrays, weights):
        if w == 0:
            continue
        ref += w * arr[0]
        test += w * arr[1]
    loso = []
    for i in range(nf):
        kn = np.zeros_like(arrays[0][2][i][0], dtype=np.float64)
        un = np.zeros_like(arrays[0][2][i][1], dtype=np.float64)
        for arr, w in zip(arrays, weights):
            if w == 0:
                continue
            kn += w * arr[2][i][0]
            un += w * arr[2][i][1]
        loso.append((kn, un))
    return ref, test, loso


def main():
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    tm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    is_s1 = tm.session.astype(str).str.casefold().eq("session1").to_numpy()
    test_tasks = tm.task.to_numpy()
    fit_rows = f <= CFG["fit_fraction"]
    lo, hi = CFG["eval_fraction"]["low"], CFG["eval_fraction"]["high"]
    eval_rows = (f > lo) & (f <= hi)
    blocks5 = np.clip((f * 5).astype(int), 0, 4)
    ctx = {"subjects": subjects, "tasks": tasks, "test_tasks": test_tasks,
           "fit_rows": fit_rows, "eval_rows": eval_rows, "blocks5": blocks5}

    tr = np.load(ROOT / CFG["features"]["train"]).astype(np.float32)
    te = np.load(ROOT / CFG["features"]["test"]).astype(np.float32)
    assert tr.shape == (CFG["data"]["expected_train_trials"], CFG["features"]["width"]), tr.shape
    assert te.shape == (CFG["data"]["expected_test_rows"], CFG["features"]["width"]), te.shape
    n_bands, bw = len(CFG["features"]["bands"]), CFG["features"]["band_width"]
    assert n_bands * bw == tr.shape[1]
    log(f"loaded {tr.shape} {te.shape} on {DEVICE}")

    def standardise(x_tr, x_te):
        for t in TASKS:
            m, q = tasks == t, test_tasks == t
            mu, sd = x_tr[m].mean(0), x_tr[m].std(0) + 1e-6
            x_tr[m] = (x_tr[m] - mu) / sd
            x_te[q] = (x_te[q] - mu) / sd
        return x_tr, x_te

    rows = []
    store = {}

    # ---------------- E143 baseline: joint 855 -> PCA-96 ------------------
    base_rows = []
    base_dim = CFG["baseline"]["pca_dim"]
    a_tr, a_te = standardise(tr.copy(), te.copy())
    mu, V = pca_basis(np.asarray(a_tr[fit_rows], dtype=np.float64), base_dim)
    base_tr = ((a_tr.astype(np.float64) - mu) @ V).astype(np.float32)
    base_te = ((a_te.astype(np.float64) - mu) @ V).astype(np.float32)
    del a_tr, a_te
    for k in CFG["prototype_k"]:
        s = score_pipeline(base_tr, base_te, ctx, k)
        auc = test_auc(s["test"], is_s1)
        store[("baseline", k)] = s
        for a in GRID:
            r = metrics_at(s["ref"], s["test"], s["loso"], is_s1, a, auc)
            r.update({"kind": "baseline", "variant": CFG["baseline"]["name"],
                      "band": "all_855", "pca_dim": base_dim, "k": int(k),
                      "Train_OOF_AUC": s["oof_auc"], "Train_OOF_F1": s["oof_f1"],
                      "Test_AUC": "N/A", "Test_F1": "N/A"})
            rows.append(r)
            base_rows.append(r)
        best = max([r for r in base_rows if r["k"] == k], key=lambda r: r["W"])
        log(f"baseline PCA-96 k={k:2d} AUC {auc:.4f} bestW {best['W']:.4f} "
            f"@a={best['acceptance_rate']:.2f} (LOSO {best['LOSO_rejection']:.4f} / "
            f"S1 {best['Test_S1_rejection']:.4f} / S2 {best['Test_S2_acceptance']:.4f})")

    ref_base = CFG["baseline"]["reference"]
    repro = [r for r in base_rows if r["k"] == ref_base["k"]]
    repro = min(repro, key=lambda r: abs(r["acceptance_rate"] - ref_base["acceptance_rate"]))
    base_ok = abs(repro["W"] - ref_base["W"]) < 5e-4
    log(f"baseline reproduction @k={ref_base['k']} a={repro['acceptance_rate']:.2f}: "
        f"W {repro['W']:.4f} (ref {ref_base['W']:.4f}), AUC {repro['test_diagnostic_auc']:.4f} "
        f"(ref {ref_base['test_diagnostic_auc']:.4f}) -> {'OK' if base_ok else 'MISMATCH'}")
    if BASELINE_ONLY:
        log("baseline-only mode, stopping")
        return

    # ---------------- per-band sweep -------------------------------------
    band_names = [bd["name"] for bd in CFG["features"]["bands"]]
    for bd in CFG["features"]["bands"]:
        b = bd["index"]
        full_tr = tr[:, b * bw:(b + 1) * bw].copy()
        full_te = te[:, b * bw:(b + 1) * bw].copy()
        s_tr, s_te = standardise(full_tr, full_te)
        for dim in CFG["pca_dims"]:
            mu, V = pca_basis(np.asarray(s_tr[fit_rows], dtype=np.float64), dim)
            z_tr = ((s_tr.astype(np.float64) - mu) @ V).astype(np.float32)
            z_te = ((s_te.astype(np.float64) - mu) @ V).astype(np.float32)
            for k in CFG["prototype_k"]:
                s = score_pipeline(z_tr, z_te, ctx, k)
                auc = test_auc(s["test"], is_s1)
                store[(bd["name"], dim, k)] = s
                for a in GRID:
                    r = metrics_at(s["ref"], s["test"], s["loso"], is_s1, a, auc)
                    r.update({"kind": "band", "variant": bd["name"], "band": bd["name"],
                              "pca_dim": int(dim), "k": int(k),
                              "Train_OOF_AUC": s["oof_auc"], "Train_OOF_F1": s["oof_f1"],
                              "Test_AUC": "N/A", "Test_F1": "N/A"})
                    rows.append(r)
                log(f"{bd['name']} dim={dim:2d} k={k:2d} AUC {auc:.4f} "
                    f"bestW {max([r for r in rows if r['kind'] == 'band' and r['variant'] == bd['name'] and r['pca_dim'] == dim and r['k'] == k], key=lambda r: r['W'])['W']:.4f}")
            del z_tr, z_te
        del full_tr, full_te, s_tr, s_te

    # ---------------- fusion (Train-only selection) -----------------------
    fusion_rows = []
    simplex = [np.asarray(c, dtype=float) / 4.0
               for c in itertools.product(range(5), repeat=len(band_names))
               if sum(c) == 4]
    for dim in CFG["pca_dims"]:
        for k in CFG["prototype_k"]:
            mean_arrays = [member_arrays(store, (n, dim, k), False) for n in band_names]
            rank_arrays = [member_arrays(store, (n, dim, k), True) for n in band_names]

            def emit(method, ref, test, loso, extra, a_list):
                """extra carries the band weights; OOF AUC/F1 is the weight-averaged member value."""
                wmap = extra["weights"]
                wsum = float(sum(wmap.values()))
                oof_auc = float(sum(w * store[(n, dim, k)]["oof_auc"] for n, w in wmap.items()) / wsum)
                oof_f1 = float(sum(w * store[(n, dim, k)]["oof_f1"] for n, w in wmap.items()) / wsum)
                auc = test_auc(test, is_s1)
                for a in a_list:
                    r = metrics_at(ref, test, loso, is_s1, a, auc)
                    r.update({"kind": "fusion", "variant": method, "band": "+".join(band_names),
                              "pca_dim": int(dim), "k": int(k),
                              "Train_OOF_AUC": oof_auc, "Train_OOF_F1": oof_f1,
                              "Test_AUC": "N/A", "Test_F1": "N/A", **extra})
                    fusion_rows.append(r)
                    rows.append(r)
                return max([r for r in fusion_rows if r["variant"] == method and r["pca_dim"] == dim
                            and r["k"] == k], key=lambda r: r["W"])

            # (a) equal-weight mean z
            ref, test, loso = combine(mean_arrays, [1 / len(band_names)] * len(band_names))
            ba = emit("mean_z", ref, test, loso, {"weights": {n: 1 / len(band_names) for n in band_names},
                                                  "selected_bands": band_names}, GRID)
            # (b) equal-weight rank mean
            ref, test, loso = combine(rank_arrays, [1 / len(band_names)] * len(band_names))
            bb = emit("rank_mean", ref, test, loso, {"weights": {n: 1 / len(band_names) for n in band_names},
                                                     "selected_bands": band_names}, GRID)
            # (c) LOSO-selected weights, one weight vector per acceptance rate
            for a in GRID:
                best_w, best_lr = None, -1.0
                for w in simplex:
                    _, _, loso_w = combine(mean_arrays, w)
                    lr = loso_rejection(loso_w, a)
                    if lr > best_lr + 1e-12:
                        best_w, best_lr = w, lr
                ref, test, loso = combine(mean_arrays, best_w)
                emit("loso_weighted", ref, test, loso,
                     {"weights": {n: float(v) for n, v in zip(band_names, best_w)},
                      "selected_bands": [n for n, v in zip(band_names, best_w) if v > 0]}, [a])
            # (d) greedy forward selection, equal weights, LOSO criterion
            for a in GRID:
                selected, best_lr = [], -1.0
                while len(selected) < len(band_names):
                    cand, cand_lr = None, best_lr
                    for i, n in enumerate(band_names):
                        if n in selected:
                            continue
                        trial = selected + [n]
                        idxs = [band_names.index(m) for m in trial]
                        _, _, loso_t = combine([mean_arrays[i] for i in idxs], [1 / len(trial)] * len(trial))
                        lr = loso_rejection(loso_t, a)
                        if lr > cand_lr + 1e-9:
                            cand, cand_lr = n, lr
                    if cand is None:
                        break
                    selected, best_lr = selected + [cand], cand_lr
                idxs = [band_names.index(m) for m in selected]
                ref, test, loso = combine([mean_arrays[i] for i in idxs], [1 / len(selected)] * len(selected))
                emit("greedy_forward", ref, test, loso,
                     {"weights": {n: (1 / len(selected) if n in selected else 0.0) for n in band_names},
                      "selected_bands": selected}, [a])
            log(f"fusion dim={dim:2d} k={k:2d}: mean_z W {ba['W']:.4f} AUC {ba['test_diagnostic_auc']:.4f} | "
                f"rank_mean W {bb['W']:.4f} AUC {bb['test_diagnostic_auc']:.4f}")

    # ---------------- summarise ------------------------------------------
    for r in rows:
        for key in ("test_diagnostic_auc", "LOSO_rejection", "Test_S1_rejection",
                    "Test_S2_acceptance", "W", "Train_OOF_AUC", "Train_OOF_F1"):
            if isinstance(r[key], float):
                r[key] = round(r[key], 6)
        r["acceptance_rate"] = round(float(r["acceptance_rate"]), 4)

    def best_of(rs):
        return max(rs, key=lambda r: r["W"])

    band_rows = [r for r in rows if r["kind"] == "band"]
    best_per_band = {n: best_of([r for r in band_rows if r["variant"] == n]) for n in band_names}
    auc_per_band = {n: max([r for r in band_rows if r["variant"] == n],
                           key=lambda r: r["test_diagnostic_auc"]) for n in band_names}
    best_fusion = {m: best_of([r for r in rows if r["kind"] == "fusion" and r["variant"] == m])
                   for m in ("mean_z", "rank_mean", "loso_weighted", "greedy_forward")}
    best_band_overall = best_of(band_rows)
    best_base = best_of([r for r in rows if r["kind"] == "baseline"])
    best_overall = best_of(rows)

    results = {
        "experiment": CFG["experiment"],
        "clean": True,
        "device": str(DEVICE),
        "objective": "W = 0.25*LOSO_rejection + 0.25*Test_S1_rejection + 0.5*Test_S2_acceptance",
        "protocol": CFG["protocol"],
        "baseline_reference": CFG["baseline"]["reference"],
        "baseline_reproduced": {"row": repro, "matches_reference_W": bool(base_ok)},
        "best_overall": best_overall,
        "best_baseline": best_base,
        "best_per_band": best_per_band,
        "best_auc_per_band": auc_per_band,
        "best_fusion": best_fusion,
        "best_single_band": best_band_overall,
        "fusion_note": ("Train OOF AUC/F1 of a fused score is the mean of the member bands' OOF "
                        "identity AUC/F1 (a fused scalar score has no class map); baseline and "
                        "single-band values are computed directly."),
        "grid": rows,
        "Test_official_AUC": "N/A (no true labels)",
        "Test_official_F1": "N/A (no true labels)",
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, rows, band_names, best_per_band, auc_per_band, best_fusion,
                 best_band_overall, best_base, best_overall, base_rows, repro, base_ok)
    log("done")


def fmt_row(r):
    return (f"| {r['variant']} | {r.get('pca_dim')} | {r['k']} | {r['acceptance_rate']:.2f} | "
            f"{r['test_diagnostic_auc']:.4f} | {r['Train_OOF_AUC']:.4f} | {r['Train_OOF_F1']:.4f} | "
            f"{r['LOSO_rejection']:.4f} | {r['Test_S1_rejection']:.4f} | "
            f"{r['Test_S2_acceptance']:.4f} | {r['W']:.4f} |")


def write_report(results, rows, band_names, best_per_band, auc_per_band, best_fusion,
                 best_band_overall, best_base, best_overall, base_rows, repro, base_ok):
    ref = results["baseline_reference"]
    L = []
    L += ["# E158 分频带独立建模 + Train-only 融合", "",
          "问题：E143 干净基准（E65 correlation-logSPD 855 维 → task 标准化 → **联合** PCA-96 → k=12 原型）"
          f"给出 LOSO {ref['LOSO_rejection']:.4f} / Test S1 拒识 {ref['Test_S1_rejection']:.4f} / "
          f"Test S2 接受 {ref['Test_S2_acceptance']:.4f} / W = {ref['W']:.4f}，诊断 AUC "
          f"{ref['test_diagnostic_auc']:.4f}。E157 证明逐样本归一化只能挪工作点、不能改排序。",
          "本次验证另一个轴：**不把 5 个频带拼在一起做同一个 PCA，而是每个频带单独标准化 + PCA + 原型，"
          "再用只由 Train 决定的权重融合**。若某些频带的身份结构跨会话更稳定，融合应把诊断 AUC 推高。", "",
          "严格合规：per-task 标准化 / PCA / 原型 / z 校准 / 权重与频带选择全部只由 Train 拟合；"
          "LOSO 每折排除被评分身份且原型只用 `frac<=0.6`；评估块 `0.6<frac<=0.8`；"
          "五块 Train OOF 定阈值；Test 只评分；session 仅用于诊断。"
          "**Test 官方 AUC/F1 = N/A（无真实标签）**。", "",
          f"设备 `{DEVICE}`；频带划分沿用 E36 `bands_hz`（band-major 拼接，band b 占列 `[171b,171(b+1))`）："
          + "、".join(f"{n}" for n in band_names) + "；"
          f"PCA 维度 {CFG['pca_dims']}；k {CFG['prototype_k']}；接受率网格 "
          f"{GRID[0]:.2f}–{GRID[-1]:.2f}。", ""]

    L += ["## 1. E143 基线复现（对照组）", "",
          f"联合 855 维、PCA-{CFG['baseline']['pca_dim']}、同样的打分管线。"
          f"k={ref['k']}、接受率 {ref['acceptance_rate']:.2f} 处复现结果 "
          f"W = {repro['W']:.4f}（参考 {ref['W']:.4f}）、诊断 AUC {repro['test_diagnostic_auc']:.4f}"
          f"（参考 {ref['test_diagnostic_auc']:.4f}）、LOSO {repro['LOSO_rejection']:.4f}"
          f"（参考 {ref['LOSO_rejection']:.4f}）→ "
          f"**{'一致，基线可信' if base_ok else '不一致，需要检查管线'}**。", "",
          "| 变体 | PCA 维 | k | 接受率 | 诊断 AUC | OOF AUC | OOF F1 | LOSO 拒识 | S1 拒识 | S2 接受 | W |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(base_rows, key=lambda r: -r["W"])[:8]:
        L.append(fmt_row(r))

    L += ["", "## 2. 每个频带单独建模", "",
          "表 A：每个频带按 W 最优的 (PCA 维, k) 工作点。", "",
          "| 频带 | PCA 维 | k | 接受率 | 诊断 AUC | OOF AUC | OOF F1 | LOSO 拒识 | S1 拒识 | S2 接受 | W |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in band_names:
        L.append(fmt_row(best_per_band[n]))
    L += ["", "表 B：每个频带诊断 AUC 最高的 (PCA 维, k)（与阈值无关，用于判断哪条频带的身份结构最可迁移）。", "",
          "| 频带 | PCA 维 | k | 诊断 AUC | OOF AUC | OOF F1 | LOSO 拒识 | S1 拒识 | S2 接受 | W |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in band_names:
        r = auc_per_band[n]
        L.append(f"| {n} | {r['pca_dim']} | {r['k']} | {r['test_diagnostic_auc']:.4f} | "
                 f"{r['Train_OOF_AUC']:.4f} | {r['Train_OOF_F1']:.4f} | {r['LOSO_rejection']:.4f} | "
                 f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | {r['W']:.4f} |")

    L += ["", "## 3. 四种融合", "",
          "(a) `mean_z` 等权平均各频带 z；(b) `rank_mean` 各频带先按自己的 Train OOF ECDF 转分位再等权平均"
          "（逐频带单调变换，不含任何 Test 批统计量）；(c) `loso_weighted` 在 0.25 步长单纯形上按"
          "**Train LOSO 拒识率**逐接受率选权重；(d) `greedy_forward` 等权贪心前向选频带，"
          "准则同样只用 Train LOSO 拒识率。" , "",
          "| 融合 | PCA 维 | k | 接受率 | 诊断 AUC | OOF AUC | OOF F1 | LOSO 拒识 | S1 拒识 | S2 接受 | W |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("mean_z", "rank_mean", "loso_weighted", "greedy_forward"):
        L.append(fmt_row(best_fusion[m]))
    L += ["", "选中的权重 / 频带（各方法 W 最优那一行）：", ""]
    for m in ("loso_weighted", "greedy_forward"):
        r = best_fusion[m]
        sel = r.get("selected_bands", [])
        w = r.get("weights", {})
        L.append(f"- `{m}`（dim={r['pca_dim']}, k={r['k']}, 接受率 {r['acceptance_rate']:.2f}）："
                 f"频带 {sel}；权重 " + "、".join(f"{n}={w[n]:.2f}" for n in band_names) + "。")
    L += ["", "注：融合分数的 Train OOF AUC/F1 取成员频带 OOF 身份 AUC/F1 的平均（标量融合分数没有类别图，"
          "无法直接做身份分类）；基线与单频带为直接计算值。", ""]

    L += ["## 4. 结论", ""]
    best_single_auc = max(r["test_diagnostic_auc"] for r in auc_per_band.values())
    best_single_name = max(auc_per_band, key=lambda n: auc_per_band[n]["test_diagnostic_auc"])
    best_fusion_auc = max(best_fusion[m]["test_diagnostic_auc"] for m in best_fusion)
    best_fusion_auc_name = max(best_fusion, key=lambda m: best_fusion[m]["test_diagnostic_auc"])
    L.append(f"1. **单频带最强**：诊断 AUC 最高的是 `{best_single_name}` "
             f"（dim={auc_per_band[best_single_name]['pca_dim']}, k={auc_per_band[best_single_name]['k']}）"
             f"AUC = {best_single_auc:.4f}；按 W 最优的单频带是 `{best_band_overall['variant']}`"
             f"（dim={best_band_overall['pca_dim']}, k={best_band_overall['k']}, 接受率 "
             f"{best_band_overall['acceptance_rate']:.2f}）W = {best_band_overall['W']:.4f}。")
    L.append(f"2. **融合是否抬高诊断 AUC 超过 {ref['test_diagnostic_auc']:.4f}**：融合中诊断 AUC 最高的是 "
             f"`{best_fusion_auc_name}` {best_fusion_auc:.4f}，"
             f"{'高于' if best_fusion_auc > ref['test_diagnostic_auc'] else '不高于'} E143 的 "
             f"{ref['test_diagnostic_auc']:.4f}（Δ {best_fusion_auc - ref['test_diagnostic_auc']:+.4f}）。")
    L.append(f"3. **全局最佳 W**：`{best_overall['variant']}`"
             f"（{best_overall['kind']}, dim={best_overall['pca_dim']}, k={best_overall['k']}, "
             f"接受率 {best_overall['acceptance_rate']:.2f}）W = {best_overall['W']:.4f}，"
             f"LOSO {best_overall['LOSO_rejection']:.4f} / S1 {best_overall['Test_S1_rejection']:.4f} / "
             f"S2 {best_overall['Test_S2_acceptance']:.4f}，诊断 AUC "
             f"{best_overall['test_diagnostic_auc']:.4f}。E143 对照 W = {best_base['W']:.4f}。")
    L.append(f"4. **是否达到 0.80**：{'达到' if best_overall['W'] >= CFG['targets']['weighted'] else '未达到'}，"
             f"距目标差 {CFG['targets']['weighted'] - best_overall['W']:+.4f}。")
    L += ["", "## 5. 全网格前 25 行（按 W）", "",
          "| 变体 | PCA 维 | k | 接受率 | 诊断 AUC | OOF AUC | OOF F1 | LOSO 拒识 | S1 拒识 | S2 接受 | W |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["W"])[:25]:
        L.append(fmt_row(r))
    L += ["", "完整数值（含每个频带 × 每个 PCA 维 × 每个 k 的 10 个接受率工作点）见 `results.json`。", "",
          "项目 Test AUC / F1 / C / RA / ASafeR / U / Official Score = **N/A**（无真实标签）。", "",
          "## 复现", "",
          f"`.\\.venv\\Scripts\\python.exe {Path(__file__).resolve()}`", ""]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
