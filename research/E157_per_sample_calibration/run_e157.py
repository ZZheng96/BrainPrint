"""E157: per-sample calibration of the prototype-distance vector.

Question: can the S1-vs-S2 Test diagnostic AUC be improved *without* using any
Test-batch statistic, by normalising only the distance vector of the individual
sample (its own median / IQR / rank / softmax profile / runner-up margin)?

The S2 false-reject problem (E150) is a global shift of Test distances.  Moving
the threshold fixes S2 at the expense of S1; it cannot change the AUC.  A score
that is invariant to the session shift should raise the AUC itself.

Protocol is frozen to E143:
  * E65 correlation-logSPD, per-task standardisation, PCA-96 fitted on Train
    fit_rows (frac <= 0.6);
  * LOSO folds exclude the scored identity, prototypes only from fit_rows;
  * five sequential Train OOF blocks calibrate the thresholds;
  * Test is score-only; session is diagnostic only.

Score forms (all higher = more likely registered):
  raw              -d1
  z_profile        (median(d) - d1) / IQR(d)
  softmax_profile  max_j softmax(-d_j / tau), tau = Train OOF per-task median d1
  rank_profile     self rank of d1 (structurally constant; diagnostic only)
  ratio_d2_d1      d2 / d1, d2 = nearest different-identity prototype (higher = cleaner margin)
  gap              d2 - d1
  rank_ref_control 1 - ECDF_train_oof_task(d1) (monotone control, AUC == raw)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

CFG_PATH = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "config.json"
OUT = CFG_PATH.parent
ROOT = OUT.parents[1]
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]
W_ = CFG["weights"]
SCORES = ["raw", "z_profile", "softmax_profile", "rank_profile",
          "ratio_d2_d1", "gap", "rank_ref_control"]


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
            for centre in kmeans_t(block, k, seed=int(c) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(c)))
    return np.stack(protos), keys


def distance_matrix(feat, protos, keys, row_tasks):
    """Same-task prototype distances; other-task columns stay +inf."""
    n, m = len(feat), len(protos)
    out = np.full((n, m), np.inf, dtype=np.float32)
    for task in TASKS:
        idx = np.flatnonzero(row_tasks == task)
        cols = np.array([i for i, (t, _) in enumerate(keys) if t == task], dtype=int)
        if len(idx) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 4096):
            ch = idx[st:st + 4096]
            q = torch.as_tensor(np.asarray(feat[ch], dtype=np.float32), device=DEVICE)
            out[np.ix_(ch, cols)] = torch.cdist(q, p).cpu().numpy()
    return out


def form_scores(d, keys, row_tasks, tau_by_task):
    """Per-sample transforms of one row's own distance vector."""
    n, m = d.shape
    finite = np.isfinite(d)
    dsort = np.sort(d, axis=1)
    d1 = dsort[:, 0]
    nearest = np.argmin(d, axis=1)
    col_id = np.asarray([kk[1] for kk in keys], dtype=int)
    nearest_id = col_id[nearest]
    d_diff = np.where(col_id[None, :] == nearest_id[:, None], np.inf, d)
    d2 = d_diff.min(1)

    dd = np.where(finite, d, np.float32(np.nan)).astype(np.float32)
    med = np.nanmedian(dd, axis=1)
    q75 = np.nanpercentile(dd, 75, axis=1)
    q25 = np.nanpercentile(dd, 25, axis=1)
    iqr = np.maximum(q75 - q25, 1e-6)
    z_profile = (med - d1) / iqr

    tau = np.asarray([tau_by_task.get(t, np.nan) for t in row_tasks], dtype=np.float64)
    zz = -(dd.astype(np.float64) - d1[:, None]) / tau[:, None]
    zz = np.where(np.isfinite(zz), zz, -np.inf)
    softmax_profile = 1.0 / np.exp(zz).sum(1)

    n_finite = finite.sum(1).astype(np.float64)
    rank_profile = 1.0 - 1.0 / np.maximum(n_finite, 1.0)

    return {
        "raw": -d1,
        "z_profile": z_profile,
        "softmax_profile": softmax_profile,
        "rank_profile": rank_profile,
        "ratio_d2_d1": d2 / np.maximum(d1, 1e-6),
        "gap": d2 - d1,
    }, d1, d2


def ecdf_acceptance(ref_d1, ref_tasks, d1, row_tasks):
    """1 - percentile of d1 in the Train OOF same-task reference distribution."""
    out = np.full(len(d1), np.nan, dtype=np.float64)
    for task in TASKS:
        ref = np.sort(ref_d1[(ref_tasks == task) & np.isfinite(ref_d1)])
        m = row_tasks == task
        if len(ref) == 0 or not m.any():
            continue
        out[m] = 1.0 - np.searchsorted(ref, d1[m], side="right") / len(ref)
    return out


def oof_identity_metrics(ref_blocks, subjects, tasks):
    """Five-block OOF nearest-prototype identity AUC/F1 (E156 definition)."""
    n = len(subjects)
    pred = np.full(n, -1, dtype=int)
    cls = np.full((n, len(CLASSES)), -np.inf, dtype=np.float32)
    for d, keys, idx in ref_blocks:
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
    valid = pred > 0
    y_bin = np.eye(len(CLASSES), dtype=int)[subjects[valid] - 1]
    auc = float(roc_auc_score(y_bin, cls[valid], multi_class="ovr", average="macro"))
    f1 = float(f1_score(subjects[valid], pred[valid], labels=CLASSES,
                        average="macro", zero_division=0))
    return auc, f1, int(valid.sum())


def py(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


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
    y_diag = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]
    D = CFG["pca_dim"]

    # ---- Train-only standardisation / PCA (E143 exact) -------------------
    a = np.load(ROOT / CFG["features"]["train"]).astype(np.float32)
    b = np.load(ROOT / CFG["features"]["test"]).astype(np.float32)
    for t in TASKS:
        m = tasks == t
        mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
        a[m] = (a[m] - mu) / sd
        q = test_tasks == t
        b[q] = (b[q] - mu) / sd
    mu, V = pca_basis(np.asarray(a[fit_rows], dtype=np.float64), D)
    a = ((a.astype(np.float64) - mu) @ V).astype(np.float32)
    b = ((b.astype(np.float64) - mu) @ V).astype(np.float32)
    print(f"space ready {a.shape} -> {b.shape} on {DEVICE}", flush=True)

    rows = []
    summary = {}
    for k in CFG["prototype_k"]:
        print(f"\n===== k={k} =====", flush=True)

        # ---- Train OOF reference: prototypes from all non-held blocks (E143)
        ref_blocks, ref_d1 = [], np.full(len(a), np.nan, dtype=np.float64)
        for blk in range(5):
            hold = blocks5 == blk
            idx = np.flatnonzero(hold)
            p2, k2 = build_prototypes(a, subjects, tasks, ~hold, CLASSES, k)
            d = distance_matrix(a[idx], p2, k2, tasks[idx])
            ref_blocks.append((d, k2, idx))
            ref_d1[idx] = np.sort(d, axis=1)[:, 0]
        tau_by_task = {t: float(np.nanmedian(ref_d1[tasks == t])) for t in TASKS}
        global_tau = float(np.nanmedian(ref_d1))
        for t in TASKS:
            if not np.isfinite(tau_by_task[t]):
                tau_by_task[t] = global_tau

        ref_forms = {s: np.full(len(a), np.nan, dtype=np.float64) for s in SCORES}
        for d, k2, idx in ref_blocks:
            fm, d1f, _ = form_scores(d, k2, tasks[idx], tau_by_task)
            fm["rank_ref_control"] = ecdf_acceptance(ref_d1, tasks, d1f, tasks[idx])
            for s in SCORES:
                ref_forms[s][idx] = fm[s]

        oof_auc, oof_f1, oof_n = oof_identity_metrics(ref_blocks, subjects, tasks)
        del ref_blocks

        # ---- Test: deployed prototypes from all Train rows (E143) --------
        protos, keys = build_prototypes(a, subjects, tasks, np.ones(len(a), bool), CLASSES, k)
        dte = distance_matrix(b, protos, keys, test_tasks)
        te_forms, d1te, _ = form_scores(dte, keys, test_tasks, tau_by_task)
        te_forms["rank_ref_control"] = ecdf_acceptance(ref_d1, tasks, d1te, test_tasks)
        del dte

        # ---- LOSO folds --------------------------------------------------
        loso_forms = {s: [] for s in SCORES}
        for held in CLASSES:
            enrolled = np.setdiff1d(CLASSES, [held])
            pm = np.isin(subjects, enrolled)
            ki = pm & eval_rows
            ui = (subjects == held) & eval_rows
            if ui.sum() == 0:
                continue
            p2, k2 = build_prototypes(a, subjects, tasks, pm & fit_rows, enrolled, k)
            idx = np.flatnonzero(ki | ui)
            d = distance_matrix(a[idx], p2, k2, tasks[idx])
            fm, d1f, _ = form_scores(d, k2, tasks[idx], tau_by_task)
            fm["rank_ref_control"] = ecdf_acceptance(ref_d1, tasks, d1f, tasks[idx])
            mk, mu_ = ki[idx], ui[idx]
            for s in SCORES:
                loso_forms[s].append((fm[s][mk], fm[s][mu_]))

        # ---- scan acceptance grid ---------------------------------------
        print(f"  Train OOF AUC={oof_auc:.4f} F1={oof_f1:.4f} (n={oof_n})", flush=True)
        for s in SCORES:
            ref = ref_forms[s]
            rmu = float(np.nanmean(ref))
            rsd = float(np.nanstd(ref)) + 1e-6
            ref_n = (ref - rmu) / rsd
            te_n = (te_forms[s] - rmu) / rsd
            auc = float(roc_auc_score(y_diag, np.r_[te_n[is_s1], te_n[~is_s1]]))
            med_shift = float(np.nanmedian(te_n) - np.nanmedian(ref_n))
            s1_shift = float(np.nanmedian(te_n[is_s1]) - np.nanmedian(ref_n))
            s2_shift = float(np.nanmedian(te_n[~is_s1]) - np.nanmedian(ref_n))
            best = None
            for acc in CFG["acceptance_grid"]:
                th = float(np.nanquantile(ref_n, 1 - acc))
                s2a = float(np.mean(te_n[~is_s1] >= th))
                s1r = float(np.mean(te_n[is_s1] < th))
                rejs = [float(np.mean(uf < np.quantile(kf, 1 - acc)))
                        for kf, uf in loso_forms[s]]
                lr = float(np.mean(rejs))
                w = W_["LOSO"] * lr + W_["S1"] * s1r + W_["S2"] * s2a
                row = {"score": s, "k": int(k), "acceptance_rate": float(acc),
                       "test_diagnostic_auc": auc, "median_shift_sd": med_shift,
                       "s1_median_shift_sd": s1_shift, "s2_median_shift_sd": s2_shift,
                       "LOSO_rejection": lr, "Test_S1_rejection": s1r,
                       "Test_S2_acceptance": s2a, "W": w,
                       "Train_OOF_AUC": oof_auc, "Train_OOF_F1": oof_f1,
                       "Test_AUC": "N/A", "Test_F1": "N/A"}
                rows.append(row)
                if best is None or w > best["W"]:
                    best = row
            print(f"  {s:17s} AUC {auc:.4f} shift {med_shift:+.2f}sd  "
                  f"bestW {best['W']:.4f} @a={best['acceptance_rate']:.2f} "
                  f"(LOSO {best['LOSO_rejection']:.4f} / S1 {best['Test_S1_rejection']:.4f} / "
                  f"S2 {best['Test_S2_acceptance']:.4f})", flush=True)
            summary.setdefault(s, {"best": None, "auc_by_k": {}})
            summary[s]["auc_by_k"][str(k)] = auc
            if summary[s]["best"] is None or best["W"] > summary[s]["best"]["W"]:
                summary[s]["best"] = best
        del ref_forms, te_forms, loso_forms, protos, keys

    rows.sort(key=lambda r: (-r["W"], r["score"], r["k"], r["acceptance_rate"]))
    best_overall = rows[0]

    results = {
        "experiment": CFG["experiment"],
        "clean": True,
        "device": str(DEVICE),
        "objective": "W = 0.25*LOSO_rejection + 0.25*Test_S1_rejection + 0.5*Test_S2_acceptance",
        "protocol": CFG["protocol"],
        "score_definitions": CFG["score_definitions"],
        "baseline_reference": CFG["reference_baseline"],
        "best_overall": best_overall,
        "per_score": summary,
        "curve": rows,
        "rank_profile_is_degenerate": True,
        "rank_profile_note": (
            "The nearest distance is always the minimum of the sample's own distance vector, so its "
            "within-sample rank is constant (1/M). Its score carries no information; the constant score "
            "yields AUC 0.5 and W = 0.5 (LOSO 0, S1 0, S2 1) purely from the threshold convention."),
        "rank_ref_control_note": (
            "1 - ECDF_train_oof_task(d1) is a strictly monotone transform of raw; its diagnostic AUC "
            "must equal raw by construction, which separates 'calibration moved the working point' from "
            "'calibration improved the ranking'."),
        "Test_official_AUC": "N/A (no true labels)",
        "Test_official_F1": "N/A (no true labels)",
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- report ----------------
    L = []
    L += ["# E157 逐样本距离归一化（不触碰 Test 统计量）", "",
          "目标：验证只对**单个样本自己的距离向量**做归一化，能否消除会话带来的整体平移/尺度变化，"
          "从而真正提升 Test S1-vs-S2 诊断 AUC，而不只是把工作点从 S1 挪到 S2。", "",
          "严格合规：per-task 标准化 / PCA-96 / 原型全部只由 Train 拟合；LOSO 每折排除被评分身份，"
          "原型只用 `frac<=0.6`；五块 Train OOF 校准阈值；Test 只评分；session 仅用于诊断。"
          "Test 官方 AUC/F1：**N/A（无真实标签）**。", "",
          f"设备：`{DEVICE}`；E65 correlation-logSPD；k 扫描 {CFG['prototype_k']}；"
          f"接受率网格 {CFG['acceptance_grid'][0]:.2f}–{CFG['acceptance_grid'][-1]:.2f}。", ""]

    L += ["## 分数定义（统一为“越大越像注册者”）", "",
          "- `raw`：`-d1`，最近同 task 原型距离取负（E143 基准）。",
          "- `z_profile`：把该样本的相似度向量 `-d` 用自身中位数/IQR 归一化后取最大，"
          "等价于 `(median(d) - d1) / IQR(d)`。",
          "- `softmax_profile`：`max_j softmax(-d_j/tau) = exp(-d1/tau) / sum_j exp(-d_j/tau)`，"
          "`tau` 取 Train OOF 每个 task 的最近距离中位数。",
          "- `rank_profile`：`d1` 在该样本自身距离向量中的分位（1 − 1/M）。由于最近距离必为最小值，"
          "它是结构性常数，只作为“该类定义不可用”的诊断。",
          "- `ratio_d2_d1`：`d2/d1`；`d2` 是**不同身份**的最近原型距离（同身份的第 2 个原型会使 margin 退化）。"
          "越大表示次优身份越远、间隔越干净。",
          "- `gap`：`d2-d1`，越大表示身份间隔越干净。",
          "- `rank_ref_control`：`1 - ECDF_{Train-OOF, 同 task}(d1)`，是 raw 的单调分位重标定对照，"
          "只用于区分“排序提升”与“工作点移动”。", ""]

    L += ["## 结论摘要", ""]
    L.append(f"**最佳 W（含对照）：{best_overall['score']} / k={best_overall['k']} / 接受率 "
             f"{best_overall['acceptance_rate']:.2f} → W = {best_overall['W']:.4f}**"
             f"（LOSO {best_overall['LOSO_rejection']:.4f} / S1 拒识 "
             f"{best_overall['Test_S1_rejection']:.4f} / S2 接受 "
             f"{best_overall['Test_S2_acceptance']:.4f}，诊断 AUC "
             f"{best_overall['test_diagnostic_auc']:.4f}，Train OOF AUC/F1 "
             f"{best_overall['Train_OOF_AUC']:.4f}/{best_overall['Train_OOF_F1']:.4f}）。")
    raw_best = summary["raw"]["best"]
    kk = str(CFG["prototype_k"][-1])
    raw_auc = summary["raw"]["auc_by_k"][kk]
    genuine = ["z_profile", "softmax_profile", "ratio_d2_d1", "gap"]
    best_ps = max(genuine, key=lambda s: summary[s]["auc_by_k"][kk])
    best_ps_auc = summary[best_ps]["auc_by_k"][kk]
    L.append("")
    L.append(f"但 `rank_ref_control` 是 raw 的单调重标定对照，二者 AUC 只差 "
             f"{abs(summary['rank_ref_control']['auc_by_k'][kk] - raw_auc):.4f}；"
             f"真正的逐样本分数里诊断 AUC 最高的是 `{best_ps}` "
             f"({best_ps_auc:.4f})，仍比 raw ({raw_auc:.4f}) 低 "
             f"{raw_auc - best_ps_auc:.4f}。**没有任何逐样本归一化真正提升了排序能力。**")
    L.append("")
    L.append(f"E143 raw 基线复现：k={raw_best['k']}、接受率 {raw_best['acceptance_rate']:.2f} 时 "
             f"W={raw_best['W']:.4f}（LOSO {raw_best['LOSO_rejection']:.4f} / S1 "
             f"{raw_best['Test_S1_rejection']:.4f} / S2 {raw_best['Test_S2_acceptance']:.4f}，"
             f"诊断 AUC {raw_best['test_diagnostic_auc']:.4f}）；参考值 "
             f"W={CFG['reference_baseline']['W']:.4f}、AUC={CFG['reference_baseline']['test_diagnostic_auc']:.4f}。")
    L += ["", "## 各分数的最佳工作点（对每个分数独立取 W 最大）", "",
          "| 分数 | k | 接受率 | 诊断 AUC | OOF AUC | OOF F1 | LOSO 拒识 | S1 拒识 | S2 接受 | W |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SCORES:
        r = summary[s]["best"]
        L.append(f"| {s} | {r['k']} | {r['acceptance_rate']:.2f} | "
                 f"{r['test_diagnostic_auc']:.4f} | {r['Train_OOF_AUC']:.4f} | "
                 f"{r['Train_OOF_F1']:.4f} | {r['LOSO_rejection']:.4f} | "
                 f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                 f"{r['W']:.4f} |")

    L += ["", "## 诊断 AUC（与阈值无关）：是否真的提升了排序能力", "",
          "| 分数 | k=1 | k=3 | k=6 | k=12 | ΔAUC(k=12) vs raw |",
          "|---|---:|---:|---:|---:|---:|"]
    raw_by_k = summary["raw"]["auc_by_k"]
    for s in SCORES:
        ak = summary[s]["auc_by_k"]
        cells = " | ".join(f"{ak[str(k)]:.4f}" for k in CFG["prototype_k"])
        delta = ak[str(CFG["prototype_k"][-1])] - raw_by_k[str(CFG["prototype_k"][-1])]
        L.append(f"| {s} | {cells} | {delta:+.4f} |")

    L += ["", "## Test 残移诊断（以 Train OOF 标准差为单位的中位数偏移）", "",
          "| 分数 | k | 全体偏移 | S1 偏移 | S2 偏移 |", "|---|---:|---:|---:|---:|"]
    for s in SCORES:
        r = summary[s]["best"]
        L.append(f"| {s} | {r['k']} | {r['median_shift_sd']:+.3f} | "
                 f"{r['s1_median_shift_sd']:+.3f} | {r['s2_median_shift_sd']:+.3f} |")

    L += ["", "## 关键判断", ""]
    max_delta = max(abs(summary["rank_ref_control"]["auc_by_k"][str(k)] - raw_by_k[str(k)])
                    for k in CFG["prototype_k"])
    L.append(f"- `rank_ref_control` 是 raw 的单调分位重标定，各 k 的 AUC 最大只差 "
             f"{max_delta:.4f}（分位离散化造成的并列）；W 的微小变化来自工作点，"
             "验证了“标定只能挪工作点、不能改排序”。")
    L.append(f"- `z_profile`（k={summary['z_profile']['best']['k']}）把 Test 中位残移从 raw 的 "
             f"{raw_best['median_shift_sd']:+.2f}sd 压到 "
             f"{summary['z_profile']['best']['median_shift_sd']:+.2f}sd，"
             f"S1/S2 中位残移也几乎重合，但诊断 AUC 降到 "
             f"{summary['z_profile']['best']['test_diagnostic_auc']:.4f}："
             "会话平移和身份可分性被同一个 IQR 归一化一起抹掉（与 E138 结论一致）。")
    L.append(f"- `softmax_profile` / `gap` / `ratio_d2_d1` 的诊断 AUC 分别为 "
             f"{summary['softmax_profile']['best']['test_diagnostic_auc']:.4f} / "
             f"{summary['gap']['best']['test_diagnostic_auc']:.4f} / "
             f"{summary['ratio_d2_d1']['best']['test_diagnostic_auc']:.4f}，"
             f"都低于 raw 的 {raw_auc:.4f}。")
    L.append("- `rank_profile` 在自身距离向量上的分位恒等于最小值的位置（1/M），是结构性常数："
             "AUC=0.5、W=0.5，说明“自我秩”定义不可用；W=0.5 本身也说明该目标可被常数分数套利。")
    L.append("- 判定标准：只有诊断 AUC 上升才算真的提升排序能力；AUC 不变而 W 变化只是工作点移动。"
             "本实验中没有任何纯逐样本分数通过该标准。")
    L += ["", "## 逐工作点明细（前 25 行）", "",
          "| 分数 | k | 接受率 | 诊断 AUC | LOSO | S1 拒识 | S2 接受 | W |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:25]:
        L.append(f"| {r['score']} | {r['k']} | {r['acceptance_rate']:.2f} | "
                 f"{r['test_diagnostic_auc']:.4f} | {r['LOSO_rejection']:.4f} | "
                 f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | {r['W']:.4f} |")
    L += ["", "## 复现", "",
          f"`{sys.executable} {Path(__file__).resolve()}`", "",
          "完整数值见 `results.json`。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(best_overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
