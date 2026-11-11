"""目标批次 AS-norm + kNN 平滑 + 自适应阈值的纯 EEG 开集决策实验。

决策层只接收分数矩阵、EEG 特征和 task；Session 仅在冻结预测后的 proxy_eval 中读取。
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from src.open_set_calibration.core import build_submission, predict_calibration_scores
from src.open_set_improvement.core import COV_COLUMNS, ROOT, TaskSpecificModel, resolve_cache

OUT = ROOT / "research" / "E05_asnorm_openset"
ALIGNS = ("none", "target_zscore")
NORMS = ("raw", "asnorm")
SMOOTHS = (0, 10)
CONFS = ("max", "margin")
THRS = ("src_q", "fixed", "gmm")
GAMMAS = (0.0, 0.5, 1.0)
UNK_FRACS = (0.15, 0.45)
FIXED_GRID = np.round(np.arange(-1.0, 4.01, 0.05), 3)


# ---------------- 决策层（session-blind） ----------------
def asnorm(s: torch.Tensor, cohort_n: int = 100) -> torch.Tensor:
    top = s.argmax(1)
    k = s.shape[1]
    mu_k = torch.empty(k, device=s.device)
    sd_k = torch.empty(k, device=s.device)
    for c in range(k):
        v = s[top != c, c]
        v = v.topk(min(cohort_n, len(v))).values if len(v) > 1 else s[:, c]
        mu_k[c], sd_k[c] = v.mean(), v.std().clamp_min(1e-6)
    masked = s.clone()
    masked[torch.arange(len(s)), top] = float("nan")
    mu_x = masked.nanmean(1, keepdim=True)
    sd_x = ((masked - mu_x) ** 2).nanmean(1, keepdim=True).sqrt().clamp_min(1e-6)
    return 0.5 * ((s - mu_x) / sd_x + (s - mu_k) / sd_k)


def confidence(s: torch.Tensor, kind: str) -> torch.Tensor:
    t2 = s.topk(2, dim=1).values
    return t2[:, 0] if kind == "max" else t2[:, 0] - t2[:, 1]


def knn_smooth(conf: torch.Tensor, feats: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return conf
    f = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = f @ f.T
    sim.fill_diagonal_(-2)
    idx = sim.topk(k, dim=1).indices
    return 0.5 * conf + 0.5 * conf[idx].mean(1)


def gmm_accept(conf: np.ndarray) -> np.ndarray:
    g = GaussianMixture(2, random_state=0).fit(conf[:, None])
    hi = int(np.argmax(g.means_.ravel()))
    return g.predict_proba(conf[:, None])[:, hi] > 0.5


def decide(scores, feats, tasks, norm, smooth, conf_kind, device):
    """返回 (pred_label_index, confidence)，按 task 在完整目标批次内计算。"""
    pred = np.empty(len(scores), int)
    conf = np.empty(len(scores))
    for t in np.unique(tasks):
        i = np.flatnonzero(tasks == t)
        s = torch.as_tensor(scores[i], dtype=torch.float32, device=device)
        if norm == "asnorm":
            s = asnorm(s)
        c = confidence(s, conf_kind)
        c = knn_smooth(c, torch.as_tensor(feats[i], dtype=torch.float32, device=device), smooth)
        pred[i] = s.argmax(1).cpu().numpy()
        conf[i] = c.cpu().numpy()
    return pred, conf


def accept_mask(conf, tasks, thr, value):
    if thr == "gmm":
        acc = np.zeros(len(conf), bool)
        for t in np.unique(tasks):
            i = np.flatnonzero(tasks == t)
            acc[i] = gmm_accept(conf[i])
        return acc
    return conf >= value


# ---------------- 内部伪目标仿真 ----------------
def split_time(meta, fit_subj):
    """已注册被试每个 (subject, task) 按 trial 顺序：前 50% 训练，50–70% 源校准，后 30% 目标。"""
    role = np.full(len(meta), "", object)
    for (subj, _), g in meta.groupby(["subject", "task"]):
        if subj not in fit_subj:
            role[g.index] = "unknown"
            continue
        order = g.sort_values("trial").index.to_numpy()
        n = len(order)
        role[order[: int(0.5 * n)]] = "fit"
        role[order[int(0.5 * n): int(0.7 * n)]] = "calib"
        role[order[int(0.7 * n):]] = "target_known"
    return role


def add_drift(x, w, subj, scale, gamma, rng):
    """目标侧合成漂移：全局 + 被试特异偏移（单位为特征标准差）；对已知/未知一视同仁。"""
    if gamma == 0:
        return x, w
    g = rng.standard_normal(x.shape[1])
    d = np.zeros_like(x)
    for s in np.unique(subj):
        d[subj == s] = gamma * scale * (0.7 * g + 0.7 * rng.standard_normal(x.shape[1]))
    return x + d, w + d[:, None, :]


def evaluate(true, pred_lab, acc, is_known):
    out = np.where(acc, pred_lab, 0)
    C = float(np.mean(out[is_known] == true[is_known]))
    U = float(np.mean(out[~is_known] == 0))
    return C, U, float(f1_score(true, out, average="macro", zero_division=0))


def simulate(args, device):
    cache = resolve_cache()
    meta = pd.read_csv(cache / "train_metadata.csv")
    subj = meta.subject.to_numpy()
    tasks = meta.task.to_numpy()
    x = np.asarray(np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    w = np.asarray(np.load(cache / "train_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
                   ).reshape(len(meta), 2, -1)
    scale = x.std(0)
    rng = np.random.default_rng(0)
    subjects = rng.permutation(np.unique(subj))
    folds = np.array_split(subjects, 5)
    rows = []
    for f, unk in enumerate(folds):
        fit_subj = set(subjects) - set(unk)
        role = split_time(meta, fit_subj)
        model = TaskSpecificModel(device).fit(x, w, subj, tasks, role == "fit")
        classes = model.classes
        calib = np.flatnonzero(role == "calib")
        for gamma, frac in itertools.product(GAMMAS, UNK_FRACS):
            frng = np.random.default_rng(1000 * f + int(10 * gamma) + int(100 * frac))
            tk = np.flatnonzero(role == "target_known")
            uk = np.flatnonzero(role == "unknown")
            n_u = min(len(uk), int(round(frac / (1 - frac) * len(tk))))
            uk = frng.choice(uk, n_u, replace=False)
            tgt = np.concatenate([tk, uk])
            xt, wt = add_drift(x[tgt], w[tgt], subj[tgt], scale, gamma, frng)
            is_known = np.r_[np.ones(len(tk), bool), np.zeros(len(uk), bool)]
            true = np.where(is_known, subj[tgt], 0)
            for align in ALIGNS:
                sc, _ = predict_calibration_scores(model, xt, wt, tasks[tgt], align)
                sc_cal, _ = predict_calibration_scores(model, x[calib], w[calib], tasks[calib], align)
                feats = (xt - xt.mean(0)) / xt.std(0).clip(1e-6)
                for norm, smooth, ck in itertools.product(NORMS, SMOOTHS, CONFS):
                    p, c = decide(sc[f"base_{align}"], feats, tasks[tgt], norm, smooth, ck, device)
                    lab = classes[p]
                    auroc = float(roc_auc_score(is_known, c))
                    _, cc = decide(sc_cal[f"base_{align}"], x[calib], tasks[calib], norm, 0, ck, device)
                    base = dict(fold=f, gamma=gamma, unk_frac=frac, align=align, norm=norm,
                                smooth=smooth, conf=ck, auroc=auroc)
                    # src_q
                    acc = c >= np.quantile(cc, 0.05)
                    rows.append({**base, "thr": "src_q", "t": None,
                                 **dict(zip(("C", "U", "macro_f1"), evaluate(true, lab, acc, is_known)))})
                    acc = accept_mask(c, tasks[tgt], "gmm", None)
                    rows.append({**base, "thr": "gmm", "t": None,
                                 **dict(zip(("C", "U", "macro_f1"), evaluate(true, lab, acc, is_known)))})
                    for t in FIXED_GRID:
                        acc = c >= t
                        rows.append({**base, "thr": "fixed", "t": float(t),
                                     **dict(zip(("C", "U", "macro_f1"), evaluate(true, lab, acc, is_known)))})
        print(f"fold {f} done", flush=True)
    df = pd.DataFrame(rows)
    df["proxy"] = np.sqrt(df.C * df.U)
    return df


def summarize(df):
    """fixed 阈值用留一折选择；其余方法无需选择。"""
    keys = ["align", "norm", "smooth", "conf"]
    out = []
    for cfg, g in df.groupby(keys):
        for thr in ("src_q", "gmm"):
            h = g[g.thr == thr]
            out.append({**dict(zip(keys, cfg)), "thr": thr, "t": None, **h[["C", "U", "auroc", "macro_f1", "proxy"]].mean().to_dict(),
                        "proxy_min": h.groupby(["gamma", "unk_frac"]).proxy.mean().min()})
        h = g[g.thr == "fixed"]
        lofo = []
        for f in h.fold.unique():
            t_best = h[h.fold != f].groupby("t").proxy.mean().idxmax()
            lofo.append(h[(h.fold == f) & (h.t == t_best)])
        lofo = pd.concat(lofo)
        out.append({**dict(zip(keys, cfg)), "thr": "fixed",
                    "t": float(h.groupby("t").proxy.mean().idxmax()),
                    **lofo[["C", "U", "auroc", "macro_f1", "proxy"]].mean().to_dict(),
                    "proxy_min": lofo.groupby(["gamma", "unk_frac"]).proxy.mean().min()})
    return pd.DataFrame(out).sort_values("proxy", ascending=False)


# ---------------- 冻结后测试 ----------------
def predict_test(cfg, device):
    """只读取 EEG 特征与 task；不读取 session 列。"""
    cache = resolve_cache()
    meta = pd.read_csv(cache / "train_metadata.csv")
    x = np.asarray(np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    w = np.asarray(np.load(cache / "train_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
                   ).reshape(len(meta), 2, -1)
    model = TaskSpecificModel(device).fit(x, w, meta.subject.to_numpy(), meta.task.to_numpy(),
                                          np.ones(len(meta), bool))
    tmeta = pd.read_csv(cache / "test_metadata.csv", usecols=["sample", "task"])
    tx = np.asarray(np.load(cache / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    tw = np.asarray(np.load(cache / "test_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
                    ).reshape(len(tmeta), 2, -1)
    tasks = tmeta.task.to_numpy()
    sc, _ = predict_calibration_scores(model, tx, tw, tasks, cfg["align"])
    feats = (tx - tx.mean(0)) / tx.std(0).clip(1e-6)
    p, c = decide(sc[f"base_{cfg['align']}"], feats, tasks, cfg["norm"], int(cfg["smooth"]), cfg["conf"], device)
    if cfg["thr"] == "src_q":
        raise ValueError("src_q 未被选中时不实现测试分支")
    acc = accept_mask(c, tasks, cfg["thr"], cfg["t"])
    pred = np.where(acc, model.classes[p], 0)
    return tmeta, pred, c


def proxy_eval(tmeta, pred, conf):
    """事后诊断：读取 session，仅用于报告，不回流到任何选择。"""
    sess = pd.read_csv(resolve_cache() / "test_metadata.csv", usecols=["sample", "session"])
    sess = tmeta[["sample"]].merge(sess, on="sample", how="left").session.to_numpy()
    s2 = sess == "session2"
    acc = pred[s2] != 0
    labs, cnt = np.unique(pred[s2][acc], return_counts=True)
    return {
        "proxy_known_accept_session2": float(acc.mean()),
        "proxy_reject_session1": float(np.mean(pred[~s2] == 0)),
        "proxy_auroc_conf_s2_vs_s1": float(roc_auc_score(s2, conf)),
        "proxy_geo": float(np.sqrt(acc.mean() * np.mean(pred[~s2] == 0))),
        "session2_accepted_label_counts": {int(a): int(b) for a, b in zip(labs, cnt)},
        "n_session1": int((~s2).sum()), "n_session2": int(s2.sum()),
        "rejected_fraction_all": float(np.mean(pred == 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_sim:
        df = simulate(args, device)
        df.to_csv(OUT / "sim_raw.csv.gz", index=False)
    df = pd.read_csv(OUT / "sim_raw.csv.gz")
    summ = summarize(df)
    summ.to_csv(OUT / "sim_summary.csv", index=False)
    # 冻结：仅按内部仿真 proxy（均值）选择；不看 test session。
    cfg = summ[summ.thr != "src_q"].iloc[0].to_dict()
    frozen = {k: cfg[k] for k in ("align", "norm", "smooth", "conf", "thr", "t")}
    (OUT / "frozen_config.json").write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
    print("frozen:", frozen, flush=True)
    results = {"frozen": frozen, "sim_selected": cfg}
    # 同时报告基线（无 AS-norm）与冻结配置的测试 proxy，基线不参与选择
    baseline = summ[(summ.norm == "raw") & (summ.smooth == 0) & (summ.thr != "src_q")].iloc[0].to_dict()
    for name, c in (("frozen", cfg), ("best_raw_no_smooth", baseline)):
        if c["thr"] == "src_q":
            continue
        tmeta, pred, conf = predict_test(c, device)
        # Session 不变性：预测函数从未读取 session 列（usecols 仅 sample, task）
        if name == "frozen":
            template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
            sub = build_submission(template, pd.read_csv(resolve_cache() / "test_metadata.csv"), pred)
            assert len(sub) == 22251 and sub.Label.between(0, 20).all()
            sub.to_csv(OUT / "submission_asnorm_frozen.csv", index=False)
        results[name] = {"config": {k: c[k] for k in frozen}, **proxy_eval(tmeta, pred, conf)}
        print(name, json.dumps(results[name], default=str)[:600], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
