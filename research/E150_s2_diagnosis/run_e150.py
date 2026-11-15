"""E150: diagnose Test S2 false rejects with frozen E143/E65 protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

CFG = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "config.json"
OUT = CFG.parent
ROOT = OUT.parents[1]
C = json.loads(CFG.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = C["tasks"]


def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x: np.ndarray, dim: int):
    mu = x.mean(0)
    xc = x - mu
    _, v = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, v[:, np.argsort(-_)[: min(dim, x.shape[1])]]


def project(train: np.ndarray, test: np.ndarray, tm: np.ndarray, qtm: np.ndarray):
    a, b = train.copy(), test.copy()
    for task in TASKS:
        m, q = tm == task, qtm == task
        mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
        a[m] = (a[m] - mu) / sd
        b[q] = (b[q] - mu) / sd
    mu, v = pca_basis(a, C["pca_dim"])
    return ((a - mu) @ v).astype(np.float32), ((b - mu) @ v).astype(np.float32)


def centroids(x, meta, allowed, fit_mask):
    out, keys = [], []
    subjects, tasks = meta.subject.to_numpy(), meta.task.to_numpy()
    k = int(C["prototype_k"])
    for task in TASKS:
        for identity in sorted(meta.subject.unique()):
            m = allowed & fit_mask & (tasks == task) & (subjects == identity)
            if not m.any():
                continue
            block = torch.as_tensor(x[m], device=DEVICE)
            if len(block) <= k:
                centres = block
            else:
                gen = torch.Generator(device=DEVICE).manual_seed(int(identity) * 17)
                centres = block[torch.randperm(len(block), generator=gen, device=DEVICE)[:k]]
                for _ in range(20):
                    labels = torch.cdist(block, centres).argmin(1)
                    sums = torch.zeros_like(centres).index_add_(0, labels, block)
                    counts = torch.bincount(labels, minlength=k).to(block.dtype)
                    new = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centres)
                    if torch.allclose(new, centres, atol=1e-6):
                        break
                    centres = new
            out.extend(centres.detach().cpu().numpy())
            keys.extend([(task, int(identity))] * len(centres))
    return np.asarray(out, dtype=np.float32), keys


def score(x, tasks, protos, keys):
    result = np.full(len(x), np.nan, dtype=np.float32)
    identity = np.full(len(x), -1, dtype=int)
    distance = np.full(len(x), np.nan, dtype=np.float32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if not len(idx) or not cols:
            continue
        d = torch.cdist(torch.as_tensor(x[idx], device=DEVICE), torch.as_tensor(protos[cols], device=DEVICE))
        j = d.argmin(1).cpu().numpy()
        vals = d[torch.arange(len(idx), device=DEVICE), torch.as_tensor(j, device=DEVICE)].cpu().numpy()
        result[idx], distance[idx] = vals, vals
        identity[idx] = np.asarray([keys[cols[k]][1] for k in j], dtype=int)
    return result, identity, distance


def summary(values):
    values = np.asarray(values, float)
    q = np.quantile(values[np.isfinite(values)], [0.02, .05, .5, .95])
    return {"N": int(np.isfinite(values).sum()), "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)), "q02": float(q[0]), "q05": float(q[1]),
            "median": float(q[2]), "q95": float(q[3])}


def dist_rows(name, values, task, session=None):
    row = {"split": name, "task": task, **summary(values)}
    if session is not None:
        row["session"] = session
    return row


def main():
    train_meta = pd.read_csv(ROOT / C["data"]["train_metadata"])
    test_meta = pd.read_csv(ROOT / C["data"]["test_metadata"])
    train = np.load(ROOT / C["features"]["train"]).astype(np.float32)
    test = np.load(ROOT / C["features"]["test"]).astype(np.float32)
    if len(train) != len(train_meta) or len(test) != len(test_meta):
        raise ValueError("feature/metadata 行数不一致")
    train_x, test_x = project(train, test, train_meta.task.to_numpy(), test_meta.task.to_numpy())
    tf = frac(train_meta)
    train_tasks = train_meta.task.to_numpy(); test_tasks = test_meta.task.to_numpy()
    fit = tf <= C["fit_fraction"]
    blocks = np.clip((tf * 5).astype(int), 0, 4)

    rows, oof_dist, oof_true, oof_pred = [], [], [], []
    for block in range(5):
        allowed = blocks != block
        p, keys = centroids(train_x, train_meta, allowed, fit)
        d, pred, _ = score(train_x, train_tasks, p, keys)
        held = blocks == block
        oof_dist.extend(d[held]); oof_true.extend(train_meta.subject.to_numpy()[held]); oof_pred.extend(pred[held])
    oof_dist = np.asarray(oof_dist); oof_true = np.asarray(oof_true); oof_pred = np.asarray(oof_pred)
    # Reconstruct OOF task order directly to avoid relying on subject ordering.
    oof_values, oof_tasks = [], []
    for block in range(5):
        p, keys = centroids(train_x, train_meta, blocks != block, fit)
        held = blocks == block
        d, pred, _ = score(train_x[held], train_tasks[held], p, keys)
        oof_values.extend(d); oof_tasks.extend(train_tasks[held])
    oof_values, oof_tasks = np.asarray(oof_values), np.asarray(oof_tasks)
    rows = [dist_rows("Train_OOF", oof_values[oof_tasks == task], task) for task in TASKS]

    p, keys = centroids(train_x, train_meta, np.ones(len(train_x), bool), fit)
    test_d, test_pred, _ = score(test_x, test_tasks, p, keys)
    train_d, _, _ = score(train_x, train_tasks, p, keys)
    sessions = test_meta.session.astype(str).str.casefold().map(
        lambda s: "S1" if s in {"session1", "s1", "1"} else "S2" if s in {"session2", "s2", "2"} else s.upper()).to_numpy()
    for task in TASKS:
        rows.append(dist_rows("Train_fit", train_d[train_tasks == task], task))
        for sess in ("S1", "S2"):
            rows.append(dist_rows("Test", test_d[(test_tasks == task) & (sessions == sess)], task, sess))
    test_s1, test_s2 = sessions == "S1", sessions == "S2"
    valid = np.isfinite(test_d) & (test_s1 | test_s2)
    auc = float(roc_auc_score(test_s1[valid].astype(int), -test_d[valid]))
    session_pred = (test_d[valid] <= np.median(train_d[np.isfinite(train_d)])).astype(int)
    diag_f1 = float(f1_score(test_s1[valid].astype(int), session_pred, zero_division=0))
    oof_auc = float("nan")
    oof_f1 = float(f1_score(oof_true, oof_pred, labels=sorted(train_meta.subject.unique()), average="macro", zero_division=0))
    results = {"experiment": C["experiment"], "protocol": {"train_only_fit": True, "test_statistics_used_for_modeling": False,
        "test_session_used_only_for_diagnosis": True, "identity_field": "metadata.subject for Train; unavailable on Test; task/session/sample-order fallback"},
        "device": str(DEVICE), "features": {"train": C["features"]["train"], "test": C["features"]["test"]},
        "distribution_rows": rows, "metrics": {"Train_OOF_AUC": "N/A(no pseudo-unknown labels)", "Train_OOF_F1": oof_f1,
        "Test_official_AUC": "N/A(no true labels)", "Test_official_F1": "N/A(no true labels)",
        "Test_session_diagnostic_AUC": auc, "Test_session_diagnostic_F1": diag_f1},
        "baseline_W": C["baseline_W"], "diagnosis": {"S1_N": int(test_s1.sum()), "S2_N": int(test_s2.sum()),
        "S2_acceptance_at_train_fit_median": float(np.mean(test_d[~test_s1] <= np.median(train_d[np.isfinite(train_d)]))),
        "S2_rejection_at_train_fit_median": float(np.mean(test_d[~test_s1] > np.median(train_d[np.isfinite(train_d)])))}}
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    best = C["baseline_W"]
    lines = ["# E150 S2 误拒诊断", "", "## 结论", "", f"当前 W 基线沿用 E143：**{best['W']:.4f}**（{best['set']}，k={best['k']}，接受率={best['acceptance_rate']:.3f}）。",
             f"Train-only 原型距离的 Test session 诊断 AUC={auc:.4f}，诊断 F1={diag_f1:.4f}；这不是官方真值指标。Test 官方 AUC/F1：**N/A（无真实标签）**。",
             f"S2 样本 {int((~test_s1).sum())} 条，在 Train-fit 距离中位数阈值下接受率={results['diagnosis']['S2_acceptance_at_train_fit_median']:.4f}，拒绝率={results['diagnosis']['S2_rejection_at_train_fit_median']:.4f}。", "",
             "## 口径与限制", "- 沿用 E143 的 E65 logSPD 特征、Train-only task 标准化/PCA 和 identity 原型；没有使用 Test 距离、分位数或其他统计量拟合模型。", "- Test identity 不在 metadata 中；仅按 task 分层，并用 session/sample 顺序字段做诊断切片。Session 不能替代真实身份标签。", "- Train OOF 为五个 trial 顺序块留出；Test session 标签只用于报告，不参与阈值、PCA、原型或模型选择。", "", "## 距离分布", "", "| split | task | session | N | mean | std | q02 | q05 | median | q95 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['split']} | {r['task']} | {r.get('session','-')} | {r['N']} | {r['mean']:.5f} | {r['std']:.5f} | {r['q02']:.5f} | {r['q05']:.5f} | {r['median']:.5f} | {r['q95']:.5f} |")
    lines += ["", "## 复现", f"`{sys.executable} {Path(__file__).resolve()}`", "", "完整数值见 `results.json`。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
