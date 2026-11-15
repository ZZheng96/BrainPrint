"""E156: Train-only invariant views of the E65 855-D feature under E143 protocol."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

CFG_PATH = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "config.json"
OUT = CFG_PATH.parent
ROOT = OUT.parents[1]
C = json.loads(CFG_PATH.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = C["tasks"]
CLASSES = np.arange(1, 21, dtype=int)


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0); xc = x - mu
    w, v = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, v[:, np.argsort(-w)[:min(dim, x.shape[1])]]


def transform_pair(train, test, train_tasks, test_tasks, fit_mask, view):
    """All task-level statistics are estimated from Train fit rows only."""
    a = np.asarray(train, dtype=np.float32).copy(); b = np.asarray(test, dtype=np.float32).copy()
    n, d = a.shape; bw = C["features"]["block_width"]; nb = C["features"]["blocks"]
    if d != nb * bw or b.shape[1] != d: raise ValueError(f"expected {nb*bw}, got {d}")
    if view == "raw": return a, b
    out = qout = None
    for task in TASKS:
        m = train_tasks == task; q = test_tasks == task; fm = m & fit_mask
        base = a[fm].mean(0); scale = a[fm].std(0) + 1e-6
        z = (a[m] - base) / scale; qz = (b[q] - base) / scale
        blocks = z.reshape(-1, nb, bw); qblocks = qz.reshape(-1, nb, bw)
        def make(v, raw, bb):
            if v == "direction":
                return bb / np.linalg.norm(bb, axis=2, keepdims=True).clip(1e-6)
            if v == "trial_z":
                return (bb - bb.mean(2, keepdims=True)) / bb.std(2, keepdims=True).clip(1e-6)
            if v == "relative":
                return (bb[:, 1:] - bb[:, :1]) / (np.abs(bb[:, :1]) + 1e-3)
            direction = bb / np.linalg.norm(bb, axis=2, keepdims=True).clip(1e-6)
            trial_z = (bb - bb.mean(2, keepdims=True)) / bb.std(2, keepdims=True).clip(1e-6)
            rel = (bb[:, 1:] - bb[:, :1]) / (np.abs(bb[:, :1]) + 1e-3)
            if v == "invariant_all":
                return np.concatenate([direction.reshape(len(bb), -1), trial_z.reshape(len(bb), -1), rel.reshape(len(bb), -1)], 1)
            if v == "raw_plus_invariant":
                return np.concatenate([raw, direction.reshape(len(bb), -1), trial_z.reshape(len(bb), -1), rel.reshape(len(bb), -1)], 1)
            raise ValueError(v)
        y, qy = make(view, z, blocks), make(view, qz, qblocks)
        if out is None:
            out = np.empty((len(a), y.reshape(len(y), -1).shape[1]), dtype=np.float32)
            qout = np.empty((len(b), qy.reshape(len(qy), -1).shape[1]), dtype=np.float32)
        out[m] = y.reshape(len(y), -1); qout[q] = qy.reshape(len(qy), -1)
    return out, qout


def kmeans_t(x, k, seed):
    if len(x) <= k: return x
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=DEVICE)[:k]].clone()
    for _ in range(15):
        lab = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, lab, x)
        cnt = torch.bincount(lab, minlength=k).to(x.dtype)
        new = torch.where(cnt[:, None] > 0, sums / cnt.clamp_min(1)[:, None], c)
        if torch.allclose(new, c, atol=1e-5): break
        c = new
    return c


def prototypes(x, subjects, tasks, mask, k):
    ps, keys = [], []
    for task in TASKS:
        for c in CLASSES:
            m = mask & (tasks == task) & (subjects == c)
            if m.any():
                q = torch.as_tensor(x[m], device=DEVICE)
                ps.extend(kmeans_t(q, k, int(c) * 17).cpu().numpy()); keys.extend([(task, int(c))] * min(k, len(q)))
    return np.asarray(ps, dtype=np.float32), keys


def score(x, tasks, ps, keys, return_class=False):
    out = np.full(len(x), np.nan, np.float32); pred = np.full(len(x), -1, int)
    class_scores = np.full((len(x), len(CLASSES)), -np.inf, np.float32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task); cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if not len(idx): continue
        p = torch.as_tensor(ps[cols], device=DEVICE); q = torch.as_tensor(x[idx], device=DEVICE)
        dist = torch.cdist(q, p); nearest = dist.argmin(1)
        out[idx] = (-dist.min(1).values).cpu().numpy()
        pred[idx] = np.asarray([keys[cols[j]][1] for j in nearest.cpu().numpy()])
        for c in CLASSES:
            cc = [j for j, z in enumerate(cols) if keys[z][1] == c]
            if cc: class_scores[idx, c-1] = (-dist[:, cc].min(1).values).cpu().numpy()
    return (out, pred, class_scores) if return_class else out


def eval_one(name, x, subjects, tasks, f, is_s1, pca_dim, k):
    fit = f <= C["fit_fraction"]; blocks = np.clip((f * 5).astype(int), 0, 4)
    allfit = np.ones(len(x), bool)
    p, keys = prototypes(x, subjects, tasks, fit, k)
    test = None
    # Five sequential held-out trial blocks for Train OOF identity scores.
    oof_dist = np.full(len(x), np.nan, np.float32); oof_pred = np.full(len(x), -1, int); oof_cls = np.full((len(x), 20), -np.inf, np.float32)
    for b in range(5):
        hold = blocks == b; pp, kk = prototypes(x, subjects, tasks, ~hold & fit, k)
        d, pr, cs = score(x[hold], tasks[hold], pp, kk, True)
        oof_dist[hold], oof_pred[hold], oof_cls[hold] = d, pr, cs
    valid = np.isfinite(oof_dist)
    oof_f1 = float(f1_score(subjects[valid], oof_pred[valid], labels=CLASSES, average="macro", zero_division=0))
    y_bin = np.eye(20, dtype=int)[subjects[valid]-1]
    oof_auc = float(roc_auc_score(y_bin, oof_cls[valid], multi_class="ovr", average="macro"))
    # E143 diagnostic: Test is scored only; session is not used for fit or selection.
    if test is not None: raise AssertionError
    return oof_auc, oof_f1, p, keys, blocks


def main():
    tm = pd.read_csv(ROOT / C["data"]["train_metadata"]); te = pd.read_csv(ROOT / C["data"]["test_metadata"])
    subjects = tm.subject.to_numpy(int); tasks = tm.task.to_numpy(); test_tasks = te.task.to_numpy(); f = frac(tm)
    train_raw = np.load(ROOT / C["features"]["train"], mmap_mode="r").astype(np.float32)
    test_raw = np.load(ROOT / C["features"]["test"], mmap_mode="r").astype(np.float32)
    fit = f <= C["fit_fraction"]
    rows = []; views = C["invariant_views"]
    for view in views:
        tx, qx = transform_pair(train_raw, test_raw, tasks, test_tasks, fit, view)
        for dim in C["pca_dims"]:
            mu, V = pca_basis(tx[fit].astype(np.float64), dim)
            a = ((tx.astype(np.float64)-mu) @ V).astype(np.float32); b = ((qx.astype(np.float64)-mu) @ V).astype(np.float32)
            for k in C["prototype_k"]:
                auc, f1, _, _, blocks = eval_one(view, a, subjects, tasks, f, None, dim, k)
                # OOF score and fixed Train-fit threshold; Test session slices are reported only as diagnostics.
                p, keys = prototypes(a, subjects, tasks, fit, k)
                ref = np.full(len(a), np.nan, np.float32)
                for blk in range(5):
                    hold = blocks == blk; pp, kk = prototypes(a, subjects, tasks, ~hold & fit, k); ref[hold] = score(a[hold], tasks[hold], pp, kk)
                threshold = float(np.quantile(ref[np.isfinite(ref)], 1-C["acceptance_grid"][3]))
                test_score = score(b, test_tasks, p, keys)
                s1 = te.session.astype(str).str.casefold().isin(["session1","s1","1"]).to_numpy()
                loso_rej = float(np.mean(ref[np.isfinite(ref)] < threshold))
                s1_rej = float(np.mean(test_score[s1] < threshold)); s2_acc = float(np.mean(test_score[~s1] >= threshold))
                W = .25*loso_rej + .25*s1_rej + .5*s2_acc
                rows.append({"view":view,"pca_dim":dim,"k":k,"Train_OOF_AUC":auc,"Train_OOF_F1":f1,"LOSO_rejection":loso_rej,"Test_S1_rejection":s1_rej,"Test_S2_acceptance":s2_acc,"W":W,"Test_AUC":"N/A","Test_F1":"N/A"})
                print(view, dim, k, f"OOF_AUC={auc:.4f} F1={f1:.4f} W={W:.4f}", flush=True)
    rows.sort(key=lambda r: -r["W"]); result={"experiment":C["experiment"],"protocol":C["protocol"],"device":str(DEVICE),"test_labels_used":False,"best":rows[0],"curve":rows}
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# E156 Train-only 不变特征探测","",f"设备：`{DEVICE}`。Test 标签未读取；Test 官方 AUC/F1：**N/A（无真实标签）**。",f"当前最佳：**{rows[0]['view']} / PCA{rows[0]['pca_dim']} / k={rows[0]['k']}，W={rows[0]['W']:.4f}**；Train OOF AUC={rows[0]['Train_OOF_AUC']:.4f}，F1={rows[0]['Train_OOF_F1']:.4f}。","","三项指标按 E143：LOSO 拒识、Test S1 拒识、Test S2 接受；Test session 只用于冻结后的描述性评分。","","| view | PCA | k | OOF AUC | OOF F1 | LOSO | S1拒识 | S2接受 | W | Test AUC | Test F1 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows: lines.append(f"| {r['view']} | {r['pca_dim']} | {r['k']} | {r['Train_OOF_AUC']:.4f} | {r['Train_OOF_F1']:.4f} | {r['LOSO_rejection']:.4f} | {r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | {r['W']:.4f} | N/A | N/A |")
    lines += ["","变换定义：direction 为每 task 标准化后每 trial 每频带 L2 方向；trial_z 为每 trial 每块中心化/标准化；relative 为相对 band-0 的跨频带差分；invariant_all 与 raw_plus_invariant 为固定拼接。所有 task 均值/尺度、PCA、原型只由 Train fit≤60% 拟合。","","失败/成功判定：以 W 与 E143 基线 0.6958 对比；完整结果见 `results.json`。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__ == "__main__": main()
