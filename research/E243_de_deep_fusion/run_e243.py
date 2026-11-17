"""E243: strict Train-only LOSO deep fusion with a task-conditioned MLP."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = list(CFG["tasks"])
CLASSES = np.arange(1, 21, dtype=int)
CANDIDATES = ["eeg_only", "eeg_logbp", "eeg_dino", "eeg_logbp_dino"]
SEED = 20261117


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.r_[neg, pos]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    s = x[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n0, n1 = len(neg), len(pos)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def binary_f1(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def pca_fit(x, dim):
    xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    mu = xt.mean(0)
    xc = xt - mu
    cov = (xc.T @ xc) / max(len(x) - 1, 1)
    w, v = torch.linalg.eigh(cov.double())
    basis = v[:, torch.argsort(w, descending=True)[: min(dim, x.shape[1])]].float()
    return mu.cpu().numpy().astype(np.float32), basis.cpu().numpy().astype(np.float32)


def apply_pca(x, mu, basis):
    out = []
    mt = torch.as_tensor(mu, device=DEVICE)
    vt = torch.as_tensor(basis, device=DEVICE)
    for i in range(0, len(x), 4096):
        out.append(((torch.as_tensor(x[i:i + 4096], device=DEVICE) - mt) @ vt).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def standardize_and_pca(raw, tasks, fit_mask, dim):
    z = raw.astype(np.float32, copy=True)
    stats = {}
    for task in TASKS:
        ref = fit_mask & (tasks == task)
        mu = z[ref].mean(0)
        sd = z[ref].std(0) + 1e-6
        z[tasks == task] = (z[tasks == task] - mu) / sd
        stats[task] = (mu.astype(np.float32), sd.astype(np.float32))
    mu, basis = pca_fit(z[fit_mask], dim)
    return apply_pca(z, mu, basis), {"task_stats": stats, "pca_mu": mu, "pca_basis": basis}


def de_from_logbp(logbp):
    # E146 is log band power; under the stated Gaussian-DE equivalence,
    # H = 0.5 * log(2*pi*e*variance), with variance represented by power.
    return (0.5 * (logbp.astype(np.float32) + np.log(2.0 * np.pi * np.e))).astype(np.float32)


def apply_states(raw, tasks, states, key):
    z = raw.astype(np.float32, copy=True)
    st = states[key]
    for task in TASKS:
        mu, sd = st["task_stats"][task]
        z[tasks == task] = (z[tasks == task] - mu) / sd
    return apply_pca(z, st["pca_mu"], st["pca_basis"])


def build_features(raw, logbp, dino, tasks, fit_mask, kind):
    parts, states = [], {}
    z, st = standardize_and_pca(raw, tasks, fit_mask, int(CFG["pca_dim"]))
    parts.append(z); states["eeg"] = st
    if "logbp" in kind:
        z, st = standardize_and_pca(de_from_logbp(logbp), tasks, fit_mask, int(CFG["pca_dim"]))
        parts.append(z); states["de_logbp"] = st
    if "dino" in kind:
        z, st = standardize_and_pca(dino, tasks, fit_mask, int(CFG["pca_dim"]))
        parts.append(z); states["dino"] = st
    return np.concatenate(parts, axis=1).astype(np.float32), states


def transform_with_states(raw, logbp, dino, tasks, states, kind):
    parts = [apply_states(raw, tasks, states, "eeg")]
    if "logbp" in kind:
        parts.append(apply_states(de_from_logbp(logbp), tasks, states, "de_logbp"))
    if "dino" in kind:
        parts.append(apply_states(dino, tasks, states, "dino"))
    return np.concatenate(parts, axis=1).astype(np.float32)


class IdentityMLP(nn.Module):
    def __init__(self, in_dim, hidden, task_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim + task_dim, hidden), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(hidden, 20))

    def forward(self, x, task_onehot):
        return self.net(torch.cat([x, task_onehot], 1))


def onehot(task_values):
    idx = np.asarray([TASKS.index(str(t)) for t in task_values])
    out = np.zeros((len(idx), len(TASKS)), np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def train_model(x, tasks, subjects, train_mask, val_mask, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    y = torch.as_tensor(subjects[train_mask] - 1, dtype=torch.long, device=DEVICE)
    xt = torch.as_tensor(x[train_mask], dtype=torch.float32, device=DEVICE)
    tt = torch.as_tensor(onehot(tasks[train_mask]), dtype=torch.float32, device=DEVICE)
    xv = torch.as_tensor(x[val_mask], dtype=torch.float32, device=DEVICE)
    tv = torch.as_tensor(onehot(tasks[val_mask]), dtype=torch.float32, device=DEVICE)
    model = IdentityMLP(x.shape[1], int(CFG["hidden"]), len(TASKS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG["lr"]), weight_decay=float(CFG["weight_decay"]))
    best, best_loss, wait = None, float("inf"), 0
    batch = int(CFG["batch_size"])
    for _ in range(int(CFG["epochs"])):
        model.train()
        order = torch.randperm(len(xt), device=DEVICE)
        for start in range(0, len(xt), batch):
            sel = order[start:start + batch]
            loss = F.cross_entropy(model(xt[sel], tt[sel]), y[sel])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.cross_entropy(model(xv, tv), torch.as_tensor(subjects[val_mask] - 1, device=DEVICE)).item())
        if vl < best_loss - 1e-5:
            best_loss, best, wait = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= int(CFG["patience"]): break
    if best is not None: model.load_state_dict(best)
    return model, best_loss


def predict(model, x, tasks):
    model.eval(); out = []
    xx = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    tt = torch.as_tensor(onehot(tasks), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for i in range(0, len(xx), 4096):
            out.append(torch.softmax(model(xx[i:i + 4096], tt[i:i + 4096]), 1).max(1).values.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def metric(known, unknown, threshold):
    y = np.r_[np.zeros(len(known), int), np.ones(len(unknown), int)]
    p = np.r_[known, unknown]
    reject = (p < threshold).astype(int)
    return {"AUC": auc(-unknown, -known), "F1": binary_f1(y, reject),
            "threshold": float(threshold), "known_acceptance": float((known >= threshold).mean()),
            "unknown_rejection": float((unknown < threshold).mean()), "n_known": int(len(known)), "n_unknown": int(len(unknown))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=20); ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args(); started = time.time()
    (OUT / "models").mkdir(exist_ok=True); (OUT / "scores").mkdir(exist_ok=True)
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects, tasks, fractions = meta.subject.to_numpy(int), meta.task.astype(str).to_numpy(), frac(meta)
    raw = np.load(ROOT / CFG["data"]["eeg"], mmap_mode="r").astype(np.float32)
    logbp = np.load(ROOT / CFG["data"]["logbp"], mmap_mode="r").astype(np.float32)
    dino = np.load(ROOT / CFG["data"]["dino"], mmap_mode="r").astype(np.float32)
    fit_all = fractions <= float(CFG["fit_fraction"])
    eval_all = (fractions > float(CFG["fit_fraction"])) & (fractions <= float(CFG["score_fraction"][1]))
    results = {n: {"folds": []} for n in CANDIDATES}; all_scores = {}
    held_ids = CLASSES[: min(args.folds, len(CLASSES))]
    for held in held_ids:
        outer_train = fit_all & (subjects != held)
        train_subjects = np.sort(np.unique(subjects[outer_train]))
        val_subject = int(train_subjects[-1])
        train_mask = outer_train & (subjects != val_subject)
        val_mask = outer_train & (subjects == val_subject)
        known_eval = eval_all & (subjects != held); unknown_eval = eval_all & (subjects == held)
        for name in CANDIDATES:
            x, states = build_features(raw, logbp, dino, tasks, train_mask, name)
            model, val_loss = train_model(x, tasks, subjects, train_mask, val_mask, SEED + int(held))
            val_prob = predict(model, x[val_mask], tasks[val_mask])
            threshold = float(np.quantile(val_prob, 1.0 - float(CFG["threshold_acceptance"])))
            # 行序按原始 metadata 排列，held 被试的行可能出现在最前，必须用布尔掩码按行对齐切分
            sel = np.flatnonzero(known_eval | unknown_eval)
            prob = predict(model, x[sel], tasks[sel])
            known_prob = prob[known_eval[sel]]
            unknown_prob = prob[unknown_eval[sel]]
            row = metric(known_prob, unknown_prob, threshold)
            row.update({"held_subject": int(held), "validation_subject": val_subject, "val_loss": val_loss})
            results[name]["folds"].append(row)
            all_scores[f"{name}_held_{held:02d}_known"] = known_prob
            all_scores[f"{name}_held_{held:02d}_unknown"] = unknown_prob
            torch.save({"model": model.state_dict(), "states": states, "candidate": name, "held_subject": int(held), "train_ids": np.flatnonzero(train_mask), "validation_ids": np.flatnonzero(val_mask), "threshold": threshold}, OUT / "models" / f"outer_{held:02d}_{name}.pt")
            print(f"held={held} {name} AUC={row['AUC']:.4f} F1={row['F1']:.4f}", flush=True)
    np.savez_compressed(OUT / "scores" / "outer_scores.npz", **all_scores)
    for name in CANDIDATES:
        fs = results[name]["folds"]
        results[name].update({"AUC": float(np.nanmean([r["AUC"] for r in fs])), "F1": float(np.nanmean([r["F1"] for r in fs]))})
    test_info = {"accessed": False, "AUC": None, "F1": None}
    if not args.no_test:
        test_meta_path = ROOT / CFG["data"]["test_metadata"]
        if test_meta_path.exists():
            tm = pd.read_csv(test_meta_path)
            traw = np.load(ROOT / CFG["data"]["eeg_test"], mmap_mode="r").astype(np.float32)
            tlbp = np.load(ROOT / CFG["data"]["logbp_test"], mmap_mode="r").astype(np.float32)
            tdino = np.load(ROOT / CFG["data"]["dino_test"], mmap_mode="r").astype(np.float32)
            test_info = {"accessed": True, "AUC": None, "F1": None, "n": int(len(tm)), "session_metrics": {}}
            for name in CANDIDATES:
                x, states = build_features(raw, logbp, dino, tasks, fit_all, name)
                # Refit final model only on the complete Train fit set; Test never affects fitting.
                all_train = fit_all
                val_subject = int(np.sort(np.unique(subjects[all_train]))[-1])
                model, _ = train_model(x, tasks, subjects, all_train, all_train & (subjects == val_subject), SEED + 900)
                test_tasks = tm.task.astype(str).to_numpy()
                test_x = transform_with_states(traw, tlbp, tdino, test_tasks, states, name)
                final_val_prob = predict(model, x[all_train & (subjects == val_subject)], tasks[all_train & (subjects == val_subject)])
                final_threshold = float(np.quantile(final_val_prob, 1.0 - float(CFG["threshold_acceptance"])))
                test_prob = predict(model, test_x, test_tasks)
                session_metrics = {"n_scored": int(len(test_x)), "max_confidence_mean": float(test_prob.mean()), "threshold_source": "final Train inner calibration quantile", "threshold": final_threshold}
                if "session" in tm.columns:
                    session = tm.session.astype(str).str.casefold().to_numpy()
                    session_metrics["S1"] = {"n": int(np.isin(session, ["session1", "s1", "1"]).sum()), "accepted": int((test_prob[np.isin(session, ["session1", "s1", "1"])] >= final_threshold).sum())}
                    session_metrics["S2"] = {"n": int((~np.isin(session, ["session1", "s1", "1"])).sum()), "accepted": int((test_prob[~np.isin(session, ["session1", "s1", "1"])] >= final_threshold).sum())}
                test_info["session_metrics"][name] = session_metrics
                torch.save({"model": model.state_dict(), "states": states, "candidate": name, "train_ids": np.flatnonzero(all_train)}, OUT / "models" / f"final_{name}.pt")
    payload = {"experiment": CFG["experiment"], "device": str(DEVICE), "runtime_seconds": time.time() - started, "protocol": "strict Train-only outer LOSO; fit fraction <= 0.6; validation is last available outer-train subject; class posterior max confidence; score=-maxprob", "de_definition": "E146 train_log_band_power is treated as log-power equivalent Gaussian variance: 0.5*(logpower+log(2*pi*e))", "results": results, "test": test_info}
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E243 DE + DINO deep fusion", "", f"设备：{DEVICE}；outer folds：{len(held_ids)}；Test AUC/F1：N/A（无真实标签）。", "", "| candidate | AUC | F1 |", "|---|---:|---:|"]
    for n in CANDIDATES: lines.append(f"| {n} | {results[n]['AUC']:.6f} | {results[n]['F1']:.6f} |")
    lines += ["", "协议：每个 outer fold 只使用非 held subject 的 fit<=0.6 行；任务内标准化和 PCA 均 fit-only。DE 明确按 E146 log-power 的等价高斯 DE 计算。阈值由内部 validation subject 的 known posterior 10% 分位校准，未查看 held subject。", "Test 仅作最终 Train 模型的描述性评分，未用于选择阈值或模型；S1/S2 仅在 metadata 有 session 字段时记录计数。"]
    (OUT / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({n: {"AUC": results[n]["AUC"], "F1": results[n]["F1"]} for n in CANDIDATES}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
