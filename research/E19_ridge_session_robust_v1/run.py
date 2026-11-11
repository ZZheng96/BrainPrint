"""GPU Ridge baseline versus fit-only robust covariance normalization."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.open_set_improvement.core import COV_COLUMNS, resolve_cache

OUT = Path(__file__).resolve().parent
METHODS = ("full", "base", "robust")
BASELINE = {"AUC": 0.965632, "identity_macro_F1": 0.936207}


def load_split(cache, split, n):
    full = np.load(cache / f"{split}_features.npy", mmap_mode="r")
    win = np.load(cache / f"{split}_window2_features.npy", mmap_mode="r")
    x = np.asarray(full[:, COV_COLUMNS], dtype=np.float32)
    w = np.asarray(win[:, COV_COLUMNS], dtype=np.float32).reshape(n, 2, len(COV_COLUMNS))
    if len(x) != n or len(w) != n or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError(f"invalid {split} feature cache")
    return x, w


def forward_mask(meta):
    mx = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return meta["trial"].to_numpy() > .8 * mx


class GpuRidge:
    def __init__(self, device, alpha=100.0):
        self.device, self.alpha = device, alpha

    def fit(self, x, y, classes):
        z = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.mean = z.mean(0); self.scale = z.std(0, unbiased=False).clamp_min(1e-6)
        z = (z - self.mean) / self.scale
        self.classes = np.asarray(classes, dtype=int)
        target = torch.full((len(y), len(classes)), -1., device=self.device)
        lookup = {int(c): i for i, c in enumerate(self.classes)}
        target[torch.arange(len(y), device=self.device), torch.as_tensor([lookup[int(v)] for v in y], device=self.device)] = 1.
        target -= target.mean(0)
        if len(y) < z.shape[1]:
            gram = z @ z.T; gram.diagonal().add_(self.alpha)
            self.coef = z.T @ torch.linalg.solve(gram, target)
        else:
            gram = z.T @ z; gram.diagonal().add_(self.alpha)
            self.coef = torch.linalg.solve(gram, z.T @ target)
        return self

    def __call__(self, x):
        z = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return ((z - self.mean) / self.scale) @ self.coef


class Model:
    def __init__(self, device): self.device = device

    def fit(self, x, w, subjects, tasks, mask):
        self.classes = np.unique(subjects[mask]).astype(int)
        self.models, self.robust_stats = {}, {}
        for task in sorted(np.unique(tasks[mask])):
            ix = np.flatnonzero(mask & (tasks == task))
            y = subjects[ix]
            xx, ww = x[ix], w[ix]
            full = GpuRidge(self.device).fit(xx, y, self.classes)
            win = GpuRidge(self.device).fit(ww.reshape(-1, x.shape[1]), np.repeat(y, 2), self.classes)
            # Fit robust location/scale only on training observations; six covariance blocks are independent.
            t = torch.as_tensor(np.concatenate([xx, ww.reshape(-1, x.shape[1])]), dtype=torch.float32, device=self.device).reshape(-1, 6, 171)
            med = t.median(0).values
            mad = (t - med).abs().median(0).values.mul(1.4826).clamp_min(1e-5)
            self.robust_stats[task] = (med, mad)
            rx = ((torch.as_tensor(xx, device=self.device).reshape(-1, 6, 171) - med) / mad).reshape(-1, x.shape[1]).cpu().numpy()
            rw = ((torch.as_tensor(ww, device=self.device).reshape(-1, 6, 171) - med) / mad).reshape(-1, 2, x.shape[1]).cpu().numpy()
            robust = GpuRidge(self.device).fit(rx, y, self.classes)
            robust_w = GpuRidge(self.device).fit(rw.reshape(-1, x.shape[1]), np.repeat(y, 2), self.classes)
            self.models[task] = (full, win, robust, robust_w)
        return self

    def scores(self, x, w, tasks):
        out = {m: np.full((len(x), len(self.classes)), np.nan, np.float32) for m in METHODS}
        for task in sorted(np.unique(tasks)):
            pos = np.flatnonzero(tasks == task); full, win, robust, robust_w = self.models[task]
            xx, ww = x[pos], w[pos]
            a = full(torch.as_tensor(xx, device=self.device))
            b = (win(torch.as_tensor(ww[:, 0], device=self.device)) + win(torch.as_tensor(ww[:, 1], device=self.device))) / 2
            med, mad = self.robust_stats[task]
            rx = ((torch.as_tensor(xx, device=self.device).reshape(-1, 6, 171) - med) / mad).reshape(-1, x.shape[1])
            rw = ((torch.as_tensor(ww, device=self.device).reshape(-1, 6, 171) - med) / mad).reshape(-1, 2, x.shape[1])
            r = robust(rx); rb = (robust_w(rw[:, 0]) + robust_w(rw[:, 1])) / 2
            out["full"][pos] = a.detach().cpu().numpy()
            out["base"][pos] = (.5 * a + .5 * b).detach().cpu().numpy()
            out["robust"][pos] = (.5 * r + .5 * rb).detach().cpu().numpy()
        if any(np.isnan(v).any() for v in out.values()): raise RuntimeError("missing task scores")
        return out


def confidence(scores):
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0] if scores.shape[1] > 1 else top[:, 1]


def threshold(known, unknown):
    y = np.r_[np.ones(len(known)), np.zeros(len(unknown))]; s = np.r_[known, unknown]
    best = None
    for t in np.unique(known):
        row = (f1_score(y, s >= t, average="macro", zero_division=0), f1_score(y, s >= t, pos_label=0, zero_division=0), roc_auc_score(y, s))
        if best is None or row > best[0]: best = (row, float(t))
    return best[1]


def report(true, scores, known_n, t, classes):
    conf = confidence(scores); pred = classes[scores.argmax(1)].copy(); pred[conf < t] = 0
    y = np.r_[np.ones(known_n, int), np.zeros(len(scores)-known_n, int)]
    all_true = np.r_[true, np.zeros(len(scores)-known_n, int)]
    accepted = conf >= t
    return {"AUC": float(roc_auc_score(y, conf)), "identity_macro_F1": float(f1_score(all_true, pred, average="macro", zero_division=0)), "detection_macro_F1": float(f1_score(y, accepted, average="macro", zero_division=0)), "unknown_F1": float(f1_score(y, accepted, pos_label=0, zero_division=0)), "U": float(np.mean(~accepted[known_n:])), "known_acceptance": float(np.mean(accepted[:known_n])), "C": float(np.mean(pred[:known_n] == true)), "threshold": float(t), "n_known": int(known_n), "n_unknown": int(len(scores)-known_n)}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required")
    if any((OUT / n).exists() for n in ("metrics.json", "report.md", "oof_scores.npy", "test_candidate.csv")):
        raise FileExistsError(f"refusing to overwrite {OUT}")
    torch.manual_seed(42); device = torch.device("cuda"); cache = resolve_cache()
    meta = pd.read_csv(cache / "train_metadata.csv"); subjects, tasks = meta.subject.to_numpy(), meta.task.to_numpy()
    x, w = load_split(cache, "train", len(meta)); all_subjects = np.unique(subjects); forward = forward_mask(meta)
    rng = np.random.default_rng(42); folds = []
    for fold, held in enumerate(np.array_split(rng.permutation(all_subjects), 5), 1):
        enrolled = np.setdiff1d(all_subjects, held); ik = {m: [] for m in METHODS}; iu = {m: [] for m in METHODS}
        for pseudo in np.array_split(rng.permutation(enrolled), 2):
            registered = np.isin(subjects, np.setdiff1d(enrolled, pseudo)); fit = registered & ~forward
            ki = np.flatnonzero(registered & forward); ui = np.flatnonzero(np.isin(subjects, pseudo) & ~forward)
            sc = Model(device).fit(x, w, subjects, tasks, fit).scores(x[np.r_[ki, ui]], w[np.r_[ki, ui]], tasks[np.r_[ki, ui]])
            for m in METHODS: ik[m].append(confidence(sc[m][:len(ki)])); iu[m].append(confidence(sc[m][len(ki):]))
        ts = {m: threshold(np.concatenate(ik[m]), np.concatenate(iu[m])) for m in METHODS}
        fit = np.isin(subjects, enrolled) & ~forward; ki = np.flatnonzero(np.isin(subjects, enrolled) & forward); ui = np.flatnonzero(np.isin(subjects, held)); sel = np.r_[ki, ui]
        model = Model(device).fit(x, w, subjects, tasks, fit); sc = model.scores(x[sel], w[sel], tasks[sel])
        row = {"outer_fold": fold, "held_subjects": held.tolist(), "methods": {m: report(subjects[ki], sc[m], len(ki), ts[m], model.classes) for m in METHODS}}
        folds.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {m: {k: float(np.mean([f["methods"][m][k] for f in folds])) for k in ("AUC", "identity_macro_F1", "detection_macro_F1", "unknown_F1", "U", "known_acceptance", "C")} for m in METHODS}
    selected = "robust" if (summary["robust"]["AUC"] >= BASELINE["AUC"] and summary["robust"]["identity_macro_F1"] >= BASELINE["identity_macro_F1"]) else "base"
    # Fresh blocked OOF, retaining scores for selected method.
    block = np.full(len(meta), -1, int)
    for ix in meta.groupby(["subject", "task"], sort=False).indices.values():
        ordered = ix[np.argsort(meta.iloc[ix].trial.to_numpy(), kind="stable")]
        for k, part in enumerate(np.array_split(ordered, 5)): block[part] = k
    oof = np.full((len(meta), len(all_subjects)), np.nan, np.float32)
    for k in range(5):
        valid = block == k; mm = Model(device).fit(x, w, subjects, tasks, ~valid); oof[valid] = mm.scores(x[valid], w[valid], tasks[valid])[selected]
    final = Model(device).fit(x, w, subjects, tasks, np.ones(len(meta), bool)); tm = pd.read_csv(cache / "test_metadata.csv", usecols=["sample", "task"]); tx, tw = load_split(cache, "test", len(tm)); test_scores = final.scores(tx, tw, tm.task.to_numpy())[selected]
    # Threshold from all-Train OOF only; test metadata session is not read until candidate is frozen.
    t = float(np.quantile(confidence(oof), .05)); labels = final.classes[test_scores.argmax(1)].copy(); labels[confidence(test_scores) < t] = 0
    submission = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv"); order = tm["sample"].map(pd.Series(np.arange(len(tm)), index=tm["sample"])); submission["Label"] = labels[order.to_numpy(dtype=int)]; submission.to_csv(OUT / "test_candidate.csv", index=False)
    audit_meta = pd.read_csv(cache / "test_metadata.csv", usecols=["session"]); audit = {str(k): int(np.sum(labels[v] == 0)) for k, v in audit_meta.groupby("session").indices.items()}
    np.save(OUT / "oof_scores.npy", oof)
    result = {"protocol": {"outer": "strict 5-fold subject holdout; enrolled final 20% trials known, held subjects unknown", "inner": "pseudo-unknown only for threshold selection", "features": "full/base Ridge and six-block robust median/MAD covariance normalization; fit samples only", "rejection": "top1 margin", "window": "full+window score average for base and robust", "session_used_for_inference_or_threshold": False, "gpu": torch.cuda.get_device_name(0), "ridge_baseline": BASELINE}, "folds": folds, "summary": summary, "selected": selected, "candidate": {"threshold": t, "rows": len(labels), "rejected": int(np.sum(labels == 0)), "session_post_output_audit": audit}}
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Ridge session robust v1 实验报告", "", "严格五折 subject holdout；outer enrolled 的后20% trial 为 known，held subjects 全部为 unknown；inner 伪未知只用于阈值选择。拒识统一使用 top-1 margin，Test Session 仅候选写出后审计。", "", "|方法|AUC|identity macro-F1|detection macro-F1|unknown-F1|U|known acceptance|C|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        r = summary[m]; lines.append(f"|{m}|{r['AUC']:.6f}|{r['identity_macro_F1']:.6f}|{r['detection_macro_F1']:.6f}|{r['unknown_F1']:.6f}|{r['U']:.6f}|{r['known_acceptance']:.6f}|{r['C']:.6f}|")
    robust_ok = summary["robust"]["AUC"] >= BASELINE["AUC"] and summary["robust"]["identity_macro_F1"] >= BASELINE["identity_macro_F1"]
    lines += ["", f"历史 Ridge 基线：AUC={BASELINE['AUC']:.6f}，identity macro-F1={BASELINE['identity_macro_F1']:.6f}。", ("稳健归一化达到或超过基线，作为候选方法。" if robust_ok else "稳健归一化低于 Ridge 基线，淘汰原因：外层平均 AUC 或 identity macro-F1 未达到 0.965632/0.936207，因此候选不采用 robust；候选使用 base Ridge 对照。"), f"最终候选方法：{selected}；Test 行数={len(labels)}，拒识={int(np.sum(labels == 0))}。Test 无真值，Test AUC/F1 不可计算。", "Session 仅在候选写出后做按会话拒识计数审计，未参与训练、阈值、模型选择或后处理。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
