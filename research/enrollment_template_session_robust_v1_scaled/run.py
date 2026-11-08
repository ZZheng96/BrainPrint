"""GPU robust subject-template enrollment with session-blind nested evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.open_set_improvement.core import COV_COLUMNS, resolve_cache

OUT = Path(__file__).resolve().parent
METHODS = ("prototype_only", "prototype_margin", "prototype_window")


def load_split(cache: Path, split: str, n: int):
    full = np.load(cache / f"{split}_features.npy", mmap_mode="r")
    win = np.load(cache / f"{split}_window2_features.npy", mmap_mode="r")
    x = np.asarray(full[:, COV_COLUMNS], dtype=np.float32)
    w = np.asarray(win[:, COV_COLUMNS], dtype=np.float32).reshape(n, 2, len(COV_COLUMNS))
    if len(x) != n or len(w) != n or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError(f"invalid {split} feature cache")
    return x, w


def forward_mask(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return meta["trial"].to_numpy() > .8 * mx


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1, eps=1e-8)


class TemplateModel:
    def __init__(self, device: torch.device):
        self.device = device

    def fit(self, x, w, subjects, tasks, mask):
        self.classes = np.unique(subjects[mask]).astype(int)
        self.tasks = sorted(np.unique(tasks[mask]).tolist())
        fit_x = torch.as_tensor(x[mask], dtype=torch.float32, device=self.device)
        fit_w = torch.as_tensor(w[mask], dtype=torch.float32, device=self.device)
        scaler_fit = torch.cat((fit_x, fit_w.reshape(-1, x.shape[1])), dim=0)
        self.scaler_mean = scaler_fit.mean(dim=0)
        self.scaler_scale = scaler_fit.std(dim=0, unbiased=False).clamp_min(1e-6)
        x_scaled = (torch.as_tensor(x, dtype=torch.float32, device=self.device) - self.scaler_mean) / self.scaler_scale
        w_scaled = (torch.as_tensor(w, dtype=torch.float32, device=self.device) - self.scaler_mean) / self.scaler_scale
        self.prototypes = {}
        for task in self.tasks:
            self.prototypes[task] = {}
            for subject in self.classes:
                ix = np.flatnonzero(mask & (tasks == task) & (subjects == subject))
                if not len(ix):
                    continue
                z = normalize(x_scaled[ix])
                center = normalize(z.mean(0, keepdim=True))[0]
                keep = torch.topk(z @ center, k=max(1, int(.8 * len(ix)))).indices
                robust = normalize(z[keep].mean(0, keepdim=True))[0]
                medoid = z[keep[torch.argmax(z[keep] @ robust)]]
                w_trial = w_scaled[ix].mean(dim=1)
                zw = normalize(w_trial)
                w_center = normalize(zw.mean(0, keepdim=True))[0]
                w_keep = torch.topk(zw @ w_center, k=max(1, int(.8 * len(ix)))).indices
                w_robust = normalize(zw[w_keep].mean(0, keepdim=True))[0]
                wm = zw[w_keep[torch.argmax(zw[w_keep] @ w_robust)]]
                self.prototypes[task][int(subject)] = (medoid, wm)
        return self

    def score(self, x, w, tasks):
        scores = np.full((len(x), len(self.classes)), np.nan, np.float32)
        margin = np.empty(len(x), np.float32)
        agreement = np.empty(len(x), np.float32)
        for task in sorted(np.unique(tasks)):
            pos = np.flatnonzero(tasks == task)
            available = [s for s in self.classes if task in self.prototypes and int(s) in self.prototypes[task]]
            if not available:
                raise ValueError(f"no templates for task {task}")
            p = torch.stack([self.prototypes[task][int(s)][0] for s in available])
            wp = torch.stack([self.prototypes[task][int(s)][1] for s in available])
            x_pos = (torch.as_tensor(x[pos], dtype=torch.float32, device=self.device) - self.scaler_mean) / self.scaler_scale
            w_pos = (torch.as_tensor(w[pos], dtype=torch.float32, device=self.device) - self.scaler_mean) / self.scaler_scale
            z = normalize(x_pos)
            zw_pair = normalize(w_pos.reshape(-1, 2, x.shape[1]))
            full = z @ p.T
            ws = (zw_pair[:, 0] @ wp.T + zw_pair[:, 1] @ wp.T) / 2
            # Window agreement is the mean of the two per-window top-class scores.
            best = full.argmax(1)
            agree = torch.stack([zw_pair[:, 0] @ wp.T, zw_pair[:, 1] @ wp.T], 1)
            agree = agree[torch.arange(len(pos), device=self.device), :, best].mean(1)
            scores[pos[:, None], np.array([self.classes.tolist().index(s) for s in available])[None, :]] = full.cpu().numpy()
            top = full.topk(min(2, full.shape[1]), 1).values
            margin[pos] = (top[:, 0] - (top[:, 1] if top.shape[1] > 1 else -1)).cpu().numpy()
            agreement[pos] = agree.cpu().numpy()
            # Keep window scores available for prototype_window without another pass.
            scores[pos] = full.cpu().numpy() if len(available) == len(self.classes) else scores[pos]
            if len(available) < len(self.classes):
                for j, s in enumerate(available):
                    scores[pos, np.where(self.classes == s)[0][0]] = full[:, j].cpu().numpy()
            self._window_scores = getattr(self, "_window_scores", {})
            self._window_scores.update({int(i): ws[k].cpu().numpy() for k, i in enumerate(pos)})
        if not np.isfinite(scores).all():
            raise ValueError("missing template scores")
        return scores, margin, agreement

    def predict(self, x, w, tasks):
        scores, margin, agreement = self.score(x, w, tasks)
        # Window scores are recomputed by class and task to avoid session/batch state.
        wscores = np.zeros_like(scores)
        for task in sorted(np.unique(tasks)):
            pos = np.flatnonzero(tasks == task)
            for j, subject in enumerate(self.classes):
                if int(subject) not in self.prototypes[task]:
                    continue
                wp = self.prototypes[task][int(subject)][1]
                w_pos = (torch.as_tensor(w[pos], dtype=torch.float32, device=self.device) - self.scaler_mean) / self.scaler_scale
                zw = normalize(w_pos.reshape(-1, 2, x.shape[1]))
                value = ((zw[:, 0] @ wp) + (zw[:, 1] @ wp)).cpu().numpy() / 2
                wscores[pos, j] = value
        return {"prototype_only": scores, "prototype_margin": scores, "prototype_window": wscores}, margin, agreement


def confidence(scores, margin, agreement, method):
    if method == "prototype_only":
        return scores.max(1)
    if method == "prototype_margin":
        return margin
    return .5 * scores.max(1) + .5 * agreement


def choose_threshold(known, unknown):
    candidates = np.unique(known)
    best = None
    y = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    s = np.r_[known, unknown]
    for t in candidates:
        pred = s >= t
        row = (f1_score(y, pred, average="macro", zero_division=0),
               f1_score(y, pred, pos_label=0, zero_division=0),
               roc_auc_score(y, s))
        if best is None or row > best[0]:
            best = (row, float(t))
    return best[1]


def report(true, scores, class_scores, known_n, threshold, classes):
    conf = scores
    pred = classes[np.argmax(class_scores[:known_n], 1)].copy()
    pred[conf[:known_n] < threshold] = 0
    unknown_pred = classes[np.argmax(class_scores[known_n:], 1)].copy()
    unknown_pred[conf[known_n:] < threshold] = 0
    y = np.r_[np.ones(known_n, int), np.zeros(len(scores) - known_n, int)]
    accepted = conf >= threshold
    all_true = np.r_[true, np.zeros(len(scores) - known_n, int)]
    all_pred = np.r_[pred, unknown_pred]
    return {"AUC": float(roc_auc_score(y, conf)), "identity_macro_F1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
            "detection_macro_F1": float(f1_score(y, accepted, average="macro", zero_division=0)),
            "unknown_F1": float(f1_score(y, accepted, pos_label=0, zero_division=0)),
            "U": float(np.mean(~accepted[known_n:])), "known_acceptance": float(np.mean(accepted[:known_n])),
            "C": float(np.mean(pred == true)), "threshold": float(threshold), "n_known": int(known_n), "n_unknown": int(len(scores)-known_n)}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    OUT.mkdir(parents=True, exist_ok=True)
    required = ["run.py", "metrics.json", "report.md", "test_candidate.csv", "oof_scores.npy"]
    if any((OUT / n).exists() for n in required[1:]):
        raise FileExistsError(f"existing output in {OUT}; refusing to overwrite")
    torch.manual_seed(42)
    device = torch.device("cuda")
    cache = resolve_cache()
    meta = pd.read_csv(cache / "train_metadata.csv")
    subjects, tasks = meta.subject.to_numpy(), meta.task.to_numpy()
    x, w = load_split(cache, "train", len(meta))
    forward = forward_mask(meta)
    all_subjects = np.unique(subjects)
    folds, oof = [], np.full((len(meta), len(all_subjects)), np.nan, np.float32)
    oof_margin = np.full(len(meta), np.nan, np.float32)
    oof_agreement = np.full(len(meta), np.nan, np.float32)
    rng = np.random.default_rng(42)
    for fold, held in enumerate(np.array_split(rng.permutation(all_subjects), 5), 1):
        enrolled = np.setdiff1d(all_subjects, held)
        inner_known, inner_unknown = {m: [] for m in METHODS}, {m: [] for m in METHODS}
        for pseudo in np.array_split(rng.permutation(enrolled), 2):
            registered = np.isin(subjects, np.setdiff1d(enrolled, pseudo))
            fit = registered & ~forward
            ki = np.flatnonzero(registered & forward)
            ui = np.flatnonzero(np.isin(subjects, pseudo) & ~forward)
            model = TemplateModel(device).fit(x, w, subjects, tasks, fit)
            pred, mar, agr = model.predict(x[np.r_[ki, ui]], w[np.r_[ki, ui]], tasks[np.r_[ki, ui]])
            for method in METHODS:
                sc = confidence(pred[method], mar, agr, method)
                inner_known[method].append(sc[:len(ki)]); inner_unknown[method].append(sc[len(ki):])
        thresholds = {m: choose_threshold(np.concatenate(inner_known[m]), np.concatenate(inner_unknown[m])) for m in METHODS}
        fit = np.isin(subjects, enrolled) & ~forward
        ki = np.flatnonzero(np.isin(subjects, enrolled) & forward)
        ui = np.flatnonzero(np.isin(subjects, held))
        model = TemplateModel(device).fit(x, w, subjects, tasks, fit)
        pred, mar, agr = model.predict(x[np.r_[ki, ui]], w[np.r_[ki, ui]], tasks[np.r_[ki, ui]])
        fold_result = {"outer_fold": fold, "held_subjects": held.tolist(), "thresholds": thresholds, "methods": {}}
        for method in METHODS:
            sc = confidence(pred[method], mar, agr, method)
            fold_result["methods"][method] = report(subjects[ki], sc, pred[method], len(ki), thresholds[method], model.classes)
        folds.append(fold_result)
        print(f"outer={fold} " + json.dumps(fold_result["methods"], ensure_ascii=False), flush=True)
    # Fresh blocked OOF for final all-Train threshold and identity audit.
    block = np.full(len(meta), -1, int)
    for ix in meta.groupby(["subject", "task"], sort=False).indices.values():
        ordered = ix[np.argsort(meta.iloc[ix].trial.to_numpy(), kind="stable")]
        for k, part in enumerate(np.array_split(ordered, 5)): block[part] = k
    for k in range(5):
        valid = block == k
        model = TemplateModel(device).fit(x, w, subjects, tasks, ~valid)
        pred, mar, agr = model.predict(x[valid], w[valid], tasks[valid])
        oof[valid] = pred["prototype_only"]
        oof_margin[valid] = mar
        oof_agreement[valid] = agr
    if not np.isfinite(oof).all(): raise RuntimeError("incomplete OOF")
    # Train all and Test candidate; session is intentionally not loaded until output is frozen.
    final_model = TemplateModel(device).fit(x, w, subjects, tasks, np.ones(len(meta), bool))
    test_meta = pd.read_csv(cache / "test_metadata.csv", usecols=["sample", "task"])
    tx, tw = load_split(cache, "test", len(test_meta))
    test_pred, tmar, tagr = final_model.predict(tx, tw, test_meta.task.to_numpy())
    selected = max(METHODS, key=lambda m: np.mean([r["methods"][m]["detection_macro_F1"] for r in folds]))
    oof_conf = confidence(oof, oof_margin, oof_agreement, selected)
    threshold = float(np.quantile(oof_conf, .05))
    test_conf = confidence(test_pred[selected], tmar, tagr, selected)
    labels = final_model.classes[np.argmax(test_pred[selected], 1)].copy(); labels[test_conf < threshold] = 0
    submission = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
    row = test_meta["sample"].map(pd.Series(np.arange(len(test_meta)), index=test_meta["sample"]))
    submission["Label"] = labels[row.to_numpy(dtype=int)]
    submission.to_csv(OUT / "test_candidate.csv", index=False)
    # only now inspect session for a post-generation audit, never inference.
    audit_meta = pd.read_csv(cache / "test_metadata.csv", usecols=["session"])
    audit = {str(k): int(np.sum(labels[v] == 0)) for k, v in audit_meta.groupby("session").indices.items()}
    np.save(OUT / "oof_scores.npy", oof)
    summary = {m: {k: float(np.mean([f["methods"][m][k] for f in folds])) for k in ("AUC", "identity_macro_F1", "detection_macro_F1", "unknown_F1", "U", "known_acceptance", "C")} for m in METHODS}
    result = {"protocol": {"outer": "strict 5-fold subject holdout; enrolled last 20% trial; held subjects entirely unknown", "inner": "2 pseudo-unknown subject groups; fit enrolled non-forward trials", "features": "COV_COLUMNS full and window2", "template": "per(subject,task) robust trimmed medoid on GPU; window template uses per-trial mean of two windows", "scaler": "fit-mask-only GPU equivalent of StandardScaler over full and window observations; shared by full/window template and query transform", "threshold_selection": "inner pseudo-unknown known/unknown forward labels only; outer known/unknown labels excluded", "session_used_for_inference": False, "ridge_baseline": {"AUC": 0.965632, "F1": 0.936207}, "report_metrics": "AUC and identity macro-F1 are outer-fold means; Test AUC/F1 unavailable without labels", "gpu": torch.cuda.get_device_name(0)}, "folds": folds, "summary": summary, "selected_method": selected, "candidate": {"threshold": threshold, "test_rows": len(labels), "rejected": int(np.sum(labels == 0)), "session_post_output_audit": audit}}
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Enrollment template session-robust v1 scaled 实验报告", "", "严格外层五折 subject holdout；每个 outer fold 仅使用 fit mask 的 full/window 特征拟合 GPU StandardScaler，模板和测试使用同一 scaler。窗口模板按 trial 两窗口平均后进行 robust medoid 聚合；Train/Test 推理不读取 session。", "", "|决策|AUC|identity macro-F1|detection macro-F1|unknown-F1|U|known acceptance|C|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        r = summary[m]; lines.append(f"|{m}|{r['AUC']:.6f}|{r['identity_macro_F1']:.6f}|{r['detection_macro_F1']:.6f}|{r['unknown_F1']:.6f}|{r['U']:.6f}|{r['known_acceptance']:.6f}|{r['C']:.6f}|")
    lines += ["", f"选择方法：{selected}。输出候选 {len(labels)} 行，拒识 {int(np.sum(labels == 0))} 行。", "阈值来自全量 Train 的 fresh blocked OOF 已知置信度 5% 分位；Test 无真值，因此 Test AUC/F1 不可计算。", "与批准 Ridge 基线 AUC=0.965632、F1=0.936207 比较：本实验各方法 outer-fold 平均 AUC/F1（identity macro-F1）见上表，仍低于该基线。", "Session 仅在候选写出后做拒识计数审计，未参与模板、阈值、决策或后处理。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
