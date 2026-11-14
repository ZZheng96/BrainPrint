"""E94: Train-only raw EEG robust identity classifier with pseudo-unknown rejection."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
from run_e36 import load_config as load_e36_config, train_metadata  # noqa: E402
from run_attacks import load_all_raw  # noqa: E402

CONFIG = OUT / "config.json"
RESULTS = OUT / "results.json"
REPORT = OUT / "report.md"
CHECKPOINT = OUT / "model_checkpoint.pt"

class EEGNet1d(nn.Module):
    def __init__(self, channels, classes, width):
        super().__init__()
        self.drop = nn.Dropout2d(0.10)
        self.net = nn.Sequential(
            nn.Conv1d(channels, width, 9, 2, 4), nn.BatchNorm1d(width), nn.ELU(),
            nn.Conv1d(width, width * 2, 9, 2, 4), nn.BatchNorm1d(width * 2), nn.ELU(),
            nn.Conv1d(width * 2, width * 4, 9, 2, 4), nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.Conv1d(width * 4, width * 4, 9, 2, 4), nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(width * 4, classes)
    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        x = x / x.flatten(1).std(1).view(-1, 1, 1).clamp_min(1e-6)
        return self.head(self.net(self.drop(x)).squeeze(-1))

def augment(x, cfg, generator):
    if cfg["model"]["channel_dropout"]:
        mask = (torch.rand((len(x), x.shape[1], 1), device=x.device, generator=generator) > cfg["model"]["channel_dropout"]).to(x.dtype)
        x = x * mask
    shift = int(cfg["training"]["shift_samples"])
    if shift:
        offsets = torch.randint(-shift, shift + 1, (len(x),), device=x.device, generator=generator)
        x = torch.stack([torch.roll(v, int(s.item()), -1) for v, s in zip(x, offsets)])
    x = x * (1.0 + float(cfg["training"]["amplitude_std"]) * torch.randn((len(x), 1, 1), device=x.device, generator=generator))
    return x

def pgd(model, clean, labels, cfg, generator):
    radius = clean.flatten(1).norm(2, 1).clamp_min(1e-6) * float(cfg["training"]["pgd_epsilon_ratio"])
    delta = torch.randn(clean.shape, device=clean.device, generator=generator)
    delta = delta / delta.flatten(1).norm(2, 1).view(-1, 1, 1).clamp_min(1e-6) * radius.view(-1, 1, 1)
    steps = int(cfg["training"]["pgd_steps"])
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        loss = F.cross_entropy(model(clean + delta), labels)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + radius.view(-1, 1, 1) / max(steps, 1) * grad.sign()
        delta = delta * torch.minimum(torch.ones_like(radius), radius / delta.flatten(1).norm(2, 1).clamp_min(1e-6)).view(-1, 1, 1)
    return (clean + delta).detach()

def train_fold(raw, fit, labels, classes, cfg, seed, device, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    model = EEGNet1d(18, len(classes), int(cfg["model"]["width"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    rng = np.random.default_rng(seed); gen = torch.Generator(device=device); gen.manual_seed(seed)
    batch = int(cfg["training"]["batch_size"])
    fit_device = torch.as_tensor(fit, device=device)
    for epoch in range(epochs):
        model.train(); order = rng.permutation(len(fit))
        for start in range(0, len(order), batch):
            sel = torch.as_tensor(order[start:start + batch], device=device)
            x, y = augment(raw[fit_device[sel]], cfg, gen), labels[sel]
            adv = pgd(model, x, y, cfg, gen)
            loss = 0.5 * F.cross_entropy(model(x), y) + 0.5 * F.cross_entropy(model(adv), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"    epoch={epoch + 1}/{epochs} loss={float(loss):.4f}", flush=True)
    return model

def predict(model, raw, indices, device):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            sel = torch.as_tensor(indices[start:start + 512], device=device)
            out.append(torch.softmax(model(raw[sel]), 1).cpu().numpy())
    return np.concatenate(out)

def choose_threshold(known, unknown):
    values = np.unique(np.r_[known, unknown])
    best = (-1.0, float(np.quantile(known, .05)))
    y = np.r_[np.ones(len(known), int), np.zeros(len(unknown), int)]
    for t in values:
        score = f1_score(y, np.r_[known >= t, unknown >= t].astype(int), zero_division=0)
        if score > best[0]: best = (float(score), float(t))
    return best[1]

def eer(known, unknown):
    y = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    fpr, tpr, _ = roc_curve(y, np.r_[known, unknown])
    return float(np.min(np.maximum(fpr, 1 - tpr)))

def evaluate(pk, true, pu, threshold, classes):
    ck, cu = pk.max(1), pu.max(1); dk, du = classes[pk.argmax(1)], classes[pu.argmax(1)]
    pred_k = np.where(ck >= threshold, dk, 0); pred_u = np.where(cu >= threshold, du, 0)
    y = np.r_[np.ones(len(ck), int), np.zeros(len(cu), int)]; accepted = np.r_[pred_k != 0, pred_u != 0]
    return {"N_known": int(len(ck)), "N_unknown": int(len(cu)), "AUC": float(roc_auc_score(y, np.r_[ck, cu])),
            "F1": float(f1_score(true, pred_k, labels=classes, average="macro", zero_division=0)),
            "known_acceptance": float(np.mean(pred_k != 0)), "unknown_FAR": float(np.mean(pred_u != 0)),
            "unknown_F1_pos_label_0": float(f1_score(y, accepted, pos_label=0, zero_division=0)), "EER": eer(ck, cu),
            "C": float(np.mean(pred_k == true)), "U": float(np.mean(pred_u == 0)), "threshold": float(threshold)}

def run(smoke):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")); e36 = load_e36_config()
    if not torch.cuda.is_available(): raise RuntimeError("E94 requires CUDA GPU")
    device = torch.device("cuda"); meta = train_metadata(e36); raw = load_all_raw(meta, e36, device)
    subjects = meta.subject.to_numpy(int); all_subjects = np.unique(subjects); trial = meta.trial.to_numpy(float)
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float); frac = trial / np.maximum(maximum, 1)
    groups = np.array_split(np.random.default_rng(cfg["protocol"]["outer_seed"]).permutation(all_subjects), cfg["protocol"]["outer_folds"])
    if smoke: groups = groups[:1]
    folds = []; last_model = None
    for oi, outer_unknown in enumerate(groups, 1):
        known = np.setdiff1d(all_subjects, outer_unknown); inners = np.array_split(np.random.default_rng(100 + oi).permutation(known), 2)
        if smoke: inners = inners[:1]
        inner_runs = []
        for ii, inner_unknown in enumerate(inners, 1):
            enrolled = np.setdiff1d(known, inner_unknown); fit = np.flatnonzero(np.isin(subjects, enrolled) & (frac <= .60))
            cal_k = np.flatnonzero(np.isin(subjects, enrolled) & (frac > .60) & (frac <= .80)); cal_u = np.flatnonzero(np.isin(subjects, inner_unknown) & (frac <= .80))
            lookup = {s: i for i, s in enumerate(enrolled)}; labels = torch.as_tensor([lookup[s] for s in subjects[fit]], device=device, dtype=torch.long)
            model = train_fold(raw, fit, labels, enrolled, cfg, cfg["seed"] + oi * 100 + ii, device, cfg["training"]["smoke_epochs"] if smoke else cfg["training"]["epochs"])
            pk, pu = predict(model, raw, cal_k, device), predict(model, raw, cal_u, device); threshold = choose_threshold(pk.max(1), pu.max(1))
            inner_runs.append({"threshold": threshold, "enrolled": enrolled.tolist(), "N_fit": int(len(fit))})
            del model; torch.cuda.empty_cache()
        outer_fit = np.flatnonzero(np.isin(subjects, known) & (frac <= .80)); outer_k = np.flatnonzero(np.isin(subjects, known) & (frac > .80)); outer_u = np.flatnonzero(np.isin(subjects, outer_unknown))
        lookup = {s: i for i, s in enumerate(known)}; labels = torch.as_tensor([lookup[s] for s in subjects[outer_fit]], device=device, dtype=torch.long)
        model = train_fold(raw, outer_fit, labels, known, cfg, cfg["seed"] + oi * 1000, device, cfg["training"]["smoke_epochs"] if smoke else cfg["training"]["epochs"])
        threshold = float(np.mean([r["threshold"] for r in inner_runs])); metrics = evaluate(predict(model, raw, outer_k, device), subjects[outer_k], predict(model, raw, outer_u, device), threshold, known)
        folds.append({"outer_fold": oi, "outer_unknown": outer_unknown.tolist(), "inner": inner_runs, "metrics": metrics}); last_model = model.state_dict()
        print(f"outer={oi} AUC={metrics['AUC']:.4f} F1={metrics['F1']:.4f} C={metrics['C']:.4f} U={metrics['U']:.4f}", flush=True)
    summary = {k: float(np.mean([f["metrics"][k] for f in folds])) for k in folds[0]["metrics"] if k not in ("N_known", "N_unknown")}; summary["N_known"] = sum(f["metrics"]["N_known"] for f in folds); summary["N_unknown"] = sum(f["metrics"]["N_unknown"] for f in folds)
    result = {"status": "complete_smoke" if smoke else "complete", "experiment": "E94", "mode": "smoke" if smoke else "full_5_outer_x_2_inner", "summary": summary, "folds": folds, "data": {"metadata": cfg["data"]["train_metadata"], "test_accessed": False, "test_metrics": "N/A"}, "protocol_checks": {"train_only": True, "gpu": torch.cuda.get_device_name(0), "pseudo_unknown_calibration": True, "test_accessed": False}}
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); torch.save({"experiment": "E94", "state_dict": last_model, "config": cfg}, CHECKPOINT)
    write_report(result); return result

def write_report(result):
    s = result["summary"]; lines = ["# E94 Train-only 原始EEG鲁棒身份+拒识", "", "仅访问 Train/session1 的 18 通道原始 EEG；项目 Test 目录、test.csv、test_metadata.csv 均未访问。项目 Test 指标：N/A。", "", "## 当前结果", "", "| AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append("| " + " | ".join(f"{s[k]:.6f}" for k in ("AUC", "F1", "known_acceptance", "unknown_FAR", "unknown_F1_pos_label_0", "EER", "C", "U")) + " |")
    lines += ["", "## 方法与边界", "", "EEGNet1d 在 GPU 训练；训练增强包含通道 dropout、循环时间平移、幅度扰动和低强度原始波形 PGD。outer 留出被试，inner 再留 pseudo-unknown 被试；阈值由 inner known/pseudo-unknown confidence 校准。C 为已知身份拒识后的正确率，U 为未知拒识率。", "", "模型 checkpoint：`model_checkpoint.pt`；结构化结果：`results.json`。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true"); a = ap.parse_args()
    if a.smoke == a.full: ap.error("choose exactly one of --smoke or --full")
    print(json.dumps(run(a.smoke)["summary"], ensure_ascii=False, indent=2))
