"""E69：图像化路线的对抗鲁棒性（补 RA / ASafeR 两项门禁）。

图像化路线此前从未测过对抗攻击。本实验在原始波形空间做 L2 FGSM/PGD，
梯度穿过「波形 → 可微图像编码 → 冻结 DINOv2 → 线性探针」整条链。

关键设计：
  1. 可微图像编码：协方差图的对数特征值分解在 PyTorch 中可微
  2. 冻结 DINOv2 前向可微（参数冻结，但梯度可回传到输入图像）
  3. 线性探针（sklearn）转成等价的 torch 线性层，梯度可回传
  4. 攻击目标：最小化真实身份的 logit（非目标）/ 最大化目标身份（目标）

评估：
  RA     = 对抗样本正确识别原身份率
  ASafeR = 对抗样本「判对或拒识」率
  阈值来自内层伪未知，与 E20D 同协议
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.run_nested import forward_holdout, inner_holdout  # noqa: E402

CACHE = ROOT / "BrainPrint_cache"
FS = 300
N_CHANNELS = 18
BANDS = ((1, 8), (8, 13), (13, 30), (30, 45), (45, 90))
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------- 可微图像编码 ----------------

def bandpass_fft_torch(data: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """data: (B, C, T) -> 带通滤波（FFT 域掩码，可微）"""
    spectrum = torch.fft.rfft(data, dim=-1)
    freqs = torch.fft.rfftfreq(data.shape[-1], d=1.0 / FS, device=data.device)
    mask = ((freqs >= low) & (freqs <= high)).to(data.dtype)
    return torch.fft.irfft(spectrum * mask, n=data.shape[-1], dim=-1)


def log_cov_torch(segment: torch.Tensor, shrinkage: float = 0.05) -> torch.Tensor:
    """segment: (B, C, T) -> (B, C, C) 对数协方差（可微）"""
    centered = segment - segment.mean(dim=-1, keepdim=True)
    cov = centered @ centered.transpose(-1, -2) / max(segment.shape[-1] - 1, 1)
    eye = torch.eye(N_CHANNELS, device=segment.device, dtype=segment.dtype)
    target = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1) * eye
    cov = (1.0 - shrinkage) * cov + shrinkage * target
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.clamp_min(1e-12)
    log_cov = (eigenvectors * eigenvalues.log().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    return log_cov


def normalize_image_torch(image: torch.Tensor) -> torch.Tensor:
    """逐图 min-max 到 [0,1]（可微，带数值保护）"""
    flat = image.flatten(1)
    low = flat.min(dim=1).values.view(-1, 1, 1)
    high = flat.max(dim=1).values.view(-1, 1, 1)
    span = (high - low).clamp_min(1e-3)
    return ((image - low) / span).clamp(0.0, 1.0)


def _resize_index_torch(size_in: int, size_out: int, device) -> torch.Tensor:
    """复现 numpy 版 _resize_to 的最近邻索引：round(linspace(0, size_in-1, size_out))"""
    positions = torch.linspace(0, size_in - 1, size_out, device=device)
    return positions.round().long().clamp(0, size_in - 1)


def covariance5_image_torch(raw: torch.Tensor) -> torch.Tensor:
    """raw: (B, C, T) -> (B, 3, 224, 224) 与 E20 的 covariance5 同构（可微）"""
    centered = raw - raw.mean(dim=-1, keepdim=True)
    maps = []
    for low, high in BANDS:
        filtered = bandpass_fft_torch(centered, low, high)
        maps.append(normalize_image_torch(log_cov_torch(filtered)))
    maps = torch.stack(maps, dim=1)  # (B, 5, 18, 18)
    # 2×3 网格平铺，每格 112×74（与 numpy 版 _tile_5_bands 一致：
    # 先缩放到 cell_h×cell_h，再裁切到 cell_h×cell_w）
    cell_h, cell_w = IMG_SIZE // 2, IMG_SIZE // 3
    row_idx = _resize_index_torch(N_CHANNELS, cell_h, raw.device)
    col_idx = _resize_index_torch(N_CHANNELS, cell_h, raw.device)
    resized = maps[:, :, row_idx][:, :, :, col_idx][:, :, :, :cell_w]  # (B, 5, cell_h, cell_w)
    canvas = torch.zeros(len(raw), 1, cell_h * 2, cell_w * 3, device=raw.device, dtype=raw.dtype)
    for index in range(5):
        row, col = divmod(index, 3)
        canvas[:, :, row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = \
            resized[:, index:index + 1]
    padded = torch.zeros(len(raw), 1, IMG_SIZE, IMG_SIZE, device=raw.device, dtype=raw.dtype)
    padded[:, :, :canvas.shape[2], :canvas.shape[3]] = canvas
    return padded.repeat(1, 3, 1, 1)


# ---------------- 冻结 DINOv2 + 线性探针 ----------------

class FrozenDinoProbe(torch.nn.Module):
    """冻结 DINOv2/DINOv3 + 可微线性探针（权重从 sklearn 迁移）"""

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m"):
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0,
                                          img_size=IMG_SIZE)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.embed_dim = self.backbone.embed_dim
        self.model_name = model_name
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.probe = None

    def set_probe(self, scaler_mean, scaler_scale, coef, intercept):
        device = next(self.backbone.parameters()).device
        self.probe = torch.nn.Linear(self.embed_dim, coef.shape[0], bias=True).to(device)
        with torch.no_grad():
            self.probe.weight.copy_(torch.as_tensor(coef, dtype=torch.float32, device=device))
            self.probe.bias.copy_(torch.as_tensor(intercept, dtype=torch.float32, device=device))
        self.scaler_mean = torch.as_tensor(scaler_mean, dtype=torch.float32, device=device)
        self.scaler_scale = torch.as_tensor(scaler_scale, dtype=torch.float32, device=device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.mean) / self.std
        tokens = self.backbone.forward_features(normalized)
        cls = self.backbone.forward_head(tokens, pre_logits=True)
        scaled = (cls - self.scaler_mean) / self.scaler_scale
        return self.probe(scaled)


# ---------------- 攻击 ----------------

def raw_attack(model: FrozenDinoProbe, raw: torch.Tensor, labels: torch.Tensor,
               snr: float, attack_name: str, steps: int, seed: int,
               target: torch.Tensor | None = None, batch_size: int = 32):
    """原始波形空间 L2 FGSM/PGD，梯度穿过图像编码 + DINOv2 + 探针。"""
    targeted = attack_name == "targeted_pgd"
    scores_out = torch.empty((len(raw), model.probe.out_features), device=raw.device)
    realized = []
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        clean = raw[start:stop].detach().float()
        centered = clean - clean.mean(dim=-1, keepdim=True)
        signal_norm = centered.flatten(1).norm(dim=1).clamp_min(1e-12)
        radius = signal_norm / (10.0 ** (snr / 20.0))
        generator = torch.Generator(device=raw.device)
        generator.manual_seed(int(seed + start))
        delta = torch.zeros_like(clean)
        if steps > 1:
            delta = torch.randn(clean.shape, generator=generator, device=raw.device)
            delta = delta / delta.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            delta = delta * radius.view(-1, 1, 1) * torch.rand(
                len(clean), generator=generator, device=raw.device).view(-1, 1, 1)
        objective = target[start:stop] if targeted else labels[start:stop]
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            images = covariance5_image_torch(clean + delta)
            scores = model(images)
            loss = F.cross_entropy(scores / 0.05, objective)
            gradient = torch.autograd.grad(-loss if targeted else loss, delta)[0]
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite gradient")
            direction = gradient / gradient.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            step_size = radius if steps == 1 else 2.0 * radius / steps
            delta = delta.detach() + step_size.view(-1, 1, 1) * direction
            norm = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
            delta = delta * torch.minimum(torch.ones_like(norm), radius / norm).view(-1, 1, 1)
        with torch.no_grad():
            scores_out[start:stop] = model(covariance5_image_torch(clean + delta))
            norm = delta.flatten(1).norm(dim=1)
            realized.extend((20 * torch.log10(signal_norm / norm.clamp_min(1e-12))).cpu().tolist())
    return scores_out, {"realized_snr_db_mean": float(np.mean(realized))}


# ---------------- 评估 ----------------

def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, nargs="+", default=[30.0, 20.0])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")

    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)

    # 加载原始波形（GPU）
    print("加载 Train 原始波形...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    from scipy.io import loadmat

    def load_one(path):
        data = np.asarray(loadmat(ROOT / "Brainprint_Competition" / path,
                                  variable_names=["data"])["data"], dtype=np.float32)
        return data.T

    paths = metadata["path"].tolist()
    raw = torch.empty((len(paths), N_CHANNELS, 1200), dtype=torch.float32, device=device)
    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(paths), 512):
            arrays = list(pool.map(load_one, paths[start:start + 512]))
            raw[start:start + len(arrays)] = torch.as_tensor(np.stack(arrays), device=device)
    print(f"raw {tuple(raw.shape)} in {time.time()-started:.0f}s", flush=True)

    # 预提 DINOv2 特征（干净）
    print("提取干净 DINOv2 特征...", flush=True)
    dino_path = CACHE / "eeg_image_dinov2" / "train_covariance5_dinov2.npy"
    if dino_path.exists():
        features = np.load(dino_path)
        print(f"reuse {dino_path.name} {features.shape}", flush=True)
    else:
        raise SystemExit("缺少 train_covariance5_dinov2.npy，请先运行 E20")

    model = FrozenDinoProbe().to(device)
    results = {}
    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward

        # 内层阈值
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_conf = []
        for inner_unknown in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask = np.isin(subjects, inner_enrolled)
            fit = mask & ~forward & ~inner
            eval_idx = np.flatnonzero(mask & inner)
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            probe.fit(features[fit], subjects[fit])
            inner_conf.append(margin(probe.decision_function(features[eval_idx])))
        threshold = float(np.quantile(np.concatenate(inner_conf), 0.05))

        # 外层探针
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        probe.fit(features[outer_fit], subjects[outer_fit])
        classes = probe.classes_
        scaler = probe.named_steps["standardscaler"]
        lr = probe.named_steps["logisticregression"]
        model.set_probe(scaler.mean_, scaler.scale_, lr.coef_, lr.intercept_)

        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        # 限制样本数以控制时间
        rng = np.random.default_rng(fold)
        if len(known_eval) > args.max_samples:
            known_eval = rng.choice(known_eval, args.max_samples, replace=False)
        if len(unknown_eval) > args.max_samples:
            unknown_eval = rng.choice(unknown_eval, args.max_samples, replace=False)

        clean_scores = probe.decision_function(features[known_eval])
        clean_conf = margin(clean_scores)
        clean_pred = classes[clean_scores.argmax(1)].copy()
        clean_pred[clean_conf < threshold] = 0
        true_labels = subjects[known_eval]

        unknown_scores = probe.decision_function(features[unknown_eval])
        unknown_conf = margin(unknown_scores)
        unknown_pred = classes[unknown_scores.argmax(1)].copy()
        unknown_pred[unknown_conf < threshold] = 0

        fold_result = {
            "fold": fold,
            "threshold": threshold,
            "clean_C": float(np.mean(clean_pred == true_labels)),
            "clean_identity_f1": float(f1_score(true_labels, clean_pred, average="macro",
                                                zero_division=0)),
            "U": float(np.mean(unknown_pred == 0)),
            "unknown_auroc": float(roc_auc_score(
                np.r_[np.ones(len(known_eval)), np.zeros(len(unknown_eval))],
                np.r_[clean_conf, unknown_conf])),
            "attacks": {},
        }

        label_tensor = torch.as_tensor(
            np.searchsorted(classes, true_labels), device=device
        )
        raw_eval = raw[torch.as_tensor(known_eval, device=device)]
        for snr in args.snr:
            for attack_name in ("untargeted_fgsm", "untargeted_pgd"):
                steps = 1 if attack_name == "untargeted_fgsm" else args.steps
                adv_scores, diag = raw_attack(
                    model, raw_eval, label_tensor, snr, attack_name, steps,
                    seed=1000 * fold + int(snr), batch_size=args.batch_size,
                )
                adv_scores = adv_scores.cpu().numpy()
                adv_conf = margin(adv_scores)
                adv_pred = classes[adv_scores.argmax(1)].copy()
                adv_pred[adv_conf < threshold] = 0
                correct = adv_pred == true_labels
                rejected = adv_pred == 0
                key = f"{attack_name}_snr{int(snr)}"
                fold_result["attacks"][key] = {
                    "RA": float(np.mean(correct)),
                    "ASafeR": float(np.mean(correct | rejected)),
                    "direct_RA": float(np.mean(classes[adv_scores.argmax(1)] == true_labels)),
                    "reject_rate": float(np.mean(rejected)),
                    "realized_snr_db": diag["realized_snr_db_mean"],
                }
                print(f"  fold{fold} {key}: RA={fold_result['attacks'][key]['RA']:.4f} "
                      f"ASafeR={fold_result['attacks'][key]['ASafeR']:.4f} "
                      f"reject={fold_result['attacks'][key]['reject_rate']:.4f}", flush=True)
        results[f"fold{fold}"] = fold_result
        print(f"fold{fold} clean: C={fold_result['clean_C']:.4f} U={fold_result['U']:.4f} "
              f"AUROC={fold_result['unknown_auroc']:.4f}", flush=True)

    # 汇总
    summary = {}
    for key in results["fold1"]["attacks"]:
        summary[key] = {
            "RA": float(np.mean([results[f"fold{f}"]["attacks"][key]["RA"] for f in range(1, 6)])),
            "ASafeR": float(np.mean([results[f"fold{f}"]["attacks"][key]["ASafeR"]
                                     for f in range(1, 6)])),
            "direct_RA": float(np.mean([results[f"fold{f}"]["attacks"][key]["direct_RA"]
                                        for f in range(1, 6)])),
        }
    summary["clean"] = {
        "C": float(np.mean([results[f"fold{f}"]["clean_C"] for f in range(1, 6)])),
        "identity_f1": float(np.mean([results[f"fold{f}"]["clean_identity_f1"]
                                      for f in range(1, 6)])),
        "U": float(np.mean([results[f"fold{f}"]["U"] for f in range(1, 6)])),
        "unknown_auroc": float(np.mean([results[f"fold{f}"]["unknown_auroc"]
                                        for f in range(1, 6)])),
    }
    macro_RA = float(np.mean([v["RA"] for k, v in summary.items() if k != "clean"]))
    macro_ASafeR = float(np.mean([v["ASafeR"] for k, v in summary.items() if k != "clean"]))
    summary["macro"] = {"RA": macro_RA, "ASafeR": macro_ASafeR}

    (OUT / "results.json").write_text(json.dumps({
        "protocol": {
            "note": "图像化路线对抗鲁棒性；攻击在原始波形空间，梯度穿过图像编码+DINOv2+探针",
            "snr": args.snr, "steps": args.steps, "max_samples": args.max_samples,
        },
        "folds": results,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 汇总 ===")
    print(f"clean: C={summary['clean']['C']:.4f} F1={summary['clean']['identity_f1']:.4f} "
          f"U={summary['clean']['U']:.4f} AUROC={summary['clean']['unknown_auroc']:.4f}")
    for key, value in summary.items():
        if key in ("clean", "macro"):
            continue
        print(f"{key}: RA={value['RA']:.4f} ASafeR={value['ASafeR']:.4f} "
              f"direct_RA={value['direct_RA']:.4f}")
    print(f"macro: RA={macro_RA:.4f} ASafeR={macro_ASafeR:.4f}")
    print(f"saved {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
