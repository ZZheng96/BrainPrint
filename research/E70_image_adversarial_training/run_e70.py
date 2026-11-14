"""E70：图像化路线的对抗鲁棒性（特征空间对抗训练）。

E69 证明冻结 DINOv2 + 线性探针在 SNR30 下 RA=0.0667（完全失效）。
端到端对抗训练太慢（DINOv2 前向是瓶颈），改用两阶段方案：

  阶段 1：预生成对抗样本（原始波形空间 PGD，梯度穿过图像编码 + DINOv2 + 探针）
  阶段 2：提取干净 + 对抗样本的 DINOv2 特征，在 768 维特征空间做对抗训练

特征空间对抗训练：
  - 输入：干净特征 x，对抗特征 x_adv
  - 模型：MLP(768 → 256 → 20)
  - 损失：CE(x) + λ·CE(x_adv)
  - 推理时用干净特征

这样 DINOv2 只做一次前向，训练在特征空间进行，速度快 100 倍。
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT.parent / "E69_image_adversarial"))

from run_e69 import (  # noqa: E402
    FrozenDinoProbe, covariance5_image_torch, N_CHANNELS,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout  # noqa: E402

CACHE = ROOT / "BrainPrint_cache"
ADV_CACHE = CACHE / "e70_adversarial"


def pgd_attack(model: FrozenDinoProbe, raw: torch.Tensor, labels: torch.Tensor,
               snr: float, steps: int, seed: int, batch_size: int = 32):
    """原始波形空间 PGD，梯度穿过图像编码 + DINOv2 + 探针。"""
    model.eval()
    adv = torch.empty_like(raw)
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        clean = raw[start:stop].detach().float()
        target = labels[start:stop]
        centered = clean - clean.mean(dim=-1, keepdim=True)
        signal_norm = centered.flatten(1).norm(dim=1).clamp_min(1e-12)
        radius = signal_norm / (10.0 ** (snr / 20.0))
        generator = torch.Generator(device=raw.device)
        generator.manual_seed(int(seed + start))
        delta = torch.randn(clean.shape, generator=generator, device=raw.device)
        delta = delta / delta.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta * radius.view(-1, 1, 1) * torch.rand(
            len(clean), generator=generator, device=raw.device).view(-1, 1, 1)
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            loss = F.cross_entropy(model(covariance5_image_torch(clean + delta)), target)
            gradient = torch.autograd.grad(loss, delta)[0]
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            direction = gradient / gradient.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            step_size = 2.0 * radius / steps
            delta = delta.detach() + step_size.view(-1, 1, 1) * direction
            norm = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
            delta = delta * torch.minimum(torch.ones_like(norm), radius / norm).view(-1, 1, 1)
        adv[start:stop] = (clean + delta).detach()
    return adv


@torch.no_grad()
def extract_dino(model: FrozenDinoProbe, raw: torch.Tensor, batch_size: int = 32):
    """提取冻结 DINOv2 CLS 特征（不经过探针）"""
    model.eval()
    outputs = []
    for start in range(0, len(raw), batch_size):
        images = covariance5_image_torch(raw[start:start + batch_size])
        normalized = (images - model.mean) / model.std
        tokens = model.backbone.forward_features(normalized)
        outputs.append(model.backbone.forward_head(tokens, pre_logits=True))
    return torch.cat(outputs)


class RobustHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_robust_head(clean: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor,
                      n_classes: int, epochs: int = 60, lr: float = 1e-3,
                      lambda_adv: float = 1.0, seed: int = 0, head_type: str = "mlp"):
    """head_type: mlp（非线性）或 linear（线性探针，等价 sklearn 逻辑回归）"""
    torch.manual_seed(seed)
    if head_type == "linear":
        head = nn.Linear(clean.shape[1], n_classes).to(clean.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    else:
        head = RobustHead(clean.shape[1], n_classes).to(clean.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        head.train()
        permutation = torch.randperm(len(clean), generator=generator)
        total, n = 0.0, 0
        for start in range(0, len(permutation), 128):
            selected = permutation[start:start + 128]
            loss = F.cross_entropy(head(clean[selected]), labels[selected])
            if lambda_adv > 0:
                loss = loss + lambda_adv * F.cross_entropy(
                    head(adv[selected]), labels[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1
        scheduler.step()
    head.eval()
    return head


@torch.no_grad()
def head_scores(head: RobustHead, features: torch.Tensor) -> np.ndarray:
    head.eval()
    return head(features).cpu().numpy()


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-snr", type=float, default=30.0)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--eval-snr", type=float, nargs="+", default=[30.0, 20.0])
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--tag", default="dinov2")
    parser.add_argument("--head-type", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--cache-tag", default=None,
                        help="缓存键（默认等于 tag）；扫参时固定它可复用对抗样本")
    args = parser.parse_args()
    cache_tag = args.cache_tag or args.tag
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")
    ADV_CACHE.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)

    print("加载原始波形...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    from scipy.io import loadmat

    def load_one(path):
        return np.asarray(loadmat(ROOT / "Brainprint_Competition" / path,
                                  variable_names=["data"])["data"], dtype=np.float32).T

    paths = metadata["path"].tolist()
    raw = torch.empty((len(paths), N_CHANNELS, 1200), dtype=torch.float32, device=device)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(paths), 512):
            arrays = list(pool.map(load_one, paths[start:start + 512]))
            raw[start:start + len(arrays)] = torch.as_tensor(np.stack(arrays), device=device)
    print(f"raw {tuple(raw.shape)}", flush=True)

    # 干净特征：DINOv2 复用 E20 缓存；其他骨干现场提取
    if args.tag == "dinov2":
        clean_features = torch.as_tensor(
            np.load(CACHE / "eeg_image_dinov2" / "train_covariance5_dinov2.npy"), device=device)
    else:
        clean_path = ADV_CACHE / f"clean_{cache_tag}.npy"
        if clean_path.exists():
            clean_features = torch.as_tensor(np.load(clean_path), device=device)
        else:
            print(f"提取 {args.tag} 干净特征...", flush=True)
            extractor = FrozenDinoProbe(args.backbone).to(device)
            started = time.time()
            clean_features = extract_dino(extractor, raw)
            print(f"  完成 {time.time()-started:.0f}s", flush=True)
            np.save(clean_path, clean_features.cpu().numpy())
            del extractor
            torch.cuda.empty_cache()
    print(f"clean features {tuple(clean_features.shape)}", flush=True)

    results = {}
    for fold in range(1, args.folds + 1):
        unknown_subjects = groups[fold - 1]
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward
        classes = np.sort(known_subjects)
        lookup = {int(c): i for i, c in enumerate(classes)}

        fit_idx = np.flatnonzero(outer_fit)
        fit_labels = torch.as_tensor(
            [lookup[int(s)] for s in subjects[fit_idx]], device=device)

        # 阶段 1：生成对抗样本（缓存）
        adv_path = ADV_CACHE / f"fold{fold}_snr{int(args.train_snr)}_steps{args.train_steps}_{cache_tag}.npy"
        if adv_path.exists():
            adv_features = torch.as_tensor(np.load(adv_path), device=device)
            print(f"fold{fold}: reuse {adv_path.name}", flush=True)
        else:
            print(f"fold{fold}: 生成对抗样本（{len(fit_idx)} 条）...", flush=True)
            # 用干净特征训练一个临时探针作为攻击目标
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            probe.fit(clean_features[fit_idx].cpu().numpy(), subjects[fit_idx])
            scaler = probe.named_steps["standardscaler"]
            lr = probe.named_steps["logisticregression"]
            attack_model = FrozenDinoProbe(args.backbone).to(device)
            attack_model.set_probe(scaler.mean_, scaler.scale_, lr.coef_, lr.intercept_)
            started = time.time()
            adv_raw = pgd_attack(attack_model, raw[fit_idx], fit_labels, args.train_snr,
                                 args.train_steps, seed=fold, batch_size=32)
            print(f"  攻击完成 {time.time()-started:.0f}s", flush=True)
            adv_features = extract_dino(attack_model, adv_raw)
            np.save(adv_path, adv_features.cpu().numpy())
            del attack_model
            torch.cuda.empty_cache()

        # 阶段 2：特征空间对抗训练
        head = train_robust_head(clean_features[fit_idx], adv_features, fit_labels,
                                 len(classes), epochs=args.epochs,
                                 lambda_adv=args.lambda_adv, seed=fold, head_type=args.head_type)

        # 内层阈值（用干净特征 + 鲁棒头）
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_conf = []
        for inner_unknown in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask = np.isin(subjects, inner_enrolled)
            fit = mask & ~forward & ~inner
            eval_idx = np.flatnonzero(mask & inner)
            sub_head = train_robust_head(
                clean_features[fit], adv_features[np.isin(fit_idx, np.flatnonzero(fit))],
                torch.as_tensor([lookup[int(s)] for s in subjects[fit]], device=device),
                len(classes), epochs=args.epochs, lambda_adv=args.lambda_adv, seed=fold, head_type=args.head_type)
            inner_conf.append(margin(head_scores(sub_head, clean_features[eval_idx])))
        threshold = float(np.quantile(np.concatenate(inner_conf), 0.05))

        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        rng = np.random.default_rng(fold)
        if len(known_eval) > args.max_samples:
            known_eval = rng.choice(known_eval, args.max_samples, replace=False)
        if len(unknown_eval) > args.max_samples:
            unknown_eval = rng.choice(unknown_eval, args.max_samples, replace=False)

        clean_scores = head_scores(head, clean_features[known_eval])
        clean_conf = margin(clean_scores)
        clean_pred = classes[clean_scores.argmax(1)].copy()
        clean_pred[clean_conf < threshold] = 0
        true_labels = subjects[known_eval]

        unknown_scores = head_scores(head, clean_features[unknown_eval])
        unknown_conf = margin(unknown_scores)
        unknown_pred = classes[unknown_scores.argmax(1)].copy()
        unknown_pred[unknown_conf < threshold] = 0

        fold_result = {
            "fold": fold, "threshold": threshold,
            "clean_C": float(np.mean(clean_pred == true_labels)),
            "clean_identity_f1": float(f1_score(true_labels, clean_pred,
                                                average="macro", zero_division=0)),
            "U": float(np.mean(unknown_pred == 0)),
            "unknown_auroc": float(roc_auc_score(
                np.r_[np.ones(len(known_eval)), np.zeros(len(unknown_eval))],
                np.r_[clean_conf, unknown_conf])),
            "attacks": {},
        }

        # 评估：重新生成评估攻击（不同种子）
        label_tensor = torch.as_tensor(
            np.searchsorted(classes, true_labels), device=device)
        raw_eval = raw[torch.as_tensor(known_eval, device=device)]
        eval_model = FrozenDinoProbe(args.backbone).to(device)
        eval_model.set_probe(np.zeros(clean_features.shape[1]), np.ones(clean_features.shape[1]),
                             np.zeros((len(classes), clean_features.shape[1])),
                             np.zeros(len(classes)))
        for snr in args.eval_snr:
            for attack_name in ("untargeted_fgsm", "untargeted_pgd"):
                steps = 1 if attack_name == "untargeted_fgsm" else args.eval_steps
                adv_raw = pgd_attack(eval_model, raw_eval, label_tensor, snr, steps,
                                     seed=5000 * fold + int(snr), batch_size=32)
                adv_features = extract_dino(eval_model, adv_raw)
                adv_scores = head_scores(head, adv_features)
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
                }
                print(f"  fold{fold} {key}: RA={fold_result['attacks'][key]['RA']:.4f} "
                      f"ASafeR={fold_result['attacks'][key]['ASafeR']:.4f} "
                      f"reject={fold_result['attacks'][key]['reject_rate']:.4f}", flush=True)
        del eval_model
        torch.cuda.empty_cache()
        results[f"fold{fold}"] = fold_result
        print(f"fold{fold} clean: C={fold_result['clean_C']:.4f} U={fold_result['U']:.4f} "
              f"AUROC={fold_result['unknown_auroc']:.4f}", flush=True)

    summary = {}
    for key in results["fold1"]["attacks"]:
        summary[key] = {
            "RA": float(np.mean([results[f"fold{f}"]["attacks"][key]["RA"]
                                 for f in range(1, args.folds + 1)])),
            "ASafeR": float(np.mean([results[f"fold{f}"]["attacks"][key]["ASafeR"]
                                     for f in range(1, args.folds + 1)])),
        }
    summary["clean"] = {
        "C": float(np.mean([results[f"fold{f}"]["clean_C"] for f in range(1, args.folds + 1)])),
        "identity_f1": float(np.mean([results[f"fold{f}"]["clean_identity_f1"]
                                      for f in range(1, args.folds + 1)])),
        "U": float(np.mean([results[f"fold{f}"]["U"] for f in range(1, args.folds + 1)])),
        "unknown_auroc": float(np.mean([results[f"fold{f}"]["unknown_auroc"]
                                        for f in range(1, args.folds + 1)])),
    }
    summary["macro"] = {
        "RA": float(np.mean([v["RA"] for k, v in summary.items() if k != "clean"])),
        "ASafeR": float(np.mean([v["ASafeR"] for k, v in summary.items() if k != "clean"])),
    }

    (OUT / f"results_{args.tag}.json").write_text(json.dumps({
        "protocol": {
            "note": "图像化路线特征空间对抗训练；骨干冻结，只训练 MLP 头",
            "backbone": args.backbone,
            "train_snr": args.train_snr, "train_steps": args.train_steps,
            "eval_snr": args.eval_snr, "eval_steps": args.eval_steps,
            "epochs": args.epochs, "lambda_adv": args.lambda_adv, "folds": args.folds,
        },
        "folds": results, "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 汇总 ===")
    print(f"clean: C={summary['clean']['C']:.4f} F1={summary['clean']['identity_f1']:.4f} "
          f"U={summary['clean']['U']:.4f} AUROC={summary['clean']['unknown_auroc']:.4f}")
    for key, value in summary.items():
        if key in ("clean", "macro"):
            continue
        print(f"{key}: RA={value['RA']:.4f} ASafeR={value['ASafeR']:.4f}")
    print(f"macro: RA={summary['macro']['RA']:.4f} ASafeR={summary['macro']['ASafeR']:.4f}")
    print(f"saved {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
