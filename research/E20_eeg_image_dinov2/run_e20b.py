"""E20B：patch token 池化对比 CLS。

对 topomap / correlation 两种最优表示，比较：
  - CLS（基线）
  - patch 均值池化
  - patch 注意力池化（可学习，训练时用内层验证早停）
  - CLS + patch 均值拼接
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eeg_image_dinov2.backbone import FrozenDINOv2

CACHE = ROOT / "BrainPrint_cache"
OUT = ROOT / "research" / "E20_eeg_image_dinov2"
METHODS = ("topomap", "correlation")


def forward_mask(metadata: pd.DataFrame) -> np.ndarray:
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return metadata["trial"].to_numpy() > 0.8 * max_trial


@torch.no_grad()
def extract_tokens(
    images: np.ndarray, device: torch.device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """返回 CLS (N,D) 与 patch tokens (N,P,D)，tokens 存 float16 以节省内存。

    images 可为 memmap，按 batch 读取避免整体转 float32 占满内存。
    """
    model = FrozenDINOv2().to(device)
    n = len(images)
    cls_out = np.empty((n, model.embed_dim), dtype=np.float32)
    token_out = None
    for start in range(0, n, batch_size):
        chunk = np.asarray(images[start : start + batch_size], dtype=np.float32) / 255.0
        batch = torch.as_tensor(chunk, device=device)
        cls, tokens = model(batch, return_tokens=True)
        if token_out is None:
            token_out = np.empty(
                (n, tokens.shape[1], tokens.shape[2]), dtype=np.float16
            )
        cls_out[start : start + len(batch)] = cls.cpu().numpy()
        token_out[start : start + len(batch)] = tokens.cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    return cls_out, token_out


class AttentionPool(nn.Module):
    """可学习注意力池化 patch tokens。"""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.float()
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)


def train_attention_pool(
    tokens: np.ndarray,
    subjects: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
) -> np.ndarray:
    """在 fit 上训练注意力池化 + 线性分类，返回全部样本的池化特征。"""
    pool = AttentionPool(tokens.shape[-1]).to(device)
    classifier = nn.Linear(tokens.shape[-1], len(np.unique(subjects[fit_mask]))).to(device)
    optimizer = torch.optim.AdamW(
        list(pool.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=1e-4
    )
    classes = np.unique(subjects[fit_mask])
    lookup = {int(c): i for i, c in enumerate(classes)}
    fit_idx = np.flatnonzero(fit_mask)
    val_idx = np.flatnonzero(val_mask)
    best_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(epochs):
        pool.train(); classifier.train()
        permutation = np.random.default_rng(epoch).permutation(fit_idx)
        for start in range(0, len(permutation), 64):
            selected = permutation[start : start + 64]
            batch = torch.as_tensor(tokens[selected], device=device)
            target = torch.as_tensor(
                [lookup[int(s)] for s in subjects[selected]], device=device
            )
            logits = classifier(pool(batch))
            loss = F.cross_entropy(logits, target, label_smoothing=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        pool.eval(); classifier.eval()
        with torch.no_grad():
            val_batch = torch.as_tensor(tokens[val_idx], device=device)
            val_target = torch.as_tensor(
                [lookup[int(s)] for s in subjects[val_idx]], device=device
            )
            val_loss = float(F.cross_entropy(classifier(pool(val_batch)), val_target).item())
        if val_loss < best_loss - 1e-4:
            best_loss, patience = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in pool.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is not None:
        pool.load_state_dict(best_state)
    pool.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tokens), 128):
            batch = torch.as_tensor(tokens[start : start + 128], device=device)
            outputs.append(pool(batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def probe(features: np.ndarray, subjects: np.ndarray, forward: np.ndarray) -> dict:
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known = np.isin(subjects, np.setdiff1d(all_subjects, unknown_subjects))
        fit = known & ~forward
        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(features[fit], subjects[fit])
        known_scores = model.decision_function(features[known_eval])
        unknown_scores = model.decision_function(features[unknown_eval])
        classes = model.classes_

        def margin(scores):
            top = np.partition(scores, -2, axis=1)[:, -2:]
            return top[:, 1] - top[:, 0]

        known_conf, unknown_conf = margin(known_scores), margin(unknown_scores)
        threshold = float(np.quantile(known_conf, 0.05))
        known_pred = classes[known_scores.argmax(axis=1)].copy()
        known_pred[known_conf < threshold] = 0
        unknown_pred = classes[unknown_scores.argmax(axis=1)].copy()
        unknown_pred[unknown_conf < threshold] = 0
        detection_true = np.r_[np.ones(len(known_eval)), np.zeros(len(unknown_eval))]
        detection_conf = np.r_[known_conf, unknown_conf]
        rows.append({
            "fold": fold,
            "C": float(np.mean(known_pred == subjects[known_eval])),
            "identity_macro_f1": float(f1_score(
                subjects[known_eval], known_pred, average="macro", zero_division=0
            )),
            "U": float(np.mean(unknown_pred == 0)),
            "unknown_auroc": float(roc_auc_score(detection_true, detection_conf)),
            "unknown_f1": float(f1_score(
                detection_true, detection_conf >= threshold, pos_label=0, zero_division=0
            )),
        })
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("C", "identity_macro_f1", "U", "unknown_auroc", "unknown_f1")
    }
    return {"folds": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=OUT / "e20b_pooling.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    forward = forward_mask(metadata)
    all_subjects = np.unique(subjects)

    results = {}
    for method in args.methods:
        print(f"\n=== {method} ===", flush=True)
        images = np.load(CACHE / "eeg_image_dinov2" / f"train_{method}.npy", mmap_mode="r")
        started = time.time()
        cls, tokens = extract_tokens(images, device, args.batch_size)
        print(f"CLS {cls.shape} tokens {tokens.shape} in {time.time()-started:.0f}s", flush=True)
        np.save(CACHE / "eeg_image_dinov2" / f"train_{method}_tokens.npy", tokens)

        variants = {
            "cls": cls,
            "patch_mean": tokens.mean(axis=1, dtype=np.float32),
            "cls_patch_mean": np.concatenate(
                [cls, tokens.mean(axis=1, dtype=np.float32)], axis=1
            ),
        }
        method_results = {}
        for name, features in variants.items():
            method_results[name] = probe(features, subjects, forward)
            print(f"  {name}: {method_results[name]['summary']}", flush=True)

        # 注意力池化：每折单独训练，避免泄漏
        groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
        pooled = np.full((len(tokens), tokens.shape[-1]), np.nan, dtype=np.float32)
        for fold, unknown_subjects in enumerate(groups, 1):
            known = np.isin(subjects, np.setdiff1d(all_subjects, unknown_subjects))
            fit = known & ~forward
            # 内层验证：注册者中间 20% trial
            max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
            trial = metadata["trial"].to_numpy()
            inner = known & (trial > 0.6 * max_trial) & (trial <= 0.8 * max_trial)
            pooled[known] = train_attention_pool(
                tokens, subjects, fit, inner, device
            )[known]
            print(f"  attention_pool fold {fold} done", flush=True)
        method_results["attention_pool"] = probe(pooled, subjects, forward)
        print(f"  attention_pool: {method_results['attention_pool']['summary']}", flush=True)
        results[method] = method_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": {
            "outer": "5-fold subject holdout; 4 unknown, 16 registered",
            "known_eval": "last 20% trials per registered subject/task",
            "backbone": "frozen vit_base_patch14_dinov2.lvd142m, img_size=224",
            "classifier": "StandardScaler + LogisticRegression(C=1.0)",
            "attention_pool": "per-fold trained on fit trials, early stop on inner 60-80% trials",
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {args.output}", flush=True)


if __name__ == "__main__":
    main()
