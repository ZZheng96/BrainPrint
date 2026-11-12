"""E20J：DINOv2 vs DINOv3 冻结特征对比（同一 covariance5 图像）。

严格嵌套纯 EEG：外层 5 折留被试，内层 2 折伪未知标定阈值。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = ROOT / "research" / "E20_eeg_image_dinov2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenBackbone(nn.Module):
    def __init__(self, model_name: str, img_size: int = 224):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=img_size
        )
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.embed_dim = self.backbone.embed_dim
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: torch.Tensor, return_tokens: bool = False):
        normalized = (images - self.mean) / self.std
        with torch.no_grad():
            tokens = self.backbone.forward_features(normalized)
        cls = self.backbone.forward_head(tokens, pre_logits=True)
        if return_tokens:
            return cls, tokens[:, 1:, :]
        return cls


@torch.no_grad()
def extract(images, model_name, device, batch_size=64):
    model = FrozenBackbone(model_name).to(device)
    n = len(images)
    cls_out = np.empty((n, model.embed_dim), dtype=np.float32)
    tokens_out = None
    for start in range(0, n, batch_size):
        chunk = np.asarray(images[start : start + batch_size], dtype=np.float32) / 255.0
        batch = torch.as_tensor(chunk, device=device)
        cls, tokens = model(batch, return_tokens=True)
        if tokens_out is None:
            tokens_out = np.empty((n, tokens.shape[1], tokens.shape[2]), dtype=np.float16)
        cls_out[start : start + len(batch)] = cls.cpu().numpy()
        tokens_out[start : start + len(batch)] = tokens.cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    return cls_out, tokens_out


def margin(scores):
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit_model(features, subjects, mask):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(features[mask], subjects[mask])
    return model


def evaluate(features, subjects, metadata):
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_conf = []
        for inner_unknown_subjects in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
            mask = np.isin(subjects, inner_enrolled)
            model = fit_model(features, subjects, mask & ~forward & ~inner)
            eval_idx = np.flatnonzero(mask & inner)
            inner_conf.append(margin(model.decision_function(features[eval_idx])))
        threshold = float(np.quantile(np.concatenate(inner_conf), 0.05))

        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = fit_model(features, subjects, outer_fit)
        classes = model.classes_
        ks = model.decision_function(features[known_eval])
        us = model.decision_function(features[unknown_eval])
        kc, uc = margin(ks), margin(us)
        kp = classes[ks.argmax(axis=1)].copy(); kp[kc < threshold] = 0
        up = classes[us.argmax(axis=1)].copy(); up[uc < threshold] = 0
        det_true = np.r_[np.ones(len(known_eval)), np.zeros(len(unknown_eval))]
        det_conf = np.r_[kc, uc]
        # 闭集（拒识前）
        closed_pred = classes[ks.argmax(axis=1)]
        rows.append({
            "fold": fold,
            "closed_accuracy": float(np.mean(closed_pred == subjects[known_eval])),
            "C": float(np.mean(kp == subjects[known_eval])),
            "identity_macro_f1": float(f1_score(
                subjects[known_eval], kp, average="macro", zero_division=0)),
            "U": float(np.mean(up == 0)),
            "unknown_auroc": float(roc_auc_score(det_true, det_conf)),
            "unknown_f1": float(f1_score(det_true, det_conf >= threshold,
                                         pos_label=0, zero_division=0)),
            "known_acceptance": float(np.mean(kc >= threshold)),
        })
    summary = {key: float(np.mean([r[key] for r in rows]))
               for key in ("closed_accuracy", "C", "identity_macro_f1", "U",
                           "unknown_auroc", "unknown_f1", "known_acceptance")}
    summary["worst_fold_auroc"] = float(min(r["unknown_auroc"] for r in rows))
    return {"folds": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")

    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    images = np.load(CACHE / "eeg_image_dinov2" / "train_covariance5.npy", mmap_mode="r")

    started = time.time()
    cls, tokens = extract(images, args.model, device, args.batch_size)
    print(f"{args.tag}: cls {cls.shape} tokens {tokens.shape} in {time.time()-started:.0f}s",
          flush=True)
    np.save(CACHE / "eeg_image_dinov2" / f"train_covariance5_{args.tag}.npy", cls)

    cov = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    variants = {
        "image_cls": cls,
        "fused": np.concatenate([cls, cov], axis=1),
    }
    results = {}
    for name, features in variants.items():
        results[name] = evaluate(features, subjects, metadata)
        s = results[name]["summary"]
        print(f"  {name:10s} closed={s['closed_accuracy']:.4f} C={s['C']:.4f} "
              f"U={s['U']:.4f} AUROC={s['unknown_auroc']:.4f} worst={s['worst_fold_auroc']:.4f}",
              flush=True)

    path = OUT / f"e20j_{args.tag}.json"
    path.write_text(json.dumps({
        "protocol": {
            "note": "纯 EEG；严格嵌套阈值；无 Session",
            "model": args.model,
            "image": "covariance5 (5-band log-covariance tiled 2x3)",
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
