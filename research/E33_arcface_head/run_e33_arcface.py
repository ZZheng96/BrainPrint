"""E33：ArcFace 身份头微调 DINO 特征（DINO 路线内）。

E30–E32 的结论：判据层面已饱和（AUROC 0.74），瓶颈在 DINO 特征的判别力。
本实验在冻结的 DINOv2 cov5 特征之上训练一个轻量投影头 + ArcFace 角间隔损失，
让同身份特征聚拢、异身份推开，再用余弦距离做开集拒识。

评估（严格嵌套，外层 5 折留被试、内层伪未知标定阈值）：
  - 闭集准确率、身份 macro-F1（Train）
  - C / U / unknown AUROC（Train）
  - Test 审计：AUROC(S2 vs S1)、S1 拒识率与 S2 接受率的工作点曲线

不使用 Session、文件顺序、样本数量、近重复配对；Test 只做审计。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eeg_image_dinov2.heads import IdentityHead
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
DINO = CACHE / "eeg_image_dinov2"
OUT = Path(__file__).resolve().parent
QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20)
EPOCHS = 40
BATCH = 256
LR = 1e-3
# 命令行可覆盖：python run_e33_arcface.py <margin> <dropout>
MARGIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
DROPOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
TAG = f"m{MARGIN}_d{DROPOUT}"


def train_head(features: np.ndarray, labels: np.ndarray, n_classes: int, seed: int) -> IdentityHead:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdentityHead(in_dim=features.shape[1], n_classes=n_classes,
                         dropout=DROPOUT, margin=MARGIN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    # 折内的被试编号不连续（可能缺少某个 subject id），必须重映射到 0..n_classes-1
    classes = np.unique(labels)
    if len(classes) != n_classes:
        raise ValueError(f"expected {n_classes} classes, got {len(classes)}")
    remap = {int(label): index for index, label in enumerate(classes)}
    y = torch.as_tensor([remap[int(label)] for label in labels], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(x), generator=generator, device=device)
        for start in range(0, len(order), BATCH):
            index = order[start : start + BATCH]
            logits = model(x[index], y[index])
            loss = F.cross_entropy(logits, y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


@torch.no_grad()
def embed(model: IdentityHead, features: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return model.embed(tensor).cpu().numpy()


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_x = np.load(DINO / "train_covariance5_dinov2.npy").astype(np.float32)
    test_x = np.load(DINO / "test_covariance5_dinov2.npy").astype(np.float32)

    report = {}
    # ---------- 嵌套评估 ----------
    folds = np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5)
    rows = []
    started = time.time()
    for fold, unknown in enumerate(folds, 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        is_known = np.isin(subjects, known_subjects)
        outer_fit = is_known & ~forward
        model = train_head(train_x[outer_fit], subjects[outer_fit], len(known_subjects), seed=fold)
        z_train = embed(model, train_x)

        eval_known = np.flatnonzero(is_known & forward)
        eval_unknown = np.flatnonzero(np.isin(subjects, unknown))
        true_known = subjects[eval_known]

        # 内层伪未知标定阈值
        inner_conf = []
        for inner_unknown in np.array_split(
            np.random.default_rng(100 + fold).permutation(known_subjects), 2
        ):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_enrolled = np.isin(subjects, enrolled)
            sub_model = train_head(train_x[mask_enrolled & ~forward & ~inner],
                                   subjects[mask_enrolled & ~forward & ~inner], len(enrolled), seed=100 + fold)
            z_sub = embed(sub_model, train_x)
            centers = np.stack([z_sub[mask_enrolled & ~forward & ~inner & (subjects == c)].mean(0) for c in enrolled])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
            cos = z_sub[np.flatnonzero(mask_enrolled & inner)] @ centers.T
            inner_conf.append(-np.max(cos, axis=1))
        threshold = float(np.quantile(np.concatenate(inner_conf), 0.95))

        centers = np.stack([z_train[outer_fit & (subjects == c)].mean(0) for c in known_subjects])
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        cos_known = z_train[eval_known] @ centers.T
        cos_unknown = z_train[eval_unknown] @ centers.T
        pred_known = known_subjects[cos_known.argmax(1)]
        pred_unknown = known_subjects[cos_unknown.argmax(1)]
        novelty_known = -cos_known.max(1)
        novelty_unknown = -cos_unknown.max(1)

        accepted_known = novelty_known <= threshold
        accepted_unknown = novelty_unknown <= threshold
        pred_known = np.where(accepted_known, pred_known, 0)
        pred_unknown = np.where(accepted_unknown, pred_unknown, 0)

        rows.append({
            "closed_acc": float(np.mean(known_subjects[cos_known.argmax(1)] == true_known)),
            "C": float(np.mean(pred_known == true_known)),
            "identity_f1": float(f1_score(true_known, pred_known, average="macro", zero_division=0)),
            "U": float(np.mean(pred_unknown == 0)),
            "auroc": float(roc_auc_score(
                np.r_[np.ones(len(eval_known)), np.zeros(len(eval_unknown))],
                np.r_[(-novelty_known), (-novelty_unknown)],
            )),
            "acceptance": float(accepted_known.mean()),
        })
        print(f"fold {fold} closed={rows[-1]['closed_acc']:.4f} C={rows[-1]['C']:.4f} "
              f"U={rows[-1]['U']:.4f} AUROC={rows[-1]['auroc']:.4f} | {time.time()-started:.0f}s", flush=True)

    nested = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    report["nested"] = nested
    print(f"\n嵌套平均：closed={nested['closed_acc']:.4f} C={nested['C']:.4f} "
          f"identity_F1={nested['identity_f1']:.4f} U={nested['U']:.4f} AUROC={nested['auroc']:.4f}", flush=True)

    # ---------- 全量模型 + Test 审计 ----------
    model = train_head(train_x, subjects, 20, seed=2026)
    z_train = embed(model, train_x)
    z_test = embed(model, test_x)
    centroid_mask = ~inner & ~forward
    centers = np.stack([z_train[centroid_mask & (subjects == c)].mean(0) for c in range(1, 21)])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)

    cos_test = z_test @ centers.T
    novelty_test = -cos_test.max(1)
    predicted = np.arange(1, 21)[cos_test.argmax(1)]
    inner_novelty = -np.max(z_train @ centers.T, axis=1)
    inner_novelty = inner_novelty[inner]

    auroc_test = float(roc_auc_score(known, -novelty_test))
    entry = {
        "test_auroc": auroc_test,
        "train_p50": float(np.median(inner_novelty)),
        "s2_p50": float(np.median(novelty_test[known])),
        "s1_p50": float(np.median(novelty_test[~known])),
        "points": {},
    }
    for q in QUANTILES:
        threshold = float(np.quantile(inner_novelty, 1 - q))
        keep = novelty_test <= threshold
        candidate = np.where(keep, predicted, 0)
        entry["points"][f"q{int(q*100):02d}"] = {
            "s2_accept": float(keep[known].mean()),
            "s1_reject": float((~keep[~known]).mean()),
            "zero_count": int((candidate == 0).sum()),
        }
        pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
            OUT / f"candidate_arcface_{TAG}_q{int(q*100):02d}.csv", index=False
        )
    report["test"] = entry
    print(f"Test AUROC={auroc_test:.4f} | train {entry['train_p50']:.3f} "
          f"S2 {entry['s2_p50']:.3f} S1 {entry['s1_p50']:.3f}", flush=True)
    for key, value in entry["points"].items():
        print(f"  {key}: S2 accept={value['s2_accept']:.3f} S1 reject={value['s1_reject']:.3f}", flush=True)

    (OUT / f"e33_arcface_{TAG}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
