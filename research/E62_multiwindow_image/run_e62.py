"""E62：多窗口时序图像 + 冻结 DINOv2，检验能否突破 Test AUROC 0.74 上限。

流程：
  1. 生成 Train/Test 多窗口图像（缓存到 BrainPrint_cache/e62_multiwindow）
  2. 冻结 DINOv2 ViT-B/14 提 CLS 特征
  3. 严格嵌套评估（外层 5 折留被试 + 内层伪未知阈值），与 E20D 同协议
  4. Test 无标签审计：AUROC(S2 vs S1)、S1/S2 拒识率

用法：
  python run_e62.py --stage images   # 生成图像缓存
  python run_e62.py --stage features # 提 DINOv2 特征
  python run_e62.py --stage nested   # 训练侧严格嵌套评估
  python run_e62.py --stage test     # Test 无标签审计
  python run_e62.py --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from encoders import encode  # noqa: E402
from src.eeg_features import load_eeg  # noqa: E402
from src.eeg_image_dinov2.backbone import extract_features  # noqa: E402
from src.open_set_improvement.core import COV_COLUMNS  # noqa: E402
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout  # noqa: E402

DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"
IMG_CACHE = CACHE / "e62_multiwindow"
METHODS = ("multiwindow_cov8", "multiwindow_cov4", "multiwindow_corr8")


def encode_path(args: tuple[str, str]) -> np.ndarray:
    path, method = args
    data, _ = load_eeg(path)
    return encode(data, method)


def build_images(method: str, split: str, metadata: pd.DataFrame, workers: int) -> np.ndarray:
    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    path = IMG_CACHE / f"{split}_{method}.npy"
    shape = (len(metadata), 3, 224, 224)
    if path.exists() and np.load(path, mmap_mode="r").shape == shape:
        print(f"reuse {path.name}", flush=True)
        return np.load(path, mmap_mode="r")
    images = np.empty(shape, dtype=np.uint8)
    paths = [(str(DATA / p), method) for p in metadata["path"]]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, image in enumerate(pool.map(encode_path, paths, chunksize=16), 1):
            images[index - 1] = np.clip(image * 255, 0, 255).astype(np.uint8)
            if index % 2000 == 0 or index == len(paths):
                print(f"{split}/{method}: {index}/{len(paths)} | {time.time()-started:.0f}s", flush=True)
    np.save(path, images)
    return np.load(path, mmap_mode="r")


def extract(method: str, split: str, device: torch.device, batch_size: int = 64) -> np.ndarray:
    path = IMG_CACHE / f"{split}_{method}_dinov2.npy"
    if path.exists():
        print(f"reuse {path.name}", flush=True)
        return np.load(path)
    images = np.load(IMG_CACHE / f"{split}_{method}.npy", mmap_mode="r")
    started = time.time()
    features = extract_features(images, device, batch_size)
    print(f"{split}/{method} DINOv2 {features.shape} in {time.time()-started:.0f}s", flush=True)
    np.save(path, features)
    return features


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit_model(features, subjects, mask):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(features[mask], subjects[mask])
    return model


def nested_evaluate(features: np.ndarray, subjects: np.ndarray, metadata: pd.DataFrame) -> dict:
    """与 E20D 完全同协议：外层 5 折留被试，内层 2 折伪未知标定阈值。"""
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
        for inner_unknown in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~forward & ~inner
            inner_eval_known = np.flatnonzero(inner_known_mask & inner)
            model = fit_model(features, subjects, inner_fit)
            inner_conf.append(margin(model.decision_function(features[inner_eval_known])))
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
        print(f"  fold {fold}: closed={rows[-1]['closed_accuracy']:.4f} C={rows[-1]['C']:.4f} "
              f"U={rows[-1]['U']:.4f} AUROC={rows[-1]['unknown_auroc']:.4f}", flush=True)
    summary = {key: float(np.mean([r[key] for r in rows]))
               for key in ("closed_accuracy", "C", "identity_macro_f1", "U",
                           "unknown_auroc", "unknown_f1", "known_acceptance")}
    summary["worst_fold_auroc"] = float(min(r["unknown_auroc"] for r in rows))
    return {"folds": rows, "summary": summary}


def test_audit(features: np.ndarray, subjects: np.ndarray, metadata: pd.DataFrame,
               test_features: np.ndarray, test_meta: pd.DataFrame) -> dict:
    """全量训练 + Test 推理，无标签审计（session 仅用于离线 AUROC）。"""
    model = fit_model(features, subjects, np.ones(len(subjects), dtype=bool))
    classes = model.classes_
    scores = model.decision_function(test_features)
    conf = margin(scores)
    predicted = classes[scores.argmax(axis=1)]

    session = test_meta["session"].astype(str).str.strip().str.casefold()
    codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                         "2": "S2", "s2": "S2", "session2": "S2"})
    if codes.isna().any():
        raise ValueError("未知 Session 取值")
    is_s1 = (codes == "S1").to_numpy()
    is_s2 = (codes == "S2").to_numpy()

    labels = np.concatenate([np.ones(is_s2.sum()), np.zeros(is_s1.sum())])
    pooled = np.concatenate([conf[is_s2], conf[is_s1]])
    auroc = float(roc_auc_score(labels, pooled))

    tau_s2 = float(np.quantile(conf[is_s2], 0.02, method="lower"))
    tau_s1 = float(np.quantile(conf[is_s1], 0.95, method="higher"))
    return {
        "auroc_s2_vs_s1": auroc,
        "auroc_flipped": max(auroc, 1 - auroc),
        "s1_median_conf": float(np.median(conf[is_s1])),
        "s2_median_conf": float(np.median(conf[is_s2])),
        "if_s2_reject_2pct_then_s1_reject": float(np.mean(conf[is_s1] < tau_s2)),
        "if_s1_reject_95pct_then_s2_reject": float(np.mean(conf[is_s2] < tau_s1)),
        "n_s1": int(is_s1.sum()),
        "n_s2": int(is_s2.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["images", "features", "nested", "test", "all"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")

    train_meta = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = train_meta["subject"].to_numpy()
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")

    results = {}
    for method in args.methods:
        print(f"\n=== {method} ===", flush=True)
        if args.stage in ("images", "all"):
            build_images(method, "train", train_meta, args.workers)
            build_images(method, "test", test_meta, args.workers)
        if args.stage in ("features", "all"):
            extract(method, "train", device, args.batch_size)
            extract(method, "test", device, args.batch_size)
        if args.stage in ("nested", "all"):
            features = np.load(IMG_CACHE / f"train_{method}_dinov2.npy")
            results.setdefault(method, {})["nested"] = nested_evaluate(
                features, subjects, train_meta
            )
            s = results[method]["nested"]["summary"]
            print(f"  nested summary: closed={s['closed_accuracy']:.4f} C={s['C']:.4f} "
                  f"F1={s['identity_macro_f1']:.4f} U={s['U']:.4f} "
                  f"AUROC={s['unknown_auroc']:.4f}", flush=True)
        if args.stage in ("test", "all"):
            features = np.load(IMG_CACHE / f"train_{method}_dinov2.npy")
            test_features = np.load(IMG_CACHE / f"test_{method}_dinov2.npy")
            results.setdefault(method, {})["test"] = test_audit(
                features, subjects, train_meta, test_features, test_meta
            )
            t = results[method]["test"]
            print(f"  test audit: AUROC(S2 vs S1)={t['auroc_s2_vs_s1']:.4f} "
                  f"(flipped {t['auroc_flipped']:.4f}) | "
                  f"S2拒识2%→S1拒识{t['if_s2_reject_2pct_then_s1_reject']:.4f} | "
                  f"S1拒识95%→S2拒识{t['if_s1_reject_95pct_then_s2_reject']:.4f}", flush=True)

    path = OUT / f"results_{args.stage}.json"
    path.write_text(json.dumps({
        "protocol": {
            "note": "纯 EEG；严格嵌套阈值；session 仅用于 Test 离线审计",
            "methods": args.methods,
            "backbone": "frozen vit_base_patch14_dinov2.lvd142m, img_size=224",
            "classifier": "StandardScaler + LogisticRegression(C=1.0)",
            "rejection": "top1-top2 margin, 内层伪未知 5% 分位",
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {path}", flush=True)


if __name__ == "__main__":
    main()
