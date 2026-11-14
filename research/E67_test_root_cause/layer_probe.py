"""E67 诊断 1：逐层追踪 S1/S2 可分性，定位信息在哪一层丢失。

层级：
  L0 原始波形统计（范数、峰度、频谱平坦度、20-40Hz 占比）
  L1 频带功率（5 频带 × 18 通道 = 90 维）
  L2 频带 log 协方差（5 频带 × 171 维 = 855 维，即 covariance5 的源特征）
  L3 DINOv2 图像特征（768 维，covariance5 图像）

对每一层：
  - 用 Train 拟合一个「S1 vs S2 二分类器」（仅诊断，不可用于决策）
  - 在 Test 上算 AUROC（交叉验证，避免过拟合）
  - 同时算 Train 内部「留被试伪未知 vs 已知」的 AUROC 作对照

关键：如果 L0/L1 的 AUROC 就接近 0.5，说明原始信号层面 S1/S2 无差异，
      问题不在图像化或 DINO，而在数据本身。
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from src.eeg_features import load_eeg  # noqa: E402

CACHE = ROOT / "BrainPrint_cache"
DATA = ROOT / "Brainprint_Competition"
FS = 300
BANDS = ((1, 8), (8, 13), (13, 30), (30, 45), (45, 90))
N_CHANNELS = 18


def load_raw(path: str) -> np.ndarray:
    data, _ = load_eeg(path)
    return data.T.astype(np.float32)  # (18, 1200)


def bandpass_fft(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """data: (C, T)"""
    spectrum = np.fft.rfft(data, axis=-1)
    freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / FS)
    mask = (freqs >= low) & (freqs <= high)
    spectrum[..., ~mask] = 0
    return np.fft.irfft(spectrum, n=data.shape[-1], axis=-1)


def log_cov_vector(segment: np.ndarray) -> np.ndarray:
    """segment: (C, T) -> 上三角 log 协方差向量（含对角）"""
    cov = np.cov(segment, rowvar=True, ddof=1)
    target = np.trace(cov) / N_CHANNELS * np.eye(N_CHANNELS)
    cov = 0.95 * cov + 0.05 * target
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    log_cov = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    triu = np.triu_indices(N_CHANNELS)
    return log_cov[triu].astype(np.float32)


def layer_features(raw: np.ndarray) -> dict[str, np.ndarray]:
    """单条 (18,1200) 波形 -> 各层特征"""
    centered = raw - raw.mean(axis=-1, keepdims=True)
    # L0 原始统计
    norm = float(np.linalg.norm(centered))
    spectrum = np.abs(np.fft.rfft(centered, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(centered.shape[-1], d=1.0 / FS)
    total = spectrum.sum(-1).clip(1e-12)
    high_ratio = float((spectrum[:, freqs >= 20].sum(-1) / total).mean())
    flatness = float((np.exp(np.log(spectrum.clip(1e-12)).mean(-1)) / spectrum.mean(-1).clip(1e-12)).mean())
    kurt = float((((centered - centered.mean(-1, keepdims=True)) ** 4).mean(-1)
                  / (centered.var(-1).clip(1e-12) ** 2)).mean())
    l0 = np.array([norm, high_ratio, flatness, kurt], dtype=np.float32)

    # L1 频带功率 + L2 频带 log 协方差
    powers, covs = [], []
    for low, high in BANDS:
        filtered = bandpass_fft(centered, low, high)
        powers.append(np.log1p((filtered ** 2).mean(-1)))
        covs.append(log_cov_vector(filtered))
    l1 = np.concatenate(powers).astype(np.float32)
    l2 = np.concatenate(covs).astype(np.float32)
    return {"L0_raw_stats": l0, "L1_band_power": l1, "L2_log_cov": l2}


def build_layer_cache(split: str, metadata: pd.DataFrame, workers: int = 8) -> dict[str, np.ndarray]:
    path = CACHE / "e63_root_cause" / f"{split}_layers.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        loaded = np.load(path)
        print(f"reuse {path.name}", flush=True)
        return {k: loaded[k] for k in loaded.files}
    paths = [str(DATA / p) for p in metadata["path"]]
    results = {"L0_raw_stats": [], "L1_band_power": [], "L2_log_cov": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, feats in enumerate(pool.map(lambda p: layer_features(load_raw(p)), paths), 1):
            for key in results:
                results[key].append(feats[key])
            if index % 5000 == 0:
                print(f"{split}: {index}/{len(paths)}", flush=True)
    out = {k: np.stack(v) for k, v in results.items()}
    np.savez(path, **out)
    print(f"saved {path.name}", flush=True)
    return out


def cv_auroc(features: np.ndarray, labels: np.ndarray, folds: int = 5) -> float:
    """交叉验证 AUROC（避免过拟合导致的虚高）"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros(len(labels), dtype=np.float64)
    for train_idx, test_idx in skf.split(features, labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(features[train_idx], labels[train_idx])
        oof[test_idx] = model.decision_function(features[test_idx])
    return float(roc_auc_score(labels, oof))


def main() -> None:
    train_meta = pd.read_csv(CACHE / "train_metadata.csv")
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")

    print("构建 Train 分层特征...", flush=True)
    train_layers = build_layer_cache("train", train_meta)
    print("构建 Test 分层特征...", flush=True)
    test_layers = build_layer_cache("test", test_meta)

    # Test 的 S1/S2 标签（仅诊断）
    session = test_meta["session"].astype(str).str.strip().str.casefold()
    codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                         "2": "S2", "s2": "S2", "session2": "S2"})
    if codes.isna().any():
        raise ValueError("未知 Session 取值")
    is_s1 = (codes == "S1").to_numpy()
    is_s2 = (codes == "S2").to_numpy()
    test_labels = np.concatenate([np.ones(is_s2.sum()), np.zeros(is_s1.sum())])

    # Train 内部对照：留被试伪未知 vs 已知
    subjects = train_meta["subject"].to_numpy()
    rng = np.random.default_rng(42)
    unknown_subjects = rng.permutation(np.unique(subjects))[:4]
    train_known = ~np.isin(subjects, unknown_subjects)
    train_labels = train_known.astype(int)

    report = {}
    print(f"\n{'layer':18s} {'dim':>6s} {'Test AUROC':>11s} {'Test flip':>10s} "
          f"{'Train AUROC':>12s}", flush=True)
    for layer in ("L0_raw_stats", "L1_band_power", "L2_log_cov"):
        test_x = np.concatenate([test_layers[layer][is_s2], test_layers[layer][is_s1]])
        train_x = train_layers[layer]
        test_auroc = cv_auroc(test_x, test_labels)
        train_auroc = cv_auroc(train_x, train_labels)
        report[layer] = {
            "dim": int(test_x.shape[1]),
            "test_auroc_s2_vs_s1": test_auroc,
            "test_auroc_flipped": max(test_auroc, 1 - test_auroc),
            "train_auroc_known_vs_unknown": train_auroc,
        }
        print(f"{layer:18s} {test_x.shape[1]:6d} {test_auroc:11.4f} "
              f"{max(test_auroc, 1-test_auroc):10.4f} {train_auroc:12.4f}", flush=True)

    # L3：DINOv2 图像特征（已有缓存）
    dino_path = CACHE / "eeg_image_dinov2" / "test_covariance5_dinov2.npy"
    if dino_path.exists():
        dino = np.load(dino_path)
        test_x = np.concatenate([dino[is_s2], dino[is_s1]])
        test_auroc = cv_auroc(test_x, test_labels)
        train_dino = np.load(CACHE / "eeg_image_dinov2" / "train_covariance5_dinov2.npy")
        train_auroc = cv_auroc(train_dino, train_labels)
        report["L3_dino_cov5"] = {
            "dim": int(test_x.shape[1]),
            "test_auroc_s2_vs_s1": test_auroc,
            "test_auroc_flipped": max(test_auroc, 1 - test_auroc),
            "train_auroc_known_vs_unknown": train_auroc,
        }
        print(f"{'L3_dino_cov5':18s} {test_x.shape[1]:6d} {test_auroc:11.4f} "
              f"{max(test_auroc, 1-test_auroc):10.4f} {train_auroc:12.4f}", flush=True)

    (OUT / "layer_auroc.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'layer_auroc.json'}", flush=True)


if __name__ == "__main__":
    main()
