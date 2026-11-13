"""E54 探针：逐样本自归一化分数能否消除跨会话尺度偏移。

约束：只用单个样本自身的分数向量，不使用 Session、批次统计或 Test 标签。

候选（全部逐样本）：
  S1 distance_ratio（基线）
  S2 z-score： (max - mean) / std，衡量「最高分相对自身分布有多突出」
  S3 归一化 margin：(max - second) / std
  S4 余弦对比度： (max - mean) / (1 - mean)
  S5 秩内归一：把该样本 K 个分数排序后，max 的秩位置（恒为 1，无信息，作对照）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))

from run_e36 import ZBAND_PATH, load_config, load_train_matrix, train_metadata  # noqa: E402
from run_e40 import confidence_values, fit_prototype  # noqa: E402

device = torch.device("cuda")
e36 = load_config()
metadata = train_metadata(e36)
subjects = metadata["subject"].to_numpy(dtype=int)
trial = metadata["trial"].to_numpy()
maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
fit_np = np.flatnonzero(trial <= 0.8 * maximum)
cal_np = np.flatnonzero(trial > 0.8 * maximum)
classes = np.unique(subjects)
proto = fit_prototype(zband, fit_np, subjects, classes,
                      float(e36["models"]["wccn_shrinkage"]),
                      float(e36["models"]["wccn_eigenvalue_floor"]))


def score_features(features):
    with torch.no_grad():
        whitened = features @ proto.whitening.to(features.dtype)
        normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (normalized @ proto.prototypes.to(features.dtype).T).cpu().numpy()


def statistics(scores):
    scores = np.asarray(scores, dtype=np.float64)
    ordered = np.sort(scores, axis=1)
    maximum, second = ordered[:, -1], ordered[:, -2]
    mean = scores.mean(1)
    std = scores.std(1).clip(1e-6)
    return {
        "S1_distance_ratio": (1.0 - second) / np.maximum(1.0 - maximum, 1e-6),
        "S2_zscore": (maximum - mean) / std,
        "S3_margin_over_std": (maximum - second) / std,
        "S4_contrast": (maximum - mean) / np.maximum(1.0 - mean, 1e-6),
        "S5_neg_std": -std,
    }


cal_stats = statistics(score_features(zband[torch.as_tensor(cal_np, device=device)]))

test_meta = pd.read_csv(ROOT / "BrainPrint_cache" / "test_metadata.csv")
paths = [ROOT / "Brainprint_Competition" / str(p) for p in test_meta["path"]]
raw = torch.empty((len(paths), e36["data"]["channels"], e36["data"]["samples_per_trial"]),
                  dtype=torch.float32, device=device)
with ThreadPoolExecutor(max_workers=int(e36["features"]["io_workers"])) as pool:
    for start in range(0, len(paths), 256):
        arrays = list(pool.map(load_train_matrix, paths[start:start + 256]))
        raw[start:start + len(arrays)] = torch.as_tensor(np.stack(arrays), dtype=torch.float32,
                                                         device=device)
from run_e36 import detrend_zband, gpu_logspd  # noqa: E402

test_features = torch.empty((len(raw), 855), dtype=torch.float32, device=device)
with torch.no_grad():
    for start in range(0, len(raw), 256):
        chunk = raw[start:start + 256]
        test_features[start:start + len(chunk)] = detrend_zband(gpu_logspd(chunk, e36), e36)
test_stats = statistics(score_features(test_features))

rows = {}
for key in cal_stats:
    cal = cal_stats[key]
    test = test_stats[key]
    tau = float(np.quantile(cal, 0.02, method="higher"))
    reject = float(np.mean(test < tau))
    overlap = float(np.mean((test >= np.quantile(cal, 0.01)) & (test <= np.quantile(cal, 0.99))))
    rows[key] = {"train_q02": tau, "train_median": float(np.median(cal)),
                 "test_median": float(np.median(test)), "test_reject_rate": reject,
                 "test_in_train_1_99_range": overlap}
    print(f"{key:22s} train_med={np.median(cal):.4f} test_med={np.median(test):.4f} "
          f"tau={tau:.4f} test_reject={reject:.4f} overlap={overlap:.4f}", flush=True)

(OUT / "probe_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
