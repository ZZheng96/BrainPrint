"""E53 探针：无监督逐样本分数标定能否修复 Test 拒识率。

约束（项目硬约束）：
  - 不使用 Session 编号、文件顺序、样本数量、批次统计
  - 不使用 Test 标签
  - 只允许「逐样本」的变换

候选变换（全部逐样本、单调）：
  A. 原始 distance_ratio（E52 现状，基线）
  B. 秩映射：把 Test 的 dr 经验分布映射到 Train 校准块分布（需要批次分位 → 违反约束，仅作诊断对照）
  C. 逐样本自归一化：dr / (1 + dr) 之类的单调变换（不改变排序，无法修复偏移）
  D. 逐样本特征归一化：在特征层面做逐 trial 标准化（不依赖批次）

本探针先量化偏移的形态，再判断哪类变换在约束内可行。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))

from run_e36 import ZBAND_PATH, detrend_zband, gpu_logspd, load_config, train_metadata  # noqa: E402
from run_attacks import load_all_raw  # noqa: E402
from run_e40 import FrozenPrototype, confidence_values, fit_prototype  # noqa: E402

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
model = FrozenPrototype(proto, e36)

def score_features(features: torch.Tensor) -> np.ndarray:
    """zband 特征 → 余弦分数（与 FrozenPrototype 同口径）。"""
    with torch.no_grad():
        whitened = features @ proto.whitening.to(features.dtype)
        normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (normalized @ proto.prototypes.to(features.dtype).T).cpu().numpy()


cal_scores = score_features(zband[torch.as_tensor(cal_np, device=device)])
cal_dr = confidence_values(cal_scores)["distance_ratio"]
cal_mc = confidence_values(cal_scores)["max_cosine"]
print(f"Train 校准块 N={len(cal_dr)}")
for name, values in (("distance_ratio", cal_dr), ("max_cosine", cal_mc)):
    print(f"  {name}: q01={np.quantile(values,0.01):.4f} q05={np.quantile(values,0.05):.4f} "
          f"q50={np.quantile(values,0.5):.4f} q95={np.quantile(values,0.95):.4f}")

# Test 侧：只读元数据与波形，不读 Session
import pandas as pd  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from run_e36 import load_train_matrix  # noqa: E402

test_meta = pd.read_csv(ROOT / "BrainPrint_cache" / "test_metadata.csv")
paths = [ROOT / "Brainprint_Competition" / str(p) for p in test_meta["path"]]
raw = torch.empty((len(paths), e36["data"]["channels"], e36["data"]["samples_per_trial"]),
                  dtype=torch.float32, device=device)
with ThreadPoolExecutor(max_workers=int(e36["features"]["io_workers"])) as pool:
    for start in range(0, len(paths), 256):
        arrays = list(pool.map(load_train_matrix, paths[start:start + 256]))
        raw[start:start + len(arrays)] = torch.as_tensor(np.stack(arrays), dtype=torch.float32,
                                                         device=device)
test_scores = model.scores(raw)
test_dr = confidence_values(test_scores)["distance_ratio"]
test_mc = confidence_values(test_scores)["max_cosine"]
print(f"\nTest N={len(test_dr)}")
for name, values in (("distance_ratio", test_dr), ("max_cosine", test_mc)):
    print(f"  {name}: q01={np.quantile(values,0.01):.4f} q05={np.quantile(values,0.05):.4f} "
          f"q50={np.quantile(values,0.5):.4f} q95={np.quantile(values,0.95):.4f}")

tau_low = float(np.quantile(cal_dr, 0.02, method="higher"))
tau_high = float(np.quantile(cal_dr, 0.05, method="higher"))
print(f"\nTrain 阈值 tau_low={tau_low:.4f} tau_high={tau_high:.4f}")
print(f"Test 拒识率（原始 dr）= {np.mean(test_dr < tau_low):.4f}")
print(f"Test 落入 logSPD 区 = {np.mean(test_dr >= tau_high):.4f}")

# 诊断对照：秩映射（违反约束，仅用于判断「偏移是否只是单调平移」）
rank = np.searchsorted(np.sort(cal_dr), test_dr, side="right") / len(cal_dr)
print(f"\n[诊断] 秩映射后 Test 拒识率 = {np.mean(rank < 0.02):.4f}（应≈0.02，说明偏移是单调的）")

# 逐样本可行方案：用 max_cosine 与 dr 的联合，检查 Test 上是否仍可分
print(f"\nTest max_cosine 中位 {np.median(test_mc):.4f} vs Train {np.median(cal_mc):.4f}")
print(f"Test dr 中位 {np.median(test_dr):.4f} vs Train {np.median(cal_dr):.4f}")

(OUT / "probe_results.json").write_text(json.dumps({
    "train": {"n": int(len(cal_dr)), "dr_q": [float(np.quantile(cal_dr, q)) for q in (0.01, 0.05, 0.5, 0.95)],
              "mc_q": [float(np.quantile(cal_mc, q)) for q in (0.01, 0.05, 0.5, 0.95)],
              "tau_low": tau_low, "tau_high": tau_high},
    "test": {"n": int(len(test_dr)), "dr_q": [float(np.quantile(test_dr, q)) for q in (0.01, 0.05, 0.5, 0.95)],
             "mc_q": [float(np.quantile(test_mc, q)) for q in (0.01, 0.05, 0.5, 0.95)],
             "reject_rate_raw": float(np.mean(test_dr < tau_low)),
             "logspd_region_raw": float(np.mean(test_dr >= tau_high)),
             "reject_rate_rank_mapped": float(np.mean(rank < 0.02))},
}, ensure_ascii=False, indent=2), encoding="utf-8")
