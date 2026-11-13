"""E60 诊断：Test 上 S1/S2 拒识率失衡的根因。

E52 实测（E49 系统）：S1 拒识 0.5875、S2 拒识 0.7467，两项门禁（S2≤2%、S1≥95%）都失败。
本诊断只做无标签结构分析，不参与任何决策：
  1. Test 的 distance_ratio 在 S1/S2 上的分布（仅离线审计）
  2. 若把阈值放到「S2 拒识≤2%」，S1 拒识会是多少（衡量可分性上限）
  3. 若把阈值放到「S1 拒识≥95%」，S2 拒识会是多少
  4. 用 S1/S2 标签算 AUROC（仅作上限诊断，不可用于调参）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))

from run_e36 import ZBAND_PATH, detrend_zband, gpu_logspd, load_config, load_train_matrix, train_metadata  # noqa: E402
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


cal_dr = confidence_values(score_features(zband[torch.as_tensor(cal_np, device=device)]))["distance_ratio"]

test_meta = pd.read_csv(ROOT / "BrainPrint_cache" / "test_metadata.csv")
paths = [ROOT / "Brainprint_Competition" / str(p) for p in test_meta["path"]]
raw = torch.empty((len(paths), e36["data"]["channels"], e36["data"]["samples_per_trial"]),
                  dtype=torch.float32, device=device)
with ThreadPoolExecutor(max_workers=int(e36["features"]["io_workers"])) as pool:
    for start in range(0, len(paths), 256):
        arrays = list(pool.map(load_train_matrix, paths[start:start + 256]))
        raw[start:start + len(arrays)] = torch.as_tensor(np.stack(arrays), dtype=torch.float32,
                                                         device=device)
features = torch.empty((len(raw), 855), dtype=torch.float32, device=device)
with torch.no_grad():
    for start in range(0, len(raw), 256):
        chunk = raw[start:start + 256]
        features[start:start + len(chunk)] = detrend_zband(gpu_logspd(chunk, e36), e36)
test_dr = confidence_values(score_features(features))["distance_ratio"]

session = test_meta["session"].astype(str).str.strip().str.casefold()
codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                     "2": "S2", "s2": "S2", "session2": "S2"})
if codes.isna().any():
    raise ValueError("未知 Session 取值")
is_s1 = (codes == "S1").to_numpy()
is_s2 = (codes == "S2").to_numpy()

print(f"Train 校准块 dr: q02={np.quantile(cal_dr,0.02):.4f} q05={np.quantile(cal_dr,0.05):.4f} "
      f"中位={np.median(cal_dr):.4f}")
for name, mask in (("S1", is_s1), ("S2", is_s2)):
    values = test_dr[mask]
    print(f"Test {name} N={mask.sum()}: q02={np.quantile(values,0.02):.4f} "
          f"q05={np.quantile(values,0.05):.4f} 中位={np.median(values):.4f} "
          f"q95={np.quantile(values,0.95):.4f}")

labels = np.concatenate([np.ones(is_s2.sum()), np.zeros(is_s1.sum())])
values = np.concatenate([test_dr[is_s2], test_dr[is_s1]])
auroc = float(roc_auc_score(labels, values))
print(f"\n[上限诊断] AUROC(S2 vs S1, 用 dr) = {auroc:.4f}")

# 若阈值使 S2 拒识 = 2%，S1 拒识是多少
tau_s2 = float(np.quantile(test_dr[is_s2], 0.02, method="lower"))
print(f"若 S2 拒识=2% (tau={tau_s2:.4f}) -> S1 拒识={np.mean(test_dr[is_s1] < tau_s2):.4f}")
# 若阈值使 S1 拒识 = 95%
tau_s1 = float(np.quantile(test_dr[is_s1], 0.95, method="higher"))
print(f"若 S1 拒识=95% (tau={tau_s1:.4f}) -> S2 拒识={np.mean(test_dr[is_s2] < tau_s1):.4f}")

# 用 Train 阈值
tau_train = float(np.quantile(cal_dr, 0.02, method="higher"))
print(f"用 Train 阈值 (tau={tau_train:.4f}) -> S1 拒识={np.mean(test_dr[is_s1] < tau_train):.4f} "
      f"S2 拒识={np.mean(test_dr[is_s2] < tau_train):.4f}")

(OUT / "probe_results.json").write_text(json.dumps({
    "train_cal": {"q02": float(np.quantile(cal_dr, 0.02)), "q05": float(np.quantile(cal_dr, 0.05)),
                  "median": float(np.median(cal_dr))},
    "test_s1": {"n": int(is_s1.sum()), "median": float(np.median(test_dr[is_s1])),
                "q95": float(np.quantile(test_dr[is_s1], 0.95))},
    "test_s2": {"n": int(is_s2.sum()), "median": float(np.median(test_dr[is_s2])),
                "q02": float(np.quantile(test_dr[is_s2], 0.02))},
    "auroc_s2_vs_s1": auroc,
    "s2_2pct_then_s1_reject": float(np.mean(test_dr[is_s1] < tau_s2)),
    "s1_95pct_then_s2_reject": float(np.mean(test_dr[is_s2] < tau_s1)),
    "train_threshold": {"tau": tau_train,
                        "s1_reject": float(np.mean(test_dr[is_s1] < tau_train)),
                        "s2_reject": float(np.mean(test_dr[is_s2] < tau_train))},
}, ensure_ascii=False, indent=2), encoding="utf-8")
