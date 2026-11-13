"""E61 诊断：系统筛查各特征在 Test 上区分 S1（未注册）与 S2（注册）的能力。

只做无标签结构诊断：Session 仅用于离线计算 AUROC，不参与任何决策。
候选特征全部逐样本、不依赖批次统计：
  - logSPD WCCN：distance_ratio / max_cosine / margin
  - 原始波形统计：范数、通道间相关、频谱平坦度、20-40Hz 能量占比、峰度
  - 协方差结构：logSPD 特征范数、特征值谱熵
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
classes = np.unique(subjects)
proto = fit_prototype(zband, fit_np, subjects, classes,
                      float(e36["models"]["wccn_shrinkage"]),
                      float(e36["models"]["wccn_eigenvalue_floor"]))

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

with torch.no_grad():
    whitened = features @ proto.whitening.to(features.dtype)
    normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scores = (normalized @ proto.prototypes.to(features.dtype).T).cpu().numpy()
conf = confidence_values(scores)

centered = raw - raw.mean(dim=-1, keepdim=True)
signal_norm = centered.reshape(len(raw), -1).norm(dim=1).cpu().numpy()
spectrum = torch.fft.rfft(centered, dim=-1).abs() ** 2
freqs = torch.linspace(0, 150, spectrum.shape[-1], device=device)
total = spectrum.sum(-1).clamp_min(1e-12)
high_ratio = (spectrum[..., freqs >= 20].sum(-1) / total).mean(1).cpu().numpy()
flatness = (torch.exp(torch.log(spectrum.clamp_min(1e-12)).mean(-1))
            / spectrum.mean(-1).clamp_min(1e-12)).mean(1).cpu().numpy()
kurtosis = (((centered - centered.mean(-1, keepdim=True)) ** 4).mean(-1)
            / (centered.var(-1, unbiased=False).clamp_min(1e-12) ** 2)).mean(1).cpu().numpy()
channel_corr = []
for start in range(0, len(raw), 512):
    chunk = centered[start:start + 512]
    chunk = chunk - chunk.mean(dim=-1, keepdim=True)
    scale = chunk.std(dim=-1, unbiased=False).clamp_min(1e-12)
    corr = (chunk / scale[:, :, None]) @ (chunk / scale[:, :, None]).transpose(-1, -2)
    corr = corr / chunk.shape[-1]
    off = corr - torch.eye(18, device=device)[None]
    channel_corr.append(off.abs().mean(dim=(1, 2)).cpu().numpy())
channel_corr = np.concatenate(channel_corr)
feature_norm = features.norm(dim=1).cpu().numpy()
feature_std = features.std(dim=1).cpu().numpy()

candidates = {
    "logspd_distance_ratio": conf["distance_ratio"],
    "logspd_max_cosine": conf["max_cosine"],
    "logspd_margin": conf["margin"],
    "raw_signal_norm": signal_norm,
    "raw_high_band_ratio": high_ratio,
    "raw_spectral_flatness": flatness,
    "raw_kurtosis": kurtosis,
    "raw_channel_corr": channel_corr,
    "logspd_feature_norm": feature_norm,
    "logspd_feature_std": feature_std,
}

session = test_meta["session"].astype(str).str.strip().str.casefold()
codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                     "2": "S2", "s2": "S2", "session2": "S2"})
if codes.isna().any():
    raise ValueError("未知 Session 取值")
is_s1 = (codes == "S1").to_numpy()
is_s2 = (codes == "S2").to_numpy()
labels = np.concatenate([np.ones(is_s2.sum()), np.zeros(is_s1.sum())])

rows = {}
print(f"S1 N={is_s1.sum()}  S2 N={is_s2.sum()}")
print(f"{'feature':26s} {'AUROC':>7s} {'S1_med':>9s} {'S2_med':>9s} {'|d|':>7s}")
for name, values in candidates.items():
    values = np.asarray(values, dtype=np.float64)
    pooled = np.concatenate([values[is_s2], values[is_s1]])
    auroc = float(roc_auc_score(labels, pooled))
    s1_med = float(np.median(values[is_s1]))
    s2_med = float(np.median(values[is_s2]))
    pooled_std = float(np.std(pooled)) or 1.0
    effect = abs(s2_med - s1_med) / pooled_std
    rows[name] = {"auroc": auroc, "auroc_flipped": max(auroc, 1 - auroc),
                  "s1_median": s1_med, "s2_median": s2_med, "effect_size": effect}
    print(f"{name:26s} {auroc:7.4f} {s1_med:9.4f} {s2_med:9.4f} {effect:7.4f}")

best = max(rows.items(), key=lambda item: item[1]["auroc_flipped"])
print(f"\n最强单特征：{best[0]}  AUROC(取方向后)={best[1]['auroc_flipped']:.4f}")
(OUT / "probe_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
