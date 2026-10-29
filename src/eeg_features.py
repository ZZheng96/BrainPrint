"""4 秒多通道 EEG 的稳健频谱与对数协方差特征。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

FS = 300
FREQ_MIN = 1
FREQ_MAX = 90
COV_BANDS = ((1, 8), (8, 13), (13, 30), (30, 45), (45, 70), (70, 90))
N_CHANNELS = 18
N_PSD_FEATURES = N_CHANNELS * (FREQ_MAX - FREQ_MIN + 1)
N_COV_FEATURES = N_CHANNELS * (N_CHANNELS + 1) // 2 * len(COV_BANDS)
N_FEATURES = N_PSD_FEATURES + N_COV_FEATURES

_FULL_BAND_SOS = butter(4, (0.5, FREQ_MAX), btype="bandpass", fs=FS, output="sos")
_NOTCH_B, _NOTCH_A = iirnotch(50, Q=30, fs=FS)
_BAND_SOS = {
    band: butter(4, band, btype="bandpass", fs=FS, output="sos")
    for band in COV_BANDS
}
_TRIU = np.triu_indices(N_CHANNELS)
_OFF_DIAGONAL = _TRIU[0] != _TRIU[1]
_DIAGONAL = ~_OFF_DIAGONAL


def normalize_log_cov_determinant(cov_features: np.ndarray) -> np.ndarray:
    """逐频带扣除平均对数方差，去除 Trial 全局功率尺度。"""
    if cov_features.shape[-1] != N_COV_FEATURES:
        raise ValueError(f"Expected last dimension {N_COV_FEATURES}")
    normalized = np.array(cov_features, copy=True)
    bands = normalized.reshape(*normalized.shape[:-1], len(COV_BANDS), -1)
    diagonal_mean = bands[..., _DIAGONAL].mean(axis=-1, keepdims=True)
    bands -= diagonal_mean * _DIAGONAL
    return normalized


def scalar_text(value) -> str:
    return str(value.reshape(-1)[0])


def load_eeg(path: str | Path) -> tuple[np.ndarray, dict]:
    mat = loadmat(path)
    data = np.asarray(mat["data"], dtype=np.float64)
    if data.ndim != 2 or data.shape != (1200, N_CHANNELS):
        raise ValueError(f"{path}: expected (1200, 18), got {data.shape}")
    metadata = {
        "fs": int(mat["fs"].reshape(-1)[0]),
        "session": scalar_text(mat["session_id"]),
        "task": scalar_text(mat["task_label"]),
        "channels": tuple(str(x.reshape(-1)[0]) for x in mat["channel_names"].reshape(-1)),
    }
    if metadata["fs"] != FS:
        raise ValueError(f"{path}: expected fs={FS}, got {metadata['fs']}")
    return data, metadata


def robust_clip(data: np.ndarray, n_mad: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """逐通道截尾，返回处理后信号和每个通道的截尾比例。"""
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    scale = 1.4826 * mad
    std = np.std(data, axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    lower = median - n_mad * scale
    upper = median + n_mad * scale
    clipped = np.clip(np.nan_to_num(data), lower, upper)
    clip_fraction = np.mean((data < lower) | (data > upper) | ~np.isfinite(data), axis=0)
    return clipped, clip_fraction


def log_covariance(data: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    cov = np.cov(data, rowvar=False, ddof=1)
    target = np.trace(cov) / N_CHANNELS * np.eye(N_CHANNELS)
    cov = (1.0 - shrinkage) * cov + shrinkage * target
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    log_cov = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    vector = log_cov[_TRIU]
    vector[_OFF_DIAGONAL] *= np.sqrt(2.0)
    return vector


def extract_features(data: np.ndarray) -> tuple[np.ndarray, dict]:
    clipped, clip_fraction = robust_clip(data)
    filtered = sosfiltfilt(_FULL_BAND_SOS, clipped, axis=0)
    filtered = filtfilt(_NOTCH_B, _NOTCH_A, filtered, axis=0)

    frequencies, psd = welch(
        filtered,
        fs=FS,
        window="hann",
        nperseg=300,
        noverlap=150,
        detrend="linear",
        axis=0,
    )
    selected = (frequencies >= FREQ_MIN) & (frequencies <= FREQ_MAX)
    log_psd = np.log10(np.maximum(psd[selected], 1e-20)).T.reshape(-1)

    covariances = []
    for band in COV_BANDS:
        band_data = sosfiltfilt(_BAND_SOS[band], filtered, axis=0)
        covariances.append(log_covariance(band_data))
    features = np.concatenate([log_psd, *covariances]).astype(np.float32)
    if features.shape != (N_FEATURES,):
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")

    quality = {
        "raw_std_mean": float(np.mean(np.std(data, axis=0))),
        "filtered_std_mean": float(np.mean(np.std(filtered, axis=0))),
        "clip_fraction_mean": float(np.mean(clip_fraction)),
        "clip_fraction_max": float(np.max(clip_fraction)),
        "nonfinite_fraction": float(np.mean(~np.isfinite(data))),
    }
    return features, quality
