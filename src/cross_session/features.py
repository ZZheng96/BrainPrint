"""协方差表征的公共变换（E21 / E22 共用）。

缓存里的特征是「log 协方差」的上三角向量（非对角乘 sqrt(2)）。
本模块负责：
  - 向量 ↔ 对称矩阵
  - log 协方差 ↔ 协方差（矩阵指数/对数）
  - 逐样本、归纳式的变换：逐频带去对角线均值、逐频带 z-score、相关矩阵 Fisher-z
"""
from __future__ import annotations

import numpy as np

N_CHANNELS = 18
N_BANDS = 6
N_TRIU = N_CHANNELS * (N_CHANNELS + 1) // 2
TRIU = np.triu_indices(N_CHANNELS)
DIAGONAL = TRIU[0] == TRIU[1]
OFF_DIAGONAL = ~DIAGONAL


def to_matrices(vectors: np.ndarray) -> np.ndarray:
    """(N, 6*171) log-cov 向量还原为 (N, 6, 18, 18) 对称矩阵。"""
    values = vectors.reshape(len(vectors), N_BANDS, N_TRIU).astype(np.float64).copy()
    values[..., OFF_DIAGONAL] /= np.sqrt(2.0)
    matrices = np.zeros((len(vectors), N_BANDS, N_CHANNELS, N_CHANNELS))
    matrices[..., TRIU[1], TRIU[0]] = values * OFF_DIAGONAL
    matrices[..., TRIU[0], TRIU[1]] = values
    return matrices


def vectorize(matrices: np.ndarray) -> np.ndarray:
    values = matrices[..., TRIU[0], TRIU[1]].copy()
    values[..., OFF_DIAGONAL] *= np.sqrt(2.0)
    return values.reshape(len(matrices), -1).astype(np.float32)


def cov_from_log(vectors: np.ndarray) -> np.ndarray:
    matrices = to_matrices(vectors)
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    return (eigenvectors * np.exp(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)


def log_from_cov(covariances: np.ndarray) -> np.ndarray:
    matrices = (covariances + np.swapaxes(covariances, -1, -2)) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return (eigenvectors * np.log(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)


def detrend(vectors: np.ndarray) -> np.ndarray:
    """逐频带只从对角线扣掉平均对数方差（与 eeg_features.normalize_log_cov_determinant 一致）。"""
    blocks = vectors.reshape(len(vectors), N_BANDS, N_TRIU).copy()
    mean = blocks[:, :, DIAGONAL].mean(axis=-1, keepdims=True)
    blocks -= mean * DIAGONAL
    return blocks.reshape(len(vectors), -1)


def zscore_bands(vectors: np.ndarray) -> np.ndarray:
    """逐样本、逐频带在 171 维内 z-score。"""
    blocks = vectors.reshape(len(vectors), N_BANDS, N_TRIU).copy()
    mean = blocks.mean(axis=-1, keepdims=True)
    std = blocks.std(axis=-1, keepdims=True)
    return ((blocks - mean) / np.maximum(std, 1e-6)).reshape(len(vectors), -1)


def unit_norm(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-6)


def corr_fisher(vectors: np.ndarray) -> np.ndarray:
    """协方差 → 相关矩阵 → Fisher-z。

    必须先把 log 协方差还原成协方差再归一化；直接在 log 空间除以 sqrt(对角) 没有数学意义。
    对角线置 0（相关系数恒为 1，不含信息），避免常数项进入后续 z-score。
    """
    covariances = cov_from_log(vectors)
    diagonal = np.diagonal(covariances, axis1=-2, axis2=-1)
    scale = np.sqrt(np.maximum(diagonal, 1e-30))
    correlation = covariances / scale[..., :, None] / scale[..., None, :]
    fisher = np.arctanh(np.clip(correlation, -0.999, 0.999))
    fisher = np.nan_to_num(fisher, nan=0.0)
    index = np.arange(N_CHANNELS)
    fisher[..., index, index] = 0.0
    return vectorize(fisher)
