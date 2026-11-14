"""E62 多窗口时序图像编码器。

动机：E20–E35 的 covariance5 把整段 4 秒压成一张图，跨会话身份证据弱
（Test AUROC(S2 vs S1) 上限 0.7327）。本编码器保留时间结构：
把 4 秒切成 N 段，每段算 5 频带 18×18 log 协方差，按 (段 × 频带) 网格平铺。

所有表示输出 (3, 224, 224) float32，值域 [0,1]，与 E20 的 DINOv2 输入口径一致。
"""
from __future__ import annotations

import numpy as np

FS = 300
N_CHANNELS = 18
IMG_SIZE = 224
BANDS = ((1, 8), (8, 13), (13, 30), (30, 45), (45, 90))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(image.min()), float(image.max())
    if high - low < 1e-8:
        return np.zeros_like(image)
    return (image - low) / (high - low)


def _resize_nearest(image: np.ndarray, out_h: int, out_w: int | None = None) -> np.ndarray:
    """最近邻缩放到 out_h×out_w（out_w 省略时按正方形）。"""
    out_w = out_h if out_w is None else out_w
    height, width = image.shape
    if (height, width) == (out_h, out_w):
        return image.astype(np.float32)
    rows = np.clip(np.round(np.linspace(0, height - 1, out_h)).astype(int), 0, height - 1)
    cols = np.clip(np.round(np.linspace(0, width - 1, out_w)).astype(int), 0, width - 1)
    return image[np.ix_(rows, cols)].astype(np.float32)


def _bandpass_fft(data: np.ndarray, low: float, high: float) -> np.ndarray:
    spectrum = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(data.shape[0], d=1.0 / FS)
    mask = (freqs >= low) & (freqs <= high)
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=data.shape[0], axis=0)


def _log_cov(segment: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    cov = np.cov(segment, rowvar=False, ddof=1)
    target = np.trace(cov) / N_CHANNELS * np.eye(N_CHANNELS)
    cov = (1.0 - shrinkage) * cov + shrinkage * target
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T


def _tile_grid(maps: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """把 len(maps) 张图按 rows×cols 网格平铺成 224×224 灰度三通道。"""
    cell_h, cell_w = IMG_SIZE // rows, IMG_SIZE // cols
    canvas = np.zeros((cell_h * rows, cell_w * cols), dtype=np.float32)
    for index, image in enumerate(maps):
        row, col = divmod(index, cols)
        if row >= rows:
            break
        canvas[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = _resize_nearest(
            image, cell_h, cell_w
        )
    padded = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    padded[: canvas.shape[0], : canvas.shape[1]] = canvas
    return np.stack([padded, padded, padded]).astype(np.float32)


def multiwindow_cov_image(data: np.ndarray, n_windows: int = 8) -> np.ndarray:
    """多窗口协方差图：n_windows 段 × 5 频带，平铺成网格。

    n_windows=8 → 8×5=40 格，用 8 行 × 5 列网格（每格 28×44）。
    """
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    edges = np.linspace(0, data.shape[0], n_windows + 1).astype(int)
    maps = []
    for start, stop in zip(edges[:-1], edges[1:]):
        segment = data[start:stop]
        for low, high in BANDS:
            filtered = _bandpass_fft(segment, low, high)
            maps.append(_normalize_image(_log_cov(filtered)))
    return _tile_grid(maps, rows=n_windows, cols=len(BANDS))


def multiwindow_corr_image(data: np.ndarray, n_windows: int = 8) -> np.ndarray:
    """多窗口相关图：n_windows 段 × 5 频带，每格为通道间相关矩阵。"""
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    edges = np.linspace(0, data.shape[0], n_windows + 1).astype(int)
    maps = []
    for start, stop in zip(edges[:-1], edges[1:]):
        segment = data[start:stop]
        for low, high in BANDS:
            filtered = _bandpass_fft(segment, low, high)
            std = filtered.std(axis=0, keepdims=True)
            std = np.where(std > 1e-12, std, 1.0)
            normalized = filtered / std
            corr = normalized.T @ normalized / max(len(filtered) - 1, 1)
            maps.append(_normalize_image(np.clip(corr, -1, 1)))
    return _tile_grid(maps, rows=n_windows, cols=len(BANDS))


ENCODERS = {
    "multiwindow_cov8": lambda data: multiwindow_cov_image(data, 8),
    "multiwindow_cov4": lambda data: multiwindow_cov_image(data, 4),
    "multiwindow_corr8": lambda data: multiwindow_corr_image(data, 8),
}


def encode(data: np.ndarray, method: str) -> np.ndarray:
    if method not in ENCODERS:
        raise ValueError(f"未知编码器: {method}，可选 {list(ENCODERS)}")
    if data.shape != (1200, N_CHANNELS):
        raise ValueError(f"期望 (1200, {N_CHANNELS})，实际 {data.shape}")
    return ENCODERS[method](data)
