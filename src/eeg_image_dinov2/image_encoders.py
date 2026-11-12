"""四种 EEG 图像表示：时频图、通道相关矩阵、头皮拓扑图、波形 GAF/MTF。

所有表示统一输出 (3, 224, 224) float32，值域 [0,1]，供 DINOv2 使用。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import stft

FS = 300
N_CHANNELS = 18
IMG_SIZE = 224

# 10-20 系统电极坐标（x 右为正，y 前为正），单位归一化到 [-1, 1]
CHANNEL_POSITIONS = {
    "Fp1": (-0.30, 0.95), "Fp2": (0.30, 0.95),
    "F7": (-0.80, 0.60), "F3": (-0.40, 0.60), "Fz": (0.0, 0.60),
    "F4": (0.40, 0.60), "F8": (0.80, 0.60),
    "T3": (-0.95, 0.0), "C3": (-0.45, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.45, 0.0), "T4": (0.95, 0.0),
    "T5": (-0.80, -0.60), "P3": (-0.40, -0.60), "Pz": (0.0, -0.60),
    "P4": (0.40, -0.60), "T6": (0.80, -0.60),
    "O1": (-0.30, -0.95), "O2": (0.30, -0.95),
}
# 数据实际通道顺序（与 eeg_features.load_eeg 一致）
CHANNEL_ORDER = (
    "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz",
    "Fp1", "Fp2", "T3", "T5", "O1", "O2", "F7", "F8", "T6", "T4",
)
BANDS = ((1, 8), (8, 13), (13, 30), (30, 45), (45, 90))
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """逐图 min-max 到 [0,1]，非有限值置 0。"""
    image = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(image.min()), float(image.max())
    if high - low < 1e-8:
        return np.zeros_like(image)
    return (image - low) / (high - low)


def _resize_to(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """双线性插值到 size×size（输入 H×W）。"""
    if image.shape == (size, size):
        return image.astype(np.float32)
    height, width = image.shape
    rows = np.linspace(0, height - 1, size)
    cols = np.linspace(0, width - 1, size)
    row_idx = np.clip(np.round(rows).astype(int), 0, height - 1)
    col_idx = np.clip(np.round(cols).astype(int), 0, width - 1)
    return image[np.ix_(row_idx, col_idx)].astype(np.float32)


def _bandpass_fft(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """FFT 频域带通，data 形状 (T, C)。"""
    spectrum = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(data.shape[0], d=1.0 / FS)
    mask = (freqs >= low) & (freqs <= high)
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=data.shape[0], axis=0)


def spectrogram_image(data: np.ndarray) -> np.ndarray:
    """V1：多频带时频图。

    18 通道 × 5 频带，每格 STFT 时频能量；R/G/B 分别聚合低/中/高频段。
    """
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    # 每个频带一张 (C, F, T) 时频图
    band_maps = []
    for low, high in BANDS:
        filtered = _bandpass_fft(data, low, high)
        _, _, zxx = stft(filtered, fs=FS, nperseg=128, noverlap=96, axis=0)
        power = np.abs(zxx) ** 2  # (F, C, T)
        power = np.log1p(power)
        # 频率方向压缩到固定高度，时间方向压缩到固定宽度
        freq_bins, n_ch, time_bins = power.shape
        power = power.transpose(1, 0, 2).reshape(n_ch, freq_bins * time_bins)
        band_maps.append(power)
    band_maps = np.stack(band_maps)  # (5, C, F*T)
    # 每个频带归一化后按通道堆叠成 (C, 5*F*T)
    normalized = np.stack([_normalize_image(m) for m in band_maps])
    # 通道方向堆叠：行=通道，列=频带×时频
    stacked = normalized.transpose(1, 0, 2).reshape(N_CHANNELS, -1)
    # 3 通道：低频段(1-13)、中频段(13-45)、高频段(45-90)
    low_map = normalized[0:2].mean(axis=0)
    mid_map = normalized[2:4].mean(axis=0)
    high_map = normalized[4]
    rgb = np.stack([
        _resize_to(_normalize_image(low_map)),
        _resize_to(_normalize_image(mid_map)),
        _resize_to(_normalize_image(high_map)),
    ])
    return rgb.astype(np.float32)


def correlation_image(data: np.ndarray) -> np.ndarray:
    """V2：通道相关矩阵图。

    5 个频带各一张 18×18 相关矩阵，取 3 个代表频带堆叠为 RGB。
    """
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    matrices = []
    for low, high in BANDS:
        filtered = _bandpass_fft(data, low, high)
        std = filtered.std(axis=0, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0)
        normalized = filtered / std
        corr = normalized.T @ normalized / max(len(filtered) - 1, 1)
        matrices.append(np.clip(corr, -1, 1))
    matrices = np.stack(matrices)  # (5, C, C)
    # 3 通道：低频(1-13)、中频(13-45)、高频(45-90)
    low_map = matrices[0:2].mean(axis=0)
    mid_map = matrices[2:4].mean(axis=0)
    high_map = matrices[4]
    rgb = np.stack([
        _resize_to(_normalize_image(low_map)),
        _resize_to(_normalize_image(mid_map)),
        _resize_to(_normalize_image(high_map)),
    ])
    return rgb.astype(np.float32)


def _tile_5_bands(maps: list[np.ndarray]) -> np.ndarray:
    """把 5 张图按 2×3 网格平铺，第 6 格留空，输出 224×224 灰度三通道。"""
    cell_h, cell_w = IMG_SIZE // 2, IMG_SIZE // 3
    canvas = np.zeros((cell_h * 2, cell_w * 3), dtype=np.float32)
    for index, image in enumerate(maps):
        row, col = divmod(index, 3)
        resized = _resize_to(image, cell_h)[:cell_h, :cell_w]
        canvas[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = resized
    # 右侧补 2 列到 224
    padded = np.zeros((cell_h * 2, IMG_SIZE), dtype=np.float32)
    padded[:, : canvas.shape[1]] = canvas
    return np.stack([padded, padded, padded]).astype(np.float32)


def topomap5_image(data: np.ndarray, grid: int = 64) -> np.ndarray:
    """V3b：五频带头皮拓扑图，5 张地形图按 2×3 网格平铺成 224×224。

    相比 topomap_image 的 3 通道压缩，这里保留全部 5 个频带，用灰度平铺。
    """
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    positions = np.array([CHANNEL_POSITIONS[name] for name in CHANNEL_ORDER])
    band_power = []
    for low, high in BANDS:
        filtered = _bandpass_fft(data, low, high)
        band_power.append(np.log1p(np.mean(filtered ** 2, axis=0)))
    band_power = np.stack(band_power)

    axis = np.linspace(-1, 1, grid)
    grid_x, grid_y = np.meshgrid(axis, axis)
    mask = grid_x ** 2 + grid_y ** 2 <= 1.0
    maps = []
    for power in band_power:
        interpolated = griddata(
            positions, power, (grid_x, grid_y), method="cubic", fill_value=float(power.mean())
        )
        interpolated = np.nan_to_num(interpolated, nan=float(power.mean()))
        interpolated = np.where(mask, interpolated, float(power.mean()))
        maps.append(_normalize_image(interpolated))
    return _tile_5_bands(maps)


def covariance5_image(data: np.ndarray) -> np.ndarray:
    """V2b：五频带对数协方差矩阵，5 张 18×18 按 2×3 网格平铺成 224×224。

    与既往协方差基线同源，但以图像形式输入视觉模型。
    """
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    matrices = []
    for low, high in BANDS:
        filtered = _bandpass_fft(data, low, high)
        cov = np.cov(filtered, rowvar=False, ddof=1)
        target = np.trace(cov) / N_CHANNELS * np.eye(N_CHANNELS)
        cov = 0.95 * cov + 0.05 * target
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        log_cov = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
        matrices.append(_normalize_image(log_cov))
    return _tile_5_bands(matrices)


def topomap_image(data: np.ndarray, grid: int = 64) -> np.ndarray:
    """V3：头皮拓扑图序列。

    按 10-20 坐标把 18 通道频带功率插值成 2D 地形图，3 个代表频带堆叠为 RGB。
    """
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    positions = np.array([CHANNEL_POSITIONS[name] for name in CHANNEL_ORDER])
    band_power = []
    for low, high in BANDS:
        filtered = _bandpass_fft(data, low, high)
        band_power.append(np.log1p(np.mean(filtered ** 2, axis=0)))
    band_power = np.stack(band_power)  # (5, C)

    axis = np.linspace(-1, 1, grid)
    grid_x, grid_y = np.meshgrid(axis, axis)
    maps = []
    for power in band_power:
        interpolated = griddata(positions, power, (grid_x, grid_y), method="cubic", fill_value=float(power.mean()))
        interpolated = np.nan_to_num(interpolated, nan=float(power.mean()))
        # 圆形头皮掩膜
        mask = grid_x ** 2 + grid_y ** 2 <= 1.0
        interpolated = np.where(mask, interpolated, float(power.mean()))
        maps.append(interpolated)
    maps = np.stack(maps)
    low_map = maps[0:2].mean(axis=0)
    mid_map = maps[2:4].mean(axis=0)
    high_map = maps[4]
    rgb = np.stack([
        _resize_to(_normalize_image(low_map)),
        _resize_to(_normalize_image(mid_map)),
        _resize_to(_normalize_image(high_map)),
    ])
    return rgb.astype(np.float32)


def _gaf(series: np.ndarray) -> np.ndarray:
    """Gramian Angular Field（求和场），输入一维序列。"""
    series = np.asarray(series, dtype=np.float64)
    low, high = series.min(), series.max()
    if high - low < 1e-12:
        return np.zeros((len(series), len(series)), dtype=np.float32)
    scaled = 2 * (series - low) / (high - low) - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def _mtf(series: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Markov Transition Field，输入一维序列。"""
    series = np.asarray(series, dtype=np.float64)
    low, high = series.min(), series.max()
    if high - low < 1e-12:
        return np.zeros((len(series), len(series)), dtype=np.float32)
    bins = np.clip(((series - low) / (high - low) * n_bins).astype(int), 0, n_bins - 1)
    transition = np.zeros((n_bins, n_bins), dtype=np.float64)
    for left, right in zip(bins[:-1], bins[1:]):
        transition[left, right] += 1
    row_sums = transition.sum(axis=1, keepdims=True)
    transition = transition / np.where(row_sums > 0, row_sums, 1.0)
    return transition[bins][:, bins].astype(np.float32)


def waveform_image(data: np.ndarray, method: str = "gaf") -> np.ndarray:
    """V4：原始波形图（GAF 或 MTF）。

    18 通道降采样到 224 点后做 GAF/MTF，按额/中央/枕三组平均堆叠为 RGB。
    """
    data = np.asarray(data, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    data = data / np.where(std > 1e-12, std, 1.0)
    # 降采样到 224 点
    indices = np.linspace(0, len(data) - 1, IMG_SIZE).astype(int)
    downsampled = data[indices]  # (224, C)
    transform = _gaf if method == "gaf" else _mtf
    channel_maps = np.stack([transform(downsampled[:, c]) for c in range(N_CHANNELS)])
    # 按电极位置分组：额区、中央区、枕区
    frontal = [CHANNEL_ORDER.index(n) for n in ("Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8")]
    central = [CHANNEL_ORDER.index(n) for n in ("C3", "C4", "Cz", "T3", "T4")]
    posterior = [CHANNEL_ORDER.index(n) for n in ("P3", "P4", "O1", "O2", "T5", "T6")]
    rgb = np.stack([
        _normalize_image(channel_maps[frontal].mean(axis=0)),
        _normalize_image(channel_maps[central].mean(axis=0)),
        _normalize_image(channel_maps[posterior].mean(axis=0)),
    ])
    return rgb.astype(np.float32)


ENCODERS = {
    "spectrogram": spectrogram_image,
    "correlation": correlation_image,
    "topomap": topomap_image,
    "topomap5": topomap5_image,
    "covariance5": covariance5_image,
    "gaf": lambda data: waveform_image(data, "gaf"),
    "mtf": lambda data: waveform_image(data, "mtf"),
}


def encode(data: np.ndarray, method: str) -> np.ndarray:
    """把 (1200, 18) EEG 转成 (3, 224, 224) 图像。"""
    if method not in ENCODERS:
        raise ValueError(f"未知图像表示: {method}，可选 {list(ENCODERS)}")
    if data.shape != (1200, N_CHANNELS):
        raise ValueError(f"期望 (1200, {N_CHANNELS})，实际 {data.shape}")
    return ENCODERS[method](data)
