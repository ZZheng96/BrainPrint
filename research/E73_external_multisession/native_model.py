"""Channel-count-compatible E54 encoder for the native 22-channel external montage."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

E73_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E73_DIR.parents[1] / "research" / "E54_outlier_exposure"))
from run_e54 import EEGNetEmbedding  # noqa: E402


def robust_preprocess_native(raw: torch.Tensor, channels: int, floor: float = 1e-8) -> torch.Tensor:
    x = raw.float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != channels or x.shape[3] != 1200:
        raise ValueError(f"Expected Bx1x{channels}x1200, got {tuple(x.shape)}")
    t = torch.linspace(-1.0, 1.0, 1200, device=x.device, dtype=x.dtype)
    centered = x - x.mean(dim=-1, keepdim=True)
    slope = (centered * t).sum(dim=-1, keepdim=True) / (t * t).sum()
    detrended = centered - slope * t
    median = detrended.median(dim=-1, keepdim=True).values
    centered = detrended - median
    mad = 1.4826 * centered.abs().median(dim=-1, keepdim=True).values
    rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
    scale = torch.where(mad > floor, mad, rms).clamp_min(floor)
    return torch.nan_to_num(centered / scale)


class NativeEEGNetEmbedding(EEGNetEmbedding):
    def __init__(self, config: dict):
        super().__init__(n_classes=1, config=config)
        cfg = config["model"]
        channels = int(config["data"]["channels"])
        f1 = int(cfg["temporal_filters"])
        depth = int(cfg["depthwise_multiplier"])
        self.channels = channels
        self.depthwise = torch.nn.Conv2d(
            f1, f1 * depth, (channels, 1), groups=f1, bias=False
        )

    def embed(self, raw: torch.Tensor) -> torch.Tensor:
        x = robust_preprocess_native(raw, self.channels, self.mad_floor)
        x = self.bn1(self.temporal(x))
        x = self.bn2(self.depthwise(x))
        x = self.dropout(self.pool(F.elu(x)))
        x = self.separable_point(self.separable_depth(x))
        x = self.bn3(x)
        x = self.dropout(self.pool(F.elu(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.projection(x)
        return F.normalize(x, dim=1, eps=1e-12)
