"""Native 22-channel multiscale temporal encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def robust_preprocess(raw: torch.Tensor, channels: int = 22) -> torch.Tensor:
    x = raw.float()
    if x.ndim != 3 or tuple(x.shape[1:]) != (channels, 1200):
        raise ValueError(f"Expected Bx{channels}x1200, got {tuple(x.shape)}")
    t = torch.linspace(-1.0, 1.0, 1200, device=x.device, dtype=x.dtype)
    centered = x - x.mean(dim=-1, keepdim=True)
    slope = (centered * t).sum(dim=-1, keepdim=True) / (t.square().sum() + 1e-12)
    centered = centered - slope * t
    centered = centered - centered.median(dim=-1, keepdim=True).values
    mad = 1.4826 * centered.abs().median(dim=-1, keepdim=True).values
    rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
    return torch.nan_to_num(centered / torch.where(mad > 1e-8, mad, rms).clamp_min(1e-8)).unsqueeze(1)


class _ScaleBranch(nn.Module):
    def __init__(self, channels: int, filters: int, kernel: int):
        super().__init__()
        self.temporal = nn.Conv2d(1, filters, (1, kernel), padding=(0, kernel // 2), bias=False)
        self.bn_temporal = nn.BatchNorm2d(filters)
        self.spatial = nn.Conv2d(filters, filters, (channels, 1), groups=filters, bias=False)
        self.bn_spatial = nn.BatchNorm2d(filters)
        self.pool = nn.AvgPool2d((1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_temporal(self.temporal(x))
        x = self.bn_spatial(self.spatial(x))
        return self.pool(F.elu(x))


class MultiScaleNative22(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = (config or {}).get("model", {})
        channels = int((config or {}).get("data", {}).get("channels", 22))
        filters = int(cfg.get("branch_filters", 8))
        self.branches = nn.ModuleList([_ScaleBranch(channels, filters, k) for k in (33, 65, 129)])
        merged = filters * 3
        self.separable_depth = nn.Conv2d(merged, merged, (1, 15), padding=(0, 7), groups=merged, bias=False)
        self.separable_point = nn.Conv2d(merged, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.projection = nn.Linear(32, int(cfg.get("embedding_dim", 64)))
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.2)))

    def embed(self, raw: torch.Tensor) -> torch.Tensor:
        x = robust_preprocess(raw, 22)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.separable_point(self.separable_depth(x))
        x = self.dropout(F.elu(self.bn(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return F.normalize(self.projection(x), dim=1, eps=1e-12)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.embed(raw)
