"""轻量身份头：投影 + ArcFace。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace 角度间隔分类器。"""

    def __init__(self, in_features: int, out_features: int, margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


class IdentityHead(nn.Module):
    """冻结骨干之上的轻量投影头 + ArcFace。"""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        n_classes: int = 20,
        dropout: float = 0.3,
        margin: float = 0.3,
        scale: float = 30.0,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.arcface = ArcMarginProduct(embed_dim, n_classes, margin, scale)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(features), dim=1)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.arcface(self.embed(features), labels)
