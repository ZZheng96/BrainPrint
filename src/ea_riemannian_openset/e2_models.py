"""E2：共享标准化口径的 Ridge、Ledoit–Wolf shrinkage LDA 与余弦原型。"""
from __future__ import annotations

import numpy as np
import torch

from src.open_set_improvement.core import TorchRidge

MODEL_NAMES = ("ridge", "shrinkage_lda", "cosine_prototype")


def ledoit_wolf_shrinkage(x: torch.Tensor) -> torch.Tensor:
    """与 sklearn ledoit_wolf_shrinkage(assume_centered=False) 等价。"""
    if x.ndim != 2 or not len(x):
        raise ValueError("x must be a nonempty 2D tensor")
    if x.shape[1] == 1:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    x = x - x.mean(dim=0)
    n_samples, n_features = x.shape
    x_squared = x.square()
    empirical_trace = x_squared.mean(dim=0)
    mu = empirical_trace.mean()
    beta_raw = (x_squared.T @ x_squared).sum()
    delta_raw = (x.T @ x).square().sum() / n_samples**2
    beta = (beta_raw / n_samples - delta_raw) / (n_features * n_samples)
    delta = (
        delta_raw
        - 2 * mu * empirical_trace.sum()
        + n_features * mu.square()
    ) / n_features
    beta = torch.minimum(beta, delta)
    return torch.where(delta > 0, beta / delta, torch.zeros_like(delta))


class TorchShrinkageLDA:
    """StandardScaler 后按类 Ledoit–Wolf/auto、经验先验加权的 lsqr LDA 分数。"""

    def __init__(self, device: torch.device):
        self.device = device

    def fit(
        self, x: np.ndarray, y: np.ndarray, classes: np.ndarray
    ) -> "TorchShrinkageLDA":
        x_tensor = torch.as_tensor(
            np.asarray(x), dtype=torch.float64, device=self.device
        )
        self.mean = x_tensor.mean(dim=0)
        self.scale = x_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        x_scaled = (x_tensor - self.mean) / self.scale
        self.classes = np.asarray(classes, dtype=int)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device)
        counts = torch.as_tensor(
            [(y == label).sum() for label in self.classes],
            dtype=torch.float64,
            device=self.device,
        )
        self.priors = counts / len(y)
        means = []
        covariance = torch.zeros(
            (x_scaled.shape[1], x_scaled.shape[1]),
            dtype=torch.float64,
            device=self.device,
        )
        self.shrinkages = []
        for index, label in enumerate(self.classes):
            class_x = x_scaled[y_tensor == label]
            class_mean = class_x.mean(dim=0)
            centered = class_x - class_mean
            class_scale = class_x.std(dim=0, unbiased=False)
            class_scale = torch.where(
                class_scale > 0, class_scale, torch.ones_like(class_scale)
            )
            standardized = centered / class_scale
            empirical_cov = standardized.T @ standardized / len(class_x)
            shrinkage = ledoit_wolf_shrinkage(standardized)
            mu = torch.diagonal(empirical_cov).mean()
            shrunk_cov = (
                (1 - shrinkage) * empirical_cov
                + shrinkage * mu * torch.eye(
                    x_scaled.shape[1], dtype=torch.float64, device=self.device
                )
            )
            shrunk_cov = class_scale[:, None] * shrunk_cov * class_scale[None, :]
            covariance += self.priors[index] * shrunk_cov
            means.append(class_mean)
            self.shrinkages.append(float(shrinkage.item()))
        self.means = torch.stack(means)
        self.coef = torch.linalg.solve(covariance, self.means.T).T
        self.intercept = (
            -0.5 * (self.means * self.coef).sum(dim=1) + self.priors.log()
        )
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = (x.to(dtype=torch.float64) - self.mean) / self.scale
        return x_scaled @ self.coef.T + self.intercept


class TorchCosinePrototype:
    """源 fit StandardScaler 后的类均值原型余弦分数。"""

    def __init__(self, device: torch.device):
        self.device = device

    def fit(
        self, x: np.ndarray, y: np.ndarray, classes: np.ndarray
    ) -> "TorchCosinePrototype":
        x_tensor = torch.as_tensor(
            np.asarray(x), dtype=torch.float32, device=self.device
        )
        self.mean = x_tensor.mean(dim=0)
        self.scale = x_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        x_scaled = (x_tensor - self.mean) / self.scale
        self.classes = np.asarray(classes, dtype=int)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device)
        prototypes = torch.stack([
            x_scaled[y_tensor == label].mean(dim=0) for label in self.classes
        ])
        self.prototypes = prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = (x - self.mean) / self.scale
        x_scaled = x_scaled / x_scaled.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return x_scaled @ self.prototypes.T


class E2TaskSpecificModel:
    """每个 task 使用同一源数据分别拟合三种分类器，类别列顺序完全一致。"""

    def __init__(self, device: torch.device, ridge_alpha: float = 100.0):
        self.device = device
        self.ridge_alpha = ridge_alpha

    def fit(
        self,
        x: np.ndarray,
        subjects: np.ndarray,
        tasks: np.ndarray,
        fit_mask: np.ndarray,
    ) -> "E2TaskSpecificModel":
        self.classes = np.unique(np.asarray(subjects)[fit_mask])
        self.models: dict[str, dict[str, object]] = {}
        for task in sorted(np.unique(tasks)):
            indices = np.flatnonzero(fit_mask & (tasks == task))
            x_task = np.asarray(x[indices])
            y_task = np.asarray(subjects)[indices]
            self.models[str(task)] = {
                "ridge": TorchRidge(self.device, self.ridge_alpha).fit(
                    x_task, y_task, self.classes
                ),
                "shrinkage_lda": TorchShrinkageLDA(self.device).fit(
                    x_task, y_task, self.classes
                ),
                "cosine_prototype": TorchCosinePrototype(self.device).fit(
                    x_task, y_task, self.classes
                ),
            }
        return self

    def scores(self, x: torch.Tensor, task: str, model_name: str) -> torch.Tensor:
        if model_name not in MODEL_NAMES:
            raise ValueError(model_name)
        return self.models[task][model_name](x)

    def predict_scores(
        self,
        x: np.ndarray,
        tasks: np.ndarray,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        outputs = {
            name: np.full((len(x), len(self.classes)), np.nan, dtype=np.float64)
            for name in MODEL_NAMES
        }
        for task in sorted(np.unique(tasks)):
            indices = np.flatnonzero(tasks == task)
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                batch = torch.as_tensor(
                    np.asarray(x[selected]), dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    for name in MODEL_NAMES:
                        outputs[name][selected] = self.scores(
                            batch, str(task), name
                        ).cpu().numpy()
        if any(np.isnan(values).any() for values in outputs.values()):
            raise RuntimeError("部分 Trial 没有有效分数")
        return outputs
