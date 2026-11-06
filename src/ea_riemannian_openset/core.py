"""从缓存 log-cov 特征恢复 SPD，并进行 EA 或仿射不变参考重居中。"""
from __future__ import annotations

import math

import numpy as np
import torch

from src.open_set_calibration.core import predict_calibration_scores
from src.open_set_improvement.core import TaskSpecificModel

N_BANDS = 6
N_CHANNELS = 18
N_TRIU = N_CHANNELS * (N_CHANNELS + 1) // 2
N_FEATURES = N_BANDS * N_TRIU
RECENTER_METHODS = ("ea", "riemannian_recenter")
_TRIU_ROWS, _TRIU_COLS = np.triu_indices(N_CHANNELS)
_OFF_DIAGONAL = _TRIU_ROWS != _TRIU_COLS


def _as_float64_tensor(features, device: torch.device | str | None) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features.to(device=device, dtype=torch.float64)
    return torch.as_tensor(
        np.asarray(features), dtype=torch.float64, device=device
    )


def _validate_log_features(features: torch.Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(f"Expected (n, {N_FEATURES}) log-cov features")


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.transpose(-1, -2)) / 2


def matrix_exp(matrix: torch.Tensor) -> torch.Tensor:
    """批量对称矩阵指数。"""
    eigenvalues, eigenvectors = torch.linalg.eigh(_symmetric(matrix))
    return _symmetric(
        (eigenvectors * torch.exp(eigenvalues).unsqueeze(-2))
        @ eigenvectors.transpose(-1, -2)
    )


def matrix_log(matrix: torch.Tensor) -> torch.Tensor:
    """批量 SPD 矩阵对数。"""
    eigenvalues, eigenvectors = torch.linalg.eigh(_symmetric(matrix))
    eigenvalues = eigenvalues.clamp_min(1e-15)
    return _symmetric(
        (eigenvectors * torch.log(eigenvalues).unsqueeze(-2))
        @ eigenvectors.transpose(-1, -2)
    )


def matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """批量 SPD 对称矩阵幂。"""
    eigenvalues, eigenvectors = torch.linalg.eigh(_symmetric(matrix))
    eigenvalues = eigenvalues.clamp_min(1e-15)
    return _symmetric(
        (eigenvectors * eigenvalues.pow(power).unsqueeze(-2))
        @ eigenvectors.transpose(-1, -2)
    )


def log_features_to_spd(
    features,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 256,
) -> torch.Tensor:
    """将 6×171 维加权上三角 log-cov 还原为 6 个原始 SPD 矩阵。"""
    values = _as_float64_tensor(features, device)
    _validate_log_features(values)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = torch.as_tensor(_TRIU_ROWS, device=values.device)
    cols = torch.as_tensor(_TRIU_COLS, device=values.device)
    off_diagonal = torch.as_tensor(_OFF_DIAGONAL, device=values.device)
    output = torch.empty(
        (len(values), N_BANDS, N_CHANNELS, N_CHANNELS),
        dtype=torch.float64,
        device=values.device,
    )
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].reshape(-1, N_BANDS, N_TRIU)
        matrix = torch.zeros(
            (*batch.shape[:-1], N_CHANNELS, N_CHANNELS),
            dtype=torch.float64,
            device=values.device,
        )
        unweighted = torch.where(
            off_diagonal, batch / math.sqrt(2.0), batch
        )
        matrix[..., rows, cols] = unweighted
        matrix[..., cols, rows] = unweighted
        output[start : start + len(batch)] = matrix_exp(matrix)
    return output


def spd_to_log_features(
    covariances: torch.Tensor,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    """对 SPD 做矩阵 log，并按 Frobenius 权重编码上三角。"""
    if (
        covariances.ndim != 4
        or covariances.shape[1] != N_BANDS
        or covariances.shape[-2:] != (N_CHANNELS, N_CHANNELS)
    ):
        raise ValueError(
            f"Expected (n, {N_BANDS}, {N_CHANNELS}, {N_CHANNELS}) SPD matrices"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    covariances = covariances.to(dtype=torch.float64)
    rows = torch.as_tensor(_TRIU_ROWS, device=covariances.device)
    cols = torch.as_tensor(_TRIU_COLS, device=covariances.device)
    off_diagonal = torch.as_tensor(_OFF_DIAGONAL, device=covariances.device)
    output = torch.empty(
        (len(covariances), N_FEATURES),
        dtype=torch.float64,
        device=covariances.device,
    )
    for start in range(0, len(covariances), batch_size):
        log_cov = matrix_log(covariances[start : start + batch_size])
        vector = log_cov[..., rows, cols]
        vector = torch.where(
            off_diagonal, vector * math.sqrt(2.0), vector
        )
        output[start : start + len(vector)] = vector.reshape(len(vector), -1)
    return output


def riemannian_mean(
    covariances: torch.Tensor,
    *,
    max_iter: int = 50,
    tol: float = 1e-7,
    return_iterations: bool = False,
):
    """SPD 的仿射不变几何均值；在白化切空间做 Karcher 固定点迭代。"""
    if covariances.ndim != 4 or covariances.shape[-2:] != (
        N_CHANNELS,
        N_CHANNELS,
    ):
        raise ValueError("Expected (n, bands, 18, 18) SPD matrices")
    if not len(covariances) or max_iter < 1 or tol <= 0:
        raise ValueError("covariances must be nonempty; max_iter>=1 and tol>0")
    covariances = covariances.to(dtype=torch.float64)
    reference = covariances.mean(dim=0)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        reference_sqrt = matrix_power(reference, 0.5)
        reference_inv_sqrt = matrix_power(reference, -0.5)
        whitened = (
            reference_inv_sqrt.unsqueeze(0)
            @ covariances
            @ reference_inv_sqrt.unsqueeze(0)
        )
        step = matrix_log(whitened).mean(dim=0)
        reference = _symmetric(
            reference_sqrt @ matrix_exp(step) @ reference_sqrt
        )
        step_norm = torch.linalg.matrix_norm(
            step, ord="fro", dim=(-2, -1)
        ).max()
        if float(step_norm.item()) < tol:
            break
    if return_iterations:
        return reference, iterations
    return reference


class SPDRecenter:
    """按 task/band 估计本域参考，并将 SPD 重居中后重新 log-vectorize。"""

    def __init__(
        self,
        method: str,
        device: torch.device | str,
        *,
        batch_size: int = 256,
        max_iter: int = 50,
        tol: float = 1e-7,
    ):
        if method not in RECENTER_METHODS:
            raise ValueError(method)
        self.method = method
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        features: np.ndarray,
        tasks: np.ndarray,
        fit_mask: np.ndarray | None = None,
    ) -> "SPDRecenter":
        """参考只使用 fit_mask；接口不接收 subject、session 或已知/未知标签。"""
        features = np.asarray(features)
        tasks = np.asarray(tasks)
        if len(features) != len(tasks):
            raise ValueError("features/tasks length mismatch")
        if fit_mask is None:
            fit_mask = np.ones(len(tasks), dtype=bool)
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != (len(tasks),):
            raise ValueError("fit_mask must have shape (n,)")
        self.references: dict[str, torch.Tensor] = {}
        self.diagnostics: dict[str, dict] = {}
        for task in sorted(np.unique(tasks[fit_mask])):
            indices = np.flatnonzero(fit_mask & (tasks == task))
            covariances = log_features_to_spd(
                features[indices], device=self.device, batch_size=self.batch_size
            )
            if self.method == "ea":
                reference = covariances.mean(dim=0)
                iterations = 0
            else:
                reference, iterations = riemannian_mean(
                    covariances,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    return_iterations=True,
                )
            self.references[str(task)] = reference
            self.diagnostics[str(task)] = {
                "n_reference_trials": int(len(indices)),
                "reference_iterations": int(iterations),
            }
            del covariances
        if not self.references:
            raise ValueError("No samples available for SPD reference")
        return self

    def transform(self, features: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        tasks = np.asarray(tasks)
        if len(features) != len(tasks):
            raise ValueError("features/tasks length mismatch")
        aligned = np.empty((len(features), N_FEATURES), dtype=np.float32)
        for task in sorted(np.unique(tasks)):
            task_name = str(task)
            if task_name not in self.references:
                raise ValueError(f"No SPD reference for task {task_name}")
            indices = np.flatnonzero(tasks == task)
            covariances = log_features_to_spd(
                features[indices], device=self.device, batch_size=self.batch_size
            )
            reference_inv_sqrt = matrix_power(
                self.references[task_name], -0.5
            )
            recentered = (
                reference_inv_sqrt.unsqueeze(0)
                @ covariances
                @ reference_inv_sqrt.unsqueeze(0)
            )
            task_features = spd_to_log_features(
                recentered, batch_size=self.batch_size
            )
            aligned[indices] = task_features.cpu().numpy().astype(np.float32)
            del covariances, recentered, task_features
        return aligned


def fit_recentered_model(
    x: np.ndarray,
    window: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fit_mask: np.ndarray,
    method: str,
    device: torch.device,
    *,
    batch_size: int = 256,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> tuple[TaskSpecificModel, SPDRecenter]:
    """源参考及源训练特征均只由 fit_mask 决定，再复用 TaskSpecificModel。"""
    fit_mask = np.asarray(fit_mask, dtype=bool)
    source_aligner = SPDRecenter(
        method,
        device,
        batch_size=batch_size,
        max_iter=max_iter,
        tol=tol,
    ).fit(x, tasks, fit_mask)
    fit_indices = np.flatnonzero(fit_mask)
    source_x = source_aligner.transform(x[fit_indices], tasks[fit_indices])
    model = TaskSpecificModel(device).fit(
        source_x,
        np.asarray(window)[fit_indices],
        np.asarray(subjects)[fit_indices],
        np.asarray(tasks)[fit_indices],
        np.ones(len(fit_indices), dtype=bool),
    )
    return model, source_aligner


def predict_recentered_full_scores(
    model: TaskSpecificModel,
    source_aligner: SPDRecenter,
    x: np.ndarray,
    window: np.ndarray,
    tasks: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict]:
    """目标参考由完整无标签 eval 批次估计；只返回 full Ridge 分数。"""
    target_aligner = SPDRecenter(
        source_aligner.method,
        model.device,
        batch_size=batch_size,
        max_iter=source_aligner.max_iter,
        tol=source_aligner.tol,
    ).fit(x, tasks)
    target_x = target_aligner.transform(x, tasks)
    outputs, _ = predict_calibration_scores(
        model, target_x, window, tasks, "none", batch_size=batch_size
    )
    diagnostics = {
        "source": source_aligner.diagnostics,
        "target": target_aligner.diagnostics,
    }
    return outputs["full_none"], diagnostics
