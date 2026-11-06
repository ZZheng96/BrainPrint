"""可复现的纯 EEG 开集模型、指标与特征空间攻击。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.metrics import decision_scores, score_predictions

ROOT = Path(__file__).resolve().parents[2]
N_PSD_FEATURES = 18 * 90
COV_COLUMNS = np.arange(N_PSD_FEATURES, N_PSD_FEATURES + 6 * 171)
SCORE_NAMES = ("max_logit", "margin", "energy")
VARIANTS = ("full", "base", "fused")


def resolve_cache() -> Path:
    for path in (ROOT / "BrainPrint_cache", ROOT.parent / "BrainPrint_cache"):
        if (path / "train_features.npy").exists():
            return path
    raise FileNotFoundError("找不到 BrainPrint_cache/train_features.npy")


def row_zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


class TorchRidge:
    """与 StandardScaler + RidgeClassifier 一致的多类线性评分器。"""

    def __init__(self, device: torch.device, alpha: float = 100.0):
        self.device = device
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> "TorchRidge":
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.mean = x_tensor.mean(dim=0)
        self.scale = x_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        x_scaled = (x_tensor - self.mean) / self.scale
        self.classes = np.asarray(classes, dtype=int)
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        targets = torch.full(
            (len(y), len(classes)), -1.0, dtype=torch.float32, device=self.device
        )
        targets[torch.arange(len(y), device=self.device), torch.as_tensor(
            [class_to_index[int(label)] for label in y], device=self.device
        )] = 1.0
        self.target_mean = targets.mean(dim=0)
        centered_targets = targets - self.target_mean
        if len(y) < x_scaled.shape[1]:
            kernel = x_scaled @ x_scaled.T
            kernel.diagonal().add_(self.alpha)
            self.coef = x_scaled.T @ torch.linalg.solve(kernel, centered_targets)
        else:
            gram = x_scaled.T @ x_scaled
            gram.diagonal().add_(self.alpha)
            self.coef = torch.linalg.solve(gram, x_scaled.T @ centered_targets)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.scale) @ self.coef + self.target_mean


class TorchNeighbor:
    """类内 top-3 余弦相似度，前向与攻击均在 GPU 上执行。"""

    def __init__(self, device: torch.device):
        self.device = device

    def fit(self, x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> "TorchNeighbor":
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.mean = x_tensor.mean(dim=0)
        self.scale = x_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        train = (x_tensor - self.mean) / self.scale
        self.train = train / train.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.classes = np.asarray(classes, dtype=int)
        self.train_targets = torch.as_tensor(y, dtype=torch.long, device=self.device)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        validation = (x - self.mean) / self.scale
        validation = validation / validation.norm(dim=1, keepdim=True).clamp_min(1e-12)
        similarities = validation @ self.train.T
        outputs = []
        for label in self.classes:
            class_scores = similarities[:, self.train_targets == label]
            k = min(3, class_scores.shape[1])
            outputs.append(class_scores.topk(k, dim=1).values.mean(dim=1))
        return torch.stack(outputs, dim=1)


@dataclass
class TaskModels:
    full: TorchRidge
    window: TorchRidge
    low: TorchRidge
    low_neighbor: TorchNeighbor
    delta_neighbor: TorchNeighbor
    attack_scale: torch.Tensor


class TaskSpecificModel:
    """固定任务特异融合；不读取 session，不做批次对齐或类别平衡。"""

    def __init__(self, device: torch.device, alpha: float = 100.0):
        self.device = device
        self.alpha = alpha

    def fit(
        self,
        x: np.ndarray,
        window: np.ndarray,
        subjects: np.ndarray,
        tasks: np.ndarray,
        fit_mask: np.ndarray,
    ) -> "TaskSpecificModel":
        self.classes = np.unique(subjects[fit_mask])
        self.models: dict[str, TaskModels] = {}
        for task in sorted(np.unique(tasks)):
            indices = np.flatnonzero(fit_mask & (tasks == task))
            if not len(indices):
                continue
            x_task = np.asarray(x[indices])
            y_task = subjects[indices]
            window_task = np.asarray(window[indices]).reshape(-1, x.shape[1])
            window_targets = np.repeat(y_task, 2)
            self.models[task] = TaskModels(
                full=TorchRidge(self.device, self.alpha).fit(x_task, y_task, self.classes),
                window=TorchRidge(self.device, self.alpha).fit(
                    window_task, window_targets, self.classes
                ),
                low=TorchRidge(self.device, self.alpha).fit(
                    x_task[:, :342], y_task, self.classes
                ),
                low_neighbor=TorchNeighbor(self.device).fit(
                    x_task[:, :342], y_task, self.classes
                ),
                delta_neighbor=TorchNeighbor(self.device).fit(
                    x_task[:, :171], y_task, self.classes
                ),
                attack_scale=torch.as_tensor(
                    x_task.std(axis=0), dtype=torch.float32, device=self.device
                ).clamp_min(1e-6),
            )
        return self

    def scores(
        self, x: torch.Tensor, window: torch.Tensor, task: str, variant: str = "fused"
    ) -> torch.Tensor:
        models = self.models[task]
        full = models.full(x)
        if variant == "full":
            return full
        window_scores = (models.window(window[:, 0]) + models.window(window[:, 1])) / 2
        base = 0.5 * full + 0.5 * window_scores
        if variant == "base":
            return base
        if variant != "fused":
            raise ValueError(variant)
        if task == "EC":
            return (
                row_zscore(base)
                + 0.25 * row_zscore(models.low_neighbor(x[:, :342]))
                + 0.5 * row_zscore(models.delta_neighbor(x[:, :171]))
            )
        if task == "EO":
            return base + 0.25 * models.low(x[:, :342])
        if task == "Imagery":
            return window_scores
        if task == "SSVEP":
            return base + 0.5 * models.low(x[:, :342])
        raise ValueError(task)

    def predict_scores(
        self,
        x: np.ndarray,
        window: np.ndarray,
        tasks: np.ndarray,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        outputs = {
            variant: np.full((len(x), len(self.classes)), np.nan, dtype=np.float32)
            for variant in VARIANTS
        }
        for task in sorted(np.unique(tasks)):
            indices = np.flatnonzero(tasks == task)
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                x_batch = torch.as_tensor(
                    np.asarray(x[selected]), dtype=torch.float32, device=self.device
                )
                window_batch = torch.as_tensor(
                    np.asarray(window[selected]), dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    for variant in VARIANTS:
                        outputs[variant][selected] = (
                            self.scores(x_batch, window_batch, task, variant)
                            .cpu().numpy()
                        )
        if any(np.isnan(values).any() for values in outputs.values()):
            raise RuntimeError("部分 Trial 没有有效分数")
        return outputs


def confidence_values(scores: np.ndarray) -> dict[str, np.ndarray]:
    return decision_scores(np.asarray(scores, dtype=np.float64))


def select_operating_point(
    known_scores: list[np.ndarray], unknown_scores: list[np.ndarray]
) -> dict:
    """仅用内层数据选择分数；阈值固定为内层已知分数的 5% 分位。"""
    known = np.concatenate(known_scores)
    unknown = np.concatenate(unknown_scores)
    candidates = []
    for name in SCORE_NAMES:
        known_confidence = confidence_values(known)[name]
        unknown_confidence = confidence_values(unknown)[name]
        threshold = float(np.quantile(known_confidence, 0.05))
        true = np.concatenate([np.ones(len(known_confidence)), np.zeros(len(unknown_confidence))])
        accepted = np.concatenate([known_confidence, unknown_confidence]) >= threshold
        candidates.append({
            "score_name": name,
            "threshold": threshold,
            "inner_detection_macro_f1": float(f1_score(true, accepted, average="macro")),
            "inner_unknown_f1": float(f1_score(true, accepted, pos_label=0)),
            "inner_known_acceptance": float(np.mean(known_confidence >= threshold)),
            "inner_U": float(np.mean(unknown_confidence < threshold)),
            "inner_auroc": float(roc_auc_score(
                true, np.concatenate([known_confidence, unknown_confidence])
            )),
        })
    return max(
        candidates,
        key=lambda row: (
            row["inner_detection_macro_f1"], row["inner_unknown_f1"], row["inner_auroc"]
        ),
    )


def open_set_report(
    known_true: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    classes: np.ndarray,
    score_name: str,
    threshold: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    known_confidence = confidence_values(known_scores)[score_name]
    unknown_confidence = confidence_values(unknown_scores)[score_name]
    known_pred = classes[known_scores.argmax(axis=1)].copy()
    known_pred[known_confidence < threshold] = 0
    unknown_pred = classes[unknown_scores.argmax(axis=1)].copy()
    unknown_pred[unknown_confidence < threshold] = 0
    true = np.concatenate([known_true, np.zeros(len(unknown_pred), dtype=int)])
    pred = np.concatenate([known_pred, unknown_pred])
    detection_true = np.concatenate([
        np.ones(len(known_pred), dtype=int), np.zeros(len(unknown_pred), dtype=int)
    ])
    detection_scores = np.concatenate([known_confidence, unknown_confidence])
    fpr, tpr, _ = roc_curve(detection_true, detection_scores)
    fnr = 1 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    labels = np.unique(np.concatenate([true, pred]))
    report = {
        "n_known": int(len(known_true)),
        "n_unknown": int(len(unknown_pred)),
        "C": float(np.mean(known_pred == known_true)),
        "direct_C_without_rejection": float(np.mean(
            classes[known_scores.argmax(axis=1)] == known_true
        )),
        "U": float(np.mean(unknown_pred == 0)),
        "known_acceptance": float(np.mean(known_confidence >= threshold)),
        "unknown_auroc": float(roc_auc_score(detection_true, detection_scores)),
        "known_aupr": float(average_precision_score(detection_true, detection_scores)),
        "unknown_aupr": float(average_precision_score(1 - detection_true, -detection_scores)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "unknown_f1": float(f1_score(
            detection_true, detection_scores >= threshold, pos_label=0, zero_division=0
        )),
        "detection_macro_f1": float(f1_score(
            detection_true, detection_scores >= threshold, average="macro", zero_division=0
        )),
        "score_name": score_name,
        "threshold": float(threshold),
    }
    records = {
        "true": true,
        "pred": pred,
        "is_known": detection_true.astype(bool),
        "confidence": detection_scores,
    }
    return report, records


def competition_report(
    clean_true: np.ndarray,
    clean_scores: np.ndarray,
    adversarial_true: np.ndarray,
    adversarial_scores: np.ndarray,
    unknown_scores: np.ndarray,
    classes: np.ndarray,
    score_name: str,
    threshold: float,
    label_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    clean_confidence = confidence_values(clean_scores)[score_name]
    adversarial_confidence = confidence_values(adversarial_scores)[score_name]
    unknown_confidence = confidence_values(unknown_scores)[score_name]
    clean_pred = classes[clean_scores.argmax(axis=1)].copy()
    adversarial_pred = classes[adversarial_scores.argmax(axis=1)].copy()
    if label_override is not None:
        clean_pred = label_override[0].copy()
        adversarial_pred = label_override[1].copy()
    unknown_pred = classes[unknown_scores.argmax(axis=1)].copy()
    clean_pred[clean_confidence < threshold] = 0
    adversarial_pred[adversarial_confidence < threshold] = 0
    unknown_pred[unknown_confidence < threshold] = 0
    report = score_predictions(
        clean_true, clean_pred, adversarial_true, adversarial_pred, unknown_pred
    )
    report.update({
        "n_known": int(len(clean_true)),
        "n_unknown": int(len(unknown_pred)),
        "clean_macro_f1_with_reject": float(f1_score(
            clean_true, clean_pred, average="macro", zero_division=0
        )),
        "adversarial_macro_f1_with_reject": float(f1_score(
            adversarial_true, adversarial_pred, average="macro", zero_division=0
        )),
        "unknown_f1": float(f1_score(
            np.zeros(len(unknown_pred), dtype=int), unknown_pred,
            labels=[0], average=None, zero_division=0
        )[0]) if len(unknown_pred) else float("nan"),
    })
    return report


def attack_batch(
    model: TaskSpecificModel,
    x: np.ndarray,
    window: np.ndarray,
    task: str,
    targets: np.ndarray,
    epsilon: float,
    method: str,
    steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """对模型输入特征做 L∞ FGSM/PGD；epsilon 单位为训练特征标准差。"""
    if epsilon <= 0 or method not in {"fgsm", "pgd"}:
        raise ValueError("epsilon 必须为正，method 必须为 fgsm 或 pgd")
    clean_x = torch.as_tensor(x, dtype=torch.float32, device=model.device)
    clean_window = torch.as_tensor(window, dtype=torch.float32, device=model.device)
    scale = model.models[task].attack_scale
    class_to_index = {label: index for index, label in enumerate(model.classes)}
    target = torch.as_tensor(
        [class_to_index[int(label)] for label in targets],
        dtype=torch.long,
        device=model.device,
    )
    delta_x = torch.zeros_like(clean_x, requires_grad=True)
    delta_window = torch.zeros_like(clean_window, requires_grad=True)
    iterations = 1 if method == "fgsm" else steps
    step_size = epsilon if method == "fgsm" else 2 * epsilon / steps
    for _ in range(iterations):
        scores = model.scores(
            clean_x + delta_x, clean_window + delta_window, task, "fused"
        )
        loss = F.cross_entropy(scores, target)
        grad_x, grad_window = torch.autograd.grad(
            loss, [delta_x, delta_window], allow_unused=True
        )
        if grad_x is None:
            grad_x = torch.zeros_like(delta_x)
        if grad_window is None:
            grad_window = torch.zeros_like(delta_window)
        with torch.no_grad():
            delta_x = (delta_x + step_size * scale * grad_x.sign()).clamp(
                -epsilon * scale, epsilon * scale
            ).detach().requires_grad_(True)
            delta_window = (
                delta_window + step_size * scale * grad_window.sign()
            ).clamp(-epsilon * scale, epsilon * scale).detach().requires_grad_(True)
    with torch.no_grad():
        scores = model.scores(
            clean_x + delta_x, clean_window + delta_window, task, "fused"
        )
    return scores.cpu().numpy(), np.stack([
        (delta_x.detach() / scale).abs().max().item(),
        (delta_window.detach() / scale).abs().max().item(),
    ])


def subject_bootstrap(records: dict[str, np.ndarray], subjects: np.ndarray, repeats: int = 200) -> dict:
    """按被试聚类重采样；不以 Trial 独立性夸大置信区间。"""
    if len(subjects) != len(records["is_known"]):
        raise ValueError(
            f"被试记录长度 {len(subjects)} 与评估记录长度 {len(records['is_known'])} 不一致"
        )
    rng = np.random.default_rng(20261106)
    unique_subjects = np.unique(subjects)
    indices_by_subject = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    values = {name: [] for name in (
        "C", "U", "macro_f1", "unknown_f1", "unknown_auroc", "known_aupr"
    )}
    for _ in range(repeats):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([indices_by_subject[subject] for subject in sampled])
        known = records["is_known"][indices]
        true = records["true"][indices]
        pred = records["pred"][indices]
        confidence = records["confidence"][indices]
        if len(np.unique(known)) < 2:
            continue
        values["C"].append(float(np.mean(pred[known] == true[known])))
        values["U"].append(float(np.mean(pred[~known] == 0)))
        values["macro_f1"].append(float(f1_score(true, pred, average="macro", zero_division=0)))
        values["unknown_f1"].append(float(f1_score(
            known.astype(int), pred != 0, pos_label=0, zero_division=0
        )))
        values["unknown_auroc"].append(float(roc_auc_score(known, confidence)))
        values["known_aupr"].append(float(average_precision_score(known, confidence)))
    return {
        name: {
            "p2.5": float(np.quantile(samples, 0.025)),
            "p97.5": float(np.quantile(samples, 0.975)),
            "n": len(samples),
        }
        for name, samples in values.items() if samples
    }
