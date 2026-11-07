"""由训练折内 clean/adv 样本学习的干净锚点与配对传播。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.open_set_improvement.core import confidence_values


def anchor_features(
    view_scores: list[np.ndarray],
    classes: np.ndarray,
    operating_points: list[dict],
) -> np.ndarray:
    """构造仅依赖三视图分数及工作点的 22 维锚点特征。"""
    if len(view_scores) != 3 or len(operating_points) != 3:
        raise ValueError("锚点特征需要三个视图及三个工作点")
    classes = np.asarray(classes)
    scores = [np.asarray(view) for view in view_scores]
    if any(view.ndim != 2 for view in scores):
        raise ValueError("每个视图的分数必须为 N×C 矩阵")
    if len({view.shape for view in scores}) != 1:
        raise ValueError("三个视图的分数形状必须一致")
    if scores[0].shape[1] != len(classes):
        raise ValueError("classes 长度必须与分数列数一致")

    confidences = [confidence_values(view) for view in scores]
    columns = []
    for name in ("max_logit", "margin", "energy"):
        columns.extend(values[name] for values in confidences)

    differences = []
    accepted = []
    for values, point in zip(confidences, operating_points):
        difference = values[point["score_name"]] - float(point["threshold"])
        differences.append(difference)
        accepted.append(difference >= 0)
    columns.extend(differences)
    columns.extend(accepted)

    predictions = [classes[view.argmax(axis=1)] for view in scores]
    pair_agreement = [
        predictions[left] == predictions[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    columns.extend(pair_agreement)
    columns.append(pair_agreement[0] & pair_agreement[1] & pair_agreement[2])
    columns.append(np.sum(np.stack(accepted), axis=0))
    columns.append(np.mean(np.stack(differences), axis=0))
    columns.append(np.min(np.stack(differences), axis=0))

    features = np.column_stack(columns).astype(np.float64, copy=False)
    limit = np.finfo(np.float32).max
    features = np.nan_to_num(features, nan=0.0, posinf=limit, neginf=-limit)
    return np.clip(features, -limit, limit).astype(np.float32)


class CleanAnchorModel:
    """标准化、类别平衡的 PyTorch 二分类 logistic regression。"""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.linear: torch.nn.Linear | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "CleanAnchorModel":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features 必须为 N×D，且 labels 长度必须为 N")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("labels 只能包含 0 (adv) 和 1 (clean)")
        n_clean = int(np.sum(y == 1))
        n_adversarial = int(np.sum(y == 0))
        if n_clean == 0 or n_adversarial == 0:
            raise ValueError("拟合干净锚点需要同时包含 clean 和 adv 样本")

        torch.manual_seed(20261107)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        self.mean = x_tensor.mean(dim=0)
        self.scale = x_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        x_scaled = (x_tensor - self.mean) / self.scale
        self.linear = torch.nn.Linear(x.shape[1], 1, device=self.device)
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=0.05)
        pos_weight = torch.as_tensor(
            n_adversarial / n_clean, dtype=torch.float32, device=self.device
        )
        for _ in range(1200):
            optimizer.zero_grad(set_to_none=True)
            logits = self.linear(x_scaled).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(
                logits, y_tensor, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.linear is None:
            raise RuntimeError("CleanAnchorModel 尚未拟合")
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.linear((x - self.mean) / self.scale).squeeze(1)
            probability = torch.sigmoid(logits)
        return probability.cpu().numpy()


def learned_propagation(
    predictions: np.ndarray,
    clean_probability: np.ndarray,
    neighbors: dict[str, np.ndarray],
    pair_positions_func,
) -> np.ndarray:
    """只在高相似同任务 pair 内，从高 P(clean) 锚点向低概率端传播。"""
    output = predictions.copy()
    for left in pair_positions_func(neighbors, gap=0.005):
        right = neighbors["nearest"][left]
        anchor, receiver = (
            (left, right)
            if clean_probability[left] >= clean_probability[right]
            else (right, left)
        )
        if (
            predictions[anchor] > 0
            and clean_probability[anchor] >= 0.5
            and clean_probability[receiver] < 0.5
        ):
            output[receiver] = predictions[anchor]
    return output


def propagation_audit(
    before: np.ndarray,
    after: np.ndarray,
    known_true: np.ndarray,
    n_known: int,
) -> dict:
    """按 clean/adv/unknown receiver 分组量化传播改变及正确性。"""
    before = np.asarray(before)
    after = np.asarray(after)
    known_true = np.asarray(known_true)
    n_known = int(n_known)
    groups = {
        "clean": (slice(0, n_known), known_true),
        "adv": (slice(n_known, 2 * n_known), known_true),
        "unknown": (
            slice(2 * n_known, None),
            np.zeros(len(before) - 2 * n_known, dtype=known_true.dtype),
        ),
    }
    audit = {}
    for name, (positions, targets) in groups.items():
        audit[name] = {
            "changed": int(np.sum(before[positions] != after[positions])),
            "correct_before": int(np.sum(before[positions] == targets)),
            "correct_after": int(np.sum(after[positions] == targets)),
        }
    return audit
