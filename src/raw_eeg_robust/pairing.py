"""配对扩覆盖候选的无 oracle 可行性门槛计算。"""
from __future__ import annotations


def feasibility_metrics(
    known_adversarial: int,
    candidate_pairs: int,
    same_identity_pairs: int,
    correct_anchor_pairs: int,
    *,
    coverage_denominator: int | None = None,
) -> dict:
    """计算内层候选的覆盖、纯度和锚点身份正确率。"""
    denominator = known_adversarial if coverage_denominator is None else coverage_denominator
    if min(known_adversarial, candidate_pairs, same_identity_pairs, correct_anchor_pairs, denominator) < 0:
        raise ValueError("计数不能为负")
    if candidate_pairs > denominator:
        raise ValueError("候选配对数不能超过覆盖分母")
    if same_identity_pairs > candidate_pairs or correct_anchor_pairs > same_identity_pairs:
        raise ValueError("配对质量计数不合法")
    return {
        "adv_coverage": candidate_pairs / denominator if denominator else 0.0,
        "same_identity_purity": same_identity_pairs / candidate_pairs if candidate_pairs else 0.0,
        "anchor_identity_accuracy": correct_anchor_pairs / same_identity_pairs if same_identity_pairs else 0.0,
        "thresholds": {
            "adv_coverage": 0.65,
            "same_identity_purity": 0.99,
            "anchor_identity_accuracy": 0.95,
        },
    }


def passes_feasibility(metrics: dict) -> bool:
    thresholds = metrics["thresholds"]
    return (
        metrics["adv_coverage"] >= thresholds["adv_coverage"]
        and metrics["same_identity_purity"] >= thresholds["same_identity_purity"]
        and metrics["anchor_identity_accuracy"] >= thresholds["anchor_identity_accuracy"]
    )
