import numpy as np
import pytest

from src.raw_eeg_robust.pairing import feasibility_metrics, passes_feasibility


def test_pairing_feasibility_gate():
    metrics = feasibility_metrics(100, 70, 70, 67)
    assert metrics["adv_coverage"] == pytest.approx(0.70)
    assert metrics["same_identity_purity"] == pytest.approx(1.0)
    assert metrics["anchor_identity_accuracy"] == pytest.approx(0.9571428571)
    assert passes_feasibility(metrics)


def test_pairing_upper_bound_fails_without_oracle():
    metrics = feasibility_metrics(9232, 3963, 3963, 3963)
    assert metrics["adv_coverage"] < 0.65
    assert not passes_feasibility(metrics)
