import numpy as np
import pandas as pd

from src.session_blind_candidate.run import (
    FORBIDDEN_COLUMNS,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
    read_metadata,
)
from src.session_blind_candidate.shift_calibration import (
    apply_conditional_rejection,
    apply_robust_normalization,
    fit_conditional_thresholds,
    fit_training_statistics,
    score_diagnostics,
)


def test_metadata_reading_uses_only_whitelisted_columns(tmp_path):
    frame = pd.DataFrame({
        "subject": [1, 2], "task": ["EC", "EO"], "session": ["session1", "session2"],
        "sample": ["a", "b"], "trial": [1, 2], "path": ["x", "y"],
    })
    path = tmp_path / "metadata.csv"
    frame.to_csv(path, index=False)
    train = read_metadata(path, TRAIN_COLUMNS)
    test = read_metadata(path, TEST_COLUMNS)
    assert list(train.columns) == TRAIN_COLUMNS
    assert list(test.columns) == TEST_COLUMNS
    assert not set(FORBIDDEN_COLUMNS).intersection(train.columns)
    assert not set(FORBIDDEN_COLUMNS).intersection(test.columns)


def test_robust_statistics_are_training_fitted_and_shape_safe():
    train = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [100.0, 13.0]])
    stats, window_stats = fit_training_statistics(train, train[:, None, :])
    transformed, transformed_window = apply_robust_normalization(
        np.array([[1000.0, 14.0]]), np.array([[[1000.0, 14.0]]]), stats, window_stats
    )
    assert transformed.shape == (1, 2)
    assert transformed_window.shape == (1, 1, 2)
    assert np.all(np.isfinite(transformed))
    assert np.all(np.isfinite(transformed_window))


def test_diagnostics_and_conditional_rejection_use_frozen_oof_calibration():
    full = np.array([[3.0, 0.0], [2.0, 1.0], [0.0, 2.0], [0.0, 1.0]])
    base = np.array([[3.0, 0.0], [1.5, 1.0], [0.0, 2.0], [1.0, 0.0]])
    diagnostics = score_diagnostics(full, base)
    calibration = fit_conditional_thresholds(base, diagnostics, n_bins=2)
    predicted = apply_conditional_rejection(base, np.array([1, 2]), diagnostics, calibration)
    assert set(np.unique(predicted)).issubset({0, 1, 2})
    assert np.all(np.isfinite(calibration.threshold(diagnostics["quality"])))
    assert "shift" in diagnostics and "full_base_consistency" in diagnostics


def test_session_changes_do_not_change_whitelisted_metadata(tmp_path):
    base = pd.DataFrame({"subject": [1, 2], "task": ["EC", "EO"], "sample": ["a", "b"]})
    altered = base.copy()
    altered["session"] = ["anything", "else"]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    base.to_csv(first, index=False)
    altered.to_csv(second, index=False)
    np.testing.assert_array_equal(
        read_metadata(first, TEST_COLUMNS).to_numpy(),
        read_metadata(second, TEST_COLUMNS).to_numpy(),
    )
