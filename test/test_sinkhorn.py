import unittest
from unittest import mock

import numpy as np

from src.balanced_sinkhorn import sinkhorn_labels
from src.evaluate_task_specific_ensemble import (
    build_task_scores,
    predict_with_task_postprocessing,
    row_zscore,
)


class SinkhornTest(unittest.TestCase):
    def test_clear_decisions_are_preserved(self):
        scores = np.array([
            [5.0, 0.0],
            [4.0, 1.0],
            [0.0, 5.0],
            [1.0, 4.0],
        ])
        labels = sinkhorn_labels(scores, np.array([0.5, 0.5]), temperature=1.0, iterations=100)
        np.testing.assert_array_equal(labels, np.array([1, 1, 2, 2]))

    def test_invalid_prior_is_not_silently_accepted(self):
        with np.testing.assert_raises(ValueError):
            sinkhorn_labels(np.ones((2, 2)), np.array([0.0, 0.0]), 1.0, 10)


class TaskSpecificScoreTest(unittest.TestCase):
    def test_task_routing_and_weights(self):
        tasks = np.array(["EC", "EO", "Imagery", "SSVEP"])
        full = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 3.0], [4.0, 0.0]])
        window = np.array([[2.0, 0.0], [1.0, 3.0], [4.0, 1.0], [2.0, 1.0]])
        low = np.array([[0.0, 1.0], [3.0, 0.0], [2.0, 2.0], [1.0, 4.0]])
        low_neighbor = np.array([[0.2, 0.4], [0.1, 0.3], [0.5, 0.2], [0.4, 0.1]])
        delta_neighbor = np.array([[0.3, 0.1], [0.4, 0.2], [0.1, 0.5], [0.2, 0.4]])
        scores = build_task_scores(tasks, full, window, low, low_neighbor, delta_neighbor)
        base = 0.5 * full + 0.5 * window
        np.testing.assert_allclose(
            scores[0], row_zscore(base)[0] + 0.25 * row_zscore(low_neighbor)[0] + 0.5 * row_zscore(delta_neighbor)[0]
        )
        np.testing.assert_allclose(scores[1], (base + 0.25 * low)[1])
        np.testing.assert_allclose(scores[2], window[2])
        np.testing.assert_allclose(scores[3], (base + 0.5 * low)[3])
        self.assertTrue(np.isfinite(scores).all())

    @mock.patch("src.evaluate_task_specific_ensemble.sinkhorn_labels")
    def test_ec_guard_only_accepts_low_confidence_changes(self, mocked_sinkhorn):
        mocked_sinkhorn.side_effect = [np.array([2, 1]), np.array([1]), np.array([1])]
        scores = np.zeros((4, 20))
        scores[0, 0], scores[0, 1] = 5.0, 0.0
        scores[1, 0], scores[1, 1] = 0.4, 0.5
        scores[2, 0], scores[3, 0] = 1.0, 1.0
        predictions = predict_with_task_postprocessing(
            scores=scores,
            subjects=np.array([1, 2, 1, 1]),
            tasks=np.array(["EC", "EC", "Imagery", "SSVEP"]),
            folds=np.zeros(4, dtype=int),
            selected=np.ones(4, dtype=bool),
            protocol="forward",
            balanced_ec_imagery=True,
            guarded_ec=True,
        )
        np.testing.assert_array_equal(predictions[:2], np.array([1, 1]))


if __name__ == "__main__":
    unittest.main()
