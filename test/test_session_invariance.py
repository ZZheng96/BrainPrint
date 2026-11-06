import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.asnorm_openset.run import predict_test
from src.open_set_improvement.core import COV_COLUMNS


class SessionInvarianceTest(unittest.TestCase):
    def test_frozen_prediction_ignores_test_session_values(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            n_train, n_test = 18, 24
            pd.DataFrame({
                "subject": np.repeat([1, 2, 3], 6),
                "task": np.tile(np.repeat(["EC", "EO"], 3), 3),
            }).to_csv(cache / "train_metadata.csv", index=False)
            test_metadata = pd.DataFrame({
                "sample": [f"test_{index}" for index in range(n_test)],
                "task": np.repeat(["EC", "EO"], n_test // 2),
                "session": ["session1"] * n_test,
            })
            test_metadata.to_csv(cache / "test_metadata.csv", index=False)

            train_x = rng.normal(size=(n_train, len(COV_COLUMNS))).astype(np.float32)
            test_x = rng.normal(size=(n_test, len(COV_COLUMNS))).astype(np.float32)
            train_window = rng.normal(size=(n_train, 2, len(COV_COLUMNS))).astype(np.float32)
            test_window = rng.normal(size=(n_test, 2, len(COV_COLUMNS))).astype(np.float32)
            for split, x, window in (
                ("train", train_x, train_window), ("test", test_x, test_window)
            ):
                features = np.zeros((len(x), COV_COLUMNS[-1] + 1), dtype=np.float32)
                features[:, COV_COLUMNS] = x
                window_features = np.zeros(
                    (len(x) * 2, COV_COLUMNS[-1] + 1), dtype=np.float32
                )
                window_features[:, COV_COLUMNS] = window.reshape(-1, len(COV_COLUMNS))
                np.save(cache / f"{split}_features.npy", features)
                np.save(cache / f"{split}_window2_features.npy", window_features)

            config = json.loads(
                (Path(__file__).parents[1] / "research" / "asnorm_openset" /
                 "frozen_config.json").read_text(encoding="utf-8")
            )
            original_config = config.copy()
            with patch("src.asnorm_openset.run.resolve_cache", return_value=cache):
                first_meta, first_pred, first_conf = predict_test(config, torch.device("cpu"))
                test_metadata["session"] = ["not-a-session"] * (n_test // 2) + ["session2"] * (
                    n_test - n_test // 2
                )
                test_metadata.to_csv(cache / "test_metadata.csv", index=False)
                second_meta, second_pred, second_conf = predict_test(config, torch.device("cpu"))

            pd.testing.assert_frame_equal(first_meta, second_meta)
            np.testing.assert_array_equal(first_pred, second_pred)
            np.testing.assert_allclose(first_conf, second_conf, atol=0, rtol=0)
            self.assertEqual(config, original_config)


if __name__ == "__main__":
    unittest.main()
