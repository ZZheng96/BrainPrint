import unittest

import numpy as np

from src.eeg_features import N_COV_FEATURES, N_FEATURES, N_PSD_FEATURES, extract_features, normalize_log_cov_determinant, robust_clip


class FeaturesTest(unittest.TestCase):
    def test_feature_shape_and_spectral_peak(self):
        rng = np.random.default_rng(42)
        time = np.arange(1200) / 300
        data = 0.05 * rng.standard_normal((1200, 18))
        data[:, 12] += np.sin(2 * np.pi * 10 * time)
        features, quality = extract_features(data)
        self.assertEqual(features.shape, (N_FEATURES,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(quality["nonfinite_fraction"], 0.0)
        log_psd = features[:N_PSD_FEATURES].reshape(18, 90)
        peak_frequency = np.argmax(log_psd[12]) + 1
        self.assertEqual(peak_frequency, 10)

    def test_robust_clip_limits_outlier(self):
        data = np.zeros((100, 18))
        data[:, 0] = np.linspace(-1, 1, 100)
        data[50, 0] = 1e6
        clipped, fraction = robust_clip(data)
        self.assertLess(np.max(np.abs(clipped)), 1e3)
        self.assertAlmostEqual(fraction[0], 0.01)

    def test_determinant_normalization(self):
        rng = np.random.default_rng(7)
        features = rng.standard_normal((2, N_COV_FEATURES))
        normalized = normalize_log_cov_determinant(features)
        bands = normalized.reshape(2, 6, 171)
        diagonal = np.flatnonzero(np.triu_indices(18)[0] == np.triu_indices(18)[1])
        np.testing.assert_allclose(bands[..., diagonal].mean(axis=-1), 0.0, atol=1e-12)
        off_diagonal = np.flatnonzero(np.triu_indices(18)[0] != np.triu_indices(18)[1])
        np.testing.assert_allclose(bands[..., off_diagonal], features.reshape(2, 6, 171)[..., off_diagonal])


if __name__ == "__main__":
    unittest.main()
