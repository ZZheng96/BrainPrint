"""Convert the E200 frozen bundle into final/team_hymn/weights.pt (weights_only-safe).

Only tensors and plain python types are stored, so the evaluator can call
torch.load(..., weights_only=True). Values are copied bit-for-bit from
research/E200_final_freeze/bundle/frozen_bundle.pt; nothing is re-fitted.
"""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/E200_final_freeze/bundle/frozen_bundle.pt"
DST = ROOT / "final/team_hymn/weights.pt"

b = torch.load(SRC, map_location="cpu", weights_only=False)
s1, s2, s3 = b["stage1"], b["stage2"], b["stage3"]
feat = s1["feature"]
assert list(s1["task_order"]) == list(s2["task_order"]) == ["EC", "EO", "Imagery", "SSVEP"]
assert np.array_equal(np.asarray(s2["classes"]), np.arange(1, 21))

proto = np.asarray(s2["prototypes"], dtype=np.float32)[:, :, 0, :]  # (4, 20, 96)
weights = {
    "format": "brainprint-final-v1",
    "source": "E200_final_freeze subwin6_pca96_k1 @ a=0.90",
    "feature": {
        "n_subwindows": int(feat["n_subwindows"]),
        "subwindow_length": int(feat["subwindow_length"]),
        "channels": int(feat["channels"]),
        "sampling_rate_hz": float(feat["sampling_rate_hz"]),
        "bands_hz": [[float(lo), float(hi)] for lo, hi in feat["bands_hz"]],
        "channel_std_ddof": int(feat["channel_std_ddof"]),
        "channel_std_epsilon": float(feat["channel_std_epsilon"]),
        "covariance_shrinkage": float(feat["covariance_shrinkage"]),
        "eigenvalue_floor": float(feat["eigenvalue_floor"]),
        "off_diagonal_multiplier": float(feat["off_diagonal_multiplier"]),
        "zband_epsilon": float(feat["zband_epsilon"]),
        "dim": int(feat["dim"]),
    },
    "task_order": ["EC", "EO", "Imagery", "SSVEP"],
    "task_mu": torch.as_tensor(np.asarray(s1["task_stats_mu"], dtype=np.float32)),
    "task_sd": torch.as_tensor(np.asarray(s1["task_stats_sd"], dtype=np.float32)),
    "pca_mean": torch.as_tensor(np.asarray(s1["pca_mean"], dtype=np.float32)),
    "pca_basis": torch.as_tensor(np.asarray(s1["pca_basis"], dtype=np.float32)),
    "prototypes": torch.as_tensor(proto),  # identities 1..20
    "tau_dist": float(s3["tau_dist"]),
}
DST.parent.mkdir(parents=True, exist_ok=True)
torch.save(weights, DST)
torch.load(DST, weights_only=True)
print(f"wrote {DST} ({DST.stat().st_size} bytes), tau={weights['tau_dist']:.6f}")
