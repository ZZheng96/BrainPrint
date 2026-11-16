"""E200 reproduction check for the frozen bundle + predict.py.

Three independent paths are compared
  A. `predict.py` (bundle only, features rebuilt from the raw `.mat` files),
  B. the frozen Train feature cache (E171 `test_subwin_mean_n6.npy`) + the same
     bundle prototypes / tau (feature-level shortcut, used as the reference),
  C. E189's stored frozen Test scores (`subwin6_pca96_k1__test`), i.e. the exact
     numbers E191's 0.7135 / 0.7069 audit was computed from.

Writes `cache/repro_check.json` and prints a summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BUNDLE = OUT / "bundle/frozen_bundle.pt"
CACHE = OUT / "cache"
TEST_META = ROOT / "BrainPrint_cache/test_metadata.csv"
TEST_FEAT = ROOT / "research/E171_multiscale_fusion/test_subwin_mean_n6.npy"
E189_SCORES = ROOT / "research/E189_balanced_gate_gates/cache/stream_scores.npz"
E191_AUDIT = ROOT / "research/E191_fix_gate1_gate7/audit_v3.json"


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def reference_labels(features: np.ndarray, tasks: np.ndarray, bundle: dict) -> tuple:
    """Feature-cache path: same stage1/stage2/tau as predict.py, numpy only."""
    stage1, stage2 = bundle["stage1"], bundle["stage2"]
    order = list(stage1["task_order"])
    mu = np.asarray(stage1["task_stats_mu"], dtype=np.float32)
    sd = np.asarray(stage1["task_stats_sd"], dtype=np.float32)
    pca_mu = np.asarray(stage1["pca_mean"], dtype=np.float64)
    basis = np.asarray(stage1["pca_basis"], dtype=np.float64)
    proto = np.asarray(stage2["prototypes"], dtype=np.float32)
    classes = np.asarray(stage2["classes"], dtype=np.int64)
    index = np.asarray([order.index(str(t)) for t in tasks], dtype=np.int64)
    z = (features.astype(np.float64) - mu[index].astype(np.float64)) \
        / sd[index].astype(np.float64)
    proj = (z - pca_mu) @ basis
    dist = np.full(len(features), np.nan, dtype=np.float32)
    ident = np.full(len(features), -1, dtype=np.int64)
    for i in range(len(order)):
        rows = np.flatnonzero(index == i)
        p = torch.as_tensor(proto[i].reshape(len(classes), -1), dtype=torch.float32)
        q = torch.as_tensor(f32(proj[rows]))
        values = torch.cdist(q, p)
        best, at = values.min(dim=1)
        dist[rows] = best.numpy()
        ident[rows] = classes[at.numpy()]
    labels = np.where(dist <= float(bundle["stage3"]["tau_dist"]), ident, 0)
    return dist, ident, labels.astype(np.int64)


def main() -> None:
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    tau = float(bundle["stage3"]["tau_dist"])
    meta = pd.read_csv(TEST_META)
    tasks = meta["task"].to_numpy(dtype=object)
    session = meta["session"].astype(str).to_numpy()
    is_s1 = np.isin(session, ["session1", "s1", "1"])

    # ---- B: feature-cache reference
    test_feat = f32(np.load(TEST_FEAT))
    ref_dist, ref_ident, ref_labels = reference_labels(test_feat, tasks, bundle)

    # ---- A1: 256-row smoke
    smoke_csv = pd.read_csv(CACHE / "smoke_256.csv")
    smoke_feat = np.load(CACHE / "smoke_256_work/features.npy")
    smoke_feature_delta = float(np.max(np.abs(smoke_feat - test_feat[:len(smoke_feat)])))
    smoke_labels = smoke_csv["Label"].to_numpy(dtype=np.int64)
    smoke_match = int((smoke_labels == ref_labels[:len(smoke_labels)]).sum())
    smoke_agreement = float(np.mean(smoke_labels == ref_labels[:len(smoke_labels)]))

    # ---- A2: full submission
    submission = pd.read_csv(OUT / "submission_frozen.csv")
    full_labels = submission["Label"].to_numpy(dtype=np.int64)
    full_agreement = float(np.mean(full_labels == ref_labels))
    full_feat = np.load(OUT / "submission_frozen_work/features.npy")
    full_feature_delta = float(np.max(np.abs(full_feat - test_feat)))

    # ---- C: E189 stored scores (what E191 audited)
    stored = np.load(E189_SCORES, allow_pickle=False)
    stored_scores = stored["subwin6_pca96_k1__test"].astype(np.float64)
    stored_mu = float(stored["subwin6_pca96_k1__oof_mu"])
    stored_sd = float(stored["subwin6_pca96_k1__oof_sd"])
    e191 = json.loads(E191_AUDIT.read_text(encoding="utf-8"))["subwin6_pca96_k1"]["points"]["0.9"]
    stored_labels = np.where((stored_scores - stored_mu) / stored_sd >= e191["tau"],
                             ref_ident, 0).astype(np.int64)
    stored_agreement = float(np.mean(stored_labels == full_labels))
    stored_dist = -stored_scores
    stored_dist_delta = float(np.max(np.abs(stored_dist - ref_dist)))

    # ---- submission compliance
    expected_sample = [f"test_{int(str(s).split('_')[-1]):03d}" if str(s).startswith("test")
                       else str(s) for s in meta["sample"]]
    compliance = {
        "rows": int(len(submission)),
        "rows_expected": int(len(meta)),
        "rows_ok": bool(len(submission) == len(meta) == 22251),
        "columns": list(submission.columns),
        "columns_ok": bool(list(submission.columns) == ["Sample", "Label"]),
        "sample_order_ok": bool(list(submission["Sample"].astype(str)) == expected_sample),
        "label_min": int(full_labels.min()),
        "label_max": int(full_labels.max()),
        "labels_in_range": bool(np.isin(full_labels, np.arange(21)).all()),
        "nan_free": bool(submission.notna().all().all()),
    }

    # ---- S1 / S2 and comparison with E191
    s1 = float(np.mean(full_labels[is_s1] == 0))
    s2 = float(np.mean(full_labels[~is_s1] != 0))
    acceptance = float(np.mean(full_labels != 0))
    window_tau = float(bundle["stage3"]["sensitivity_window_tau_dist"])
    alt_labels = np.where(ref_dist <= window_tau, ref_ident, 0).astype(np.int64)
    report = {
        "bundle_sha256": __import__("hashlib").sha256(BUNDLE.read_bytes()).hexdigest(),
        "tau_dist": tau,
        "tau_norm": float(bundle["stage3"]["tau_norm"]),
        "train_known_acceptance": float(bundle["stage3"]["train_known_acceptance"]),
        "smoke_256": {
            "rows": int(len(smoke_labels)),
            "feature_max_abs_vs_cache": smoke_feature_delta,
            "feature_tolerance_1e-3": bool(smoke_feature_delta <= 1e-3),
            "labels_identical": int(smoke_match),
            "agreement": smoke_agreement,
            "bitwise_identical": bool(smoke_match == len(smoke_labels)),
        },
        "full": {
            "rows": int(len(full_labels)),
            "feature_max_abs_vs_cache": full_feature_delta,
            "agreement_with_feature_cache_path": full_agreement,
            "disagreements": int(round((1.0 - full_agreement) * len(full_labels))),
            "agreement_with_e189_stored_scores": stored_agreement,
            "distance_max_abs_vs_e189": stored_dist_delta,
            "rejected": int((full_labels == 0).sum()),
            "rejected_rate": float(np.mean(full_labels == 0)),
            "accepted": int((full_labels != 0).sum()),
            "s1_rejection": s1,
            "s2_acceptance": s2,
            "e191_s1_rejection": float(e191["test_s1_rejection"]),
            "e191_s2_acceptance": float(e191["test_s2_acceptance"]),
            "s1_delta": s1 - float(e191["test_s1_rejection"]),
            "s2_delta": s2 - float(e191["test_s2_acceptance"]),
            "label_counts": {int(k): int(v) for k, v in
                             zip(*np.unique(full_labels, return_counts=True))},
            "per_session": {
                "session1": {"n": int(is_s1.sum()),
                             "rejected": int((full_labels[is_s1] == 0).sum()),
                             "rejection_rate": s1},
                "session2": {"n": int((~is_s1).sum()),
                             "rejected": int((full_labels[~is_s1] == 0).sum()),
                             "rejection_rate": float(np.mean(full_labels[~is_s1] == 0))},
            },
            "window_tau_variant": {
                "tau_dist": window_tau,
                "s1_rejection": float(np.mean(alt_labels[is_s1] == 0)),
                "s2_acceptance": float(np.mean(alt_labels[~is_s1] != 0)),
                "rejected": int((alt_labels == 0).sum()),
            },
        },
        "compliance": compliance,
    }
    (CACHE / "repro_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"tau_dist {tau:.6f} (tau_norm {report['tau_norm']:.6f}, "
          f"Train acceptance {report['train_known_acceptance']:.4f})")
    print(f"smoke 256 : feature max|d| {smoke_feature_delta:.3e} (<=1e-3 "
          f"{report['smoke_256']['feature_tolerance_1e-3']}), labels identical "
          f"{smoke_match}/{len(smoke_labels)}")
    print(f"full      : feature max|d| {full_feature_delta:.3e}, labels vs cache path "
          f"{full_agreement:.6f}, vs E189 stored scores {stored_agreement:.6f}, "
          f"distance max|d| vs E189 {stored_dist_delta:.3e}")
    print(f"rejected  : {report['full']['rejected']}/{report['full']['rows']} "
          f"({report['full']['rejected_rate']:.4f})")
    print(f"S1        : {s1:.6f} (E191 {e191['test_s1_rejection']:.6f}, "
          f"delta {s1 - e191['test_s1_rejection']:+.6f})")
    print(f"S2        : {s2:.6f} (E191 {e191['test_s2_acceptance']:.6f}, "
          f"delta {s2 - e191['test_s2_acceptance']:+.6f})")
    print(f"window-tau variant: S1 {report['full']['window_tau_variant']['s1_rejection']:.4f} "
          f"S2 {report['full']['window_tau_variant']['s2_acceptance']:.4f}")
    print(f"compliance: {compliance}")


if __name__ == "__main__":
    main()
