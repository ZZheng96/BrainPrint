"""在测试 Session2 内寻找 clean/adversarial 候选配对并用原始波形核验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from eeg_features import load_eeg

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT.parent / "BrainPrint_cache"


def quantiles(values: np.ndarray) -> dict:
    return {
        str(q): float(np.quantile(values, q))
        for q in (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
    }


def waveform_correlation(path_a: str, path_b: str) -> float:
    a, _ = load_eeg(DATA / path_a)
    b, _ = load_eeg(DATA / path_b)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    numerator = np.sum(a * b, axis=0)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--raw-checks", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "test_pair_analysis.json")
    args = parser.parse_args()

    metadata = pd.read_csv(CACHE / "test_metadata.csv")
    features = np.load(CACHE / "test_features.npy", mmap_mode="r")
    known_indices = np.flatnonzero(metadata["session"].to_numpy() == "session2")
    known = np.asarray(features[known_indices])
    known_metadata = metadata.iloc[known_indices].reset_index(drop=True)

    print("standardizing", known.shape, flush=True)
    scaled = StandardScaler().fit_transform(known)
    print("PCA", args.components, flush=True)
    embedding = PCA(n_components=args.components, svd_solver="randomized", random_state=42).fit_transform(scaled)
    del scaled, known

    nearest = np.empty(len(known_indices), dtype=np.int32)
    distances = np.empty((len(known_indices), 3), dtype=np.float32)
    reciprocal = np.zeros(len(known_indices), dtype=bool)
    same_task = np.zeros(len(known_indices), dtype=bool)
    report = {"n_session2": len(known_indices), "pca_components": args.components}

    for task in sorted(known_metadata["task"].unique()):
        local = np.flatnonzero(known_metadata["task"].to_numpy() == task)
        task_embedding = embedding[local]
        task_embedding = task_embedding / np.maximum(np.linalg.norm(task_embedding, axis=1, keepdims=True), 1e-12)
        nn = NearestNeighbors(n_neighbors=3, metric="euclidean", n_jobs=-1).fit(task_embedding)
        distance, neighbor = nn.kneighbors(task_embedding)
        candidate = local[neighbor[:, 1]]
        nearest[local] = candidate
        distances[local] = distance
        reciprocal[local] = nearest[candidate] == local
        same_task[local] = True
        print(f"{task}: n={len(local)}, reciprocal={reciprocal[local].mean():.4f}", flush=True)

    assert same_task.all()
    report["reciprocal_fraction"] = float(reciprocal.mean())
    report["nearest_distance_quantiles"] = quantiles(distances[:, 0])
    report["nearest_to_second_distance_ratio"] = quantiles(
        distances[:, 0] / np.maximum(distances[:, 1], 1e-12)
    )
    candidate_pairs = np.flatnonzero(reciprocal & (np.arange(len(nearest)) < nearest))
    report["n_reciprocal_pairs"] = len(candidate_pairs)

    rng = np.random.default_rng(42)
    check_pairs = rng.choice(candidate_pairs, min(args.raw_checks, len(candidate_pairs)), replace=False)
    pair_correlations = []
    control_correlations = []
    for count, i in enumerate(check_pairs, 1):
        j = nearest[i]
        pair_correlations.append(waveform_correlation(known_metadata.at[i, "path"], known_metadata.at[j, "path"]))
        same_task_pool = np.flatnonzero(known_metadata["task"].to_numpy() == known_metadata.at[i, "task"])
        k = int(rng.choice(same_task_pool[same_task_pool != i]))
        control_correlations.append(waveform_correlation(known_metadata.at[i, "path"], known_metadata.at[k, "path"]))
        if count % 20 == 0:
            print(f"raw checks: {count}/{len(check_pairs)}", flush=True)
    report["raw_pair_channel_correlation"] = quantiles(np.asarray(pair_correlations))
    report["raw_random_same_task_channel_correlation"] = quantiles(np.asarray(control_correlations))

    pair_table = known_metadata[["sample", "path", "task"]].copy()
    pair_table["nearest_sample"] = known_metadata.iloc[nearest]["sample"].to_numpy()
    pair_table["nearest_distance"] = distances[:, 0]
    pair_table["second_distance"] = distances[:, 1]
    pair_table["reciprocal"] = reciprocal
    pair_table.to_csv(CACHE / "test_pair_candidates.csv", index=False)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
