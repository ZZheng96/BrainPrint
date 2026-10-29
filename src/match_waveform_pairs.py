"""用降采样时域波形指纹匹配测试集中的近重复/对抗配对。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT.parent / "BrainPrint_cache"


def quantiles(values: np.ndarray) -> dict:
    return {str(q): float(np.quantile(values, q)) for q in (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)}


def fingerprint(path: Path, stride: int = 20) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    data = data - data.mean(axis=0, keepdims=True)
    data = data / np.maximum(data.std(axis=0, keepdims=True), 1e-12)
    vector = data[::stride].reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "waveform_pair_analysis.json")
    args = parser.parse_args()
    metadata = pd.read_csv(CACHE / "test_metadata.csv")
    known_indices = np.flatnonzero(metadata["session"].to_numpy() == "session2")
    known = metadata.iloc[known_indices].reset_index(drop=True)
    fingerprints = np.empty((len(known), 1200 // args.stride * 18), dtype=np.float32)
    for i, path in enumerate(known["path"], 1):
        fingerprints[i - 1] = fingerprint(DATA / path, args.stride)
        if i % 1000 == 0 or i == len(known):
            print(f"fingerprints: {i}/{len(known)}", flush=True)

    nearest = np.empty(len(known), dtype=np.int32)
    similarity = np.empty(len(known), dtype=np.float32)
    second_similarity = np.empty(len(known), dtype=np.float32)
    for task in sorted(known["task"].unique()):
        local = np.flatnonzero(known["task"].to_numpy() == task)
        nn = NearestNeighbors(n_neighbors=3, metric="cosine", n_jobs=-1).fit(fingerprints[local])
        distances, neighbors = nn.kneighbors(fingerprints[local])
        nearest[local] = local[neighbors[:, 1]]
        similarity[local] = 1 - distances[:, 1]
        second_similarity[local] = 1 - distances[:, 2]
        print(
            f"{task}: n={len(local)}, median similarity={np.median(similarity[local]):.6f}",
            flush=True,
        )

    reciprocal = nearest[nearest] == np.arange(len(known))
    result = known[["sample", "path", "session", "task"]].copy()
    result["nearest_sample"] = known.iloc[nearest]["sample"].to_numpy()
    result["similarity"] = similarity
    result["second_similarity"] = second_similarity
    result["reciprocal"] = reciprocal
    result.to_csv(CACHE / "waveform_pairs.csv", index=False)
    report = {
        "n_session2": len(known),
        "stride": args.stride,
        "reciprocal_fraction": float(reciprocal.mean()),
        "reciprocal_high_confidence_0.99_fraction": float(np.mean(reciprocal & (similarity > 0.99))),
        "similarity_quantiles": quantiles(similarity),
        "second_similarity_quantiles": quantiles(second_similarity),
        "reciprocal_similarity_quantiles": quantiles(similarity[reciprocal]),
        "n_unique_reciprocal_pairs_similarity_gt_0.99": int(
            np.sum(reciprocal & (similarity > 0.99) & (np.arange(len(known)) < nearest))
        ),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
