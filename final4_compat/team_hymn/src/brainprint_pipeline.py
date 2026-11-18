from pathlib import Path
import numpy as np
import torch
import euclidean_gate as gate
import cosine_rank as rank


def load_model(model_dir: str, device: str = "cpu"):
    w = torch.load(Path(model_dir) / "weights.pt", weights_only=True, map_location="cpu")
    return {"gate": gate.from_weights(w["euclidean_gate"], device), "rank": rank.from_weights(w["cosine_rank"], device)}


def predict_batch(artifact, x, meta):
    g = gate.predict_batch(artifact["gate"], x, meta)
    s = rank.predict_batch(artifact["rank"], x, meta)
    best = s[:, 1:].max(axis=1)
    s[:, 0] = np.where(g.argmax(axis=1) == 0, best + 1, best - 1)
    s[~np.isfinite(best), 0] = 0.0
    return s


def enroll(artifact, x, meta, labels):
    gate.enroll(artifact["gate"], x, meta, labels)
    rank.enroll(artifact["rank"], x, meta, labels)
    return artifact
