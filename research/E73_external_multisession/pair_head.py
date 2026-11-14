"""Generic symmetric sample-to-template matching head for frozen EEG embeddings."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def subject_templates(embeddings: torch.Tensor, subjects: np.ndarray,
                      classes: np.ndarray) -> torch.Tensor:
    """Average each class's enrollment trials without using query samples."""
    subjects = np.asarray(subjects)
    classes = np.asarray(classes, dtype=int)
    return torch.stack([
        embeddings[torch.as_tensor(subjects == subject, device=embeddings.device)].mean(dim=0)
        for subject in classes
    ])


def same_different_labels(query_subjects: np.ndarray, classes: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    query_subjects = np.asarray(query_subjects, dtype=int)
    classes = np.asarray(classes, dtype=int)
    return torch.as_tensor(
        query_subjects[:, None] == classes[None, :], dtype=torch.float32, device=device
    )


def symmetric_pair_features(query: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Return N x K x (2D+1) exchange-invariant sample/template features."""
    if query.ndim != 2 or template.ndim != 2 or query.shape[1] != template.shape[1]:
        raise ValueError("Expected query NxD and template KxD with equal D")
    query = F.normalize(query.float(), dim=1, eps=1e-12)
    template = F.normalize(template.float(), dim=1, eps=1e-12)
    q = query[:, None, :]
    t = template[None, :, :]
    cosine = (q * t).sum(dim=-1, keepdim=True)
    return torch.cat([q * t, (q - t).abs(), cosine], dim=-1)


class SampleTemplatePairHead(nn.Module):
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * self.embedding_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        features = symmetric_pair_features(query, template)
        return self.network(features).squeeze(-1)
