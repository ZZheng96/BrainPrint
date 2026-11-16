from __future__ import annotations
import numpy as np

class UnknownGate:
    def __init__(self):
        self.mean = None
        self.std = None
        self.w = None
        self.b = 0.0
        self.tau = 0.5

    def fit(self, features, known, epochs=500, lr=0.05):
        x = np.asarray(features, dtype=np.float64)
        y = known.astype(np.float64)
        self.mean = x.mean(0)
        self.std = x.std(0)
        self.std[self.std < 1e-8] = 1.0
        x = (x - self.mean) / self.std
        self.w = np.zeros(x.shape[1], dtype=np.float64)
        self.b = 0.0
        pos_weight = max(1.0, float((y == 0).sum()) / max(1, (y == 1).sum()))
        for _ in range(epochs):
            q = np.clip(x @ self.w + self.b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-q))
            weight = np.where(y == 1, pos_weight, 1.0)
            err = (p - y) * weight
            self.w -= lr * (x.T @ err / len(y) + 1e-3 * self.w)
            self.b -= lr * float(err.mean())
        p = self.predict_known(features)
        self.tau = float(np.quantile(p[known == 1], 0.10))
        return self

    def predict_known(self, features):
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.std
        q = np.clip(x @ self.w + self.b, -30, 30)
        return 1.0 / (1.0 + np.exp(-q))

def rejection_features(z, proto, e65, e162):
    z = np.asarray(z, dtype=np.float32)
    proto = np.asarray(proto, dtype=np.float32)
    sim = z @ proto.T
    order = np.argsort(sim, axis=1)[:, ::-1]
    best = sim[np.arange(len(z)), order[:, 0]]
    second = sim[np.arange(len(z)), order[:, 1]]
    e65 = np.asarray(e65, dtype=np.float32)
    e162 = np.asarray(e162, dtype=np.float32)
    e65n = e65 / np.maximum(np.linalg.norm(e65, axis=1, keepdims=True), 1e-8)
    e162n = e162 / np.maximum(np.linalg.norm(e162, axis=1, keepdims=True), 1e-8)
    return np.column_stack([best, best-second, np.linalg.norm(z, axis=1), e65n.mean(1), e162n.mean(1)]), order[:, 0], best

def score_gate(gate, features, s_a, tau_a=None):
    p = gate.predict_known(features)
    accepted = p >= gate.tau
    if tau_a is not None:
        accepted &= s_a >= tau_a
    return p, accepted

def binary_report(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= 0.5))}
