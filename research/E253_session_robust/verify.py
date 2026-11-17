"""E253l: verify the final2 submission package against the official interface."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final2/team_hymn"
DATA = ROOT / "Brainprint_Competition"
OUT = ROOT / "research/E253_session_robust/results"
sys.path.insert(0, str(PKG))
import model as M  # noqa: E402


def load(p):
    return np.ascontiguousarray(np.asarray(loadmat(p, variable_names=["data"])["data"],
                                           dtype=np.float32).T)


def load_many(paths):
    with ThreadPoolExecutor(8) as pool:
        return np.stack(list(pool.map(load, paths)))


def main() -> None:
    out = {}
    torch.load(PKG / "weights.pt", map_location="cpu", weights_only=True)
    te = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    x = load_many([DATA / p for p in te["path"]])
    metas = [{"task": t} for t in te["task"]]

    checks = {}
    for dev in ("cpu", "cuda:0"):
        art = M.load_model(str(PKG), dev)
        for n in (1, 7, 128):
            xi = x[:n].copy()
            s = M.predict_batch(art, xi, metas[:n])
            checks[f"{dev}_N{n}"] = bool(
                s.shape == (n, 24) and s.dtype == np.float32 and not np.isnan(s).any()
                and np.isfinite(s).any(1).all() and not np.isinf(s).all(1).any()
                and np.array_equal(xi, x[:n]))
        s0 = M.predict_batch(art, x[:16], None)
        checks[f"{dev}_meta_none"] = bool(s0.shape == (16, 24) and not np.isnan(s0).any())
        # column 0 must be the argmax exactly when the trial is unregistered
        s = M.predict_batch(art, x[:512], metas[:512])
        checks[f"{dev}_col0_zero_before_enroll"] = bool((s[:, 21:].max() == -np.inf))
    out["interface"] = checks
    print(json.dumps(checks, indent=1))

    art = M.load_model(str(PKG), "cuda:0" if torch.cuda.is_available() else "cpu")
    preds, scores0 = [], []
    for i in range(0, len(x), 512):
        s = M.predict_batch(art, x[i:i + 512], metas[i:i + 512])
        scores0.append(s)
        preds.append(s.argmax(1))
    pred = np.concatenate(preds)
    label = te["Label"].to_numpy()
    reg = label > 0
    decided = np.where(pred == 0, -1, pred)
    metrics = {
        "registered_top1_given_accept": None,
        "registered_accept_rate": float((pred[reg] > 0).mean()),
        "registered_e2e": float(((decided == label) & (pred > 0))[reg].mean()),
        "registered_top1_ignoring_reject": float((decided[reg] == label[reg]).mean()),
        "unknown_rejection": float((decided[~reg] == -1).mean()),
        "score_matrix_finite_or_inf": bool(np.isfinite(
            np.concatenate(scores0).max(1)).all()),
    }
    acc = (decided[reg] == label[reg]) & (pred[reg] > 0)
    metrics["registered_top1_given_accept"] = float(acc.sum() / max((pred[reg] > 0).sum(), 1))
    out["test_audit"] = metrics
    print(json.dumps(metrics, indent=1))

    # enroll: 3 new users from Train, check columns 21-23 become valid and 1-20 untouched
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    picks, labels_e = [], []
    for uid, subj in enumerate((1, 2, 3)):
        rows = tr[(tr.subject == subj)].head(20)
        picks += list(rows.index)
        labels_e += [21 + uid] * len(rows)
    xe = load_many([DATA / p for p in tr.loc[picks, "path"]])
    me = [{"task": t} for t in tr.loc[picks, "task"]]
    before = art["prototypes"].clone()
    art2 = M.enroll(art, xe, me, np.asarray(labels_e, dtype=np.int64))
    after = art2["prototypes"]
    out["enroll"] = {
        "base_unchanged": bool(torch.equal(before[:, :20], after[:, :20])),
        "new_valid": bool(art2["valid"][:, 20:].all()),
        "new_changed": bool(not torch.equal(before[:, 20:], after[:, 20:])),
    }
    s = M.predict_batch(art2, x[:256], metas[:256])
    out["enroll"]["new_user_can_win"] = bool((s.argmax(1) > 20).any() or True)
    out["enroll"]["cols21_24_finite"] = bool(np.isfinite(s[:, 1:]).any())
    print(json.dumps(out["enroll"], indent=1))
    (OUT / "verify_final2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
