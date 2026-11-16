"""Self-test for final/brainprint (not part of the submission zip).

1. interface: weights_only load, (N,24) for N=1/7/128, finite, x untouched,
   meta without task, cpu + cuda.
2. reproduction: first 1024 Test rows vs E200 submission_frozen.csv (labels must match).
3. final-round simulation on Train (session1 only, optimistic: PCA/stats saw all 20):
   3 subjects act as new users (base prototype removed, 20 enroll samples from the
   early trials frac<=0.6, tested on late trials frac>0.8); 3 other subjects act as
   unregistered (removed, never enrolled); the remaining 14 are registered users
   tested on frac>0.8. Repeated over random draws.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final/brainprint"
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache/e201_final_selftest"
sys.path.insert(0, str(PKG))
import model as M  # noqa: E402


def load(path):
    return np.ascontiguousarray(np.asarray(loadmat(path, variable_names=["data"])["data"],
                                           dtype=np.float32).T)


def load_many(paths):
    with ThreadPoolExecutor(8) as pool:
        return np.stack(list(pool.map(load, paths)))


def metas(frame):
    return [{"task": t} for t in frame["task"]]


def main():
    out = {}
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.load(PKG / "weights.pt", weights_only=True)

    # ---------------- 1. interface
    test_meta = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    x = load_many([DATA / p for p in test_meta["path"][:128]])
    checks = {}
    for d in ["cpu", dev]:
        art = M.load_model(str(PKG), d)
        for n in (1, 7, 128):
            xi = x[:n].copy()
            s = M.predict_batch(art, xi, metas(test_meta[:n]))
            checks[f"{d}_N{n}"] = bool(s.shape == (n, 24) and not np.isnan(s).any()
                                       and np.isfinite(s).any(axis=1).all()
                                       and np.array_equal(xi, x[:n]))
        s_nometa = M.predict_batch(art, x[:16], None)
        checks[f"{d}_meta_none"] = bool(s_nometa.shape == (16, 24) and not np.isnan(s_nometa).any())
    art = M.load_model(str(PKG), dev)
    s = M.predict_batch(art, x[:32], metas(test_meta[:32]))
    checks["no_new_user_predicted_before_enroll"] = bool((s.argmax(1) <= 20).all())
    out["interface"] = checks
    print("interface", checks)

    # ---------------- 2. reproduction vs E200
    ref = pd.read_csv(ROOT / "research/E200_final_freeze/submission_frozen.csv")
    n_rep = 1024
    xr = load_many([DATA / p for p in test_meta["path"][:n_rep]])
    pred = []
    for i in range(0, n_rep, 128):
        sc = M.predict_batch(art, xr[i:i + 128], metas(test_meta[i:i + 128]))
        pred.append(sc.argmax(1))
    pred = np.concatenate(pred)
    agree = float((pred == ref["Label"].to_numpy()[:n_rep]).mean())
    out["reproduction_vs_E200_first1024"] = agree
    print("reproduction agreement", agree)

    # ---------------- 3. Train simulation
    tm = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    tm["frac"] = (tm.groupby(["subject", "task"]).cumcount() + 1) / tm.groupby(
        ["subject", "task"])["trial"].transform("size")
    CACHE.mkdir(parents=True, exist_ok=True)
    fpath = CACHE / "train_raw_scores_inputs.npy"
    if fpath.is_file():
        feats = torch.as_tensor(np.load(fpath), device=art["device"])
    else:
        t0 = time.time()
        chunks = []
        for i in range(0, len(tm), 512):
            xb = load_many([DATA / p for p in tm["path"][i:i + 512]])
            chunks.append(M._features(art, xb).cpu().numpy())
        np.save(fpath, np.concatenate(chunks))
        feats = torch.as_tensor(np.load(fpath), device=art["device"])
        print(f"train features {time.time() - t0:.0f}s")
    tasks = M._task_indices(metas(tm), len(tm))
    subj = tm["subject"].to_numpy()
    frac = tm["frac"].to_numpy()

    def scores_from_feats(a, rows):
        idx = np.nonzero(rows)[0]
        f = feats[torch.as_tensor(idx, device=a["device"])]
        dist = torch.full((len(idx), 23), float("inf"), device=a["device"])
        for t in range(4):
            take = torch.as_tensor(tasks[idx] == t, device=a["device"])
            if take.any():
                d = torch.cdist(M._project(a, f[take], t), a["prototypes"][t])
                dist[take] = d.masked_fill(~a["valid"][t][None], float("inf"))
        sc = torch.cat([torch.full((len(idx), 1), -a["tau"], device=a["device"]), -dist], 1)
        return sc.argmax(1).cpu().numpy()

    rng = np.random.default_rng(20261116)
    runs = []
    for rep in range(20):
        perm = rng.permutation(np.arange(1, 21))
        new_s, unk_s, known_s = perm[:3], perm[3:6], perm[6:]
        a = M.load_model(str(PKG), dev)
        for s_ in list(new_s) + list(unk_s):
            a["valid"][:, s_ - 1] = False
        # enrollment: 20 random early trials per new user (task mix as it comes)
        ex, em, el = [], [], []
        mapping = {int(s_): 21 + k for k, s_ in enumerate(new_s)}
        for s_ in new_s:
            pool = np.nonzero((subj == s_) & (frac <= 0.6))[0]
            pick = rng.choice(pool, 20, replace=False)
            ex.append(pick)
        pick = np.concatenate(ex)
        xe = load_many([DATA / p for p in tm["path"].iloc[pick]])
        labels = np.asarray([mapping[int(s_)] for s_ in subj[pick]], dtype=np.int64)
        a = M.enroll(a, xe, metas(tm.iloc[pick]), labels)
        late = frac > 0.8
        r_known = late & np.isin(subj, known_s)
        r_new = late & np.isin(subj, new_s)
        r_unk = np.isin(subj, unk_s)
        acc_known = float((scores_from_feats(a, r_known) == subj[r_known]).mean())
        truth_new = np.asarray([mapping[int(s_)] for s_ in subj[r_new]])
        acc_new = float((scores_from_feats(a, r_new) == truth_new).mean())
        rej_unk = float((scores_from_feats(a, r_unk) == 0).mean())
        runs.append({"rep": rep, "new": new_s.tolist(), "unknown": unk_s.tolist(),
                     "acc_known": acc_known, "acc_new": acc_new, "rej_unknown": rej_unk})
        print(runs[-1])
    df = pd.DataFrame(runs)
    out["train_simulation"] = {
        "reps": len(df),
        "acc_known_mean": df.acc_known.mean(), "acc_new_mean": df.acc_new.mean(),
        "rej_unknown_mean": df.rej_unknown.mean(),
        "acc_new_min": df.acc_new.min(), "rej_unknown_min": df.rej_unknown.min(),
        "runs": runs,
    }
    (ROOT / "final/tools/selftest_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["train_simulation"].items() if k != "runs"},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
