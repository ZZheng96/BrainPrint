"""E105: leave-one-subject-out (LOSO) evaluation of the promising candidates.

Protocol requested by the user: enrol 19 of the 20 project identities, hold the
remaining one out as an unregistered person, and measure how well each candidate
separates the held-out identity from the enrolled ones.

This is a Train-only protocol. The project Test is never read. The held-out
identity is a *registered* person in the real task, so this is still a proxy,
but it is the closest proxy available without touching Test.

Candidates compared on identical folds:
  A. E102 task-conditioned distance ratio (correlation-logSPD prototypes)
  B. E95 raw identity max-cosine
  C. E95 raw identity distance ratio
  D. E104 session-invariant encoder distance ratio
  E. mean fusion of A and C
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
E73 = ROOT / "research" / "E73_external_multisession"
E88 = ROOT / "research" / "E88_spatial_channel_projection"
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(E88))
sys.path.insert(0, str(E73))
import run_e54 as E54
from e88_model import E88Spatial18Embedding

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CFG["data"]["channels_18"] = 18
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / "BrainPrint_cache/train_metadata.csv"
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"
E95_CKPT = ROOT / "research" / "E95_e54_final20_candidate" / "model_checkpoint.pt"
E104_ENC = ROOT / "research" / "E104_session_invariant_pretrain" / "encoder.pt"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def build_task_prototypes(feat, subjects, tasks, fit_mask, classes):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(feat[sel], dtype=np.float32).mean(0))
            keys.append((task, int(c)))
    return np.stack(protos), keys


def distance_ratio(feat, protos, keys, tasks):
    ratio = np.full(len(feat), np.nan, dtype=np.float32)
    pred = np.zeros(len(feat), dtype=np.int64)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        cls = np.asarray([keys[i][1] for i in cols])
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
            pred[chunk] = cls[order[:, 0]]
    return ratio, pred


def max_cosine(emb, protos, keys, tasks):
    """Max cosine to any prototype of the same task."""
    conf = np.full(len(emb), np.nan, dtype=np.float32)
    pred = np.zeros(len(emb), dtype=np.int64)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        p = torch.nn.functional.normalize(p, dim=1)
        cls = np.asarray([keys[i][1] for i in cols])
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
            q = torch.nn.functional.normalize(q, dim=1)
            s = (q @ p.T).cpu().numpy()
            conf[chunk] = s.max(1)
            pred[chunk] = cls[s.argmax(1)]
    return conf, pred


def embed_raw(model, raw, batch=256):
    out = []
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_fold(held, feat, raw_emb, inv_emb, subjects, tasks, f):
    """Enrol the other 19 identities; held-out identity is the unknown."""
    enrolled = np.setdiff1d(CLASSES, [held])
    fit = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
    known_idx = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
    unknown_idx = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
    if len(unknown_idx) == 0:
        return None

    out = {}
    # A: correlation-logSPD distance ratio
    p_spd, k_spd = build_task_prototypes(feat, subjects, tasks, fit, enrolled)
    r_spd, pred_spd = distance_ratio(feat, p_spd, k_spd, tasks)
    out["A_spd_ratio"] = (r_spd, pred_spd)
    # B: E95 raw max cosine
    p_raw, k_raw = build_task_prototypes(raw_emb, subjects, tasks, fit, enrolled)
    c_raw, pred_raw = max_cosine(raw_emb, p_raw, k_raw, tasks)
    out["B_raw_cosine"] = (c_raw, pred_raw)
    # C: E95 raw distance ratio
    r_raw, pred_raw2 = distance_ratio(raw_emb, p_raw, k_raw, tasks)
    out["C_raw_ratio"] = (r_raw, pred_raw2)
    # D: E104 session-invariant distance ratio
    p_inv, k_inv = build_task_prototypes(inv_emb, subjects, tasks, fit, enrolled)
    r_inv, pred_inv = distance_ratio(inv_emb, p_inv, k_inv, tasks)
    out["D_inv_ratio"] = (r_inv, pred_inv)

    rows = {}
    for name, (score, pred) in out.items():
        k = score[known_idx]; u = score[unknown_idx]
        if not (np.isfinite(k).all() and np.isfinite(u).all()):
            continue
        auc = float(roc_auc_score(np.r_[np.ones(len(k)), np.zeros(len(u))], np.r_[k, u]))
        # Best joint point under the relaxed target.
        grid = np.unique(np.quantile(np.r_[k, u], np.linspace(0, 1, 401)))
        best = None
        for t in grid:
            ka = float(np.mean(k >= t)); ur = float(np.mean(u < t))
            gap = (1 - ka) + (1 - ur)
            if best is None or gap < best[0]:
                best = (gap, float(t), ka, ur)
        _, th, ka, ur = best
        feas = sum(1 for t in grid
                   if np.mean(k >= t) >= CFG["targets"]["known_acceptance"]
                   and np.mean(u < t) >= CFG["targets"]["unknown_rejection"])
        rows[name] = {"AUC": auc, "best_known_acceptance": ka, "best_unknown_rejection": ur,
                      "feasible_points": int(feas), "threshold": th}
    # E: mean fusion of A and C (rank-normalized on the fold pool)
    if "A_spd_ratio" in out and "C_raw_ratio" in out:
        def pooled(k, u):
            both = np.r_[k, u]
            o = both.argsort().argsort().astype(float) / max(len(both) - 1, 1)
            return o[:len(k)], o[len(k):]
        a_k, a_u = pooled(out["A_spd_ratio"][0][known_idx], out["A_spd_ratio"][0][unknown_idx])
        c_k, c_u = pooled(out["C_raw_ratio"][0][known_idx], out["C_raw_ratio"][0][unknown_idx])
        k = 0.5 * (a_k + c_k); u = 0.5 * (a_u + c_u)
        auc = float(roc_auc_score(np.r_[np.ones(len(k)), np.zeros(len(u))], np.r_[k, u]))
        grid = np.unique(np.quantile(np.r_[k, u], np.linspace(0, 1, 401)))
        best = None
        for t in grid:
            ka = float(np.mean(k >= t)); ur = float(np.mean(u < t))
            gap = (1 - ka) + (1 - ur)
            if best is None or gap < best[0]:
                best = (gap, float(t), ka, ur)
        _, th, ka, ur = best
        feas = sum(1 for t in grid
                   if np.mean(k >= t) >= CFG["targets"]["known_acceptance"]
                   and np.mean(u < t) >= CFG["targets"]["unknown_rejection"])
        rows["E_mean_fusion"] = {"AUC": auc, "best_known_acceptance": ka, "best_unknown_rejection": ur,
                                 "feasible_points": int(feas), "threshold": th}
    return {"held_out": int(held), "N_known": int(len(known_idx)), "N_unknown": int(len(unknown_idx)),
            "candidates": rows}


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E105 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != 11304 or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    e95 = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    e95.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    e95.eval()
    raw_emb = embed_raw(e95, raw)

    inv = E88Spatial18Embedding(CFG).to(DEVICE)
    inv.load_state_dict(torch.load(E104_ENC, map_location=DEVICE, weights_only=False)["state_dict"])
    inv.eval()
    inv_emb = embed_raw(inv, raw)

    folds = []
    for held in CLASSES:
        r = evaluate_fold(held, feat, raw_emb, inv_emb, subjects, tasks, f)
        if r is not None:
            folds.append(r)
            print(f"E105 held={held} done", flush=True)

    names = sorted({n for r in folds for n in r["candidates"]})
    summary = {}
    for n in names:
        vals = [r["candidates"][n] for r in folds if n in r["candidates"]]
        if not vals:
            continue
        summary[n] = {
            "AUC_mean": float(np.mean([v["AUC"] for v in vals])),
            "AUC_std": float(np.std([v["AUC"] for v in vals])),
            "best_known_acceptance_mean": float(np.mean([v["best_known_acceptance"] for v in vals])),
            "best_unknown_rejection_mean": float(np.mean([v["best_unknown_rejection"] for v in vals])),
            "folds_with_feasible_point": int(sum(1 for v in vals if v["feasible_points"] > 0)),
            "n_folds": len(vals),
        }
    result = {"experiment": CFG["experiment"], "test_accessed": False,
              "protocol": "leave-one-identity-out; 19 enrolled, 1 held out as unknown",
              "targets": CFG["targets"], "summary": summary, "folds": folds}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E105 leave-one-identity-out candidate comparison", "",
             "19 个身份注册，留出 1 个身份作为未注册者。未访问项目 Test。", "",
             f"目标：known acceptance ≥ {CFG['targets']['known_acceptance']}，"
             f"unknown rejection ≥ {CFG['targets']['unknown_rejection']}。", "",
             "| 候选 | AUC 均值 | AUC 标准差 | 最佳 known acc | 最佳 unknown rej | 有可行点的折数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for n, s in summary.items():
        lines.append(f"| {n} | {s['AUC_mean']:.4f} | {s['AUC_std']:.4f} | "
                     f"{s['best_known_acceptance_mean']:.4f} | {s['best_unknown_rejection_mean']:.4f} | "
                     f"{s['folds_with_feasible_point']}/{s['n_folds']} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
