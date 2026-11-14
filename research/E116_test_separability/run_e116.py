"""E116: diagnostic only. How separable are Test S1 and S2 on the existing scores?

This experiment does NOT produce a submission candidate. It answers one
question: given the scores we already have, what is the best achievable
(S1 rejection, S2 acceptance) pair on the project Test?

That is an upper bound, not a deployable result, because the threshold is
chosen using the Session grouping. It tells us whether the remaining gap is a
threshold problem or a representation problem.

Scores analysed (all already computed, no new training):
  E95 raw cosine        research/E95_e54_final20_candidate/predictions.csv
  E102 SPD ratio        research/E102_task_distance_ratio/predictions.csv
  E115 V3 cosine        research/E115_v3_frozen_audit/predictions.csv

For each score we report, over the Test Session split:
  - the S1/S2 score distributions (quantiles)
  - the diagnostic AUC of S1 vs S2
  - the best achievable S2 acceptance at S1 rejection >= 0.80 / 0.90 / 0.95
  - the best achievable S1 rejection at S2 acceptance >= 0.85 / 0.90 / 0.95

The predictions.csv files only contain hard labels, so we recompute the
continuous scores from the frozen checkpoints where needed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e54 as E54
import run_e65
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / "BrainPrint_cache/train_metadata.csv"
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"
E95_CKPT = ROOT / "research" / "E95_e54_final20_candidate" / "model_checkpoint.pt"
E115_CKPT = ROOT / "research" / "E115_v3_frozen_audit" / "model_checkpoint.pt"
E115_PROTO = ROOT / "research" / "E115_v3_frozen_audit" / "frozen_prototypes.npz"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def load_test_raw(mapping):
    root = ROOT / CFG["data"]["train_root"]
    chunks = []
    for st in range(0, len(mapping), 64):
        paths = [root / p for p in mapping.path.iloc[st:st + 64]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"],
                                   dtype=np.float32).T for p in paths])
        chunks.append(torch.as_tensor(arr, dtype=torch.float32, device=DEVICE))
    return chunks


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
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
    return ratio


def max_cosine(emb, protos, keys, tasks):
    out = np.full(len(emb), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        p = torch.nn.functional.normalize(p, dim=1)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
            q = torch.nn.functional.normalize(q, dim=1)
            out[chunk] = (q @ p.T).max(1).values.cpu().numpy()
    return out


def embed(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def separability(score, is_s1):
    """Diagnostic upper bound: best (S1 rejection, S2 acceptance) pairs."""
    s1 = score[is_s1]
    s2 = score[~is_s1]
    auc = float(roc_auc_score(np.r_[np.zeros(len(s1)), np.ones(len(s2))], np.r_[s1, s2]))
    grid = np.unique(np.quantile(score, np.linspace(0, 1, 401)))
    rows = []
    for th in grid:
        rows.append((float(th), float(np.mean(s1 < th)), float(np.mean(s2 >= th))))
    rows = np.asarray(rows)

    def best_s2_at(min_s1):
        m = rows[:, 1] >= min_s1
        return float(rows[m, 2].max()) if m.any() else 0.0

    def best_s1_at(min_s2):
        m = rows[:, 2] >= min_s2
        return float(rows[m, 1].max()) if m.any() else 0.0

    # Best joint point: maximise min(S1 rejection, S2 acceptance).
    joint = float(np.max(np.minimum(rows[:, 1], rows[:, 2])))
    return {
        "diagnostic_AUC_S1_vs_S2": auc,
        "S1_quantiles": {str(q): float(np.quantile(s1, q)) for q in (0.1, 0.5, 0.9)},
        "S2_quantiles": {str(q): float(np.quantile(s2, q)) for q in (0.1, 0.5, 0.9)},
        "best_S2_accept_at_S1_rej_0.80": best_s2_at(0.80),
        "best_S2_accept_at_S1_rej_0.90": best_s2_at(0.90),
        "best_S2_accept_at_S1_rej_0.95": best_s2_at(0.95),
        "best_S1_rej_at_S2_accept_0.85": best_s1_at(0.85),
        "best_S1_rej_at_S2_accept_0.90": best_s1_at(0.90),
        "best_S1_rej_at_S2_accept_0.95": best_s1_at(0.95),
        "best_joint_min": joint,
    }


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E116 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)

    # ---- E95 raw cosine
    e95 = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    e95.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    e95.eval()

    # ---- E115 V3 fine-tuned encoder + multi-prototype
    e115 = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    e115.load_state_dict(torch.load(E115_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    e115.eval()
    proto_npz = np.load(E115_PROTO, allow_pickle=False)
    v3_protos = proto_npz["protos"]
    v3_keys = [tuple(s.split("|")) for s in proto_npz["keys"]]
    v3_keys = [(t, int(c)) for t, c in v3_keys]

    # ---- E102 SPD ratio prototypes (all 20 identities, fit <= 0.6)
    fit = f <= CFG["calibration"]["fit_fraction"]
    sp_protos, sp_keys = build_task_prototypes(feat, subjects, tasks, fit, CLASSES)

    scores = {"E95_raw_cosine": [], "E102_spd_ratio": [], "E115_V3_cosine": []}
    for chunk in load_test_raw(mapping):
        n = len(chunk)
        with torch.no_grad():
            s95 = e95.scores(chunk).cpu().numpy()
        scores["E95_raw_cosine"].append(s95.max(1))

        emb115 = embed(e115, chunk)
        scores["E115_V3_cosine"].append(max_cosine(emb115, v3_protos, v3_keys, test_tasks[:n]))

        e36 = load_e36_config()
        e65cfg = json.loads((ROOT / "research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
        with torch.no_grad():
            tf = run_e65.gpu_correlation_logspd(chunk, e36, e65cfg).cpu().numpy()
        scores["E102_spd_ratio"].append(distance_ratio(tf, sp_protos, sp_keys, test_tasks[:n]))

    result = {"experiment": CFG["experiment"], "diagnostic_only": True,
              "warning": "Thresholds here are chosen with the Session grouping. "
                         "This is an upper bound, not a deployable candidate.",
              "N_S1": int(is_s1.sum()), "N_S2": int((~is_s1).sum()), "scores": {}}
    for name, parts in scores.items():
        score = np.concatenate(parts)
        result["scores"][name] = separability(score, is_s1)
        print(f"E116 {name} done", flush=True)

    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E116 Test S1/S2 separability upper bound (diagnostic only)", "",
             "**这不是可提交的候选**：阈值用 Session 分组选出，只用于判断瓶颈在阈值还是表征。", "",
             f"Test: S1 N={result['N_S1']}, S2 N={result['N_S2']}。", "",
             "## 诊断 AUC 与分布", "",
             "| 分数 | S1 vs S2 AUC | S1 p50 | S2 p50 |", "|---|---:|---:|---:|"]
    for name, s in result["scores"].items():
        lines.append(f"| {name} | {s['diagnostic_AUC_S1_vs_S2']:.4f} | "
                     f"{s['S1_quantiles']['0.5']:.4f} | {s['S2_quantiles']['0.5']:.4f} |")
    lines += ["", "## 可达上限", "",
              "| 分数 | S1≥0.80 时 S2 接受 | S1≥0.90 | S1≥0.95 | S2≥0.85 时 S1 拒识 | S2≥0.90 | S2≥0.95 | 最佳均衡点 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in result["scores"].items():
        lines.append(
            f"| {name} | {s['best_S2_accept_at_S1_rej_0.80']:.4f} | "
            f"{s['best_S2_accept_at_S1_rej_0.90']:.4f} | "
            f"{s['best_S2_accept_at_S1_rej_0.95']:.4f} | "
            f"{s['best_S1_rej_at_S2_accept_0.85']:.4f} | "
            f"{s['best_S1_rej_at_S2_accept_0.90']:.4f} | "
            f"{s['best_S1_rej_at_S2_accept_0.95']:.4f} | "
            f"{s['best_joint_min']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
