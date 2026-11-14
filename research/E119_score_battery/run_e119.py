"""E119: score battery for the Test S1/S2 separability problem.

Why this experiment. E105 established that E95's raw cosine, scored against
task-conditioned prototypes rebuilt from the enrolled identities only, reaches
LOSO AUC 0.9353 with unknown rejection 0.899 at known acceptance 0.862. E95
also reaches Test S2 acceptance 0.9414. So E95 already satisfies two of the
three requirements:

    LOSO rejection  0.899   ok
    Test S2 accept  0.9414  ok
    Test S1 reject  0.1543  missing

(The E109 figure of 0.157 for E95 was contaminated: that run scored against the
frozen 20-class prototype matrix, which still contains the held-out identity.)

So the only remaining gap is Test S1 rejection. This experiment measures, on the
project Test, the diagnostic AUC of a battery of deployable scores built on the
E95 embedding and on the correlation-logSPD features.

All scores are computed without using the Session column. Session is used only
afterwards to measure separability, which makes this diagnostic, not a
submission candidate.

Scores:
  S01 e95_max          max cosine to task-conditioned prototypes
  S02 e95_margin       max - second
  S03 e95_ratio        (1-second)/(1-max)
  S04 e95_neg_energy   -logsumexp(scores)
  S05 e95_entropy      entropy of softmax(scores/T)
  S06 e95_max_taskz    e95_max z-scored within task (Train known reference)
  S07 e95_max_idcal    e95_max divided by the predicted identity's Train median
  S08 e95_mahal        negative Mahalanobis distance to the Train embedding Gaussian
  S09 spd_ratio        E102 task-conditioned distance ratio
  S10 spd_max          max cosine on correlation-logSPD prototypes
  S11 mahal_x_spd      geometric mean of the two rank-normalised OOD scores
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
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
E95_CKPT = ROOT / "research" / "E95_e54_final20_candidate" / "model_checkpoint.pt"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def build_task_prototypes(emb, subjects, tasks, fit_mask, classes):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(emb[sel], dtype=np.float32).mean(0))
            keys.append((task, int(c)))
    return np.stack(protos), keys


def task_scores(emb, protos, keys, tasks):
    """Full cosine matrix; columns outside the sample's own task stay -inf."""
    n = len(emb)
    out = np.full((n, len(keys)), -np.inf, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        p = torch.nn.functional.normalize(p, dim=1, eps=1e-12)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
            q = torch.nn.functional.normalize(q, dim=1, eps=1e-12)
            out[np.ix_(chunk, cols)] = (q @ p.T).cpu().numpy()
    return out


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


def embed(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def core_scores(S):
    """Derive the embedding-side scores from a cosine matrix."""
    order = np.argsort(-S, axis=1)
    top1 = S[np.arange(len(S)), order[:, 0]]
    top2 = S[np.arange(len(S)), order[:, 1]]
    T = 0.05
    logits = S / T
    logits = logits - logits.max(1, keepdims=True)
    P = np.exp(logits)
    P = P / P.sum(1, keepdims=True)
    entropy = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    neg_energy = -np.log(np.exp(logits).sum(1) + 1e-30) * T
    return top1, top2, order[:, 0], entropy, neg_energy


def separability(score, is_s1):
    s1 = score[is_s1]
    s2 = score[~is_s1]
    auc = float(roc_auc_score(np.r_[np.zeros(len(s1)), np.ones(len(s2))], np.r_[s1, s2]))
    grid = np.unique(np.quantile(score, np.linspace(0, 1, 401)))
    ka = np.asarray([np.mean(s1 < t) for t in grid])
    ur = np.asarray([np.mean(s2 >= t) for t in grid])

    def best_at(min_s1):
        m = ka >= min_s1
        return float(ur[m].max()) if m.any() else 0.0

    return {
        "diagnostic_AUC": auc,
        "best_S2_accept_at_S1_rej_0.80": best_at(0.80),
        "best_S2_accept_at_S1_rej_0.90": best_at(0.90),
        "best_joint_min": float(np.max(np.minimum(ka, ur))),
    }


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E119 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["calibration"]["fit_fraction"]

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    e95 = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    e95.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
    e95.eval()

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()

    # ---- Train references
    train_emb = embed(e95, raw)
    protos, keys = build_task_prototypes(train_emb, subjects, tasks, fit_rows, CLASSES)
    train_S = task_scores(train_emb, protos, keys, tasks)
    t1, t2, tpred, _, _ = core_scores(train_S)

    sp_protos, sp_keys = build_task_prototypes(feat, subjects, tasks, fit_rows, CLASSES)
    sp_ratio_train = distance_ratio(feat, sp_protos, sp_keys, tasks)

    ref_emb = np.asarray(train_emb[fit_rows], dtype=np.float64)
    mu = ref_emb.mean(0)
    cov = np.cov(ref_emb, rowvar=False) + np.eye(ref_emb.shape[1]) * 1e-3
    inv = np.linalg.inv(cov)
    d_tr = ref_emb - mu
    ref_mahal = np.einsum("ij,jk,ik->i", d_tr, inv, d_tr)

    # ---- Test scores
    root = ROOT / CFG["data"]["train_root"]
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research" / "E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    emb_parts, feat_parts, task_parts = [], [], []
    for st in range(0, len(mapping), 64):
        stop = min(st + 64, len(mapping))
        paths = [root / p for p in mapping.path.iloc[st:stop]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"],
                                   dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        emb_parts.append(embed(e95, x))
        with torch.no_grad():
            feat_parts.append(run_e65.gpu_correlation_logspd(x, e36, e65cfg).cpu().numpy())
        task_parts.append(test_tasks[st:stop])
        print(f"E119 test chunk {stop}/{len(mapping)}", flush=True)

    test_emb = np.concatenate(emb_parts)
    test_feat = np.concatenate(feat_parts)
    test_task = np.concatenate(task_parts)

    test_S = task_scores(test_emb, protos, keys, test_task)
    top1, top2, pred_col, entropy, neg_energy = core_scores(test_S)

    scores = {
        "S01_e95_max": top1,
        "S02_e95_margin": top1 - top2,
        "S03_e95_ratio": (1.0 - top2) / np.maximum(1.0 - top1, 1e-6),
        "S04_e95_neg_energy": neg_energy,
        "S05_e95_entropy": entropy,
    }

    task_z = np.full(len(test_S), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        sel = fit_rows & (tasks == task)
        mu_t = float(t1[sel].mean())
        sd_t = float(t1[sel].std() + 1e-9)
        m = test_task == task
        task_z[m] = (top1[m] - mu_t) / sd_t
    scores["S06_e95_max_taskz"] = task_z

    id_med = {}
    for col in np.unique(tpred[fit_rows]):
        sel = fit_rows & (tpred == col)
        if sel.sum() > 0:
            id_med[int(col)] = float(np.median(t1[sel]))
    scores["S07_e95_max_idcal"] = np.array(
        [top1[i] / max(id_med.get(int(pred_col[i]), 1.0), 1e-6) for i in range(len(top1))],
        dtype=np.float32)

    d_te = np.asarray(test_emb, dtype=np.float64) - mu
    mahal_test = np.einsum("ij,jk,ik->i", d_te, inv, d_te)
    scores["S08_e95_mahal"] = -mahal_test.astype(np.float32)

    scores["S09_spd_ratio"] = distance_ratio(test_feat, sp_protos, sp_keys, test_task)
    scores["S10_spd_max"] = task_scores(test_feat, sp_protos, sp_keys, test_task).max(1)

    r1 = rank_normalize(scores["S08_e95_mahal"], -ref_mahal)
    r2 = rank_normalize(scores["S09_spd_ratio"], sp_ratio_train[fit_rows])
    scores["S11_mahal_x_spd"] = np.sqrt(np.clip(r1 * r2, 0, None))

    result = {"experiment": CFG["experiment"], "diagnostic_only": True,
              "warning": "Session is used only to measure separability, never to build a score.",
              "N_S1": int(is_s1.sum()), "N_S2": int((~is_s1).sum()), "scores": {}}
    for name, sc in scores.items():
        sc = np.asarray(sc, dtype=np.float64)
        if not np.isfinite(sc).all():
            continue
        result["scores"][name] = separability(sc, is_s1)

    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E119 Test S1/S2 score battery (diagnostic only)", "",
             "E105 已证明 E95 raw cosine 的 LOSO 拒识是 **0.899**；E109 报的 0.157 是被冻结 20 类原型矩阵污染的错数。"
             "加上 Test S2 接受率 0.9414，E95 已满足三项中的两项，唯一缺口是 Test S1 拒识（0.1543）。", "",
             "本实验在 Test 上测量一批可部署分数的 S1-vs-S2 诊断 AUC。"
             "Session 只用于事后测量，不参与打分。", "",
             "| 分数 | 诊断 AUC | S1≥0.80 时 S2 接受 | S1≥0.90 | 最佳均衡点 |",
             "|---|---:|---:|---:|---:|"]
    for name in sorted(result["scores"]):
        s = result["scores"][name]
        lines.append(f"| {name} | {s['diagnostic_AUC']:.4f} | "
                     f"{s['best_S2_accept_at_S1_rej_0.80']:.4f} | "
                     f"{s['best_S2_accept_at_S1_rej_0.90']:.4f} | "
                     f"{s['best_joint_min']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
