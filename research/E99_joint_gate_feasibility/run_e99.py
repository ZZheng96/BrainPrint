"""E99: Train-only feasibility audit of the S1/S2 joint gate.

No model is trained and the project Test is never read. The question is purely
statistical: on Train/session1, using leave-identity pseudo-unknown as the
unknown proxy, does any accept rule reach both
    known acceptance >= 0.98  (S2 proxy)
    unknown rejection >= 0.95  (S1 proxy)
for the representations already available (E95 raw identity, E65 SPD)?
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
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
E95 = ROOT / "research" / "E95_e54_final20_candidate"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def raw_scores(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.scores(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def spd_scores(proto, feat, batch=1024):
    p = torch.as_tensor(proto, dtype=torch.float32, device=DEVICE)
    out = []
    for st in range(0, len(feat), batch):
        q = torch.as_tensor(np.array(feat[st:st + batch], dtype=np.float32), device=DEVICE)
        out.append((-torch.cdist(q, p)).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_spd_prototypes(meta, feat, fit_mask):
    subjects = meta.subject.to_numpy(int)
    rows = np.flatnonzero(fit_mask)
    present = np.unique(subjects[rows])
    proto = np.stack([np.asarray(feat[rows[subjects[rows] == c]], dtype=np.float32).mean(0) for c in present])
    return present, proto


def joint_curve(known_conf, unknown_conf, targets):
    """Sweep a single threshold; report the best joint operating points."""
    grid = np.unique(np.quantile(np.r_[known_conf, unknown_conf], np.linspace(0, 1, 401)))
    rows = []
    for t in grid:
        ka = float(np.mean(known_conf >= t))
        ur = float(np.mean(unknown_conf < t))
        rows.append({"threshold": float(t), "known_acceptance": ka, "unknown_rejection": ur})
    df = pd.DataFrame(rows)
    feasible = df[(df.known_acceptance >= targets["known_acceptance"]) &
                  (df.unknown_rejection >= targets["unknown_rejection"])]
    # Closest point to the joint target in (1-ka, 1-ur) space.
    gap = (1 - df.known_acceptance) + (1 - df.unknown_rejection)
    best = df.loc[gap.idxmin()]
    return df, feasible, best


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E99 requires CUDA")
    e54cfg = E54.load_config()
    meta = E54.train_metadata(e54cfg)
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != 11304 or feat.shape != (len(meta), 855) or not np.isfinite(feat).all():
        raise ValueError("Train metadata/feature invariant failed")
    model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    ck = torch.load(E95 / "model_checkpoint.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()

    f = frac(meta); subjects = meta.subject.to_numpy(int)
    pseudo = np.asarray(CFG["calibration"]["pseudo_unknown_subjects"])
    known = np.setdiff1d(CLASSES, pseudo)
    fit = np.isin(subjects, known) & (f <= .6)
    cal_known = np.flatnonzero(np.isin(subjects, known) & (f > .6) & (f <= .8))
    cal_unknown = np.flatnonzero(np.isin(subjects, pseudo) & (f > .6) & (f <= .8))
    proto_classes, proto = build_spd_prototypes(meta, feat, fit)

    raw_all = raw_scores(model, raw)
    spd_all = spd_scores(proto, feat)
    branches = {
        "raw": (raw_all[cal_known].max(1), raw_all[cal_unknown].max(1)),
        "spd": (spd_all[cal_known].max(1), spd_all[cal_unknown].max(1)),
    }
    # Rank-normalized min fusion, computed on the calibration pool only.
    def rank_pair(k, u):
        both = np.r_[k, u]
        order = both.argsort().argsort().astype(float) / max(len(both) - 1, 1)
        return order[:len(k)], order[len(k):]
    rk, ru = rank_pair(*branches["raw"])
    sk, su = rank_pair(*branches["spd"])
    branches["min_fusion"] = (np.minimum(rk, sk), np.minimum(ru, su))

    targets = CFG["targets"]
    report = {"experiment": CFG["experiment"], "test_accessed": False,
              "N_known": int(len(cal_known)), "N_pseudo_unknown": int(len(cal_unknown)),
              "targets": targets, "branches": {}}
    curves = {}
    for name, (k, u) in branches.items():
        auc = float(roc_auc_score(np.r_[np.ones(len(k)), np.zeros(len(u))], np.r_[k, u]))
        df, feasible, best = joint_curve(k, u, targets)
        curves[name] = df
        report["branches"][name] = {
            "AUC": auc,
            "feasible_points": int(len(feasible)),
            "best_joint": {"threshold": float(best.threshold),
                           "known_acceptance": float(best.known_acceptance),
                           "unknown_rejection": float(best.unknown_rejection)},
            "known_acceptance_at_unknown_rejection_95": float(
                df.loc[(df.unknown_rejection >= targets["unknown_rejection"]).idxmax(), "known_acceptance"]
            ) if (df.unknown_rejection >= targets["unknown_rejection"]).any() else 0.0,
        }
    report["any_feasible"] = any(v["feasible_points"] > 0 for v in report["branches"].values())
    report["conclusion"] = (
        "Train-only 代理下存在同时满足 S1/S2 目标的工作点" if report["any_feasible"]
        else "Train-only 代理下不存在同时满足 S1/S2 目标的工作点；瓶颈是分布重叠"
    )
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, df in curves.items():
        df.to_csv(OUT / f"curve_{name}.csv", index=False)

    lines = ["# E99 Train-only joint gate feasibility audit", "",
             "未训练新模型，未读取项目 Test。未知代理为留身份 pseudo-unknown。", "",
             f"目标：known acceptance ≥ {targets['known_acceptance']}，unknown rejection ≥ {targets['unknown_rejection']}。", "",
             "| 分支 | AUC | 可行工作点数 | 最佳联合点 known acc | 最佳联合点 unknown rej | U@95 时 known acc |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, v in report["branches"].items():
        lines.append(f"| {name} | {v['AUC']:.6f} | {v['feasible_points']} | "
                     f"{v['best_joint']['known_acceptance']:.6f} | {v['best_joint']['unknown_rejection']:.6f} | "
                     f"{v['known_acceptance_at_unknown_rejection_95']:.6f} |")
    lines += ["", f"**结论**：{report['conclusion']}。", "",
              "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
