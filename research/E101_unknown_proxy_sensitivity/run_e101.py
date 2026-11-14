"""E101: is the leave-identity proxy too pessimistic?

E99/E100 used 4 held-out identities as pseudo-unknown. Those identities are
*registered* people in the real task, so their EEG may be closer to the enrolled
set than a true unregistered person. This audit compares several unknown proxies
on the same frozen E95 encoder to see how much the proxy choice moves the curve.

No project Test access.
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


def curve(k, u, targets):
    grid = np.unique(np.quantile(np.r_[k, u], np.linspace(0, 1, 801)))
    rows = [{"threshold": float(t), "known_acceptance": float(np.mean(k >= t)),
             "unknown_rejection": float(np.mean(u < t))} for t in grid]
    df = pd.DataFrame(rows)
    feas = df[(df.known_acceptance >= targets["known_acceptance"]) &
              (df.unknown_rejection >= targets["unknown_rejection"])]
    gap = (1 - df.known_acceptance) + (1 - df.unknown_rejection)
    best = df.loc[gap.idxmin()]
    return df, len(feas), best


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E101 requires CUDA")
    e54cfg = E54.load_config()
    meta = E54.train_metadata(e54cfg)
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    feat = np.load(TRAIN_FEATURES)
    model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    ck = torch.load(E95 / "model_checkpoint.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()

    f = frac(meta); subjects = meta.subject.to_numpy(int)
    raw_all = raw_scores(model, raw)

    # Enrolment set: all 20 identities, first 60% of each task block.
    fit = f <= .6
    proto_classes, proto = build_spd_prototypes(meta, feat, fit)
    spd_all = spd_scores(proto, feat)

    def rank_pair(k, u):
        both = np.r_[k, u]
        order = both.argsort().argsort().astype(float) / max(len(both) - 1, 1)
        return order[:len(k)], order[len(k):]

    targets = CFG["targets"]
    report = {"experiment": CFG["experiment"], "test_accessed": False, "targets": targets, "proxies": {}}

    # Known pool: late blocks of all identities (0.6-0.8), same for every proxy.
    known_idx = np.flatnonzero((f > .6) & (f <= .8))
    k_raw = raw_all[known_idx].max(1)
    k_spd = spd_all[known_idx].max(1)

    proxies = {}
    # Proxy A: held-out identities (E99/E100 definition).
    pseudo = np.asarray(CFG["proxies"]["held_identity"])
    proxies["held_identity"] = np.flatnonzero(np.isin(subjects, pseudo) & (f > .6) & (f <= .8))
    # Proxy B: same identities, but a *different task* than the enrolment block.
    # Enrolment used all tasks, so instead use the last 20% of each block.
    proxies["late_block"] = np.flatnonzero((f > .8))
    # Proxy C: held-out identities, late block only (hardest).
    proxies["held_identity_late"] = np.flatnonzero(np.isin(subjects, pseudo) & (f > .8))

    for name, idx in proxies.items():
        if len(idx) == 0:
            continue
        u_raw = raw_all[idx].max(1)
        u_spd = spd_all[idx].max(1)
        # Rank-normalize on the pooled known+unknown pool so both branches share a scale.
        def pooled_rank(k, u):
            both = np.r_[k, u]
            order = both.argsort().argsort().astype(float) / max(len(both) - 1, 1)
            return order[:len(k)], order[len(k):]
        rk, ru = pooled_rank(k_raw, u_raw)
        sk, su = pooled_rank(k_spd, u_spd)
        variants = {
            "raw": (k_raw, u_raw),
            "spd": (k_spd, u_spd),
            "mean_fusion": (0.5 * (rk + sk), 0.5 * (ru + su)),
        }
        entry = {"N_unknown": int(len(idx)), "branches": {}}
        for bname, (k, u) in variants.items():
            auc = float(roc_auc_score(np.r_[np.ones(len(k)), np.zeros(len(u))], np.r_[k, u]))
            df, nfeas, best = curve(k, u, targets)
            entry["branches"][bname] = {
                "AUC": auc, "feasible_points": int(nfeas),
                "best_known_acceptance": float(best.known_acceptance),
                "best_unknown_rejection": float(best.unknown_rejection),
            }
        report["proxies"][name] = entry

    report["any_feasible"] = any(
        b["feasible_points"] > 0
        for p in report["proxies"].values() for b in p["branches"].values()
    )
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E101 unknown-proxy sensitivity audit", "",
             "未训练新模型，未读取项目 Test。比较三种未知代理在同一冻结 E95 编码器上的表现。", "",
             f"目标：known acceptance ≥ {targets['known_acceptance']}，unknown rejection ≥ {targets['unknown_rejection']}。", "",
             "| 未知代理 | N | 分支 | AUC | 可行点数 | 最佳 known acc | 最佳 unknown rej |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for pname, p in report["proxies"].items():
        for bname, b in p["branches"].items():
            lines.append(f"| {pname} | {p['N_unknown']} | {bname} | {b['AUC']:.6f} | {b['feasible_points']} | "
                         f"{b['best_known_acceptance']:.6f} | {b['best_unknown_rejection']:.6f} |")
    lines += ["", f"**是否存在可行工作点**：{'是' if report['any_feasible'] else '否'}。", "",
              "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
