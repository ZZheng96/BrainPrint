"""E131: fuse the global and per-identity acceptance scores.

E130 produced two complementary operating points on the same PCA-96 space:
  global threshold         Test S1 0.7933 / Test S2 0.5872  (AUC 0.7655)
  per-identity thresholds  Test S1 0.6109 / Test S2 0.7214  (AUC 0.7138)

The global score rejects well and accepts poorly; the per-identity score accepts
well and rejects poorly. If their errors are complementary, a rank fusion should
beat both on the joint objective.

Variants:
  A  global only
  B  per-identity only
  C  rank mean of A and B, reference = out-of-fold Train scores
  D  rank mean of A, B and per-task (from E130)
  E  per-identity with stricter quantiles (0.90, 0.93, 0.95) to trade acceptance
     back towards rejection, reporting the full (S1, S2) curve

All thresholds and references come from out-of-fold Train scores only.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "E130_per_identity_thresholds"))
import run_e130 as E130  # reuse the out-of-fold machinery

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CLASSES = np.arange(1, 21, dtype=int)


def three_way(train_sig, test_sig, is_s1, target):
    finite = np.isfinite(train_sig)
    th = float(np.quantile(train_sig[finite], 1 - target))
    lab = (test_sig >= th).astype(int)
    return {
        "train_known_acceptance": float(np.mean(train_sig[finite] >= th)),
        "Test_S1_rejection": float(np.mean(lab[is_s1] == 0)),
        "Test_S2_acceptance": float(np.mean(lab[~is_s1] == 1)),
        "diagnostic_AUC": float(roc_auc_score(
            np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
            np.r_[test_sig[is_s1], test_sig[~is_s1]])),
    }


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def run():
    meta = pd.read_csv(E130.TRAIN_META)
    feat = np.load(E130.TRAIN_FEATURES)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = E130.frac(meta)

    mapping = pd.read_csv(ROOT / E130.CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = np.load(E130.TEST_CACHE)

    target = CFG["targets"]["known_acceptance"]
    mu, V = E130.pca_basis(np.asarray(feat[f <= 0.6], dtype=np.float64), CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    blocks = np.clip((f * CFG["oof_blocks"]).astype(int), 0, CFG["oof_blocks"] - 1)
    d_oof, keys = E130.oof_distances(tr_p, subjects, tasks, blocks, CFG["oof_blocks"])
    nearest = np.argmin(d_oof, axis=1)

    protos, keys_full = E130.build_prototypes(tr_p, subjects, tasks, np.ones(len(tr_p), bool), CLASSES)
    d_test = E130.own_task_distances(te_p, protos, keys_full, test_tasks)

    results = {}

    # A: global
    g_tr = E130.global_score(d_oof)
    g_te = E130.global_score(d_test)
    results["A_global"] = three_way(g_tr, g_te, is_s1, target)

    # B: per-identity at the nominal target
    th_id = E130.calibrate(d_oof, keys, nearest, target, "identity")
    b_tr = E130.per_identity_score(d_oof, keys, th_id, "identity")
    b_te = E130.per_identity_score(d_test, keys, th_id, "identity")
    results["B_identity"] = three_way(b_tr, b_te, is_s1, target)

    # C: rank fusion of A and B
    c_tr = 0.5 * (rank_normalize(g_tr, g_tr) + rank_normalize(b_tr, b_tr))
    c_te = 0.5 * (rank_normalize(g_te, g_tr) + rank_normalize(b_te, b_tr))
    results["C_fuse_global_identity"] = three_way(c_tr, c_te, is_s1, target)

    # D: add per-task
    th_task = E130.calibrate(d_oof, keys, nearest, target, "task")
    t_tr = E130.per_identity_score(d_oof, keys, th_task, "task")
    t_te = E130.per_identity_score(d_test, keys, th_task, "task")
    d_tr = (1 / 3) * (rank_normalize(g_tr, g_tr) + rank_normalize(b_tr, b_tr)
                      + rank_normalize(t_tr, t_tr))
    d_te = (1 / 3) * (rank_normalize(g_te, g_tr) + rank_normalize(b_te, b_tr)
                      + rank_normalize(t_te, t_tr))
    results["D_fuse_three"] = three_way(d_tr, d_te, is_s1, target)

    # E: per-identity stricter quantiles
    curve = []
    for q in CFG["identity_quantiles"]:
        th = E130.calibrate(d_oof, keys, nearest, 1 - q, "identity")
        s_tr = E130.per_identity_score(d_oof, keys, th, "identity")
        s_te = E130.per_identity_score(d_test, keys, th, "identity")
        row = three_way(s_tr, s_te, is_s1, target)
        row["identity_quantile"] = q
        curve.append(row)
    results["E_identity_curve"] = curve

    # F: fusion curve, sweeping the global weight
    fcurve = []
    for wg in CFG["global_weights"]:
        tr_s = wg * rank_normalize(g_tr, g_tr) + (1 - wg) * rank_normalize(b_tr, b_tr)
        te_s = wg * rank_normalize(g_te, g_tr) + (1 - wg) * rank_normalize(b_te, b_tr)
        row = three_way(tr_s, te_s, is_s1, target)
        row["global_weight"] = wg
        fcurve.append(row)
    results["F_fusion_curve"] = fcurve

    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    def best_joint(rows):
        return max(rows, key=lambda r: min(r["Test_S1_rejection"], r["Test_S2_acceptance"]))

    lines = ["# E131 Fusing global and per-identity acceptance", "",
             "E130 给出两个互补工作点：全局阈值（S1 高、S2 低）与逐身份阈值（S2 高、S1 低）。"
             "本实验测试秩融合能否同时抬高两者。", "",
             "## 固定变体", "",
             "| 变体 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | 两者较小值 |",
             "|---|---:|---:|---:|---:|"]
    for name in ("A_global", "B_identity", "C_fuse_global_identity", "D_fuse_three"):
        v = results[name]
        lines.append(f"| {name} | {v['diagnostic_AUC']:.4f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | "
                     f"{min(v['Test_S1_rejection'], v['Test_S2_acceptance']):.4f} |")
    lines += ["", "## 逐身份阈值分位扫描", "",
              "| 分位 | Test S1 拒识 | Test S2 接受 | 两者较小值 |", "|---:|---:|---:|---:|"]
    for v in results["E_identity_curve"]:
        lines.append(f"| {v['identity_quantile']:.2f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | "
                     f"{min(v['Test_S1_rejection'], v['Test_S2_acceptance']):.4f} |")
    lines += ["", "## 融合权重扫描", "",
              "| 全局权重 | Test S1 拒识 | Test S2 接受 | 两者较小值 |", "|---:|---:|---:|---:|"]
    for v in results["F_fusion_curve"]:
        lines.append(f"| {v['global_weight']:.2f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | "
                     f"{min(v['Test_S1_rejection'], v['Test_S2_acceptance']):.4f} |")
    bj_e = best_joint(results["E_identity_curve"])
    bj_f = best_joint(results["F_fusion_curve"])
    lines += ["", f"**逐身份曲线最佳均衡：S1 {bj_e['Test_S1_rejection']:.4f} / "
              f"S2 {bj_e['Test_S2_acceptance']:.4f}（分位 {bj_e['identity_quantile']:.2f}）**",
              f"**融合曲线最佳均衡：S1 {bj_f['Test_S1_rejection']:.4f} / "
              f"S2 {bj_f['Test_S2_acceptance']:.4f}（全局权重 {bj_f['global_weight']:.2f}）**",
              "", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
