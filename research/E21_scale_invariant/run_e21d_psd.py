"""E21D：PSD 相对功率表示的跨会话检验。

E21A-C 表明协方差表征的跨会话分数整体衰减且无法区分未知/注册者。
本脚本改为检验频谱形状（逐通道相对功率）这一更不依赖电极阻抗的表征：

  psd_log      1–40 Hz 逐通道 log10 PSD 原样
  psd_rel      逐通道扣除跨频段均值（频谱形状）
  band_rel     5 个频带相对总功率的 log 比值（90 维）
  fused         band_rel + cov_detrend_zband

评估分两部分：
  1) Train 严格嵌套（5 折留被试 + 内层伪未知阈值）：闭集准确率、C、U、AUROC；
  2) Test 无标签审计（写出后才读 session）：S2 接受率、S2 vs S1 的分数 AUROC。
     「S2 为注册者、S1 为未注册者」是项目记忆中的强推断，仅用于审计，不参与任何决策。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
FEATURE_CACHE = CACHE / "e21_scale_invariant"
OUT = Path(__file__).resolve().parent
PSD_DIM = 18 * 90
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))


def band_relative(psd: np.ndarray) -> np.ndarray:
    """逐通道 5 频带功率占总功率比例的对数。"""
    block = psd.reshape(len(psd), 18, 90)
    band_power = np.stack(
        [block[:, :, low - 1 : high - 1].mean(axis=2) for low, high in BANDS], axis=2
    )
    total = band_power.mean(axis=2, keepdims=True)
    return np.log(np.maximum(band_power, 1e-8) / np.maximum(total, 1e-8)).reshape(len(psd), -1)


def psd_relative(psd: np.ndarray) -> np.ndarray:
    block = psd.reshape(len(psd), 18, 90)[:, :, :40]
    block = block - block.mean(axis=2, keepdims=True)
    return block.reshape(len(psd), -1)


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit_model(features: np.ndarray, subjects: np.ndarray, mask: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features[mask], subjects[mask])
    return model


def nested(features: np.ndarray, subjects: np.ndarray, metadata: pd.DataFrame) -> dict:
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_conf = []
        for inner_unknown in np.array_split(inner_order, 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_known = np.isin(subjects, enrolled)
            model = fit_model(features, subjects, mask_known & ~forward & ~inner)
            inner_conf.append(margin(model.decision_function(
                features[np.flatnonzero(mask_known & inner)]
            )))
        threshold = float(np.quantile(np.concatenate(inner_conf), 0.05))

        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = fit_model(features, subjects, outer_fit)
        classes = model.classes_
        known_scores = model.decision_function(features[known_eval])
        unknown_scores = model.decision_function(features[unknown_eval])
        known_conf, unknown_conf = margin(known_scores), margin(unknown_scores)
        known_pred = classes[known_scores.argmax(axis=1)].copy()
        known_pred[known_conf < threshold] = 0
        unknown_pred = classes[unknown_scores.argmax(axis=1)].copy()
        unknown_pred[unknown_conf < threshold] = 0
        detection_true = np.r_[np.ones(len(known_eval)), np.zeros(len(unknown_eval))]
        detection_conf = np.r_[known_conf, unknown_conf]
        rows.append({
            "C": float(np.mean(known_pred == subjects[known_eval])),
            "identity_f1": float(f1_score(subjects[known_eval], known_pred, average="macro", zero_division=0)),
            "U": float(np.mean(unknown_pred == 0)),
            "auroc": float(roc_auc_score(detection_true, detection_conf)),
            "unknown_f1": float(f1_score(detection_true, detection_conf >= threshold, pos_label=0, zero_division=0)),
            "closed_acc": float(np.mean(classes[known_scores.argmax(axis=1)] == subjects[known_eval])),
            "errors": int(np.sum(classes[known_scores.argmax(axis=1)] != subjects[known_eval])),
            "acceptance": float(np.mean(known_conf >= threshold)),
        })
    summary = {k: float(np.mean([r[k] for r in rows])) for k in
               ("C", "identity_f1", "U", "auroc", "unknown_f1", "closed_acc", "acceptance")}
    summary["errors"] = int(sum(r["errors"] for r in rows))
    summary["worst_fold_auroc"] = float(min(r["auroc"] for r in rows))
    return summary


def audit(name: str, train: np.ndarray, test: np.ndarray, subjects: np.ndarray,
          metadata: pd.DataFrame, test_meta: pd.DataFrame) -> dict:
    model = fit_model(train, subjects, np.ones(len(subjects), dtype=bool))
    inner = inner_holdout(metadata)
    scores = model.decision_function(train[inner])
    i_max = np.sort(scores, axis=1)[:, -1]
    i_margin = margin(scores)
    outer = forward_holdout(metadata)
    o_scores = model.decision_function(train[outer])
    o_max, o_margin = np.sort(o_scores, 1)[:, -1], margin(o_scores)

    t_scores = model.decision_function(test)
    t_max, t_margin = np.sort(t_scores, 1)[:, -1], margin(t_scores)
    known = (test_meta["session"] == "session2").to_numpy()

    margin_threshold = float(np.quantile(i_margin, 0.05))
    margin_low = float(np.quantile(i_margin, 0.001))
    max_threshold = float(np.quantile(i_max, 0.05))
    accept_q05 = t_margin >= margin_threshold
    accept_two = (t_margin >= margin_low) & (t_max >= max_threshold)
    return {
        "name": name,
        "train_inner_max_p50": float(np.median(i_max)),
        "train_inner_margin_p50": float(np.median(i_margin)),
        "train_outer_max_p50": float(np.median(o_max)),
        "train_outer_margin_p50": float(np.median(o_margin)),
        "test_s2_max_p50": float(np.median(t_max[known])),
        "test_s2_margin_p50": float(np.median(t_margin[known])),
        "test_s1_max_p50": float(np.median(t_max[~known])),
        "test_s1_margin_p50": float(np.median(t_margin[~known])),
        "audit_auroc_s2_vs_s1_max": float(roc_auc_score(known, t_max)),
        "audit_auroc_s2_vs_s1_margin": float(roc_auc_score(known, t_margin)),
        "s2_accept_margin_q05": float(accept_q05[known].mean()),
        "s1_accept_margin_q05": float(accept_q05[~known].mean()),
        "s2_accept_two_stage": float(accept_two[known].mean()),
        "s1_accept_two_stage": float(accept_two[~known].mean()),
        "thresholds": {"margin_q05": margin_threshold, "margin_q0001": margin_low,
                       "max_q05": max_threshold},
        "s2_pred_counts": np.bincount(
            model.classes_[t_scores[known].argmax(1)], minlength=21
        ).tolist(),
    }


def main() -> None:
    train_raw = np.load(CACHE / "train_features.npy", mmap_mode="r")
    test_raw = np.load(CACHE / "test_features.npy", mmap_mode="r")
    train_psd = np.asarray(train_raw[:, :PSD_DIM], dtype=np.float32)
    test_psd = np.asarray(test_raw[:, :PSD_DIM], dtype=np.float32)
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["session", "task"])

    cov_zband_train = np.load(FEATURE_CACHE / "train_cov_detrend_zband.npy")
    cov_zband_test = np.load(FEATURE_CACHE / "test_cov_detrend_zband.npy")

    sets = {
        "psd_log": (train_psd, test_psd),
        "psd_rel": (psd_relative(train_psd), psd_relative(test_psd)),
        "band_rel": (band_relative(train_psd), band_relative(test_psd)),
        "fused_band_rel_cov": (
            np.concatenate([band_relative(train_psd), cov_zband_train], axis=1),
            np.concatenate([band_relative(test_psd), cov_zband_test], axis=1),
        ),
    }
    report = {}
    for name, (train, test) in sets.items():
        train = np.asarray(train, dtype=np.float32)
        test = np.asarray(test, dtype=np.float32)
        nested_result = nested(train, subjects, metadata)
        audit_result = audit(name, train, test, subjects, metadata, test_meta)
        report[name] = {"dim": int(train.shape[1]), "nested": nested_result, "audit": audit_result}
        print(
            f"{name:22s} dim={train.shape[1]:5d} closed={nested_result['closed_acc']:.4f} "
            f"err={nested_result['errors']:3d} C={nested_result['C']:.4f} U={nested_result['U']:.4f} "
            f"AUROC={nested_result['auroc']:.4f} | S2 accept(q05)={audit_result['s2_accept_margin_q05']:.4f} "
            f"S2/S1 AUROC max={audit_result['audit_auroc_s2_vs_s1_max']:.4f}",
            flush=True,
        )
    (OUT / "e21d_psd.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
