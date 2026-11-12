"""E24A：扰动检测（范数门控）与候选生成。

E23 发现 Test S2 特征范数双峰：低范数 7,985 条（均值 33.9）、高范数 11,233 条（均值 58.0），
Train 范数集中在 56–62。本实验检验：

1. 低范数是否与「对抗扰动」一致：对 Train 原始信号注入不同 SNR 的白噪声，看特征范数是否掉到 ~34；
2. 范数门控（阈值只由 Train 分位给出）能在几乎不损失 C 的前提下拒掉疑似扰动样本；
3. 三种接受策略的差异：只门控 / 门控 + 极松 margin / 门控 + 常规 margin。

Test 只做无标签审计；Test 的 AUC/F1/C/RA/ASafeR/U 均为 N/A。
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

from src.eeg_features import extract_features, load_eeg
from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
DATA = ROOT / "Brainprint_Competition"
OUT = Path(__file__).resolve().parent
COV_SLICE = slice(int(COV_COLUMNS[0]), int(COV_COLUMNS[-1]) + 1)
LOW_GROUP_NORM = 45.0


def cov_of(path: str, noise_snr_db: float | None = None, seed: int = 0) -> np.ndarray:
    data, _ = load_eeg(DATA / path)
    if noise_snr_db is not None:
        generator = np.random.default_rng(seed)
        noise_power = np.mean(data**2) / (10 ** (noise_snr_db / 10))
        data = data + generator.normal(0.0, np.sqrt(noise_power), data.shape)
    features, _ = extract_features(data)
    return features[COV_SLICE]


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features, labels)
    return model


def nested_evaluate(train, subjects, metadata, train_gate, margin_quantile: float | None) -> dict:
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    groups = np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5)
    rows = []
    for fold, unknown in enumerate(groups, 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        is_known = np.isin(subjects, known_subjects)
        fit_mask = is_known & ~forward
        confidence = []
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_known = np.isin(subjects, enrolled)
            model = fit(train[mask_known & ~forward & ~inner], subjects[mask_known & ~forward & ~inner])
            confidence.append(margin(model.decision_function(train[np.flatnonzero(mask_known & inner)])))
        threshold = float(np.quantile(np.concatenate(confidence), margin_quantile)) if margin_quantile is not None else -np.inf

        evaluate = np.flatnonzero(is_known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown))
        model = fit(train[fit_mask], subjects[fit_mask])
        classes = model.classes_
        scores = model.decision_function(train[evaluate])
        unknown_scores = model.decision_function(train[unknown_eval])
        conf, unknown_conf = margin(scores), margin(unknown_scores)
        accepted = (conf >= threshold) & ~train_gate[evaluate]
        pred = classes[scores.argmax(1)].copy()
        pred[~accepted] = 0
        unknown_accepted = (unknown_conf >= threshold) & ~train_gate[unknown_eval]
        unknown_pred = classes[unknown_scores.argmax(1)].copy()
        unknown_pred[~unknown_accepted] = 0
        detection_true = np.r_[np.ones(len(evaluate)), np.zeros(len(unknown_eval))]
        rows.append({
            "C": float(np.mean(pred == subjects[evaluate])),
            "identity_f1": float(f1_score(subjects[evaluate], pred, average="macro", zero_division=0)),
            "U": float(np.mean(unknown_pred == 0)),
            "auroc": float(roc_auc_score(detection_true, np.r_[conf, unknown_conf])),
            "acceptance": float(accepted.mean()),
        })
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    summary["worst_fold_auroc"] = float(min(row["auroc"] for row in rows))
    return summary


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial", "path"])
    subjects = metadata["subject"].to_numpy()
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session", "task"])
    known = (test_meta["session"] == "session2").to_numpy()

    train = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    test = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    train_norm = np.linalg.norm(train, axis=1)
    test_norm = np.linalg.norm(test, axis=1)

    # 1. 噪声注入验证
    rng = np.random.default_rng(0)
    sample = rng.choice(len(metadata), 150, replace=False)
    noise_report = {}
    for snr in (30.0, 20.0, 15.0):
        norms = [np.linalg.norm(cov_of(metadata["path"].iloc[i], snr, seed=int(i))) for i in sample]
        noise_report[f"snr_{int(snr)}db"] = {
            "p25": float(np.quantile(norms, 0.25)),
            "p50": float(np.median(norms)),
            "p75": float(np.quantile(norms, 0.75)),
        }
        print(f"噪声 SNR {snr:4.0f} dB -> 范数 p25/p50/p75 = "
              f"{noise_report[f'snr_{int(snr)}db']['p25']:.1f} / "
              f"{noise_report[f'snr_{int(snr)}db']['p50']:.1f} / "
              f"{noise_report[f'snr_{int(snr)}db']['p75']:.1f}", flush=True)
    clean_norms = [np.linalg.norm(cov_of(metadata["path"].iloc[i])) for i in sample]
    noise_report["clean"] = {"p50": float(np.median(clean_norms))}
    print(f"干净样本范数 p50 = {np.median(clean_norms):.1f}")

    # 2. 门控阈值（只用 Train 分位）
    low_threshold = float(np.quantile(train_norm, 0.005))
    high_threshold = float(np.quantile(train_norm, 0.995))
    train_gate = (train_norm < low_threshold) | (train_norm > high_threshold)
    test_gate = (test_norm < low_threshold) | (test_norm > high_threshold)
    print(f"\n门控阈值 [{low_threshold:.2f}, {high_threshold:.2f}]；Train 触发率 {train_gate.mean():.4f}")

    model = fit(train, subjects)
    test_scores = model.decision_function(test)
    test_conf = margin(test_scores)
    inner_conf = margin(model.decision_function(train[inner]))
    predicted = model.classes_[test_scores.argmax(1)]
    low_group = known & (test_norm < LOW_GROUP_NORM)
    high_group = known & (test_norm >= LOW_GROUP_NORM)

    variants = {}
    for name, quantile in (("gate_only", None), ("gate_margin0001", 0.001), ("gate_margin005", 0.05)):
        threshold = float(np.quantile(inner_conf, quantile)) if quantile is not None else -np.inf
        accepted = (test_conf >= threshold) & ~test_gate
        candidate = np.where(accepted, predicted, 0)
        variants[name] = {
            "margin_quantile": quantile,
            "margin_threshold": threshold,
            "nested": nested_evaluate(train, subjects, metadata, train_gate, quantile),
            "audit": {
                "s2_accept": float(accepted[known].mean()),
                "s2_high_group_accept": float(accepted[high_group].mean()),
                "s2_low_group_reject": float((~accepted[low_group]).mean()),
                "s1_reject": float((~accepted[~known]).mean()),
                "zero_count": int((candidate == 0).sum()),
                "zero_session1": int((candidate[~known] == 0).sum()),
                "zero_session2": int((candidate[known] == 0).sum()),
            },
        }
        print(f"{name:16s} margin_thr={threshold:8.3f} | nested C={variants[name]['nested']['C']:.4f} "
              f"F1={variants[name]['nested']['identity_f1']:.4f} U={variants[name]['nested']['U']:.4f} "
              f"AUROC={variants[name]['nested']['auroc']:.4f} | "
              f"S2 accept={variants[name]['audit']['s2_accept']:.3f} "
              f"(high {variants[name]['audit']['s2_high_group_accept']:.3f}) "
              f"S1 reject={variants[name]['audit']['s1_reject']:.3f}", flush=True)

    best_name = "gate_only"
    threshold = variants[best_name]["margin_threshold"]
    accepted = (test_conf >= threshold) & ~test_gate
    candidate = np.where(accepted, predicted, 0)
    pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
        OUT / "candidate_norm_gate.csv", index=False
    )

    report = {
        "noise_injection": noise_report,
        "gate": {"low": low_threshold, "high": high_threshold, "train_trigger": float(train_gate.mean()),
                 "test_trigger_session1": float(test_gate[~known].mean()),
                 "test_trigger_session2": float(test_gate[known].mean())},
        "variants": variants,
        "candidate": best_name,
    }
    (OUT / "e24a_norm_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
