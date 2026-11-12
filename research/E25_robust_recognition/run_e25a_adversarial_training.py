"""E25A：对抗增广训练，目标是「扰动后仍识别原身份」（RA），不是拒识。

攻击模型：在原始波形上加白噪声，SNR 30 dB / 20 dB。
SNR 30 dB 的选择依据是 E24A 的可观测事实：注入后特征范数 31.7，与 Test S2 低范数组（34.1）一致。
不使用 Test 标签，不使用 Session。

评估（Train 有标签，可真实计算）：
  clean 闭集准确率、SNR30 准确率、SNR20 准确率（=RA 代理）、拒识后的 ASafeR 代理
Test 只做无标签审计（低/高范数组的接受率与预测分布）。
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eeg_features import extract_features, load_eeg
from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
DATA = ROOT / "Brainprint_Competition"
FEATURE_CACHE = CACHE / "e25_robust"
OUT = Path(__file__).resolve().parent
COV_SLICE = slice(int(COV_COLUMNS[0]), int(COV_COLUMNS[-1]) + 1)
SNR_TRAIN = 30.0


def noisy_cov(payload) -> np.ndarray:
    path, snr, seed = payload
    data, _ = load_eeg(DATA / path)
    generator = np.random.default_rng(seed)
    noise_power = np.mean(data**2) / (10 ** (snr / 10))
    data = data + generator.normal(0.0, np.sqrt(noise_power), data.shape)
    features, _ = extract_features(data)
    return features[COV_SLICE]


def build_noisy_cache(metadata: pd.DataFrame) -> np.ndarray:
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURE_CACHE / f"train_cov_snr{int(SNR_TRAIN)}.npy"
    if cache_path.exists() and np.load(cache_path, mmap_mode="r").shape[0] == len(metadata):
        print(f"reuse {cache_path.name}", flush=True)
        return np.asarray(np.load(cache_path), dtype=np.float32)
    payloads = [(row.path, SNR_TRAIN, int(row.Index)) for row in metadata.itertuples()]
    output = np.empty((len(metadata), int(COV_COLUMNS.size)), dtype=np.float32)
    started = time.time()
    with ProcessPoolExecutor(max_workers=6) as pool:
        for index, features in enumerate(pool.map(noisy_cov, payloads, chunksize=16), 1):
            output[index - 1] = features
            if index % 2000 == 0 or index == len(payloads):
                print(f"noisy features {index}/{len(payloads)} | {time.time()-started:.0f}s", flush=True)
    np.save(cache_path, output)
    return output


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features, labels)
    return model


def nested(train_clean, train_noisy, subjects, metadata) -> dict:
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    groups = np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5)
    rows = []
    for fold, unknown in enumerate(groups, 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        is_known = np.isin(subjects, known_subjects)
        fit_mask = is_known & ~forward
        inner_conf = []
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_known = np.isin(subjects, enrolled)
            model = fit(train_clean[mask_known & ~forward & ~inner], subjects[mask_known & ~forward & ~inner])
            inner_conf.append(margin(model.decision_function(train_clean[np.flatnonzero(mask_known & inner)])))
        threshold = float(np.quantile(np.concatenate(inner_conf), 0.05))
        outer = np.flatnonzero(is_known & forward)
        labels = subjects[outer]

        result = {}
        for name, model in (
            ("clean", fit(train_clean[fit_mask], subjects[fit_mask])),
            ("noisy1", fit(np.concatenate([train_clean[fit_mask], train_noisy[fit_mask]]),
                           np.concatenate([subjects[fit_mask]] * 2))),
            ("noisy2", fit(np.concatenate([train_clean[fit_mask], train_noisy[fit_mask], train_noisy[fit_mask]]),
                           np.concatenate([subjects[fit_mask]] * 3))),
        ):
            clean_scores = model.decision_function(train_clean[outer])
            noisy_scores = model.decision_function(train_noisy[outer])
            classes = model.classes_
            clean_pred = classes[clean_scores.argmax(1)]
            noisy_pred = classes[noisy_scores.argmax(1)]
            clean_conf, noisy_conf = margin(clean_scores), margin(noisy_scores)
            result[name] = {
                "clean_acc": float(np.mean(clean_pred == labels)),
                "clean_accept": float(np.mean(clean_conf >= threshold)),
                "noisy_acc": float(np.mean(noisy_pred == labels)),
                "asafe": float(np.mean((noisy_pred == labels) | (noisy_conf < threshold))),
                "noisy_accept": float(np.mean(noisy_conf >= threshold)),
            }
        rows.append(result)
    summary = {
        config: {key: float(np.mean([row[config][key] for row in rows])) for key in rows[0][config]}
        for config in rows[0]
    }
    return summary


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial", "path"])
    subjects = metadata["subject"].to_numpy()
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session", "task"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_clean = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    train_noisy = build_noisy_cache(metadata)
    test = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    test_norm = np.linalg.norm(test, axis=1)
    low_group = known & (test_norm < 45.0)
    high_group = known & (test_norm >= 45.0)

    summary = nested(train_clean, train_noisy, subjects, metadata)
    for config, row in summary.items():
        print(f"{config:7s} clean_acc={row['clean_acc']:.4f} clean_accept={row['clean_accept']:.3f} | "
              f"noisy_acc(RA代理)={row['noisy_acc']:.4f} ASafeR代理={row['asafe']:.4f} "
              f"noisy_accept={row['noisy_accept']:.3f}", flush=True)

    # Test 审计：三种训练方式在低/高范数组上的分数
    audit = {}
    for name, features in (
        ("clean", train_clean),
        ("noisy1", np.concatenate([train_clean, train_noisy])),
        ("noisy2", np.concatenate([train_clean, train_noisy, train_noisy])),
    ):
        labels = np.concatenate([subjects] * (len(features) // len(train_clean)))
        model = fit(features, labels)
        inner_conf = margin(model.decision_function(train_clean[inner]))
        threshold = float(np.quantile(inner_conf, 0.05))
        scores = model.decision_function(test)
        conf = margin(scores)
        audit[name] = {
            "low_group_accept": float(np.mean(conf[low_group] >= threshold)),
            "high_group_accept": float(np.mean(conf[high_group] >= threshold)),
            "s1_accept": float(np.mean(conf[~known] >= threshold)),
            "low_group_margin_p50": float(np.median(conf[low_group])),
            "high_group_margin_p50": float(np.median(conf[high_group])),
        }
        print(f"{name:7s} test: low accept={audit[name]['low_group_accept']:.3f} "
              f"high accept={audit[name]['high_group_accept']:.3f} "
              f"S1 accept={audit[name]['s1_accept']:.3f}", flush=True)

    (OUT / "e25a_adversarial_training.json").write_text(
        json.dumps({"nested": summary, "test_audit": audit, "snr_train": SNR_TRAIN},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
