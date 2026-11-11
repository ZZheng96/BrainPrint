"""仅基于无标签 Test 波形近重复对估计差分扰动强度和频谱分布。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from src.eeg_features import COV_BANDS, FS, N_CHANNELS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"
OUTPUT = ROOT / "research" / "E08_raw_eeg_anchor" / "test_perturbation.json"
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
EPS = 1e-12


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}


def load_waveform(path: Path) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "fs", "task_label"])
    data = np.asarray(mat["data"], dtype=np.float32)
    if data.shape != (1200, N_CHANNELS):
        raise ValueError(f"{path}: expected (1200, {N_CHANNELS}), got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: EEG contains non-finite values")
    if int(np.asarray(mat["fs"]).reshape(-1)[0]) != FS:
        raise ValueError(f"{path}: unexpected sampling rate")
    return data


def select_pairs(table: pd.DataFrame) -> pd.DataFrame:
    required = {"sample", "path", "task", "nearest_sample", "similarity"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"waveform_pairs.csv 缺少列: {sorted(missing)}")
    metadata = table.set_index("sample", verify_integrity=True)
    selected = table.loc[table["similarity"] > 0.99].copy()
    selected = selected.loc[
        selected.apply(
            lambda row: row["task"] == metadata.at[row["nearest_sample"], "task"],
            axis=1,
        )
    ].copy()
    ordered = np.sort(selected[["sample", "nearest_sample"]].to_numpy(), axis=1)
    selected["left_sample"] = ordered[:, 0]
    selected["right_sample"] = ordered[:, 1]
    selected = selected.drop_duplicates(["left_sample", "right_sample"], keep="first")
    selected["left_path"] = selected["left_sample"].map(metadata["path"])
    selected["right_path"] = selected["right_sample"].map(metadata["path"])
    selected["task"] = selected["left_sample"].map(metadata["task"])
    if selected[["left_path", "right_path", "task"]].isna().any().any():
        raise ValueError("近重复对的 Test 元数据不完整")
    return selected.sort_values(["left_sample", "right_sample"]).reset_index(drop=True)


def batch_statistics(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    centered_a = a - a.mean(dim=1, keepdim=True)
    centered_b = b - b.mean(dim=1, keepdim=True)
    flat_a = a.reshape(len(a), -1).double()
    flat_b = b.reshape(len(b), -1).double()
    flat_ca = centered_a.reshape(len(a), -1).double()
    flat_cb = centered_b.reshape(len(b), -1).double()
    difference = flat_a - flat_b

    norm_a = flat_ca.norm(dim=1)
    norm_b = flat_cb.norm(dim=1)
    difference_norm = difference.norm(dim=1)
    nonzero = difference_norm > 0
    snr = 20.0 * torch.log10(
        torch.sqrt(norm_a * norm_b) / torch.where(nonzero, difference_norm, 1.0)
    )
    snr = torch.where(nonzero, snr, torch.full_like(snr, float("inf")))

    raw_denominator = flat_a.norm(dim=1) * flat_b.norm(dim=1)
    centered_denominator = norm_a * norm_b
    raw_correlation = (flat_a * flat_b).sum(dim=1) / raw_denominator.clamp_min(EPS)
    centered_correlation = (flat_ca * flat_cb).sum(dim=1) / centered_denominator.clamp_min(EPS)

    centered_difference = centered_a - centered_b
    spectrum = torch.fft.rfft(centered_difference, dim=1)
    frequencies = torch.fft.rfftfreq(a.shape[1], d=1.0 / FS, device=a.device)
    band_energy = []
    for low, high in COV_BANDS:
        # 相邻边界只计入低频带，避免共享频率重复计数。
        mask = (frequencies >= low) & (
            (frequencies <= high) if high == COV_BANDS[-1][1] else (frequencies < high)
        )
        power = spectrum[:, mask, :].abs().square().double()
        # 单边谱的非 DC/Nyquist 频率乘 2；本频带不含 Nyquist。
        power[:, 0, :] *= 0.5
        band_energy.append(2.0 * power.sum(dim=(1, 2)))
    band_energy = torch.stack(band_energy, dim=1)
    return {
        "snr_db": snr,
        "raw_correlation": raw_correlation,
        "demeaned_correlation": centered_correlation,
        "band_energy": band_energy,
        "difference_norm": difference_norm,
    }


def summarize(values: dict[str, np.ndarray]) -> dict:
    band_energy = values["band_energy"]
    total_energy = band_energy.sum(axis=1)
    nonzero = values["difference_norm"] > 0
    proportions = band_energy[nonzero] / total_energy[nonzero, None]
    band_names = [f"{low}-{high}" for low, high in COV_BANDS]
    dominant = np.argmax(band_energy[nonzero], axis=1)
    return {
        "n_pairs": int(len(values["snr_db"])),
        "n_exact_duplicate_pairs": int(np.sum(~nonzero)),
        "snr_db_quantiles_basis": "仅 ||a-b||>0 的对；完全相同对的理论 SNR 为 +inf，另报数量",
        "snr_db_quantiles": quantiles(values["snr_db"][nonzero]),
        "raw_correlation_quantiles": quantiles(values["raw_correlation"]),
        "demeaned_correlation_quantiles": quantiles(values["demeaned_correlation"]),
        "difference_band_energy": {
            "definition": "非零差分对的去均值 a-b rFFT 单边功率按 COV_BANDS 汇总；每对比例不含零差分对",
            "pooled_energy_fraction": {
                name: float(band_energy[:, index].sum() / total_energy.sum())
                for index, name in enumerate(band_names)
            },
            "per_pair_fraction_quantiles": {
                name: quantiles(proportions[:, index]) for index, name in enumerate(band_names)
            },
            "dominant_band_fraction_of_nonzero_pairs": {
                name: float(np.mean(dominant == index)) for index, name in enumerate(band_names)
            },
            "mean_per_pair_fraction": {
                name: float(proportions[:, index].mean()) for index, name in enumerate(band_names)
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size 必须为正数")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = select_pairs(pd.read_csv(CACHE / "waveform_pairs.csv"))
    if pairs.empty:
        raise ValueError("没有 similarity>0.99 的同任务近重复对")

    collected: dict[str, list[np.ndarray]] = {
        "snr_db": [],
        "raw_correlation": [],
        "demeaned_correlation": [],
        "band_energy": [],
        "difference_norm": [],
    }
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs.iloc[start:start + args.batch_size]
            left = np.stack([load_waveform(DATA / path) for path in batch["left_path"]])
            right = np.stack([load_waveform(DATA / path) for path in batch["right_path"]])
            stats = batch_statistics(
                torch.from_numpy(left).to(device), torch.from_numpy(right).to(device)
            )
            for name in collected:
                collected[name].append(stats[name].cpu().numpy())
            print(f"pairs: {min(start + len(batch), len(pairs))}/{len(pairs)}", flush=True)
    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    overall = summarize(values)

    task_summaries = {}
    for task in sorted(pairs["task"].unique()):
        indices = np.flatnonzero(pairs["task"].to_numpy() == task)
        task_summaries[task] = summarize({name: array[indices] for name, array in values.items()})

    snr = values["snr_db"]
    pooled = overall["difference_band_energy"]["pooled_energy_fraction"]
    leading_band = max(pooled, key=pooled.get)
    report = {
        "protocol": {
            "pair_source": "BrainPrint_cache/waveform_pairs.csv",
            "selection": "similarity > 0.99；校验同 task；无向对去重；不要求 reciprocal，不增加 gap 条件",
            "n_pairs": len(pairs),
            "n_unique_test_samples": int(len(set(pairs["left_sample"]) | set(pairs["right_sample"]))),
            "task_pair_counts": pairs["task"].value_counts().sort_index().to_dict(),
            "label_or_oracle_used": False,
            "device": str(device),
            "sampling_rate_hz": FS,
            "cov_bands_hz": [list(band) for band in COV_BANDS],
            "definitions": {
                "snr_db": "20*log10(sqrt(||centered_a||*||centered_b||)/||a-b||)；均值按通道扣除，范数展平全时程与通道",
                "raw_correlation": "sum(a*b)/(||a||*||b||)，未去均值的全波形余弦相关",
                "demeaned_correlation": "sum(centered_a*centered_b)/(||centered_a||*||centered_b||)，全波形 Pearson 相关",
            },
        },
        "overall": overall,
        "by_task": task_summaries,
        "internal_snr_descriptive_comparison": {
            "note": "仅比较 L2 相对强度；近重复对不一定是 clean/adv 真配对，不能据此识别攻击或验证防御效果",
            "fraction_snr_below_20_db": float(np.mean(snr < 20.0)),
            "fraction_snr_20_to_30_db": float(np.mean((snr >= 20.0) & (snr < 30.0))),
            "fraction_snr_at_least_30_db": float(np.mean(snr >= 30.0)),
            "median_observed_minus_internal20_db": float(np.median(snr) - 20.0),
            "median_observed_minus_internal30_db": float(np.median(snr) - 30.0),
        },
        "conclusion": {
            "leading_band_by_pooled_energy_hz": leading_band,
            "text": (
                f"可见近重复差分的 SNR 中位数为 {np.median(snr):.2f} dB；"
                f"六频带汇总能量以 {leading_band} Hz 为主。"
                "应将该分布作为内部 SNR20/30 攻击强度的描述性参照，而非真实攻击强度证明。"
            ),
        },
        "limitations": [
            "波形指纹经过逐通道标准化和降采样；余弦相似度>0.99只说明波形形状接近，不代表幅度、直流偏置或各频带能量一致。",
            "无标签近重复对可能包含重复录制、自然相似波形或其他非攻击差异；无法确定 clean/adv 身份、方向或因果来源。",
            "六频带 rFFT 能量是差分信号的频谱描述；相邻带共享频率只计一次，带外能量不纳入频带比例，且未复现 Butterworth 特征滤波器。",
            "本诊断不计算 AUC 或 F1，不使用测试标签，不修改正式提交，也不处理训练折传播审计。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_pairs": len(pairs),
        "snr_db_quantiles": overall["snr_db_quantiles"],
        "pooled_band_energy_fraction": pooled,
        "comparison": report["internal_snr_descriptive_comparison"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
