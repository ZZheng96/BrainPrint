"""E36E 最终冻结推理；默认仅允许 Train-only 冒烟，Test 需显式冻结候选。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_e36 import detrend_zband, gpu_logspd, load_train_matrix, sha256
from run_e36e_calibration import (
    TorchWCCNPrototype,
    distance_ratio,
    load_train_metadata,
    predict_scores,
)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
CALIBRATION_PATH = OUT / "calibration_results.json"
TEMPLATE_PATH = OUT / "frozen_candidate.template.json"
CACHE = ROOT / "BrainPrint_cache" / "e36_pure_eeg"
ZBAND_PATH = CACHE / "train_detrend_zband_cov5.npy"
F20_PATH = CACHE / "f20_known_oof_distance_ratio.npy"
TRAIN_METADATA_PATH = ROOT / "BrainPrint_cache" / "train_metadata.csv"
TEST_METADATA_PATH = ROOT / "BrainPrint_cache" / "test_metadata.csv"
EXPECTED_TEST_ROWS = 22251
TEST_METRICS = {
    metric: "N/A（无真实标签）" for metric in ("AUC", "F1", "C", "RA", "ASafeR", "U")
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(actual: str, expected: str, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} SHA256 与冻结候选不一致")


def load_candidate(path: Path, require_frozen: bool) -> dict:
    """Test 元数据加载之前完成全部冻结与来源校验。"""
    resolved = path.resolve()
    if require_frozen:
        if resolved == TEMPLATE_PATH.resolve() or resolved.name != "frozen_candidate.json":
            raise ValueError("--run-test 必须显式加载另一个名为 frozen_candidate.json 的文件")
        if candidate_is_not_frozen(path):
            raise ValueError("Test 冻结门禁未通过")
    candidate = read_json(resolved)
    expected = {
        "feature_variant": "detrend_zband_cov5",
        "score": "distance_ratio",
        "calibration": "known_oof_ecdf",
    }
    for key, value in expected.items():
        if candidate.get(key) != value:
            raise ValueError(f"冻结字段 {key} 必须为 {value}")
    shrinkage = float(candidate["wccn_shrinkage"])
    threshold = float(candidate["threshold"])
    if not 0.0 <= shrinkage <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("WCCN shrinkage 或 ECDF threshold 超出 [0,1]")
    if not isinstance(candidate.get("seed"), int):
        raise ValueError("seed 必须为整数")

    calibration = read_json(CALIBRATION_PATH)
    if calibration["status"] != "complete" or not calibration["gates"]["all_C_F1_AUROC_U"]:
        raise ValueError("E36E Train 校准结果不完整或训练门禁失败")
    if threshold != calibration["inner_threshold_selection"]["u"]["proxy_threshold"]:
        raise ValueError("threshold 不是内层 K8 pooled known-OOF ECDF 冻结阈值")
    if shrinkage != calibration["wccn_shrinkage"] or candidate["seed"] != calibration["seed"]:
        raise ValueError("WCCN shrinkage 或 seed 与 E36E 校准来源不一致")

    code_hashes = candidate["code_sha256"]
    for name, path_to_hash in (
        ("config.json", CONFIG_PATH),
        ("run_e36.py", OUT / "run_e36.py"),
        ("run_e36e_calibration.py", OUT / "run_e36e_calibration.py"),
        ("run_e36e_inference.py", Path(__file__).resolve()),
    ):
        require_hash(sha256(path_to_hash), code_hashes[name], name)
    cache_hashes = candidate["cache_sha256"]
    require_hash(sha256(TRAIN_METADATA_PATH), cache_hashes["train_metadata"], "Train metadata")
    require_hash(sha256(ZBAND_PATH), cache_hashes["train_detrend_zband_cov5"], "Train zband")
    require_hash(
        calibration["zband_cache_sha256"], cache_hashes["train_detrend_zband_cov5"],
        "校准来源的 Train zband",
    )
    require_hash(
        calibration["final_20class_continuous_block_oof"]["F20_reference_sha256"],
        cache_hashes["f20_known_oof_distance_ratio"], "校准来源的 F20 reference",
    )
    return candidate


def candidate_is_not_frozen(path: Path) -> bool:
    candidate = read_json(path)
    return candidate.get("frozen") is not True or candidate.get("train_gates_passed") is not True


def load_train_zband(config: dict, device: torch.device) -> tuple[pd.DataFrame, torch.Tensor]:
    metadata = load_train_metadata(config)
    cached = np.load(ZBAND_PATH, mmap_mode="r", allow_pickle=False)
    if cached.shape != (len(metadata), 855) or not np.isfinite(cached).all():
        raise ValueError("Train detrend/zband 缓存形状或数值无效")
    return metadata, torch.as_tensor(np.asarray(cached), dtype=torch.float32, device=device)


def fit_full_train_model(
    metadata: pd.DataFrame, zband: torch.Tensor, candidate: dict, config: dict,
) -> TorchWCCNPrototype:
    subjects = metadata["subject"].to_numpy(dtype=int)
    classes = np.unique(subjects)
    if not np.array_equal(classes, np.arange(1, 21)):
        raise ValueError("最终模型必须从全部 Train 拟合固定的 20 类")
    return TorchWCCNPrototype(
        zband, subjects, classes, float(candidate["wccn_shrinkage"]),
        float(config["models"]["wccn_eigenvalue_floor"]),
    )


def get_f20_reference(
    metadata: pd.DataFrame, zband: torch.Tensor, candidate: dict, config: dict,
    device: torch.device, save: bool,
) -> np.ndarray:
    """只用20类前80%连续块fit、后20%已知OOF产生F20；绝不接触Test。"""
    expected_hash = candidate["cache_sha256"]["f20_known_oof_distance_ratio"]
    if F20_PATH.exists():
        require_hash(sha256(F20_PATH), expected_hash, F20_PATH.name)
        reference = np.load(F20_PATH, allow_pickle=False)
    else:
        subjects = metadata["subject"].to_numpy(dtype=int)
        trial = metadata["trial"].to_numpy()
        maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
        oof_np = np.flatnonzero(trial > 0.8 * maximum)
        fit_np = np.flatnonzero(trial <= 0.8 * maximum)
        model = TorchWCCNPrototype(
            zband[torch.as_tensor(fit_np, device=device)], subjects[fit_np],
            np.arange(1, 21), float(candidate["wccn_shrinkage"]),
            float(config["models"]["wccn_eigenvalue_floor"]),
        )
        scores = predict_scores(
            model, zband, oof_np, device, int(config["models"]["prediction_batch_size"])
        )
        reference = np.sort(distance_ratio(scores)).astype(np.float64, copy=False)
        del model, scores
        actual_hash = __import__("hashlib").sha256(
            np.ascontiguousarray(reference, dtype=np.float64).tobytes()
        ).hexdigest()
        require_hash(actual_hash, expected_hash, "重算的 F20 reference")
        if save:
            np.save(F20_PATH, reference, allow_pickle=False)
            require_hash(sha256(F20_PATH), expected_hash, F20_PATH.name)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1 or len(reference) < 2 or not np.isfinite(reference).all():
        raise ValueError("F20 reference 无效")
    return np.sort(reference)


def decisions_from_scores(
    scores: np.ndarray, model: TorchWCCNPrototype, reference: np.ndarray, threshold: float,
) -> dict[str, np.ndarray]:
    """纯分数函数：无文件顺序、Sample、Session、批次统计或标签先验。"""
    ratio = distance_ratio(scores)
    u = np.searchsorted(reference, ratio, side="right") / len(reference)
    direct = model.classes[np.asarray(scores).argmax(axis=1)].astype(int)
    label = np.where(u < threshold, 0, direct).astype(int)
    return {"label": label, "direct": direct, "distance_ratio": ratio, "u": u}


def extract_single_trial(path: Path, config: dict, device: torch.device) -> torch.Tensor:
    """单个18×1200 Trial在GPU独立生成与E36完全同口径的五频带特征。"""
    array = load_train_matrix(path)
    raw = torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        features = detrend_zband(gpu_logspd(raw, config), config)
    if features.shape != (1, 855) or not torch.isfinite(features).all():
        raise ValueError(f"{path}: 单样本特征无效")
    return features


def predict_trial(
    path: Path, model: TorchWCCNPrototype, reference: np.ndarray, candidate: dict,
    config: dict, device: torch.device,
) -> dict[str, np.ndarray]:
    """生产推理唯一入口；参数中不存在顺序、Sample或Session。"""
    features = extract_single_trial(path, config, device)
    with torch.no_grad():
        scores = model(features).cpu().numpy()
    return decisions_from_scores(scores, model, reference, float(candidate["threshold"]))


def column(frame: pd.DataFrame, name: str) -> str:
    matches = [item for item in frame.columns if item.casefold() == name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"元数据必须且只能包含一个 {name} 字段")
    return matches[0]


def load_test_metadata_after_freeze() -> pd.DataFrame:
    """此函数只能由通过冻结校验的 run_test 调用；冒烟路径不调用。"""
    metadata = pd.read_csv(TEST_METADATA_PATH)
    if len(metadata) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Test 预测数必须为 {EXPECTED_TEST_ROWS}，实际 {len(metadata)}")
    sample = column(metadata, "Sample")
    path = column(metadata, "path")
    if metadata[sample].isna().any() or not metadata[sample].is_unique:
        raise ValueError("Sample 缺失或不唯一")
    if not metadata[path].astype(str).str.startswith("Test/").all():
        raise ValueError("Test 元数据含非 Test 路径")
    return metadata


def offline_session_audit(metadata: pd.DataFrame, labels: np.ndarray) -> dict:
    """仅在全部预测和CSV写出后调用；Session永不进入预测函数。"""
    session = metadata[column(metadata, "Session")].astype(str).str.strip().str.casefold()
    codes = session.map({
        "1": "S1", "s1": "S1", "session1": "S1",
        "2": "S2", "s2": "S2", "session2": "S2",
    })
    if codes.isna().any():
        raise ValueError("离线审计遇到未知 Session 取值")
    output = {"session_used_by_inference": False}
    for code in ("S1", "S2"):
        selected = codes.to_numpy() == code
        rate = float(np.mean(labels[selected] == 0)) if selected.any() else None
        output[f"{code}_n"] = int(selected.sum())
        output[f"{code}_rejection_rate"] = rate
    output["S2_rejection_le_0.02"] = bool(output["S2_rejection_rate"] <= 0.02)
    output["S1_rejection_ge_0.95"] = bool(output["S1_rejection_rate"] >= 0.95)
    return output


def run_test(candidate_path: Path, candidate: dict, config: dict, args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    metadata_train, zband = load_train_zband(config, device)
    reference = get_f20_reference(metadata_train, zband, candidate, config, device, save=True)
    model = fit_full_train_model(metadata_train, zband, candidate, config)

    # 只有显式冻结候选通过全部校验后，才到达此处的 Test 元数据读取。
    metadata = load_test_metadata_after_freeze()
    sample_col = column(metadata, "Sample")
    path_col = column(metadata, "path")
    test_root = ROOT / "Brainprint_Competition"
    labels = np.empty(len(metadata), dtype=np.int64)
    for index, relative in enumerate(metadata[path_col].astype(str)):
        result = predict_trial(
            test_root / relative, model, reference, candidate, config, device
        )
        labels[index] = int(result["label"][0])
        if (index + 1) % 1024 == 0 or index + 1 == len(metadata):
            print(f"单样本推理 {index + 1}/{len(metadata)}", flush=True)
    if not np.isin(labels, np.arange(21)).all():
        raise RuntimeError("预测标签超出 0–20")

    submission = pd.DataFrame({"Sample": metadata[sample_col].to_numpy(), "Label": labels})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)

    # 所有预测与CSV均完成后，Session才用于一次离线审计，不反馈模型或阈值。
    audit = {
        "frozen": True,
        "candidate": str(candidate_path.resolve()),
        "n_prediction_rows": int(len(submission)),
        "csv_has_header": True,
        "prediction_rule": "单样本 distance_ratio -> F20 right-continuous ECDF；u<threshold输出0，否则最高余弦身份",
        "test_metrics": TEST_METRICS,
        "test_metrics_unavailable_reason": "官方 Test 无真实标签；预测分布、配对或离线 Session 审计不能替代真值指标",
        "offline_session_audit_after_all_predictions": offline_session_audit(metadata, labels),
        "test_labels_used": False,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CSV：{args.output_csv}\n审计：{args.audit_json}", flush=True)


def concatenate_decisions(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([item[key] for item in items]) for key in items[0]}


def train_smoke(candidate: dict, config: dict) -> dict:
    """只读取Train；F20缺失时仅内存重算，避免冒烟产生额外文件。"""
    device = torch.device("cuda")
    metadata, zband = load_train_zband(config, device)
    reference = get_f20_reference(metadata, zband, candidate, config, device, save=False)
    model = fit_full_train_model(metadata, zband, candidate, config)
    rng = np.random.default_rng(int(candidate["seed"]))
    selected = np.sort(rng.choice(len(metadata), size=8, replace=False))
    rows = metadata.iloc[selected].copy()
    paths = [ROOT / config["data"]["train_root"] / path for path in rows["path"]]
    arrays = [load_train_matrix(path) for path in paths]
    raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
    with torch.no_grad():
        batch_features = detrend_zband(gpu_logspd(raw, config), config)
        single_features = torch.cat([
            detrend_zband(gpu_logspd(raw[index:index + 1], config), config)
            for index in range(len(paths))
        ])
        batch_scores = model(batch_features).cpu().numpy()
        single_scores = torch.cat([model(single_features[index:index + 1]) for index in range(len(paths))]).cpu().numpy()
    batch = decisions_from_scores(batch_scores, model, reference, float(candidate["threshold"]))
    singles = concatenate_decisions([
        predict_trial(path, model, reference, candidate, config, device) for path in paths
    ])

    permutation = rng.permutation(len(paths))
    inverse = np.argsort(permutation)
    with torch.no_grad():
        permuted_features = detrend_zband(gpu_logspd(raw[permutation], config), config)
        permuted_scores = model(permuted_features).cpu().numpy()
    permuted = decisions_from_scores(permuted_scores, model, reference, float(candidate["threshold"]))

    changed_session = rows.copy()
    changed_session[column(changed_session, "Session")] = "session_changed_not_used"
    changed_paths = [
        ROOT / config["data"]["train_root"] / path for path in changed_session["path"]
    ]
    changed = concatenate_decisions([
        predict_trial(path, model, reference, candidate, config, device) for path in changed_paths
    ])

    def max_difference(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))

    feature_single = max_difference(batch_features.cpu().numpy(), single_features.cpu().numpy())
    feature_permutation = max_difference(
        batch_features.cpu().numpy(), permuted_features.cpu().numpy()[inverse]
    )
    score_single = max_difference(batch_scores, single_scores)
    score_permutation = max_difference(batch_scores, permuted_scores[inverse])
    labels_equal = {
        "single_vs_batch": bool(np.array_equal(batch["label"], singles["label"])),
        "permutation": bool(np.array_equal(batch["label"], permuted["label"][inverse])),
        "session_field_modification": bool(np.array_equal(singles["label"], changed["label"])),
    }
    values_close = {
        "single_vs_batch": bool(np.allclose(batch_scores, single_scores, atol=1e-5, rtol=1e-5)),
        "permutation": bool(np.allclose(batch_scores, permuted_scores[inverse], atol=1e-5, rtol=1e-5)),
        "session_field_modification": bool(
            np.array_equal(singles["distance_ratio"], changed["distance_ratio"])
            and np.array_equal(singles["u"], changed["u"])
        ),
    }
    result = {
        "scope": "Train-only；未读取、枚举或推理Test",
        "n_trials": len(paths),
        "f20_reference_source": "existing cache" if F20_PATH.exists() else "recomputed in memory",
        "f20_reference_sha256": candidate["cache_sha256"]["f20_known_oof_distance_ratio"],
        "single_vs_batch_max_abs_feature": feature_single,
        "permutation_max_abs_feature": feature_permutation,
        "single_vs_batch_max_abs_score": score_single,
        "permutation_max_abs_score": score_permutation,
        "labels_equal": labels_equal,
        "scores_allclose": values_close,
        "batch_statistics_used": False,
        "passed": bool(
            all(labels_equal.values()) and all(values_close.values())
            and np.allclose(batch_features.cpu().numpy(), single_features.cpu().numpy(), atol=1e-5, rtol=1e-5)
            and np.allclose(batch_features.cpu().numpy(), permuted_features.cpu().numpy()[inverse], atol=1e-5, rtol=1e-5)
        ),
    }
    if not result["passed"]:
        raise RuntimeError("Train-only 冒烟失败：" + json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-train", action="store_true", help="仅Train单样本/批次/重排/Session不变性")
    mode.add_argument("--run-test", action="store_true", help="需显式已冻结候选；本次编写与冒烟不执行")
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=OUT / "e36e_predictions.csv")
    parser.add_argument("--audit-json", type=Path, default=OUT / "audit.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E36E 特征、WCCN 与逐 Trial 推理要求 CUDA GPU")
    if args.run_test and args.candidate is None:
        parser.error("--run-test 必须显式提供 --candidate .../frozen_candidate.json")
    candidate_path = args.candidate or TEMPLATE_PATH
    candidate = load_candidate(candidate_path, require_frozen=args.run_test)
    config = read_json(CONFIG_PATH)
    torch.manual_seed(int(candidate["seed"]))
    np.random.seed(int(candidate["seed"]))
    if args.smoke_train:
        print(json.dumps(train_smoke(candidate, config), ensure_ascii=False, indent=2), flush=True)
    else:
        run_test(candidate_path, candidate, config, args)


if __name__ == "__main__":
    main()
