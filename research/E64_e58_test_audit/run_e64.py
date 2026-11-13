"""E64：E58最佳方案的20类冻结迁移与一次性Test工程审计。

Train阶段只用<=60%连续块拟合20类新原型与logSPD/WCCN，60--70%校准，
70--80%做已知迁移sanity。冻结配置、阈值、代码和模型指纹后才读取Test
样本索引并推理；全部预测CSV写出后才读取Test session做离线审计。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))

from run_e36 import (  # noqa: E402
    detrend_zband,
    gpu_logspd,
    load_config as load_e36_config,
    load_train_matrix,
    sha256,
    train_metadata,
)
from run_e36e_calibration import TorchWCCNPrototype, distance_ratio  # noqa: E402
import run_e54 as E54  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
PREDICTIONS_PATH = OUT / "e64_predictions.csv"
CACHE = ROOT / "BrainPrint_cache" / "e64_e58_test_audit"
MODEL_PATH = CACHE / "frozen_model.pt"
MANIFEST_PATH = CACHE / "frozen_manifest.json"
E36_DIR = ROOT / "research" / "E36_pure_eeg"
E54_DIR = ROOT / "research" / "E54_outlier_exposure"
E54_CACHE = ROOT / "BrainPrint_cache" / "e54_outlier_exposure"
ZBAND_PATH = ROOT / "BrainPrint_cache" / "e36_pure_eeg" / "train_detrend_zband_cov5.npy"
CLASSES = np.arange(1, 21, dtype=int)
TEST_METRICS = {
    name: "N/A（无真实标签）"
    for name in ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")
}
E58_REFERENCE = {
    "protocol": "E58 Train inner: 5 outer groups x 2 inner runs; 8-class enrolled identities; clean known 60-80% and pseudo-unknown subjects; six white-box attacks macro",
    "source": "research/E58_oe_three_region/results.json and report.md; read-only reference, not rerun by E64",
    "C": 0.9714303861788618,
    "RA": 0.9370733824525745,
    "ASafeR": 0.975827828590786,
    "U": 0.9875,
    "Score": 96.7363424924926,
    "identity_macro_F1_direct": 0.9848762685722227,
    "identity_macro_F1_threshold_clean": 0.9821224405130554,
    "pseudo_unknown_AUROC_reported_in_E58_report": 0.9984,
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def source_paths(config: dict) -> dict[str, Path]:
    paths = {
        "e64_config": CONFIG_PATH,
        "e64_code": Path(__file__).resolve(),
        "e36_config": E36_DIR / "config.json",
        "e36_code": E36_DIR / "run_e36.py",
        "e36_calibration_code": E36_DIR / "run_e36e_calibration.py",
        "e54_config": E54_DIR / "config.json",
        "e54_code": E54_DIR / "run_e54.py",
        "train_metadata": ROOT / config["data"]["train_metadata"],
        "train_zband_cache": ZBAND_PATH,
    }
    for name in config["identity_migration"]["backbones"]:
        paths[f"e54_artifact:{name}"] = E54_CACHE / name
    return paths


def block_indices(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    result = {
        "fit": np.flatnonzero(trial <= 0.60 * maximum),
        "calibration": np.flatnonzero((trial > 0.60 * maximum) & (trial <= 0.70 * maximum)),
        "sanity": np.flatnonzero((trial > 0.70 * maximum) & (trial <= 0.80 * maximum)),
    }
    subjects = metadata["subject"].to_numpy(dtype=int)
    for name, indices in result.items():
        if len(indices) == 0 or not np.array_equal(np.unique(subjects[indices]), CLASSES):
            raise ValueError(f"{name}连续块未覆盖全部20名被试")
    if len(np.intersect1d(result["fit"], result["calibration"])) or len(
        np.intersect1d(result["fit"], result["sanity"])
    ) or len(np.intersect1d(result["calibration"], result["sanity"])):
        raise ValueError("fit/calibration/sanity连续块不互斥")
    return result


def load_source_backbone(name: str, e54_config: dict, device: torch.device) -> torch.nn.Module:
    path = E54_CACHE / name
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item["config_sha256"] != sha256(E54_DIR / "config.json"):
        raise ValueError(f"E54配置指纹变化：{path}")
    if item["info"]["variant"] != "oe" or len(item["classes"]) != 8:
        raise ValueError(f"E64只接受E54八类inner OE backbone：{path}")
    model = E54.EEGNetEmbedding(20, e54_config, task_adversarial=False).to(device)
    encoder_state = {
        key: value for key, value in item["model_state"].items()
        if not key.startswith(("prototypes", "radius_raw", "task_head."))
    }
    incompatible = model.load_state_dict(encoder_state, strict=False)
    if set(incompatible.missing_keys) != {"prototypes", "radius_raw"} or incompatible.unexpected_keys:
        raise ValueError(f"通用backbone迁移参数异常：{path}: {incompatible}")
    model.eval()
    model.requires_grad_(False)
    return model


def embed_indices(
    model: torch.nn.Module,
    raw: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = torch.as_tensor(
                indices[start:start + batch_size], dtype=torch.long, device=raw.device
            )
            chunks.append(model.embed(raw[selected]))
    return torch.cat(chunks)


def model_scores(
    model: torch.nn.Module,
    raw: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    output = np.empty((len(indices), 20), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = torch.as_tensor(
                indices[start:start + batch_size], dtype=torch.long, device=raw.device
            )
            scores = model.scores(raw[selected])
            output[start:start + len(selected)] = scores.cpu().numpy()
    return output


def logspd_scores(
    model: TorchWCCNPrototype,
    zband: torch.Tensor,
    indices: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    output = np.empty((len(indices), 20), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = torch.as_tensor(
                indices[start:start + batch_size], dtype=torch.long, device=zband.device
            )
            output[start:start + len(selected)] = model(zband[selected]).cpu().numpy()
    return output


def fit_twenty_class_migration(
    raw: torch.Tensor,
    zband: torch.Tensor,
    metadata: pd.DataFrame,
    blocks: dict[str, np.ndarray],
    config: dict,
    e54_config: dict,
    e36_config: dict,
) -> tuple[list[torch.nn.Module], TorchWCCNPrototype, dict[str, dict[str, np.ndarray]], list[dict]]:
    device = raw.device
    subjects = metadata["subject"].to_numpy(dtype=int)
    fit_np = blocks["fit"]
    fit_labels = torch.as_tensor(subjects[fit_np], dtype=torch.long, device=device)
    embedding_batch = int(config["identity_migration"]["embedding_batch_size"])
    eeg = {name: np.zeros((len(indices), 20), dtype=np.float64) for name, indices in blocks.items()}
    models: list[torch.nn.Module] = []
    backbone_rows = []
    expected_weights = np.full(10, 0.1, dtype=np.float64)
    if not np.array_equal(np.asarray(config["identity_migration"]["ensemble_weights"]), expected_weights):
        raise ValueError("E64十个backbone必须固定等权")

    for model_index, name in enumerate(config["identity_migration"]["backbones"], start=1):
        model = load_source_backbone(name, e54_config, device)
        fit_embedding = embed_indices(model, raw, fit_np, embedding_batch)
        means = torch.stack([
            fit_embedding[fit_labels == label].mean(dim=0) for label in CLASSES
        ])
        prototypes = torch.nn.functional.normalize(means, dim=1, eps=1e-12)
        if not torch.isfinite(prototypes).all():
            raise FloatingPointError(f"{name}的20类新原型含非有限值")
        model.prototypes.copy_(prototypes)
        model.eval()
        fit_score = (fit_embedding @ prototypes.T).cpu().numpy()
        eeg["fit"] += fit_score / 10.0
        for split in ("calibration", "sanity"):
            eeg[split] += model_scores(
                model, raw, blocks[split], embedding_batch
            ) / 10.0
        backbone_rows.append({
            "artifact": name,
            "artifact_sha256": sha256(E54_CACHE / name),
            "new_prototype_shape": list(prototypes.shape),
            "source_eight_class_prototypes_used": False,
            "encoder_frozen": True,
            "ensemble_weight": 0.1,
        })
        models.append(model)
        print(f"20类原型迁移 {model_index}/10: {name}", flush=True)
        del fit_embedding, fit_score, means, prototypes
        torch.cuda.empty_cache()

    fit_index = torch.as_tensor(fit_np, dtype=torch.long, device=device)
    logspd = TorchWCCNPrototype(
        zband[fit_index], subjects[fit_np], CLASSES,
        float(config["logspd_wccn"]["wccn_shrinkage"]),
        float(config["logspd_wccn"]["wccn_eigenvalue_floor"]),
    )
    log_scores = {
        name: logspd_scores(logspd, zband, indices).astype(np.float64)
        for name, indices in blocks.items()
    }
    return models, logspd, {"eeg": eeg, "logspd": log_scores}, backbone_rows


def decide(
    eeg_scores: np.ndarray,
    log_scores: np.ndarray,
    tau_low: float,
    tau_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dr = distance_ratio(log_scores)
    eeg_pred = CLASSES[np.asarray(eeg_scores).argmax(axis=1)]
    log_pred = CLASSES[np.asarray(log_scores).argmax(axis=1)]
    direct = np.where(dr >= tau_high, log_pred, eeg_pred).astype(int)
    pred = np.where(dr < tau_low, 0, direct).astype(int)
    return pred, direct, dr, log_pred


def known_metrics(
    true: np.ndarray,
    eeg_scores: np.ndarray,
    log_scores: np.ndarray,
    tau_low: float,
    tau_high: float,
) -> dict:
    pred, direct, dr, _ = decide(eeg_scores, log_scores, tau_low, tau_high)
    eeg_pred = CLASSES[np.asarray(eeg_scores).argmax(axis=1)]
    log_pred = CLASSES[np.asarray(log_scores).argmax(axis=1)]
    return {
        "N": int(len(true)),
        "direct_C": float(np.mean(direct == true)),
        "threshold_C": float(np.mean(pred == true)),
        "identity_20class_macro_F1_direct": float(f1_score(
            true, direct, labels=CLASSES, average="macro", zero_division=0
        )),
        "identity_20class_macro_F1_threshold": float(f1_score(
            true, pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "EEGNet_ensemble_only_C": float(np.mean(eeg_pred == true)),
        "logSPD_only_C": float(np.mean(log_pred == true)),
        "known_acceptance": float(np.mean(pred != 0)),
        "rejection_rate": float(np.mean(pred == 0)),
        "logSPD_route_rate": float(np.mean(dr >= tau_high)),
        "EEGNet_route_rate": float(np.mean((dr >= tau_low) & (dr < tau_high))),
        "AUC": "N/A（final20全20名Train被试均为已注册者，无pseudo-unknown）",
    }


def compute_raw_scores(
    raw: torch.Tensor,
    models: list[torch.nn.Module],
    logspd: TorchWCCNPrototype,
    e36_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        features = detrend_zband(gpu_logspd(raw, e36_config), e36_config)
        log_score = logspd(features).cpu().numpy().astype(np.float64)
        eeg_score = np.mean([
            model.scores(raw).cpu().numpy().astype(np.float64) for model in models
        ], axis=0)
    return eeg_score, log_score


def train_numerical_check(
    raw: torch.Tensor,
    metadata: pd.DataFrame,
    blocks: dict[str, np.ndarray],
    models: list[torch.nn.Module],
    logspd: TorchWCCNPrototype,
    e36_config: dict,
    config: dict,
    tau_low: float,
    tau_high: float,
) -> dict:
    selected = blocks["sanity"][:int(config["numerical_check"]["n_train_trials"])]
    index = torch.as_tensor(selected, dtype=torch.long, device=raw.device)
    batch_raw = raw[index]
    batch_eeg, batch_log = compute_raw_scores(batch_raw, models, logspd, e36_config)
    singles = [compute_raw_scores(batch_raw[i:i + 1], models, logspd, e36_config) for i in range(len(selected))]
    single_eeg = np.concatenate([item[0] for item in singles])
    single_log = np.concatenate([item[1] for item in singles])
    permutation = np.random.default_rng(int(config["seed"])).permutation(len(selected))
    inverse = np.argsort(permutation)
    perm_eeg, perm_log = compute_raw_scores(batch_raw[permutation], models, logspd, e36_config)
    batch_pred = decide(batch_eeg, batch_log, tau_low, tau_high)[0]
    single_pred = decide(single_eeg, single_log, tau_low, tau_high)[0]
    perm_pred = decide(perm_eeg, perm_log, tau_low, tau_high)[0][inverse]
    atol = float(config["numerical_check"]["score_atol"])
    rtol = float(config["numerical_check"]["score_rtol"])
    result = {
        "scope": "Train final20 sanity trials only; Test not accessed",
        "N": int(len(selected)),
        "single_vs_batch_EEGNet_max_abs": float(np.max(np.abs(batch_eeg - single_eeg))),
        "single_vs_batch_logSPD_max_abs": float(np.max(np.abs(batch_log - single_log))),
        "permutation_EEGNet_max_abs": float(np.max(np.abs(batch_eeg - perm_eeg[inverse]))),
        "permutation_logSPD_max_abs": float(np.max(np.abs(batch_log - perm_log[inverse]))),
        "single_vs_batch_scores_allclose": bool(
            np.allclose(batch_eeg, single_eeg, atol=atol, rtol=rtol)
            and np.allclose(batch_log, single_log, atol=atol, rtol=rtol)
        ),
        "permutation_scores_allclose": bool(
            np.allclose(batch_eeg, perm_eeg[inverse], atol=atol, rtol=rtol)
            and np.allclose(batch_log, perm_log[inverse], atol=atol, rtol=rtol)
        ),
        "single_vs_batch_labels_equal": bool(np.array_equal(batch_pred, single_pred)),
        "permutation_labels_equal": bool(np.array_equal(batch_pred, perm_pred)),
        "cross_sample_statistics_used": False,
    }
    result["passed"] = bool(all(result[key] for key in (
        "single_vs_batch_scores_allclose", "permutation_scores_allclose",
        "single_vs_batch_labels_equal", "permutation_labels_equal",
    )))
    if not result["passed"]:
        raise RuntimeError("Train单条/批次/重排一致性检查失败：" + json.dumps(result, ensure_ascii=False))
    return result


def freeze_models(
    models: list[torch.nn.Module],
    logspd: TorchWCCNPrototype,
    config: dict,
    tau_low: float,
    tau_high: float,
) -> dict:
    payload = {
        "format": "E64 generic frozen encoders + new 20-class prototypes + E36 WCCN",
        "classes": CLASSES,
        "backbones": [
            {
                "encoder_state": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                    if not key.startswith(("prototypes", "radius_raw", "task_head."))
                },
                "new_prototypes": model.prototypes.detach().cpu(),
            }
            for model in models
        ],
        "logspd": {
            "classes": logspd.classes,
            "whitening": logspd.whitening.detach().cpu(),
            "prototypes": logspd.prototypes.detach().cpu(),
        },
    }
    torch.save(payload, MODEL_PATH)
    hashes = {name: sha256(path) for name, path in source_paths(config).items()}
    manifest = {
        "frozen": True,
        "experiment": "E64",
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "threshold_quantiles": {"low": 0.02, "high": 0.05, "method": "higher"},
        "decision": config["decision"]["rule"],
        "ensemble_weights": config["identity_migration"]["ensemble_weights"],
        "model_sha256": sha256(MODEL_PATH),
        "source_sha256": hashes,
        "test_session_used_for_freeze": False,
        "train_gates_passed_claim": False,
        "authorization": config["authorization"],
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def verify_freeze(config: dict) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest.get("frozen") or manifest.get("experiment") != "E64":
        raise ValueError("E64冻结清单无效")
    if manifest["decision"] != config["decision"]["rule"] or not np.array_equal(
        np.asarray(manifest["ensemble_weights"], dtype=float),
        np.asarray(config["identity_migration"]["ensemble_weights"], dtype=float),
    ):
        raise ValueError("冻结决策或ensemble权重与配置不一致")
    for name, path in source_paths(config).items():
        actual = sha256(path)
        if actual != manifest["source_sha256"].get(name):
            raise ValueError(f"冻结指纹校验失败：{name} ({path})")
    if sha256(MODEL_PATH) != manifest["model_sha256"]:
        raise ValueError("E64迁移模型指纹校验失败")
    if not 0.0 < float(manifest["tau_low"]) <= float(manifest["tau_high"]):
        raise ValueError("冻结阈值顺序无效")
    return manifest


def load_frozen_models(
    config: dict,
    e54_config: dict,
    device: torch.device,
) -> tuple[list[torch.nn.Module], TorchWCCNPrototype]:
    item = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if not np.array_equal(np.asarray(item["classes"], dtype=int), CLASSES) or len(item["backbones"]) != 10:
        raise ValueError("冻结模型必须为十个20类backbone")
    models = []
    for backbone in item["backbones"]:
        model = E54.EEGNetEmbedding(20, e54_config, task_adversarial=False).to(device)
        incompatible = model.load_state_dict(backbone["encoder_state"], strict=False)
        if set(incompatible.missing_keys) != {"prototypes", "radius_raw"} or incompatible.unexpected_keys:
            raise ValueError(f"冻结backbone加载异常：{incompatible}")
        model.requires_grad_(False)
        with torch.no_grad():
            model.prototypes.copy_(backbone["new_prototypes"].to(device))
        model.eval()
        models.append(model)
    saved = item["logspd"]
    logspd = TorchWCCNPrototype.__new__(TorchWCCNPrototype)
    logspd.classes = np.asarray(saved["classes"], dtype=int)
    logspd.whitening = saved["whitening"].to(device)
    logspd.prototypes = saved["prototypes"].to(device)
    if not np.array_equal(logspd.classes, CLASSES):
        raise ValueError("冻结logSPD类别异常")
    return models, logspd


def run_train() -> dict:
    started = time.time()
    config = load_config()
    e36_config = load_e36_config()
    e54_config = E54.load_config()
    device = torch.device("cuda")
    metadata = train_metadata(e36_config)
    if len(metadata) != int(config["data"]["expected_train_trials"]):
        raise ValueError("Train行数与E64配置不一致")
    subjects = metadata["subject"].to_numpy(dtype=int)
    if not np.array_equal(np.unique(subjects), CLASSES):
        raise ValueError("Train必须包含固定1--20类")
    blocks = block_indices(metadata)
    raw = E54.load_all_raw(metadata, e54_config, device)
    cached = np.load(ZBAND_PATH, allow_pickle=False)
    if cached.shape != (len(metadata), 855) or not np.isfinite(cached).all():
        raise ValueError("E36 Train zband缓存无效")
    zband = torch.as_tensor(cached, dtype=torch.float32, device=device)
    models, logspd, scores, backbone_rows = fit_twenty_class_migration(
        raw, zband, metadata, blocks, config, e54_config, e36_config
    )
    calibration_dr = distance_ratio(scores["logspd"]["calibration"])
    tau_low = float(np.quantile(calibration_dr, 0.02, method="higher"))
    tau_high = float(np.quantile(calibration_dr, 0.05, method="higher"))
    if not tau_low <= tau_high:
        raise ValueError("clean known分位阈值顺序异常")
    metrics = {
        name: known_metrics(
            subjects[blocks[name]], scores["eeg"][name], scores["logspd"][name], tau_low, tau_high
        )
        for name in ("fit", "calibration", "sanity")
    }
    invariance = train_numerical_check(
        raw, metadata, blocks, models, logspd, e36_config, config, tau_low, tau_high
    )
    manifest = freeze_models(models, logspd, config, tau_low, tau_high)
    result = {
        "status": "train_frozen_test_pending",
        "experiment": "E64",
        "date": config["date"],
        "train_authorization_boundary": config["authorization"],
        "train_gates_passed": "not evaluated / not claimed; one-shot Test engineering audit explicitly authorized by user",
        "data_split": {
            name: {"N": int(len(indices)), "rule": config["continuous_blocks"][
                "final20_sanity" if name == "sanity" else name
            ]}
            for name, indices in blocks.items()
        },
        "twenty_class_migration": {
            "backbones": backbone_rows,
            "all_twenty_registered_OE_or_further_training": False,
            "score_aggregation": "fixed arithmetic mean of ten 20-class cosine score vectors",
            "logSPD_WCCN_fit": "all 20 subjects, <=60% only",
        },
        "calibration": {
            "source": "60-70% clean known logSPD distance_ratio only",
            "N": int(len(calibration_dr)),
            "tau_low": tau_low,
            "tau_high": tau_high,
            "quantile_method": "higher",
            "dr_quantiles": {
                key: float(value) for key, value in zip(
                    ("q00", "q02", "q05", "q50", "q95", "q100"),
                    np.quantile(calibration_dr, [0, 0.02, 0.05, 0.5, 0.95, 1], method="higher"),
                )
            },
        },
        "train_metrics": metrics,
        "train_AUC": "N/A：final20无pseudo-unknown；E58参考AUROC仅列于reference，不是E64新AUC",
        "train_numerical_invariance": invariance,
        "e58_inner_reference": E58_REFERENCE,
        "freeze": {
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": sha256(MANIFEST_PATH),
            "model": str(MODEL_PATH),
            "model_sha256": manifest["model_sha256"],
            "config_sha256": manifest["source_sha256"]["e64_config"],
            "code_sha256": manifest["source_sha256"]["e64_code"],
            "completed_before_any_test_index_or_waveform_read": True,
        },
        "test": {"status": "pending", "metrics": TEST_METRICS},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute": "Train raw/features/models and Test trial-local feature/score computation on CUDA; MAT I/O and metrics on CPU",
        },
        "train_elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result)
    print(
        f"E64 Train冻结：tau_low={tau_low:.6f} tau_high={tau_high:.6f} "
        f"sanity directC={metrics['sanity']['direct_C']:.4f} "
        f"thresholdC={metrics['sanity']['threshold_C']:.4f} "
        f"F1={metrics['sanity']['identity_20class_macro_F1_threshold']:.4f}",
        flush=True,
    )
    del raw, zband, models, logspd
    torch.cuda.empty_cache()
    return result


def load_test_index_after_freeze(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """冻结校验后只读Sample/sample/path；不读Label、Session、task或配对信息。"""
    sample_order = pd.read_csv(ROOT / config["data"]["test_csv"], usecols=["Sample"])
    paths = pd.read_csv(ROOT / config["data"]["test_metadata"], usecols=["sample", "path"])
    expected = int(config["data"]["expected_test_rows"])
    if len(sample_order) != expected or len(paths) != expected:
        raise ValueError(f"Test必须为{expected}行")
    if sample_order["Sample"].isna().any() or not sample_order["Sample"].is_unique:
        raise ValueError("Test Sample缺失或不唯一")
    if not np.array_equal(
        sample_order["Sample"].astype(str).to_numpy(), paths["sample"].astype(str).to_numpy()
    ):
        raise ValueError("test_metadata路径顺序未与test.csv Sample顺序对齐")
    if not paths["path"].astype(str).str.startswith("Test/").all():
        raise ValueError("Test索引包含非Test路径")
    return sample_order, paths


def offline_session_audit_after_csv(
    config: dict,
    sample_order: pd.DataFrame,
    labels: np.ndarray,
    dr: np.ndarray,
    tau_low: float,
    tau_high: float,
) -> dict:
    """只能在全部预测和CSV落盘后调用；Session绝不反馈到预测。"""
    sessions = pd.read_csv(
        ROOT / config["data"]["test_metadata"], usecols=["sample", "session"]
    )
    if not np.array_equal(
        sample_order["Sample"].astype(str).to_numpy(), sessions["sample"].astype(str).to_numpy()
    ):
        raise ValueError("离线Session审计与预测Sample顺序不一致")

    def session_code(value: object) -> str:
        text = str(value).strip().casefold().replace("_", "").replace("-", "")
        mapping = {"1": "S1", "s1": "S1", "session1": "S1", "2": "S2", "s2": "S2", "session2": "S2"}
        if text not in mapping:
            raise ValueError(f"未知Test Session取值：{value}")
        return mapping[text]

    codes = sessions["session"].map(session_code).to_numpy()
    output = {
        "N": int(len(labels)),
        "overall_rejection_rate": float(np.mean(labels == 0)),
        "session_used_by_inference": False,
        "audit_occurred_after_all_predictions_and_csv_write": True,
        "S1_rejection_target": 0.95,
        "S2_acceptance_target": 0.98,
    }
    for code in ("S1", "S2"):
        selected = codes == code
        if not selected.any():
            raise ValueError(f"Test离线审计缺少{code}")
        rejection = float(np.mean(labels[selected] == 0))
        output[f"{code}_N"] = int(selected.sum())
        output[f"{code}_rejection_rate"] = rejection
        output[f"{code}_acceptance_rate"] = float(1.0 - rejection)
    output["S1_rejection_ge_0.95"] = bool(output["S1_rejection_rate"] >= 0.95)
    output["S2_acceptance_ge_0.98"] = bool(output["S2_acceptance_rate"] >= 0.98)
    counts = {str(label): int(np.sum(labels == label)) for label in CLASSES if np.any(labels == label)}
    nonzero = int(np.sum(labels != 0))
    output["nonzero_N"] = nonzero
    output["nonzero_identity_distribution"] = {
        label: {
            "N": count,
            "fraction_of_all_Test": float(count / len(labels)),
            "fraction_of_nonzero": float(count / nonzero) if nonzero else 0.0,
        }
        for label, count in counts.items()
    }
    output["decision_route_counts"] = {
        "reject_dr_lt_tau_low": int(np.sum(dr < tau_low)),
        "EEGNet_dr_middle": int(np.sum((dr >= tau_low) & (dr < tau_high))),
        "logSPD_dr_ge_tau_high": int(np.sum(dr >= tau_high)),
    }
    output["audit_boundary"] = "S1/S2只描述预测拒识/接受，不是真实unknown/clean标签；不能据此计算官方指标或回调阈值"
    return output


def run_test() -> dict:
    started = time.time()
    config = load_config()
    manifest = verify_freeze(config)
    e36_config = load_e36_config()
    e54_config = E54.load_config()
    device = torch.device("cuda")
    models, logspd = load_frozen_models(config, e54_config, device)

    # 所有冻结与指纹校验已完成后，才读取Test的Sample与MAT路径。
    sample_order, paths = load_test_index_after_freeze(config)
    expected = len(sample_order)
    labels = np.empty(expected, dtype=np.int64)
    all_dr = np.empty(expected, dtype=np.float64)
    batch_size = int(config["identity_migration"]["inference_batch_size"])
    tau_low, tau_high = float(manifest["tau_low"]), float(manifest["tau_high"])
    test_root = ROOT / "Brainprint_Competition"
    with torch.no_grad():
        for start in range(0, expected, batch_size):
            stop = min(start + batch_size, expected)
            # 每个MAT仍逐条读取；堆叠仅加速完全trial-local的GPU计算。
            arrays = [
                load_train_matrix(test_root / str(relative))
                for relative in paths["path"].iloc[start:stop]
            ]
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
            eeg_scores, log_scores = compute_raw_scores(raw, models, logspd, e36_config)
            pred, _, dr, _ = decide(eeg_scores, log_scores, tau_low, tau_high)
            labels[start:stop] = pred
            all_dr[start:stop] = dr
            if stop % 1024 < batch_size or stop == expected:
                print(f"E64 Test独立推理 {stop}/{expected}", flush=True)
    if not np.isin(labels, np.arange(21)).all() or not np.isfinite(all_dr).all():
        raise RuntimeError("Test预测标签或distance_ratio无效")
    submission = pd.DataFrame({"Sample": sample_order["Sample"].to_numpy(), "Label": labels})
    submission.to_csv(PREDICTIONS_PATH, index=False)
    if len(pd.read_csv(PREDICTIONS_PATH)) != expected:
        raise RuntimeError("E64预测CSV行数校验失败")

    # 所有预测和CSV写出后，才首次读取Test session进行一次性离线审计。
    audit = offline_session_audit_after_csv(
        config, sample_order, labels, all_dr, tau_low, tau_high
    )
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if result["freeze"]["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("results.json与当前冻结模型不一致")
    result["status"] = "complete"
    result["test"] = {
        "status": "complete_one_shot_frozen_engineering_audit",
        "predictions_csv": str(PREDICTIONS_PATH),
        "prediction_rows": int(len(submission)),
        "csv_header": ["Sample", "Label"],
        "sample_order_source": config["data"]["test_csv"],
        "raw_input": "18x1200 MAT per trial",
        "single_trial_independent_features_and_scores": True,
        "gpu_batching_only_for_acceleration": True,
        "batch_zscore_ECDF_distribution_matching_threshold_adjustment_identity_quota": False,
        "session_file_order_task_count_pairing_used_for_decision": False,
        "test_labels_read_or_used": False,
        "metrics": TEST_METRICS,
        "metrics_unavailable_reason": "官方Test无真实标签；Session审计和预测分布均不能替代真值指标",
        "offline_session_audit_after_prediction": audit,
        "post_audit_threshold_or_variant_change": False,
        "inference_elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result)
    print(
        f"E64 Test审计：N={audit['N']} overallRej={audit['overall_rejection_rate']:.4f} "
        f"S1Rej={audit['S1_rejection_rate']:.4f} S2Accept={audit['S2_acceptance_rate']:.4f}",
        flush=True,
    )
    del models, logspd
    torch.cuda.empty_cache()
    return result


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(result: dict) -> None:
    train = result["train_metrics"]
    calibration = result["calibration"]
    lines = [
        "# E64：E58最佳方案20类冻结迁移与Test工程审计",
        "",
        "日期：2026-11-13。主方案严格为E54 OE-EEGNet通用backbone迁移 + E36 logSPD/WCCN三段式，不是E59。",
        "",
        "> 本次是用户明确授权的一次性冻结Test工程审计；**不声称Train全门禁已通过**。Test结果未用于调阈值、选变体或改集合划分。",
        "",
        "## 冻结方法与复现",
        "",
        "- 加载E54十个inner `oe` EEGNet并冻结encoder，丢弃原八类 prototypes/radii；每个encoder仅用全部20名Train被试的≤60%连续块embedding建立20个归一化均值原型。没有对20名已注册者同时做OE或继续训练。",
        "- 推理身份分数为十个20类cosine向量的固定等权算术平均。logSPD/WCCN也仅用20名被试≤60%块拟合。",
        "- 60–70% clean known dr独立校准：`tau_low=q0.02`、`tau_high=q0.05`，NumPy `method=higher`；70–80%仅作final20已知迁移sanity，>80%不使用。",
        f"- 冻结阈值：`tau_low={calibration['tau_low']:.9f}`，`tau_high={calibration['tau_high']:.9f}`。规则：dr≥tau_high→logSPD身份；中间区→EEGNet ensemble身份；dr<tau_low→0。",
        "- Test逐条读取原始18×1200 MAT；GPU batch仅为加速，所有预处理、logSPD和embedding均为trial-local，无批次z-score、ECDF、分布匹配、阈值调整、身份配额、Session/任务/数量/配对/文件顺序决策。Sample顺序只用于对齐路径和写CSV。",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e64.py'}\" --stage train",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e64.py'}\" --stage test",
        "# 或依次自动执行：",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e64.py'}\" --stage all",
        "```",
        "",
        f"配置：`{CONFIG_PATH}`；代码：`{OUT / 'run_e64.py'}`；模型：`{MODEL_PATH}`；冻结清单：`{MANIFEST_PATH}`。",
        f"冻结模型SHA256：`{result['freeze']['model_sha256']}`；E64代码SHA256：`{result['freeze']['code_sha256']}`。",
        "",
        "## E58 inner参考口径（只读参考，E64未重跑inner）",
        "",
        "E58为5 outer groups × 2 inner runs、每run八类注册身份、60–80% clean known与pseudo-unknown、六种白盒攻击宏均值；与E64 final20全20类连续块口径不同。",
        "",
        "| C | RA | ASafeR | U | Score | direct身份F1 | clean阈值身份F1 | pseudo-unknown AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(E58_REFERENCE['C'])} | {fmt(E58_REFERENCE['RA'])} | {fmt(E58_REFERENCE['ASafeR'])} | {fmt(E58_REFERENCE['U'])} | {E58_REFERENCE['Score']:.2f} | {fmt(E58_REFERENCE['identity_macro_F1_direct'])} | {fmt(E58_REFERENCE['identity_macro_F1_threshold_clean'])} | {fmt(E58_REFERENCE['pseudo_unknown_AUROC_reported_in_E58_report'])}（E58报告表列值） |",
        "",
        "## E64 final20 Train fit、校准与sanity",
        "",
        "direct C/F1保留三段路由但不拒识；threshold C/F1包含dr<tau_low拒识。F1固定为1–20类macro-F1，0不计作身份类。",
        "",
        "| Train块 | N | direct C | threshold C | direct身份F1 | threshold身份F1 | EEGNet-only C | logSPD-only C | 拒识率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, title in (("fit", "fit ≤60%"), ("calibration", "calibration 60–70%"), ("sanity", "final20 sanity 70–80%")):
        row = train[name]
        lines.append(
            f"| {title} | {row['N']} | {fmt(row['direct_C'])} | {fmt(row['threshold_C'])} | "
            f"{fmt(row['identity_20class_macro_F1_direct'])} | {fmt(row['identity_20class_macro_F1_threshold'])} | "
            f"{fmt(row['EEGNet_ensemble_only_C'])} | {fmt(row['logSPD_only_C'])} | {fmt(row['rejection_rate'])} |"
        )
    check = result["train_numerical_invariance"]
    lines.extend([
        "",
        f"- **Train final20 AUC：N/A**。全部20名Train被试均为注册者，本阶段没有pseudo-unknown；上表AUC不能由身份C/F1替代。E58参考AUROC不属于E64新评估。",
        f"- Train数值不变性：N={check['N']}，单条/批次与重排分数容差、标签一致均passed={check['passed']}；未使用跨样本统计。",
        "- Train fit只是拟合/容量诊断；60–70%用于阈值，70–80%为sanity，二者都不是真实跨Session证据。",
        "",
        "## Test冻结预测与Session离线审计",
        "",
    ])
    test = result.get("test", {})
    if test.get("status") == "pending":
        lines.append("Test尚未运行；AUC/F1/C/RA/ASafeR/U/Score均N/A（无真值）。")
    else:
        audit = test["offline_session_audit_after_prediction"]
        lines.extend([
            f"预测文件：`{PREDICTIONS_PATH}`，共{test['prediction_rows']}行，表头`Sample,Label`。全部预测和CSV写出后才读取Test Session。",
            "",
            "| 项目 | N | 拒识率 | 接受率 | 目标 | 审计判定 |",
            "|---|---:|---:|---:|---|---|",
            f"| 整体Test | {audit['N']} | {fmt(audit['overall_rejection_rate'])} | {fmt(1-audit['overall_rejection_rate'])} | — | 描述性 |",
            f"| S1 | {audit['S1_N']} | {fmt(audit['S1_rejection_rate'])} | {fmt(audit['S1_acceptance_rate'])} | 拒识≥95% | {'达到' if audit['S1_rejection_ge_0.95'] else '未达到'} |",
            f"| S2 | {audit['S2_N']} | {fmt(audit['S2_rejection_rate'])} | {fmt(audit['S2_acceptance_rate'])} | 接受≥98% | {'达到' if audit['S2_acceptance_ge_0.98'] else '未达到'} |",
            "",
            "非0身份分布（N；占全部Test比例）：",
            "",
            "| 身份 | N | 比例 |",
            "|---:|---:|---:|",
        ])
        for label, row in audit["nonzero_identity_distribution"].items():
            lines.append(f"| {label} | {row['N']} | {fmt(row['fraction_of_all_Test'])} |")
        routes = audit["decision_route_counts"]
        lines.extend([
            "",
            f"决策路由计数：拒识={routes['reject_dr_lt_tau_low']}，EEGNet中间区={routes['EEGNet_dr_middle']}，logSPD高dr区={routes['logSPD_dr_ge_tau_high']}。",
            "",
            "**Test AUC、F1、C、RA、ASafeR、U、Score：全部N/A（无真实标签）。** S1拒识/S2接受只是对预测0/非0的工程审计，不等于U或clean C；非0身份分布也不提供身份正确性。即使目标未达到，本实验也不会据Test回头调整。",
        ])
    lines.extend([
        "",
        "## 证据边界",
        "",
        "- E64只使用Train Session1连续块拟合/校准/sanity；不是跨天、跨Session或官方新被试真值评估。E58 inner数值仅是参考，不能外推为Test得分。",
        "- 十个backbone来自E54各自八类inner OE训练；E64迁移阶段没有全20类OE。新原型与WCCN严格使用≤60%块。",
        "- Test无真值，因此不报告或暗示官方AUC/F1/C/RA/ASafeR/U/Score。Session审计完成后未修改阈值、模型、代码路线或预测。",
        "- 本次审计获得用户对Train门禁未全通过状态的明确豁免，仅授权工程运行，不构成门禁通过声明。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E64特征、模型迁移与推理要求CUDA GPU")
    config = load_config()
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    if args.stage in ("train", "all"):
        run_train()
    if args.stage in ("test", "all"):
        run_test()
    print(f"结果：{RESULTS_PATH}\n报告：{REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
