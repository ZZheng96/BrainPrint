"""E72: Train-only correctness audit and nested query/template verification.

Formal results use only the two pure-EEG feature caches. The legacy DINO cache is
restricted to smoke. No Test data, Test index, or Test labels are read.
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

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
import run_e36 as E36  # noqa: E402
import run_e65 as E65  # noqa: E402
from src.eeg_image_dinov2 import image_encoders  # noqa: E402
from pair_protocol import (  # noqa: E402
    calibrate_q05, confidence_values, continuous_masks, evaluate_queries,
    fit_spaces, inner_splits, load_train_metadata, make_template_pairs,
    mean_metrics, outer_groups, pooled_metrics, predict_spaces, sha256,
    std_metrics, subject_roles,
)

CONFIG_PATH = OUT / "config.json"
MANIFEST_PATH = OUT / "input_manifest.json"
RESULTS_PATH = OUT / "results.json"
SMOKE_RESULTS_PATH = OUT / "smoke_results.json"
REPORT_PATH = OUT / "report.md"
MODEL_PATH = OUT / "outer_templates.pt"
TEST_METRICS = {
    key: "N/A（无公开真值，且未读取 Test 数据、索引或标签）"
    for key in ("AUC", "F1", "C", "RA", "ASafeR", "A", "U", "Score")
}


def write_json(path: Path, value: dict) -> None:
    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def invocation(stage: str) -> dict:
    return {
        "command": f".venv/Scripts/python.exe research/E72_pair_protocol/run_e72.py --stage {stage}",
        "executable": sys.executable,
        "argv": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
    }


def finite_on_gpu(array: np.ndarray, device: torch.device, batch_size: int) -> bool:
    for start in range(0, len(array), batch_size):
        chunk = torch.as_tensor(
            np.array(array[start:start + batch_size], dtype=np.float32, copy=True), device=device
        )
        if not bool(torch.isfinite(chunk).all().item()):
            return False
    return True


def audit_formal_features(metadata: pd.DataFrame, config: dict,
                          device: torch.device) -> tuple[dict, torch.Tensor, np.ndarray]:
    started = time.time()
    e36_config = E36.load_config()
    e65_config = E65.load_config()
    metadata_path = ROOT / config["data"]["train_metadata"]
    metadata_hash = sha256(metadata_path)
    e65_old_manifest = json.loads(E65.FEATURE_MANIFEST.read_text(encoding="utf-8"))
    arrays = {
        name: np.load(ROOT / relative, mmap_mode="r")
        for name, relative in config["features"]["formal"].items()
    }
    expected_shape = tuple(config["features"]["expected_formal_shape"])
    if expected_shape != (len(metadata), 855):
        raise ValueError("E72 expected feature shape does not match Train metadata")
    rng = np.random.default_rng(int(config["features"]["audit_seed"]))
    selected = np.sort(rng.choice(
        len(metadata), int(config["features"]["audit_rebuild_samples"]), replace=False
    ))
    paths = [ROOT / config["data"]["train_root"] / path for path in metadata.iloc[selected]["path"]]
    raw = torch.as_tensor(
        np.stack([E36.load_train_matrix(path) for path in paths]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        rebuilt = {
            "e36_covariance": E36.detrend_zband(E36.gpu_logspd(raw, e36_config), e36_config),
            "e65_correlation": E65.gpu_correlation_logspd(raw, e36_config, e65_config),
        }
    cache_reports = {}
    for name, array in arrays.items():
        actual = torch.as_tensor(np.asarray(array[selected]), device=device)
        difference = (actual - rebuilt[name]).abs()
        finite = finite_on_gpu(array, device, int(config["features"]["finite_check_batch_size"]))
        cache_hash = sha256(ROOT / config["features"]["formal"][name])
        allclose = bool(torch.allclose(
            actual, rebuilt[name],
            atol=float(config["features"]["audit_atol"]),
            rtol=float(config["features"]["audit_rtol"]),
        ))
        cache_reports[name] = {
            "path": config["features"]["formal"][name],
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": finite,
            "sha256": cache_hash,
            "metadata_sha256": metadata_hash,
            "rebuild_sample_indices": selected.tolist(),
            "rebuild_sample_train_paths": metadata.iloc[selected]["path"].tolist(),
            "cache_vs_rebuilt_max_abs": float(difference.max().item()),
            "row_alignment_allclose": allclose,
            "single_trial_rebuild_on_gpu": True,
            "passed": bool(
                array.shape == expected_shape
                and str(array.dtype) == config["features"]["expected_dtype"]
                and finite and allclose
            ),
        }
        if name == "e65_correlation":
            cache_reports[name]["sha256_matches_e65_manifest"] = bool(
                cache_hash == e65_old_manifest["cache_sha256"]
            )
            cache_reports[name]["metadata_hash_matches_e65_manifest"] = bool(
                metadata_hash == e65_old_manifest["metadata_sha256"]
            )
            cache_reports[name]["passed"] = bool(
                cache_reports[name]["passed"]
                and cache_reports[name]["sha256_matches_e65_manifest"]
                and cache_reports[name]["metadata_hash_matches_e65_manifest"]
            )
    report = {
        "formal_features": cache_reports,
        "all_formal_cache_checks_passed": all(row["passed"] for row in cache_reports.values()),
        "metadata": {
            "path": config["data"]["train_metadata"],
            "sha256": metadata_hash,
            "rows": len(metadata),
            "subjects": int(metadata["subject"].nunique()),
            "sessions": sorted(metadata["session"].unique().tolist()),
            "tasks": sorted(metadata["task"].unique().tolist()),
            "all_paths_train_only": bool(metadata["path"].str.startswith("Train/").all()),
        },
        "source_sha256": {
            "e36_run": sha256(ROOT / "research" / "E36_pure_eeg" / "run_e36.py"),
            "e36_config": sha256(ROOT / "research" / "E36_pure_eeg" / "config.json"),
            "e65_run": sha256(ROOT / "research" / "E65_correlation_spd" / "run_e65.py"),
            "e65_config": sha256(ROOT / "research" / "E65_correlation_spd" / "config.json"),
        },
        "elapsed_seconds": time.time() - started,
    }
    if not report["all_formal_cache_checks_passed"]:
        raise RuntimeError(f"Formal cache audit failed: {cache_reports}")
    return report, raw, selected


def audit_images_and_legacy_dino(raw: torch.Tensor, selected: np.ndarray,
                                 config: dict, device: torch.device) -> dict:
    method = config["image_audit"]["method"]
    images = []
    for index in range(int(config["image_audit"]["samples"])):
        data = raw[index].detach().cpu().numpy().T
        images.append(image_encoders.encode(data, method))
    shape_ok = all(list(image.shape) == config["image_audit"]["required_shape"] for image in images)
    dtype_ok = all(str(image.dtype) == config["image_audit"]["required_dtype"] for image in images)
    finite = all(bool(np.isfinite(image).all()) for image in images)
    minimum = float(min(image.min() for image in images))
    maximum = float(max(image.max() for image in images))
    low, high = config["image_audit"]["required_range"]
    direct_encode_passed = bool(shape_ok and dtype_ok and finite and minimum >= low and maximum <= high)
    bands = [list(map(float, band)) for band in image_encoders.BANDS]
    official = config["image_audit"]["official_band_hz"]
    band_violations = [band for band in bands if band[0] < official[0] or band[1] > official[1]]
    legacy_image_path = ROOT / config["features"]["legacy_image_cache"]
    legacy_images = np.load(legacy_image_path, mmap_mode="r")
    legacy_samples = np.asarray(legacy_images[selected[:len(images)]])
    legacy_dino_report = {}
    for name, relative in config["features"]["smoke_only"].items():
        path = ROOT / relative
        array = np.load(path, mmap_mode="r")
        legacy_dino_report[name] = {
            "path": relative,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": finite_on_gpu(array, device, int(config["features"]["finite_check_batch_size"])),
            "sha256": sha256(path),
            "formal_use": False,
            "row_alignment_rebuilt": False,
            "restriction": config["image_audit"]["legacy_dino_formal_use"],
        }
    return {
        "direct_encode": {
            "method": method,
            "samples": len(images),
            "shape": list(images[0].shape),
            "dtype": str(images[0].dtype),
            "finite": finite,
            "min": minimum,
            "max": maximum,
            "passed": direct_encode_passed,
        },
        "encoder_bands_hz": bands,
        "official_band_hz": official,
        "out_of_official_band_definitions": band_violations,
        "all_bands_within_official_1_40hz": not band_violations,
        "legacy_uint8_image_cache": {
            "path": config["features"]["legacy_image_cache"],
            "shape": list(legacy_images.shape),
            "dtype": str(legacy_images.dtype),
            "sample_min": int(legacy_samples.min()),
            "sample_max": int(legacy_samples.max()),
            "not_directly_encoder_compatible": True,
            "historical_extraction_scaling": "E20 run_e20a.py converts uint8 to float32 and divides by 255 before DINO; old caches were not modified",
        },
        "legacy_dino_features": legacy_dino_report,
        "formal_image_conclusions_allowed": False,
        "reason": "45-90 Hz definitions exceed the official 1-40 Hz preprocessing band; legacy DINO lineage was not rebuilt under E72",
    }


def run_audit(metadata: pd.DataFrame, config: dict, device: torch.device,
              stage: str) -> dict:
    formal, raw, selected = audit_formal_features(metadata, config, device)
    manifest = {
        "experiment": config["experiment"],
        "invocation": invocation(stage),
        "config_sha256": sha256(CONFIG_PATH),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        **formal,
        "image_audit": audit_images_and_legacy_dino(raw, selected, config, device),
        "data_access": {
            "train_metadata_read": config["data"]["train_metadata"],
            "train_waveforms_read": formal["formal_features"]["e36_covariance"]["rebuild_sample_train_paths"],
            "test_accessed": False,
            "test_data_index_or_labels_read": False,
            "legacy_test_cache_read": False,
        },
    }
    if MANIFEST_PATH.exists():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        history = previous.get("audit_history", [])
        if previous.get("invocation"):
            history.append(previous["invocation"])
    else:
        history = []
    history.append(manifest["invocation"])
    manifest["audit_history"] = history[-10:]
    write_json(MANIFEST_PATH, manifest)
    print(
        f"输入审计通过：两个正式缓存行对齐；图像直接编码 passed="
        f"{manifest['image_audit']['direct_encode']['passed']}，频带合规="
        f"{manifest['image_audit']['all_bands_within_official_1_40hz']}", flush=True,
    )
    return manifest


def load_features(config: dict, device: torch.device, smoke: bool) -> dict[str, torch.Tensor]:
    paths = dict(config["features"]["formal"])
    if smoke:
        paths.update(config["features"]["smoke_only"])
    features = {}
    for name, relative in paths.items():
        array = np.load(ROOT / relative)
        features[name] = torch.as_tensor(array, dtype=torch.float32, device=device)
        print(f"GPU features {name}: {tuple(features[name].shape)}", flush=True)
    return features


def candidate_id(feature: str, spec: dict) -> str:
    return f"{feature}/{spec['space']}/{spec['score']}"


def model_state(model) -> dict:
    return {
        "classes": np.asarray(model.classes, dtype=int),
        "prototypes": model.prototypes.detach().cpu(),
        "whitening": None if model.whitening is None else model.whitening.detach().cpu(),
    }


def assert_inner_isolation(subjects: np.ndarray, fit: np.ndarray, known: np.ndarray,
                           unknown: np.ndarray, roles: dict) -> dict:
    checks = {
        "fit_known_indices_disjoint": not np.intersect1d(fit, known).size,
        "fit_excludes_all_unknown_subjects": not np.isin(
            subjects[fit], np.concatenate([roles["outer_unknown"], roles["calibration_unknown"]])
        ).any(),
        "known_excludes_all_unknown_subjects": not np.isin(
            subjects[known], np.concatenate([roles["outer_unknown"], roles["calibration_unknown"]])
        ).any(),
        "unknown_is_exactly_calibration_subjects": set(subjects[unknown].tolist()) == set(
            roles["calibration_unknown"].tolist()
        ),
        "outer_unknown_absent_from_every_inner_role": not np.isin(
            subjects[np.concatenate([fit, known, unknown])], roles["outer_unknown"]
        ).any(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Inner isolation failed: {checks}")
    return checks


def run_protocol(features: dict[str, torch.Tensor], metadata: pd.DataFrame,
                 config: dict, manifest: dict, stage: str, smoke: bool) -> dict:
    started = time.time()
    subjects = metadata["subject"].to_numpy(dtype=int)
    masks = continuous_masks(metadata)
    all_subjects = np.unique(subjects)
    groups = outer_groups(all_subjects, config)
    if smoke:
        groups = groups[:1]
    specs = config["protocol"]["methods"]
    candidate_ids = [candidate_id(feature, spec) for feature in features for spec in specs]
    result = {
        "status": "running_smoke" if smoke else "running",
        "experiment": config["experiment"],
        "mode": "smoke" if smoke else "full_5_outer_x_2_inner",
        "formal_conclusions_allowed": not smoke,
        "invocation": invocation(stage),
        "config_sha256": sha256(CONFIG_PATH),
        "input_manifest_sha256": sha256(MANIFEST_PATH),
        "feature_roles": {
            name: ("smoke_only_legacy" if name in config["features"]["smoke_only"] else "formal_pure_eeg")
            for name in features
        },
        "protocol": config["protocol"],
        "folds": [],
        "data_access": manifest["data_access"],
        "score_availability": {
            "available": ["C", "U"],
            "unavailable": ["RA", "ASafeR", "A"],
            "official_Score": "N/A: official Score requires C, A and U; no attacks were run",
            "C_U_proxy_not_official": "100*C^0.40*U^0.25",
        },
        "test": TEST_METRICS,
    }
    outer_records = {name: [] for name in candidate_ids}
    saved_models = {}
    batch_size = int(config["features"]["prediction_batch_size"])
    for outer_index, outer_unknown in enumerate(groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, outer_unknown)
        splits = inner_splits(known_subjects, outer_fold, config)
        if smoke:
            splits = splits[:1]
        inner_runs = []
        for inner_index, inner_unknown in enumerate(splits):
            inner_fold = inner_index + 1
            roles = subject_roles(all_subjects, outer_unknown, inner_unknown)
            enrolled = roles["enrolled"]
            enrolled_mask = np.isin(subjects, enrolled)
            fit_np = np.flatnonzero(enrolled_mask & masks["inner_fit"])
            known_np = np.flatnonzero(enrolled_mask & masks["inner_known"])
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & masks["inner_unknown"])
            run = {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "outer_unknown_subjects": outer_unknown.tolist(),
                "registered_subjects": enrolled.tolist(),
                "calibration_unknown_subjects": inner_unknown.tolist(),
                "N_fit": int(len(fit_np)),
                "N_known": int(len(known_np)),
                "N_unknown": int(len(unknown_np)),
                "isolation": assert_inner_isolation(subjects, fit_np, known_np, unknown_np, roles),
                "fit_accuracy": {},
                "features": {},
            }
            for feature, x in features.items():
                fit_index = torch.as_tensor(fit_np, dtype=torch.long, device=x.device)
                models = fit_spaces(x[fit_index], subjects[fit_np], enrolled, config)
                eval_np = np.concatenate([known_np, unknown_np])
                scores = predict_spaces(models, x, eval_np, batch_size)
                fit_scores = predict_spaces(models, x, fit_np, batch_size)
                run["fit_accuracy"][feature] = {
                    space: float(np.mean(enrolled[values.argmax(axis=1)] == subjects[fit_np]))
                    for space, values in fit_scores.items()
                }
                run["features"][feature] = {
                    "known_scores": {space: value[:len(known_np)] for space, value in scores.items()},
                    "unknown_scores": {space: value[len(known_np):] for space, value in scores.items()},
                    "known_true": subjects[known_np],
                    "classes": enrolled,
                }
                if smoke and outer_fold == 1 and inner_fold == 1 and feature == next(iter(features)):
                    pairs = make_template_pairs(scores["cosine"][:len(known_np)], enrolled, subjects[known_np])
                    if int(pairs["same"].sum()) != len(known_np):
                        raise RuntimeError("Each known query must have exactly one same template pair")
                    result["smoke_pair_audit"] = {
                        "feature": feature,
                        "N_pairs": int(len(pairs["same"])),
                        "N_same": int(pairs["same"].sum()),
                        "N_different": int((1 - pairs["same"]).sum()),
                        "one_same_template_per_known_query": True,
                        "trial_by_trial_N_squared_materialized": False,
                    }
                del models, scores, fit_scores
            inner_runs.append(run)
            print(
                f"outer={outer_fold} inner={inner_fold}: fit={len(fit_np)} "
                f"known={len(known_np)} calibration_unknown={len(unknown_np)}", flush=True,
            )
        outer_fit_np = np.flatnonzero(np.isin(subjects, known_subjects) & masks["outer_fit"])
        outer_known_np = np.flatnonzero(np.isin(subjects, known_subjects) & masks["outer_known"])
        outer_unknown_np = np.flatnonzero(np.isin(subjects, outer_unknown))
        outer_isolation = {
            "fit_and_continuous_query_disjoint": not np.intersect1d(outer_fit_np, outer_known_np).size,
            "outer_unknown_exact": set(subjects[outer_unknown_np].tolist()) == set(outer_unknown.tolist()),
            "outer_unknown_excluded_from_fit": not np.isin(subjects[outer_fit_np], outer_unknown).any(),
            "outer_unknown_excluded_from_known_query": not np.isin(subjects[outer_known_np], outer_unknown).any(),
            "outer_unknown_all_trials": len(outer_unknown_np) == int(np.isin(subjects, outer_unknown).sum()),
        }
        if not all(outer_isolation.values()):
            raise RuntimeError(f"Outer isolation failed: {outer_isolation}")
        fold = {
            "outer_fold": outer_fold,
            "outer_unknown_subjects": outer_unknown.tolist(),
            "registered_subjects": known_subjects.tolist(),
            "inner_roles": [
                {key: value for key, value in run.items() if key != "features"}
                for run in inner_runs
            ],
            "N_outer_fit": int(len(outer_fit_np)),
            "N_outer_known_continuous": int(len(outer_known_np)),
            "N_outer_unknown": int(len(outer_unknown_np)),
            "outer_isolation": outer_isolation,
            "outer_fit_accuracy": {},
            "candidates": {},
        }
        for feature, x in features.items():
            fit_index = torch.as_tensor(outer_fit_np, dtype=torch.long, device=x.device)
            models = fit_spaces(x[fit_index], subjects[outer_fit_np], known_subjects, config)
            eval_np = np.concatenate([outer_known_np, outer_unknown_np])
            scores = predict_spaces(models, x, eval_np, batch_size)
            fit_scores = predict_spaces(models, x, outer_fit_np, batch_size)
            fold["outer_fit_accuracy"][feature] = {
                space: float(np.mean(known_subjects[values.argmax(axis=1)] == subjects[outer_fit_np]))
                for space, values in fit_scores.items()
            }
            if not smoke:
                for space, model in models.items():
                    saved_models[f"outer_{outer_fold}/{feature}/{space}"] = model_state(model)
            for spec in specs:
                name = candidate_id(feature, spec)
                feature_inner_runs = [run["features"][feature] for run in inner_runs]
                threshold = calibrate_q05(feature_inner_runs, spec["space"], spec["score"])
                inner_metrics = []
                for run in feature_inner_runs:
                    metrics, _ = evaluate_queries(
                        run["known_true"], run["known_scores"][spec["space"]],
                        run["unknown_scores"][spec["space"]], run["classes"],
                        spec["score"], threshold,
                    )
                    inner_metrics.append(metrics)
                metrics, records = evaluate_queries(
                    subjects[outer_known_np], scores[spec["space"]][:len(outer_known_np)],
                    scores[spec["space"]][len(outer_known_np):], known_subjects,
                    spec["score"], threshold,
                )
                fold["candidates"][name] = {
                    "threshold": threshold,
                    "threshold_source": "pooled inner-known q05 only",
                    "inner_runs": inner_metrics,
                    "inner_mean": mean_metrics(inner_metrics),
                    "outer": metrics,
                }
                outer_records[name].append(records)
                print(
                    f"outer={outer_fold} {name}: C={metrics['C']:.4f} "
                    f"F1={metrics['identity_macro_F1']:.4f} AUROC={metrics['AUROC']:.4f} "
                    f"U@95={metrics['U_at_95']:.4f} U={metrics['U']:.4f}", flush=True,
                )
            del models, scores, fit_scores
            torch.cuda.empty_cache()
        result["folds"].append(fold)
        result["completed_outer_folds"] = len(result["folds"])
        write_json(SMOKE_RESULTS_PATH if smoke else RESULTS_PATH, result)
    summary = {}
    for name in candidate_ids:
        rows = [fold["candidates"][name]["outer"] for fold in result["folds"]]
        inner_rows = [
            inner for fold in result["folds"] for inner in fold["candidates"][name]["inner_runs"]
        ]
        outer_mean = mean_metrics(rows)
        item = {
            "inner_run_mean": mean_metrics(inner_rows),
            "outer_fold_mean": outer_mean,
            "outer_fold_std": std_metrics(rows),
            "pooled_outer": pooled_metrics(outer_records[name]),
        }
        if not smoke:
            gates = {
                "C": bool(outer_mean["C"] >= float(config["gates"]["C_min"])),
                "AUROC": bool(outer_mean["AUROC"] >= float(config["gates"]["AUROC_min"])),
                "U_at_95": bool(outer_mean["U_at_95"] >= float(config["gates"]["U_at_95_min"])),
            }
            item["gates"] = {**gates, "all_passed": all(gates.values())}
        summary[name] = item
    result["summary"] = summary
    result["status"] = "complete_smoke" if smoke else "complete"
    result["elapsed_seconds"] = time.time() - started
    result["data_isolation_all_passed"] = bool(all(
        all(role["isolation"].values()) for fold in result["folds"] for role in fold["inner_roles"]
    ) and all(all(fold["outer_isolation"].values()) for fold in result["folds"]))
    if not smoke:
        result["any_candidate_all_gates_passed"] = any(
            item["gates"]["all_passed"] for item in summary.values()
        )
        artifact = {
            "experiment": config["experiment"],
            "config_sha256": sha256(CONFIG_PATH),
            "metadata_sha256": manifest["metadata"]["sha256"],
            "feature_sha256": {
                name: row["sha256"] for name, row in manifest["formal_features"].items()
            },
            "models": saved_models,
        }
        torch.save(artifact, MODEL_PATH)
        result["outer_template_artifact"] = {
            "path": str(MODEL_PATH.relative_to(ROOT)),
            "sha256": sha256(MODEL_PATH),
            "N_outer_folds": len(groups),
            "includes_inner_or_smoke_models": False,
        }
    write_json(SMOKE_RESULTS_PATH if smoke else RESULTS_PATH, result)
    return result


def fmt(value: float) -> str:
    return f"{value:.6f}"


def metric_table(result: dict, section: str, only_legacy: bool = False) -> list[str]:
    lines = [
        "| 特征 / 模板空间 / 置信度 | direct C | C | 身份 macro-F1 | AUROC | U@95 | 工作点 U | unknown-F1 | 非官方 C-U proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["summary"].items():
        if only_legacy and not name.startswith("legacy_"):
            continue
        metrics = item[section]
        f1_key = "identity_macro_F1"
        lines.append(
            f"| {name} | {fmt(metrics['direct_C'])} | {fmt(metrics['C'])} | "
            f"{fmt(metrics[f1_key])} | {fmt(metrics['AUROC'])} | {fmt(metrics['U_at_95'])} | "
            f"{fmt(metrics['U'])} | {fmt(metrics['unknown_F1'])} | "
            f"{fmt(metrics['C_U_proxy_not_official'])} |"
        )
    return lines


def write_report(result: dict) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    smoke = None
    if SMOKE_RESULTS_PATH.exists():
        smoke = json.loads(SMOKE_RESULTS_PATH.read_text(encoding="utf-8"))
    passed = [name for name, item in result["summary"].items() if item["gates"]["all_passed"]]
    known_n = result["summary"][next(iter(result["summary"]))]["pooled_outer"]["N_known"]
    unknown_n = result["summary"][next(iter(result["summary"]))]["pooled_outer"]["N_unknown"]
    lines = [
        "# E72：配对协议正确性审计与模板匹配基线", "",
        "## 结论", "",
        f"- 已完成 **5 个外层留被试折 × 每折 2 个内层标定折**；正式比较仅使用纯 EEG 协方差与相关 logSPD 缓存。外层 known 连续块累计 N={known_n}，outer unknown 全 Trial 累计 N={unknown_n}。",
        f"- 预注册出口为 C≥0.95、AUROC≥0.98、U@95≥0.95。满足三项的候选：**{', '.join(passed) if passed else '无'}**。这里只作冻结候选的描述性确认，不依据外层重新选模或调阈值。",
        "- Test AUC、F1、C、RA、ASafeR、A、U、Score：**N/A**；无公开真值，且本实验未读取 Test 数据、索引或标签，也未生成 Test 预测。",
        "- 官方 Score：**N/A**，因为 clean-only 实验没有 RA、ASafeR、A。表中 `100*C^0.40*U^0.25` 仅为明确标注的非官方 C-U proxy，不能当作官方 Score。", "",
        "## 实际执行命令", "",
        "```powershell",
        ".venv/Scripts/python.exe research/E72_pair_protocol/test_pair_protocol.py",
    ]
    commands = []
    for item in manifest.get("audit_history", []):
        if item["command"] not in commands:
            commands.append(item["command"])
    if smoke and smoke["invocation"]["command"] not in commands:
        commands.append(smoke["invocation"]["command"])
    if result["invocation"]["command"] not in commands:
        commands.append(result["invocation"]["command"])
    lines.extend(commands)
    lines.extend(["```", "", f"完整验证耗时：{result['elapsed_seconds']:.2f} 秒。逐折模型已保存至 `{result['outer_template_artifact']['path']}`。", "",
                  "## 数据、配对与隔离口径", "",
                  "- 仅 Train/session1：20 人、11,304 trials、18 通道、1,200 点、300 Hz，任务 EC/EO/Imagery/SSVEP。任务和时间块只构造划分，不作为推理输入。",
                  "- RNG42 将 20 人分成五组，每组 4 人 outer unknown；其余 16 人以 RNG=100+outer 分成两组 8 人。每内折 enrolled=其余 8 人，8 名留出者全部只作 calibration unknown；**没有 OE 角色**。",
                  "- inner fit：enrolled 且 trial≤0.60×subject/task 内最大 trial；inner known query：0.60–0.80；inner unknown：8 名留出者的 ≤0.80 数据。",
                  "- outer 模板：16 名 enrolled 的 ≤0.80 数据；outer known query：**trial>0.80×最大值的连续末端文件块**；outer unknown：4 人全部 trials，仅最终评估。",
                  "- 配对为查询样本对全部注册模板：每个 known query 恰有一个 same pair，其余为 different；unknown query 的所有模板对均为 different。没有物化 trial×trial 的 N² 矩阵。",
                  "- 每个候选阈值只取对应 outer group 内全部 inner-known confidence 的 q05（linear）；未扫描阈值，未使用 inner/outer unknown 调阈值。",
                  f"- 全部已执行折的数据隔离断言通过：**{result['data_isolation_all_passed']}**。", "",
                  "## 输入正确性与哈希审计", "",
                  "| 缓存 | shape | dtype | finite | 重算最大绝对差 | 行对齐 | SHA256 |",
                  "|---|---|---|---|---:|---|---|"])
    for name, row in manifest["formal_features"].items():
        lines.append(
            f"| {name} | {row['shape']} | {row['dtype']} | {row['finite']} | "
            f"{row['cache_vs_rebuilt_max_abs']:.3g} | {row['row_alignment_allclose']} | `{row['sha256']}` |"
        )
    lines.extend([
        "", f"- Train metadata SHA256：`{manifest['metadata']['sha256']}`；两个正式缓存均用抽样 Train 原始波形在 GPU 逐 Trial 重算并核对行对齐。",
        f"- 图像直接编码审计：shape={manifest['image_audit']['direct_encode']['shape']}、dtype={manifest['image_audit']['direct_encode']['dtype']}、finite={manifest['image_audit']['direct_encode']['finite']}、范围=[{manifest['image_audit']['direct_encode']['min']:.6g}, {manifest['image_audit']['direct_encode']['max']:.6g}]，通过={manifest['image_audit']['direct_encode']['passed']}。",
        f"- 旧图像缓存为 `{manifest['image_audit']['legacy_uint8_image_cache']['dtype']}`，抽样范围 [{manifest['image_audit']['legacy_uint8_image_cache']['sample_min']}, {manifest['image_audit']['legacy_uint8_image_cache']['sample_max']}]，不能直接当作 float32 [0,1] 输入。历史 E20 提取源码在 DINO 前有 `/255.0`；E72 未修改旧缓存。",
        f"- 共享图像编码器频带为 {manifest['image_audit']['encoder_bands_hz']}，其中 {manifest['image_audit']['out_of_official_band_definitions']} 超出官方 1–40 Hz；因此旧 DINO 仅用于 smoke，不进入正式结论。", "",
        "## 内层标定结果（10 inner runs 均值）", "",
        "C/U/F1 使用同一 pooled inner-known q05；inner U@95 则按各 inner run 自己的 known q05 计算，两者不必完全相等。内层结果只用于协议检查，不用于从外层结果反向选模。", "",
    ])
    lines.extend(metric_table(result, "inner_run_mean"))
    lines.extend(["", "## 外层五折均值（正式结果）", "",
                  "AUROC 以 known=1、unknown=0，confidence 越大越像 known。`U@95` 严格复用 `src.metrics.open_set_metrics`：在相应 outer known confidence 的 q05 下计算 unknown 拒识率；C、U、身份 macro-F1 和 unknown-F1 则使用内层冻结 q05 工作阈值。direct C 即连续时间块不拒识身份准确率。", ""])
    lines.extend(metric_table(result, "outer_fold_mean"))
    lines.extend(["", "### 三项出口门禁", "",
                  "| 候选 | C≥0.95 | AUROC≥0.98 | U@95≥0.95 | 全部 |",
                  "|---|---|---|---|---|"])
    for name, item in result["summary"].items():
        gates = item["gates"]
        lines.append(
            f"| {name} | {'通过' if gates['C'] else '未通过'} | "
            f"{'通过' if gates['AUROC'] else '未通过'} | {'通过' if gates['U_at_95'] else '未通过'} | "
            f"{'通过' if gates['all_passed'] else '未通过'} |"
        )
    lines.extend(["", "### WCCN + distance_ratio 逐外层折", "",
                  "| 折 | outer unknown | 特征 | C | 身份 macro-F1 | AUROC | U@95 | U |",
                  "|---:|---|---|---:|---:|---:|---:|---:|"])
    for fold in result["folds"]:
        for feature in ("e36_covariance", "e65_correlation"):
            name = f"{feature}/wccn/distance_ratio"
            metrics = fold["candidates"][name]["outer"]
            lines.append(
                f"| {fold['outer_fold']} | {','.join(map(str, fold['outer_unknown_subjects']))} | {feature} | "
                f"{fmt(metrics['C'])} | {fmt(metrics['identity_macro_F1'])} | {fmt(metrics['AUROC'])} | "
                f"{fmt(metrics['U_at_95'])} | {fmt(metrics['U'])} |"
            )
    if smoke:
        lines.extend(["", "## Smoke（非正式证据）", "",
                      f"- smoke 仅执行第一个 outer group、第一个 inner split，状态 `{smoke['status']}`；配对接口审计：{smoke.get('smoke_pair_audit', {})}。",
                      "- 下表旧 DINO 数字只证明流程可运行，**不得作为新结论或与正式纯 EEG 结果比较**。", ""])
        lines.extend(metric_table(smoke, "outer_fold_mean", only_legacy=True))
    lines.extend(["", "## 证据边界与问题", "",
                  "- 所有 unknown 都是 Train 留被试 pseudo-unknown；连续末端文件块只是同 Session 时间隔离代理，不是真实跨天/跨 Session 证据，不能推出官方 Test 门禁可达。",
                  "- Cosine/WCCN 决定模板空间，max cosine、top1−top2 margin、distance_ratio 决定置信度；所有候选使用完全相同的被试折、时间块和 q05 规则。",
                  "- 完整逐折阈值、样本数、身份/未知/联合 F1、AUPR、fold std、pooled 指标、拟合准确率及角色 ID 见 `results.json`；缓存与图像细节见 `input_manifest.json`。",
                  "- 阶段出口由两路 WCCN+margin 达到；其余未通过候选不再扫阈值。Train 出口通过不等于跨 Session 问题解决，后续仍需真实跨 Session 监督与独立验证。", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "full", "report", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "report":
        write_report(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
        print(f"报告：{REPORT_PATH}", flush=True)
        return
    if not torch.cuda.is_available():
        parser.error("E72 要求 CUDA GPU")
    config = load_config()
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda")
    metadata = load_train_metadata(config)
    manifest = run_audit(metadata, config, device, args.stage)
    if args.stage == "audit":
        print(f"输入清单：{MANIFEST_PATH}", flush=True)
        return
    if args.stage in ("smoke", "all"):
        features = load_features(config, device, smoke=True)
        smoke_result = run_protocol(features, metadata, config, manifest, args.stage, smoke=True)
        print(f"Smoke 完成：{SMOKE_RESULTS_PATH} ({smoke_result['status']})", flush=True)
        del features, smoke_result
        torch.cuda.empty_cache()
    if args.stage in ("full", "all"):
        features = load_features(config, device, smoke=False)
        result = run_protocol(features, metadata, config, manifest, args.stage, smoke=False)
        write_report(result)
        print(f"完整结果：{RESULTS_PATH}\n报告：{REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
