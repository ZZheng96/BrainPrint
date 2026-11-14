"""E66：冻结 NeuroShield-FM 的四窗嵌入、归一化原型与开放集门禁。"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
from run_e36 import load_config as load_e36_config, load_train_matrix, sha256, train_metadata  # noqa: E402
import run_e54 as E54  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
PREDICTIONS_PATH = OUT / "e66_predictions.csv"
CACHE = ROOT / "BrainPrint_cache" / "e66_neuroshield_fm"
EMBEDDINGS_PATH = CACHE / "train_trial_embeddings.npy"
FEATURE_MANIFEST = CACHE / "feature_manifest.json"
MODEL_PATH = CACHE / "frozen_prototypes.pt"
FREEZE_MANIFEST = CACHE / "frozen_manifest.json"
CLASSES = np.arange(1, 21, dtype=int)
TEST_METRICS = {key: "N/A（无真实标签）" for key in ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_foundation(config: dict, device: torch.device):
    """严格按官方 minimal_inference 的导入、建模与权重加载路径。"""
    repo = ROOT / config["foundation"]["repo"]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--cuda_devices", "0"]
    try:
        module = importlib.import_module("eeg_auth_train_optuna")
    finally:
        sys.argv = old_argv
    meta_path = ROOT / config["foundation"]["meta"]
    checkpoint = ROOT / config["foundation"]["checkpoint"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    params = dict(meta["trial_params_effective"])
    model, embed_dim = module.build_model(
        trial=None, num_channels=len(module.REF_CLIST_NORM), device=device, fixed_params=params
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    if embed_dim != int(config["foundation"]["expected_embedding_dim"]):
        raise ValueError(f"NeuroShield embedding维度异常：{embed_dim}")
    names = [module.normalize_label(name) for name in config["channel_order"]]
    positions = module.names_norm_to_ref_pos3d(names)
    if positions.shape != (18, 3) or not np.isfinite(positions).all() or not np.abs(positions).any(axis=1).all():
        raise ValueError("18通道的standard_1005坐标不完整")
    info = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "meta": str(meta_path), "meta_sha256": sha256(meta_path),
        "official_code_sha256": sha256(repo / "eeg_auth_train_optuna.py"),
        "embedding_dim": int(embed_dim), "channel_names_normalized": names,
        "positions": positions.tolist(), "strict_load": True, "frozen": True,
    }
    return model, torch.as_tensor(positions, dtype=torch.float32, device=device), info


def preprocess_windows(raw: torch.Tensor, config: dict) -> torch.Tensor:
    """B×18×1200 -> B×4×18×500；所有统计均限制在单条trial内部。"""
    if raw.ndim != 3 or raw.shape[1:] != (18, 1200):
        raise ValueError(f"期望B×18×1200，得到{tuple(raw.shape)}")
    x = raw.float()
    median = x.median(dim=-1, keepdim=True).values
    centered = x - median
    eps = float(config["preprocess"]["scale_epsilon"])
    mad = 1.4826 * centered.abs().median(dim=-1, keepdim=True).values
    rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
    scale = torch.where(mad > eps, mad, rms).clamp_min(eps)
    x = centered / scale
    x = x - x.mean(dim=1, keepdim=True)  # CAR
    spectrum = torch.fft.rfft(x, n=1200, dim=-1)
    frequency = torch.fft.rfftfreq(1200, d=1.0 / 300.0, device=x.device)
    low, high = config["preprocess"]["bandpass_hz"]
    spectrum = spectrum * ((frequency >= low) & (frequency <= high)).to(x.dtype)
    padded = torch.zeros((len(x), 18, 1001), dtype=spectrum.dtype, device=x.device)
    padded[..., :spectrum.shape[-1]] = spectrum
    resampled = torch.fft.irfft(padded, n=2000, dim=-1) * (2000.0 / 1200.0)
    return resampled.reshape(len(x), 18, 4, 500).permute(0, 2, 1, 3).contiguous()


def embed_raw(raw: torch.Tensor, model: torch.nn.Module, positions: torch.Tensor,
              config: dict) -> torch.Tensor:
    windows = preprocess_windows(raw, config)
    batch = len(windows)
    x = windows.reshape(batch * 4, 18, 500)
    mask = torch.ones((batch * 4, 18), dtype=torch.bool, device=x.device)
    pos = positions.unsqueeze(0).expand(batch * 4, -1, -1)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        window_embedding = model(x, mask, pos3d=pos)
    embedding = F.normalize(window_embedding.float().reshape(batch, 4, -1).mean(dim=1), dim=1, eps=1e-12)
    if not torch.isfinite(embedding).all():
        raise FloatingPointError("NeuroShield嵌入含非有限值")
    return embedding


def embedding_invariance(raw: torch.Tensor, model: torch.nn.Module, positions: torch.Tensor,
                         config: dict) -> dict:
    batch = embed_raw(raw, model, positions, config)
    singles = torch.cat([embed_raw(raw[i:i + 1], model, positions, config) for i in range(len(raw))])
    permutation = np.random.default_rng(int(config["seed"])).permutation(len(raw))
    inverse = np.argsort(permutation)
    permuted = embed_raw(raw[torch.as_tensor(permutation, device=raw.device)], model, positions, config)
    atol = float(config["numerical_check"]["embedding_atol"])
    rtol = float(config["numerical_check"]["embedding_rtol"])
    result = {
        "N": int(len(raw)),
        "single_vs_batch_max_abs": float((batch - singles).abs().max()),
        "permutation_max_abs": float((batch - permuted[inverse]).abs().max()),
        "single_vs_batch_allclose": bool(torch.allclose(batch, singles, atol=atol, rtol=rtol)),
        "permutation_allclose": bool(torch.allclose(batch, permuted[inverse], atol=atol, rtol=rtol)),
        "cross_trial_or_batch_statistics": False,
    }
    result["passed"] = bool(result["single_vs_batch_allclose"] and result["permutation_allclose"])
    return result


def run_smoke(config: dict, device: torch.device) -> dict:
    model, positions, info = load_foundation(config, device)
    metadata = train_metadata(load_e36_config())
    paths = [ROOT / config["data"]["train_root"] / path for path in metadata["path"].iloc[:2]]
    raw = torch.as_tensor(np.stack([load_train_matrix(path) for path in paths]), dtype=torch.float32, device=device)
    embedding = embed_raw(raw, model, positions, config)
    result = {
        "passed": bool(embedding.shape == (2, 128) and torch.isfinite(embedding).all()),
        "input_shape": list(raw.shape), "embedding_shape": list(embedding.shape),
        "embedding_norm": embedding.norm(dim=1).cpu().tolist(), "foundation": info,
        "gpu": torch.cuda.get_device_name(0), "test_accessed": False,
    }
    write_json(CACHE / "smoke.json", result)
    print(f"E66 smoke {json.dumps({k: v for k, v in result.items() if k != 'foundation'}, ensure_ascii=False)}", flush=True)
    if not result["passed"]:
        raise RuntimeError("E66官方权重smoke失败")
    return result


def build_train_embeddings(config: dict, device: torch.device, resume: bool = True) -> dict:
    if resume and EMBEDDINGS_PATH.exists() and FEATURE_MANIFEST.exists():
        report = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
        if report["config_sha256"] == sha256(CONFIG_PATH) and report["script_sha256"] == sha256(Path(__file__)):
            cached = np.load(EMBEDDINGS_PATH, mmap_mode="r")
            if cached.shape == (11304, 128) and np.isfinite(cached).all():
                print("复用E66冻结Train嵌入", flush=True)
                return report
    started = time.time()
    model, positions, foundation_info = load_foundation(config, device)
    metadata = train_metadata(load_e36_config())
    rng = np.random.default_rng(int(config["seed"]))
    selected = np.sort(rng.choice(len(metadata), int(config["numerical_check"]["trials"]), replace=False))
    # 上次运行已完成嵌入但止于数值校验时，只重算8条样本来验证并接管现有缓存。
    if resume and EMBEDDINGS_PATH.exists() and not FEATURE_MANIFEST.exists():
        cached = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        if cached.shape == (len(metadata), 128) and np.isfinite(cached).all():
            paths = [ROOT / config["data"]["train_root"] / metadata["path"].iloc[index] for index in selected]
            raw = torch.as_tensor(np.stack([load_train_matrix(path) for path in paths]),
                                  dtype=torch.float32, device=device)
            checks = embedding_invariance(raw, model, positions, config)
            rebuilt = embed_raw(raw, model, positions, config).cpu().numpy()
            checks["cache_vs_rebuilt_max_abs"] = float(np.max(np.abs(cached[selected] - rebuilt)))
            checks["cache_vs_rebuilt_allclose"] = bool(np.allclose(
                cached[selected], rebuilt, atol=float(config["numerical_check"]["embedding_atol"]),
                rtol=float(config["numerical_check"]["embedding_rtol"])))
            if not checks["passed"] or not checks["cache_vs_rebuilt_allclose"]:
                raise RuntimeError(f"E66现有嵌入缓存复验失败：{checks}")
            report = {
                "shape": [len(metadata), 128], "dtype": "float32", "finite": True,
                "cache": str(EMBEDDINGS_PATH), "cache_sha256": sha256(EMBEDDINGS_PATH),
                "config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__)),
                "metadata_sha256": sha256(ROOT / config["data"]["train_metadata"]),
                "checks": checks, "foundation": foundation_info, "test_accessed": False,
                "recovered_unmanifested_cache_after_numerical_check_stop": True,
                "elapsed_seconds": time.time() - started, "gpu": torch.cuda.get_device_name(0),
            }
            write_json(FEATURE_MANIFEST, report)
            print(f"E66现有Train嵌入复验通过：{checks}", flush=True)
            return report
    raw = E54.load_all_raw(metadata, E54.load_config(), device)
    output = np.lib.format.open_memmap(EMBEDDINGS_PATH, mode="w+", dtype=np.float32,
                                       shape=(len(metadata), 128))
    batch_size = int(config["identity"]["feature_batch_size"])
    for start in range(0, len(metadata), batch_size):
        stop = min(start + batch_size, len(metadata))
        output[start:stop] = embed_raw(raw[start:stop], model, positions, config).cpu().numpy()
        if stop % 512 == 0 or stop == len(metadata):
            print(f"E66 Train embeddings {stop}/{len(metadata)}", flush=True)
    output.flush()
    del output
    rng = np.random.default_rng(int(config["seed"]))
    selected = np.sort(rng.choice(len(metadata), int(config["numerical_check"]["trials"]), replace=False))
    index = torch.as_tensor(selected, dtype=torch.long, device=device)
    checks = embedding_invariance(raw[index], model, positions, config)
    cached = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    checks["cache_vs_rebuilt_max_abs"] = float(np.max(np.abs(
        cached[selected] - embed_raw(raw[index], model, positions, config).cpu().numpy())))
    if not checks["passed"]:
        raise RuntimeError(f"E66嵌入逐样本不变性失败：{checks}")
    report = {
        "shape": [len(metadata), 128], "dtype": "float32", "finite": bool(np.isfinite(cached).all()),
        "cache": str(EMBEDDINGS_PATH), "cache_sha256": sha256(EMBEDDINGS_PATH),
        "config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__)),
        "metadata_sha256": sha256(ROOT / config["data"]["train_metadata"]),
        "checks": checks, "foundation": foundation_info, "test_accessed": False,
        "elapsed_seconds": time.time() - started, "gpu": torch.cuda.get_device_name(0),
    }
    write_json(FEATURE_MANIFEST, report)
    del raw, model
    torch.cuda.empty_cache()
    return report


def normalized_prototypes(embedding: torch.Tensor, labels: np.ndarray,
                          classes: np.ndarray) -> torch.Tensor:
    target = torch.as_tensor(labels, dtype=torch.long, device=embedding.device)
    means = torch.stack([embedding[target == int(label)].mean(dim=0) for label in classes])
    return F.normalize(means, dim=1, eps=1e-12)


def cosine_scores(embedding: torch.Tensor, prototypes: torch.Tensor,
                  indices: np.ndarray | None = None) -> np.ndarray:
    if indices is not None:
        embedding = embedding[torch.as_tensor(indices, dtype=torch.long, device=embedding.device)]
    return (embedding @ prototypes.T).cpu().numpy().astype(np.float64)


def confidence(scores: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(scores, dtype=np.float64), axis=1)


def thresholds(known_scores: np.ndarray, config: dict, final20: bool = False) -> tuple[float, float]:
    method = config["identity"]["final20_quantile_method" if final20 else "inner_quantile_method"]
    values = confidence(known_scores)
    return (float(np.quantile(values, float(config["identity"]["tau_low_quantile"]), method=method)),
            float(np.quantile(values, float(config["identity"]["u95_quantile"]), method=method)))


def decide(scores: np.ndarray, classes: np.ndarray, tau_low: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct = classes[np.asarray(scores).argmax(axis=1)].astype(int)
    conf = confidence(scores)
    return np.where(conf < tau_low, 0, direct).astype(int), direct, conf


def detection(known_conf: np.ndarray, unknown_conf: np.ndarray) -> dict:
    labels = np.concatenate([np.ones(len(known_conf), dtype=int), np.zeros(len(unknown_conf), dtype=int)])
    values = np.concatenate([known_conf, unknown_conf])
    return {
        "AUROC": float(roc_auc_score(labels, values)),
        "unknown_AUPR": float(average_precision_score(1 - labels, -values)),
        "known_AUPR": float(average_precision_score(labels, values)),
    }


def confidence_summary(values: np.ndarray) -> dict:
    quantiles = np.quantile(values, [0.02, 0.05, 0.5, 0.95])
    return {"N": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
            "q02": float(quantiles[0]), "q05": float(quantiles[1]),
            "median": float(quantiles[2]), "q95": float(quantiles[3])}


def clean_run_metrics(run: dict) -> dict:
    classes = run["classes"]
    true = run["known_true"]
    tau_low, tau95 = run["tau_low"], run["tau95"]
    known_pred, direct, known_conf = decide(run["known_scores"], classes, tau_low)
    unknown_pred, _, unknown_conf = decide(run["unknown_scores"], classes, tau_low)
    detection_true = np.concatenate([np.ones(len(true), dtype=int), np.zeros(len(unknown_pred), dtype=int)])
    accepted = np.concatenate([known_pred != 0, unknown_pred != 0])
    return {
        "N_known": int(len(true)), "N_unknown": int(len(unknown_pred)),
        "direct_C": float(np.mean(direct == true)),
        "threshold_C": float(np.mean(known_pred == true)),
        "continuous_block_accuracy": float(np.mean(direct == true)),
        "identity_macro_F1_direct": float(f1_score(true, direct, labels=classes, average="macro", zero_division=0)),
        "identity_macro_F1_threshold": float(f1_score(true, known_pred, labels=classes, average="macro", zero_division=0)),
        **detection(known_conf, unknown_conf),
        "U_at_95": float(np.mean(unknown_conf < tau95)),
        "working_U": float(np.mean(unknown_pred == 0)),
        "known_acceptance": float(np.mean(known_pred != 0)),
        "unknown_F1": float(f1_score(detection_true, accepted, pos_label=0, zero_division=0)),
        "joint_F1": float(f1_score(np.concatenate([true, np.zeros(len(unknown_pred), dtype=int)]),
                                    np.concatenate([known_pred, unknown_pred]),
                                    labels=np.concatenate([[0], classes]), average="macro", zero_division=0)),
        "identity_fit_accuracy": float(run["identity_fit_accuracy"]),
        "tau_low": float(tau_low), "tau95": float(tau95),
    }


def mean_rows(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
            if isinstance(rows[0][key], (int, float, np.integer, np.floating))
            and not isinstance(rows[0][key], (bool, np.bool_))}


def pooled_clean(runs: list[dict]) -> dict:
    true_parts, known_parts, unknown_parts, direct_parts = [], [], [], []
    known_conf, unknown_conf = [], []
    for run in runs:
        kp, direct, kc = decide(run["known_scores"], run["classes"], run["tau_low"])
        up, _, uc = decide(run["unknown_scores"], run["classes"], run["tau_low"])
        true_parts.append(run["known_true"])
        known_parts.append(kp)
        unknown_parts.append(up)
        direct_parts.append(direct)
        known_conf.append(kc)
        unknown_conf.append(uc)
    true = np.concatenate(true_parts)
    kp = np.concatenate(known_parts)
    up = np.concatenate(unknown_parts)
    direct = np.concatenate(direct_parts)
    detection_true = np.concatenate([np.ones(len(true), dtype=int), np.zeros(len(up), dtype=int)])
    accepted = np.concatenate([kp != 0, up != 0])
    return {
        "N_known": int(len(true)), "N_unknown": int(len(up)),
        "direct_C": float(np.mean(direct == true)), "threshold_C": float(np.mean(kp == true)),
        "identity_20class_macro_F1_direct": float(f1_score(true, direct, labels=CLASSES, average="macro", zero_division=0)),
        "identity_20class_macro_F1_threshold": float(f1_score(true, kp, labels=CLASSES, average="macro", zero_division=0)),
        **detection(np.concatenate(known_conf), np.concatenate(unknown_conf)),
        "working_U": float(np.mean(up == 0)),
        "unknown_F1": float(f1_score(detection_true, accepted, pos_label=0, zero_division=0)),
        "joint_21class_F1": float(f1_score(np.concatenate([true, np.zeros(len(up), dtype=int)]),
                                           np.concatenate([kp, up]), labels=np.arange(21),
                                           average="macro", zero_division=0)),
    }


def drift_summary(runs: list[dict]) -> dict:
    output = {}
    for block in ("fit", "calibration", "sanity"):
        rows = []
        for run in runs:
            row = confidence_summary(run["confidence_blocks"][block])
            fit = confidence_summary(run["confidence_blocks"]["fit"])
            row["drift_vs_fit"] = {key: float(row[key] - fit[key]) for key in ("mean", "q02", "q05", "median", "q95")}
            rows.append(row)
        output[block] = {
            "run_mean": mean_rows(rows),
            "drift_vs_fit_run_mean": mean_rows([row["drift_vs_fit"] for row in rows]),
        }
    return output


def make_inner_run(embeddings: torch.Tensor, metadata: pd.DataFrame, outer: int, inner: int,
                   unknown_subjects: np.ndarray, known_subjects: np.ndarray,
                   inner_unknown: np.ndarray, config: dict, e54_config: dict) -> dict:
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    roles = E54.build_inner_roles(known_subjects, inner_unknown, unknown_subjects, outer, inner, e54_config)
    enrolled = roles["enrolled"]
    masks = {
        "fit": trial <= 0.6 * maximum,
        "calibration": (trial > 0.6 * maximum) & (trial <= 0.7 * maximum),
        "sanity": (trial > 0.7 * maximum) & (trial <= 0.8 * maximum),
        "known": (trial > 0.6 * maximum) & (trial <= 0.8 * maximum),
        "unknown": trial <= 0.8 * maximum,
    }
    fit_np = np.flatnonzero(np.isin(subjects, enrolled) & masks["fit"])
    known_np = np.flatnonzero(np.isin(subjects, enrolled) & masks["known"])
    unknown_np = np.flatnonzero(np.isin(subjects, roles["calibration"]) & masks["unknown"])
    known_subset = E54.stratified_subset(metadata, known_np, int(config["protocol"]["formal_trials_per_subject_task"]))
    unknown_subset = E54.stratified_subset(metadata, unknown_np, int(config["protocol"]["formal_trials_per_subject_task"]))
    prototypes = normalized_prototypes(embeddings[torch.as_tensor(fit_np, device=embeddings.device)],
                                       subjects[fit_np], enrolled)
    fit_scores = cosine_scores(embeddings, prototypes, fit_np)
    known_scores = cosine_scores(embeddings, prototypes, known_subset)
    unknown_scores = cosine_scores(embeddings, prototypes, unknown_subset)
    tau_low, tau95 = thresholds(known_scores, config)
    confidence_blocks = {}
    for block in ("fit", "calibration", "sanity"):
        indices = np.flatnonzero(np.isin(subjects, enrolled) & masks[block])
        confidence_blocks[block] = confidence(cosine_scores(embeddings, prototypes, indices))
    return {
        "experiment": "E66", "config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__)),
        "outer_fold": outer, "inner_fold": inner, "classes": enrolled,
        "roles": {key: np.asarray(value).tolist() for key, value in roles.items() if key != "pairwise_disjoint"},
        "known_true": subjects[known_subset], "known_indices": known_subset, "unknown_indices": unknown_subset,
        "known_scores": known_scores, "unknown_scores": unknown_scores,
        "identity_fit_accuracy": float(np.mean(enrolled[fit_scores.argmax(1)] == subjects[fit_np])),
        "tau_low": tau_low, "tau95": tau95, "confidence_blocks": confidence_blocks,
    }


def summarize_runs(runs: list[dict]) -> dict:
    folds = []
    for outer in sorted(set(run["outer_fold"] for run in runs)):
        pair = [run for run in runs if run["outer_fold"] == outer]
        clean_runs = [clean_run_metrics(run) for run in pair]
        folds.append({"outer_fold": outer, "inner_runs": [run["inner_fold"] for run in pair],
                      "clean_runs": clean_runs, "clean": mean_rows(clean_runs)})
    return {"outer_folds": folds,
            "summary": {"clean_outer_mean": mean_rows([fold["clean"] for fold in folds]),
                        "pooled_clean": pooled_clean(runs), "confidence_drift": drift_summary(runs)}}


def gates_for(summary: dict, config: dict) -> dict:
    clean = summary["clean_outer_mean"]
    gate = config["test_gate"]
    values = {
        "clean_direct_C": (float(clean["direct_C"]), float(gate["clean_direct_C_min"])),
        "clean_identity_macro_F1_direct": (float(clean["identity_macro_F1_direct"]),
                                             float(gate["clean_identity_macro_F1_direct_min"])),
        "clean_AUROC": (float(clean["AUROC"]), float(gate["clean_AUROC_min"])),
        "clean_U_at_95": (float(clean["U_at_95"]), float(gate["clean_U_at_95_min"])),
    }
    return {key: {"value": value, "minimum": minimum, "passed": bool(value >= minimum)}
            for key, (value, minimum) in values.items()}


def run_inner(config: dict, device: torch.device, resume: bool = True) -> dict:
    started = time.time()
    feature_report = build_train_embeddings(config, device, resume)
    metadata = train_metadata(load_e36_config())
    embeddings = torch.as_tensor(np.load(EMBEDDINGS_PATH, allow_pickle=False), dtype=torch.float32, device=device)
    e54_config = E54.load_config()
    subjects = metadata["subject"].to_numpy(dtype=int)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(np.random.default_rng(int(config["protocol"]["outer_seed"])).permutation(all_subjects), 5)
    runs = []
    for outer, unknown_subjects in enumerate(outer_groups, start=1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        inner_order = np.random.default_rng(100 + outer).permutation(known_subjects)
        for inner, inner_unknown in enumerate(np.array_split(inner_order, 2), start=1):
            checkpoint = CACHE / f"inner_o{outer}_i{inner}.pt"
            run = None
            if resume and checkpoint.exists():
                candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if candidate.get("config_sha256") == sha256(CONFIG_PATH) and candidate.get("script_sha256") == sha256(Path(__file__)):
                    run = candidate
                    print(f"复用E66断点 o{outer}i{inner}", flush=True)
            if run is None:
                print(f"E66 inner o{outer}i{inner}开始", flush=True)
                run = make_inner_run(embeddings, metadata, outer, inner, unknown_subjects,
                                     known_subjects, inner_unknown, config, e54_config)
                torch.save(run, checkpoint)
            runs.append(run)
            clean = clean_run_metrics(run)
            print(f"E66 o{outer}i{inner}: directC={clean['direct_C']:.4f} C={clean['threshold_C']:.4f} "
                  f"F1={clean['identity_macro_F1_threshold']:.4f} AUROC={clean['AUROC']:.4f} "
                  f"U95={clean['U_at_95']:.4f} U={clean['working_U']:.4f}", flush=True)
        partial = summarize_runs(runs)
        write_json(RESULTS_PATH, {"status": "running_inner", "completed_inner_runs": len(runs), **partial})
    summary = summarize_runs(runs)
    gates = gates_for(summary["summary"], config)
    all_clean_pass = all(row["passed"] for row in gates.values())
    result = {
        "status": "inner_complete_test_authorized" if all_clean_pass else "inner_complete_test_not_authorized",
        "experiment": "E66", "date": config["date_preregistered"],
        "feature_report": feature_report,
        "run_roles": [{"outer_fold": run["outer_fold"], "inner_fold": run["inner_fold"],
                       "classes": run["classes"].tolist(), "roles": run["roles"],
                       "identity_fit_accuracy": run["identity_fit_accuracy"]} for run in runs],
        **summary,
        "test_gates": {"checks": gates, "all_clean_passed": all_clean_pass,
                       "all_train_gates_passed_claim": False,
                       "reason": "E66未运行攻击套件，RA/ASafeR=N/A；四项clean门禁只授权冻结工程审计"},
        "test": {"status": "pending_authorized_single_audit" if all_clean_pass else "N/A_clean_gate_failed",
                 "test_accessed": False, "metrics": TEST_METRICS},
        "evidence_boundary": config["reporting"]["evidence_boundary"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "mne": importlib.import_module("mne").__version__,
                        "gpu": torch.cuda.get_device_name(0),
                        "compute": "FFT preprocessing and frozen transformer inference on CUDA; MAT I/O and metrics on CPU"},
        "script_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG_PATH),
        "inner_elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result, config)
    print(f"E66 INNER SUMMARY {json.dumps(summary['summary']['clean_outer_mean'], ensure_ascii=False)}", flush=True)
    print(f"E66 TEST GATES {json.dumps(gates, ensure_ascii=False)}", flush=True)
    return result


def final_blocks(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return {
        "fit": np.flatnonzero(trial <= 0.6 * maximum),
        "calibration": np.flatnonzero((trial > 0.6 * maximum) & (trial <= 0.7 * maximum)),
        "sanity": np.flatnonzero((trial > 0.7 * maximum) & (trial <= 0.8 * maximum)),
    }


def final_known_metrics(true: np.ndarray, scores: np.ndarray, tau_low: float) -> dict:
    pred, direct, conf = decide(scores, CLASSES, tau_low)
    return {
        "N": int(len(true)), "direct_C": float(np.mean(direct == true)),
        "threshold_C": float(np.mean(pred == true)),
        "identity_20class_macro_F1_direct": float(f1_score(true, direct, labels=CLASSES, average="macro", zero_division=0)),
        "identity_20class_macro_F1_threshold": float(f1_score(true, pred, labels=CLASSES, average="macro", zero_division=0)),
        "known_acceptance": float(np.mean(pred != 0)), "rejection_rate": float(np.mean(pred == 0)),
        "AUC": "N/A（final20无pseudo-unknown）",
    }


def source_paths(config: dict) -> dict[str, Path]:
    repo = ROOT / config["foundation"]["repo"]
    return {
        "e66_config": CONFIG_PATH, "e66_code": Path(__file__).resolve(),
        "official_code": repo / "eeg_auth_train_optuna.py",
        "checkpoint": ROOT / config["foundation"]["checkpoint"], "meta": ROOT / config["foundation"]["meta"],
        "train_metadata": ROOT / config["data"]["train_metadata"], "train_embeddings": EMBEDDINGS_PATH,
    }


def run_final20_and_test(config: dict, device: torch.device) -> dict:
    started = time.time()
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if not result.get("test_gates", {}).get("all_clean_passed", False):
        result["test"] = {"status": "N/A_clean_gate_failed", "test_accessed": False, "metrics": TEST_METRICS,
                          "reason": "至少一项预注册inner clean门禁失败；未读取Test，未生成预测"}
        write_json(RESULTS_PATH, result)
        write_report(result, config)
        print("E66 Test=N/A：inner clean门禁未全部通过", flush=True)
        return result
    if PREDICTIONS_PATH.exists():
        raise RuntimeError("E66一次性Test预测已存在，拒绝重跑或尝试变体")
    metadata = train_metadata(load_e36_config())
    subjects = metadata["subject"].to_numpy(dtype=int)
    blocks = final_blocks(metadata)
    embeddings = torch.as_tensor(np.load(EMBEDDINGS_PATH, allow_pickle=False), dtype=torch.float32, device=device)
    fit_np = blocks["fit"]
    prototypes = normalized_prototypes(embeddings[torch.as_tensor(fit_np, device=device)], subjects[fit_np], CLASSES)
    scores = {block: cosine_scores(embeddings, prototypes, indices) for block, indices in blocks.items()}
    tau_low, tau95 = thresholds(scores["calibration"], config, final20=True)
    final_metrics = {block: final_known_metrics(subjects[indices], scores[block], tau_low)
                     for block, indices in blocks.items()}
    final_conf = {block: confidence_summary(confidence(scores[block])) for block in blocks}
    for block in ("calibration", "sanity"):
        final_conf[block]["drift_vs_fit"] = {key: float(final_conf[block][key] - final_conf["fit"][key])
                                              for key in ("mean", "q02", "q05", "median", "q95")}
    model, positions, foundation_info = load_foundation(config, device)
    selected = blocks["sanity"][:int(config["numerical_check"]["trials"])]
    paths = [ROOT / config["data"]["train_root"] / metadata["path"].iloc[index] for index in selected]
    raw = torch.as_tensor(np.stack([load_train_matrix(path) for path in paths]), dtype=torch.float32, device=device)
    batch_embedding = embed_raw(raw, model, positions, config)
    batch_scores = (batch_embedding @ prototypes.T).cpu().numpy().astype(np.float64)
    single_scores = np.concatenate([(embed_raw(raw[i:i + 1], model, positions, config) @ prototypes.T).cpu().numpy()
                                    for i in range(len(raw))]).astype(np.float64)
    permutation = np.random.default_rng(int(config["seed"])).permutation(len(raw))
    inverse = np.argsort(permutation)
    perm_embedding = embed_raw(raw[torch.as_tensor(permutation, device=device)], model, positions, config)
    perm_scores = (perm_embedding @ prototypes.T).cpu().numpy().astype(np.float64)
    batch_pred = decide(batch_scores, CLASSES, tau_low)[0]
    single_pred = decide(single_scores, CLASSES, tau_low)[0]
    perm_pred = decide(perm_scores, CLASSES, tau_low)[0][inverse]
    atol = float(config["numerical_check"]["score_atol"])
    rtol = float(config["numerical_check"]["score_rtol"])
    numerical = {
        "N": int(len(raw)), "single_vs_batch_score_max_abs": float(np.max(np.abs(batch_scores - single_scores))),
        "permutation_score_max_abs": float(np.max(np.abs(batch_scores - perm_scores[inverse]))),
        "single_vs_batch_scores_allclose": bool(np.allclose(batch_scores, single_scores, atol=atol, rtol=rtol)),
        "permutation_scores_allclose": bool(np.allclose(batch_scores, perm_scores[inverse], atol=atol, rtol=rtol)),
        "single_vs_batch_labels_equal": bool(np.array_equal(batch_pred, single_pred)),
        "permutation_labels_equal": bool(np.array_equal(batch_pred, perm_pred)),
        "cross_sample_statistics_used": False,
    }
    numerical["passed"] = bool(all(numerical[key] for key in (
        "single_vs_batch_scores_allclose", "permutation_scores_allclose",
        "single_vs_batch_labels_equal", "permutation_labels_equal")))
    if not numerical["passed"]:
        raise RuntimeError(f"E66 final20逐样本数值检查失败：{numerical}")
    torch.save({"experiment": "E66", "classes": CLASSES, "prototypes": prototypes.cpu(),
                "channel_order": config["channel_order"], "foundation_checkpoint_sha256": foundation_info["checkpoint_sha256"]}, MODEL_PATH)
    manifest = {
        "frozen": True, "experiment": "E66", "tau_low": tau_low, "tau95_diagnostic_only": tau95,
        "threshold_quantiles": {"low": 0.02, "u95": 0.05, "method": "higher"},
        "decision": config["identity"]["decision"], "model_sha256": sha256(MODEL_PATH),
        "source_sha256": {name: sha256(path) for name, path in source_paths(config).items()},
        "test_session_used_for_freeze": False, "post_audit_changes_forbidden": True,
    }
    write_json(FREEZE_MANIFEST, manifest)
    result["final20_train"] = {
        "blocks": final_metrics, "confidence": final_conf, "numerical_invariance": numerical,
        "fit": "all 20 Train subjects <=60%; frozen encoder and normalized prototypes only",
        "calibration": "60-70% clean known max-cosine q02/q05", "sanity": "70-80% known continuous block",
        "tau_low": tau_low, "tau95": tau95,
    }
    result["freeze"] = {"model": str(MODEL_PATH), "manifest": str(FREEZE_MANIFEST),
                        "model_sha256": manifest["model_sha256"],
                        "completed_before_any_test_index_or_waveform_read": True}
    result["status"] = "final20_frozen_test_in_progress"
    write_json(RESULTS_PATH, result)

    # 冻结及全部指纹完成后才读取Test索引；此前没有Test访问。
    for name, path in source_paths(config).items():
        if sha256(path) != manifest["source_sha256"][name]:
            raise ValueError(f"E66冻结源指纹变化：{name} ({path})")
    sample_order = pd.read_csv(ROOT / config["data"]["test_csv"], usecols=["Sample"])
    test_paths = pd.read_csv(ROOT / config["data"]["test_metadata"], usecols=["sample", "path"])
    expected = int(config["data"]["expected_test_rows"])
    if len(sample_order) != expected or len(test_paths) != expected:
        raise ValueError(f"Test必须为{expected}行")
    if not np.array_equal(sample_order["Sample"].astype(str).to_numpy(), test_paths["sample"].astype(str).to_numpy()):
        raise ValueError("Test Sample与路径顺序不一致")
    if not test_paths["path"].astype(str).str.startswith("Test/").all():
        raise ValueError("Test索引含非Test路径")
    labels = np.empty(expected, dtype=np.int64)
    batch_size = int(config["identity"]["test_batch_size"])
    test_root = ROOT / "Brainprint_Competition"
    with ThreadPoolExecutor(max_workers=int(config["identity"]["io_workers"])) as pool:
        for start in range(0, expected, batch_size):
            stop = min(start + batch_size, expected)
            arrays = list(pool.map(load_train_matrix,
                                   [test_root / str(relative) for relative in test_paths["path"].iloc[start:stop]]))
            raw_batch = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
            trial_embedding = embed_raw(raw_batch, model, positions, config)
            batch_score = (trial_embedding @ prototypes.T).cpu().numpy().astype(np.float64)
            labels[start:stop] = decide(batch_score, CLASSES, tau_low)[0]
            if stop % 1024 < batch_size or stop == expected:
                print(f"E66 Test独立推理 {stop}/{expected}", flush=True)
    if not np.isin(labels, np.arange(21)).all():
        raise RuntimeError("E66 Test预测标签无效")
    pd.DataFrame({"Sample": sample_order["Sample"].to_numpy(), "Label": labels}).to_csv(PREDICTIONS_PATH, index=False)
    if len(pd.read_csv(PREDICTIONS_PATH)) != expected:
        raise RuntimeError("E66预测CSV行数校验失败")

    # 全部预测与CSV完成后，才首次读取Session做描述性审计。
    sessions = pd.read_csv(ROOT / config["data"]["test_metadata"], usecols=["sample", "session"])
    if not np.array_equal(sample_order["Sample"].astype(str).to_numpy(), sessions["sample"].astype(str).to_numpy()):
        raise ValueError("Session审计顺序不一致")
    def session_code(value: object) -> str:
        text = str(value).strip().casefold().replace("_", "").replace("-", "")
        mapping = {"1": "S1", "s1": "S1", "session1": "S1", "2": "S2", "s2": "S2", "session2": "S2"}
        if text not in mapping:
            raise ValueError(f"未知Test Session：{value}")
        return mapping[text]
    codes = sessions["session"].map(session_code).to_numpy()
    audit = {"N": expected, "overall_rejection_rate": float(np.mean(labels == 0)),
             "session_used_by_inference": False, "audit_after_predictions_and_csv": True,
             "S1_rejection_target": 0.95, "S2_acceptance_target": 0.98}
    for code in ("S1", "S2"):
        selected_mask = codes == code
        rejection = float(np.mean(labels[selected_mask] == 0))
        audit[f"{code}_N"] = int(selected_mask.sum())
        audit[f"{code}_rejection_rate"] = rejection
        audit[f"{code}_acceptance_rate"] = 1.0 - rejection
    audit["S1_rejection_ge_0.95"] = bool(audit["S1_rejection_rate"] >= 0.95)
    audit["S2_acceptance_ge_0.98"] = bool(audit["S2_acceptance_rate"] >= 0.98)
    result["status"] = "complete_one_shot_test_audit"
    result["test"] = {
        "status": "complete_one_shot_frozen_engineering_audit", "test_accessed": True,
        "predictions_csv": str(PREDICTIONS_PATH), "prediction_rows": expected,
        "single_trial_independent": True, "batch_statistics_or_test_calibration": False,
        "test_labels_read_or_used": False, "metrics": TEST_METRICS,
        "offline_session_audit_after_prediction": audit, "post_audit_threshold_or_variant_change": False,
        "test_elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result, config)
    print(f"E66 Test审计：S1Rej={audit['S1_rejection_rate']:.4f} "
          f"S2Accept={audit['S2_acceptance_rate']:.4f}", flush=True)
    return result


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(result: dict, config: dict) -> None:
    if "summary" not in result:
        return
    clean = result["summary"]["clean_outer_mean"]
    gates = result["test_gates"]
    lines = [
        "# E66：冻结 NeuroShield-FM 四窗嵌入与归一化身份原型",
        "",
        "日期：2026-11-13。目标是检验外部大规模EEG身份预训练能否改善开放集与跨会话表征；不根据Test结果回调阈值或选择变体。",
        "",
        "## 结论与当前 AUC / F1",
        "",
        f"十个inner run按五外层折聚合：**direct C={fmt(clean['direct_C'])}，threshold C={fmt(clean['threshold_C'])}，direct身份F1={fmt(clean['identity_macro_F1_direct'])}，含拒识身份F1={fmt(clean['identity_macro_F1_threshold'])}，pseudo-unknown AUROC={fmt(clean['AUROC'])}，unknown AUPR={fmt(clean['unknown_AUPR'])}，U@95={fmt(clean['U_at_95'])}，工作点U={fmt(clean['working_U'])}**。",
        "",
        f"拟合块身份准确率均值={fmt(clean['identity_fit_accuracy'])}；该值只作流程诊断。E66未运行攻击套件，**Train RA/ASafeR=N/A（未评估）**，不声称Train全部门禁通过。",
        "",
        "## 冻结表征与决策",
        "",
        "- 官方NeuroShield-FM（MIT）权重严格加载并全部冻结；预训练数据源为TUH-EEG、DVS-rsEEG、PREDICT，官方仓库未列BrainPrint。输出每个1秒窗128维L2归一化嵌入。",
        "- 通道顺序固定为P3/C3/F3/Fz/F4/C4/P4/Cz/Fp1/Fp2/T3/T5/O1/O2/F7/F8/T6/T4，3D位置由官方MNE standard_1005接口得到。",
        "- 单条4秒trial内逐通道median/1.4826×MAD稳健归一化（近零时RMS回退），随后CAR；GPU FFT保留1–45Hz并从300Hz/1200点重采样到500Hz/2000点。无跨trial或批次统计。",
        "- BrainPrint幅值单位未确认，因此不采用官方绝对µV质量阈值或±400µV裁剪；这是与官方输入协议的明确偏差。",
        "- 四个不重叠500点窗分别推理，嵌入算术平均后再次L2归一化。身份模型仅为fit≤60%块的类均值归一化原型，无WCCN、无encoder微调。",
        "- 置信度为最大原型余弦相似度；低于clean-known q02输出0，否则输出最近身份。q05仅用于U@95诊断。阈值不使用unknown或Test。",
        "",
        "## Train inner 五外层折均值",
        "",
        "| direct C | threshold C | direct身份F1 | 含拒识身份F1 | 连续块 | AUROC | unknown AUPR | U@95 | 工作点U | 已知接受 | unknown F1 | joint F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(clean['direct_C'])} | {fmt(clean['threshold_C'])} | {fmt(clean['identity_macro_F1_direct'])} | {fmt(clean['identity_macro_F1_threshold'])} | {fmt(clean['continuous_block_accuracy'])} | {fmt(clean['AUROC'])} | {fmt(clean['unknown_AUPR'])} | {fmt(clean['U_at_95'])} | {fmt(clean['working_U'])} | {fmt(clean['known_acceptance'])} | {fmt(clean['unknown_F1'])} | {fmt(clean['joint_F1'])} |",
        "",
        "逐外层折：",
        "",
        "| outer | direct C | threshold C | direct F1 | 含拒识F1 | AUROC | AUPR | U@95 | U | fit准确率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in result["outer_folds"]:
        row = fold["clean"]
        lines.append(f"| {fold['outer_fold']} | {fmt(row['direct_C'])} | {fmt(row['threshold_C'])} | "
                     f"{fmt(row['identity_macro_F1_direct'])} | {fmt(row['identity_macro_F1_threshold'])} | "
                     f"{fmt(row['AUROC'])} | {fmt(row['unknown_AUPR'])} | {fmt(row['U_at_95'])} | "
                     f"{fmt(row['working_U'])} | {fmt(row['identity_fit_accuracy'])} |")
    lines.extend([
        "",
        "## 连续块置信度漂移",
        "",
        "漂移为当前块统计减去同run的fit≤60%统计；不是跨Session漂移。",
        "",
        "| 块 | mean | q02 | q05 | median | mean漂移 | q02漂移 | q05漂移 | median漂移 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for block, title in (("fit", "fit≤60%"), ("calibration", "60–70%"), ("sanity", "70–80%")):
        item = result["summary"]["confidence_drift"][block]
        row, delta = item["run_mean"], item["drift_vs_fit_run_mean"]
        lines.append(f"| {title} | {fmt(row['mean'])} | {fmt(row['q02'])} | {fmt(row['q05'])} | {fmt(row['median'])} | "
                     f"{fmt(delta['mean'])} | {fmt(delta['q02'])} | {fmt(delta['q05'])} | {fmt(delta['median'])} |")
    lines.extend([
        "",
        "## 预注册 Test 门禁",
        "",
        "| 门禁 | 实测 | 最低值 | 结果 |",
        "|---|---:|---:|---|",
    ])
    for name, row in gates["checks"].items():
        lines.append(f"| {name} | {fmt(row['value'])} | {fmt(row['minimum'])} | {'通过' if row['passed'] else '未通过'} |")
    lines.extend(["", f"四项clean门禁全部通过：**{gates['all_clean_passed']}**。", ""])
    if "final20_train" in result:
        final = result["final20_train"]
        lines.extend([
            "## 获准后的 final20 Train 冻结",
            "",
            f"20类原型只用fit≤60%；60–70% clean known校准tau_low={final['tau_low']:.9f}、q05={final['tau95']:.9f}（method=higher）；70–80%仅作sanity。",
            "",
            "| 块 | N | direct C | threshold C | direct F1 | 含拒识F1 | 已知接受率 | 拒识率 | confidence median | median相对fit漂移 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for block in ("fit", "calibration", "sanity"):
            row = final["blocks"][block]
            conf = final["confidence"][block]
            shift = conf.get("drift_vs_fit", {}).get("median", 0.0)
            lines.append(f"| {block} | {row['N']} | {fmt(row['direct_C'])} | {fmt(row['threshold_C'])} | "
                         f"{fmt(row['identity_20class_macro_F1_direct'])} | {fmt(row['identity_20class_macro_F1_threshold'])} | "
                         f"{fmt(row['known_acceptance'])} | {fmt(row['rejection_rate'])} | {fmt(conf['median'])} | {fmt(shift)} |")
        lines.extend(["", f"单条/批次/置换数值与标签不变性：passed={final['numerical_invariance']['passed']}。", ""])
    lines.extend(["## Test 执行状态与工程审计", ""])
    test = result["test"]
    if test["status"] == "complete_one_shot_frozen_engineering_audit":
        audit = test["offline_session_audit_after_prediction"]
        lines.extend([
            f"已按门禁执行一次冻结Test；`{PREDICTIONS_PATH}`共{test['prediction_rows']}行。Session仅在全部预测和CSV落盘后读取。",
            "",
            "| Session | N | 拒识率 | 接受率 | 目标 | 判定 |",
            "|---|---:|---:|---:|---|---|",
            f"| S1 | {audit['S1_N']} | {fmt(audit['S1_rejection_rate'])} | {fmt(audit['S1_acceptance_rate'])} | 拒识≥95% | {'达到' if audit['S1_rejection_ge_0.95'] else '未达到'} |",
            f"| S2 | {audit['S2_N']} | {fmt(audit['S2_rejection_rate'])} | {fmt(audit['S2_acceptance_rate'])} | 接受≥98% | {'达到' if audit['S2_acceptance_ge_0.98'] else '未达到'} |",
            "",
            "审计后未调整阈值、模型或变体。",
        ])
    elif gates["all_clean_passed"]:
        lines.append("四项clean门禁通过，已授权一次冻结Test工程审计；当前报告生成时预测尚未完成。")
    else:
        failed = "、".join(name for name, row in gates["checks"].items() if not row["passed"])
        lines.append(f"**Test=N/A，未读取Test，未生成e66_predictions.csv**。原因：{failed}未达到预注册门禁；没有尝试任何Test变体。")
    lines.extend([
        "",
        "**Test AUC/F1/C/RA/ASafeR/U/Score=N/A（无真实标签）**。S1拒识/S2接受若存在仅为预测工程审计，不是官方真值指标。",
        "",
        "## 证据边界与复现",
        "",
        "- 所有AUC/F1/C/U均来自Train Session1 pseudo-unknown与连续文件块代理，不是跨Session或官方新被试真值证据。",
        "- inner阈值和known指标使用同一60–80%已知连续块；unknown来自互斥被试。final20才把60–70%校准与70–80% sanity分开。",
        "- E66不依赖E58/E65的CNN、logSPD或Test预测；E54仅用于复用被试角色与分层抽样。没有Session、文件顺序、批次身份分布或配对信息进入推理。",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e66.py'}\" --stage smoke",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e66.py'}\" --stage inner",
        "# 仅当四项clean门禁全部通过：",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e66.py'}\" --stage test",
        "```",
        "",
        f"配置：`{CONFIG_PATH}`；结果：`{RESULTS_PATH}`；缓存：`{CACHE}`。断点绑定配置和代码SHA256。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "features", "inner", "test", "all"), default="inner")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E66要求CUDA GPU")
    config = load_config()
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda")
    if args.stage == "smoke":
        run_smoke(config, device)
        return
    if args.stage == "features":
        build_train_embeddings(config, device, not args.no_resume)
        return
    result = None
    if args.stage in ("inner", "all"):
        if args.stage == "all":
            run_smoke(config, device)
        result = run_inner(config, device, not args.no_resume)
    if args.stage == "test" or (args.stage == "all" and result["test_gates"]["all_clean_passed"]):
        run_final20_and_test(config, device)
    print(f"结果：{RESULTS_PATH}\n报告：{REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
