"""训练折内原始 EEG 对抗增强，仅重拟合任务模型的 Ridge 组件。"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch

from src.open_set_improvement.core import TaskSpecificModel, TorchRidge
from src.raw_eeg_defense.gpu_attack import GPUCovExtractor, raw_pgd
from src.raw_eeg_defense.run import extract_view


def refit_ridge_components(
    model: TaskSpecificModel,
    x: np.ndarray,
    window: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fit_mask: np.ndarray,
    adversarial_indices: np.ndarray,
    adversarial_full: np.ndarray,
    adversarial_window: np.ndarray,
) -> dict:
    """clean 全量加训练折内 adv；近邻组件与攻击尺度保持 clean 拟合结果。"""
    adversarial_indices = np.asarray(adversarial_indices, dtype=int)
    if adversarial_indices.ndim != 1 or not len(adversarial_indices):
        raise ValueError("adversarial_indices 必须是一维非空数组")
    if len(np.unique(adversarial_indices)) != len(adversarial_indices):
        raise ValueError("adversarial_indices 不能重复")
    if not np.all(fit_mask[adversarial_indices]):
        raise ValueError("对抗增强样本必须来自当前拟合折")
    if adversarial_full.shape != (len(adversarial_indices), x.shape[1]):
        raise ValueError("adversarial_full 形状与增强索引不一致")
    if adversarial_window.shape != (len(adversarial_indices), 2, x.shape[1]):
        raise ValueError("adversarial_window 形状与增强索引不一致")

    diagnostics = {"clean": {}, "adversarial": {}}
    for task, models in model.models.items():
        clean_indices = np.flatnonzero(fit_mask & (tasks == task))
        adversarial_positions = np.flatnonzero(tasks[adversarial_indices] == task)
        if not len(clean_indices) or not len(adversarial_positions):
            raise ValueError(f"任务 {task} 缺少 clean 或 adv 训练样本")
        clean_full = np.asarray(x[clean_indices])
        clean_window = np.asarray(window[clean_indices]).reshape(-1, x.shape[1])
        clean_targets = subjects[clean_indices]
        adv_full = np.asarray(adversarial_full[adversarial_positions])
        adv_window = np.asarray(adversarial_window[adversarial_positions]).reshape(-1, x.shape[1])
        adv_targets = subjects[adversarial_indices[adversarial_positions]]

        full = np.concatenate([clean_full, adv_full])
        full_targets = np.concatenate([clean_targets, adv_targets])
        windows = np.concatenate([clean_window, adv_window])
        window_targets = np.concatenate([np.repeat(clean_targets, 2), np.repeat(adv_targets, 2)])
        models.full = TorchRidge(model.device, model.alpha).fit(
            full, full_targets, model.classes
        )
        models.window = TorchRidge(model.device, model.alpha).fit(
            windows, window_targets, model.classes
        )
        models.low = TorchRidge(model.device, model.alpha).fit(
            full[:, :342], full_targets, model.classes
        )
        diagnostics["clean"][task] = int(len(clean_indices))
        diagnostics["adversarial"][task] = int(len(adversarial_positions))
    return diagnostics


def fit_adversarial_ridge(
    clean_model: TaskSpecificModel,
    x: np.ndarray,
    window: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fit_mask: np.ndarray,
    adversarial_indices: np.ndarray,
    adversarial_full: np.ndarray,
    adversarial_window: np.ndarray,
) -> tuple[TaskSpecificModel, dict]:
    """保留 clean 基线不变，返回独立、确定性的对抗 Ridge 模型。"""
    robust = TaskSpecificModel(clean_model.device, clean_model.alpha).fit(
        x, window, subjects, tasks, fit_mask
    )
    diagnostics = refit_ridge_components(
        robust, x, window, subjects, tasks, fit_mask,
        adversarial_indices, adversarial_full, adversarial_window,
    )
    return robust, diagnostics


def adversarial_original_features(
    model: TaskSpecificModel,
    extractor: GPUCovExtractor,
    raw: np.ndarray,
    tasks: np.ndarray,
    targets: np.ndarray,
    snr_db: float,
    targeted: bool,
    steps: int,
    seed: int,
    pool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """GPU 原始波形 PGD 后，以既有 SciPy 提取器精确重提原始视图特征。"""
    raw = np.asarray(raw, dtype=np.float32)
    tasks = np.asarray(tasks)
    targets = np.asarray(targets)
    if raw.ndim != 3 or len(raw) != len(tasks) or len(raw) != len(targets):
        raise ValueError("raw、tasks、targets 的样本数必须一致")
    adversarial = np.empty_like(raw)
    surrogate_success = []
    for task in sorted(np.unique(tasks)):
        positions = np.flatnonzero(tasks == task)
        for start in range(0, len(positions), batch_size):
            selected = positions[start : start + batch_size]
            data = torch.as_tensor(raw[selected], device=model.device)
            attack_targets = targets[selected]
            if targeted:
                class_to_index = {
                    int(label): index for index, label in enumerate(model.classes)
                }
                target_indices = np.asarray([
                    (class_to_index[int(label)] + 1) % len(model.classes)
                    for label in attack_targets
                ])
                attack_targets = model.classes[target_indices]
            adv, diagnostics = raw_pgd(
                model, extractor, data, task, attack_targets, snr_db,
                targeted=targeted, steps=steps,
                seed=seed + start + int(snr_db) * 1000 + int(targeted) * 100000,
            )
            adversarial[selected] = adv.cpu().numpy()
            surrogate_scores = diagnostics["final_scores"]
            if isinstance(surrogate_scores, torch.Tensor):
                surrogate_scores = surrogate_scores.cpu().numpy()
            predicted = model.classes[np.asarray(surrogate_scores).argmax(axis=1)]
            surrogate_success.append(predicted == targets[selected])
            print(
                f"raw PGD snr={snr_db:g} targeted={targeted} task={task}: "
                f"{min(start + batch_size, len(positions))}/{len(positions)}",
                flush=True,
            )

    centered = raw - raw.mean(axis=1, keepdims=True)
    delta = adversarial - raw
    signal_norm = np.linalg.norm(centered.reshape(len(raw), -1), axis=1)
    delta_norm = np.linalg.norm(delta.reshape(len(raw), -1), axis=1)
    actual_snr = 20 * np.log10(signal_norm / np.maximum(delta_norm, 1e-12))
    adv_full = np.empty((len(raw), model.models[next(iter(model.models))].full.mean.numel()), dtype=np.float32)
    adv_window = np.empty((len(raw), 2, adv_full.shape[1]), dtype=np.float32)
    extract = partial(extract_view, view="original")
    for index, (full, window) in enumerate(pool.map(extract, adversarial, chunksize=4), 1):
        adv_full[index - 1] = full
        adv_window[index - 1] = window
        if index % 250 == 0 or index == len(raw):
            print(f"exact original features: {index}/{len(raw)}", flush=True)
    diagnostics = {
        "actual_snr_min": float(actual_snr.min()),
        "actual_snr_median": float(np.median(actual_snr)),
        "actual_snr_max": float(actual_snr.max()),
        "surrogate_identity_retention": float(np.mean(np.concatenate(surrogate_success))),
        "evaluation_features": "existing SciPy extractor re-applied to attacked raw EEG",
        "snr_db": float(snr_db),
        "targeted": bool(targeted),
        "steps": int(steps),
    }
    return adv_full, adv_window, diagnostics
