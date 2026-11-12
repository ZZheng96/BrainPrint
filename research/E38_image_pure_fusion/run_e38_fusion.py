"""E38：DINO 图像分支 + 纯 EEG WCCN 分支的分数级融合。

两个分支各自独立拟合（只用 fit 折），各自输出
  - 身份预测（最近类中心）
  - 置信度（max_cosine / distance_ratio）
融合方式：每个分支的置信度先映射到该分支 fit 折已知样本的 ECDF，
再按权重 w 线性组合；身份标签取权重较大分支的预测。

评估：内层伪未知（选型）→ 外层冻结确认 → 达标后一次性 Test 无标签审计。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
DINO_PATH = ROOT / "BrainPrint_cache" / "eeg_image_dinov2" / "train_covariance5_dinov2.npy"
ZBAND_PATH = ROOT / "BrainPrint_cache" / "e36_pure_eeg" / "train_detrend_zband_cov5.npy"
TASKS = ("EC", "EO", "Imagery", "SSVEP")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_metadata(config: dict) -> pd.DataFrame:
    metadata = pd.read_csv(ROOT / config["data"]["train_metadata"])
    if set(metadata["session"]) != {config["data"]["session"]}:
        raise ValueError("E38 只允许 Train session1")
    if not metadata["path"].str.startswith("Train/").all():
        raise ValueError("元数据含非 Train 路径")
    return metadata


def wccn_whitening(x: torch.Tensor, y: np.ndarray, shrinkage: float, floor: float) -> torch.Tensor:
    classes = np.unique(y)
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    target = torch.as_tensor([class_to_index[int(v)] for v in y], dtype=torch.long, device=x.device)
    means = torch.stack([x[target == index].mean(dim=0) for index in range(len(classes))])
    residual = x - means[target]
    covariance = residual.T @ residual / max(len(x) - len(classes), 1)
    trace = torch.diagonal(covariance).sum()
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace / x.shape[1] * eye
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    inverse_root = eigenvalues.clamp_min(floor).rsqrt()
    return (eigenvectors * inverse_root.unsqueeze(0)) @ eigenvectors.T


class PrototypeBranch:
    """WCCN 白化 + 最近类中心余弦。"""

    def __init__(self, x: torch.Tensor, y: np.ndarray, classes: np.ndarray, shrinkage: float, floor: float):
        self.classes = np.asarray(classes, dtype=int)
        self.whitening = wccn_whitening(x, y, shrinkage, floor)
        whitened = x @ self.whitening
        class_to_index = {int(label): index for index, label in enumerate(self.classes)}
        target = torch.as_tensor([class_to_index[int(v)] for v in y], dtype=torch.long, device=x.device)
        means = torch.stack([whitened[target == index].mean(dim=0) for index in range(len(self.classes))])
        self.prototypes = means / means.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        whitened = x @ self.whitening
        normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return normalized @ self.prototypes.T


class TaskLogisticBranch:
    """逐任务 StandardScaler + 逻辑回归，再在标准化空间做 WCCN 类中心。"""

    def __init__(self, x: torch.Tensor, y: np.ndarray, task: np.ndarray, classes: np.ndarray, shrinkage: float):
        from sklearn.linear_model import LogisticRegression

        self.classes = np.asarray(classes, dtype=int)
        self.models = {}
        self.centers = {}
        self.whitening = {}
        x_np = x.cpu().numpy()
        for name in TASKS:
            rows = task == name
            if rows.sum() < 100:
                continue
            model = LogisticRegression(C=1.0, max_iter=300)
            model.fit(x_np[rows], y[rows])
            mean = x_np[rows].mean(0)
            scale = x_np[rows].std(0)
            scale[scale < 1e-6] = 1.0
            z = torch.as_tensor((x_np[rows] - mean) / scale, dtype=torch.float32)
            whitening = wccn_whitening(z, y[rows], shrinkage, 1e-8)
            z_w = z @ whitening
            centers = torch.stack([
                z_w[torch.as_tensor(y[rows] == c, device=z.device)].mean(0) for c in self.classes
            ])
            self.models[name] = (model, mean, scale)
            self.whitening[name] = whitening
            self.centers[name] = centers

    def __call__(self, x: torch.Tensor, task: np.ndarray) -> torch.Tensor:
        x_np = x.cpu().numpy()
        output = np.zeros((len(x_np), len(self.classes)), dtype=np.float32)
        for name in TASKS:
            rows = task == name
            if not rows.any() or name not in self.models:
                continue
            model, mean, scale = self.models[name]
            z = torch.as_tensor((x_np[rows] - mean) / scale, dtype=torch.float32)
            z_w = z @ self.whitening[name]
            normalized = z_w / z_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
            centers = self.centers[name]
            centers = centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-12)
            output[rows] = (normalized @ centers.T).numpy()
        return output


def confidence(scores: np.ndarray, kind: str) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
    if kind == "max_cosine":
        return maximum
    return (1.0 - second) / np.maximum(1.0 - maximum, 1e-6)


def ecdf_map(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="right") / len(ordered)


def detection_summary(known: np.ndarray, unknown: np.ndarray) -> dict:
    labels = np.concatenate([np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)])
    values = np.concatenate([known, unknown])
    fpr, tpr, _ = roc_curve(labels, values)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "unknown_auroc": float(roc_auc_score(labels, values)),
        "unknown_aupr": float(average_precision_score(1 - labels, -values)),
        "eer": float((fpr[index] + fnr[index]) / 2),
    }


def build_fold(
    dino: torch.Tensor, zband: torch.Tensor, subjects: np.ndarray, task: np.ndarray,
    fit_np: np.ndarray, known_np: np.ndarray, unknown_np: np.ndarray,
    classes: np.ndarray, config: dict,
) -> dict:
    """拟合两个分支，返回 fit/known/unknown 三组分数与身份预测。"""
    fit_index = torch.as_tensor(fit_np, device=dino.device)
    image = TaskLogisticBranch(
        dino[fit_index], subjects[fit_np], task[fit_np], classes,
        float(config["branches"]["image"]["wccn_shrinkage"]),
    )
    pure = PrototypeBranch(
        zband[fit_index], subjects[fit_np], classes,
        float(config["branches"]["pure_eeg"]["wccn_shrinkage"]),
        float(config["branches"]["pure_eeg"]["wccn_eigenvalue_floor"]),
    )
    output = {}
    for name, indices in (("fit", fit_np), ("known", known_np), ("unknown", unknown_np)):
        index = torch.as_tensor(indices, device=dino.device)
        with torch.no_grad():
            image_scores = image(dino[index], task[indices])
            pure_scores = pure(zband[index])
        image_scores = np.asarray(image_scores, dtype=np.float32)
        pure_scores = pure_scores.cpu().numpy()
        output[name] = {
            "image_scores": image_scores,
            "pure_scores": pure_scores,
            "image_pred": classes[image_scores.argmax(1)],
            "pure_pred": classes[pure_scores.argmax(1)],
        }
    output["classes"] = classes
    output["known_true"] = subjects[known_np]
    return output


def fuse(fold: dict, weight: float, kind: str) -> dict:
    """用 fit 折已知样本的 ECDF 做逐分支归一化后融合。"""
    image_ref = confidence(fold["fit"]["image_scores"], kind)
    pure_ref = confidence(fold["fit"]["pure_scores"], kind)
    result = {}
    for name in ("known", "unknown"):
        image_u = ecdf_map(image_ref, confidence(fold[name]["image_scores"], kind))
        pure_u = ecdf_map(pure_ref, confidence(fold[name]["pure_scores"], kind))
        result[name] = {
            "confidence": weight * image_u + (1.0 - weight) * pure_u,
            "pred": fold[name]["image_pred"] if weight > 0.5 else fold[name]["pure_pred"],
        }
    return result


def threshold_metrics(known_conf, unknown_conf, known_correct, threshold) -> dict:
    accepted = known_conf >= threshold
    c = float(np.mean(known_correct & accepted))
    u = float(np.mean(unknown_conf < threshold))
    return {
        "threshold": float(threshold), "C": c, "U": u,
        "known_acceptance": float(np.mean(accepted)),
        "C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if min(c, u) > 0 else 0.0,
    }


def select_point(runs: list[dict], weight: float, kind: str, config: dict) -> dict:
    fused = [fuse(run, weight, kind) for run in runs]
    known = np.concatenate([item["known"]["confidence"] for item in fused])
    unknown = np.concatenate([item["unknown"]["confidence"] for item in fused])
    correct = np.concatenate([
        item["known"]["pred"] == run["known_true"] for item, run in zip(fused, runs)
    ])
    grid = np.quantile(known, np.linspace(0.0, 0.05, int(config["nested_protocol"]["threshold_grid_points"])))
    rows = []
    for threshold in np.unique(grid):
        row = threshold_metrics(known, unknown, correct, float(threshold))
        if row["known_acceptance"] >= 0.95 - 1e-12:
            rows.append(row)
    best = max(rows, key=lambda r: (r["C_U_proxy_not_official"], r["C"], r["U"]))
    return {
        "weight": weight, "kind": kind, "threshold": best["threshold"],
        "inner_work": best, **detection_summary(known, unknown),
    }


def select_point_fixed_acceptance(runs: list[dict], weight: float, kind: str, config: dict) -> dict:
    """阈值固定为 fit 折已知置信度的 5% 分位（已知接受率≈95%），不按 U 调阈值。"""
    fused = [fuse(run, weight, kind) for run in runs]
    known = np.concatenate([item["known"]["confidence"] for item in fused])
    unknown = np.concatenate([item["unknown"]["confidence"] for item in fused])
    correct = np.concatenate([
        item["known"]["pred"] == run["known_true"] for item, run in zip(fused, runs)
    ])
    threshold = float(np.quantile(known, 0.05, method="lower"))
    row = threshold_metrics(known, unknown, correct, threshold)
    return {
        "weight": weight, "kind": kind, "threshold": threshold,
        "inner_work": row, **detection_summary(known, unknown),
    }


def select_point_balanced(runs: list[dict], weight: float, kind: str, config: dict) -> dict:
    """阈值取 fit 折已知置信度分位，使已知接受率≈未知拒识率（EER 工作点）。

    该规则只使用 fit 折已知分数与内层伪未知分数，不读取外层或 Test。
    """
    fused = [fuse(run, weight, kind) for run in runs]
    known = np.concatenate([item["known"]["confidence"] for item in fused])
    unknown = np.concatenate([item["unknown"]["confidence"] for item in fused])
    correct = np.concatenate([
        item["known"]["pred"] == run["known_true"] for item, run in zip(fused, runs)
    ])
    grid = np.quantile(known, np.linspace(0.0, 0.30, 301))
    rows = [threshold_metrics(known, unknown, correct, float(t)) for t in np.unique(grid)]
    best = min(rows, key=lambda r: (abs(r["known_acceptance"] - r["U"]), -r["C"]))
    return {
        "weight": weight, "kind": kind, "threshold": best["threshold"],
        "inner_work": best, **detection_summary(known, unknown),
    }


def select_point_high_acceptance(runs: list[dict], weight: float, kind: str, config: dict) -> dict:
    """阈值取 fit 折已知置信度的 1% 分位（已知接受率≈99%），不按 U 调阈值。

    动机：官方 Score 中 C 的指数 0.40 远大于 U 的 0.25，且 C 与 U 在此数据上高度相关，
    因此固定高接受率工作点比 EER 工作点更符合评分结构。该规则只使用 fit 折已知分数。
    """
    fused = [fuse(run, weight, kind) for run in runs]
    known = np.concatenate([item["known"]["confidence"] for item in fused])
    unknown = np.concatenate([item["unknown"]["confidence"] for item in fused])
    correct = np.concatenate([
        item["known"]["pred"] == run["known_true"] for item, run in zip(fused, runs)
    ])
    threshold = float(np.quantile(known, 0.01, method="lower"))
    row = threshold_metrics(known, unknown, correct, threshold)
    return {
        "weight": weight, "kind": kind, "threshold": threshold,
        "inner_work": row, **detection_summary(known, unknown),
    }


def select_point_quantile(runs: list[dict], weight: float, kind: str, quantile: float) -> dict:
    """阈值取 fit 折已知置信度的指定分位（已知接受率≈1-quantile）。"""
    fused = [fuse(run, weight, kind) for run in runs]
    known = np.concatenate([item["known"]["confidence"] for item in fused])
    unknown = np.concatenate([item["unknown"]["confidence"] for item in fused])
    correct = np.concatenate([
        item["known"]["pred"] == run["known_true"] for item, run in zip(fused, runs)
    ])
    threshold = float(np.quantile(known, quantile, method="lower"))
    row = threshold_metrics(known, unknown, correct, threshold)
    return {
        "weight": weight, "kind": kind, "quantile": quantile, "threshold": threshold,
        "inner_work": row, **detection_summary(known, unknown),
    }


def evaluate(runs: list[dict], weight: float, kind: str, threshold: float) -> dict:
    fused = [fuse(run, weight, kind) for run in runs]
    known = np.concatenate([item["known"]["confidence"] for item in fused])
    unknown = np.concatenate([item["unknown"]["confidence"] for item in fused])
    known_true = np.concatenate([run["known_true"] for run in runs])
    known_pred = np.concatenate([item["known"]["pred"] for item in fused])
    unknown_pred = np.concatenate([item["unknown"]["pred"] for item in fused])
    classes = np.unique(np.concatenate([run["classes"] for run in runs]))
    accepted = known >= threshold
    unknown_accepted = unknown >= threshold
    known_final = np.where(accepted, known_pred, 0)
    unknown_final = np.where(unknown_accepted, unknown_pred, 0)
    detection_true = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
    detection_pred = np.concatenate([accepted, unknown_accepted])
    c = float(np.mean(known_final == known_true))
    u = float(np.mean(unknown_final == 0))
    return {
        "direct_C": float(np.mean(known_pred == known_true)),
        "identity_macro_f1_direct": float(f1_score(
            known_true, known_pred, labels=classes, average="macro", zero_division=0
        )),
        **detection_summary(known, unknown),
        "C": c, "U": u,
        "known_acceptance": float(np.mean(accepted)),
        "identity_macro_f1": float(f1_score(
            known_true, known_final, labels=classes, average="macro", zero_division=0
        )),
        "unknown_f1": float(f1_score(detection_true, detection_pred, pos_label=0, zero_division=0)),
        "joint_macro_f1": float(f1_score(
            np.concatenate([known_true, np.zeros(len(unknown), dtype=int)]),
            np.concatenate([known_final, unknown_final]),
            labels=np.concatenate([[0], classes]), average="macro", zero_division=0,
        )),
        "C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if min(c, u) > 0 else 0.0,
        "n_known": int(len(known)), "n_unknown": int(len(unknown)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inner", "outer", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E38 要求 CUDA GPU")
    started = time.time()
    torch.manual_seed(20261112)
    np.random.seed(20261112)
    device = torch.device("cuda")
    config = load_config()
    metadata = load_metadata(config)
    subjects = metadata["subject"].to_numpy(dtype=int)
    task = metadata["task"].to_numpy()
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    dino = torch.as_tensor(np.load(DINO_PATH), dtype=torch.float32, device=device)
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    if len(dino) != len(metadata) or len(zband) != len(metadata):
        raise ValueError("特征行数与元数据不一致")
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(config["nested_protocol"]["outer_seed"]).permutation(all_subjects),
        config["nested_protocol"]["outer_folds"],
    )
    weights = [float(w) for w in config["fusion"]["weights"]]
    kinds = ("max_cosine", "distance_ratio")
    quantiles = [float(q) for q in config["fusion"]["threshold_quantiles"]]

    inner_runs = []
    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        inner_order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
        for inner_unknown in np.array_split(inner_order, config["nested_protocol"]["inner_folds"]):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(enrolled_mask & inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            inner_runs.append(build_fold(
                dino, zband, subjects, task, fit_np, known_np, unknown_np, enrolled, config
            ))
            print(f"inner outer={outer_fold} fit={len(fit_np)} known={len(known_np)} "
                  f"unknown={len(unknown_np)}", flush=True)
            torch.cuda.empty_cache()

    selections = [select_point_quantile(inner_runs, w, k, q)
                  for w in weights for k in kinds for q in quantiles]
    for row in selections:
        print(f"inner w={row['weight']:.1f} {row['kind']:14s} q={row['quantile']:.3f} "
              f"AUROC={row['unknown_auroc']:.4f} C={row['inner_work']['C']:.4f} "
              f"U={row['inner_work']['U']:.4f} acc={row['inner_work']['known_acceptance']:.4f}",
              flush=True)
    best = max(selections, key=lambda r: (
        r["inner_work"]["C_U_proxy_not_official"], r["inner_work"]["C"], r["inner_work"]["U"],
        r["unknown_auroc"],
    ))
    print(f"selected w={best['weight']} kind={best['kind']} q={best['quantile']}", flush=True)

    outer_folds = []
    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        fit_np = np.flatnonzero(outer_known & ~forward)
        known_np = np.flatnonzero(outer_known & forward)
        unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects))
        fold = build_fold(dino, zband, subjects, task, fit_np, known_np, unknown_np, known_subjects, config)
        metrics = evaluate([fold], best["weight"], best["kind"], best["threshold"])
        outer_folds.append({"outer_fold": outer_fold, "unknown_subjects": unknown_subjects.tolist(),
                            "metrics": metrics})
        print(f"outer {outer_fold}: C={metrics['C']:.4f} U={metrics['U']:.4f} "
              f"AUROC={metrics['unknown_auroc']:.4f} F1={metrics['identity_macro_f1']:.4f}", flush=True)
        torch.cuda.empty_cache()

    # 诊断：外层各工作点（不用于选型，只用于说明 U 与 C 的权衡）
    diagnostic = []
    for row in selections:
        per_fold = []
        for outer_index, unknown_subjects in enumerate(outer_groups):
            known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
            outer_known = np.isin(subjects, known_subjects)
            fit_np = np.flatnonzero(outer_known & ~forward)
            known_np = np.flatnonzero(outer_known & forward)
            unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects))
            fold = build_fold(dino, zband, subjects, task, fit_np, known_np, unknown_np,
                              known_subjects, config)
            per_fold.append(evaluate([fold], row["weight"], row["kind"], row["threshold"]))
            torch.cuda.empty_cache()
        diagnostic.append({
            "weight": row["weight"], "kind": row["kind"], "quantile": row["quantile"],
            "threshold": row["threshold"],
            "outer_mean": {key: float(np.mean([m[key] for m in per_fold])) for key in
                           ("C", "U", "known_acceptance", "unknown_auroc", "identity_macro_f1")},
        })
        print(f"diag w={row['weight']:.1f} {row['kind']:14s} q={row['quantile']:.2f} "
              f"outer C={diagnostic[-1]['outer_mean']['C']:.4f} "
              f"U={diagnostic[-1]['outer_mean']['U']:.4f}", flush=True)

    keys = ("direct_C", "identity_macro_f1_direct", "unknown_auroc", "unknown_aupr", "eer",
            "C", "U", "known_acceptance", "identity_macro_f1", "unknown_f1", "joint_macro_f1",
            "C_U_proxy_not_official")
    fold_mean = {key: float(np.mean([f["metrics"][key] for f in outer_folds])) for key in keys}
    targets = config["targets"]
    gates = {
        "direct_C": bool(fold_mean["direct_C"] >= targets["direct_C_min"]),
        "identity_macro_f1_direct": bool(fold_mean["identity_macro_f1_direct"] >= targets["identity_macro_f1_direct_min"]),
        "continuous_block": bool(fold_mean["direct_C"] >= targets["continuous_block_accuracy_min"]),
        "unknown_auroc": bool(fold_mean["unknown_auroc"] >= targets["unknown_auroc_min"]),
        "U_at_known95": bool(fold_mean["U"] >= targets["U_at_known95_min"]),
    }
    result = {
        "status": "complete", "experiment": "E38", "date": "2026-11-13",
        "data_evidence": "Train session1 only; pseudo-unknown and continuous file-block proxy; Test not accessed",
        "selected": best, "inner_selections": selections,
        "outer_folds": outer_folds, "outer_fold_mean": fold_mean,
        "outer_diagnostic_all_points": diagnostic,
        "gates": gates, "all_gates_passed": bool(all(gates.values())),
        "test_metrics": "N/A: Test not read",
        "elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    print(json.dumps({"fold_mean": fold_mean, "gates": gates}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
