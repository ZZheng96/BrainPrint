"""E39：类数不变的 ECDF 校准与 K=8 → K=20 阈值迁移。

E38 证明图像分支无增益、q=0.05 工作点 U 只有 0.68。E36E 证明 ECDF 映射后的
distance_ratio 从 K=8 内层迁移到 K=16 外层几乎无损。本实验检验同一校准能否迁移到
与官方 Test 一致的 K=20 全注册场景，并检查只用内层 K=8 选出的阈值能否同时满足
U≥0.95 与 C≥0.99。

所有阈值只来自内层 K=8 的已知 OOF 分数；外层 K=16 与 K=20 只做迁移确认。
"""
from __future__ import annotations

import json
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
ZBAND_PATH = ROOT / "BrainPrint_cache" / "e36_pure_eeg" / "train_detrend_zband_cov5.npy"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_metadata(config: dict) -> pd.DataFrame:
    metadata = pd.read_csv(ROOT / config["data"]["train_metadata"])
    if set(metadata["session"]) != {config["data"]["session"]}:
        raise ValueError("E39 只允许 Train session1")
    if not metadata["path"].str.startswith("Train/").all():
        raise ValueError("元数据含非 Train 路径")
    return metadata


class WCCNPrototype:
    def __init__(self, x: torch.Tensor, y: np.ndarray, classes: np.ndarray,
                 shrinkage: float, floor: float):
        self.classes = np.asarray(classes, dtype=int)
        class_to_index = {int(label): index for index, label in enumerate(self.classes)}
        target = torch.as_tensor([class_to_index[int(v)] for v in y], dtype=torch.long, device=x.device)
        means = torch.stack([x[target == index].mean(dim=0) for index in range(len(self.classes))])
        residual = x - means[target]
        covariance = residual.T @ residual / max(len(x) - len(self.classes), 1)
        trace = torch.diagonal(covariance).sum()
        eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * trace / x.shape[1] * eye
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        inverse_root = eigenvalues.clamp_min(floor).rsqrt()
        self.whitening = (eigenvectors * inverse_root.unsqueeze(0)) @ eigenvectors.T
        whitened = means @ self.whitening
        self.prototypes = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        whitened = x @ self.whitening
        normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return normalized @ self.prototypes.T


def predict(model: WCCNPrototype, x: torch.Tensor, indices: np.ndarray, batch: int = 512) -> np.ndarray:
    output = np.empty((len(indices), len(model.classes)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            selected = torch.as_tensor(indices[start:start + batch], device=x.device)
            output[start:start + len(selected)] = model(x[selected]).cpu().numpy()
    return output


def distance_ratio(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
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


def make_record(name: str, model: WCCNPrototype, zband: torch.Tensor,
                known_np: np.ndarray, unknown_np: np.ndarray, subjects: np.ndarray) -> dict:
    known_scores = predict(model, zband, known_np)
    known_raw = distance_ratio(known_scores)
    reference = np.sort(known_raw)
    record = {
        "name": name,
        "K": int(len(model.classes)),
        "classes": model.classes,
        "known_true": subjects[known_np],
        "known_direct": model.classes[known_scores.argmax(1)],
        "known_raw": known_raw,
        "known_u": ecdf_map(reference, known_raw),
        "n_known": int(len(known_np)),
    }
    if unknown_np is not None and len(unknown_np):
        unknown_scores = predict(model, zband, unknown_np)
        unknown_raw = distance_ratio(unknown_scores)
        record.update({
            "unknown_direct": model.classes[unknown_scores.argmax(1)],
            "unknown_raw": unknown_raw,
            "unknown_u": ecdf_map(reference, unknown_raw),
            "n_unknown": int(len(unknown_np)),
        })
    return record


def metrics_at(records: list[dict], key: str, threshold: float) -> dict:
    known_true = np.concatenate([r["known_true"] for r in records])
    known_direct = np.concatenate([r["known_direct"] for r in records])
    known = np.concatenate([r[f"known_{key}"] for r in records])
    classes = np.unique(np.concatenate([r["classes"] for r in records]))
    accepted = known >= threshold
    known_pred = np.where(accepted, known_direct, 0)
    c = float(np.mean(known_pred == known_true))
    result = {
        "threshold": float(threshold),
        "C": c,
        "known_acceptance": float(np.mean(accepted)),
        "identity_macro_f1": float(f1_score(
            known_true, known_pred, labels=classes, average="macro", zero_division=0
        )),
        "n_known": int(len(known)),
    }
    if "unknown_u" in records[0]:
        unknown = np.concatenate([r[f"unknown_{key}"] for r in records])
        unknown_direct = np.concatenate([r["unknown_direct"] for r in records])
        unknown_pred = np.where(unknown >= threshold, unknown_direct, 0)
        u = float(np.mean(unknown_pred == 0))
        detection_true = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
        detection_pred = np.concatenate([accepted, unknown >= threshold])
        result.update({
            "U": u,
            "unknown_f1": float(f1_score(detection_true, detection_pred, pos_label=0, zero_division=0)),
            "C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if min(c, u) > 0 else 0.0,
            "n_unknown": int(len(unknown)),
        })
    return result


def summarize(records: list[dict], key: str, threshold: float) -> dict:
    known = np.concatenate([r[f"known_{key}"] for r in records])
    output = {"work": metrics_at(records, key, threshold)}
    if "unknown_u" in records[0]:
        unknown = np.concatenate([r[f"unknown_{key}"] for r in records])
        output.update(detection_summary(known, unknown))
        output["local_q05_diagnostic"] = metrics_at(
            records, key, float(np.quantile(known, 0.05, method="lower"))
        )
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("E39 要求 CUDA GPU")
    started = time.time()
    torch.manual_seed(20261112)
    np.random.seed(20261112)
    device = torch.device("cuda")
    config = load_config()
    metadata = load_metadata(config)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    if len(zband) != len(metadata):
        raise ValueError("特征行数与元数据不一致")
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(config["protocol"]["outer_seed"]).permutation(all_subjects),
        config["protocol"]["outer_folds"],
    )
    shrinkage = float(config["model"]["wccn_shrinkage"])
    floor = float(config["model"]["wccn_eigenvalue_floor"])

    inner_records, outer_records = [], []
    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        inner_order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
        for inner_index, inner_unknown in enumerate(np.array_split(
            inner_order, config["protocol"]["inner_folds"]
        )):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(enrolled_mask & inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            model = WCCNPrototype(zband[torch.as_tensor(fit_np, device=device)],
                                  subjects[fit_np], enrolled, shrinkage, floor)
            inner_records.append(make_record(
                f"outer{outer_fold}_inner{inner_index + 1}", model, zband,
                known_np, unknown_np, subjects,
            ))
            print(f"inner outer={outer_fold} K={len(enrolled)} fit={len(fit_np)} "
                  f"known={len(known_np)} unknown={len(unknown_np)}", flush=True)
            del model
            torch.cuda.empty_cache()
        fit_np = np.flatnonzero(outer_known & ~forward)
        known_np = np.flatnonzero(outer_known & forward)
        unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = WCCNPrototype(zband[torch.as_tensor(fit_np, device=device)],
                              subjects[fit_np], known_subjects, shrinkage, floor)
        outer_records.append(make_record(
            f"outer{outer_fold}", model, zband, known_np, unknown_np, subjects,
        ))
        print(f"outer {outer_fold} K={len(known_subjects)} fit={len(fit_np)} "
              f"known={len(known_np)} unknown={len(unknown_np)}", flush=True)
        del model
        torch.cuda.empty_cache()

    # 阈值只在 K=8 内层 pooled 已知 OOF 上选择
    inner_known = np.concatenate([r["known_u"] for r in inner_records])
    inner_unknown = np.concatenate([r["unknown_u"] for r in inner_records])
    inner_correct = np.concatenate([
        r["known_direct"] == r["known_true"] for r in inner_records
    ])
    grid = np.unique(np.quantile(inner_known, np.linspace(0.0, 0.05, 101)))
    rows = []
    for threshold in grid:
        accepted = inner_known >= threshold
        c = float(np.mean(inner_correct & accepted))
        u = float(np.mean(inner_unknown < threshold))
        if accepted.mean() >= 0.95 - 1e-12:
            rows.append({
                "threshold": float(threshold), "C": c, "U": u,
                "known_acceptance": float(accepted.mean()),
                "C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if min(c, u) > 0 else 0.0,
            })
    selected = max(rows, key=lambda r: (r["C_U_proxy_not_official"], r["C"], r["U"]))
    threshold = selected["threshold"]
    print(f"selected ECDF threshold={threshold:.6f} inner C={selected['C']:.4f} "
          f"U={selected['U']:.4f}", flush=True)

    inner_summary = summarize(inner_records, "u", threshold)
    outer_summary = summarize(outer_records, "u", threshold)

    # K=20 全注册连续块 OOF：只有 C，没有 held-out unknown
    fit20 = np.flatnonzero(~forward)
    oof20 = np.flatnonzero(forward)
    model20 = WCCNPrototype(zband[torch.as_tensor(fit20, device=device)],
                            subjects[fit20], all_subjects, shrinkage, floor)
    record20 = make_record("k20_oof", model20, zband, oof20, None, subjects)
    k20 = metrics_at([record20], "u", threshold)
    k20["direct_C"] = float(np.mean(record20["known_direct"] == record20["known_true"]))
    k20["identity_macro_f1_direct"] = float(f1_score(
        record20["known_true"], record20["known_direct"], labels=all_subjects,
        average="macro", zero_division=0,
    ))
    k20["unknown_metrics"] = "N/A: 20 类全注册 OOF 没有 held-out unknown"
    del model20
    torch.cuda.empty_cache()

    targets = config["targets"]
    gates = {
        "inner_direct_C": bool(inner_summary["work"]["C"] >= targets["direct_C_min"]),
        "inner_unknown_auroc": bool(inner_summary["unknown_auroc"] >= targets["unknown_auroc_min"]),
        "inner_U_at_known95": bool(inner_summary["work"]["U"] >= targets["U_at_known95_min"]),
        "outer_direct_C": bool(outer_summary["work"]["C"] >= targets["direct_C_min"]),
        "outer_unknown_auroc": bool(outer_summary["unknown_auroc"] >= targets["unknown_auroc_min"]),
        "outer_U_at_known95": bool(outer_summary["work"]["U"] >= targets["U_at_known95_min"]),
        "k20_direct_C": bool(k20["direct_C"] >= targets["k20_direct_C_min"]),
        "k20_identity_macro_f1": bool(k20["identity_macro_f1_direct"] >= targets["k20_identity_macro_f1_min"]),
    }
    result = {
        "status": "complete", "experiment": "E39", "date": "2026-11-13",
        "data_evidence": "Train session1 only; pseudo-unknown and continuous file-block proxy; Test not accessed",
        "selected_threshold": selected,
        "inner_K8": inner_summary,
        "outer_K16": outer_summary,
        "k20_oof": k20,
        "transfer_delta_K16_minus_K8": {
            key: float(outer_summary["work"][key] - inner_summary["work"][key])
            for key in ("C", "U", "known_acceptance", "identity_macro_f1")
        },
        "gates": gates, "all_gates_passed": bool(all(gates.values())),
        "test_metrics": "N/A: Test not read",
        "elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    print(json.dumps({"inner": inner_summary["work"], "outer": outer_summary["work"],
                      "k20": {k: k20[k] for k in ("C", "direct_C", "identity_macro_f1_direct")},
                      "gates": gates}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
