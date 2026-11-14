"""E75: frozen EEG-image template recognition under the E74 nested protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

E75_DIR = Path(__file__).resolve().parent
ROOT = E75_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "E73_external_multisession"))
from external_data import load_external_data, subject_folds  # noqa: E402
from src.eeg_image_dinov2.backbone import extract_features  # noqa: E402
from src.eeg_image_dinov2.image_encoders import encode  # noqa: E402

CONFIG_PATH = E75_DIR / "config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def confidence(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] < 2:
        return scores[:, 0]
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def image_batch(raw: torch.Tensor, indices: np.ndarray, method: str) -> np.ndarray:
    images = [encode(raw[int(index)].detach().cpu().numpy().T, method) for index in indices]
    return np.stack(images).astype(np.float32)


def load_or_extract(data, config: dict, feature_path: Path, force: bool) -> tuple[np.ndarray, dict]:
    source_hashes = {
        "config": sha256(CONFIG_PATH),
        "run_e75.py": sha256(Path(__file__).resolve()),
        "external_data.py": sha256(ROOT / "research" / "E73_external_multisession" / "external_data.py"),
        "image_encoders.py": sha256(ROOT / "src" / "eeg_image_dinov2" / "image_encoders.py"),
        "backbone.py": sha256(ROOT / "src" / "eeg_image_dinov2" / "backbone.py"),
    }
    metadata_path = feature_path.with_suffix(".json")
    if feature_path.is_file() and metadata_path.is_file() and not force:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved.get("source_hashes") != source_hashes:
            raise ValueError("Feature cache source hash mismatch; use --force deliberately")
        return np.load(feature_path, mmap_mode="r"), source_hashes
    indices = np.arange(len(data.subject), dtype=int)
    method = config["image"]["encoder"]
    images = image_batch(data.raw, indices, method)
    device = data.raw.device
    features = extract_features(images, device, int(config["image"]["batch_size"]), config["image"]["backbone"], int(config["image"]["img_size"]))
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, features.astype(np.float32))
    metadata_path.write_text(json.dumps({"source_hashes": source_hashes, "shape": list(features.shape), "encoder": method}, indent=2), encoding="utf-8")
    return features, source_hashes


def fit_fold(features: np.ndarray, data, roles: dict, config: dict, fold: int) -> dict:
    enrolled = roles["enrolled"]
    protocol = config["protocol"]
    train_index = data.indices(enrolled, session="T", runs=protocol["train_runs"])
    template_index = data.indices(enrolled, session="T", runs=[protocol["template_run"]])
    calibration_known = data.indices(enrolled, session="E", runs=[protocol["calibration_known_query_run"]])
    calibration_unknown = data.indices(roles["calibration_unknown"], session="E", runs=[protocol["calibration_unknown_query_run"]])
    outer_known = data.indices(enrolled, session="E", runs=[protocol["evaluation_query_run"]])
    outer_unknown = data.indices(roles["outer_unknown"], session="E", runs=[protocol["outer_unknown_query_run"]])
    if any(len(index) == 0 for index in (train_index, template_index, calibration_known, calibration_unknown, outer_known, outer_unknown)):
        raise ValueError(f"Empty E75 fold role at fold {fold}")

    mean = np.asarray(features[train_index]).mean(axis=0)
    std = np.asarray(features[train_index]).std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)

    def transform(index):
        value = (np.asarray(features[index], dtype=np.float32) - mean) / std
        return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)

    template = transform(template_index)
    prototypes = np.stack([template[data.subject[template_index] == subject].mean(axis=0) for subject in enrolled])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)

    def scores(index):
        return transform(index) @ prototypes.T

    calibration_scores = scores(calibration_known)
    threshold = float(np.quantile(confidence(calibration_scores), 0.05))
    known_scores = scores(outer_known)
    unknown_scores = scores(outer_unknown)
    known_conf = confidence(known_scores)
    unknown_conf = confidence(unknown_scores)
    known_pred = enrolled[known_scores.argmax(axis=1)]
    unknown_pred = enrolled[unknown_scores.argmax(axis=1)]
    accepted_known = known_conf >= threshold
    accepted_unknown = unknown_conf >= threshold
    detection_true = np.r_[np.ones(len(known_conf), dtype=int), np.zeros(len(unknown_conf), dtype=int)]
    detection_score = np.r_[known_conf, unknown_conf]
    detection_pred = detection_score >= threshold
    identity_true = np.r_[data.subject[outer_known], np.zeros(len(outer_unknown), dtype=int)]
    identity_pred = np.r_[np.where(accepted_known, known_pred, 0), np.where(accepted_unknown, unknown_pred, 0)]
    return {
        "outer_fold": fold,
        "roles": {key: values.tolist() for key, values in roles.items()},
        "N": {"train": len(train_index), "template": len(template_index), "calibration_known": len(calibration_known), "calibration_unknown": len(calibration_unknown), "outer_known": len(outer_known), "outer_unknown": len(outer_unknown)},
        "threshold": threshold,
        "C": float(np.mean(known_pred == data.subject[outer_known])),
        "identity_macro_F1": float(f1_score(identity_true, identity_pred, labels=np.r_[enrolled, 0], average="macro", zero_division=0)),
        "unknown_F1": float(f1_score(detection_true, detection_pred, zero_division=0)),
        "detection_AUROC": float(roc_auc_score(detection_true, detection_score)),
        "known_acceptance": float(np.mean(accepted_known)),
        "unknown_rejection": float(np.mean(~accepted_unknown)),
        "checkpoint": None,
    }


def write_report(results: dict, path: Path) -> None:
    summary = results["summary"]
    path.write_text("\n".join([
        "# E75 冻结 EEG 图像模板实验", "", "## 协议", "",
        "- 数据：E73 派生的 BCIC IV-2a 外部数据；训练 run 0–2，模板 run 3，标定 run 4，评估 run 5。",
        "- 三折 enrolled / calibration-unknown / outer-unknown 被试隔离；unknown 不参与训练、模型选择或阈值选择。",
        "- 图像：现有 `topomap5` 编码器 + 冻结 DINOv2 ViT-B/14；未访问竞赛 Test。", "",
        "## 三折均值", "", "| 指标 | 数值 |", "|---|---:|",
        f"| Detection AUC | {summary['detection_AUROC']:.6f} |",
        f"| Unknown F1 | {summary['unknown_F1']:.6f} |",
        f"| Identity macro-F1 | {summary['identity_macro_F1']:.6f} |",
        f"| Known identity accuracy | {summary['C']:.6f} |",
        f"| Unknown rejection | {summary['unknown_rejection']:.6f} |", "",
        "Official Score=N/A；Test AUC/F1=N/A（竞赛 Test 未访问）。", "",
    ]), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="仅验证一折流程；仍需可用的 DINOv2 权重")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(E75_DIR) or not config_path.is_file():
        raise ValueError("E75 config must be inside the experiment directory")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not torch.cuda.is_available():
        raise RuntimeError("E75 requires CUDA for the frozen DINOv2 image backbone")
    device = torch.device("cuda")
    data = load_external_data(config, device)
    feature_path = E75_DIR / "outputs" / "image_features.npy"
    started = time.time()
    features, source_hashes = load_or_extract(data, config, feature_path, args.force)
    folds = subject_folds(config)
    if args.smoke:
        folds = folds[:1]
    rows = [fit_fold(features, data, roles, config, fold) for fold, roles in enumerate(folds, 1)]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("C", "identity_macro_F1", "unknown_F1", "detection_AUROC", "known_acceptance", "unknown_rejection")}
    output = {"experiment": config["experiment"], "completed_at": utc_now(), "smoke": args.smoke, "config_sha256": sha256(config_path), "source_sha256": source_hashes, "data": {"dataset": config["data"]["dataset"], "N": len(data.subject), "competition_train_accessed": False, "competition_test_accessed": False}, "image": config["image"], "folds": rows, "summary": summary, "elapsed_seconds": time.time() - started}
    out_dir = E75_DIR / config["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output, out_dir / "report.md")
    for fold, row in enumerate(rows, 1):
        torch.save({"experiment": "E75", "outer_fold": fold, "source_hashes": source_hashes, "config_sha256": output["config_sha256"], "roles": row["roles"], "feature_cache": str(feature_path)}, out_dir / f"image_template_outer{fold}.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
