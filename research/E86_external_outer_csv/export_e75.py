"""Export E75 outer-test per-sample predictions (frozen EEG-image template)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E73_DIR = ROOT / "research" / "E73_external_multisession"
E75_DIR = ROOT / "research" / "E75_image_template"
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
for path in (E86_DIR, E73_DIR, ROOT, E74_DIR):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402

CONFIG_PATH = E75_DIR / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
RESULTS = json.loads((E75_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))
METHOD = CONFIG["protocol"]["score"].replace(" ", "_")
DIRECTION = "T_to_E"


def sha256(path: Path) -> str:
    return common.sha256(path)


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def confidence(scores: np.ndarray) -> np.ndarray:
    """Same definition as run_e75.py: top1 - top2 cosine prototype margin."""
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def load_features(data) -> np.ndarray:
    path = E75_DIR / "outputs" / "image_features.npy"
    meta = json.loads((E75_DIR / "outputs" / "image_features.json").read_text(encoding="utf-8"))
    if meta["source_hashes"]["config"] != sha256(CONFIG_PATH):
        raise ValueError("E75 image feature cache was built from a different config")
    if tuple(meta["shape"]) != (len(data.subject), int(meta["shape"][1])):
        raise ValueError(f"E75 image feature shape mismatch: {meta['shape']}")
    features = np.load(path, mmap_mode="r")
    if features.shape != tuple(meta["shape"]):
        raise ValueError(f"E75 image feature array mismatch: {features.shape}")
    return features


def export_fold(features: np.ndarray, data, roles: dict[str, np.ndarray], fold: int):
    protocol = CONFIG["protocol"]
    saved = torch.load(E75_DIR / "outputs" / f"image_template_outer{fold}.pt",
                       map_location="cpu", weights_only=False)
    if saved.get("roles") != {key: value.tolist() for key, value in roles.items()}:
        raise ValueError(f"E75 checkpoint roles mismatch: fold {fold}")
    if saved.get("config_sha256") != sha256(CONFIG_PATH):
        raise ValueError(f"E75 checkpoint config hash mismatch: fold {fold}")

    enrolled = roles["enrolled"]
    train_index = data.indices(enrolled, session="T", runs=[int(run) for run in protocol["train_runs"]])
    template_index = data.indices(enrolled, session="T", runs=[int(protocol["template_run"])])
    calibration_index = data.indices(enrolled, session="E", runs=[int(protocol["calibration_known_query_run"])])
    outer_known_index = data.indices(enrolled, session="E", runs=[int(protocol["evaluation_query_run"])])
    outer_unknown_index = data.indices(roles["outer_unknown"], session="E", runs=[int(protocol["outer_unknown_query_run"])])

    mean = np.asarray(features[train_index]).mean(axis=0)
    std = np.asarray(features[train_index]).std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)

    def transform(index):
        value = (np.asarray(features[index], dtype=np.float32) - mean) / std
        return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)

    template = transform(template_index)
    prototypes = np.stack([template[data.subject[template_index] == subject].mean(axis=0)
                           for subject in enrolled])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)

    def scores(index):
        return transform(index) @ prototypes.T

    threshold = float(np.quantile(confidence(scores(calibration_index)), 0.05))
    known_scores = scores(outer_known_index)
    unknown_scores = scores(outer_unknown_index)
    known_conf = confidence(known_scores)
    unknown_conf = confidence(unknown_scores)
    known_direct = enrolled[known_scores.argmax(axis=1)]
    unknown_direct = enrolled[unknown_scores.argmax(axis=1)]
    known_pred = np.where(known_conf >= threshold, known_direct, 0).astype(int)
    unknown_pred = np.where(unknown_conf >= threshold, unknown_direct, 0).astype(int)

    frame = common.build_records(
        experiment="E75", outer_fold=fold, direction=DIRECTION, method=METHOD,
        roles=np.r_[np.full(len(outer_known_index), "enrolled"),
                    np.full(len(outer_unknown_index), "outer_unknown")],
        sample_id=np.r_[sample_ids(data, outer_known_index), sample_ids(data, outer_unknown_index)],
        session="E",
        run=np.r_[data.run[outer_known_index], data.run[outer_unknown_index]],
        task=np.r_[data.task[outer_known_index], data.task[outer_unknown_index]],
        artifact=np.r_[data.artifact[outer_known_index], data.artifact[outer_unknown_index]],
        true_identity=np.r_[data.subject[outer_known_index], data.subject[outer_unknown_index]],
        pred_label=np.r_[known_pred, unknown_pred],
        accepted=np.r_[known_pred != 0, unknown_pred != 0],
        confidence=np.r_[known_conf, unknown_conf],
        threshold=threshold,
    )
    reference = RESULTS["folds"][fold - 1]
    checks = compare(frame, reference, enrolled)
    checks["C_replay_direct"] = {
        "reference": float(reference["C"]),
        "recomputed_from_cached_features": float(np.mean(known_direct == data.subject[outer_known_index])),
        "abs_diff": abs(float(reference["C"]) - float(np.mean(known_direct == data.subject[outer_known_index]))),
    }
    return frame, checks


def compare(frame, reference: dict, enrolled: np.ndarray) -> dict:
    metrics = common.metrics_from_frame(frame)
    known = frame[frame["role"] == "enrolled"]
    unknown = frame[frame["role"] == "outer_unknown"]
    # run_e75.py scores identity macro-F1 over enrolled classes plus the reject class,
    # and unknown-F1 with the known/accept class as positive.
    identity_true = np.r_[known["true_identity"].to_numpy(), np.zeros(len(unknown), dtype=int)]
    identity_pred = np.r_[known["pred_label"].to_numpy(), unknown["pred_label"].to_numpy()]
    identity_f1 = float(f1_score(identity_true, identity_pred, labels=np.r_[enrolled, 0],
                                 average="macro", zero_division=0))
    detection_true = np.r_[np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)]
    unknown_f1 = float(f1_score(detection_true, frame["accepted"].to_numpy(dtype=bool), zero_division=0))
    checks = {
        "AUROC": pair(metrics["AUROC"], reference["detection_AUROC"]),
        "identity_macro_F1": pair(identity_f1, reference["identity_macro_F1"]),
        "unknown_F1": pair(unknown_f1, reference["unknown_F1"]),
        "known_acceptance": pair(metrics["known_acceptance"], reference["known_acceptance"]),
        "unknown_rejection": pair(metrics["unknown_rejection"], reference["unknown_rejection"]),
        "threshold": pair(metrics["threshold"], reference["threshold"]),
        "C": {
            "reference": float(reference["C"]),
            "csv_recomputed_with_rejection": float(metrics["C"]),
            "abs_diff_with_rejection": abs(float(reference["C"]) - float(metrics["C"])),
            "comparable": False,
            "reason": "E75 的 C 是 argmax 直接身份准确率（不含拒识）；统一 CSV 用 pred_label=0 表示拒识，"
                      "被拒样本的 argmax 身份无法从逐样本预测行恢复",
        },
    }
    return checks


def pair(recomputed: float, reference: float) -> dict:
    return {"reference": float(reference), "recomputed": float(recomputed),
            "abs_diff": abs(float(reference) - float(recomputed))}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E75 export requires CUDA")
    data = load_external_data(CONFIG, torch.device("cuda"))
    features = load_features(data)
    frames, checked = [], {}
    for fold, roles in enumerate(subject_folds(CONFIG), start=1):
        frame, checks = export_fold(features, data, roles, fold)
        frames.append(frame)
        checked[f"fold{fold}/{DIRECTION}/{METHOD}"] = checks
    comparable = [
        {key: value for key, value in checks.items() if key not in ("C", "C_replay_direct")}
        for checks in checked.values()
    ]
    worst = max(item["abs_diff"] for group in comparable for item in group.values())
    replay = max(checks["C_replay_direct"]["abs_diff"] for checks in checked.values())
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E75 results.json 不一致：max abs diff {worst}")
    if replay > 1e-9:
        raise ValueError(f"E75 直接身份准确率无法复现：max abs diff {replay}")
    frame = __import__("pandas").concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], E75_DIR / "outputs" / "outer_test_predictions.csv")
    groups = common.metrics_by_group(frame)
    summary = {key: float(np.mean([group[key] for group in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "detection_EER")}
    reference_means = {key: float(np.mean([RESULTS["folds"][index][key] for index in range(3)]))
                       for key in ("C", "identity_macro_F1", "unknown_F1", "detection_AUROC",
                                   "known_acceptance", "unknown_rejection")}
    output = {
        "experiment": "E75",
        "method": METHOD, "primary_method": CONFIG["protocol"]["score"],
        "image_encoder": CONFIG["image"]["encoder"], "backbone": CONFIG["image"]["backbone"],
        "csv": manifest, "groups": groups,
        "group_mean": summary, "reference_group_mean": reference_means,
        "definition_note": {
            "AUROC": "known-vs-unknown confidence AUROC",
            "identity_macro_F1": "enrolled 四类 + 拒识类的 joint macro-F1（run_e75.py 定义）",
            "unknown_F1": "accepted 为正类的检测 F1（run_e75.py 定义）",
            "C": "run_e75.py 的 C 不含拒识；CSV 只能复现含拒识的准确率",
        },
        "reference_check": {"max_abs_diff": worst, "C_replay_max_abs_diff": replay,
                            "per_group": checked},
        "role_definition": {"enrolled": "run5 E-session known query", "outer_unknown": "run5 E-session held-out subject query"},
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(E75_DIR / "outputs" / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "reference_group_mean": reference_means,
                      "max_abs_diff": worst, "C_replay_max_abs_diff": replay,
                      "n_rows": manifest["n_rows"], "n_groups": manifest["n_groups"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
