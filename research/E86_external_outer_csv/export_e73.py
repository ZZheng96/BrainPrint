"""Export E73 outer-test per-sample predictions (native22 and native22_pair)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E73_DIR = ROOT / "research" / "E73_external_multisession"
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
E72_DIR = ROOT / "research" / "E72_pair_protocol"
for path in (E86_DIR, E73_DIR, ROOT, E72_DIR, E74_DIR):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from pair_head import SampleTemplatePairHead, subject_templates  # noqa: E402
from pair_protocol import confidence_values, fit_spaces, predict_spaces  # noqa: E402
from pretrain import embed_indices, make_model, sha256  # noqa: E402

NATIVE_CONFIG = json.loads((E73_DIR / "config_native22.json").read_text(encoding="utf-8"))
PAIR_CONFIG = json.loads((E73_DIR / "config_pair.json").read_text(encoding="utf-8"))
NATIVE_RESULTS = json.loads((E73_DIR / "native22" / "results.json").read_text(encoding="utf-8"))
PAIR_RESULTS = json.loads((E73_DIR / "native22_pair" / "results.json").read_text(encoding="utf-8"))

PRIMARY = NATIVE_CONFIG["protocol"]["primary_method"]
NATIVE_SCORE = f"{PRIMARY['space']}:{PRIMARY['score']}"
NATIVE_METHOD = "native22"
PAIR_METHOD = "native22_pair"

# Outer fields in results.json that a per-sample CSV can reproduce exactly.
FIELDS = (
    ("AUROC", "AUROC"),
    ("C", "C"),
    ("identity_macro_F1", "identity_macro_F1"),
    ("unknown_F1", "unknown_F1"),
    ("known_acceptance", "known_acceptance"),
    ("unknown_rejection", "U"),
    ("U_at_95", "U_at_95"),
    ("DIR_at_FPIR_1", "DIR_at_FPIR_1"),
    ("detection_EER", "detection_EER"),
    ("threshold", "threshold"),
)
NOT_EXPORTABLE = {
    "pair_AUROC": "需要完整的 query x template 分数矩阵；CSV 只保存逐样本置信度",
    "pair_EER": "同上",
    "working_98_threshold": "run4 已知样本 q02 阈值未写入 CSV（CSV 只有单一 operating threshold）",
    "working_98_known_acceptance": "同上",
    "working_98_unknown_rejection": "同上",
}


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def build_frame(*, fold: int, direction: str, method: str, session: str, data,
                known_index: np.ndarray, unknown_index: np.ndarray,
                known_pred: np.ndarray, unknown_pred: np.ndarray,
                known_conf: np.ndarray, unknown_conf: np.ndarray,
                threshold: float):
    return common.build_records(
        experiment="E73", outer_fold=fold, direction=direction, method=method,
        roles=np.r_[np.full(len(known_index), "enrolled"), np.full(len(unknown_index), "outer_unknown")],
        sample_id=np.r_[sample_ids(data, known_index), sample_ids(data, unknown_index)],
        session=np.r_[np.full(len(known_index), session), np.full(len(unknown_index), session)],
        run=np.r_[data.run[known_index], data.run[unknown_index]],
        task=np.r_[data.task[known_index], data.task[unknown_index]],
        artifact=np.r_[data.artifact[known_index], data.artifact[unknown_index]],
        true_identity=np.r_[data.subject[known_index], data.subject[unknown_index]],
        pred_label=np.r_[known_pred, unknown_pred],
        accepted=np.r_[known_pred != 0, unknown_pred != 0],
        confidence=np.r_[known_conf, unknown_conf],
        threshold=threshold,
    )


def check_roles(saved: dict, roles: dict[str, np.ndarray], path: Path) -> None:
    if saved.get("roles") != {key: value.tolist() for key, value in roles.items()}:
        raise ValueError(f"E73 checkpoint roles mismatch: {path}")


def load_native_encoder(roles: dict[str, np.ndarray], fold: int, data):
    path = E73_DIR / "native22" / "models" / f"supcon_outer{fold}.pt"
    saved = torch.load(path, map_location=data.raw.device, weights_only=False)
    expected = {
        "config": sha256(E73_DIR / "config_native22.json"),
        "pretrain": sha256(E73_DIR / "pretrain.py"),
        "external_data": sha256(E73_DIR / "external_data.py"),
        "native_model": sha256(E73_DIR / "native_model.py"),
    }
    if saved.get("source_hashes") != expected:
        raise ValueError(f"E73 native22 checkpoint source hash mismatch: {path}")
    check_roles(saved, roles, path)
    model = make_model(NATIVE_CONFIG, data.raw.device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(path), "sha256": sha256(path)}


def load_pair_head(roles: dict[str, np.ndarray], fold: int, data):
    path = E73_DIR / "native22_pair" / "models" / f"pair_outer{fold}.pt"
    saved = torch.load(path, map_location=data.raw.device, weights_only=False)
    expected = {
        "pair_config": sha256(E73_DIR / "config_pair.json"),
        "pair_head": sha256(E73_DIR / "pair_head.py"),
        "run_pair": sha256(E73_DIR / "run_pair.py"),
    }
    if saved.get("source_hashes") != expected:
        raise ValueError(f"E73 pair checkpoint source hash mismatch: {path}")
    check_roles(saved, roles, path)
    cfg = PAIR_CONFIG["pair_head"]
    head = SampleTemplatePairHead(int(cfg["embedding_dim"]), int(cfg["hidden_dim"])).to(data.raw.device)
    head.load_state_dict(saved["head_state"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, {"path": str(path), "sha256": sha256(path)}


def pair_scores(head, query: torch.Tensor, template: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(query, template).float().cpu().numpy()


def export_native(fold, roles, model, data) -> tuple[dict, list]:
    protocol = NATIVE_CONFIG["protocol"]
    enrolled = roles["enrolled"]
    frames, checks = [], {}
    for source, target in (("T", "E"), ("E", "T")):
        direction = f"{source}_to_{target}"
        template_i = data.indices(enrolled, session=source, runs=[int(protocol["template_run"])])
        inner_known_i = data.indices(enrolled, session=target, runs=[int(protocol["calibration_known_query_run"])])
        outer_known_i = data.indices(enrolled, session=target, runs=[int(protocol["evaluation_query_run"])])
        outer_unknown_i = data.indices(roles["outer_unknown"], session=target, runs=[int(protocol["outer_unknown_query_run"])])
        template, inner_known, outer_known, outer_unknown = (
            embed_indices(model, data.raw, index, NATIVE_CONFIG)
            for index in (template_i, inner_known_i, outer_known_i, outer_unknown_i)
        )
        spaces = fit_spaces(template, data.subject[template_i], enrolled, NATIVE_CONFIG)
        inner_scores = predict_spaces(spaces, inner_known, np.arange(len(inner_known)), len(inner_known))[PRIMARY["space"]]
        inner_conf = confidence_values(inner_scores, PRIMARY["score"])
        threshold = float(np.quantile(inner_conf, 0.05, method="linear"))
        known_scores = predict_spaces(spaces, outer_known, np.arange(len(outer_known)), len(outer_known))[PRIMARY["space"]]
        unknown_scores = predict_spaces(spaces, outer_unknown, np.arange(len(outer_unknown)), len(outer_unknown))[PRIMARY["space"]]
        known_conf = confidence_values(known_scores, PRIMARY["score"])
        unknown_conf = confidence_values(unknown_scores, PRIMARY["score"])
        known_direct = enrolled[known_scores.argmax(axis=1)]
        unknown_direct = enrolled[unknown_scores.argmax(axis=1)]
        known_pred = np.where(known_conf >= threshold, known_direct, 0).astype(int)
        unknown_pred = np.where(unknown_conf >= threshold, unknown_direct, 0).astype(int)
        frame = build_frame(
            fold=fold, direction=direction, method=NATIVE_METHOD, session=target, data=data,
            known_index=outer_known_i, unknown_index=outer_unknown_i,
            known_pred=known_pred, unknown_pred=unknown_pred,
            known_conf=known_conf, unknown_conf=unknown_conf, threshold=threshold,
        )
        frames.append(frame)
        reference = NATIVE_RESULTS["runs"][fold - 1]["directions"][direction]["methods"][NATIVE_SCORE]["outer"]
        checks[f"fold{fold}/{direction}/{NATIVE_METHOD}"] = compare(frame, reference)
    return checks, frames


def export_pair(fold, roles, model, head, data) -> tuple[dict, list]:
    protocol = PAIR_CONFIG["protocol"]
    enrolled = roles["enrolled"]
    frames, checks = [], {}
    for source, target in (("T", "E"), ("E", "T")):
        direction = f"{source}_to_{target}"
        template_i = data.indices(enrolled, session=source, runs=[int(protocol["template_run"])])
        calibration_i = data.indices(enrolled, session=target, runs=[int(protocol["calibration_known_query_run"])])
        outer_known_i = data.indices(enrolled, session=target, runs=[int(protocol["evaluation_query_run"])])
        outer_unknown_i = data.indices(roles["outer_unknown"], session=target, runs=[int(protocol["evaluation_query_run"])])
        if any(len(index) == 0 for index in (template_i, calibration_i, outer_known_i, outer_unknown_i)):
            raise ValueError("Empty E73 pair evaluation role")
        template = subject_templates(
            embed_indices(model, data.raw, template_i, NATIVE_CONFIG),
            data.subject[template_i], enrolled,
        )
        calibration_scores = pair_scores(head, embed_indices(model, data.raw, calibration_i, NATIVE_CONFIG), template)
        threshold = float(np.quantile(calibration_scores.max(axis=1), 0.05, method="linear"))
        known_scores = pair_scores(head, embed_indices(model, data.raw, outer_known_i, NATIVE_CONFIG), template)
        unknown_scores = pair_scores(head, embed_indices(model, data.raw, outer_unknown_i, NATIVE_CONFIG), template)
        known_conf = known_scores.max(axis=1)
        unknown_conf = unknown_scores.max(axis=1)
        known_direct = enrolled[known_scores.argmax(axis=1)]
        unknown_direct = enrolled[unknown_scores.argmax(axis=1)]
        known_pred = np.where(known_conf >= threshold, known_direct, 0).astype(int)
        unknown_pred = np.where(unknown_conf >= threshold, unknown_direct, 0).astype(int)
        frame = build_frame(
            fold=fold, direction=direction, method=PAIR_METHOD, session=target, data=data,
            known_index=outer_known_i, unknown_index=outer_unknown_i,
            known_pred=known_pred, unknown_pred=unknown_pred,
            known_conf=known_conf, unknown_conf=unknown_conf, threshold=threshold,
        )
        frames.append(frame)
        reference = PAIR_RESULTS["runs"][fold - 1]["directions"][direction]["outer"]
        checks[f"fold{fold}/{direction}/{PAIR_METHOD}"] = compare(frame, reference)
        saved = json.loads((E73_DIR / "native22_pair" / "models" / f"calibration_outer{fold}.json").read_text(encoding="utf-8"))
        checks[f"fold{fold}/{direction}/{PAIR_METHOD}"]["saved_calibration_q05"] = {
            "reference": float(saved["calibrations"][direction]["q05"]),
            "recomputed": threshold,
            "abs_diff": abs(float(saved["calibrations"][direction]["q05"]) - threshold),
        }
    return checks, frames


def compare(frame, reference: dict) -> dict:
    recomputed = common.metrics_from_frame(frame)
    return {
        key: {"reference": float(reference[ref]), "recomputed": float(recomputed[key]),
              "abs_diff": abs(float(reference[ref]) - float(recomputed[key]))}
        for key, ref in FIELDS
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E73 export requires CUDA")
    data = load_external_data(NATIVE_CONFIG, torch.device("cuda"))
    frames, checked, encoders = [], {}, {}
    for fold, roles in enumerate(subject_folds(NATIVE_CONFIG), start=1):
        model, encoder = load_native_encoder(roles, fold, data)
        head, pair_encoder = load_pair_head(roles, fold, data)
        encoders[str(fold)] = {"encoder": encoder, "pair_head": pair_encoder}
        native_checks, native_frames = export_native(fold, roles, model, data)
        pair_checks, pair_frames = export_pair(fold, roles, model, head, data)
        checked.update(native_checks)
        checked.update(pair_checks)
        frames.extend(native_frames)
        frames.extend(pair_frames)
        del model, head
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values())
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E73 results.json 不一致：max abs diff {worst}")
    frame = __import__("pandas").concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], E73_DIR / "outputs" / "outer_test_predictions.csv")
    groups = common.metrics_by_group(frame)
    keys = ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
            "unknown_false_acceptance", "unknown_rejection", "U_at_95",
            "DIR_at_FPIR_1", "detection_EER")
    summary = {key: float(np.mean([group[key] for group in groups.values()])) for key in keys}
    method_means = {
        method: {key: float(np.mean([group[key] for name, group in groups.items() if name.endswith(f"/{method}")]))
                 for key in keys}
        for method in (NATIVE_METHOD, PAIR_METHOD)
    }
    output = {
        "experiment": "E73",
        "methods": {NATIVE_METHOD: NATIVE_SCORE, PAIR_METHOD: PAIR_RESULTS["primary_method"]},
        "csv": manifest, "checkpoints": encoders,
        "groups": groups, "group_mean": summary, "method_group_mean": method_means,
        "reference_check": {"max_abs_diff": worst, "per_group": checked},
        "not_exportable_from_csv": NOT_EXPORTABLE,
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(E73_DIR / "outputs" / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "method_group_mean": method_means,
                      "max_abs_diff": worst, "n_rows": manifest["n_rows"],
                      "n_groups": manifest["n_groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
