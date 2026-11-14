"""Export E74 outer-test per-sample predictions (primary wccn:margin)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
for path in (E86_DIR, E74_DIR, ROOT, ROOT / "research" / "E72_pair_protocol",
             ROOT / "research" / "E73_external_multisession"):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from multiscale_model import MultiScaleNative22  # noqa: E402
from pair_protocol import confidence_values, fit_spaces, predict_spaces  # noqa: E402
from run_e74 import embed_indices, sha256  # noqa: E402

CONFIG = json.loads((E74_DIR / "config.json").read_text(encoding="utf-8-sig"))
RESULTS = json.loads((E74_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))
PRIMARY = CONFIG["protocol"]["primary_method"]
METHOD = f"{PRIMARY['space']}:{PRIMARY['score']}"


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def load_encoder(data, roles, fold: int):
    path = E74_DIR / "outputs" / "models" / f"supcon_outer{fold}.pt"
    saved = torch.load(path, map_location=data.raw.device, weights_only=False)
    expected = {
        "config": sha256(E74_DIR / "config.json"),
        "run_e74": sha256(E74_DIR / "run_e74.py"),
        "external_data": sha256(E74_DIR / "external_data.py"),
        "multiscale_model": sha256(E74_DIR / "multiscale_model.py"),
    }
    if saved.get("source_hashes") != expected:
        raise ValueError(f"E74 checkpoint source hash mismatch: {path}")
    if saved.get("roles") != {k: v.tolist() for k, v in roles.items()}:
        raise ValueError(f"E74 checkpoint roles mismatch: {path}")
    model = MultiScaleNative22(CONFIG).to(data.raw.device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E74 export requires CUDA")
    device = torch.device("cuda")
    data = load_external_data(CONFIG, device)
    protocol = CONFIG["protocol"]
    frames, checked = [], {}
    for fold, roles in enumerate(subject_folds(CONFIG), start=1):
        model, encoder = load_encoder(data, roles, fold)
        enrolled = roles["enrolled"]
        for source, target in (("T", "E"), ("E", "T")):
            template_i = data.indices(enrolled, session=source, runs=[int(protocol["template_run"])])
            inner_known_i = data.indices(enrolled, session=target, runs=[int(protocol["calibration_known_query_run"])])
            outer_known_i = data.indices(enrolled, session=target, runs=[int(protocol["evaluation_query_run"])])
            outer_unknown_i = data.indices(roles["outer_unknown"], session=target, runs=[int(protocol["outer_unknown_query_run"])])
            template, inner_known, outer_known, outer_unknown = (
                embed_indices(model, data.raw, index, CONFIG)
                for index in (template_i, inner_known_i, outer_known_i, outer_unknown_i)
            )
            spaces = fit_spaces(template, data.subject[template_i], enrolled, CONFIG)
            space, score_name = PRIMARY["space"], PRIMARY["score"]
            inner_conf = confidence_values(predict_spaces(spaces, inner_known, np.arange(len(inner_known)), len(inner_known))[space], score_name)
            threshold = float(np.quantile(inner_conf, 0.05, method="linear"))
            known_scores = predict_spaces(spaces, outer_known, np.arange(len(outer_known)), len(outer_known))[space]
            unknown_scores = predict_spaces(spaces, outer_unknown, np.arange(len(outer_unknown)), len(outer_unknown))[space]
            known_conf = confidence_values(known_scores, score_name)
            unknown_conf = confidence_values(unknown_scores, score_name)
            direct_known = enrolled[known_scores.argmax(axis=1)]
            direct_unknown = enrolled[unknown_scores.argmax(axis=1)]
            known_pred = np.where(known_conf >= threshold, direct_known, 0).astype(int)
            unknown_pred = np.where(unknown_conf >= threshold, direct_unknown, 0).astype(int)
            direction = f"{source}_to_{target}"
            frame = common.build_records(
                experiment="E74", outer_fold=fold, direction=direction, method=METHOD,
                roles=np.r_[np.full(len(outer_known_i), "enrolled"), np.full(len(outer_unknown_i), "outer_unknown")],
                sample_id=np.r_[sample_ids(data, outer_known_i), sample_ids(data, outer_unknown_i)],
                session=np.r_[np.full(len(outer_known_i), target), np.full(len(outer_unknown_i), target)],
                run=np.r_[data.run[outer_known_i], data.run[outer_unknown_i]],
                task=np.r_[data.task[outer_known_i], data.task[outer_unknown_i]],
                artifact=np.r_[data.artifact[outer_known_i], data.artifact[outer_unknown_i]],
                true_identity=np.r_[data.subject[outer_known_i], data.subject[outer_unknown_i]],
                pred_label=np.r_[known_pred, unknown_pred],
                accepted=np.r_[known_pred != 0, unknown_pred != 0],
                confidence=np.r_[known_conf, unknown_conf],
                threshold=threshold,
            )
            frames.append(frame)
            reference = RESULTS["runs"][fold - 1]["directions"][direction]["methods"][METHOD]["outer"]
            recomputed = common.metrics_from_frame(frame)
            pairs = {key: key for key in ("AUROC", "C", "identity_macro_F1", "unknown_F1",
                                          "known_acceptance", "U_at_95", "DIR_at_FPIR_1",
                                          "detection_EER", "threshold")}
            pairs["unknown_rejection"] = "U"
            checked[f"fold{fold}/{direction}"] = {
                key: {"reference": float(reference[ref]), "recomputed": float(recomputed[key]),
                      "abs_diff": abs(float(reference[ref]) - float(recomputed[key]))}
                for key, ref in pairs.items()
            }
        del model
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values())
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E74 results.json 不一致：max abs diff {worst}")
    frame = __import__("pandas").concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], E74_DIR / "outputs" / "outer_test_predictions.csv")
    groups = common.metrics_by_group(frame)
    summary = {key: float(np.mean([g[key] for g in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "U_at_95", "DIR_at_FPIR_1", "detection_EER")}
    output = {
        "experiment": "E74", "method": METHOD,
        "csv": manifest, "groups": groups, "group_mean": summary,
        "reference_check": {"max_abs_diff": worst, "per_group": checked},
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(E74_DIR / "outputs" / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "max_abs_diff": worst, "n_rows": manifest["n_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
