"""Export E82 outer-test per-sample predictions (calibration-unknown FPIR 1%/2%)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E82_DIR = ROOT / "research" / "E82_unknown_calibration"
E79_DIR = ROOT / "research" / "E79_pair_open_set"
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
for path in (E86_DIR, E82_DIR, E79_DIR, E74_DIR, ROOT / "research" / "E72_pair_protocol",
             ROOT / "research" / "E73_external_multisession"):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from pair_head import subject_templates  # noqa: E402
from run_e74 import embed_indices  # noqa: E402
from run_e79 import fit_pair_head, load_frozen_encoder, pair_scores  # noqa: E402
from run_e82 import unknown_thresholds  # noqa: E402

BASE_CONFIG = json.loads((E74_DIR / "config.json").read_text(encoding="utf-8-sig"))
PAIR_CONFIG = json.loads((E79_DIR / "config.json").read_text(encoding="utf-8"))
CONFIG = json.loads((E82_DIR / "config.json").read_text(encoding="utf-8"))
RESULTS = json.loads((E82_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))
OUTPUT_DIR = E82_DIR / "outputs"
METHODS = {0.01: "fpir01", 0.02: "fpir02"}


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def compare_entry(reference: float, recomputed: float) -> dict:
    reference, recomputed = float(reference), float(recomputed)
    return {"reference": reference, "recomputed": recomputed,
            "abs_diff": abs(reference - recomputed)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E82 export requires CUDA")
    device = torch.device("cuda")
    data = load_external_data(BASE_CONFIG, device)
    protocol = CONFIG["protocol"]
    fpirs = [float(value) for value in protocol["target_fpirs"]]
    frames, checked = [], {}
    for fold, roles in enumerate(subject_folds(BASE_CONFIG), start=1):
        model, _ = load_frozen_encoder(BASE_CONFIG, E74_DIR / "config.json", roles, fold, device)
        head, _ = fit_pair_head(
            model, data, roles, BASE_CONFIG, PAIR_CONFIG, E79_DIR / "config.json", fold,
            OUTPUT_DIR / "models" / f"pair_outer{fold}.pt", False,
        )
        enrolled = roles["enrolled"]
        for source, target in (("T", "E"), ("E", "T")):
            template_i = data.indices(enrolled, session=source, runs=[int(protocol["template_run"])])
            calibration_i = data.indices(roles["calibration_unknown"], session=target,
                                         runs=[int(protocol["calibration_unknown_run"])])
            known_i = data.indices(enrolled, session=target,
                                   runs=[int(protocol["outer_evaluation_run"])])
            unknown_i = data.indices(roles["outer_unknown"], session=target,
                                     runs=[int(protocol["outer_evaluation_run"])])
            template = subject_templates(
                embed_indices(model, data.raw, template_i, BASE_CONFIG),
                data.subject[template_i], enrolled,
            )

            def score(indices: np.ndarray) -> np.ndarray:
                embedding = embed_indices(model, data.raw, indices, BASE_CONFIG)
                return 1.0 / (1.0 + np.exp(-pair_scores(head, embedding, template)))

            calibration, known, unknown = map(score, (calibration_i, known_i, unknown_i))
            thresholds, _ = unknown_thresholds(calibration, fpirs)
            known_conf = known.max(axis=1)
            unknown_conf = unknown.max(axis=1)
            direct_known = enrolled[known.argmax(axis=1)]
            direct_unknown = enrolled[unknown.argmax(axis=1)]
            direction = f"{source}_to_{target}"
            for fpir in fpirs:
                threshold = float(thresholds[str(fpir)])
                known_pred = np.where(known_conf >= threshold, direct_known, 0).astype(int)
                unknown_pred = np.where(unknown_conf >= threshold, direct_unknown, 0).astype(int)
                frame = common.build_records(
                    experiment="E82", outer_fold=fold, direction=direction,
                    method=METHODS[fpir],
                    roles=np.r_[np.full(len(known_i), "enrolled"),
                                np.full(len(unknown_i), "outer_unknown")],
                    sample_id=np.r_[sample_ids(data, known_i), sample_ids(data, unknown_i)],
                    session=np.r_[np.full(len(known_i), target), np.full(len(unknown_i), target)],
                    run=np.r_[data.run[known_i], data.run[unknown_i]],
                    task=np.r_[data.task[known_i], data.task[unknown_i]],
                    artifact=np.r_[data.artifact[known_i], data.artifact[unknown_i]],
                    true_identity=np.r_[data.subject[known_i], data.subject[unknown_i]],
                    pred_label=np.r_[known_pred, unknown_pred],
                    accepted=np.r_[known_pred != 0, unknown_pred != 0],
                    confidence=np.r_[known_conf, unknown_conf],
                    threshold=threshold,
                )
                frames.append(frame)
                recomputed = common.metrics_from_frame(frame)
                detection_true = np.r_[np.ones(len(known_i), dtype=int),
                                       np.zeros(len(unknown_i), dtype=int)]
                accepted = np.r_[known_pred != 0, unknown_pred != 0]
                # E82 run_e82.evaluate 使用 pos_label=1（known 类）计算其 unknown_F1。
                e82_unknown_f1 = float(f1_score(detection_true, accepted, pos_label=1, zero_division=0))
                reference = RESULTS["runs"][fold - 1]["directions"][direction]["outer"][str(fpir)]
                checked[f"fold{fold}/{direction}/{METHODS[fpir]}"] = {
                    "AUROC": compare_entry(reference["AUROC"], recomputed["AUROC"]),
                    "identity_macro_F1": compare_entry(reference["identity_macro_F1"],
                                                       recomputed["identity_macro_F1"]),
                    "known_acceptance": compare_entry(reference["known_acceptance"],
                                                      recomputed["known_acceptance"]),
                    "unknown_false_acceptance": compare_entry(reference["unknown_false_acceptance"],
                                                              recomputed["unknown_false_acceptance"]),
                    "unknown_F1_as_reported_pos_label_1": compare_entry(reference["unknown_F1"],
                                                                        e82_unknown_f1),
                    "unknown_F1_common_pos_label_0": {
                        "value": float(recomputed["unknown_F1"]),
                        "note": "E82 报告值用 pos_label=1（known 接受类）；E86 common 固定 pos_label=0（unknown 拒识类），定义不同不参与 1e-9 校验",
                    },
                    "threshold": {"reference": None, "recomputed": threshold,
                                  "abs_diff": 0.0, "note": "E82 results.json 未记录阈值"},
                }
        del model
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values()
                if isinstance(item, dict) and item.get("reference") is not None)
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E82 results.json 不一致：max abs diff {worst}")
    frame = pd.concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], OUTPUT_DIR / "outer_test_predictions.csv")
    primary = frame[frame["method"] == METHODS[0.02]]
    groups = common.metrics_by_group(primary)
    summary = {key: float(np.mean([group[key] for group in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "U_at_95",
                           "DIR_at_FPIR_1", "detection_EER")}
    output = {
        "experiment": "E82", "method": METHODS[0.02], "methods": list(METHODS.values()),
        "csv": manifest, "groups": groups, "group_mean": summary,
        "reference_check": {"max_abs_diff": worst, "per_group": checked},
        "metric_definition_note": ("E82 run_e82.evaluate 的 unknown_F1 用 pos_label=1 计算（known 接受类的 F1），"
                                   "E86 common.metrics_from_frame 固定 pos_label=0（unknown 拒识类的 F1）；"
                                   "CSV 重算的 pos_label=1 值已与 results.json 逐组比对"),
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(OUTPUT_DIR / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "max_abs_diff": worst,
                      "n_rows": manifest["n_rows"], "n_groups": manifest["n_groups"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
