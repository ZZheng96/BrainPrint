"""Export E80 outer-test per-sample predictions (robust Mahalanobis min-fusion, q02)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E80_DIR = ROOT / "research" / "E80_robust_open_set"
E79_DIR = ROOT / "research" / "E79_pair_open_set"
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
for path in (E86_DIR, E80_DIR, E79_DIR, E74_DIR, ROOT / "research" / "E72_pair_protocol",
             ROOT / "research" / "E73_external_multisession"):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from pair_head import subject_templates  # noqa: E402
from run_e74 import embed_indices  # noqa: E402
from pretrain import pair_metrics  # noqa: E402
from run_e79 import fit_pair_head, load_frozen_encoder, pair_scores  # noqa: E402
from run_e80 import fused_scores, maha_confidence, robust_stats  # noqa: E402

BASE_CONFIG = json.loads((E74_DIR / "config.json").read_text(encoding="utf-8-sig"))
PAIR_CONFIG = json.loads((E79_DIR / "config.json").read_text(encoding="utf-8"))
CONFIG = json.loads((E80_DIR / "config.json").read_text(encoding="utf-8"))
RESULTS = json.loads((E80_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))
OUTPUT_DIR = E80_DIR / "outputs"
METHOD = "min_fusion_q02"


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def compare_entry(reference: float, recomputed: float) -> dict:
    reference, recomputed = float(reference), float(recomputed)
    return {"reference": reference, "recomputed": recomputed,
            "abs_diff": abs(reference - recomputed)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E80 export requires CUDA")
    device = torch.device("cuda")
    data = load_external_data(BASE_CONFIG, device)
    protocol = CONFIG["protocol"]
    frames, checked = [], {}
    for fold, roles in enumerate(subject_folds(BASE_CONFIG), start=1):
        model, _ = load_frozen_encoder(BASE_CONFIG, E74_DIR / "config.json", roles, fold, device)
        head, _ = fit_pair_head(
            model, data, roles, BASE_CONFIG, PAIR_CONFIG, E79_DIR / "config.json", fold,
            E79_DIR / "outputs" / "models" / f"pair_outer{fold}.pt", False,
        )
        enrolled = roles["enrolled"]
        for source, target in (("T", "E"), ("E", "T")):
            template_i = data.indices(enrolled, session=source, runs=[int(protocol["template_run"])])
            known_cal_i = data.indices(enrolled, session=target,
                                       runs=[int(protocol["calibration_known_query_run"])])
            known_i = data.indices(enrolled, session=target,
                                   runs=[int(protocol["evaluation_query_run"])])
            unknown_i = data.indices(roles["outer_unknown"], session=target,
                                     runs=[int(protocol["evaluation_query_run"])])
            template = subject_templates(
                embed_indices(model, data.raw, template_i, BASE_CONFIG),
                data.subject[template_i], enrolled,
            )
            stats = robust_stats(model, data, roles, BASE_CONFIG, CONFIG, source)

            def score(indices: np.ndarray) -> np.ndarray:
                embedding = embed_indices(model, data.raw, indices, BASE_CONFIG)
                return fused_scores(pair_scores(head, embedding, template),
                                    maha_confidence(embedding, stats, CONFIG), CONFIG)

            known_cal, known, unknown = map(score, (known_cal_i, known_i, unknown_i))
            threshold = float(np.quantile(known_cal.max(axis=1), 0.02, method="linear"))
            known_conf = known.max(axis=1)
            unknown_conf = unknown.max(axis=1)
            direct_known = enrolled[known.argmax(axis=1)]
            direct_unknown = enrolled[unknown.argmax(axis=1)]
            known_pred = np.where(known_conf >= threshold, direct_known, 0).astype(int)
            unknown_pred = np.where(unknown_conf >= threshold, direct_unknown, 0).astype(int)
            direction = f"{source}_to_{target}"
            reference = RESULTS["runs"][fold - 1]["directions"][direction]["outer"]
            # pair_AUROC 需要完整 K 类分数矩阵，无法从逐样本 CSV 重算；用重建的分数矩阵直接计算。
            pair_auroc = float(pair_metrics(known, unknown, data.subject[known_i], enrolled)["pair_AUROC"])
            frame = common.build_records(
                experiment="E80", outer_fold=fold, direction=direction, method=METHOD,
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
            checked[f"fold{fold}/{direction}"] = {
                "AUROC": compare_entry(reference["AUROC"], recomputed["AUROC"]),
                "identity_macro_F1": compare_entry(reference["identity_macro_F1"],
                                                   recomputed["identity_macro_F1"]),
                "unknown_F1": compare_entry(reference["unknown_F1"], recomputed["unknown_F1"]),
                "known_acceptance": compare_entry(reference["known_acceptance"],
                                                  recomputed["known_acceptance"]),
                "unknown_false_acceptance": compare_entry(reference["q02_unknown_false_acceptance"],
                                                          recomputed["unknown_false_acceptance"]),
                "C": compare_entry(reference["C"], recomputed["C"]),
                "pair_AUROC_from_scores": compare_entry(reference["pair_AUROC"], pair_auroc),
                "threshold": compare_entry(reference["threshold"], recomputed["threshold"]),
            }
        del model
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values())
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E80 results.json 不一致：max abs diff {worst}")
    frame = pd.concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], OUTPUT_DIR / "outer_test_predictions.csv")
    groups = common.metrics_by_group(frame)
    summary = {key: float(np.mean([group[key] for group in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "U_at_95",
                           "DIR_at_FPIR_1", "detection_EER")}
    output = {
        "experiment": "E80", "method": METHOD, "methods": [METHOD],
        "csv": manifest, "groups": groups, "group_mean": summary,
        "reference_check": {"max_abs_diff": worst, "per_group": checked},
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(OUTPUT_DIR / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "max_abs_diff": worst,
                      "n_rows": manifest["n_rows"], "n_groups": manifest["n_groups"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
