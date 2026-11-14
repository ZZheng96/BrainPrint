"""Export E81 outer-test per-sample predictions (pair head + per-identity EVT tail calibration)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E81_DIR = ROOT / "research" / "E81_evt_open_set"
E79_DIR = ROOT / "research" / "E79_pair_open_set"
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
for path in (E86_DIR, E81_DIR, E79_DIR, E74_DIR, ROOT / "research" / "E72_pair_protocol",
             ROOT / "research" / "E73_external_multisession"):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from pair_head import subject_templates  # noqa: E402
from pretrain import pair_metrics  # noqa: E402
from run_e74 import embed_indices, sha256  # noqa: E402
from run_e79 import fit_pair_head, load_frozen_encoder, pair_scores  # noqa: E402
from run_e81 import fused_scores, maha_confidence, robust_stats, tail_thresholds  # noqa: E402

BASE_CONFIG = json.loads((E74_DIR / "config.json").read_text(encoding="utf-8-sig"))
PAIR_CONFIG = json.loads((E79_DIR / "config.json").read_text(encoding="utf-8"))
CONFIG = json.loads((E81_DIR / "config.json").read_text(encoding="utf-8"))
RESULTS = json.loads((E81_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))
OUTPUT_DIR = E81_DIR / "outputs"
METHOD = "evt_q02"


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def compare_entry(reference: float, recomputed: float) -> dict:
    reference, recomputed = float(reference), float(recomputed)
    return {"reference": reference, "recomputed": recomputed,
            "abs_diff": abs(reference - recomputed)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E81 export requires CUDA")
    device = torch.device("cuda")
    data = load_external_data(BASE_CONFIG, device)
    protocol = CONFIG["protocol"]
    fraction = float(protocol["tail_fraction"])
    frames, checked, checkpoints = [], {}, {}
    for fold, roles in enumerate(subject_folds(BASE_CONFIG), start=1):
        model, _ = load_frozen_encoder(BASE_CONFIG, E74_DIR / "config.json", roles, fold, device)
        pair_ckpt = OUTPUT_DIR / "models" / f"pair_outer{fold}.pt"
        head, _ = fit_pair_head(model, data, roles, BASE_CONFIG, PAIR_CONFIG,
                                E79_DIR / "config.json", fold, pair_ckpt, False)
        evt_ckpt = OUTPUT_DIR / "models" / f"evt_outer{fold}.pt"
        saved = torch.load(evt_ckpt, map_location="cpu", weights_only=False)
        if saved.get("roles") != {key: value.tolist() for key, value in roles.items()}:
            raise ValueError(f"E81 evt checkpoint roles mismatch: {evt_ckpt}")
        checkpoints[f"fold{fold}"] = {
            "pair": {"path": str(pair_ckpt), "sha256": sha256(pair_ckpt)},
            "evt": {"path": str(evt_ckpt), "sha256": sha256(evt_ckpt)},
        }
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
            thresholds, _ = tail_thresholds(known_cal, data.subject[known_cal_i], enrolled, fraction)
            known_conf = known.max(axis=1)
            unknown_conf = unknown.max(axis=1)
            direct_known = enrolled[known.argmax(axis=1)]
            direct_unknown = enrolled[unknown.argmax(axis=1)]
            known_thr = np.asarray([thresholds[int(c)] for c in direct_known])
            unknown_thr = np.asarray([thresholds[int(c)] for c in direct_unknown])
            known_pred = np.where(known_conf >= known_thr, direct_known, 0).astype(int)
            unknown_pred = np.where(unknown_conf >= unknown_thr, direct_unknown, 0).astype(int)
            direction = f"{source}_to_{target}"
            # pair_AUROC 需要完整 K 类分数矩阵，无法从逐样本 CSV 重算；用重建的分数矩阵直接计算。
            pair_auroc = float(pair_metrics(known, unknown, data.subject[known_i], enrolled)["pair_AUROC"])
            frame = common.build_records(
                experiment="E81", outer_fold=fold, direction=direction, method=METHOD,
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
                threshold=float(np.mean(list(thresholds.values()))),
            )
            frames.append(frame)
            recomputed = common.metrics_from_frame(frame)
            reference = RESULTS["runs"][fold - 1]["directions"][direction]["outer"]
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
            }
        del model
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values())
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E81 results.json 不一致：max abs diff {worst}")
    frame = pd.concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], OUTPUT_DIR / "outer_test_predictions.csv")
    groups = common.metrics_by_group(frame)
    summary = {key: float(np.mean([group[key] for group in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "U_at_95",
                           "DIR_at_FPIR_1", "detection_EER")}
    output = {
        "experiment": "E81", "method": METHOD, "methods": [METHOD],
        "csv": manifest, "groups": groups, "group_mean": summary,
        "reference_check": {"max_abs_diff": worst, "per_group": checked},
        "threshold_definition": ("per-identity run4 known lower-tail q02 阈值；CSV 的 threshold 列"
                                 "为该方向各身份阈值的均值，逐样本判定用其所属身份的阈值"),
        "checkpoints": checkpoints,
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(OUTPUT_DIR / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "max_abs_diff": worst,
                      "n_rows": manifest["n_rows"], "n_groups": manifest["n_groups"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
