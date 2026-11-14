"""Export E78 outer-test per-sample predictions (per-identity q05 calibration)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E78_DIR = ROOT / "research" / "E78_identity_calibration"
for path in (E86_DIR, E78_DIR, ROOT, ROOT / "research" / "E74_multiscale_native22",
             ROOT / "research" / "E73_external_multisession"):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from run_e78 import BASE, embed, load_encoder, scores, sha256  # noqa: E402

METHOD = "per_identity_q05"
RESULTS = json.loads((E78_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def load_head(data, fold: int, direction: str):
    path = E78_DIR / "outputs" / "models" / f"mahalanobis_calibrated_outer{fold}_{direction}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing E78 calibrated head checkpoint: {path}")
    saved = torch.load(path, map_location=data.raw.device, weights_only=False)
    return saved, {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E78 export requires CUDA")
    device = torch.device("cuda")
    data = load_external_data(BASE, device)
    frames, checked, heads = [], {}, {}
    for fold, roles in enumerate(subject_folds(BASE), start=1):
        model, encoder = load_encoder(data, roles, fold)
        reference_run = RESULTS["runs"][fold - 1]
        if encoder["sha256"] != reference_run["frozen_encoder"]["sha256"]:
            raise ValueError(f"E78 frozen encoder hash mismatch: fold {fold}")
        enrolled = roles["enrolled"]
        for source, target in (("T", "E"), ("E", "T")):
            direction = f"{source}_to_{target}"
            head, head_meta = load_head(data, fold, direction)
            classes = np.asarray(head["classes"], dtype=int)
            if not np.array_equal(classes, enrolled):
                raise ValueError(f"E78 head classes mismatch: fold {fold}/{direction}")
            per_identity = head["per_identity_thresholds"]["q05"]
            if sorted(per_identity) != sorted(str(int(c)) for c in enrolled):
                raise ValueError(f"E78 per-identity threshold keys mismatch: fold {fold}/{direction}")
            means = head["means"].to(device)
            inv = head["inverse_covariance"].to(device)
            # Same split as run_e78.main: run5 outer known/unknown; per-identity q05 from run4.
            outer_known_i = data.indices(enrolled, session=target, runs=[5])
            outer_unknown_i = data.indices(roles["outer_unknown"], session=target, runs=[5])
            known_s = scores(embed(model, data.raw, outer_known_i), means, inv)
            unknown_s = scores(embed(model, data.raw, outer_unknown_i), means, inv)
            known_idx = known_s.argmax(1)
            unknown_idx = unknown_s.argmax(1)
            known_conf = known_s[np.arange(len(known_s)), known_idx]
            unknown_conf = unknown_s[np.arange(len(unknown_s)), unknown_idx]
            known_pred_id = classes[known_idx]
            unknown_pred_id = classes[unknown_idx]
            known_threshold = np.asarray([per_identity[str(int(c))] for c in known_pred_id])
            unknown_threshold = np.asarray([per_identity[str(int(c))] for c in unknown_pred_id])
            known_accept = known_conf >= known_threshold
            unknown_accept = unknown_conf >= unknown_threshold
            known_pred = np.where(known_accept, known_pred_id, 0).astype(int)
            unknown_pred = np.where(unknown_accept, unknown_pred_id, 0).astype(int)
            threshold = np.r_[known_threshold, unknown_threshold].astype(np.float64)
            frame = common.build_records(
                experiment="E78", outer_fold=fold, direction=direction, method=METHOD,
                roles=np.r_[np.full(len(outer_known_i), "enrolled"), np.full(len(outer_unknown_i), "outer_unknown")],
                sample_id=np.r_[sample_ids(data, outer_known_i), sample_ids(data, outer_unknown_i)],
                session=np.r_[np.full(len(outer_known_i), target), np.full(len(outer_unknown_i), target)],
                run=np.r_[data.run[outer_known_i], data.run[outer_unknown_i]],
                task=np.r_[data.task[outer_known_i], data.task[outer_unknown_i]],
                artifact=np.r_[data.artifact[outer_known_i], data.artifact[outer_unknown_i]],
                true_identity=np.r_[data.subject[outer_known_i], data.subject[outer_unknown_i]],
                pred_label=np.r_[known_pred, unknown_pred],
                accepted=np.r_[known_accept, unknown_accept],
                confidence=np.r_[known_conf, unknown_conf],
                threshold=float(threshold[0]),
            )
            frame["threshold"] = threshold
            frames.append(frame)
            reference = reference_run["directions"][direction]["outer"]["q05"]
            recomputed = common.metrics_from_frame(frame)
            if (recomputed["N_known"], recomputed["N_unknown"]) != (reference["N_known"], reference["N_unknown"]):
                raise ValueError(f"E78 sample count mismatch: fold {fold}/{direction}")
            pairs = {"AUROC": "AUROC", "identity_macro_F1": "identity_macro_F1",
                     "unknown_F1": "unknown_F1", "C": "C", "known_acceptance": "known_acceptance",
                     "unknown_false_acceptance": "unknown_false_acceptance",
                     "unknown_rejection": "unknown_rejection"}
            heads[f"fold{fold}/{direction}"] = head_meta
            checked[f"fold{fold}/{direction}"] = {
                key: {"reference": float(reference[ref]), "recomputed": float(recomputed[key]),
                      "abs_diff": abs(float(reference[ref]) - float(recomputed[key]))}
                for key, ref in pairs.items()
            }
            checked[f"fold{fold}/{direction}"]["threshold"] = {
                "reference": "no scalar threshold in E78 results.json",
                "recomputed": {"per_identity_q05": {k: float(v) for k, v in sorted(per_identity.items())},
                               "csv_threshold_min": float(threshold.min()),
                               "csv_threshold_max": float(threshold.max())},
                "abs_diff": 0.0,
            }
        del model
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values())
    threshold_diff = max(group["threshold"]["abs_diff"] for group in checked.values())
    if worst > 1e-9 or threshold_diff > 1e-9:
        raise ValueError(f"CSV 重算指标与 E78 results.json 不一致：max abs diff {worst}, "
                         f"threshold diff {threshold_diff}")
    frame = pd.concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], E78_DIR / "outputs" / "outer_test_predictions.csv")
    groups = common.metrics_by_group(frame)
    summary = {key: float(np.mean([g[key] for g in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "C", "U_at_95",
                           "DIR_at_FPIR_1", "detection_EER")}
    output = {
        "experiment": "E78", "method": METHOD,
        "csv": manifest, "groups": groups, "group_mean": summary,
        "reference_check": {"max_abs_diff": worst, "threshold_max_abs_diff": threshold_diff,
                            "per_group": checked},
        "checkpoints": heads,
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "threshold_definition": "per-identity q05 from enrolled run4 known; CSV threshold column is the "
                                "applied per-sample threshold (class of argmax for unknowns)",
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(E78_DIR / "outputs" / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "max_abs_diff": worst, "n_rows": manifest["n_rows"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
