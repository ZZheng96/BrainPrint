"""Export E77 outer-test per-sample predictions (cosine/Mahalanobis fusion, q05)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E77_DIR = ROOT / "research" / "E77_score_fusion"
for path in (E86_DIR, E77_DIR, ROOT, ROOT / "research" / "E74_multiscale_native22",
             ROOT / "research" / "E73_external_multisession"):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from run_e77 import CFG, BASE, cosine, embed, fuse, load_encoder, mahal, sha256  # noqa: E402

METHOD = "fusion_q05"
RESULTS = json.loads((E77_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def load_head(data, fold: int, direction: str):
    path = E77_DIR / "outputs" / "models" / f"fusion_outer{fold}_{direction}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing E77 fusion checkpoint: {path}")
    saved = torch.load(path, map_location=data.raw.device, weights_only=False)
    return saved, {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E77 export requires CUDA")
    device = torch.device("cuda")
    data = load_external_data(BASE, device)
    frames, checked, heads = [], {}, {}
    for fold, roles in enumerate(subject_folds(BASE), start=1):
        model, encoder = load_encoder(data, roles, fold)
        reference_run = RESULTS["runs"][fold - 1]
        if encoder["sha256"] != reference_run["frozen_encoder"]["sha256"]:
            raise ValueError(f"E77 frozen encoder hash mismatch: fold {fold}")
        enrolled = roles["enrolled"]
        for source, target in (("T", "E"), ("E", "T")):
            direction = f"{source}_to_{target}"
            head, head_meta = load_head(data, fold, direction)
            classes = np.asarray(head["classes"], dtype=int)
            if not np.array_equal(classes, enrolled):
                raise ValueError(f"E77 head classes mismatch: fold {fold}/{direction}")
            if head["weights"] != CFG["fusion"]["weights"]:
                raise ValueError(f"E77 fusion weights mismatch: fold {fold}/{direction}")
            means = head["means"].to(device)
            inv = head["inverse_covariance"].to(device)
            normalizer = head["normalizer"]
            # Same split as run_e77.main: run4 known calibration, run5 outer known/unknown.
            known_i = data.indices(enrolled, session=target, runs=[4])
            outer_known_i = data.indices(enrolled, session=target, runs=[5])
            outer_unknown_i = data.indices(roles["outer_unknown"], session=target, runs=[5])
            known_e = embed(model, data.raw, known_i)
            known_s = fuse(cosine(known_e, means), mahal(known_e, means, inv), normalizer)
            outer_known_e = embed(model, data.raw, outer_known_i)
            outer_known_s = fuse(cosine(outer_known_e, means), mahal(outer_known_e, means, inv), normalizer)
            outer_unknown_e = embed(model, data.raw, outer_unknown_i)
            outer_unknown_s = fuse(cosine(outer_unknown_e, means), mahal(outer_unknown_e, means, inv), normalizer)
            threshold = float(np.quantile(known_s.max(1), 0.05))
            known_conf = outer_known_s.max(1)
            unknown_conf = outer_unknown_s.max(1)
            known_pred = np.where(known_conf >= threshold, classes[outer_known_s.argmax(1)], 0).astype(int)
            unknown_pred = np.where(unknown_conf >= threshold, classes[outer_unknown_s.argmax(1)], 0).astype(int)
            frame = common.build_records(
                experiment="E77", outer_fold=fold, direction=direction, method=METHOD,
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
            reference = reference_run["directions"][direction]["outer"]["q05"]
            recomputed = common.metrics_from_frame(frame)
            if (recomputed["N_known"], recomputed["N_unknown"]) != (reference["N_known"], reference["N_unknown"]):
                raise ValueError(f"E77 sample count mismatch: fold {fold}/{direction}")
            pairs = {"AUROC": "AUROC", "identity_macro_F1": "identity_macro_F1",
                     "unknown_F1": "unknown_F1", "known_acceptance": "known_acceptance",
                     "unknown_false_acceptance": "unknown_false_acceptance",
                     "unknown_rejection": "unknown_rejection", "threshold": "threshold"}
            heads[f"fold{fold}/{direction}"] = head_meta
            checked[f"fold{fold}/{direction}"] = {
                key: {"reference": float(reference[ref]), "recomputed": float(recomputed[key]),
                      "abs_diff": abs(float(reference[ref]) - float(recomputed[key]))}
                for key, ref in pairs.items()
            }
        del model
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values())
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E77 results.json 不一致：max abs diff {worst}")
    frame = pd.concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], E77_DIR / "outputs" / "outer_test_predictions.csv")
    groups = common.metrics_by_group(frame)
    summary = {key: float(np.mean([g[key] for g in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "C", "U_at_95",
                           "DIR_at_FPIR_1", "detection_EER")}
    output = {
        "experiment": "E77", "method": METHOD,
        "csv": manifest, "groups": groups, "group_mean": summary,
        "reference_check": {"max_abs_diff": worst, "per_group": checked},
        "checkpoints": heads,
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "threshold_definition": "q05 of enrolled run4 known fused confidence; identical across all rows",
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(E77_DIR / "outputs" / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "max_abs_diff": worst, "n_rows": manifest["n_rows"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
