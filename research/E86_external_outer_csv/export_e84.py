"""Export E84 outer-test per-sample predictions (frozen E74 encoder + hard-negative pair head).

E84 main outer threshold is the frozen run4 q05; the frozen q02 working threshold
is exported as a second method row set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
E84_DIR = ROOT / "research" / "E84_hard_negative_pair"
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
for path in (E86_DIR, E84_DIR, E74_DIR, ROOT / "research" / "E72_pair_protocol",
             ROOT / "research" / "E73_external_multisession"):
    sys.path.insert(0, str(path))

import common  # noqa: E402
from external_data import load_external_data, subject_folds  # noqa: E402
from pair_head import SampleTemplatePairHead, subject_templates  # noqa: E402
from run_e74 import embed_indices, sha256  # noqa: E402
from pretrain import pair_metrics  # noqa: E402
from run_e84 import load_frozen_encoder, pair_scores  # noqa: E402

BASE_CONFIG = json.loads((E74_DIR / "config.json").read_text(encoding="utf-8-sig"))
PAIR_CONFIG = json.loads((E84_DIR / "config.json").read_text(encoding="utf-8"))
RESULTS = json.loads((E84_DIR / "outputs" / "results.json").read_text(encoding="utf-8"))
OUTPUT_DIR = E84_DIR / "outputs"
METHODS = {"q05": "pair_q05", "q02": "pair_q02"}


def sample_ids(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([f"{data.source_file[i]}#{i}" for i in indices])


def compare_entry(reference: float, recomputed: float) -> dict:
    reference, recomputed = float(reference), float(recomputed)
    return {"reference": reference, "recomputed": recomputed,
            "abs_diff": abs(reference - recomputed)}


def load_pair_head(data, roles: dict, path: Path) -> tuple[torch.nn.Module, dict]:
    """直接加载已保存的 pair head 权重（绝不重训）。

    E84 的 pair checkpoint 记录了训练时 run_e84.py 的 sha256，与当前仓库版本不同
    （训练后脚本被改动），因此这里不复用 run_e84.fit_pair_head 的哈希门禁，而是校验
    roles、config 与 pair_head.py 哈希后直接 load_state_dict。
    """
    saved = torch.load(path, map_location=data.raw.device, weights_only=False)
    if saved.get("roles") != {key: value.tolist() for key, value in roles.items()}:
        raise ValueError(f"E84 pair checkpoint roles mismatch: {path}")
    hashes = saved.get("source_hashes", {})
    if hashes.get("pair_config") != sha256(E84_DIR / "config.json"):
        raise ValueError(f"E84 pair checkpoint config mismatch: {path}")
    if hashes.get("pair_head") != sha256(E84_DIR / "pair_head.py"):
        raise ValueError(f"E84 pair checkpoint pair_head mismatch: {path}")
    cfg = PAIR_CONFIG["pair_head"]
    head = SampleTemplatePairHead(int(cfg["embedding_dim"]), int(cfg["hidden_dim"])).to(data.raw.device)
    head.load_state_dict(saved["head_state"])
    head.eval()
    info = {"path": str(path), "sha256": sha256(path),
            "source_hashes": hashes, "current_run_e84_sha256": sha256(E84_DIR / "run_e84.py")}
    return head, info


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E86 E84 export requires CUDA")
    device = torch.device("cuda")
    data = load_external_data(BASE_CONFIG, device)
    protocol = PAIR_CONFIG["protocol"]
    frames, checked, checkpoints = [], {}, {}
    for fold, roles in enumerate(subject_folds(BASE_CONFIG), start=1):
        model, _ = load_frozen_encoder(BASE_CONFIG, E74_DIR / "config.json", roles, fold, device)
        head, head_info = load_pair_head(
            data, roles, OUTPUT_DIR / "models" / f"pair_outer{fold}.pt",
        )
        checkpoints[f"fold{fold}"] = head_info
        calibrations = json.loads(
            (OUTPUT_DIR / "models" / f"calibration_outer{fold}.json").read_text(encoding="utf-8")
        )["calibrations"]
        enrolled = roles["enrolled"]
        for source, target in (("T", "E"), ("E", "T")):
            template_i = data.indices(enrolled, session=source, runs=[int(protocol["template_run"])])
            known_i = data.indices(enrolled, session=target, runs=[int(protocol["evaluation_query_run"])])
            unknown_i = data.indices(roles["outer_unknown"], session=target,
                                     runs=[int(protocol["evaluation_query_run"])])
            template = subject_templates(
                embed_indices(model, data.raw, template_i, BASE_CONFIG),
                data.subject[template_i], enrolled,
            )
            known_scores = pair_scores(
                head, embed_indices(model, data.raw, known_i, BASE_CONFIG), template
            )
            unknown_scores = pair_scores(
                head, embed_indices(model, data.raw, unknown_i, BASE_CONFIG), template
            )
            known_conf = known_scores.max(axis=1)
            unknown_conf = unknown_scores.max(axis=1)
            direct_known = enrolled[known_scores.argmax(axis=1)]
            direct_unknown = enrolled[unknown_scores.argmax(axis=1)]
            direction = f"{source}_to_{target}"
            reference = RESULTS["runs"][fold - 1]["directions"][direction]["outer"]
            # pair_AUROC 需要完整 K 类分数矩阵，无法从逐样本 CSV 重算；用重建的分数矩阵直接计算。
            pair_auroc = float(pair_metrics(known_scores, unknown_scores,
                                            data.subject[known_i], enrolled)["pair_AUROC"])
            for rule, method in METHODS.items():
                threshold = float(calibrations[direction][rule])
                known_pred = np.where(known_conf >= threshold, direct_known, 0).astype(int)
                unknown_pred = np.where(unknown_conf >= threshold, direct_unknown, 0).astype(int)
                frame = common.build_records(
                    experiment="E84", outer_fold=fold, direction=direction, method=method,
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
                if rule == "q05":
                    entry = {
                        "AUROC": compare_entry(reference["AUROC"], recomputed["AUROC"]),
                        "identity_macro_F1": compare_entry(reference["identity_macro_F1"],
                                                           recomputed["identity_macro_F1"]),
                        "unknown_F1": compare_entry(reference["unknown_F1"], recomputed["unknown_F1"]),
                        "known_acceptance": compare_entry(reference["known_acceptance"],
                                                          recomputed["known_acceptance"]),
                        "unknown_false_acceptance": compare_entry(1.0 - reference["U"],
                                                                  recomputed["unknown_false_acceptance"]),
                        "C": compare_entry(reference["C"], recomputed["C"]),
                        "pair_AUROC_from_scores": compare_entry(reference["pair_AUROC"], pair_auroc),
                        "threshold": compare_entry(reference["threshold"], recomputed["threshold"]),
                    }
                else:
                    entry = {
                        "known_acceptance": compare_entry(reference["working_98_known_acceptance"],
                                                          recomputed["known_acceptance"]),
                        "unknown_rejection": compare_entry(reference["working_98_unknown_rejection"],
                                                           recomputed["unknown_rejection"]),
                        "threshold": compare_entry(reference["working_98_threshold"],
                                                   recomputed["threshold"]),
                    }
                checked[f"fold{fold}/{direction}/{method}"] = entry
        del model
        torch.cuda.empty_cache()
    worst = max(item["abs_diff"] for group in checked.values() for item in group.values())
    if worst > 1e-9:
        raise ValueError(f"CSV 重算指标与 E84 results.json 不一致：max abs diff {worst}")
    frame = pd.concat(frames, ignore_index=True)
    manifest = common.write_csv([frame], OUTPUT_DIR / "outer_test_predictions.csv")
    primary = frame[frame["method"] == METHODS["q05"]]
    groups = common.metrics_by_group(primary)
    summary = {key: float(np.mean([group[key] for group in groups.values()]))
               for key in ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
                           "unknown_false_acceptance", "unknown_rejection", "U_at_95",
                           "DIR_at_FPIR_1", "detection_EER")}
    output = {
        "experiment": "E84", "method": METHODS["q05"],
        "methods": list(METHODS.values()),
        "method_note": ("results.json 的 outer 主指标使用冻结 run4 q05 阈值（pair_q05）；"
                        "pair_q02 为同分数下冻结 q02 working 阈值，对应 working_98_* 字段"),
        "csv": manifest, "groups": groups, "group_mean": summary,
        "reference_check": {"max_abs_diff": worst, "per_group": checked},
        "checkpoints": checkpoints,
        "checkpoint_note": ("E84 pair checkpoint 训练时记录的 run_e84.py sha256 与当前文件不同；"
                            "导出脚本直接加载权重并校验 roles/config/pair_head 哈希，未重训"),
        "role_definition": {"enrolled": "run5 known query", "outer_unknown": "run5 held-out subject query"},
        "competition_test": "N/A; 外部 BCIC IV-2a outer test，未访问项目 Test",
    }
    common.write_json(OUTPUT_DIR / "outer_test_metrics.json", output)
    print(json.dumps({"summary": summary, "max_abs_diff": worst,
                      "n_rows": manifest["n_rows"], "n_groups": manifest["n_groups"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
