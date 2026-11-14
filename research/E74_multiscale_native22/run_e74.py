﻿"""Train and evaluate E74 with fixed cross-session-only supervised contrastive loss.

Only derived public BCIC IV-2a files under this experiment are read.  The three
outer folds and all hyperparameters are fixed before outer evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

E74_DIR = Path(__file__).resolve().parent
ROOT = E74_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(ROOT / "research" / "E72_pair_protocol"))
sys.path.insert(0, str(ROOT / "research" / "E73_external_multisession"))
from external_data import ExternalData, load_external_data, subject_folds  # noqa: E402
from multiscale_model import MultiScaleNative22  # noqa: E402
from pair_protocol import (  # noqa: E402
    confidence_values,
    evaluate_queries,
    fit_spaces,
    make_template_pairs,
    predict_spaces,
)
from run_e54 import EEGNetEmbedding  # noqa: E402

CONFIG_PATH = E74_DIR / "config.json"
RESULTS_PATH = E74_DIR / "results.json"
REPORT_PATH = E74_DIR / "report.md"
MODEL_DIR = E74_DIR / "models"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_model(config: dict, device: torch.device) -> MultiScaleNative22:
    return MultiScaleNative22(config).to(device)


def cross_session_supcon(embedding: torch.Tensor, subject: torch.Tensor,
                         session: torch.Tensor, temperature: float) -> torch.Tensor:
    """Each anchor is contrasted only with same-subject opposite-session positives."""
    z = torch.nn.functional.normalize(embedding.float(), dim=1, eps=1e-12)
    logits = z @ z.T / float(temperature)
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (subject[:, None] == subject[None, :]) & (session[:, None] != session[None, :])
    if not torch.all(positives.sum(dim=1) > 0):
        raise ValueError("Every SupCon anchor needs an opposite-session same-subject positive")
    logits = logits.masked_fill(self_mask, -torch.inf)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probability = torch.where(
        positives, log_probability, torch.zeros_like(log_probability)
    ).sum(dim=1)
    return -(positive_log_probability / positives.sum(dim=1)).mean()


def encoder_parameters(model: torch.nn.Module):
    return [parameter for name, parameter in model.named_parameters()
            if not name.startswith(("prototypes", "radius_raw", "task_head"))]


def training_groups(data: ExternalData, enrolled: np.ndarray, config: dict) -> list[np.ndarray]:
    groups = []
    for subject in enrolled:
        for session in config["data"]["sessions"]:
            index = data.indices(
                np.asarray([subject]), session=session, runs=config["training"]["train_runs"]
            )
            if not len(index):
                raise ValueError("Empty cross-session training group")
            groups.append(index)
    return groups


def fit_diagnostic(model: MultiScaleNative22, data: ExternalData, enrolled: np.ndarray,
                   config: dict) -> dict:
    outputs = {}
    for source, target in (("T", "E"), ("E", "T")):
        template_index = data.indices(enrolled, session=source, runs=config["training"]["train_runs"])
        query_index = data.indices(enrolled, session=target, runs=config["training"]["train_runs"])
        template = embed_indices(model, data.raw, template_index, config)
        query = embed_indices(model, data.raw, query_index, config)
        templates = []
        for subject in enrolled:
            selected = torch.as_tensor(data.subject[template_index] == subject, device=template.device)
            templates.append(template[selected].mean(dim=0))
        prototypes = torch.nn.functional.normalize(torch.stack(templates), dim=1)
        scores = torch.nn.functional.normalize(query, dim=1) @ prototypes.T
        predicted = np.asarray(enrolled)[scores.argmax(dim=1).cpu().numpy()]
        true = data.subject[query_index]
        outputs[f"{source}_to_{target}"] = {
            "C": float(np.mean(predicted == true)),
            "identity_macro_F1": float(f1_score(true, predicted, labels=enrolled, average="macro", zero_division=0)),
        }
    return outputs


def train_encoder(data: ExternalData, roles: dict[str, np.ndarray], config: dict,
                  outer_fold: int, checkpoint: Path, force: bool = False,
                  config_path: Path = CONFIG_PATH) -> tuple[EEGNetEmbedding, dict]:
    seed = int(config["seed_base"]) + 1000 * outer_fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(config, data.raw.device)
    source_hashes = {
        "config": sha256(config_path),
        "run_e74": sha256(Path(__file__).resolve()),
        "external_data": sha256(E74_DIR / "external_data.py"),
        "multiscale_model": sha256(E74_DIR / "multiscale_model.py"),
    }
    if checkpoint.is_file() and not force:
        saved = torch.load(checkpoint, map_location=data.raw.device, weights_only=False)
        if saved.get("source_hashes") != source_hashes:
            raise ValueError(f"Checkpoint source hash mismatch; use --force deliberately: {checkpoint}")
        model.load_state_dict(saved["model_state"])
        model.eval()
        return model, saved["training_info"]

    optimizer = torch.optim.AdamW(
        encoder_parameters(model),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    groups = training_groups(data, roles["enrolled"], config)
    k = int(config["training"]["trials_per_subject_session"])
    class_lookup = {int(subject): index for index, subject in enumerate(roles["enrolled"])}
    rng = np.random.default_rng(seed)
    history = []
    started = time.time()
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        permutations = [rng.permutation(group) for group in groups]
        steps = min(len(group) for group in permutations) // k
        epoch_loss = 0.0
        epoch_started = time.time()
        for step in range(steps):
            selected_np = np.concatenate([group[step * k:(step + 1) * k] for group in permutations])
            selected = torch.as_tensor(selected_np, dtype=torch.long, device=data.raw.device)
            subjects = torch.as_tensor(
                [class_lookup[int(value)] for value in data.subject[selected_np]],
                dtype=torch.long, device=data.raw.device,
            )
            sessions = torch.as_tensor(data.session[selected_np], dtype=torch.long, device=data.raw.device)
            optimizer.zero_grad(set_to_none=True)
            amp = bool(config["training"]["amp"]) and data.raw.device.type == "cuda"
            with torch.autocast(device_type=data.raw.device.type, dtype=torch.float16, enabled=amp):
                embedding = model.embed(data.raw[selected])
            loss = cross_session_supcon(
                embedding, subjects, sessions, float(config["training"]["temperature"])
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite E74 SupCon loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_parameters(model), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(selected_np)
        row = {
            "epoch": epoch,
            "loss": epoch_loss / (steps * len(groups) * k),
            "steps": steps,
            "seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(f"E74 outer={outer_fold} epoch={epoch} loss={row['loss']:.5f} sec={row['seconds']:.2f}", flush=True)
    model.eval()
    info = {
        "seed": seed,
        "epochs": int(config["training"]["max_epochs"]),
        "parameter_count": int(sum(parameter.numel() for parameter in encoder_parameters(model))),
        "n_encoder_train_trials": int(sum(len(group) for group in groups)),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "fit_diagnostic_not_generalization": fit_diagnostic(model, data, roles["enrolled"], config),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "E74",
        "outer_fold": outer_fold,
        "created_at": utc_now(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "roles": {key: value.tolist() for key, value in roles.items()},
        "source_hashes": source_hashes,
        "config": config,
        "training_info": info,
    }, checkpoint)
    return model, info


def embed_indices(model: MultiScaleNative22, raw: torch.Tensor, indices: np.ndarray,
                  config: dict) -> torch.Tensor:
    outputs = []
    batch = int(config["features"]["prediction_batch_size"])
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            selected = torch.as_tensor(indices[start:start + batch], dtype=torch.long, device=raw.device)
            outputs.append(model.embed(raw[selected]).float())
    if not outputs:
        raise ValueError("Cannot embed an empty index set")
    return torch.cat(outputs)


def eer(known: np.ndarray, unknown: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)])
    scores = np.concatenate([known, unknown])
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.min(np.maximum(fpr, 1.0 - tpr)))


def pair_metrics(known_scores: np.ndarray, unknown_scores: np.ndarray,
                 known_true: np.ndarray, classes: np.ndarray) -> dict:
    scores = np.concatenate([known_scores, unknown_scores])
    true = np.concatenate([known_true, np.zeros(len(unknown_scores), dtype=int)])
    pairs = make_template_pairs(scores, classes, true)
    flat_scores = scores.reshape(-1)
    fpr, tpr, _ = roc_curve(pairs["same"], flat_scores)
    return {
        "pair_AUROC": float(roc_auc_score(pairs["same"], flat_scores)),
        "pair_EER": float(np.min(np.maximum(fpr, 1.0 - tpr))),
    }


def operating_metrics(known: np.ndarray, unknown: np.ndarray,
                      known_correct: np.ndarray) -> dict:
    result = {}
    for acceptance in (0.95, 0.98):
        threshold = float(np.quantile(known, 1.0 - acceptance, method="linear"))
        result[f"U_at_{int(100 * acceptance)}"] = float(np.mean(unknown < threshold))
    for fpir in (0.01, 0.05):
        threshold = float(np.quantile(unknown, 1.0 - fpir, method="linear"))
        accepted = known >= threshold
        result[f"DIR_at_FPIR_{int(100 * fpir)}"] = float(np.mean(known_correct & accepted))
    return result


def evaluate_direction(model: MultiScaleNative22, data: ExternalData, roles: dict[str, np.ndarray],
                       source: str, target: str, config: dict,
                       development_only: bool = False) -> dict:
    protocol = config["protocol"]
    enrolled = roles["enrolled"]
    template_index = data.indices(enrolled, session=source, runs=[int(protocol["template_run"])])
    inner_known_index = data.indices(enrolled, session=target, runs=[int(protocol["calibration_known_query_run"])])
    outer_known_index = data.indices(enrolled, session=target, runs=[int(protocol["evaluation_query_run"])])
    inner_unknown_index = data.indices(roles["calibration_unknown"], session=target,
                                       runs=[int(protocol["calibration_unknown_query_run"])])
    outer_unknown_index = data.indices(roles["outer_unknown"], session=target,
                                       runs=[int(protocol["outer_unknown_query_run"])])
    indices = ([template_index, inner_known_index, inner_unknown_index] if development_only else
               [template_index, inner_known_index, outer_known_index, inner_unknown_index, outer_unknown_index])
    if any(len(index) == 0 for index in indices):
        raise ValueError("Empty E74 evaluation role")
    embeddings = [embed_indices(model, data.raw, index, config) for index in indices]
    spaces = fit_spaces(embeddings[0], data.subject[template_index], enrolled, config)
    scores = [predict_spaces(spaces, embedding, np.arange(len(embedding)), len(embedding))
              for embedding in embeddings[1:]]
    output = {"N": {
        "template": len(template_index), "inner_known": len(inner_known_index),
        "inner_unknown": len(inner_unknown_index),
        "outer_known": 0 if development_only else len(outer_known_index),
        "outer_unknown": 0 if development_only else len(outer_unknown_index),
    }, "methods": {}}
    for method in protocol["methods"]:
        space, score_name = method["space"], method["score"]
        inner_confidence = confidence_values(scores[0][space], score_name)
        threshold = float(np.quantile(inner_confidence, 0.05, method="linear"))
        inner_unknown_scores = scores[1 if development_only else 2][space]
        inner, _ = evaluate_queries(
            data.subject[inner_known_index], scores[0][space], inner_unknown_scores,
            enrolled, score_name, threshold
        )
        if development_only:
            output["methods"][f"{space}:{score_name}"] = {
                "space": space, "score": score_name, "inner": inner,
                "outer": "not_accessed_in_smoke",
            }
            continue
        outer, _ = evaluate_queries(
            data.subject[outer_known_index], scores[1][space], scores[3][space], enrolled, score_name, threshold
        )
        known_confidence = confidence_values(scores[1][space], score_name)
        unknown_confidence = confidence_values(scores[3][space], score_name)
        frozen_q02 = float(np.quantile(inner_confidence, 0.02, method="linear"))
        outer["working_98_threshold"] = frozen_q02
        outer["working_98_known_acceptance"] = float(np.mean(known_confidence >= frozen_q02))
        outer["working_98_unknown_rejection"] = float(np.mean(unknown_confidence < frozen_q02))
        direct = enrolled[scores[1][space].argmax(axis=1)]
        outer.update(pair_metrics(scores[1][space], scores[3][space], data.subject[outer_known_index], enrolled))
        outer["detection_EER"] = eer(known_confidence, unknown_confidence)
        outer.update(operating_metrics(
            known_confidence, unknown_confidence, direct == data.subject[outer_known_index]
        ))
        output["methods"][f"{space}:{score_name}"] = {
            "space": space, "score": score_name, "inner": inner, "outer": outer
        }
    return output


def numeric_mean(rows: list[dict]) -> dict:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    }


def write_report(results: dict, config: dict, path: Path = REPORT_PATH) -> None:
    primary = results["primary_summary"]
    gates = results["gates"]
    lines = [
        "# E74 外部跨 Session 身份验证实验",
        "",
        "## 协议",
        "",
        "- 数据：公开 BCIC IV-2a，9 名被试，T/E 双 Session；未读取竞赛 Train/Test。",
        "- 模型：E54 EEGNet 编码器，单损失 cross-session-only SupCon，固定训练轮数。",
        "- 隔离：run 0–2 训练、run 3 注册模板、run 4 内层标定、run 5 最终查询；三外层折的未知被试完全留出。",
        "- 方向：T→E 与 E→T；主方法在任何外层评估前固定为 WCCN + top1/top2 margin。",
        "",
        "## 主结果（3 外层折 × 2 方向的均值）",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Pair AUC | {primary['pair_AUROC']:.6f} |",
        f"| Detection AUC | {primary['AUROC']:.6f} |",
        f"| Detection EER | {primary['detection_EER']:.6f} |",
        f"| C | {primary['C']:.6f} |",
        f"| Identity macro-F1 | {primary['identity_macro_F1']:.6f} |",
        f"| Unknown-F1 | {primary['unknown_F1']:.6f} |",
        f"| U@95 | {primary['U_at_95']:.6f} |",
        f"| U@98（描述性） | {primary['U_at_98']:.6f} |",
        f"| 冻结 q02 的已知接受率 | {primary['working_98_known_acceptance']:.6f} |",
        f"| 冻结 q02 的未知拒识率 | {primary['working_98_unknown_rejection']:.6f} |",
        f"| DIR@FPIR1% | {primary['DIR_at_FPIR_1']:.6f} |",
        "",
        "## 阶段门禁",
        "",
    ]
    lines.extend(f"- {'通过' if value else '未通过'}：`{key}`" for key, value in gates.items())
    lines.extend([
        "",
        f"**总体结论：{'通过' if results['stage_passed'] else '未通过'} E74 阶段出口。**",
        "",
        "Official Score=N/A（无对抗评估）；Test AUC/F1=N/A（未访问 Test）。",
        "",
        "逐折数值、所有预声明评分方法、训练历史和模型哈希见 results.json 与 models/。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Experiment JSON inside E74")
    parser.add_argument("--force", action="store_true", help="Retrain even when a source-matched checkpoint exists")
    parser.add_argument("--smoke", action="store_true", help="One fold/two epochs; writes only ignored cache artifacts")
    args = parser.parse_args()
    config_path = (E74_DIR / args.config).resolve()
    if not config_path.is_relative_to(E74_DIR) or not config_path.is_file():
        raise ValueError("E74 config must be an existing JSON inside this experiment")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output_dir = (E74_DIR / config.get("output_subdir", "")).resolve()
    if not output_dir.is_relative_to(E74_DIR):
        raise ValueError("E74 output directory escapes the experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        config["training"]["max_epochs"] = 2
    if not torch.cuda.is_available():
        raise RuntimeError("E74 formal training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = load_external_data(config, device)
    folds = subject_folds(config)
    if args.smoke:
        folds = folds[:1]
    print(f"E74 raw={tuple(data.raw.shape)} device={device} folds={len(folds)}", flush=True)
    run_records = []
    for outer_fold, roles in enumerate(folds, start=1):
        if args.smoke:
            checkpoint = E74_DIR / "cache" / "smoke" / config.get("output_subdir", "target4") / f"supcon_outer{outer_fold}.pt"
        else:
            checkpoint = output_dir / "models" / f"supcon_outer{outer_fold}.pt"
        model, training_info = train_encoder(
            data, roles, config, outer_fold, checkpoint, args.force, config_path
        )
        record = {
            "outer_fold": outer_fold,
            "roles": {key: value.tolist() for key, value in roles.items()},
            "training": training_info,
            "directions": {},
        }
        for source, target in (("T", "E"), ("E", "T")):
            direction = f"{source}_to_{target}"
            record["directions"][direction] = evaluate_direction(
                model, data, roles, source, target, config, development_only=args.smoke
            )
            print(f"E74 evaluated outer={outer_fold} {direction} smoke={args.smoke}", flush=True)
        run_records.append(record)
        del model
        torch.cuda.empty_cache()
    if args.smoke:
        write_json(E74_DIR / "cache" / "smoke" / config.get("output_subdir", "target4") / "results.json", {
            "experiment": config["experiment"], "completed_at": utc_now(), "smoke": True,
            "scope": "inner known/calibration unknown only; outer metrics not computed",
            "runs": run_records, "test_accessed": False,
        })
        print("E74 smoke completed without outer evaluation", flush=True)
        return
    primary = config["protocol"]["primary_method"]
    method_key = f"{primary['space']}:{primary['score']}"
    primary_rows = [
        record["directions"][direction]["methods"][method_key]["outer"]
        for record in run_records for direction in record["directions"]
    ]
    primary_summary = numeric_mean(primary_rows)
    gate_config = config["gates"]
    gates = {
        "pair_AUROC": bool(primary_summary["pair_AUROC"] >= float(gate_config["pair_AUROC_min"])),
        "detection_AUROC": bool(primary_summary["AUROC"] >= float(gate_config["detection_AUROC_min"])),
        "detection_EER": bool(primary_summary["detection_EER"] <= float(gate_config["detection_EER_max"])),
        "working_98_known_acceptance": bool(primary_summary["working_98_known_acceptance"] >= float(gate_config["known_acceptance_at_frozen_inner_q02_min"])),
        "working_98_unknown_rejection": bool(primary_summary["working_98_unknown_rejection"] >= float(gate_config["unknown_rejection_at_frozen_inner_q02_min"])),
    }
    results = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "completed_at": utc_now(),
        "smoke": args.smoke,
        "config_sha256": sha256(config_path),
        "source_sha256": {"run_e74.py": sha256(Path(__file__).resolve()),
                          "external_data.py": sha256(E74_DIR / "external_data.py"),
                          "multiscale_model.py": sha256(E74_DIR / "multiscale_model.py")},
        "data": {
            "dataset": config["data"]["dataset"], "N": int(len(data.subject)),
            "subjects": sorted(set(data.subject.tolist())),
            "artifact_trials_retained": int(data.artifact.sum()),
            "observed_target_channels": int(data.channel_mask.sum()),
            "competition_train_accessed": False, "competition_test_accessed": False,
        },
        "runs": run_records,
        "primary_method": method_key,
        "primary_summary": primary_summary,
        "gates": gates,
        "stage_passed": bool(all(gates.values())),
        "official_score": "N/A",
        "test": {"AUC": "N/A", "F1": "N/A", "accessed": False},
    }
    write_json(output_dir / "results.json", results)
    write_report(results, config, output_dir / "report.md")
    print(json.dumps({"primary_summary": primary_summary, "gates": gates}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

