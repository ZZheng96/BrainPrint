"""E73c: train only a generic pair head on frozen native22 SupCon embeddings.

All fold heads and known-only thresholds are serialized before outer run-5
queries are scored.  No calibration-unknown or outer-unknown sample is used for
training, calibration, model selection, or threshold selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from external_data import ExternalData, load_external_data, subject_folds
from pair_head import SampleTemplatePairHead, same_different_labels, subject_templates
from pretrain import (
    E73_DIR,
    eer,
    embed_indices,
    make_model,
    numeric_mean,
    operating_metrics,
    pair_metrics,
    sha256,
    utc_now,
    write_json,
)
from pair_protocol import evaluate_queries

DIRECTIONS = (("T", "E"), ("E", "T"))


def resolve_config(name: str) -> Path:
    path = (E73_DIR / name).resolve()
    if not path.is_relative_to(E73_DIR) or not path.is_file():
        raise ValueError("Configuration must be an existing JSON inside E73")
    return path


def training_index_plan(data: ExternalData, roles: dict[str, np.ndarray],
                        config: dict) -> list[dict]:
    """Construct the complete training plan using enrolled run 0-2 data only."""
    plan = []
    for source, target in DIRECTIONS:
        for run in config["training"]["train_runs"]:
            template = data.indices(roles["enrolled"], session=source, runs=[int(run)])
            query = data.indices(roles["enrolled"], session=target, runs=[int(run)])
            if not len(template) or not len(query):
                raise ValueError("Empty pair-head training role")
            plan.append({"source": source, "target": target, "run": int(run),
                         "template": template, "query": query})
    return plan


def load_frozen_encoder(base_config: dict, base_path: Path, roles: dict[str, np.ndarray],
                        outer_fold: int, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = E73_DIR / base_config.get("output_subdir", "") / "models" / f"supcon_outer{outer_fold}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing frozen E73b checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    expected_hashes = {
        "config": sha256(base_path),
        "pretrain": sha256(E73_DIR / "pretrain.py"),
        "external_data": sha256(E73_DIR / "external_data.py"),
        "native_model": sha256(E73_DIR / "native_model.py"),
    }
    if saved.get("source_hashes") != expected_hashes:
        raise ValueError("E73b checkpoint no longer matches its frozen sources")
    expected_roles = {key: value.tolist() for key, value in roles.items()}
    if saved.get("roles") != expected_roles:
        raise ValueError("E73b checkpoint subject roles differ from the fixed folds")
    model = make_model(base_config, device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(checkpoint.relative_to(E73_DIR)), "sha256": sha256(checkpoint)}


def fit_pair_head(model: torch.nn.Module, data: ExternalData, roles: dict[str, np.ndarray],
                  base_config: dict, pair_config: dict, pair_path: Path,
                  outer_fold: int, checkpoint: Path, force: bool) -> tuple[SampleTemplatePairHead, dict]:
    cfg = pair_config["pair_head"]
    seed = int(pair_config["seed_base"]) + 1000 * outer_fold
    source_hashes = {
        "pair_config": sha256(pair_path), "pair_head": sha256(E73_DIR / "pair_head.py"),
        "run_pair": sha256(Path(__file__).resolve()),
    }
    if checkpoint.is_file() and not force:
        saved = torch.load(checkpoint, map_location=data.raw.device, weights_only=False)
        if saved.get("source_hashes") != source_hashes:
            raise ValueError("Pair checkpoint source mismatch; use --force deliberately")
        head = SampleTemplatePairHead(int(cfg["embedding_dim"]), int(cfg["hidden_dim"])).to(data.raw.device)
        head.load_state_dict(saved["head_state"])
        head.eval()
        return head, saved["training_info"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    groups = []
    for item in training_index_plan(data, roles, base_config):
        query = embed_indices(model, data.raw, item["query"], base_config)
        enrollment = embed_indices(model, data.raw, item["template"], base_config)
        template = subject_templates(enrollment, data.subject[item["template"]], roles["enrolled"])
        labels = same_different_labels(data.subject[item["query"]], roles["enrolled"], data.raw.device)
        groups.append((query, template, labels))

    head = SampleTemplatePairHead(int(cfg["embedding_dim"]), int(cfg["hidden_dim"])).to(data.raw.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    positive_weight = torch.tensor(float(cfg["positive_class_weight"]), device=data.raw.device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    rng = np.random.default_rng(seed)
    history = []
    started = time.time()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        head.train()
        total_loss = 0.0
        for group_index in rng.permutation(len(groups)):
            query, template, labels = groups[int(group_index)]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(query, template), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pair-head loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        history.append({"epoch": epoch, "loss": total_loss / len(groups)})
        if epoch == 1 or epoch % 10 == 0 or epoch == int(cfg["max_epochs"]):
            print(f"E73c outer={outer_fold} epoch={epoch} loss={history[-1]['loss']:.5f}", flush=True)
    head.eval()
    info = {
        "seed": seed, "epochs": int(cfg["max_epochs"]),
        "parameter_count": int(sum(parameter.numel() for parameter in head.parameters())),
        "n_query_template_pairs": int(sum(group[2].numel() for group in groups)),
        "n_training_queries": int(sum(len(group[0]) for group in groups)),
        "history": history, "elapsed_seconds": time.time() - started,
        "unknown_training_samples": 0,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": pair_config["experiment"], "outer_fold": outer_fold,
        "created_at": utc_now(), "head_state": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "roles": {key: value.tolist() for key, value in roles.items()},
        "source_hashes": source_hashes, "config": pair_config, "training_info": info,
    }, checkpoint)
    return head, info


def pair_scores(head: SampleTemplatePairHead, query: torch.Tensor,
                template: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(query, template).float().cpu().numpy()


def freeze_calibration(model: torch.nn.Module, head: SampleTemplatePairHead,
                       data: ExternalData, roles: dict[str, np.ndarray], base_config: dict,
                       pair_config: dict, outer_fold: int, output_dir: Path) -> tuple[dict, dict]:
    calibrations, templates = {}, {}
    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        template_index = data.indices(roles["enrolled"], session=source,
                                      runs=[int(pair_config["protocol"]["template_run"])])
        known_index = data.indices(roles["enrolled"], session=target,
                                   runs=[int(pair_config["protocol"]["calibration_known_query_run"])])
        enrollment = embed_indices(model, data.raw, template_index, base_config)
        template = subject_templates(enrollment, data.subject[template_index], roles["enrolled"])
        known = embed_indices(model, data.raw, known_index, base_config)
        scores = pair_scores(head, known, template)
        confidence = scores.max(axis=1)
        calibrations[direction] = {
            "N_known": int(len(known_index)),
            "q05": float(np.quantile(confidence, 0.05, method="linear")),
            "q02": float(np.quantile(confidence, 0.02, method="linear")),
            "unknown_samples_used": 0,
        }
        templates[direction] = template
    calibration_path = output_dir / "models" / f"calibration_outer{outer_fold}.json"
    write_json(calibration_path, {
        "experiment": pair_config["experiment"], "outer_fold": outer_fold,
        "created_at": utc_now(), "head_sha256": sha256(output_dir / "models" / f"pair_outer{outer_fold}.pt"),
        "calibrations": calibrations, "outer_scored": False,
    })
    return calibrations, templates


def evaluate_outer(model: torch.nn.Module, head: SampleTemplatePairHead, data: ExternalData,
                   roles: dict[str, np.ndarray], base_config: dict, pair_config: dict,
                   calibrations: dict, templates: dict) -> dict:
    output = {}
    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        known_index = data.indices(roles["enrolled"], session=target,
                                   runs=[int(pair_config["protocol"]["evaluation_query_run"])])
        unknown_index = data.indices(roles["outer_unknown"], session=target,
                                     runs=[int(pair_config["protocol"]["evaluation_query_run"])])
        known = embed_indices(model, data.raw, known_index, base_config)
        unknown = embed_indices(model, data.raw, unknown_index, base_config)
        known_scores = pair_scores(head, known, templates[direction])
        unknown_scores = pair_scores(head, unknown, templates[direction])
        threshold = float(calibrations[direction]["q05"])
        # max_cosine is the shared helper's generic row-maximum operation; values here are logits.
        outer, _ = evaluate_queries(data.subject[known_index], known_scores, unknown_scores,
                                    roles["enrolled"], "max_cosine", threshold)
        known_confidence = known_scores.max(axis=1)
        unknown_confidence = unknown_scores.max(axis=1)
        direct = roles["enrolled"][known_scores.argmax(axis=1)]
        frozen_q02 = float(calibrations[direction]["q02"])
        outer["working_98_threshold"] = frozen_q02
        outer["working_98_known_acceptance"] = float(np.mean(known_confidence >= frozen_q02))
        outer["working_98_unknown_rejection"] = float(np.mean(unknown_confidence < frozen_q02))
        outer.update(pair_metrics(known_scores, unknown_scores, data.subject[known_index], roles["enrolled"]))
        outer["detection_EER"] = eer(known_confidence, unknown_confidence)
        outer.update(operating_metrics(known_confidence, unknown_confidence,
                                       direct == data.subject[known_index]))
        output[direction] = {"inner": calibrations[direction], "outer": outer}
    return output


def write_pair_report(results: dict, path: Path) -> None:
    primary = results["primary_summary"]
    lines = [
        "# E73c native22 冻结表征 + 通用样本—模板匹配头", "", "## 协议", "",
        "- 仅改变匹配头；E73b native22 SupCon 编码器、三外层折和 run 隔离全部冻结。",
        "- 匹配头仅使用 enrolled 的 run0–2 跨 Session same/different 对；不使用两类 unknown 训练或调参。",
        "- 所有折的匹配头及 run4 已知样本 q05/q02 阈值先落盘，再一次性评估 run5 外层已知/未知样本。",
        "", "## 主结果（3 外层折 × 2 方向的均值）", "", "| 指标 | 数值 |", "|---|---:|",
        f"| Pair AUC | {primary['pair_AUROC']:.6f} |",
        f"| Detection AUC | {primary['AUROC']:.6f} |",
        f"| Detection EER | {primary['detection_EER']:.6f} |",
        f"| C | {primary['C']:.6f} |",
        f"| Identity macro-F1 | {primary['identity_macro_F1']:.6f} |",
        f"| Unknown-F1 | {primary['unknown_F1']:.6f} |",
        f"| U@95 | {primary['U_at_95']:.6f} |",
        f"| 冻结 q02 的已知接受率 | {primary['working_98_known_acceptance']:.6f} |",
        f"| 冻结 q02 的未知拒识率 | {primary['working_98_unknown_rejection']:.6f} |",
        f"| DIR@FPIR1% | {primary['DIR_at_FPIR_1']:.6f} |", "", "## 阶段门禁", "",
    ]
    lines.extend(f"- {'通过' if value else '未通过'}：`{key}`" for key, value in results["gates"].items())
    lines.extend(["", f"**总体结论：{'通过' if results['stage_passed'] else '未通过'} E73 阶段出口。**", "",
                  "Official Score=N/A；Test AUC/F1=N/A（未访问竞赛 Train/Test）。", "",
                  "模型、逐折数值与训练历史见 models/ 和 results.json。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_pair.json")
    parser.add_argument("--force", action="store_true", help="Retrain pair heads only; never retrain the encoder")
    parser.add_argument("--smoke", action="store_true", help="Two epochs/one fold; no outer scoring")
    args = parser.parse_args()
    pair_path = resolve_config(args.config)
    pair_config = json.loads(pair_path.read_text(encoding="utf-8"))
    base_path = resolve_config(pair_config["base_config"])
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    output_dir = (E73_DIR / pair_config["output_subdir"]).resolve()
    if not output_dir.is_relative_to(E73_DIR):
        raise ValueError("Output directory escapes E73")
    if args.smoke:
        pair_config["pair_head"]["max_epochs"] = 2
        output_dir = E73_DIR / "cache" / "smoke" / "native22_pair"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("E73c requires CUDA")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = load_external_data(base_config, device)
    folds = subject_folds(base_config)
    if args.smoke:
        folds = folds[:1]
    prepared = []
    for outer_fold, roles in enumerate(folds, start=1):
        model, encoder_info = load_frozen_encoder(base_config, base_path, roles, outer_fold, device)
        head, training_info = fit_pair_head(
            model, data, roles, base_config, pair_config, pair_path, outer_fold,
            output_dir / "models" / f"pair_outer{outer_fold}.pt", args.force,
        )
        calibrations, templates = freeze_calibration(
            model, head, data, roles, base_config, pair_config, outer_fold, output_dir
        )
        prepared.append((outer_fold, roles, model, head, encoder_info, training_info, calibrations, templates))
        print(f"E73c froze outer={outer_fold} head and known-only thresholds", flush=True)
    if args.smoke:
        write_json(output_dir / "results.json", {
            "experiment": pair_config["experiment"], "smoke": True,
            "scope": "enrolled training and known-only calibration; outer not scored",
            "folds": [item[0] for item in prepared], "test_accessed": False,
        })
        print("E73c smoke completed without outer evaluation", flush=True)
        return

    run_records = []
    for outer_fold, roles, model, head, encoder_info, training_info, calibrations, templates in prepared:
        directions = evaluate_outer(model, head, data, roles, base_config, pair_config, calibrations, templates)
        run_records.append({
            "outer_fold": outer_fold, "roles": {key: value.tolist() for key, value in roles.items()},
            "frozen_encoder": encoder_info, "pair_training": training_info, "directions": directions,
        })
        print(f"E73c evaluated frozen outer={outer_fold}", flush=True)
    primary_rows = [record["directions"][direction]["outer"]
                    for record in run_records for direction in record["directions"]]
    primary_summary = numeric_mean(primary_rows)
    gate_config = base_config["gates"]
    gates = {
        "pair_AUROC": bool(primary_summary["pair_AUROC"] >= float(gate_config["pair_AUROC_min"])),
        "detection_AUROC": bool(primary_summary["AUROC"] >= float(gate_config["detection_AUROC_min"])),
        "detection_EER": bool(primary_summary["detection_EER"] <= float(gate_config["detection_EER_max"])),
        "working_98_known_acceptance": bool(primary_summary["working_98_known_acceptance"] >= float(gate_config["known_acceptance_at_frozen_inner_q02_min"])),
        "working_98_unknown_rejection": bool(primary_summary["working_98_unknown_rejection"] >= float(gate_config["unknown_rejection_at_frozen_inner_q02_min"])),
    }
    results = {
        "schema_version": 1, "experiment": pair_config["experiment"], "completed_at": utc_now(),
        "config_sha256": sha256(pair_path), "base_config_sha256": sha256(base_path),
        "source_sha256": {"run_pair.py": sha256(Path(__file__).resolve()),
                          "pair_head.py": sha256(E73_DIR / "pair_head.py")},
        "data": {"dataset": base_config["data"]["dataset"], "N": int(len(data.subject)),
                 "competition_train_accessed": False, "competition_test_accessed": False},
        "runs": run_records, "primary_method": "pair_head:max_logit",
        "primary_summary": primary_summary, "gates": gates,
        "stage_passed": bool(all(gates.values())), "official_score": "N/A",
        "test": {"AUC": "N/A", "F1": "N/A", "accessed": False},
    }
    write_json(output_dir / "results.json", results)
    write_pair_report(results, output_dir / "report.md")
    print(json.dumps({"primary_summary": primary_summary, "gates": gates}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
