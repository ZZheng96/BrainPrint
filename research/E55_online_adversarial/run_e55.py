"""E55: true online raw-waveform PGD7 for every fit and OE training batch.

Only Train/session1 is read.  E54 supplies the frozen backbone, OE split and
E36 evaluation primitives; E54's cached PGD3 training path is never called.
Formal FGSM/PGD20 is regenerated after the selected current model is restored.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
CACHE = ROOT / "BrainPrint_cache" / "e55_online_adversarial"
BASE_PATH = ROOT / "research" / "E54_outlier_exposure" / "run_e54.py"
BASE_CONFIG_PATH = BASE_PATH.with_name("config.json")
BASE_RESULTS_PATH = BASE_PATH.with_name("results.json")
MAIN = "main_online_pgd7"
UNTARGETED = "untargeted_online_pgd7"
VARIANTS = (MAIN, UNTARGETED)
EARLY_ATTACKS = ("untargeted_fgsm_snr20", "untargeted_pgd_snr20", "targeted_pgd_snr20")

# Reuse only the explicitly allowed E54 implementation.  Redirect all artifact
# globals before calling any primitive so no E54 result/cache is overwritten.
spec = importlib.util.spec_from_file_location("e55_allowed_e54_primitives", BASE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load allowed E54 primitives: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.OUT = OUT
base.CONFIG_PATH = CONFIG_PATH
base.RESULTS_PATH = RESULTS_PATH
base.CACHE = CACHE


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return base.sha256(path)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def online_training_attack(model: torch.nn.Module, raw: torch.Tensor,
                           labels: torch.Tensor | None, config: dict, seed: int,
                           targeted_fraction: float, oe_batch: bool = False) -> tuple[torch.Tensor, dict]:
    """Current-model PGD7 on raw waveforms; labels=None means current pseudo-labels for OE."""
    steps = int(config["training"]["online_pgd_steps"])
    if steps != 7:
        raise ValueError("E55 training attacks are frozen at exactly seven PGD steps")
    if raw.device.type != "cuda":
        raise ValueError("E55 online attacks must run on CUDA")
    generator = torch.Generator(device=raw.device)
    generator.manual_seed(int(seed))
    snr = torch.where(torch.rand(len(raw), generator=generator, device=raw.device) < 0.5, 20.0, 30.0)
    targeted = torch.rand(len(raw), generator=generator, device=raw.device) < targeted_fraction
    clean = raw.detach().float()
    centered = clean-clean.mean(dim=-1, keepdim=True)
    signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
    radius = signal_norm/(10.0**(snr/20.0))
    delta = torch.randn(clean.shape, generator=generator, device=raw.device)
    delta = delta/delta.reshape(len(clean), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta = delta*radius.view(-1, 1, 1)*torch.rand(
        len(clean), generator=generator, device=raw.device
    ).view(-1, 1, 1)
    was_training = model.training
    model.eval()
    if labels is None:
        # OE-only subjects have no registered identity.  Pseudo-labels only set
        # the local attack objective; they never become training labels/prototypes.
        with torch.no_grad():
            labels = model.scores(clean).argmax(dim=1)
    targets = (labels+1) % model.n_classes
    objective_labels = torch.where(targeted, targets, labels)
    executed = 0
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        per_sample = F.cross_entropy(model.scores(clean+delta)/0.05, objective_labels, reduction="none")
        objective = (per_sample*torch.where(targeted, -1.0, 1.0)).sum()
        gradient = torch.autograd.grad(objective, delta)[0]
        executed += 1
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite E55 online PGD7 gradient")
        direction = gradient/gradient.reshape(len(gradient), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach()+(2.0*radius/steps).view(-1, 1, 1)*direction
        norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
        delta = delta*torch.minimum(torch.ones_like(norm), radius/norm).view(-1, 1, 1)
    attacked = (clean+delta).detach()
    if not torch.isfinite(attacked).all():
        raise FloatingPointError("Non-finite E55 online PGD7 waveform")
    norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
    diagnostics = {
        "n": int(len(raw)), "steps_executed": int(executed),
        "gradient_finite": True, "waveform_finite": True,
        "waveform_changed": bool(torch.any(delta != 0)),
        "pseudo_labeled_oe": bool(oe_batch),
        "targeted_fraction_realized": float(targeted.float().mean()),
        "realized_snr_db_min": float((20*torch.log10(signal_norm/norm)).min()),
        "radius_relative_error_max": float(((norm-radius).abs()/radius.clamp_min(1e-12)).max()),
    }
    if was_training:
        model.train()
    return attacked, diagnostics


def paired_oe_from_embeddings(model: torch.nn.Module, known_embedding: torch.Tensor,
                              unknown_embedding: torch.Tensor, config: dict) -> torch.Tensor:
    """E54 OE formula, reusing fit embeddings to avoid duplicate forwards."""
    cfg = config["outlier_exposure"]
    prototypes = model.normalized_prototypes()
    r_y = (known_embedding@prototypes.T).max(dim=1).values
    r_u = (unknown_embedding@prototypes.detach().T).max(dim=1).values
    paired = F.softplus((r_u-r_y+float(cfg["margin"]))/float(cfg["temperature"])).mean()
    return float(cfg["lambda_u"])*paired


def train_model(variant: str, raw: torch.Tensor, subjects: np.ndarray, fit_np: np.ndarray,
                oe_np: np.ndarray, known_np: np.ndarray | None, unknown_np: np.ndarray | None,
                classes: np.ndarray, metadata, config: dict, outer: int, inner: int,
                fixed_epochs: int | None = None):
    if variant not in VARIANTS:
        raise ValueError("Only the two preregistered E55 variants are allowed")
    if config["training"]["attack_training_mode"] != "true_online_raw_waveform_pgd7_every_fit_and_oe_batch":
        raise ValueError("E55 forbids cached/fallback training attacks")
    seed = int(config["seed_base"]+1000*outer+inner)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = base.EEGNetEmbedding(len(classes), config, task_adversarial=False).to(raw.device)
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    fit_index = torch.as_tensor(fit_np, device=raw.device)
    fit_raw = raw[fit_index]
    fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                                 dtype=torch.long, device=raw.device)
    oe_raw = raw[torch.as_tensor(oe_np, device=raw.device)]
    base.initialize_prototypes(model, fit_raw, fit_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    arpl_weight = float(config["model"]["arpl_weight"])
    targeted_fraction = 0.0 if variant == UNTARGETED else float(config["training"]["main_targeted_fraction"])
    temperature = float(config["model"]["cosine_temperature"])
    margin = float(config["model"]["arpl_margin"])
    batch = int(config["training"]["batch_size"])
    max_epochs = int(fixed_epochs or config["training"]["max_epochs"])
    rng = np.random.default_rng(seed)
    oe_rng = np.random.default_rng(seed+77)
    best = {"key": None, "epoch": max_epochs, "state": None, "early": None}
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, max_epochs+1):
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.time()
        order = rng.permutation(len(fit_np))
        repeats = int(math.ceil(len(fit_np)/len(oe_np)))
        oe_order = np.concatenate([oe_rng.permutation(len(oe_np)) for _ in range(repeats)])[:len(fit_np)]
        epoch_loss = epoch_oe = 0.0
        fit_attack_batches = oe_attack_batches = attack_samples = attack_steps = 0
        expected_batches = int(math.ceil(len(fit_np)/batch))
        for start in range(0, len(order), batch):
            selected = torch.as_tensor(order[start:start+batch], device=raw.device)
            labels = fit_labels[selected]
            clean = fit_raw[selected]
            attack_seed = int(config["seed_base"]+300000+10000*outer+1000*inner+100000*epoch+start)
            attacked, fit_diag = online_training_attack(
                model, clean, labels, config, attack_seed, targeted_fraction
            )
            if fit_diag["steps_executed"] != 7 or not fit_diag["waveform_changed"]:
                raise RuntimeError("A fit mini-batch did not execute online PGD7")
            fit_attack_batches += 1
            attack_samples += len(selected)
            attack_steps += fit_diag["steps_executed"]
            clean_embedding = model.embed(clean)
            adv_embedding = model.embed(attacked)
            clean_loss = base.embedding_loss(model, clean_embedding, labels, temperature, arpl_weight, margin)
            adv_loss = base.embedding_loss(model, adv_embedding, labels, temperature, arpl_weight, margin)
            loss = (float(config["training"]["clean_ce_weight"])*clean_loss
                    + float(config["training"]["adversarial_ce_weight"])*adv_loss)

            oe_selected = torch.as_tensor(oe_order[start:start+len(selected)], device=raw.device)
            oe_clean = oe_raw[oe_selected]
            oe_attacked, oe_diag = online_training_attack(
                model, oe_clean, None, config, attack_seed+50000000, targeted_fraction, oe_batch=True
            )
            if oe_diag["steps_executed"] != 7 or not oe_diag["waveform_changed"]:
                raise RuntimeError("An OE mini-batch did not execute online PGD7")
            oe_attack_batches += 1
            attack_samples += len(oe_selected)
            attack_steps += oe_diag["steps_executed"]
            oe_clean_embedding = model.embed(oe_clean)
            oe_adv_embedding = model.embed(oe_attacked)
            oe_clean_term = paired_oe_from_embeddings(model, clean_embedding, oe_clean_embedding, config)
            oe_adv_term = paired_oe_from_embeddings(model, adv_embedding, oe_adv_embedding, config)
            oe_term = 0.5*(oe_clean_term+oe_adv_term)
            loss = loss+oe_term

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite E55 training loss")
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Non-finite E55 optimizer gradient")
            optimizer.step()
            epoch_loss += float(loss.detach())*len(selected)
            epoch_oe += float(oe_term.detach())*len(selected)
        if (fit_attack_batches != expected_batches or oe_attack_batches != expected_batches
                or attack_steps != 7*(fit_attack_batches+oe_attack_batches)):
            raise RuntimeError("E55 epoch coverage invariant failed for fit/OE online PGD7")
        torch.cuda.synchronize()
        train_seconds = time.time()-epoch_started
        row = {
            "epoch": epoch, "loss": epoch_loss/len(fit_np), "oe_loss": epoch_oe/len(fit_np),
            "seconds": train_seconds, "fit_pgd7_batches": fit_attack_batches,
            "oe_pgd7_batches": oe_attack_batches, "expected_fit_batches": expected_batches,
            "expected_oe_batches": expected_batches, "pgd7_steps_executed": attack_steps,
            "pgd7_steps_per_batch": 7, "pgd7_attack_samples": attack_samples,
            "pgd7_gradient_finite": True, "optimizer_gradient_finite": True,
            "fit_oe_samples_per_second": attack_samples/train_seconds,
            "fit_oe_batches_per_second": (fit_attack_batches+oe_attack_batches)/train_seconds,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
            "targeted_fraction_configured": targeted_fraction,
        }
        if fixed_epochs is None and epoch > 1 and epoch % int(config["training"]["early_eval_every_epochs"]) == 0:
            early = base.make_run(model, raw, metadata, subjects, known_np, unknown_np, classes,
                                  config, 10+epoch, outer, inner,
                                  int(config["training"]["early_trials_per_subject_task"]),
                                  int(config["training"]["early_pgd_steps"]), EARLY_ATTACKS, False)
            selection = base.select_threshold([early], EARLY_ATTACKS, config)
            clean_early = base.clean_metrics([early], selection["score_name"], selection["threshold"],
                                             selection["q05_threshold"])
            key = (selection["macro"]["Score"], clean_early["direct_C"], clean_early["unknown_auroc"],
                   clean_early["q05_U"], -epoch)
            row.update({"Score": key[0], "direct_C": key[1], "unknown_auroc": key[2], "U_at_95": key[3]})
            if best["key"] is None or key > best["key"]:
                best = {"key": key, "epoch": epoch, "early": {"selection": selection, "clean": clean_early},
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(f"  {variant} o{outer}i{inner} epoch={epoch} loss={row['loss']:.4f} "
              f"oe={row['oe_loss']:.4f} fit/OE batches={fit_attack_batches}/{oe_attack_batches} "
              f"PGD7steps={attack_steps} sec={train_seconds:.1f} best={best['epoch']}", flush=True)
        if fixed_epochs is None and stale >= int(config["training"]["early_stopping_patience_evaluations"]):
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    fit_scores = base.predict_scores(model, fit_raw)
    fit_pred = np.asarray(classes)[fit_scores.argmax(1)]
    info = {
        "variant": variant, "seed": seed, "best_epoch": int(best["epoch"]), "history": history,
        "best_early": best["early"], "attack_training_mode": config["training"]["attack_training_mode"],
        "n_fit": int(len(fit_np)), "n_oe": int(len(oe_np)),
        "fit_accuracy": float(np.mean(fit_pred == subjects[fit_np])),
        "fit_identity_macro_f1": float(base.f1_score(subjects[fit_np], fit_pred, labels=classes,
                                                   average="macro", zero_division=0)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "elapsed_seconds": time.time()-started, "model_sha256": model_fingerprint(model),
        "every_fit_and_oe_batch_online_pgd7": all(
            row["fit_pgd7_batches"] == row["expected_fit_batches"]
            and row["oe_pgd7_batches"] == row["expected_oe_batches"]
            and row["pgd7_steps_executed"] == 7*(row["fit_pgd7_batches"]+row["oe_pgd7_batches"])
            and row["pgd7_gradient_finite"] for row in history
        ),
    }
    if not info["every_fit_and_oe_batch_online_pgd7"]:
        raise RuntimeError("E55 did not prove online PGD7 coverage in every epoch")
    del fit_raw, fit_index, oe_raw, optimizer
    torch.cuda.empty_cache()
    return model, info


def artifact_path(variant: str, outer: int, inner: int) -> Path:
    name = f"{variant}_outer{outer}_inner{inner}.pt" if inner else f"{variant}_outer{outer}_full.pt"
    return CACHE/name


def save_artifact(path: Path, model: torch.nn.Module, info: dict, run: dict | None) -> None:
    torch.save({"config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__).resolve()),
                "base_script_sha256": sha256(BASE_PATH),
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": run["classes"] if run is not None else np.arange(1, 21),
                "info": info, "run": run}, path)


def load_artifact(path: Path, device: torch.device, need_model: bool = False):
    item = torch.load(path, map_location="cpu", weights_only=False)
    if (item.get("config_sha256") != sha256(CONFIG_PATH)
            or item.get("script_sha256") != sha256(Path(__file__).resolve())
            or item.get("base_script_sha256") != sha256(BASE_PATH)):
        raise ValueError(f"Frozen E55 implementation changed; rebuild required: {path}")
    model = None
    if need_model:
        model = base.EEGNetEmbedding(len(item["classes"]), load_config(), task_adversarial=False).to(device)
        model.load_state_dict(item["model_state"])
        model.eval()
        if model_fingerprint(model) != item["info"]["model_sha256"]:
            raise ValueError("Artifact model fingerprint mismatch")
    if item["run"] is not None and item["run"].get("attack_model_sha256") != item["info"]["model_sha256"]:
        raise ValueError("Formal attacks were not generated against the saved current model")
    return item["info"], item["run"], model


def inner_layout(metadata, subjects: np.ndarray, outer_unknown: np.ndarray,
                 known_subjects: np.ndarray, inner_unknown: np.ndarray,
                 outer: int, inner: int, config: dict) -> dict:
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8*maximum
    inner_block = (trial > 0.6*maximum) & (trial <= 0.8*maximum)
    roles = base.build_inner_roles(known_subjects, inner_unknown, outer_unknown, outer, inner, config)
    enrolled, oe_only, calibration = roles["enrolled"], roles["oe_only"], roles["calibration"]
    enrolled_mask = np.isin(subjects, enrolled)
    fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
    known_np = np.flatnonzero(enrolled_mask & inner_block)
    oe_np = np.flatnonzero(np.isin(subjects, oe_only) & ~forward & ~inner_block)
    unknown_np = np.flatnonzero(np.isin(subjects, calibration) & ~forward)
    role_payload = {
        "outer_unknown_subjects": np.asarray(outer_unknown).tolist(), "registered_subjects": enrolled.tolist(),
        "oe_only_subjects": oe_only.tolist(), "calibration_eval_unknown_subjects": calibration.tolist(),
        "oe_split_seed": roles["oe_split_seed"], "model_seed": int(config["seed_base"]+1000*outer+inner),
        "n_fit": int(len(fit_np)), "n_oe_available": int(len(oe_np)),
        "n_known_validation": int(len(known_np)), "n_calibration_unknown": int(len(unknown_np)),
    }
    isolation = {
        "subject_roles_pairwise_disjoint": roles["pairwise_disjoint"],
        "subject_role_union_is_all_20": roles["union_is_all_20"],
        "fit_excludes_oe": bool(not np.isin(subjects[fit_np], oe_only).any()),
        "known_validation_excludes_oe": bool(not np.isin(subjects[known_np], oe_only).any()),
        "unknown_evaluation_excludes_oe": bool(not np.isin(subjects[unknown_np], oe_only).any()),
        "unknown_evaluation_is_calibration_only": bool(np.isin(subjects[unknown_np], calibration).all()),
        "oe_is_fit_block_only": bool(np.all(~forward[oe_np] & ~inner_block[oe_np])),
    }
    if not all(isolation.values()):
        raise ValueError(f"E55 data isolation failed: {isolation}")
    return {"roles": roles, "enrolled": enrolled, "fit_np": fit_np, "known_np": known_np,
            "oe_np": oe_np, "unknown_np": unknown_np, "role_payload": role_payload,
            "isolation": isolation}


def formal_run(model, raw, metadata, subjects, layout, config, outer: int, inner: int) -> dict:
    run = base.make_run(model, raw, metadata, subjects, layout["known_np"], layout["unknown_np"],
                        layout["enrolled"], config, 1, outer, inner,
                        int(config["protocol"]["formal_trials_per_subject_task"]),
                        int(config["attacks"]["pgd_steps"]), config["attacks"]["order"], True)
    run.update(layout["role_payload"])
    run["unknown_subjects"] = layout["role_payload"]["calibration_eval_unknown_subjects"]
    run["attack_model_sha256"] = model_fingerprint(model)
    run["attack_config_sha256"] = sha256(CONFIG_PATH)
    run["formal_attack_regenerated_for_current_complete_model"] = True
    run["formal_pgd_steps"] = int(config["attacks"]["pgd_steps"])
    return run


def get_inner_fold(variant: str, outer: int, inner: int, raw, metadata, subjects: np.ndarray,
                   known_subjects: np.ndarray, inner_unknown: np.ndarray, outer_unknown: np.ndarray,
                   config: dict):
    path = artifact_path(variant, outer, inner)
    layout = inner_layout(metadata, subjects, outer_unknown, known_subjects, inner_unknown,
                          outer, inner, config)
    if path.exists():
        try:
            info, run, _ = load_artifact(path, raw.device)
            print(f"Reuse {path.name}", flush=True)
        except (ValueError, KeyError) as error:
            print(f"Rebuild {path.name}: {error}", flush=True)
            path.unlink()
            return get_inner_fold(variant, outer, inner, raw, metadata, subjects, known_subjects,
                                  inner_unknown, outer_unknown, config)
    else:
        model, info = train_model(variant, raw, subjects, layout["fit_np"], layout["oe_np"],
                                  layout["known_np"], layout["unknown_np"], layout["enrolled"],
                                  metadata, config, outer, inner)
        run = formal_run(model, raw, metadata, subjects, layout, config, outer, inner)
        if run["attack_model_sha256"] != info["model_sha256"]:
            raise RuntimeError("Formal attack/model fingerprint mismatch")
        save_artifact(path, model, info, run)
        del model
    info["data_roles"] = layout["role_payload"]
    info["data_isolation"] = {**layout["isolation"], "run_contains_no_oe_scores": "oe_scores" not in run}
    torch.cuda.empty_cache()
    return info, run


def train_fit_summary(infos: list[dict]) -> dict:
    values = np.asarray([info["fit_accuracy"] for info in infos], dtype=float)
    return {"accuracy": float(values.mean()), "minimum_fold_accuracy": float(values.min()),
            "N": int(sum(info["n_fit"] for info in infos)), "folds": int(len(infos)),
            "strict_100pct": bool(np.all(values == 1.0)),
            "definition": "ten inner fit-known accuracies; no all-20 OE model was trained"}


def baseline_result(config: dict) -> dict:
    result = {
        "status": "running", "experiment": "E55 true online PGD7 paired outlier exposure",
        "date": "2026-11-13", "data_evidence": config["reporting"]["evidence_boundary"],
        "config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__).resolve()),
        "base_script_sha256": sha256(BASE_PATH), "base_config_sha256": sha256(BASE_CONFIG_PATH),
        "base_results_sha256": sha256(BASE_RESULTS_PATH),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory/2**20,
                        "compute": "all raw data, online PGD7, models and formal attacks resident on CUDA; metrics on CPU"},
        "variants": {}, "e54_oe_baseline": config["e54_oe_baseline"],
        "outer": "N/A: gated until a preregistered E55 variant passes every inner gate",
        "test_metrics": "N/A: Test was not read and has no true labels",
        "final_20_migration": "N/A: not run in E55; no OE was applied to all 20 registered subjects",
    }
    if RESULTS_PATH.exists():
        try:
            previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if (previous.get("config_sha256") == result["config_sha256"]
                    and previous.get("script_sha256") == result["script_sha256"]
                    and "smoke" in previous):
                result["smoke"] = previous["smoke"]
        except (json.JSONDecodeError, OSError):
            pass
    return result


def run_smoke(raw, metadata, config: dict) -> dict:
    subjects = metadata["subject"].to_numpy(dtype=int)
    all_subjects = np.unique(subjects)
    outer_unknown = list(np.array_split(np.random.default_rng(42).permutation(all_subjects), 5))[0]
    known_subjects = np.setdiff1d(all_subjects, outer_unknown)
    inner_unknown = list(np.array_split(np.random.default_rng(101).permutation(known_subjects), 2))[0]
    layout = inner_layout(metadata, subjects, outer_unknown, known_subjects, inner_unknown, 1, 1, config)
    model, info = train_model(MAIN, raw, subjects, layout["fit_np"], layout["oe_np"],
                              layout["known_np"], layout["unknown_np"], layout["enrolled"],
                              metadata, config, 1, 1, fixed_epochs=int(config["smoke"]["epochs"]))
    info["numerical_checks"] = base.numerical_checks(model, raw, layout["known_np"])
    info["oe_gradient_checks"] = base.oe_gradient_checks(model, raw, layout["fit_np"], layout["oe_np"], config)
    gradient = info["oe_gradient_checks"]
    if not gradient["finite"] or not gradient["nonzero"] or not gradient["unknown_branch_prototype_gradient_zero"]:
        raise FloatingPointError("E55 OE smoke gradient/prototype isolation check failed")
    info["data_roles"] = layout["role_payload"]
    info["data_isolation"] = layout["isolation"]
    result = baseline_result(config)
    result["smoke"] = info
    result["status"] = "smoke_complete"
    write_json(RESULTS_PATH, result)
    del model
    torch.cuda.empty_cache()
    return result


def summarize_variant(pairs: list[dict], infos: list[dict], config: dict) -> dict:
    summary = {
        "macro_fold_mean": base.fold_mean([p["macro"] for p in pairs]),
        "clean_fold_mean": base.fold_mean([p["clean"] for p in pairs]),
        "per_attack_fold_mean": {name: base.fold_mean([p["per_attack"][name] for p in pairs])
                                 for name in config["attacks"]["order"]},
        "train_fit": train_fit_summary(infos),
        "all_rejection": bool(any(p["macro"]["all_rejection"] for p in pairs)),
        "all_fit_and_oe_batches_online_pgd7": bool(all(info["every_fit_and_oe_batch_online_pgd7"] for info in infos)),
        "total_fit_pgd7_batches": int(sum(row["fit_pgd7_batches"] for info in infos for row in info["history"])),
        "total_oe_pgd7_batches": int(sum(row["oe_pgd7_batches"] for info in infos for row in info["history"])),
        "total_online_pgd7_steps": int(sum(row["pgd7_steps_executed"] for info in infos for row in info["history"])),
    }
    summary["gates"] = base.gates_for(summary, config["targets"])
    return summary


def run_inner_stage(raw, metadata, config: dict) -> dict:
    subjects = metadata["subject"].to_numpy(dtype=int)
    all_subjects = np.unique(subjects)
    outer_groups = list(np.array_split(np.random.default_rng(int(config["protocol"]["outer_seed"])).permutation(all_subjects), 5))
    result = baseline_result(config)
    for variant in VARIANTS:
        pairs, all_infos = [], []
        for outer_index, outer_unknown in enumerate(outer_groups):
            outer = outer_index+1
            known_subjects = np.setdiff1d(all_subjects, outer_unknown)
            inner_order = np.random.default_rng(100+outer).permutation(known_subjects)
            splits = list(np.array_split(inner_order, 2))
            runs, infos = [], []
            for inner_index, inner_unknown in enumerate(splits):
                inner = inner_index+1
                info, run = get_inner_fold(variant, outer, inner, raw, metadata, subjects,
                                           known_subjects, inner_unknown, outer_unknown, config)
                infos.append(info)
                runs.append(run)
                all_infos.append(info)
            selection = base.select_threshold(runs, config["attacks"]["order"], config)
            evaluated = base.evaluate_point(runs, selection["score_name"], selection["threshold"], config["attacks"]["order"])
            clean = base.clean_metrics(runs, selection["score_name"], selection["threshold"], selection["q05_threshold"])
            pair = {"outer_fold": outer, "train_info": infos, "selection": selection,
                    "clean": clean, **evaluated,
                    "attack_diagnostics": [{name: run["attacks"][name]["diagnostics"]
                                            for name in config["attacks"]["order"]} for run in runs]}
            pairs.append(pair)
            print(f"INNER {variant} outer={outer}: Score={evaluated['macro']['Score']:.3f} "
                  f"directRA={evaluated['macro']['direct_RA']:.4f} RA={evaluated['macro']['RA']:.4f} "
                  f"ASafeR={evaluated['macro']['ASafeR']:.4f} C={clean['direct_C']:.4f} "
                  f"AUROC={clean['unknown_auroc']:.4f} U@95={clean['q05_U']:.4f}", flush=True)
            result["variants"][variant] = {"folds": pairs}
            result["status"] = "inner_partial"
            write_json(RESULTS_PATH, result)
        result["variants"][variant].update(summarize_variant(pairs, all_infos, config))
        result["status"] = "inner_partial" if variant != VARIANTS[-1] else "inner_complete"
        write_json(RESULTS_PATH, result)
    passing = [name for name in VARIANTS if result["variants"][name]["gates"]["all_pass"]]
    result["selected_variant_for_outer"] = (max(passing, key=lambda name: result["variants"][name]["macro_fold_mean"]["Score"])
                                             if passing else None)
    result["best_inner_variant_by_score"] = max(VARIANTS, key=lambda name: result["variants"][name]["macro_fold_mean"]["Score"])
    if not passing:
        result["outer"] = "N/A: no E55 variant passed every inner gate; outer was not run"
        result["status"] = "complete_inner_gates_failed"
    write_json(RESULTS_PATH, result)
    return result


def run_outer_stage(result: dict, raw, metadata, config: dict) -> dict:
    if result.get("selected_variant_for_outer") is None:
        result["outer"] = "N/A: no E55 variant passed every inner gate; outer was not run"
        result["status"] = "complete_inner_gates_failed"
        write_json(RESULTS_PATH, result)
        return result
    # E54's frozen transfer protocol uses the redirected E55 artifact paths and
    # regenerates PGD20 against each transferred current complete model.
    result = base.run_outer_stage(result, raw, metadata, config)
    result["outer"]["selected_variant"] = result["selected_variant_for_outer"]
    write_json(RESULTS_PATH, result)
    return result


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def summary_row(name: str, item: dict, baseline: bool = False) -> str:
    clean, macro = item["clean_fold_mean"], item["macro_fold_mean"]
    fit = item["train_fit"]["accuracy"]
    return (f"| {name} | {fmt(fit)} | {fmt(clean['direct_C'])} | {fmt(macro['C'])} | "
            f"{fmt(clean['identity_macro_f1_direct'])}/{fmt(clean['identity_macro_f1'])} | {fmt(clean['continuous_block_accuracy'])} | "
            f"{fmt(macro['identity_macro_f1_direct'])}/{fmt(macro['identity_macro_f1'])} | "
            f"{fmt(clean['unknown_auroc'])}/{fmt(clean['unknown_aupr'])} | {fmt(macro['unknown_auroc'])}/{fmt(macro['unknown_aupr'])} | "
            f"{fmt(clean['q05_U'])} | {fmt(macro['U'])} | {fmt(macro['unknown_f1'])} | {fmt(macro['joint_f1'])} | "
            f"{fmt(macro['direct_RA'])} | {fmt(macro['RA'])} | {fmt(macro['ASafeR'])} | {fmt(macro['A'])} | "
            f"{fmt(macro['Score'])} | {macro['N_adversarial_total']:.0f} |")


def write_report(result: dict, config: dict) -> None:
    lines = [
        "# E55 全 fit/OE 当前批次在线原始波形 PGD7 纯 EEG 实验", "",
        "日期：2026-11-13。仅使用 Train Session1；Test 未读取。", "",
        "## 实际执行与复现", "", "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e55.py'}\" --stage smoke",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e55.py'}\" --stage inner",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e55.py'}\" --stage outer  # 仅变体全部门禁通过时执行",
        "```", "",
        f"状态：`{result['status']}`。实现：`{OUT / 'run_e55.py'}`；配置：`{CONFIG_PATH}`；完整结果：`{RESULTS_PATH}`。", "",
        "- E55唯一主要改动：每个当前fit mini-batch及配对OE mini-batch均在GPU上对当前模型执行真实原始波形PGD7；没有调用E54的PGD3波形缓存或refresh路径。",
        "- fit的clean/adv身份+ARPL损失各0.5；OE保持λu=0.1、m=0.2、T=0.05及r_u原型detach，clean/adv配对OE项各0.5。OE攻击只用当前模型伪标签，不设未知类或新原型。",
        "- 主变体目标攻击比例0.5；唯一消融untargeted_online_pgd7比例0。早停轻量PGD3不变，正式评估在恢复早停最佳完整模型后重新生成FGSM/PGD20。",
        "- 阈值、E36六攻击宏指标、互斥8 known/4 OE-only/4 calibration unknown、SNR20/30、40 epoch上限与早停协议均与E54一致。", "",
        "## 十 inner 折汇总及E54同划分OE基线", "",
        "AUC/AUROC为known=1、unknown=0的检测指标；AUPR为unknown类。clean F1在连续块全量已知上计算；攻击F1/AUC/AUPR为六攻击宏均值。N为每折六攻击对抗样本合计的折均值。", "",
        "| 变体 | Train fit | direct C | 阈值C | clean F1 direct/拒识 | 连续块 | 攻击F1 direct/拒识 | clean AUROC/AUPR | 攻击AUROC/AUPR | U@95 | 工作点U | unknown F1 | joint F1 | direct RA | 阈值RA | ASafeR | A | Score | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_row("E54 OE（PGD3缓存，参考）", config["e54_oe_baseline"], True),
    ]
    for name, item in result.get("variants", {}).items():
        if "train_fit" in item:
            lines.append(summary_row(name, item))
    for variant, item in result.get("variants", {}).items():
        if "per_attack_fold_mean" not in item:
            continue
        lines.extend(["", f"## {variant} 逐攻击（inner折均值）", "",
                      "| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for attack, row in item["per_attack_fold_mean"].items():
            lines.append(f"| {attack} | {fmt(row['C'])} | {fmt(row['direct_RA'])} | {fmt(row['RA'])} | {fmt(row['ASafeR'])} | "
                         f"{fmt(row['A'])} | {fmt(row['identity_macro_f1_direct'])}/{fmt(row['identity_macro_f1'])} | "
                         f"{fmt(row['unknown_auroc'])} | {fmt(row['unknown_aupr'])} | {fmt(row['unknown_f1'])} | {fmt(row['U'])} | "
                         f"{fmt(row['joint_f1'])} | {fmt(row['Score'])} | {row['N_adversarial']:.0f} |")
    role_variant = result.get("variants", {}).get(MAIN)
    if role_variant and role_variant.get("folds"):
        lines.extend(["", "## 数据角色与隔离", "",
                      "每折outer unknown 4人、registered/fit-known 8人、OE-only 4人、calibration/eval unknown 4人两两互斥。划分种子与E54相同；fit/OE仅取≤60%，known为60–80%，calibration unknown≤80%；outer封存。", "",
                      "| outer | inner | outer unknown ID | registered ID | OE-only ID | calibration/eval unknown ID | OE split seed | model seed |",
                      "|---:|---:|---|---|---|---|---:|---:|"])
        for pair in role_variant["folds"]:
            for inner_index, info in enumerate(pair["train_info"], start=1):
                role = info["data_roles"]
                ids = lambda key: ",".join(str(v) for v in role[key])
                lines.append(f"| {pair['outer_fold']} | {inner_index} | {ids('outer_unknown_subjects')} | {ids('registered_subjects')} | "
                             f"{ids('oe_only_subjects')} | {ids('calibration_eval_unknown_subjects')} | {role['oe_split_seed']} | {role['model_seed']} |")
        rows = [info["data_isolation"] for pair in role_variant["folds"] for info in pair["train_info"]]
        lines.extend(["", f"- 全部已执行折的数据隔离断言通过：{all(all(row.values()) for row in rows)}；OE-only没有评分数组，也不参与原型初始化、阈值或unknown评估。"])
    lines.extend(["", "## 门禁", "",
                  "| 变体 | Train fit | direct/阈值C | 身份F1 | 连续块 | AUROC | U@95 | direct RA | 阈值RA | ASafeR | 非全拒识 | 全部 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"])
    for name, item in result.get("variants", {}).items():
        gates = item.get("gates")
        if not gates:
            continue
        marks = lambda key: "通过" if gates[key] else "未通过"
        lines.append(f"| {name} | {marks('train_fit_100pct')} | {marks('direct_C')}/{marks('threshold_C')} | "
                     f"{marks('identity_f1_direct')} | {marks('continuous_block_accuracy')} | {marks('unknown_auroc')} | "
                     f"{marks('U_at_95')} | {marks('direct_RA')} | {marks('threshold_RA')} | {marks('ASafeR')} | "
                     f"{marks('not_all_rejection')} | {marks('all_pass')} |")
    if result.get("best_inner_variant_by_score"):
        lines.extend(["", f"- best_inner_variant_by_score={result.get('best_inner_variant_by_score')}；selected_variant_for_outer={result.get('selected_variant_for_outer')}。",
                      "- direct RA未达0.95时不宣称鲁棒达标；任何变体未通过全部门禁均不触发outer。仅比较两个预注册变体，无额外网格。"])
    smoke = result.get("smoke")
    if smoke:
        lines.extend(["", "## SMOKE：逐epoch在线PGD7覆盖、显存与吞吐", "",
                      "| epoch | fit PGD7 batches | OE PGD7 batches | 实际PGD步数 | 梯度finite | GPU峰值allocated/reserved MiB | fit+OE samples/s | 训练秒数 |",
                      "|---:|---:|---:|---:|---|---:|---:|---:|"])
        for row in smoke["history"]:
            lines.append(f"| {row['epoch']} | {row['fit_pgd7_batches']}/{row['expected_fit_batches']} | "
                         f"{row['oe_pgd7_batches']}/{row['expected_oe_batches']} | {row['pgd7_steps_executed']} | "
                         f"{row['pgd7_gradient_finite'] and row['optimizer_gradient_finite']} | "
                         f"{row['gpu_peak_allocated_mib']:.1f}/{row['gpu_peak_reserved_mib']:.1f} | "
                         f"{row['fit_oe_samples_per_second']:.1f} | {row['seconds']:.2f} |")
        check = smoke["numerical_checks"]
        gradient = smoke["oe_gradient_checks"]
        lines.extend(["", f"- smoke覆盖断言={smoke['every_fit_and_oe_batch_online_pgd7']}；参数量={smoke['parameter_count']}。",
                      f"- 单样本/批次最大分数差={check['single_vs_batch_max_abs']:.3g}；置换差={check['permutation_max_abs']:.3g}；有限差分相对误差={check['finite_difference_relative_error']:.3g}。",
                      f"- OE梯度finite/nonzero={gradient['finite']}/{gradient['nonzero']}；r_u原型梯度L2={gradient['unknown_branch_prototype_gradient_l2']:.3g}。"])
    if isinstance(result.get("outer"), dict) and "macro_fold_mean" in result["outer"]:
        outer = result["outer"]
        lines.extend(["", "## Outer 一次性冻结确认", "",
                      f"macro：direct RA={fmt(outer['macro_fold_mean']['direct_RA'])}，RA={fmt(outer['macro_fold_mean']['RA'])}，"
                      f"ASafeR={fmt(outer['macro_fold_mean']['ASafeR'])}，Score={fmt(outer['macro_fold_mean']['Score'])}；"
                      f"clean AUROC={fmt(outer['clean_fold_mean']['unknown_auroc'])}，direct F1={fmt(outer['clean_fold_mean']['identity_macro_f1_direct'])}。"])
    else:
        lines.extend(["", "## Outer 与 Test", "", f"- Outer：{result.get('outer', 'N/A')}。",
                      "- Test AUC、F1、C、RA、ASafeR、A、U、Score：N/A（无真实标签，且未读取Test）。"])
    lines.extend(["", "## 证据边界", "",
                  "- unknown指标只来自Train calibration/eval pseudo-unknown，不等于官方新被试；OE-only只参与训练，不进入未知评估。连续文件块是时间隔离代理，不是跨天/跨Session证据。",
                  "- Train fit是10个inner fit-known准确率均值及严格逐折100%门禁，不是全20类OE模型或泛化证据。SMOKE只检查攻击覆盖、梯度、显存/吞吐和数值，不用于选模。",
                  "- 正式攻击对保存信息所指纹标识的当前完整模型生成；复用 artifact 时同时校验E55配置、脚本、E54基础脚本、模型与攻击指纹，任一变化即重建。",
                  "- 本地攻击覆盖FGSM、非目标/目标PGD20×SNR30/20，不保证覆盖全部可能扰动；Score是Train内代理，不是官方Test得分。若门禁失败，本报告为负结果，不削弱攻击或借助outer/Test调参。", ""])
    (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "inner", "outer", "all", "report"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E55 requires CUDA")
    config = load_config()
    CACHE.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed_base"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    if args.stage == "report":
        result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        write_report(result, config)
        return
    metadata = base.train_metadata(config)
    raw = base.load_all_raw(metadata, config, device)
    if args.stage in ("smoke", "all"):
        result = run_smoke(raw, metadata, config)
        write_report(result, config)
    if args.stage in ("inner", "all"):
        result = run_inner_stage(raw, metadata, config)
        write_report(result, config)
    if args.stage in ("outer", "all"):
        if args.stage == "outer":
            result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if result["status"] not in ("inner_complete", "complete_inner_gates_failed"):
                raise ValueError("Run the complete inner stage first")
            if (result["config_sha256"] != sha256(CONFIG_PATH)
                    or result["script_sha256"] != sha256(Path(__file__).resolve())
                    or result["base_script_sha256"] != sha256(BASE_PATH)):
                raise ValueError("Implementation changed after inner freeze")
        result = run_outer_stage(result, raw, metadata, config)
        write_report(result, config)
    print(f"Results: {RESULTS_PATH}", flush=True)
    print(f"Report: {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
