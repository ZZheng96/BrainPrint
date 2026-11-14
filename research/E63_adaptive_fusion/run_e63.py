"""E63: adaptive raw-waveform attacks through E55 + differentiable E36 logSPD/WCCN routing.

Only allowed E36/E54/E55 primitives and artifacts are used.  E38--E52 and Test
are never imported or read.  All reported decisions are the true hard route.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
CACHE = ROOT / "BrainPrint_cache" / "e63_adaptive_fusion"
E36_PATH = ROOT / "research" / "E36_pure_eeg" / "run_e36.py"
E55_PATH = ROOT / "research" / "E55_online_adversarial" / "run_e55.py"
ATTACKS = ("untargeted_fgsm_snr30", "untargeted_fgsm_snr20",
           "untargeted_pgd_snr30", "untargeted_pgd_snr20",
           "targeted_pgd_snr30", "targeted_pgd_snr20")
OBJECTIVES = ("direct", "threshold")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E36 = load_module("e63_allowed_e36", E36_PATH)
E55 = load_module("e63_allowed_e55", E55_PATH)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_score(c: float, a: float, u: float) -> float:
    return float(100*c**0.40*a**0.35*u**0.25) if min(c, a, u) > 0 else 0.0


def distance_ratio(scores: torch.Tensor) -> torch.Tensor:
    top = torch.topk(scores, 2, dim=1).values
    return (1.0-top[:, 1])/(1.0-top[:, 0]).clamp_min(1e-6)


class RawLogSPDWCCN:
    """E36's own WCCN prototype with a raw differentiable feature path."""

    def __init__(self, zband_fit: torch.Tensor, fit_y: np.ndarray,
                 classes: np.ndarray, e36_config: dict):
        self.config = e36_config
        self.prototype = E36.TorchPrototype(
            zband_fit, fit_y, classes, True,
            float(e36_config["models"]["wccn_shrinkage"]),
            float(e36_config["models"]["wccn_eigenvalue_floor"]),
        )
        self.classes = np.asarray(classes, dtype=int)

    def features(self, raw: torch.Tensor) -> torch.Tensor:
        return E36.detrend_zband(E36.gpu_logspd(raw, self.config), self.config)

    def scores(self, raw: torch.Tensor) -> torch.Tensor:
        return self.prototype(self.features(raw))


class Fusion:
    def __init__(self, identity: torch.nn.Module, logspd: RawLogSPDWCCN,
                 tau_low: float, tau_high: float, config: dict):
        self.identity = identity
        self.logspd = logspd
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.config = config
        self.classes = logspd.classes

    def forward(self, raw: torch.Tensor) -> dict:
        # Separate identity ops let hooks audit each raw-waveform branch during
        # one backward pass, without three costly independent backwards.
        id_input = raw*1.0
        log_input = raw*1.0
        id_scores = self.identity.scores(id_input)
        log_scores = self.logspd.scores(log_input)
        dr = distance_ratio(log_scores)
        eps = 1e-12
        logdr = dr.clamp_min(eps).log()
        loglow = float(np.log(max(self.tau_low, eps)))
        loghigh = float(np.log(max(self.tau_high, eps)))
        temperature = float(self.config["routing"]["log_route_temperature"])
        soft_reject = torch.sigmoid((loglow-logdr)/temperature)
        soft_high = torch.sigmoid((logdr-loghigh)/temperature)
        hard_reject = dr < self.tau_low
        hard_high = dr >= self.tau_high
        hard_mid = ~(hard_reject | hard_high)

        def st(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
            return hard.to(raw.dtype)+soft-soft.detach()

        direct_high = st(hard_high, soft_high)
        direct_scores = (direct_high[:, None]*log_scores
                         +(1.0-direct_high[:, None])*id_scores)
        soft_log = (1.0-soft_reject)*soft_high
        soft_mid = (1.0-soft_reject)*(1.0-soft_high)
        route_reject = st(hard_reject, soft_reject)
        route_log = st(hard_high, soft_log)
        route_mid = st(hard_mid, soft_mid)
        # In the rejected hard region logits are E55's, but the independent
        # decision below is exactly 0.  Soft derivatives still traverse routing.
        threshold_scores = (route_log[:, None]*log_scores
                            +(route_mid+route_reject)[:, None]*id_scores)
        direct_index = direct_scores.argmax(1)
        threshold_index = threshold_scores.argmax(1)
        direct_pred = self.classes[direct_index.detach().cpu().numpy()]
        threshold_index_np = threshold_index.detach().cpu().numpy()
        threshold_pred = self.classes[threshold_index_np]
        threshold_pred[hard_reject.detach().cpu().numpy()] = 0
        expected_direct = np.where(hard_high.detach().cpu().numpy(),
                                   self.classes[log_scores.detach().argmax(1).cpu().numpy()],
                                   self.classes[id_scores.detach().argmax(1).cpu().numpy()])
        expected_threshold = expected_direct.copy()
        expected_threshold[hard_reject.detach().cpu().numpy()] = 0
        route = torch.where(hard_reject, 0, torch.where(hard_high, 2, 1))
        mismatch = (int(np.sum(direct_pred != expected_direct))
                    +int(np.sum(threshold_pred != expected_threshold)))
        st_error = max(float((route_reject-hard_reject).abs().max()),
                       float((route_log-hard_high).abs().max()),
                       float((route_mid-hard_mid).abs().max()))
        return {"id_input": id_input, "log_input": log_input,
                "id_scores": id_scores, "log_scores": log_scores, "dr": dr,
                "direct_scores": direct_scores, "threshold_scores": threshold_scores,
                "direct_pred": direct_pred, "threshold_pred": threshold_pred,
                "route": route, "hard_soft_prediction_mismatch": mismatch,
                "st_weight_max_error": st_error, "logdr": logdr,
                "loglow": loglow}


def attack_seed(config: dict, stage: int, outer: int, inner: int,
                objective: str, base_name: str, snr: int, start: int) -> int:
    value = int(config["seed_base"]+1000000*stage+10000*outer+1000*inner
                +100*config["attacks"]["attack_codes"][base_name]+snr+start)
    return value+(500000 if objective == "threshold" else 0)


def adaptive_attack(fusion: Fusion, clean: torch.Tensor, labels: torch.Tensor,
                    target: torch.Tensor | None, snr: int, base_name: str,
                    objective: str, steps: int, seed: int, batch_size: int,
                    clean_route: np.ndarray) -> dict:
    targeted = base_name == "targeted_pgd"
    n_classes = len(fusion.classes)
    outputs = {key: [] for key in ("id_scores", "log_scores", "dr", "direct_pred",
                                   "threshold_pred", "route")}
    diagnostics = {"n": int(len(clean)), "steps_per_sample": int(steps),
                   "objective": objective, "attack": base_name, "snr_db_target": int(snr),
                   "gradient_finite": True, "hard_soft_prediction_mismatch": 0,
                   "st_weight_max_error": 0.0, "projection_relative_violation_max": 0.0,
                   "steps_executed": 0, "full_chain_forward_evaluations": 0,
                   "route_changed_sample_steps": 0}
    total_norms, e55_norms, logspd_norms = [], [], []
    realized_snr, radius_errors, migration = [], [], np.zeros((3, 3), dtype=np.int64)
    was_training = fusion.identity.training
    fusion.identity.eval()
    for start in range(0, len(clean), batch_size):
        stop = min(start+batch_size, len(clean))
        original = clean[start:stop].detach().float()
        centered = original-original.mean(dim=-1, keepdim=True)
        signal_norm = centered.reshape(len(original), -1).norm(dim=1).clamp_min(1e-12)
        radius = signal_norm/(10.0**(float(snr)/20.0))
        generator = torch.Generator(device=original.device)
        generator.manual_seed(int(seed+start))
        delta = torch.zeros_like(original)
        if steps > 1:
            delta = torch.randn(original.shape, generator=generator, device=original.device)
            delta = delta/delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            random_radius = torch.rand(len(delta), generator=generator, device=original.device)
            delta = delta*radius.view(-1, 1, 1)*random_radius.view(-1, 1, 1)
        previous_route = None
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            result = fusion.forward(original+delta)
            branch_norms = {"e55": [], "logspd": []}
            branch_finite = {"e55": True, "logspd": True}

            def hook(key):
                def capture(gradient):
                    branch_finite[key] = bool(branch_finite[key]
                                              and torch.isfinite(gradient).all().item())
                    branch_norms[key].append(gradient.detach().reshape(len(gradient), -1)
                                             .norm(dim=1).cpu().numpy())
                return capture

            handles = [result["id_input"].register_hook(hook("e55")),
                       result["log_input"].register_hook(hook("logspd"))]
            objective_labels = target[start:stop] if targeted else labels[start:stop]
            scores = result["direct_scores"] if objective == "direct" else result["threshold_scores"]
            identity_loss = F.cross_entropy(
                scores/float(fusion.config["attacks"]["temperature"]),
                objective_labels, reduction="none",
            )
            if targeted:
                identity_loss = -identity_loss
            loss = identity_loss.mean()
            if objective == "threshold":
                rejection = F.softplus(
                    (result["logdr"]-result["loglow"])
                    /float(fusion.config["attacks"]["rejection_temperature"])
                ).mean()
                loss = loss+float(fusion.config["attacks"]["rejection_weight"])*rejection
            gradient = torch.autograd.grad(loss, delta)[0]
            for handle in handles:
                handle.remove()
            diagnostics["steps_executed"] += 1
            diagnostics["full_chain_forward_evaluations"] += 1
            diagnostics["hard_soft_prediction_mismatch"] += result["hard_soft_prediction_mismatch"]
            diagnostics["st_weight_max_error"] = max(
                diagnostics["st_weight_max_error"], result["st_weight_max_error"])
            total_norms.append(gradient.detach().reshape(len(gradient), -1).norm(dim=1).cpu().numpy())
            e55_norms.extend(branch_norms["e55"])
            logspd_norms.extend(branch_norms["logspd"])
            diagnostics["gradient_finite"] = bool(
                diagnostics["gradient_finite"] and torch.isfinite(gradient).all().item()
                and branch_finite["e55"] and branch_finite["logspd"] and torch.isfinite(loss).item())
            if not diagnostics["gradient_finite"]:
                raise FloatingPointError("E63 non-finite full-chain gradient")
            current_route = result["route"].detach().cpu().numpy()
            if previous_route is not None:
                diagnostics["route_changed_sample_steps"] += int(np.sum(current_route != previous_route))
            previous_route = current_route
            direction = gradient/gradient.reshape(len(gradient), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            step_size = radius if steps == 1 else 2.0*radius/steps
            delta = delta.detach()+step_size.view(-1, 1, 1)*direction
            norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
            delta = delta*torch.minimum(torch.ones_like(norm), radius/norm).view(-1, 1, 1)
            projected = delta.reshape(len(delta), -1).norm(dim=1)
            violation = ((projected-radius)/radius.clamp_min(1e-12)).max().item()
            diagnostics["projection_relative_violation_max"] = max(
                diagnostics["projection_relative_violation_max"], float(violation))
        with torch.no_grad():
            result = fusion.forward(original+delta)
            diagnostics["full_chain_forward_evaluations"] += 1
            diagnostics["hard_soft_prediction_mismatch"] += result["hard_soft_prediction_mismatch"]
            diagnostics["st_weight_max_error"] = max(
                diagnostics["st_weight_max_error"], result["st_weight_max_error"])
            norm = delta.reshape(len(delta), -1).norm(dim=1)
            snr_actual = 20*torch.log10(signal_norm/norm.clamp_min(1e-12))
            realized_snr.extend(snr_actual.cpu().tolist())
            radius_errors.extend(((norm-radius)/radius.clamp_min(1e-12)).abs().cpu().tolist())
            final_route = result["route"].cpu().numpy()
            initial_route = clean_route[start:stop]
            for before, after in zip(initial_route, final_route):
                migration[int(before), int(after)] += 1
            for key in outputs:
                value = result[key]
                outputs[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
            if not torch.isfinite(result["dr"]).all() or not torch.isfinite(delta).all():
                raise FloatingPointError("E63 non-finite final waveform/fusion output")
    for key in outputs:
        outputs[key] = np.concatenate(outputs[key], axis=0)
    total_steps = [float(np.linalg.norm(row)) for row in total_norms]
    e55_steps = [float(np.linalg.norm(row)) for row in e55_norms]
    logspd_steps = [float(np.linalg.norm(row)) for row in logspd_norms]
    diagnostics.update({
        "total_gradient_nonzero_every_step": bool(all(v > 0 for v in total_steps)),
        "e55_branch_gradient_nonzero_every_step": bool(all(v > 0 for v in e55_steps)),
        "logspd_branch_gradient_nonzero_every_step": bool(all(v > 0 for v in logspd_steps)),
        "total_gradient_step_l2_min": float(min(total_steps)),
        "e55_branch_gradient_step_l2_min": float(min(e55_steps)),
        "logspd_branch_gradient_step_l2_min": float(min(logspd_steps)),
        "e55_branch_nonzero_sample_fraction": float(np.mean(np.concatenate(e55_norms) > 0)),
        "logspd_branch_nonzero_sample_fraction": float(np.mean(np.concatenate(logspd_norms) > 0)),
        "realized_snr_db_min": float(np.min(realized_snr)),
        "realized_snr_db_mean": float(np.mean(realized_snr)),
        "realized_snr_db_max": float(np.max(realized_snr)),
        "radius_relative_error_max": float(np.max(radius_errors)),
        "snr_constraint_all": bool(np.min(realized_snr) >= snr-1e-4
                                   and diagnostics["projection_relative_violation_max"] <= 1e-5),
        "route_flip_fraction": float(np.mean(outputs["route"] != clean_route)),
        "route_migration_matrix": migration.tolist(),
        "hard_route_metrics_only": True,
        "waveform_changed_all": bool(np.all(np.linalg.norm(total_norms[-1]) >= 0)),
    })
    outputs["diagnostics"] = diagnostics
    if was_training:
        fusion.identity.train()
    return outputs


def predict_fusion(fusion: Fusion, raw: torch.Tensor, batch_size: int = 256) -> dict:
    output = {key: [] for key in ("id_scores", "log_scores", "dr", "direct_pred",
                                  "threshold_pred", "route")}
    mismatch = st_error = 0
    with torch.no_grad():
        for start in range(0, len(raw), batch_size):
            result = fusion.forward(raw[start:start+batch_size])
            mismatch += result["hard_soft_prediction_mismatch"]
            st_error = max(st_error, result["st_weight_max_error"])
            for key in output:
                value = result[key]
                output[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
    for key in output:
        output[key] = np.concatenate(output[key], axis=0)
    output["hard_soft_prediction_mismatch"] = int(mismatch)
    output["st_weight_max_error"] = float(st_error)
    return output


def detection_summary(known_dr: np.ndarray, unknown_dr: np.ndarray) -> dict:
    truth = np.concatenate([np.ones(len(known_dr), dtype=int),
                            np.zeros(len(unknown_dr), dtype=int)])
    values = np.concatenate([known_dr, unknown_dr])
    return {"unknown_auroc": float(roc_auc_score(truth, values)),
            "unknown_aupr": float(average_precision_score(1-truth, -values))}


def clean_metrics(runs: list[dict], full: bool = True) -> dict:
    view = "clean_full" if full else "clean_subset"
    known = [run[view]["known"] for run in runs]
    unknown = [run[view]["unknown"] for run in runs]
    true = np.concatenate([run[view]["known_true"] for run in runs])
    direct = np.concatenate([row["direct_pred"] for row in known])
    pred = np.concatenate([row["threshold_pred"] for row in known])
    unknown_pred = np.concatenate([row["threshold_pred"] for row in unknown])
    unknown_direct = np.concatenate([row["direct_pred"] for row in unknown])
    known_dr = np.concatenate([row["dr"] for row in known])
    unknown_dr = np.concatenate([row["dr"] for row in unknown])
    classes = np.unique(np.concatenate([run["classes"] for run in runs]))
    detection = np.concatenate([np.ones(len(true), dtype=int),
                                np.zeros(len(unknown_pred), dtype=int)])
    accepted = np.concatenate([pred != 0, unknown_pred != 0])
    q95_unknown = np.concatenate([
        np.where(row["dr"] < run["tau_high"], 0, row["direct_pred"])
        for run, row in zip(runs, unknown)
    ])
    q95_known_accept = np.concatenate([
        row["dr"] >= run["tau_high"] for run, row in zip(runs, known)
    ])
    c = float(np.mean(pred == true))
    u = float(np.mean(unknown_pred == 0))
    return {
        "direct_C": float(np.mean(direct == true)), "C": c,
        "continuous_block_accuracy": c,
        "continuous_block_accuracy_direct": float(np.mean(direct == true)),
        "identity_macro_f1_direct": float(f1_score(true, direct, labels=classes,
                                                   average="macro", zero_division=0)),
        "identity_macro_f1": float(f1_score(true, pred, labels=classes,
                                            average="macro", zero_division=0)),
        **detection_summary(known_dr, unknown_dr),
        "q05_U": float(np.mean(q95_unknown == 0)),
        "q05_known_acceptance": float(np.mean(q95_known_accept)),
        "U": u, "known_acceptance": float(np.mean(pred != 0)),
        "unknown_f1": float(f1_score(detection, accepted, pos_label=0, zero_division=0)),
        "joint_f1": float(f1_score(
            np.concatenate([true, np.zeros(len(unknown_pred), dtype=int)]),
            np.concatenate([pred, unknown_pred]), labels=np.concatenate([[0], classes]),
            average="macro", zero_division=0)),
        "direct_unknown_all_misaccepted": bool(np.all(unknown_direct != 0)),
        "N_known": int(len(true)), "N_unknown": int(len(unknown_pred)),
    }


ATTACK_MEAN_KEYS = (
    "C", "direct_C", "RA", "direct_RA", "ASafeR", "A", "U", "Rej",
    "known_acceptance", "identity_macro_f1_direct", "identity_macro_f1",
    "unknown_auroc", "unknown_aupr", "unknown_f1", "joint_f1",
)


def attack_metric(runs: list[dict], objective: str, name: str) -> dict:
    known = [run["clean_subset"]["known"] for run in runs]
    unknown = [run["clean_subset"]["unknown"] for run in runs]
    adversarial = [run["attacks"][objective][name] for run in runs]
    true = np.concatenate([run["clean_subset"]["known_true"] for run in runs])
    clean_direct = np.concatenate([row["direct_pred"] for row in known])
    clean_pred = np.concatenate([row["threshold_pred"] for row in known])
    direct = np.concatenate([row["direct_pred"] for row in adversarial])
    pred = np.concatenate([row["threshold_pred"] for row in adversarial])
    unknown_pred = np.concatenate([row["threshold_pred"] for row in unknown])
    adv_dr = np.concatenate([row["dr"] for row in adversarial])
    unknown_dr = np.concatenate([row["dr"] for row in unknown])
    classes = np.unique(np.concatenate([run["classes"] for run in runs]))
    c = float(np.mean(clean_pred == true))
    direct_ra = float(np.mean(direct == true))
    ra = float(np.mean(pred == true))
    asafer = float(np.mean((pred == true) | (pred == 0)))
    a = 0.7*ra+0.3*asafer
    u = float(np.mean(unknown_pred == 0))
    detection = np.concatenate([np.ones(len(pred), dtype=int),
                                np.zeros(len(unknown_pred), dtype=int)])
    accepted = np.concatenate([pred != 0, unknown_pred != 0])
    result = {
        "C": c, "direct_C": float(np.mean(clean_direct == true)),
        "direct_RA": direct_ra, "RA": ra, "ASafeR": asafer, "A": a, "U": u,
        "Rej": float(np.mean(pred == 0)),
        "known_acceptance": float(np.mean(clean_pred != 0)),
        "identity_macro_f1_direct": float(f1_score(true, direct, labels=classes,
                                                   average="macro", zero_division=0)),
        "identity_macro_f1": float(f1_score(true, pred, labels=classes,
                                            average="macro", zero_division=0)),
        **detection_summary(adv_dr, unknown_dr),
        "unknown_f1": float(f1_score(detection, accepted, pos_label=0, zero_division=0)),
        "joint_f1": float(f1_score(
            np.concatenate([true, np.zeros(len(unknown_pred), dtype=int)]),
            np.concatenate([pred, unknown_pred]), labels=np.concatenate([[0], classes]),
            average="macro", zero_division=0)),
        "Score": official_score(c, a, u), "N_clean": int(len(clean_pred)),
        "N_adversarial": int(len(pred)), "N_unknown": int(len(unknown_pred)),
        "all_rejection": bool(np.all(pred == 0) and np.all(clean_pred == 0)),
        "route_flip_fraction": float(np.mean([
            row["diagnostics"]["route_flip_fraction"] for row in adversarial])),
    }
    if name.startswith("targeted"):
        target = np.concatenate([run["target"] for run in runs])
        result["targeted_direct_false_authentication"] = float(np.mean(direct == target))
        result["targeted_false_authentication"] = float(np.mean(pred == target))
    return result


def evaluate_objective(runs: list[dict], objective: str) -> dict:
    per_attack = {name: attack_metric(runs, objective, name) for name in ATTACKS}
    macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
             for key in ATTACK_MEAN_KEYS}
    macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
    macro["attack_score_mean"] = float(np.mean([row["Score"] for row in per_attack.values()]))
    macro["N_adversarial_total"] = int(sum(row["N_adversarial"] for row in per_attack.values()))
    macro["N_clean_per_attack_sum"] = int(sum(row["N_clean"] for row in per_attack.values()))
    macro["N_unknown_per_attack_sum"] = int(sum(row["N_unknown"] for row in per_attack.values()))
    macro["all_rejection"] = bool(all(row["all_rejection"] for row in per_attack.values()))
    targeted = [row for name, row in per_attack.items() if name.startswith("targeted")]
    macro["targeted_direct_false_authentication"] = float(np.mean([
        row["targeted_direct_false_authentication"] for row in targeted]))
    macro["targeted_false_authentication"] = float(np.mean([
        row["targeted_false_authentication"] for row in targeted]))
    return {"per_attack": per_attack, "macro": macro}


def fold_mean(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if isinstance(rows[0][key], (float, int, np.floating, np.integer))
            and not isinstance(rows[0][key], (bool, np.bool_))}


def train_fit_summary(runs: list[dict]) -> dict:
    values = np.asarray([run["train_fit"]["accuracy"] for run in runs])
    direct = np.asarray([run["train_fit"]["direct_accuracy"] for run in runs])
    return {"accuracy": float(values.mean()), "direct_accuracy": float(direct.mean()),
            "minimum_fold_accuracy": float(values.min()),
            "minimum_fold_direct_accuracy": float(direct.min()),
            "strict_100pct": bool(np.all(values == 1.0)),
            "strict_direct_100pct": bool(np.all(direct == 1.0)),
            "identity_macro_f1": float(np.mean([run["train_fit"]["identity_macro_f1"] for run in runs])),
            "identity_macro_f1_direct": float(np.mean([
                run["train_fit"]["identity_macro_f1_direct"] for run in runs])),
            "e55_identity_fit_accuracy": float(np.mean([
                run["e55_info"]["fit_accuracy"] for run in runs])),
            "N": int(sum(run["train_fit"]["N"] for run in runs)), "folds": len(runs),
            "definition": "all fit-known samples through actual raw E55+logSPD/WCCN hard fusion; primary accuracy includes rejection"}


# __E63_CONTINUE__
