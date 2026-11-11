"""Strict transformed-feature PGD evaluation for logcov and corrSPD."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from src.session_blind_candidate.run import load_inputs
from src.open_set_improvement.core import TaskSpecificModel, attack_batch, confidence_values, resolve_cache
from src.open_set_calibration.core import predict_calibration_scores
from src.ea_riemannian_openset.core import log_features_to_spd, spd_to_log_features

LAMBDA = 1e-3
EPSILON = 0.05
STEPS = 10
BATCH_SIZE = 128
METHODS = (("logcov", False), ("corrspd", True))


def corr_transform(features: np.ndarray, device: torch.device) -> np.ndarray:
    cov = log_features_to_spd(features, device=device, batch_size=256)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12)
    scale = diag.rsqrt()
    corr = scale.unsqueeze(-1) * cov * scale.unsqueeze(-2)
    eye = torch.eye(corr.shape[-1], dtype=corr.dtype, device=device)
    corr = (1.0 - LAMBDA) * corr + LAMBDA * eye
    result = spd_to_log_features(corr, batch_size=256).float().cpu().numpy()
    del cov, corr
    return result


def transform(x, w, device, corr):
    if not corr:
        return np.asarray(x), np.asarray(w)
    tx = corr_transform(x, device)
    flat = np.asarray(w).reshape(-1, w.shape[-1])
    tw = corr_transform(flat, device).reshape(len(w), w.shape[1], -1)
    return tx, tw


def pred(scores, classes, threshold):
    confidence = confidence_values(scores)["max_logit"]
    labels = classes[scores.argmax(axis=1)].astype(int)
    labels[confidence < threshold] = 0
    return labels, confidence


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda")
    cache = resolve_cache()
    x, w, meta, _, _, _ = load_inputs(cache)
    y = meta.subject.to_numpy()
    tasks = meta.task.to_numpy()
    split = pd.read_csv(cache / "train_blocked_oof_predictions_cov_alpha100.csv", usecols=["subject", "task", "fold"])
    if not split[["subject", "task"]].equals(meta[["subject", "task"]]):
        raise ValueError("split misaligned")
    folds = split.fold.to_numpy()
    methods = {}
    prediction_rows = []
    for method, corr in METHODS:
        fx, fw = transform(x, w, device, corr)
        fold_results = []
        all_clean, all_adv, all_true, all_unknown, all_unknown_true = [], [], [], [], []
        for fold in range(5):
            val = folds == fold
            calibration = folds == ((fold + 1) % 5)
            pseudo_id = np.sort(np.unique(y[calibration]))[0]
            pseudo = calibration & (y == pseudo_id)
            fit = ~(val | calibration) | (calibration & ~pseudo)
            fit &= y != pseudo_id
            known_mask = val & (y != pseudo_id)
            model = TaskSpecificModel(device, alpha=100.0).fit(fx, fw, y, tasks, fit)
            known_idx = np.flatnonzero(known_mask)
            pseudo_idx = np.flatnonzero(pseudo)
            with torch.no_grad():
                known_scores = model.predict_scores(fx[known_idx], fw[known_idx], tasks[known_idx])["fused"]
                pseudo_scores = model.predict_scores(fx[pseudo_idx], fw[pseudo_idx], tasks[pseudo_idx])["fused"]
            threshold = float(np.quantile(confidence_values(pseudo_scores)["max_logit"], 0.95))
            clean_pred, clean_conf = pred(known_scores, model.classes, threshold)
            adv_scores = np.empty_like(known_scores)
            observed = []
            for task in sorted(np.unique(tasks[known_idx])):
                positions = np.flatnonzero(tasks[known_idx] == task)
                for start in range(0, len(positions), BATCH_SIZE):
                    selected = positions[start:start + BATCH_SIZE]
                    indices = known_idx[selected]
                    scores, norm = attack_batch(model, fx[indices], fw[indices], task, y[indices], EPSILON, "pgd", STEPS)
                    adv_scores[selected] = scores
                    observed.append(float(norm.max()))
            adv_pred, adv_conf = pred(adv_scores, model.classes, threshold)
            pseudo_pred, _ = pred(pseudo_scores, model.classes, threshold)
            true_known = y[known_idx].astype(int)
            c = float(np.mean(clean_pred == true_known))
            ra = float(np.mean(adv_pred == true_known))
            asafer = float(np.mean((adv_pred == true_known) | (adv_pred == 0)))
            u = float(np.mean(pseudo_pred == 0))
            a = 0.7 * ra + 0.3 * asafer
            score = float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0
            f1_clean = float(f1_score(true_known, clean_pred, labels=np.arange(1, 21), average="macro", zero_division=0))
            f1_adv = float(f1_score(true_known, adv_pred, labels=np.arange(1, 21), average="macro", zero_division=0))
            row = {"fold": fold, "pseudo_unknown_subject": int(pseudo_id), "n_known": int(len(known_idx)), "n_pseudo_unknown": int(len(pseudo_idx)), "threshold": threshold, "C": c, "F1_clean": f1_clean, "F1_adv": f1_adv, "RA": ra, "ASafeR": asafer, "A": a, "U": u, "S_proxy": score, "observed_max_linf_standardized": max(observed)}
            fold_results.append(row)
            all_clean.append(clean_pred); all_adv.append(adv_pred); all_true.append(true_known)
            all_unknown.append(pseudo_pred); all_unknown_true.append(np.zeros(len(pseudo_pred), dtype=int))
            prediction_rows.extend([{"method": method, "fold": fold, "kind": "known", "true": int(t), "clean_pred": int(cp), "adv_pred": int(ap)} for t, cp, ap in zip(true_known, clean_pred, adv_pred)])
            prediction_rows.extend([{"method": method, "fold": fold, "kind": "pseudo_unknown", "true": 0, "clean_pred": int(pp), "adv_pred": None} for pp in pseudo_pred])
            del model
            torch.cuda.empty_cache()
        clean = np.concatenate(all_clean); adv = np.concatenate(all_adv); true = np.concatenate(all_true); unknown = np.concatenate(all_unknown)
        c = float(np.mean(clean == true)); ra = float(np.mean(adv == true)); asafer = float(np.mean((adv == true) | (adv == 0))); u = float(np.mean(unknown == 0)); a = 0.7 * ra + 0.3 * asafer
        score = float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0
        methods[method] = {"n_known": int(len(true)), "n_pseudo_unknown": int(len(unknown)), "C": c, "F1_clean": float(f1_score(true, clean, labels=np.arange(1, 21), average="macro", zero_division=0)), "F1_adv": float(f1_score(true, adv, labels=np.arange(1, 21), average="macro", zero_division=0)), "RA": ra, "ASafeR": asafer, "A": a, "U": u, "S_proxy": score, "folds": fold_results}
        del fx, fw
        torch.cuda.empty_cache()
    (OUT / "prediction_summary.csv").write_text(pd.DataFrame(prediction_rows).to_csv(index=False), encoding="utf-8")
    result = {"protocol": {"attack": "feature-space PGD against fused scores", "epsilon": EPSILON, "steps": STEPS, "step_size": 2 * EPSILON / STEPS, "epsilon_unit": "per-method training feature standard deviation", "threshold": "frozen per outer fold at pseudo-unknown max_logit 95th percentile", "known_eval": "outer validation samples excluding deliberately held-out pseudo identity", "U": "clean pseudo-unknown samples from same outer fold; not real unknown/test labels", "raw_eeg_attack": False, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_train": int(len(x)), "n_known_total": int(sum(r["n_known"] for r in methods["logcov"]["folds"])), "command": "py -3.11 research/E11_session_blind_candidate/nested_pseudounknown_corrspd_v1/attack_eval/run_attack_eval.py"}, "methods": methods, "prediction_summary": "prediction_summary.csv"}
    (OUT / "attack_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
