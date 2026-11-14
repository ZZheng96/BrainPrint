from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
import run_e54 as E54

CFG_PATH = OUT / "config.json"
RESULTS = OUT / "results.json"
REPORT = OUT / "report.md"
CHECKPOINT = OUT / "model_checkpoint.pt"
PRED = OUT / "predictions.csv"


def save_json(value):
    RESULTS.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), encoding="utf-8")


def frac_blocks(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def scores(model, raw, indices):
    model.eval()
    out = []
    with torch.no_grad():
        for st in range(0, len(indices), 256):
            ix = torch.as_tensor(indices[st:st + 256], device=raw.device)
            out.append(model.scores(raw[ix]).detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.n_classes), dtype=np.float32)


def select_threshold(known_scores, unknown_scores, known_labels, classes, cfg):
    known_max = known_scores.max(1)
    unknown_max = unknown_scores.max(1)
    best = None
    for q in cfg["threshold"]["grid_quantiles"]:
        threshold = float(np.quantile(known_max, q))
        known_pred = np.where(known_max >= threshold, classes[known_scores.argmax(1)], 0)
        unknown_accept = unknown_max >= threshold
        acceptance = float(np.mean(known_pred != 0))
        if acceptance + 1e-12 < cfg["threshold"]["known_acceptance_target"]:
            continue
        rejection = float(np.mean(~unknown_accept)) if len(unknown_accept) else 0.0
        identity = float(np.mean(known_pred == known_labels))
        value = rejection + 0.1 * identity + 0.01 * acceptance
        candidate = (value, threshold, acceptance, rejection, identity)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        threshold = float(np.quantile(known_max, 0.05))
        best = (0.0, threshold, float(np.mean(known_max >= threshold)), float(np.mean(unknown_max < threshold)), 0.0)
    _, threshold, acceptance, rejection, identity = best
    return {"score": "max_cosine", "threshold": threshold, "known_acceptance": acceptance, "unknown_rejection": rejection, "identity_accuracy": identity, "selection_source": "Train-only calibration model; no Test access"}


def apply_gate(score, params, classes):
    max_score = score.max(1)
    pred = np.where(max_score >= params["threshold"], classes[score.argmax(1)], 0)
    return pred, max_score


def train_variant(raw, meta, subjects, frac, cfg, device, epochs, classes, fit_mask, seed_offset):
    e54cfg = E54.load_config()
    e54cfg["seed_base"] = int(cfg["seed"] + seed_offset)
    e54cfg["training"]["max_epochs"] = int(epochs)
    e54cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])
    e54cfg["training"]["cache_refresh_epochs"] = int(cfg["training"]["cache_refresh_epochs"])
    e54cfg["training"]["cache_pgd_steps"] = int(cfg["training"]["pgd_steps"])
    e54cfg["training"]["clean_ce_weight"] = float(cfg["training"]["clean_weight"])
    e54cfg["training"]["adversarial_ce_weight"] = float(cfg["training"]["adversarial_weight"])
    fit_np = np.flatnonzero(fit_mask)
    if len(fit_np) == 0:
        raise ValueError("empty fit block")
    labels = np.unique(subjects[fit_np])
    if not np.array_equal(np.sort(labels), np.sort(classes)):
        raise ValueError("fit block does not contain all requested classes")
    model, info = E54.train_model("control", raw, subjects, fit_np, np.empty(0, dtype=int), None, None, classes, meta, e54cfg, 95, 0, fixed_epochs=int(epochs))
    return model, info


def load_test_predictions(model, raw_device, params, classes, cfg):
    from scipy.io import loadmat
    order = pd.read_csv(ROOT / cfg["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / cfg["data"]["test_metadata"], usecols=["sample", "path", "session"])
    if len(order) != cfg["data"]["expected_test_rows"] or len(mapping) != len(order):
        raise ValueError("Test mapping mismatch")
    root = ROOT / cfg["data"]["train_root"]
    labels = []
    batch = 128
    for st in range(0, len(mapping), batch):
        paths = [root / p for p in mapping.path.iloc[st:st + batch]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"], dtype=np.float32).T for p in paths])
        if arr.shape[1:] != (18, 1200) or not np.isfinite(arr).all():
            raise ValueError("invalid Test MAT")
        x = torch.as_tensor(arr, dtype=torch.float32, device=raw_device)
        with torch.no_grad():
            s = model.scores(x).cpu().numpy()
        labels.extend(apply_gate(s, params, classes)[0].tolist())
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(PRED, index=False)
    return order, mapping


def run(smoke=False):
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("E95 requires CUDA for GPU training/inference")
    device = torch.device("cuda")
    e54cfg = E54.load_config()
    meta = E54.train_metadata(e54cfg)
    raw = E54.load_all_raw(meta, e54cfg, device)
    subjects = meta.subject.to_numpy(int)
    frac = frac_blocks(meta)
    classes = np.arange(1, 21, dtype=int)
    rng = np.random.default_rng(cfg["seed"])
    pseudo_unknown = np.sort(rng.choice(classes, size=4, replace=False))
    pseudo_known = np.setdiff1d(classes, pseudo_unknown)
    cal_known = np.flatnonzero(np.isin(subjects, pseudo_known) & (frac > .60) & (frac <= .80))
    cal_unknown = np.flatnonzero(np.isin(subjects, pseudo_unknown) & (frac > .60) & (frac <= .80))
    cal_fit = np.flatnonzero(np.isin(subjects, pseudo_known) & (frac <= .60))
    epochs = cfg["training"]["smoke_epochs"] if smoke else cfg["training"]["epochs"]
    # Leave-identity calibration model: pseudo-unknown identities are absent from fitting.
    cal_model, cal_info = train_variant(raw, meta, subjects, frac, cfg, device, epochs, pseudo_known, np.isin(subjects, pseudo_known) & (frac <= .60), 100)
    cal_known_scores = scores(cal_model, raw, cal_known)
    cal_unknown_scores = scores(cal_model, raw, cal_unknown)
    params = select_threshold(cal_known_scores, cal_unknown_scores, subjects[cal_known], pseudo_known, cfg)
    cal_pred, cal_conf = apply_gate(cal_known_scores, params, pseudo_known)
    cal_auc = float(roc_auc_score(np.r_[np.ones(len(cal_known_scores)), np.zeros(len(cal_unknown_scores))], np.r_[cal_conf, cal_unknown_scores.max(1)]))
    cal_f1 = float(f1_score(subjects[cal_known], cal_pred, labels=pseudo_known, average="macro", zero_division=0))
    # Final model: all 20 identities, every Train trial through the 0.8 time block.
    final_model, final_info = train_variant(raw, meta, subjects, frac, cfg, device, epochs, classes, frac <= .80, 200)
    torch.save({"state_dict": final_model.state_dict(), "classes": classes, "threshold": params, "config": cfg, "calibration_subjects": {"pseudo_unknown": pseudo_unknown.tolist(), "pseudo_known": pseudo_known.tolist()}}, CHECKPOINT)
    result = {"status": "complete_train_only" if smoke else "frozen_before_test", "experiment": cfg["experiment"], "calibration": {"pseudo_unknown": pseudo_unknown.tolist(), "pseudo_known": pseudo_known.tolist(), "N_known": int(len(cal_known)), "N_unknown": int(len(cal_unknown)), "AUC": cal_auc, "F1": cal_f1, "threshold": params, "model_info": cal_info}, "final_model_info": final_info, "train_only_metrics": {"AUC": cal_auc, "F1": cal_f1}, "test_metrics": {k: "N/A" for k in ["AUC", "F1", "C", "RA", "ASafeR", "U", "Official Score"]}, "protocol_checks": {"train_session1_only": True, "train_only_oof_time_calibration": True, "leave_identity_pseudo_unknown": True, "final_20_class_fit_fraction": 0.8, "gpu": torch.cuda.get_device_name(0), "test_accessed": False}}
    if not smoke:
        order, mapping = load_test_predictions(final_model, raw.device, params, classes, cfg)
        written = pd.read_csv(PRED)
        rej = written.Label.to_numpy() == 0
        session = mapping.session.astype(str).str.casefold().map({"session1": "S1", "s1": "S1", "1": "S1", "session2": "S2", "s2": "S2", "2": "S2"}).to_numpy()
        audit = {"N": int(len(written)), "predictions_complete_before_session_audit": True, "session_used_by_inference": False}
        for name in ("S1", "S2"):
            m = session == name
            audit[f"{name}_N"] = int(m.sum())
            audit[f"{name}_rejection_rate"] = float(rej[m].mean()) if m.any() else "N/A"
            audit[f"{name}_acceptance_rate"] = float((~rej[m]).mean()) if m.any() else "N/A"
        result["audit"] = audit
        result["status"] = "complete_one_shot_frozen_audit"
        result["protocol_checks"].update({"test_accessed": True, "test_read_once_after_freeze": True})
    save_json(result)
    REPORT.write_text(f"# E95 E54 final20 candidate\n\nTrain/session1 only for training and calibration; the final 20-class model is frozen before the sole Test read.\n\n## Train-only calibration\n\nAUC={cal_auc:.6f}，F1={cal_f1:.6f}。阈值={params['threshold']:.6f}，known acceptance={params['known_acceptance']:.6f}，pseudo-unknown rejection={params['unknown_rejection']:.6f}。\n\n## Test\n\n真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n" + (f"S1 拒识率={result['audit']['S1_rejection_rate']:.6f}；S2 接受率={result['audit']['S2_acceptance_rate']:.6f}。predictions.csv 已完整落盘后才读取 session 做审计。\n" if not smoke else "Smoke 模式未读取 Test。\n"), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true"); args = ap.parse_args()
    if args.smoke == args.full: ap.error("choose --smoke or --full")
    print(json.dumps(run(args.smoke), ensure_ascii=False, indent=2))
