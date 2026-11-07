"""Nested blocked session-blind evaluation of log-cov and correlation-SPD candidates."""
from __future__ import annotations
import hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.session_blind_candidate.run import load_inputs
from src.open_set_improvement.core import TaskSpecificModel, resolve_cache, confidence_values
from src.open_set_calibration.core import build_submission, predict_calibration_scores
from src.ea_riemannian_openset.core import log_features_to_spd, spd_to_log_features

OUT = Path(__file__).resolve().parent
LAMBDA = 1e-3
METHODS = ("logcov", "corrspd")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def corr_transform(features, device):
    """Convert each band SPD to shrinkage correlation SPD, then log-vectorize."""
    cov = log_features_to_spd(features, device=device, batch_size=256)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12)
    scale = diag.rsqrt()
    corr = scale.unsqueeze(-1) * cov * scale.unsqueeze(-2)
    eye = torch.eye(corr.shape[-1], dtype=corr.dtype, device=device)
    corr = (1.0 - LAMBDA) * corr + LAMBDA * eye
    result = spd_to_log_features(corr, batch_size=256).float().cpu().numpy()
    del cov, corr
    return result


def transform_all(x, window, device, corr):
    if not corr: return np.asarray(x), np.asarray(window)
    tx = corr_transform(x, device)
    flat = np.asarray(window).reshape(-1, window.shape[-1])
    tw = corr_transform(flat, device).reshape(len(window), window.shape[1], -1)
    return tx, tw


def score_model(x, w, subjects, tasks, fit_mask, tx, tw, tt, device):
    model = TaskSpecificModel(device, alpha=100.0).fit(x, w, subjects, tasks, fit_mask)
    scores, _ = predict_calibration_scores(model, tx, tw, tt, "none")
    return scores["full_none"], model.classes.copy()


def pseudo_threshold(scores, pseudo_scores):
    # Set the cutoff at the 95th percentile of leave-identity-out pseudo-unknown confidence.
    return float(np.quantile(confidence_values(pseudo_scores)["max_logit"], 0.95))


def classify(scores, classes, threshold):
    conf = confidence_values(scores)["max_logit"]
    pred = classes[scores.argmax(1)].astype(int)
    pred[conf < threshold] = 0
    return pred, conf


def metrics_known(y, pred, conf, threshold):
    return {"n": int(len(y)), "C": float(np.mean(pred == y)),
            "F1": float(f1_score(y, pred, labels=np.arange(1,21), average="macro", zero_division=0)),
            "known_acceptance": float(np.mean(conf >= threshold)), "reject_rate": float(np.mean(pred == 0))}


def main():
    start = time.time()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU fallback")
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda")
    cache = resolve_cache()
    x, w, meta, test_x, test_w, test_meta = load_inputs(cache)
    y = meta.subject.to_numpy(); tasks = meta.task.to_numpy(); tt = test_meta.task.to_numpy()
    split_path = cache / "train_blocked_oof_predictions_cov_alpha100.csv"
    split = pd.read_csv(split_path, usecols=["subject", "task", "fold"])
    if not split[["subject","task"]].equals(meta[["subject","task"]]): raise ValueError("split misaligned")
    folds = split.fold.to_numpy(); classes = np.unique(y)
    oof_scores = {m: np.full((len(x), len(classes)), np.nan, np.float32) for m in METHODS}
    oof_pred = {m: np.full(len(x), -1, int) for m in METHODS}
    fold_rows = {m: [] for m in METHODS}; pseudo_rows = {m: [] for m in METHODS}
    fold_info = {}
    for fold in range(5):
        val = folds == fold
        calibration = folds == ((fold + 1) % 5)
        # One identity from the calibration block is held out from fitting as pseudo-unknown.
        pseudo_id = np.sort(np.unique(y[calibration]))[0]
        pseudo = calibration & (y == pseudo_id)
        fit = ~(val | calibration) | (calibration & ~pseudo)
        # Explicitly remove the pseudo identity even if it occurs in other blocks.
        fit &= y != pseudo_id
        fold_info[str(fold)] = {"pseudo_unknown_subject": int(pseudo_id), "n_pseudo": int(pseudo.sum())}
        for method, corr in (("logcov", False), ("corrspd", True)):
            fx, fw = transform_all(x, w, device, corr)
            vx, vw = fx[val], fw[val]
            px, pw = fx[pseudo], fw[pseudo]
            scores, model_classes = score_model(fx, fw, y, tasks, fit, vx, vw, tasks[val], device)
            pseudo_scores, pseudo_classes = score_model(fx, fw, y, tasks, fit, px, pw, tasks[pseudo], device)
            if not np.array_equal(model_classes, pseudo_classes): raise ValueError("class order mismatch")
            cut = pseudo_threshold(scores, pseudo_scores)
            pred, conf = classify(scores, model_classes, cut)
            # Exclude deliberately held-out identity from known C/F1; retain all scoring for audit.
            known_mask = y[val] != pseudo_id
            fold_rows[method].append({**metrics_known(y[val][known_mask], pred[known_mask], conf[known_mask], cut), "fold": fold, "threshold": cut})
            pseudo_rows[method].append({"n": int(len(pseudo_scores)), "U": float(np.mean(confidence_values(pseudo_scores)["max_logit"] < cut))})
            oof_scores[method][val] = -np.inf
            oof_scores[method][val][:, np.searchsorted(classes, model_classes)] = scores
            oof_pred[method][val] = pred
            del fx, fw
        print(f"outer fold {fold+1}/5 complete; pseudo subject={pseudo_id}", flush=True)
    # Deployment: full fit, threshold from all five pseudo-unknown outer calibration records.
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv", usecols=["Sample"])
    deployment = {}; metrics = {"methods": {}}
    for method, corr in (("logcov", False), ("corrspd", True)):
        fx, fw = transform_all(x, w, device, corr)
        tx, tw = transform_all(test_x, test_w, device, corr)
        model = TaskSpecificModel(device, alpha=100.0).fit(fx, fw, y, tasks, np.ones(len(y), bool))
        test_scores, _ = predict_calibration_scores(model, tx, tw, tt, "none")
        # Fixed rule: deployment cutoff is median of outer pseudo-unknown 95th-percentile cutoffs.
        cut = float(np.median([r["threshold"] for r in fold_rows[method]]))
        pred, conf = classify(test_scores["full_none"], model.classes, cut)
        sub = build_submission(template, test_meta, pred)
        if len(sub) != 22251 or not sub.Sample.is_unique or not sub.Sample.equals(template.Sample) or not sub.Label.isin(range(21)).all(): raise ValueError("invalid submission")
        sub.to_csv(OUT / f"candidate_{method}.csv", index=False)
        if method == "corrspd": sub.to_csv(OUT / "candidate_corrspd.csv", index=False)
        known = np.concatenate([r["n"] * [1] for r in []]) if False else None
        fr = fold_rows[method]; pr = pseudo_rows[method]
        known_n = sum(r["n"] for r in fr); pseudo_n = sum(r["n"] for r in pr)
        # Pooled OOF known rows use only non-pseudo identities, reconstructed from fold predictions.
        mask = np.ones(len(y), bool)
        for fold in range(5): mask &= ~((folds == fold) & (y == fold_info[str(fold)]["pseudo_unknown_subject"]))
        op = oof_pred[method][mask]; oy = y[mask]
        oo_conf = confidence_values(oof_scores[method][mask])["max_logit"]
        auc = float(roc_auc_score(np.r_[np.ones(known_n), np.zeros(pseudo_n)], np.r_[oo_conf, np.zeros(pseudo_n)])) if False else None
        metrics["methods"][method] = {"outer_known": fr, "worst_fold_C": float(min(r["C"] for r in fr)), "worst_fold_F1": float(min(r["F1"] for r in fr)), "known_C": float(np.mean(op == oy)), "known_F1": float(f1_score(oy, op, labels=np.arange(1,21), average="macro", zero_division=0)), "pseudo_unknown_U": float(np.average([r["U"] for r in pr], weights=[r["n"] for r in pr])), "AUC": None, "F1": float(f1_score(oy, op, labels=np.arange(1,21), average="macro", zero_division=0)), "known_acceptance": float(np.mean(oo_conf >= np.median([r["threshold"] for r in fr]))), "test_reject_rate": float(np.mean(pred == 0)), "n_test": int(len(sub)), "deployment_threshold": cut, "RA": None, "ASafeR": None, "proxy": None}
        deployment[method] = {"file": f"candidate_{method}.csv", "threshold": cut, "n_reject": int(np.sum(pred == 0)), "reject_rate": float(np.mean(pred == 0))}
        del fx, fw, tx, tw
    manifest = {"seed":42,"device":torch.cuda.get_device_name(0),"torch":torch.__version__,"cache":str(cache),"n_train":len(x),"n_test":len(test_x),"metadata_columns_read":{"train":["subject","task"],"test":["sample","task"],"split":["subject","task","fold"]},"forbidden_columns_read":["session","trial","path","fs"],"test_labels_used":False,"lambda":LAMBDA,"methods":METHODS,"split_sha256":sha256(split_path),"outer_folds":fold_info,"deployment":deployment,"elapsed_seconds":time.time()-start}
    (OUT / "metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame({"subject":y,"task":tasks,"fold":folds,"logcov_pred":oof_pred["logcov"],"corrspd_pred":oof_pred["corrspd"]}).to_csv(OUT/"nested_blocked_oof_predictions.csv",index=False)
    report = ["# nested pseudounknown corrSPD v1", "", "固定 lambda=1e-3；corrSPD 对每个频带恢复 SPD、归一化为相关矩阵、shrinkage 后重新 log-vectorize。严格 blocked 外层评估，并在每折留出一个身份作为伪未知校准；未读取 session/trial/path/fs/Test 标签。", "", "|方法|外层已知 C|外层 F1|伪未知 U|AUC|已知接受率|最差折 C|最差折 F1|Test 拒识率|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        z=metrics["methods"][m]; report.append(f"|{m}|{z['known_C']:.6f}|{z['F1']:.6f}|{z['pseudo_unknown_U']:.6f}|null|{z['known_acceptance']:.6f}|{z['worst_fold_C']:.6f}|{z['worst_fold_F1']:.6f}|{z['test_reject_rate']:.6f}|")
    report += ["", "AUC、RA、ASafeR、代理分均为 null：本实验严格使用留身份伪未知校准，不构造攻击或真实未知标签，不能补值。", "", "复现：`py -3.11 research/session_blind_candidate/nested_pseudounknown_corrspd_v1/run.py`"]
    (OUT / "report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report),flush=True)

if __name__ == "__main__": main()
