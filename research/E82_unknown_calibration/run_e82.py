"""E82: strict nested pseudo-unknown calibration with frozen E74 and saved E79 pair scores."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

E82_DIR = Path(__file__).resolve().parent
ROOT = E82_DIR.parents[1]
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
E79_DIR = ROOT / "research" / "E79_pair_open_set"
E73_DIR = ROOT / "research" / "E73_external_multisession"
for p in (E82_DIR, E74_DIR, E79_DIR, E73_DIR, ROOT / "research" / "E72_pair_protocol"):
    sys.path.insert(0, str(p))
from external_data import load_external_data, subject_folds
from run_e74 import embed_indices, sha256, utc_now, write_json
from run_e79 import load_frozen_encoder, fit_pair_head, pair_scores
from pair_head import subject_templates
from pretrain import pair_metrics, operating_metrics, numeric_mean

DIRECTIONS = (("T", "E"), ("E", "T"))

def robust_stats(model, data, roles, base, cfg, source):
    idx = data.indices(roles["enrolled"], session=source, runs=cfg["protocol"]["fit_runs"])
    emb = embed_indices(model, data.raw, idx, base).cpu().numpy()
    means, invcov, medians, mads = [], [], [], []
    for subject in roles["enrolled"]:
        x = emb[data.subject[idx] == subject]; mean = x.mean(0); centered = x - mean
        cov = centered.T @ centered / max(len(x) - 1, 1)
        diag = np.trace(cov) / cov.shape[0]
        shrink = cfg["mahalanobis"]["covariance_shrinkage"]
        cov = (1 - shrink) * cov + shrink * diag * np.eye(cov.shape[0])
        cov += cfg["mahalanobis"]["covariance_floor"] * np.eye(cov.shape[0])
        inv = np.linalg.pinv(cov); dist = np.sqrt(np.maximum(0, np.einsum("nd,dd,nd->n", centered, inv, centered)))
        med = np.median(dist); mad = max(np.median(np.abs(dist - med)) * 1.4826, cfg["mahalanobis"]["mad_floor"])
        means.append(mean); invcov.append(inv); medians.append(med); mads.append(mad)
    return {"classes": roles["enrolled"].copy(), "means": np.asarray(means), "invcov": np.asarray(invcov), "medians": np.asarray(medians), "mads": np.asarray(mads)}

def maha_confidence(embedding, stats, cfg):
    x = embedding.cpu().numpy()
    d = np.stack([np.sqrt(np.maximum(0, np.einsum("nd,dd,nd->n", x - m, inv, x - m))) for m, inv in zip(stats["means"], stats["invcov"])], 1)
    z = (d - stats["medians"][None, :]) / stats["mads"][None, :]
    return -z / float(cfg["fusion"]["mahalanobis_temperature"])

def fused_scores(pair_logits, maha_z, cfg):
    p = 1 / (1 + np.exp(-pair_logits / float(cfg["fusion"]["pair_temperature"])))
    m = 1 / (1 + np.exp(-maha_z))
    return np.minimum(p, m)

def unknown_thresholds(calibration_unknown_scores, target_fpirs):
    confidence = calibration_unknown_scores.max(axis=1)
    return {str(fpir): float(np.quantile(confidence, 1.0 - fpir, method="linear")) for fpir in target_fpirs}, confidence

def tail_thresholds(known_scores, known_true, classes, fraction):
    lookup = {int(c): i for i, c in enumerate(classes)}
    values = []
    thresholds = {}
    for c in classes:
        x = known_scores[known_true == c, lookup[int(c)]]
        if not len(x): raise ValueError(f"No calibration samples for identity {c}")
        thresholds[int(c)] = float(np.quantile(x, fraction, method="linear"))
        values.extend(x.tolist())
    return thresholds, values

def evaluate(known_scores, unknown_scores, known_true, classes, threshold, target_fpir):
    known_arg = known_scores.argmax(1); unknown_arg = unknown_scores.argmax(1)
    known_conf = known_scores[np.arange(len(known_scores)), known_arg]
    unknown_conf = unknown_scores[np.arange(len(unknown_scores)), unknown_arg]
    known_pred_direct = classes[known_arg]; unknown_pred_direct = classes[unknown_arg]
    known_accept = known_conf >= threshold
    unknown_accept = unknown_conf >= threshold
    known_pred = np.where(known_accept, known_pred_direct, 0)
    unknown_pred = np.where(unknown_accept, unknown_pred_direct, 0)
    detection_scores = np.r_[known_conf, unknown_conf]
    detection_true = np.r_[np.ones(len(known_conf)), np.zeros(len(unknown_conf))]
    accepted = np.r_[known_accept, unknown_accept]
    identity_f1 = f1_score(known_true, known_pred, labels=classes, average="macro", zero_division=0)
    unknown_f1 = f1_score(detection_true, accepted, pos_label=1, zero_division=0)
    result = {"N_known": int(len(known_true)), "N_unknown": int(len(unknown_scores)),
        "target_fpir": float(target_fpir), "identity_macro_F1": float(identity_f1), "unknown_F1": float(unknown_f1),
        "known_acceptance": float(np.mean(known_accept)), "unknown_false_acceptance": float(np.mean(unknown_accept)),
        "AUROC": float(roc_auc_score(detection_true, detection_scores))}
    result.update(operating_metrics(known_conf, unknown_conf, known_pred_direct == known_true))
    return result

def evaluate_direction(model, head, data, roles, base, cfg, source, target):
    protocol = cfg["protocol"]
    ti = data.indices(roles["enrolled"], session=source, runs=[protocol["template_run"]])
    ki = data.indices(roles["enrolled"], session=target, runs=[protocol["inner_known_run"]])
    ci = data.indices(roles["calibration_unknown"], session=target, runs=[protocol["calibration_unknown_run"]])
    oi = data.indices(roles["enrolled"], session=target, runs=[protocol["outer_evaluation_run"]])
    ui = data.indices(roles["outer_unknown"], session=target, runs=[protocol["outer_evaluation_run"]])
    template = subject_templates(embed_indices(model, data.raw, ti, base), data.subject[ti], roles["enrolled"])
    def score(idx): return 1 / (1 + np.exp(-pair_scores(head, embed_indices(model, data.raw, idx, base), template)))
    known_cal, calibration_unknown, known_outer, unknown_outer = map(score, (ki, ci, oi, ui))
    thresholds, calibration_conf = unknown_thresholds(calibration_unknown, protocol["target_fpirs"])
    inner = {str(f): evaluate(known_cal, calibration_unknown, data.subject[ki], roles["enrolled"], thresholds[str(f)], f) for f in protocol["target_fpirs"]}
    outer = {str(f): evaluate(known_outer, unknown_outer, data.subject[oi], roles["enrolled"], thresholds[str(f)], f) for f in protocol["target_fpirs"]}
    for f in protocol["target_fpirs"]:
        inner[str(f)]["calibration_unknown_false_acceptance"] = float(np.mean(calibration_conf >= thresholds[str(f)]))
    return {"inner": inner, "outer": outer}, {"thresholds": thresholds, "calibration_unknown_N": len(calibration_conf)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); path = E82_DIR / args.config; cfg = json.loads(path.read_text(encoding="utf-8"))
    base_path = E74_DIR / "config.json"; base = json.loads(base_path.read_text(encoding="utf-8-sig")); pair_path = E79_DIR / "config.json"; pair_cfg = json.loads(pair_path.read_text(encoding="utf-8"))
    out = E82_DIR / cfg["output_subdir"]; out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("E82 requires CUDA")
    device = torch.device("cuda"); data = load_external_data(base, device); folds = subject_folds(base)[:1] if args.smoke else subject_folds(base)
    records = []
    for fold, roles in enumerate(folds, 1):
        model, enc = load_frozen_encoder(base, base_path, roles, fold, device)
        head, pair_info = fit_pair_head(model, data, roles, base, pair_cfg, pair_path, fold, out / "models" / f"pair_outer{fold}.pt", False)
        dirs = {}
        for source, target in DIRECTIONS:
            dirs[f"{source}_to_{target}"], calibration = evaluate_direction(model, head, data, roles, base, cfg, source, target)
        torch.save({"experiment": cfg["experiment"], "outer_fold": fold, "roles": {k: v.tolist() for k,v in roles.items()}, "calibration": calibration, "source_hashes": {"config": sha256(path), "run_e82": sha256(Path(__file__).resolve()), "e74_checkpoint": enc["sha256"]}}, out / "models" / f"unknown_calibration_outer{fold}.pt")
        records.append({"outer_fold": fold, "roles": {k: v.tolist() for k,v in roles.items()}, "frozen_encoder": enc, "pair_training": pair_info, "directions": dirs})
    if args.smoke:
        write_json(out / "results.json", {"experiment": cfg["experiment"], "smoke": True, "runs": records, "test_accessed": False}); return
    summaries = {}
    for fpir in cfg["protocol"]["target_fpirs"]:
        rows = [r["directions"][d]["outer"][str(fpir)] for r in records for d in r["directions"]]
        summaries[str(fpir)] = numeric_mean(rows)
    gates = {str(f): {"known_acceptance_ge_95": summaries[str(f)]["known_acceptance"] >= cfg["gates"]["known_acceptance_min"], "unknown_false_acceptance_le_02": summaries[str(f)]["unknown_false_acceptance"] <= cfg["gates"]["unknown_false_acceptance_max"]} for f in cfg["protocol"]["target_fpirs"]}
    results = {"schema_version": 1, "experiment": cfg["experiment"], "completed_at": utc_now(), "config_sha256": sha256(path), "data": {"dataset": base["data"]["dataset"], "N": int(len(data.subject)), "competition_train_accessed": False, "competition_test_accessed": False}, "runs": records, "primary_summaries": summaries, "gates": gates, "stage_passed": bool(all(all(x.values()) for x in gates.values())), "official_score": "N/A", "test": {"AUC": "N/A", "F1": "N/A", "accessed": False}}
    write_json(out / "results.json", results)
    lines = ["# E82 严格 nested 留被试 pseudo-unknown 标定", "", "- 冻结 E74 encoder，复用 E79 pair score；拟合仅 enrolled run0–3，inner known 为 run4。", "- calibration_unknown 为 roles calibration_unknown 的 run5，仅用于固定 FPIR=1%/2% 分位阈值；outer unknown run5 仅最终一次评估。", "", "| target FPIR | AUROC | F1 | known acceptance | unknown false acceptance | DIR@FPIR1 |", "|---:|---:|---:|---:|---:|---:|"]
    for f, p in summaries.items(): lines.append(f"| {f} | {p['AUROC']:.6f} | {p['unknown_F1']:.6f} | {p['known_acceptance']:.6f} | {p['unknown_false_acceptance']:.6f} | {p['DIR_at_FPIR_1']:.6f} |")
    lines += ["", "Test AUC/F1=N/A；未访问竞赛 Test。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
