"""E81: frozen E74 encoder with preregistered per-identity lower-tail calibration."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

E81_DIR = Path(__file__).resolve().parent
ROOT = E81_DIR.parents[1]
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
E79_DIR = ROOT / "research" / "E79_pair_open_set"
E73_DIR = ROOT / "research" / "E73_external_multisession"
for p in (E81_DIR, E74_DIR, E79_DIR, E73_DIR, ROOT / "research" / "E72_pair_protocol"):
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

def evaluate(known_scores, unknown_scores, known_true, classes, thresholds, fraction):
    known_arg = known_scores.argmax(1); unknown_arg = unknown_scores.argmax(1)
    known_conf = known_scores[np.arange(len(known_scores)), known_arg]
    unknown_conf = unknown_scores[np.arange(len(unknown_scores)), unknown_arg]
    known_pred_direct = classes[known_arg]; unknown_pred_direct = classes[unknown_arg]
    known_thr = np.asarray([thresholds[int(c)] for c in known_pred_direct])
    unknown_thr = np.asarray([thresholds[int(c)] for c in unknown_pred_direct])
    known_pred = np.where(known_conf >= known_thr, known_pred_direct, 0)
    unknown_pred = np.where(unknown_conf >= unknown_thr, unknown_pred_direct, 0)
    detection_scores = np.r_[known_conf, unknown_conf]
    detection_true = np.r_[np.ones(len(known_conf)), np.zeros(len(unknown_conf))]
    accepted = np.r_[known_pred != 0, unknown_pred != 0]
    identity_f1 = f1_score(known_true, known_pred, labels=classes, average="macro", zero_division=0)
    unknown_f1 = f1_score(detection_true, accepted, pos_label=0, zero_division=0)
    result = {
        "N_known": int(len(known_true)), "N_unknown": int(len(unknown_scores)),
        "threshold_rule_tail_fraction": float(fraction),
        "identity_macro_F1": float(identity_f1), "unknown_F1": float(unknown_f1),
        "C": float(np.mean(known_pred == known_true)), "U": float(np.mean(unknown_pred == 0)),
        "known_acceptance": float(np.mean(known_pred != 0)),
        "q02_known_acceptance": float(np.mean(known_conf >= known_thr)),
        "q02_unknown_false_acceptance": float(np.mean(unknown_conf >= unknown_thr)),
        "q02_unknown_rejection": float(np.mean(unknown_conf < unknown_thr)),
        "AUROC": float(roc_auc_score(detection_true, detection_scores)),
    }
    result.update(operating_metrics(known_conf, unknown_conf, known_pred_direct == known_true))
    return result

def evaluate_direction(model, head, data, roles, base, cfg, source, target):
    ti = data.indices(roles["enrolled"], session=source, runs=[cfg["protocol"]["template_run"]])
    ki = data.indices(roles["enrolled"], session=target, runs=[cfg["protocol"]["calibration_known_query_run"]])
    ci = data.indices(roles["calibration_unknown"], session=target, runs=[cfg["protocol"]["calibration_unknown_query_run"]])
    oi = data.indices(roles["enrolled"], session=target, runs=[cfg["protocol"]["evaluation_query_run"]])
    ui = data.indices(roles["outer_unknown"], session=target, runs=[cfg["protocol"]["evaluation_query_run"]])
    template = subject_templates(embed_indices(model, data.raw, ti, base), data.subject[ti], roles["enrolled"])
    stats = robust_stats(model, data, roles, base, cfg, source)
    def score(idx):
        emb = embed_indices(model, data.raw, idx, base)
        return fused_scores(pair_scores(head, emb, template), maha_confidence(emb, stats, cfg), cfg)
    known_cal, diag, known_outer, unknown_outer = map(score, (ki, ci, oi, ui))
    thresholds, tail_values = tail_thresholds(known_cal, data.subject[ki], roles["enrolled"], cfg["protocol"]["tail_fraction"])
    inner = evaluate(known_cal, diag, data.subject[ki], roles["enrolled"], thresholds, cfg["protocol"]["tail_fraction"])
    outer = evaluate(known_outer, unknown_outer, data.subject[oi], roles["enrolled"], thresholds, cfg["protocol"]["tail_fraction"])
    outer.update(pair_metrics(known_outer, unknown_outer, data.subject[oi], roles["enrolled"]))
    return {"inner": {**inner, "calibration_unknown_false_acceptance_diagnostic": inner["q02_unknown_false_acceptance"], "thresholds": thresholds, "tail_samples": len(tail_values)}, "outer": outer}, stats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); path = E81_DIR / args.config; cfg = json.loads(path.read_text(encoding="utf-8"))
    base_path = E74_DIR / "config.json"; base = json.loads(base_path.read_text(encoding="utf-8-sig")); pair_path = E79_DIR / "config.json"; pair_cfg = json.loads(pair_path.read_text(encoding="utf-8"))
    out = E81_DIR / cfg["output_subdir"]; out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("E81 formal evaluation requires CUDA")
    device = torch.device("cuda"); torch.backends.cudnn.deterministic = True; data = load_external_data(base, device); folds = subject_folds(base)
    if args.smoke: folds = folds[:1]
    records = []
    for fold, roles in enumerate(folds, 1):
        model, enc = load_frozen_encoder(base, base_path, roles, fold, device)
        head, pair_info = fit_pair_head(model, data, roles, base, pair_cfg, pair_path, fold, out / "models" / f"pair_outer{fold}.pt", False)
        dirs = {}; stat_info = {}
        for source, target in DIRECTIONS:
            result, stats = evaluate_direction(model, head, data, roles, base, cfg, source, target)
            dirs[f"{source}_to_{target}"] = result
            stat_info[f"{source}_to_{target}"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items() if k not in ("invcov", "means")}
        torch.save({"experiment": cfg["experiment"], "outer_fold": fold, "roles": {k: v.tolist() for k,v in roles.items()}, "stats": stat_info, "source_hashes": {"config": sha256(path), "run_e81": sha256(Path(__file__).resolve()), "e74_checkpoint": enc["sha256"]}}, out / "models" / f"evt_outer{fold}.pt")
        records.append({"outer_fold": fold, "roles": {k: v.tolist() for k,v in roles.items()}, "frozen_encoder": enc, "pair_training": pair_info, "directions": dirs})
    if args.smoke:
        write_json(out / "results.json", {"experiment": cfg["experiment"], "smoke": True, "runs": records, "test_accessed": False}); return
    rows = [r["directions"][d]["outer"] for r in records for d in r["directions"]]; primary = numeric_mean(rows)
    gates = {"S1_known_acceptance_ge_95": bool(primary["q02_known_acceptance"] >= cfg["gates"]["S1_known_acceptance_min"]), "S2_unknown_false_acceptance_le_02": bool(primary["q02_unknown_false_acceptance"] <= cfg["gates"]["S2_unknown_false_acceptance_max"])}
    results = {"schema_version": 1, "experiment": cfg["experiment"], "completed_at": utc_now(), "config_sha256": sha256(path), "data": {"dataset": base["data"]["dataset"], "N": int(len(data.subject)), "competition_train_accessed": False, "competition_test_accessed": False}, "runs": records, "primary_method": "per-identity q02 of fused confidence", "primary_summary": primary, "gates": gates, "stage_passed": bool(all(gates.values())), "official_score": "N/A", "test": {"AUC": "N/A", "F1": "N/A", "accessed": False}}
    write_json(out / "results.json", results)
    p = primary; lines = ["# E81 预注册 EVT/尾部校准开放集", "", "## 协议", "", "- 冻结 E74 encoder；pair head 复用 E79，表示拟合和 pair head 训练只使用 enrolled run0–2；run3 建模板。", "- run4 enrolled known 按身份拟合固定 q02 下尾分位数；calibration unknown 仅诊断，run5 outer 只评估，不扫描阈值。", "", "## 主结果（3 折 × 2 方向均值）", "", "| 指标 | 数值 |", "|---|---:|", f"| Pair AUROC | {p['pair_AUROC']:.6f} |", f"| Detection AUROC | {p['AUROC']:.6f} |", f"| Identity macro-F1 | {p['identity_macro_F1']:.6f} |", f"| Unknown-F1 | {p['unknown_F1']:.6f} |", f"| q02 known acceptance | {p['q02_known_acceptance']:.6f} |", f"| q02 unknown false acceptance | {p['q02_unknown_false_acceptance']:.6f} |", f"| q02 unknown rejection | {p['q02_unknown_rejection']:.6f} |", f"| DIR@FPIR1% | {p['DIR_at_FPIR_1']:.6f} |", "", "## S1/S2 代理门禁", "", f"- S1 known acceptance ≥ 0.95：{'通过' if gates['S1_known_acceptance_ge_95'] else '未通过'}", f"- S2 unknown false acceptance ≤ 0.02：{'通过' if gates['S2_unknown_false_acceptance_le_02'] else '未通过'}", "", "无竞赛 Train/Test 访问；Official Score、Test AUC/F1=N/A。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8"); print(json.dumps(primary, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
