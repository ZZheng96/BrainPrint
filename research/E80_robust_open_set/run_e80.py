"""E80: frozen E74 embeddings with robust classwise Mahalanobis min-fusion."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

E80_DIR = Path(__file__).resolve().parent
ROOT = E80_DIR.parents[1]
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
E79_DIR = ROOT / "research" / "E79_pair_open_set"
E73_DIR = ROOT / "research" / "E73_external_multisession"
for p in (E80_DIR, E74_DIR, E79_DIR, E73_DIR, ROOT / "research" / "E72_pair_protocol"):
    sys.path.insert(0, str(p))
from external_data import load_external_data, subject_folds
from run_e74 import embed_indices, sha256, utc_now, write_json
from run_e79 import load_frozen_encoder, pair_scores
from pair_head import subject_templates
from pretrain import evaluate_queries, operating_metrics, pair_metrics, eer, numeric_mean

DIRECTIONS = (("T", "E"), ("E", "T"))

def robust_stats(model, data, roles, base_config, cfg, source):
    idx = data.indices(roles["enrolled"], session=source, runs=cfg["protocol"]["fit_runs"])
    emb = embed_indices(model, data.raw, idx, base_config).detach().cpu().numpy()
    classes = roles["enrolled"]
    means, invcov, medians, mads = [], [], [], []
    for subject in classes:
        x = emb[data.subject[idx] == subject]
        mean = x.mean(axis=0)
        centered = x - mean
        cov = centered.T @ centered / max(len(x) - 1, 1)
        diag = np.trace(cov) / cov.shape[0]
        cov = (1.0 - cfg["mahalanobis"]["covariance_shrinkage"]) * cov + cfg["mahalanobis"]["covariance_shrinkage"] * diag * np.eye(cov.shape[0])
        cov += cfg["mahalanobis"]["covariance_floor"] * np.eye(cov.shape[0])
        inv = np.linalg.pinv(cov)
        dist = np.sqrt(np.maximum(0.0, np.einsum("nd,dd,nd->n", centered, inv, centered)))
        med = np.median(dist)
        mad = np.median(np.abs(dist - med)) * 1.4826
        means.append(mean); invcov.append(inv); medians.append(med); mads.append(max(mad, cfg["mahalanobis"]["mad_floor"]))
    return {"classes": classes.copy(), "means": np.asarray(means), "invcov": np.asarray(invcov), "medians": np.asarray(medians), "mads": np.asarray(mads)}

def maha_confidence(embedding, stats, cfg):
    x = embedding.detach().cpu().numpy()
    d = np.stack([np.sqrt(np.maximum(0.0, np.einsum("nd,dd,nd->n", x - mean, inv, x - mean))) for mean, inv in zip(stats["means"], stats["invcov"])], axis=1)
    z = (d - stats["medians"][None, :]) / stats["mads"][None, :]
    return -z / float(cfg["fusion"]["mahalanobis_temperature"])

def fused_scores(pair_logits, maha_z, cfg):
    pair_conf = 1.0 / (1.0 + np.exp(-pair_logits / float(cfg["fusion"]["pair_temperature"])))
    maha_conf = 1.0 / (1.0 + np.exp(-maha_z))
    return np.minimum(pair_conf, maha_conf)

def evaluate_direction(model, head, data, roles, base_config, cfg, source, target):
    template_idx = data.indices(roles["enrolled"], session=source, runs=[cfg["protocol"]["template_run"]])
    known_idx = data.indices(roles["enrolled"], session=target, runs=[cfg["protocol"]["calibration_known_query_run"]])
    diagnostic_idx = data.indices(roles["calibration_unknown"], session=target, runs=[cfg["protocol"]["calibration_unknown_query_run"]])
    outer_known_idx = data.indices(roles["enrolled"], session=target, runs=[cfg["protocol"]["evaluation_query_run"]])
    outer_unknown_idx = data.indices(roles["outer_unknown"], session=target, runs=[cfg["protocol"]["evaluation_query_run"]])
    template = subject_templates(embed_indices(model, data.raw, template_idx, base_config), data.subject[template_idx], roles["enrolled"])
    stats = robust_stats(model, data, roles, base_config, cfg, source)
    def score(idx):
        emb = embed_indices(model, data.raw, idx, base_config)
        return fused_scores(pair_scores(head, emb, template), maha_confidence(emb, stats, cfg), cfg)
    known, diagnostic, outer_known, outer_unknown = map(score, (known_idx, diagnostic_idx, outer_known_idx, outer_unknown_idx))
    q02 = float(np.quantile(known.max(axis=1), 0.02, method="linear"))
    inner, _ = evaluate_queries(data.subject[known_idx], known, diagnostic, roles["enrolled"], "max_cosine", q02)
    outer_metrics, records = evaluate_queries(data.subject[outer_known_idx], outer_known, outer_unknown, roles["enrolled"], "max_cosine", q02)
    outer_known_confidence = outer_known.max(axis=1)
    outer_unknown_confidence = outer_unknown.max(axis=1)
    direct = roles["enrolled"][outer_known.argmax(axis=1)]
    outer_metrics["q02_known_acceptance"] = float(np.mean(outer_known_confidence >= q02))
    outer_metrics["q02_unknown_false_acceptance"] = float(np.mean(outer_unknown_confidence >= q02))
    outer_metrics["q02_unknown_rejection"] = 1.0 - outer_metrics["q02_unknown_false_acceptance"]
    outer_metrics.update(pair_metrics(outer_known, outer_unknown, data.subject[outer_known_idx], roles["enrolled"]))
    outer_metrics["detection_EER"] = eer(outer_known_confidence, outer_unknown_confidence)
    outer_metrics.update(operating_metrics(outer_known_confidence, outer_unknown_confidence, direct == data.subject[outer_known_idx]))
    return {"inner": {**inner, "calibration_unknown_false_acceptance_diagnostic": float(np.mean(diagnostic.max(axis=1) >= q02)), "threshold": q02}, "outer": outer_metrics}, stats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); path = (E80_DIR / args.config).resolve(); cfg = json.loads(path.read_text(encoding="utf-8"))
    base_path = E74_DIR / "config.json"; base = json.loads(base_path.read_text(encoding="utf-8-sig")); pair_cfg_path = E79_DIR / "config.json"; pair_cfg = json.loads(pair_cfg_path.read_text(encoding="utf-8"))
    out = E80_DIR / cfg["output_subdir"]; out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("E80 formal training/evaluation requires CUDA")
    device = torch.device("cuda"); torch.backends.cudnn.deterministic = True; data = load_external_data(base, device); folds = subject_folds(base)
    if args.smoke: folds = folds[:1]
    records = []
    for fold, roles in enumerate(folds, 1):
        model, encoder_info = load_frozen_encoder(base, base_path, roles, fold, device)
        from run_e79 import fit_pair_head
        pair_ckpt = E79_DIR / pair_cfg["output_subdir"] / "models" / f"pair_outer{fold}.pt"
        head, pair_info = fit_pair_head(model, data, roles, base, pair_cfg, pair_cfg_path, fold, pair_ckpt, False)
        dirs = {}; stats_info = {}
        for source, target in DIRECTIONS:
            result, stats = evaluate_direction(model, head, data, roles, base, cfg, source, target)
            direction = f"{source}_to_{target}"; dirs[direction] = result
            stats_info[direction] = {"classes": stats["classes"].tolist(), "means": stats["means"].tolist(), "medians": stats["medians"].tolist(), "mads": stats["mads"].tolist()}
        torch.save({"experiment": cfg["experiment"], "outer_fold": fold, "roles": {k: v.tolist() for k,v in roles.items()}, "stats": stats_info, "source_hashes": {"config": sha256(path), "run_e80": sha256(Path(__file__).resolve()), "e74_checkpoint": encoder_info["sha256"], "e79_pair_checkpoint": sha256(pair_ckpt)}}, out / "models" / f"robust_outer{fold}.pt")
        records.append({"outer_fold": fold, "roles": {k: v.tolist() for k,v in roles.items()}, "frozen_encoder": encoder_info, "pair_training": pair_info, "directions": dirs})
    if args.smoke:
        write_json(out / "results.json", {"experiment": cfg["experiment"], "smoke": True, "runs": records, "test_accessed": False}); return
    rows = [r["directions"][d]["outer"] for r in records for d in r["directions"]]; primary = numeric_mean(rows)
    primary["unknown_false_acceptance"] = primary["q02_unknown_false_acceptance"]
    results = {"schema_version": 1, "experiment": cfg["experiment"], "completed_at": utc_now(), "config_sha256": sha256(path), "data": {"dataset": base["data"]["dataset"], "N": int(len(data.subject)), "competition_train_accessed": False, "competition_test_accessed": False}, "runs": records, "primary_method": "min(pair_sigmoid, robust_mahalanobis_sigmoid)", "primary_summary": primary, "gates": {}, "stage_passed": False, "official_score": "N/A", "test": {"AUC": "N/A", "F1": "N/A", "accessed": False}}
    write_json(out / "results.json", results)
    p = primary; lines = ["# E80 robust open-set score", "", "## 预注册协议", "", "- 冻结 E74 encoder；pair head 与 E79 固定。classwise Mahalanobis 仅用 enrolled run0–2 拟合，run3 建模板。", "- 每类距离用训练集 robust median/MAD 标准化；最终 confidence 为 pair sigmoid 与 Mahalanobis sigmoid 的固定 min。", "- q02 仅由 enrolled run4 known confidence 标定；calibration unknown 仅诊断；run5 outer 只评估一次。", "", "## 主结果（3 折 × 2 方向均值）", "", "| 指标 | 数值 |", "|---|---:|", f"| Detection AUROC | {p['AUROC']:.6f} |", f"| Identity macro-F1 | {p['identity_macro_F1']:.6f} |", f"| Unknown-F1 | {p['unknown_F1']:.6f} |", f"| q02 known acceptance | {p['q02_known_acceptance']:.6f} |", f"| q02 unknown false acceptance | {p['q02_unknown_false_acceptance']:.6f} |", f"| DIR@FPIR1% | {p['DIR_at_FPIR_1']:.6f} |", f"| Pair AUROC | {p['pair_AUROC']:.6f} |", "", "结果未设置通过门槛；E80 仅报告预注册指标。", "", "Official Score=N/A；Test AUC/F1=N/A（未访问竞赛 Train/Test）。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(p, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
