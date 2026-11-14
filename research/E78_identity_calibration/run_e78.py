"""E78: preregistered per-identity calibration on the frozen E74/E76 head."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

E78_DIR = Path(__file__).resolve().parent
ROOT = E78_DIR.parents[1]
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
sys.path.insert(0, str(E74_DIR)); sys.path.insert(0, str(ROOT / "research" / "E73_external_multisession"))
from external_data import load_external_data, subject_folds
from multiscale_model import MultiScaleNative22

CFG = json.loads((E78_DIR / "config.json").read_text(encoding="utf-8"))
BASE = json.loads((E74_DIR / "config.json").read_text(encoding="utf-8-sig"))

def sha256(path):
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def embed(model, raw, indices, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(indices), batch):
            idx = torch.as_tensor(indices[i:i + batch], dtype=torch.long, device=raw.device)
            out.append(model.embed(raw[idx]).float())
    return torch.cat(out) if out else torch.empty((0, 64), device=raw.device)

def load_encoder(data, roles, fold):
    ckpt = E74_DIR / "outputs" / "models" / f"supcon_outer{fold}.pt"
    if not ckpt.is_file(): raise FileNotFoundError(f"Missing frozen E74 checkpoint: {ckpt}")
    saved = torch.load(ckpt, map_location=data.raw.device, weights_only=False)
    expected = {"config": sha256(E74_DIR / "config.json"), "run_e74": sha256(E74_DIR / "run_e74.py"), "external_data": sha256(E74_DIR / "external_data.py"), "multiscale_model": sha256(E74_DIR / "multiscale_model.py")}
    if saved.get("source_hashes") != expected or saved.get("roles") != {k: v.tolist() for k, v in roles.items()}:
        raise ValueError("E74 checkpoint provenance mismatch")
    model = MultiScaleNative22(BASE).to(data.raw.device); model.load_state_dict(saved["model_state"]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, {"path": str(ckpt), "sha256": sha256(ckpt)}

def fit_head(train_x, train_y, template_x, template_y, classes):
    means = torch.stack([template_x[torch.as_tensor(template_y == c, device=template_x.device)].mean(0) for c in classes])
    lookup = {int(c): i for i, c in enumerate(classes)}
    residual = train_x - torch.stack([means[lookup[int(y)]] for y in train_y])
    cov = (residual.T @ residual) / max(1, len(residual) - 1)
    sh = float(CFG["head"]["shrinkage"]); cov = (1 - sh) * cov + sh * torch.diag(torch.diag(cov))
    floor = float(CFG["head"]["eigenvalue_floor"])
    inv = torch.linalg.pinv(cov + floor * torch.eye(cov.shape[0], device=cov.device))
    return means, inv

def scores(x, means, inv):
    d = x[:, None, :] - means[None, :, :]
    return -torch.einsum("nkd,df,nkf->nk", d, inv, d).cpu().numpy()

def thresholds(known_scores, known_labels, classes, q):
    return {str(int(c)): float(np.quantile(known_scores[known_labels == c, list(classes).index(c)], q)) for c in classes}

def metrics(known_s, unknown_s, true, classes, per_id_thresholds):
    ks = known_s.max(1); kp = classes[known_s.argmax(1)]; kt = np.asarray([per_id_thresholds[str(int(c))] for c in kp])
    us_idx = unknown_s.argmax(1); us = unknown_s[np.arange(len(unknown_s)), us_idx]; ut = np.asarray([per_id_thresholds[str(int(c))] for c in classes[us_idx]])
    accepted = ks >= kt; unknown_accept = us >= ut
    labels = np.r_[np.ones(len(ks)), np.zeros(len(us))]; det = np.r_[accepted, unknown_accept]
    pred = np.where(accepted, kp, 0)
    return {"N_known": len(ks), "N_unknown": len(us), "AUROC": float(roc_auc_score(labels, np.r_[ks, us])), "identity_macro_F1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)), "unknown_F1": float(f1_score(labels, det, pos_label=0, zero_division=0)), "known_acceptance": float(accepted.mean()), "unknown_false_acceptance": float(unknown_accept.mean()), "unknown_rejection": float((~unknown_accept).mean()), "C": float(np.mean(pred == true))}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("E78 formal evaluation requires CUDA")
    data = load_external_data(BASE, torch.device("cuda")); folds = subject_folds(BASE)[:1] if args.smoke else subject_folds(BASE); runs = []
    for fold, roles in enumerate(folds, 1):
        model, enc = load_encoder(data, roles, fold); enrolled = roles["enrolled"]; directions = {}
        for source, target in (("T", "E"), ("E", "T")):
            train_i = data.indices(enrolled, session=source, runs=[0, 1, 2]); train_j = data.indices(enrolled, session=target, runs=[0, 1, 2]); templ_i = data.indices(enrolled, session=source, runs=[3]); known_i = data.indices(enrolled, session=target, runs=[4]); cal_u = data.indices(roles["calibration_unknown"], session=target, runs=[5]); outer_i = data.indices(enrolled, session=target, runs=[5]); outer_u = data.indices(roles["outer_unknown"], session=target, runs=[5])
            tx, ex, template = embed(model, data.raw, train_i), embed(model, data.raw, train_j), embed(model, data.raw, templ_i)
            means, inv = fit_head(torch.cat([tx, ex]), np.r_[data.subject[train_i], data.subject[train_j]], template, data.subject[templ_i], enrolled)
            known = scores(embed(model, data.raw, known_i), means, inv); cal = scores(embed(model, data.raw, cal_u), means, inv); y = data.subject[known_i]
            all_thresholds = {}
            for q in (0.05, 0.02): all_thresholds[f"q{int(q * 100):02d}"] = thresholds(known, y, enrolled, q)
            inner = {name: metrics(known, cal, y, enrolled, ts) for name, ts in all_thresholds.items()}
            outer = "not_accessed_in_smoke" if args.smoke else {name: metrics(scores(embed(model, data.raw, outer_i), means, inv), scores(embed(model, data.raw, outer_u), means, inv), data.subject[outer_i], enrolled, ts) for name, ts in all_thresholds.items()}
            hp = E78_DIR / "outputs" / "models" / f"mahalanobis_calibrated_outer{fold}_{source}_to_{target}.pt"; hp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"means": means.detach().cpu(), "inverse_covariance": inv.detach().cpu(), "classes": np.asarray(enrolled), "per_identity_thresholds": all_thresholds, "encoder": enc}, hp)
            directions[f"{source}_to_{target}"] = {"inner": inner, "outer": outer, "checkpoint": str(hp), "threshold_source": "enrolled known run4 only", "calibration_unknown_diagnostic_only": True}
        runs.append({"outer_fold": fold, "roles": {k: v.tolist() for k, v in roles.items()}, "frozen_encoder": enc, "directions": directions})
    result = {"experiment": CFG["experiment"], "smoke": args.smoke, "runs": runs, "test": {"AUC": "N/A", "F1": "N/A", "accessed": False}, "competition_test_accessed": False}
    if not args.smoke:
        for q in ("q05", "q02"):
            rows = [r["directions"][d]["outer"][q] for r in runs for d in r["directions"]]
            result.setdefault("summary", {})[q] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float))}
        p = result["summary"]["q05"]; result["gates"] = {"S1_known_acceptance_ge_95": p["known_acceptance"] >= .95, "S2_unknown_false_acceptance_le_2": p["unknown_false_acceptance"] <= .02}
    out = E78_DIR / "outputs"; out.mkdir(exist_ok=True); (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        p = result["summary"]["q05"]; q = result["summary"]["q02"]; g = result["gates"]
        (out / "report.md").write_text(f"# E78 per-identity calibration\n\n冻结 E74 encoder，复用 E76 Mahalanobis score。每折/方向仅用 enrolled run0–2 训练，run3 建 prototype；每个身份阈值仅由 enrolled run4 known 的该身份 score 计算 q05，q02 作为预注册报告；calibration unknown 不参与阈值。run5 为 outer 评估。\n\n| 指标 | q05 三折两方向均值 | q02 三折两方向均值 |\n|---|---:|---:|\n| AUROC | {p['AUROC']:.6f} | {q['AUROC']:.6f} |\n| Identity macro-F1 | {p['identity_macro_F1']:.6f} | {q['identity_macro_F1']:.6f} |\n| Unknown-F1 | {p['unknown_F1']:.6f} | {q['unknown_F1']:.6f} |\n| S1 known acceptance | {p['known_acceptance']:.6f} | {q['known_acceptance']:.6f} |\n| S2 unknown false acceptance | {p['unknown_false_acceptance']:.6f} | {q['unknown_false_acceptance']:.6f} |\n\nS1 门禁：{'通过' if g['S1_known_acceptance_ge_95'] else '失败'}；S2 门禁：{'通过' if g['S2_unknown_false_acceptance_le_2'] else '失败'}。失败如实记录，未扫阈值。Test AUC/F1=N/A，未访问 Test。逐折阈值和结果见 results.json，head checkpoint 见 outputs/models。\n", encoding="utf-8")

if __name__ == "__main__": main()
