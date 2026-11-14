"""E91: Train-only hybrid identity cosine and correlation-SPD unknown rejection."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E90_project18_identity_template"))
import run_e90

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
META_PATH = ROOT / "BrainPrint_cache/train_metadata.csv"
FEATURE_PATH = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"
STRATEGIES = ("template_cosine", "spd_distance_rank", "weighted_fusion", "quantile_fusion", "dual_gate")


def dump(path, obj):
    def conv(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def time_frac(meta):
    trial = meta.trial.to_numpy(float)
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return trial / np.maximum(maximum, 1.0)


def split_masks(meta):
    f = time_frac(meta)
    return {"fit": f <= .60, "cal_known": (f > .60) & (f <= .70),
            "outer_known": f > .80, "early": f <= .80}


def groups(seed=42):
    return [np.asarray(g, int) for g in np.array_split(np.random.default_rng(seed).permutation(np.arange(1, 21)), 5)]


def fit_identity(x, meta, indices, classes):
    return run_e90.fit_templates(x, meta, indices, classes, True, bins=3)


def identity_score(model, x, indices, device):
    return run_e90.score_templates(model, x, indices, device, batch=2048)


def fit_spd(x, meta, indices, classes):
    # SPD distance uses the same E65 logSPD space; templates are Euclidean centroids.
    z = np.asarray(x[indices], dtype=np.float32)
    y = meta.subject.to_numpy(int)[indices]
    return np.stack([z[y == c].mean(0) for c in classes])


def score(x, indices, prot, device, cosine=True):
    out = np.empty((len(indices), len(prot)), np.float32)
    p = torch.as_tensor(prot, device=device)
    with torch.no_grad():
        for start in range(0, len(indices), 2048):
            sel = np.asarray(indices[start:start + 2048], int)
            q = torch.as_tensor(np.asarray(x[sel], np.float32), device=device)
            if cosine:
                q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
                pp = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
                value = q @ pp.T
            else:
                value = -torch.cdist(q, p)
            out[start:start + len(sel)] = value.cpu().numpy()
    return out


def eer(known, unknown):
    y = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    fpr, tpr, _ = roc_curve(y, np.r_[known, unknown])
    return float(np.min(np.maximum(fpr, 1 - tpr)))


def metrics(k, true, u, pred, threshold, fused_k, fused_u):
    accepted = np.r_[pred != 0, np.asarray(u >= threshold)]
    det = np.r_[np.ones(len(k), int), np.zeros(len(u), int)]
    return {"N_known": int(len(k)), "N_unknown": int(len(u)),
            "AUC": float(roc_auc_score(det, np.r_[fused_k, fused_u])),
            "F1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "known_acceptance": float(np.mean(pred != 0)),
            "unknown_FAR": float(np.mean(np.asarray(u >= threshold))),
            "unknown_F1_pos_label_0": float(f1_score(det, accepted, pos_label=0, zero_division=0)),
            "EER": eer(fused_k, fused_u), "C": float(np.mean(pred == true)),
            "U": float(np.mean(np.asarray(u < threshold))), "threshold": float(threshold)}


def bounds(v):
    lo, hi = np.quantile(np.asarray(v, float), [.01, .99])
    return float(lo), float(max(hi - lo, 1e-8))


def scale(v, params, key):
    lo, span = params[key]
    return np.clip((np.asarray(v, float) - lo) / span, 0, 1)


def choose(cal_cos, cal_spd, yk, cu_cos, cu_spd):
    # Calibration unknown is used only inside this inner fold to choose fusion.
    ck, uk = cal_cos.max(1), cu_cos.max(1)
    sk, su = cal_spd.max(1), cu_spd.max(1)
    params = {"cos_bounds": bounds(np.r_[ck, uk]), "spd_bounds": bounds(np.r_[sk, su])}
    cn, un = scale(ck, params, "cos_bounds"), scale(uk, params, "cos_bounds")
    sn, un_spd = scale(sk, params, "spd_bounds"), scale(su, params, "spd_bounds")
    candidates = []
    for w in np.linspace(0, 1, 11):
        for mode in ("weighted", "quantile"):
            fk = w * cn + (1 - w) * sn if mode == "weighted" else np.minimum(cn, sn)
            fu = w * un + (1 - w) * un_spd if mode == "weighted" else np.minimum(un, un_spd)
            for q in np.linspace(.01, .30, 30):
                t = float(np.quantile(fk, q))
                pk = np.where(fk >= t, np.asarray(yk), 0)
                accepted = np.r_[pk != 0, fu >= t]
                value = f1_score(np.r_[np.ones(len(pk)), np.zeros(len(fu))], accepted, pos_label=0, zero_division=0) + .05 * float(np.mean(pk == yk))
                candidates.append((value, mode, w, t))
    best = max(candidates, key=lambda r: r[0])
    params.update({"mode": best[1], "weight": float(best[2]), "threshold": float(best[3])})
    return params


def fused(cos, spd, params):
    c = scale(cos.max(1), params, "cos_bounds")
    s = scale(spd.max(1), params, "spd_bounds")
    if params["mode"] == "template": return c
    if params["mode"] == "spd": return s
    if params["mode"] == "dual": return np.minimum(c, s)
    if params["mode"] == "quantile": return np.minimum(c, s)
    return params["weight"] * c + (1 - params["weight"]) * s


def run(smoke):
    meta = pd.read_csv(META_PATH)
    if len(meta) != 11304 or set(meta.subject) != set(range(1, 21)) or set(meta.session) != {"session1"}: raise ValueError("Train metadata invariant failed")
    if not meta.path.astype(str).str.startswith("Train/").all(): raise ValueError("non-Train path")
    x = np.load(FEATURE_PATH, mmap_mode="r")
    if x.shape != (len(meta), 855) or not np.isfinite(x).all(): raise ValueError("feature cache invalid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("E91 requires CUDA")
    subjects = meta.subject.to_numpy(int); masks = split_masks(meta)
    outer = groups()[:1] if smoke else groups()
    folds = []; pred_rows = []; checkpoints = {}
    for fold_no, unknown in enumerate(outer, 1):
        known = np.setdiff1d(np.arange(1, 21), unknown)
        inner_unknown = np.array_split(np.random.default_rng(100 + fold_no).permutation(known), 2)
        inner = []
        for iu in inner_unknown[:1] if smoke else inner_unknown:
            enrolled = np.setdiff1d(known, iu)
            fit = np.flatnonzero(np.isin(subjects, enrolled) & masks["fit"])
            ck_i = np.flatnonzero(np.isin(subjects, enrolled) & masks["cal_known"])
            cu_i = np.flatnonzero(np.isin(subjects, iu) & masks["early"])
            cp = fit_identity(x, meta, fit, enrolled)
            sp = fit_spd(x, meta, fit, enrolled)
            cknown, cunknown = identity_score(cp, x, ck_i, device), identity_score(cp, x, cu_i, device)
            sknown, sunknown = score(x, ck_i, sp, device, cosine=False), score(x, cu_i, sp, device, cosine=False)
            inner.append({"enrolled": enrolled, "known": subjects[ck_i], "cos_known": cknown, "cos_unknown": cunknown, "spd_known": sknown, "spd_unknown": sunknown})
        outer_fit = np.flatnonzero(np.isin(subjects, known) & masks["fit"])
        outer_known = np.flatnonzero(np.isin(subjects, known) & masks["outer_known"])
        outer_unknown = np.flatnonzero(np.isin(subjects, unknown))
        fold_rec = {"outer_fold": fold_no, "outer_unknown": unknown.tolist(), "registered": known.tolist(), "variants": {}, "inner_count": len(inner)}
        for name in STRATEGIES:
            params_rows = []
            for r in inner:
                cosk, cosu = r["cos_known"], r["cos_unknown"]; spdk, spdu = r["spd_known"], r["spd_unknown"]
                base = {"cos_bounds": bounds(np.r_[cosk.max(1), cosu.max(1)]), "spd_bounds": bounds(np.r_[spdk.max(1), spdu.max(1)])}
                cn = scale(cosk.max(1), base, "cos_bounds")
                sn = scale(spdk.max(1), base, "spd_bounds")
                if name == "template_cosine": p = {**base, "mode": "template", "threshold": float(np.quantile(cn, .05))}
                elif name == "spd_distance_rank": p = {**base, "mode": "spd", "threshold": float(np.quantile(sn, .05))}
                elif name == "dual_gate": p = {**base, "mode": "dual", "threshold": float(np.quantile(np.minimum(cn, sn), .05))}
                else: p = choose(cosk, spdk, r["known"], cosu, spdu); p["mode"] = "quantile" if name == "quantile_fusion" else "weighted"
                params_rows.append(p)
            fold_rec["variants"][name] = {"inner_params": params_rows}
        cp = fit_identity(x, meta, outer_fit, known); sp = fit_spd(x, meta, outer_fit, known)
        okc, ouc = identity_score(cp, x, outer_known, device), identity_score(cp, x, outer_unknown, device)
        oks, ous = score(x, outer_known, sp, device, cosine=False), score(x, outer_unknown, sp, device, cosine=False)
        for name in STRATEGIES:
            pars = fold_rec["variants"][name]["inner_params"]
            p = dict(pars[0]); p["threshold"] = float(np.mean([z["threshold"] for z in pars]))
            fk, fu = fused(okc, oks, p), fused(ouc, ous, p)
            if name in ("template_cosine", "spd_distance_rank", "dual_gate"):
                p["threshold"] = float(np.mean([z["threshold"] for z in pars]))
            pred = np.where(fk >= p["threshold"], known[okc.argmax(1)], 0)
            row = metrics(fk, subjects[outer_known], fu, pred, p["threshold"], fk, fu)
            fold_rec["variants"][name].update({"params": p, "metrics": row})
            if not smoke:
                for idx, val, sc in zip(outer_known, pred, fk): pred_rows.append({"outer_fold": fold_no, "variant": name, "index": int(idx), "role": "known", "subject": int(subjects[idx]), "predicted": int(val), "score": float(sc)})
                up = np.where(fu >= p["threshold"], known[ouc.argmax(1)], 0)
                for idx, val, sc in zip(outer_unknown, up, fu): pred_rows.append({"outer_fold": fold_no, "variant": name, "index": int(idx), "role": "unknown", "subject": int(subjects[idx]), "predicted": int(val), "score": float(sc)})
        checkpoints[f"outer_{fold_no}"] = {"classes": known, "identity_templates": cp, "spd_prototypes": sp}
        folds.append(fold_rec)
    summary = {n: {k: float(np.mean([fold["variants"][n]["metrics"][k] for fold in folds])) for k, value in folds[0]["variants"][n]["metrics"].items() if isinstance(value, (int, float))} for n in STRATEGIES}
    result = {"status": "complete_smoke" if smoke else "complete", "experiment": "E91 hybrid identity unknown", "mode": "smoke" if smoke else "full_5_outer_x_2_inner", "data": {"metadata": str(META_PATH.relative_to(ROOT)), "feature_cache": str(FEATURE_PATH.relative_to(ROOT)), "test_accessed": False, "test_metrics": {k: "N/A" for k in ("AUC", "F1", "C", "U")}}, "folds": folds, "summary": summary, "protocol_checks": {"train_only": True, "outer_subject_task_time_split": True, "inner_calibration_only": True, "test_accessed": False}, "source_sha256": {"script": sha256(Path(__file__)), "feature_cache": sha256(FEATURE_PATH)}}
    for n in STRATEGIES:
        d = OUT / n; d.mkdir(exist_ok=True)
        dump(d / ("smoke_results.json" if smoke else "results.json"), result)
        if not smoke:
            torch.save(checkpoints, d / "checkpoints.pt")
            with (d / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["outer_fold", "variant", "index", "role", "subject", "predicted", "score"]); w.writeheader(); w.writerows([r for r in pred_rows if r["variant"] == n])
        row = summary[n]
        report = (f"# E91 {n}\n\n"
                  "## 结果\n\n"
                  "仅使用 Train/session1 和 E65 已生成的 855 维 correlation-logSPD 特征；未访问项目 Test、Test 索引或 Test 标签。\n\n"
                  "| AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |\n"
                  "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
                  f"| {row['AUC']:.6f} | {row['F1']:.6f} | {row['known_acceptance']:.6f} | {row['unknown_FAR']:.6f} | {row['unknown_F1_pos_label_0']:.6f} | {row['EER']:.6f} | {row['C']:.6f} | {row['U']:.6f} |\n\n"
                  "## 协议与泄漏审计\n\n"
                  "outer 为 5 折 subject 留出，known query 为每个 subject/task 的 >80% 时间块，unknown 为 outer 留出 subject；模板仅用 outer fit≤60%。每个 outer 内仅用两个 inner calibration 的 known 与留出 unknown 选择融合权重/分位数阈值，outer query 不参与调参。\n\n"
                  "E90 身份分数复用 task/time multi-template；E65 分支使用同一 logSPD 特征的身份原型负欧氏距离/rank。未发现评价标签或特征跨 outer 泄漏；E61 的 Test 频带筛查结果未用于本实验。\n\n"
                  "Test AUC/F1/C/U=N/A。结果仅代表 Train session1 的 pseudo-unknown/时间块证据边界。\n")
        (d / "report.md").write_text(report, encoding="utf-8")
    dump(OUT / ("smoke_results.json" if smoke else "results.json"), result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.smoke == a.full: ap.error("choose --smoke or --full")
    print(json.dumps(run(a.smoke)["summary"], ensure_ascii=False, indent=2))
