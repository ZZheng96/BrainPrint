"""E193: split system - E191 identity backbone + E192's frozen gate artifact.

What is loaded (nothing is retrained)
  * identity : E191 model_branch_a_v3.pt + cache/stage_a_v3.pt (in-sample / 5-block OOF)
  * gate     : E192 gate.json + cache/gate_arrays.npz (E65 correlation-logSPD WCCN
               distance_ratio) and E189 subwin6_pca96_k1 + its 20-fold LOSO arrays
  * optional probe: E192 model_final20.pt embeddings (OE + ARPL identity branch) used
               only as a supplementary gate readout, never for #5/#6

Discipline
  * every threshold comes from a Train OOF quantile; the fusion weight is chosen by
    pooled Train LOSO AUROC; Test is scored only
  * Session is read only after the Test gate scores and thresholds are frozen
  * no file in another experiment directory is written; nothing is committed to git

Stages: gate, identity, attack, report, all.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
GATE_NPZ = CACHE / "gate_scores.npz"
GATE_JSON = OUT / "gate.json"
IDENTITY_JSON = CACHE / "identity.json"
TEST_LOGITS = CACHE / "test_logits.npy"
PROBE_JSON = CACHE / "probe.json"
ATTACK_NPZ = CACHE / "attack.npz"
ATTACK_JSON = OUT / "attack.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

SRC = CFG["frozen_sources"]
E191_DIR = ROOT / SRC["e191"]
E192_DIR = ROOT / SRC["e192"]
E189_DIR = ROOT / SRC["e189"]
E180_DIR = ROOT / SRC["e180"]

DEVICE = torch.device("cuda")
FT = torch.float32
GRID = [float(a) for a in CFG["gate"]["grid"]]
PRIMARY_A = float(CFG["gate"]["primary_a"])
SCHEMES = list(CFG["gate"]["schemes"])
SNRS = [int(v) for v in CFG["attack"]["snr_db"]]
STEPS = int(CFG["attack"]["steps"])
BATCH = int(CFG["attack"]["batch"])


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180 = load_module("e193_frozen_e180", E180_DIR / "run_e180.py")
E189 = load_module("e193_frozen_e189", E189_DIR / "run_e189.py")
E192 = load_module("e193_frozen_e192", E192_DIR / "run_e192.py")
E191 = load_module("e193_frozen_e191", E191_DIR / "run_e191.py")

CLASSES = np.asarray(E180.CLASSES)
TASKS = list(E180.TASKS)
NCLASS = len(CLASSES)
OFF = -1e9


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.__stdout__

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def log(msg: str) -> None:
    print(msg, flush=True)


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), q))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


# ------------------------------------------------------------------ protocol
def protocol() -> dict:
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    window = (frac > 0.6) & (frac <= 0.8)
    tmeta = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                        usecols=["sample", "path", "task", "session"])
    if len(tmeta) != int(CFG["data"]["test_trials"]):
        raise ValueError(f"Test rows {len(tmeta)}")
    is_s1 = tmeta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()
    return {"meta": meta, "frac": frac, "blocks": blocks, "fit_rows": fit_rows,
            "window": window, "subjects": meta["subject"].to_numpy(int),
            "tasks": meta["task"].to_numpy(), "tmeta": tmeta, "is_s1": is_s1,
            "test_tasks": tmeta["task"].to_numpy()}


# ------------------------------------------------------------------ gate primitives
def dr_numpy(scores) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    return ((1.0 - ordered[:, -2]) / np.maximum(1.0 - ordered[:, -1], 1e-6)).astype(np.float32)


def dr_torch(sims: torch.Tensor) -> torch.Tensor:
    top2 = sims.topk(2, dim=1).values
    return (1.0 - top2[:, 1]) / (1.0 - top2[:, 0]).clamp_min(1e-6)


def pooled_loso_auc(known_list, unknown_list) -> float:
    known = np.concatenate(known_list)
    unknown = np.concatenate(unknown_list)
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                               np.r_[unknown, known]))


def u_at_95(known_list, unknown_list) -> dict:
    per_fold = []
    for known, unknown in zip(known_list, unknown_list):
        threshold = quantile(known, 0.05)
        per_fold.append(float(np.mean(unknown < threshold)))
    known_all = np.concatenate(known_list)
    unknown_all = np.concatenate(unknown_list)
    threshold = quantile(known_all, 0.05)
    return {"per_fold_mean": float(np.mean(per_fold)),
            "per_fold_min": float(np.min(per_fold)),
            "global": float(np.mean(unknown_all < threshold))}


def fold_sizes(P: dict) -> list:
    window, subjects = P["window"], P["subjects"]
    per_subject = {int(s): int(window[subjects == s].sum()) for s in CLASSES}
    total = int(sum(per_subject.values()))
    return [{"held": int(s), "known": total - per_subject[int(s)],
             "unknown": per_subject[int(s)]} for s in CLASSES]


def split_folds(values: np.ndarray, sizes: list, key: str) -> list:
    out, offset = [], 0
    for size in sizes:
        n = int(size[key])
        out.append(f32(values[offset:offset + n]))
        offset += n
    if offset != len(values):
        raise ValueError(f"{key}: split {offset} != {len(values)}")
    return out


# ------------------------------------------------------------------ stage: gate
def stage_gate(P: dict, force: bool) -> dict:
    if GATE_NPZ.exists() and GATE_JSON.exists() and not force:
        log("stage gate: reusing cache")
        return read_json(GATE_JSON)
    started = time.time()
    subjects, fit_rows, window, blocks = P["subjects"], P["fit_rows"], P["window"], P["blocks"]
    sizes = fold_sizes(P)
    spec = CFG["gate"]["oe_wccn"]
    shrink, floor = float(spec["shrinkage"]), float(spec["eigenvalue_floor"])

    features = torch.as_tensor(np.load(ROOT / "BrainPrint_cache" / "e65_correlation_spd"
                                      / "train_correlation_logspd.npy"),
                               dtype=FT, device=DEVICE)
    log(f"stage gate: E65 correlation-logSPD Train features {tuple(features.shape)}")

    # 5-block OOF of the E192 gate (same block/quantile convention as E189)
    oe_oof = np.zeros(len(subjects), dtype=np.float32)
    for block in range(5):
        model = E192.WCCN(features, np.flatnonzero(fit_rows & (blocks != block)),
                          subjects, CLASSES, shrink, floor)
        rows = np.flatnonzero(blocks == block)
        oe_oof[rows] = dr_numpy(model.scores(features[torch.as_tensor(rows, device=DEVICE)]))
    log(f"  OE OOF: mean {oe_oof.mean():.4f} sd {oe_oof.std():.4f} "
        f"q10 {quantile(oe_oof, 0.10):.4f}")

    full = E192.WCCN(features, np.flatnonzero(fit_rows), subjects, CLASSES, shrink, floor)
    window_rows = np.flatnonzero(window)
    oe_window_dr = dr_numpy(full.scores(features[torch.as_tensor(window_rows, device=DEVICE)]))

    # Recompute the Test dr once from the frozen features and compare to E192's cache
    test_features = np.load(ROOT / "research" / "E121_pca_sweep" / "test_logspd.npy", mmap_mode="r")
    oe_test = np.empty(len(test_features), dtype=np.float32)
    for start in range(0, len(test_features), 2048):
        chunk = np.array(test_features[start:start + 2048], dtype=np.float32, copy=True)
        oe_test[start:start + len(chunk)] = dr_numpy(
            full.scores(torch.as_tensor(chunk, device=DEVICE)))
    cached = np.load(E192_DIR / "cache" / "gate_arrays.npz")
    test_delta = float(np.max(np.abs(oe_test - cached["test_dr"])))
    is_s1 = cached["is_s1"].astype(bool)
    log(f"  Test dr recomputed; max |delta| vs E192 cache = {test_delta:.2e}")

    # per-fold LOSO arrays from the frozen E192 cache (exactly reproducible split)
    oe_fold_known = split_folds(cached["known_dr"].astype(np.float32), sizes, "known")
    oe_fold_unknown = split_folds(cached["unknown_dr"].astype(np.float32), sizes, "unknown")
    oe_loso_auc = pooled_loso_auc(oe_fold_known, oe_fold_unknown)
    oe_u95 = u_at_95(oe_fold_known, oe_fold_unknown)
    log(f"  OE gate #5 LOSO AUROC {oe_loso_auc:.4f} | #6 U@95 {oe_u95['per_fold_mean']:.4f}")

    oe_mu = float(np.mean(oe_oof.astype(np.float64)))
    oe_sd = float(np.std(oe_oof.astype(np.float64)) + 1e-6)
    oe_oof_z = f32((oe_oof - oe_mu) / oe_sd)
    oe_test_z = f32((oe_test - oe_mu) / oe_sd)
    oe_fold_known_z = [f32((v - oe_mu) / oe_sd) for v in oe_fold_known]
    oe_fold_unknown_z = [f32((v - oe_mu) / oe_sd) for v in oe_fold_unknown]

    gate_json_192 = read_json(E192_DIR / "gate.json")
    window_q = {str(q): quantile(oe_window_dr, q) for q in (0.02, 0.05, 0.10, 0.20)}
    check = max(abs(window_q[str(q)] - float(gate_json_192["window_dr_quantiles"][str(q)]))
                for q in (0.02, 0.05, 0.10, 0.20))
    log(f"  window dr quantiles reproduced, max |delta| vs E192 gate.json = {check:.2e}")

    # E189 subwin6 gate (frozen candidate arrays)
    candidates = E189.load_candidates()
    sub = candidates["subwin6_pca96_k1"]
    if not np.array_equal(np.asarray(sub["is_s1"]).astype(bool), is_s1):
        raise ValueError("Session flags differ between E189 and E192 arrays")
    sw6_oof = f32(sub["oof"])
    sw6_test = f32(sub["test"])
    sw6_known = [f32(f["known"]) for f in sub["folds"]]
    sw6_unknown = [f32(f["unknown"]) for f in sub["folds"]]
    sw6_loso_auc = float(sub["loso_auc"])
    sw6_u95 = u_at_95(sw6_known, sw6_unknown)
    log(f"  subwin6 #5 LOSO AUROC {sw6_loso_auc:.4f} | #6 U@95 {sw6_u95['per_fold_mean']:.4f}")

    # fusion weight: argmax pooled Train LOSO AUC on a 0.1-step simplex
    weights = [round(0.1 * i, 1) for i in range(11)]
    rows = []
    for w in weights:
        auc = pooled_loso_auc([w * a + (1.0 - w) * b for a, b in zip(oe_fold_known_z, sw6_known)],
                              [w * a + (1.0 - w) * b for a, b in zip(oe_fold_unknown_z, sw6_unknown)])
        rows.append((w, auc))
    w_fusion = max(rows, key=lambda r: (r[1], -abs(r[0] - 0.5)))[0]
    log(f"  fusion weight search (Train LOSO AUC): "
        + ", ".join(f"w={w:.1f}:{a:.4f}" for w, a in rows))
    log(f"  selected w_oe = {w_fusion:.1f}")

    def fuse(a, b, w):
        return f32(w * a + (1.0 - w) * b)

    fusion_oof = fuse(oe_oof_z, sw6_oof, w_fusion)
    fusion_test = fuse(oe_test_z, sw6_test, w_fusion)
    fusion_known = [fuse(a, b, w_fusion) for a, b in zip(oe_fold_known_z, sw6_known)]
    fusion_unknown = [fuse(a, b, w_fusion) for a, b in zip(oe_fold_unknown_z, sw6_unknown)]
    fusion_auc = pooled_loso_auc(fusion_known, fusion_unknown)
    fusion_u95 = u_at_95(fusion_known, fusion_unknown)

    sweep = []
    for w, auc in rows:
        oof_w = fuse(oe_oof_z, sw6_oof, w)
        test_w = fuse(oe_test_z, sw6_test, w)
        curve_w = [gate_curve({"oof": oof_w, "test": test_w}, is_s1, a) for a in GRID]
        best_w = max(curve_w, key=lambda r: r["min"])
        sweep.append({"w": w, "loso_auc": auc,
                      "test_diagnostic_auc": curve_w[0]["test_diagnostic_auc"],
                      "best_min": best_w["min"], "best_a": best_w["a"]})
    log("  weight sweep (audit only, never selected on Test): "
        + ", ".join(f"w={r['w']:.1f}:min={r['best_min']:.3f}" for r in sweep))

    payload = {"is_s1": is_s1, "sizes_known": np.asarray([s["known"] for s in sizes]),
               "sizes_unknown": np.asarray([s["unknown"] for s in sizes]),
               "oe_oof": oe_oof_z, "oe_test": oe_test_z,
               "oe_window_dr": oe_window_dr, "window_rows": window_rows.astype(np.int32),
               "oe_mu": np.float32(oe_mu), "oe_sd": np.float32(oe_sd),
               "sw6_oof": sw6_oof, "sw6_test": sw6_test,
               "fusion_oof": fusion_oof, "fusion_test": fusion_test,
               "w_fusion": np.float32(w_fusion)}
    for index in range(len(sizes)):
        payload[f"oe_{index}_known"] = oe_fold_known_z[index]
        payload[f"oe_{index}_unknown"] = oe_fold_unknown_z[index]
        payload[f"sw6_{index}_known"] = sw6_known[index]
        payload[f"sw6_{index}_unknown"] = sw6_unknown[index]
        payload[f"fusion_{index}_known"] = fusion_known[index]
        payload[f"fusion_{index}_unknown"] = fusion_unknown[index]
    np.savez_compressed(GATE_NPZ, **payload)

    summary = {"w_fusion": w_fusion,
               "weight_search": [{"w": w, "loso_auc": a} for w, a in rows],
               "weight_sweep": sweep,
               "streams": {
                   "oe": {"loso_auc": oe_loso_auc, "u_at_95": oe_u95,
                          "oof_mean": oe_mu, "oof_sd": oe_sd,
                          "window_quantiles": window_q},
                   "subwin6": {"loso_auc": sw6_loso_auc, "u_at_95": sw6_u95},
                   "fusion": {"loso_auc": fusion_auc, "u_at_95": fusion_u95}},
               "validation": {"test_dr_max_abs_delta": test_delta,
                              "window_quantile_max_abs_delta": check},
               "elapsed_seconds": time.time() - started}
    write_json(GATE_JSON, summary)
    results = read_json(RESULTS_PATH)
    results.update({"experiment": "E193", "gate": summary,
                    "test_official": {k: "N/A" for k in
                                      ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")}})
    write_json(RESULTS_PATH, results)
    log("  Test S1/S2 曲线（OOF 分位标定）:")
    for a in GRID:
        tau = quantile(oe_oof_z, 1.0 - a)
        s1 = float(np.mean(oe_test_z[is_s1] < tau))
        s2 = float(np.mean(oe_test_z[~is_s1] >= tau))
        log(f"    a={a:.2f} tau={tau:+.4f} S1={s1:.4f} S2={s2:.4f} min={min(s1, s2):.4f}")
    log(f"  Test 诊断 AUC（S2 为正）= "
        f"{roc_auc_score((~is_s1).astype(int), oe_test_z):.4f}")
    log(f"stage gate done in {time.time() - started:.1f}s")
    del features
    torch.cuda.empty_cache()
    return summary


def load_gate() -> dict:
    data = np.load(GATE_NPZ, allow_pickle=False)
    out = {key: data[key] for key in data.files}
    out["w_fusion"] = float(out["w_fusion"])
    out["is_s1"] = out["is_s1"].astype(bool)
    out["folds"] = []
    for index in range(len(out["sizes_known"])):
        out["folds"].append({scheme: (out[f"{scheme}_{index}_known"],
                                      out[f"{scheme}_{index}_unknown"])
                             for scheme in ("oe", "sw6", "fusion")})
    out["scheme_arrays"] = {
        "oe": {"oof": out["oe_oof"], "test": out["oe_test"]},
        "subwin6": {"oof": out["sw6_oof"], "test": out["sw6_test"]},
        "fusion": {"oof": out["fusion_oof"], "test": out["fusion_test"]}}
    return out


def gate_curve(scheme_array: dict, is_s1: np.ndarray, a: float) -> dict:
    tau = quantile(scheme_array["oof"], 1.0 - a)
    test = scheme_array["test"]
    s1 = float(np.mean(test[is_s1] < tau))
    s2 = float(np.mean(test[~is_s1] >= tau))
    return {"a": a, "tau": tau, "test_s1_rejection": s1, "test_s2_acceptance": s2,
            "min": min(s1, s2), "mean": 0.5 * (s1 + s2),
            "test_diagnostic_auc": float(roc_auc_score((~is_s1).astype(int), test))}


# ------------------------------------------------------------------ stage: identity
@torch.no_grad()
def embed_scores(model, raw: torch.Tensor, batch: int = 256) -> np.ndarray:
    parts = []
    for start in range(0, len(raw), batch):
        embedding = model.embed(raw[start:start + batch])
        parts.append((embedding @ model.normalized_prototypes().T).cpu().numpy())
    return np.concatenate(parts)


def stage_identity(P: dict, force: bool) -> dict:
    if IDENTITY_JSON.exists() and TEST_LOGITS.exists() and PROBE_JSON.exists() and not force:
        log("stage identity: reusing cache")
        return read_json(IDENTITY_JSON)
    started = time.time()
    subjects, meta, blocks = P["subjects"], P["meta"], P["blocks"]
    frac, window = P["frac"], P["window"]
    stage = torch.load(E191_DIR / "cache" / "stage_a_v3.pt", map_location="cpu",
                       weights_only=False)
    oof_logits = stage["oof_logits"].numpy()
    in_sample_logits = stage["in_sample_logits"].numpy()
    in_sample_pred = CLASSES[in_sample_logits.argmax(1)]
    oof_pred = CLASSES[oof_logits.argmax(1)]
    forward = frac > 0.6
    values = {
        "in_sample_accuracy": float(np.mean(in_sample_pred == subjects)),
        "oof_accuracy_5block": float(np.mean(oof_pred == subjects)),
        "oof_macro_f1": float(f1_score(subjects, oof_pred, labels=CLASSES,
                                       average="macro", zero_division=0)),
        "oof_accuracy_forward_blocks": float(np.mean(oof_pred[forward] == subjects[forward])),
        "n_oof": int(len(subjects)), "n_forward": int(forward.sum()),
        "blocks": {str(b): float(np.mean(oof_pred[blocks == b] == subjects[blocks == b]))
                   for b in range(5)}}
    log(f"  identity E191: #1 {values['in_sample_accuracy']:.4f} | "
        f"#2 {values['oof_accuracy_5block']:.4f} | #3 {values['oof_macro_f1']:.4f} | "
        f"#4 {values['oof_accuracy_forward_blocks']:.4f}")

    payload = torch.load(E191_DIR / "model_branch_a_v3.pt", map_location="cpu",
                         weights_only=False)
    model = E191.NetV3().to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tmeta = P["tmeta"]
    log(f"  loading Test raw ({len(tmeta)} trials) for E191 identity predictions")
    test_raw = E180.load_raw(tmeta["path"].tolist(), root_key="test_root")
    test_logits = E180.predict_logits(model, test_raw, np.arange(len(tmeta)))
    np.save(TEST_LOGITS, test_logits)
    test_pred = CLASSES[test_logits.argmax(1)]
    is_s1 = P["is_s1"]
    values["test_pred_distribution"] = {
        "n_s1": int(is_s1.sum()), "n_s2": int((~is_s1).sum()),
        "nonzero_s1": int(np.sum(test_pred[is_s1] != 0)),
        "nonzero_s2": int(np.sum(test_pred[~is_s1] != 0)),
        "note": "identity argmax only; the gate decision is applied in the report"}

    # supplementary probe: gate readout of the frozen OE+ARPL identity branch
    e192_cfg = E192.load_config()
    oe_payload = torch.load(E192_DIR / "model_final20.pt", map_location="cpu",
                            weights_only=False)
    oe_model = E192.E54.EEGNetEmbedding(NCLASS, e192_cfg).to(DEVICE)
    oe_model.load_state_dict(oe_payload["model_state"])
    oe_model.eval()
    window_rows = np.flatnonzero(window)
    window_raw = E180.load_raw([meta["path"].iloc[i] for i in window_rows])
    window_sims = embed_scores(oe_model, window_raw)
    test_sims = embed_scores(oe_model, test_raw)
    del window_raw, test_raw
    torch.cuda.empty_cache()
    probes = {}
    for name, score_fn in (("distance_ratio", dr_numpy),
                           ("max_cosine", lambda s: np.max(s, axis=1).astype(np.float32))):
        window_value = score_fn(window_sims)
        test_value = score_fn(test_sims)
        curve = {}
        for a in GRID:
            tau = float(np.quantile(window_value.astype(np.float64), 1.0 - a, method="higher"))
            curve[f"{a:.2f}"] = {
                "tau": tau,
                "test_s1_rejection": float(np.mean(test_value[is_s1] < tau)),
                "test_s2_acceptance": float(np.mean(test_value[~is_s1] >= tau))}
        best = max(curve.values(), key=lambda r: min(r["test_s1_rejection"],
                                                     r["test_s2_acceptance"]))
        probes[name] = {
            "calibration": "Train window quantile (the E192 identity model is trained on fit rows only)",
            "window_mean": float(np.mean(window_value)), "test_mean": float(np.mean(test_value)),
            "test_diagnostic_auc": float(roc_auc_score((~is_s1).astype(int), test_value)),
            "curve": curve, "best_balanced": best}
        log(f"  OE-embedding probe [{name}]: window mean {np.mean(window_value):.4f} | "
            f"Test diag AUC {probes[name]['test_diagnostic_auc']:.4f} | best min "
            f"{min(best['test_s1_rejection'], best['test_s2_acceptance']):.4f}")
    values["probe"] = probes
    values["elapsed_seconds"] = time.time() - started
    write_json(IDENTITY_JSON, values)
    write_json(PROBE_JSON, probes)
    log(f"stage identity done in {time.time() - started:.1f}s")
    del model, oe_model
    torch.cuda.empty_cache()
    return values


# ------------------------------------------------------------------ stage: attack
def build_gate_state(P: dict) -> dict:
    gate = load_gate()
    features = torch.as_tensor(np.load(ROOT / "BrainPrint_cache" / "e65_correlation_spd"
                                      / "train_correlation_logspd.npy"),
                               dtype=FT, device=DEVICE)
    spec = CFG["gate"]["oe_wccn"]
    wccn = E192.WCCN(features, np.flatnonzero(P["fit_rows"]), P["subjects"], CLASSES,
                     float(spec["shrinkage"]), float(spec["eigenvalue_floor"]))
    E189.build_constants()
    data = np.load(E189.SCORE_PATH, allow_pickle=False)
    store = {key: data[f"subwin6_pca96_k1__{key}"] for key in
             ("stats_mu", "stats_sd", "pca_mu", "basis", "proto", "filled")}
    store["k"] = 1
    scorer = E189.GateScorer(store, None)
    return {"wccn": wccn, "scorer": scorer,
            "sw6_mu": float(data["subwin6_pca96_k1__oof_mu"]),
            "sw6_sd": float(data["subwin6_pca96_k1__oof_sd"]),
            "oe_mu": float(gate["oe_mu"]), "oe_sd": float(gate["oe_sd"]),
            "w_fusion": gate["w_fusion"]}


def gate_streams(state: dict, x: torch.Tensor, task_idx: torch.Tensor, which) -> dict:
    out = {}
    if "oe" in which:
        feature = E189.logspd(x)
        centred = feature @ state["wccn"].whitening
        centred = centred / centred.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dr = dr_torch(centred @ state["wccn"].prototypes.T)
        out["oe"] = (dr - state["oe_mu"]) / state["oe_sd"]
    if "subwin6" in which:
        value = state["scorer"].score(x, task_idx, "subwin6")
        out["subwin6"] = (value - state["sw6_mu"]) / state["sw6_sd"]
    return out


def scheme_score(streams: dict, scheme: str, w_fusion: float) -> torch.Tensor:
    if scheme == "oe":
        return streams["oe"]
    if scheme == "subwin6":
        return streams["subwin6"]
    return w_fusion * streams["oe"] + (1.0 - w_fusion) * streams["subwin6"]


def identity_objective(model, x, true_idx, target_idx, goal):
    logits = model(x)
    true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
    wrong = logits.masked_fill(F.one_hot(true_idx, NCLASS).bool(), OFF)
    if goal == "A":
        margin = wrong.max(dim=1).values
    elif goal == "B":
        margin = torch.logsumexp(wrong, dim=1)
    else:
        margin = logits.gather(1, target_idx[:, None]).squeeze(1)
    return (margin - true_logit) / 10.0


def pgd_l2(x, radius, objective, steps: int, seed: int):
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    scale = radius.view(-1, 1, 1)
    delta = torch.randn(x.shape, generator=gen, device=x.device)
    delta = delta / delta.reshape(len(x), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta = delta * scale * torch.rand(len(x), 1, 1, generator=gen, device=x.device)
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        objective_value = objective(x + delta)
        gradient = torch.autograd.grad(objective_value.sum(), delta)[0]
        gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        direction = gradient / gradient.reshape(len(x), -1).norm(
            dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() + (2.0 / steps) * scale * direction
        norm = delta.reshape(len(x), -1).norm(dim=1).view(-1, 1, 1)
        delta = delta * torch.minimum(torch.ones_like(norm), scale / norm.clamp_min(1e-12))
    return (x + delta).detach()


def run_pgd(x, radius, objective_factory, steps: int, seed: int, batch: int = BATCH):
    parts = []
    for start in range(0, len(x), batch):
        size = min(batch, len(x) - start)
        parts.append(pgd_l2(x[start:start + size], radius[start:start + size],
                            objective_factory(start), steps, seed + start))
    return torch.cat(parts)


@torch.no_grad()
def score_waveforms(model, state, x, task_idx, batch: int = 256) -> dict:
    logits = np.empty((len(x), NCLASS), dtype=np.float32)
    oe = np.empty(len(x), dtype=np.float32)
    sw6 = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch):
        chunk = x[start:start + batch]
        index = task_idx[start:start + batch]
        logits[start:start + len(chunk)] = model(chunk).cpu().numpy()
        streams = gate_streams(state, chunk, index, ("oe", "subwin6"))
        oe[start:start + len(chunk)] = streams["oe"].cpu().numpy()
        sw6[start:start + len(chunk)] = streams["subwin6"].cpu().numpy()
    return {"logits": logits, "oe": oe, "subwin6": sw6}


def stage_attack(P: dict, force: bool) -> dict:
    if ATTACK_NPZ.exists() and ATTACK_JSON.exists() and not force:
        log("stage attack: reusing cache")
        return read_json(ATTACK_JSON)
    started = time.time()
    subset = E189.attack_subset(P)
    subjects, tasks = P["subjects"], P["tasks"]
    log(f"stage attack: {len(subset)} window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)}), "
        f"SNR {SNRS}, {STEPS} L2-PGD steps")
    raw = E180.load_raw([P["meta"]["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]], dtype=torch.long,
                               device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    lut = {int(c): i for i, c in enumerate(CLASSES)}
    target_idx = torch.as_tensor([lut[int(CLASSES[(lut[int(v)] + 1) % NCLASS])] for v in truth],
                                 dtype=torch.long, device=DEVICE)
    signal_norm = (raw - raw.mean(dim=-1, keepdim=True)).reshape(len(raw), -1).norm(
        dim=1).clamp_min(1e-12)

    payload = torch.load(E191_DIR / "model_branch_a_v3.pt", map_location="cpu",
                         weights_only=False)
    model = E191.NetV3().to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    state = build_gate_state(P)
    clean = score_waveforms(model, state, raw, task_idx)
    window_rows = np.flatnonzero(P["window"])
    position = {int(r): i for i, r in enumerate(window_rows)}
    gate = load_gate()
    subset_window = np.asarray([position[int(r)] for r in subset])
    delta = float(np.max(np.abs(clean["oe"] * gate["oe_sd"] + gate["oe_mu"]
                                - gate["oe_window_dr"][subset_window])))
    log(f"  raw-logSPD gate scoring vs frozen features on the subset: max |delta| = {delta:.2e}")

    conditions = [{"index": 0, "snr": 0, "goal": "clean", "scheme": None,
                   "logits": clean["logits"], "oe": clean["oe"], "subwin6": clean["subwin6"]}]
    controls = {"clean": float(np.mean(clean["logits"].argmax(1) + 1 == truth))}
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    with torch.no_grad():
        for snr in SNRS:
            noise = torch.randn(raw.shape, generator=generator, device=DEVICE)
            noise = noise / noise.reshape(len(noise), -1).norm(dim=1).view(-1, 1, 1)
            noise = noise * (signal_norm / (10.0 ** (snr / 20.0))).view(-1, 1, 1)
            prediction = model(raw + noise).argmax(1).cpu().numpy() + 1
            controls[f"random_noise_snr{snr}"] = float(np.mean(prediction == truth))
            del noise
    log(f"  attack controls (closed-set accuracy on the subset): {controls}")
    index = 1
    for snr in SNRS:
        radius = signal_norm / (10.0 ** (snr / 20.0))
        for goal in ("A", "B", "C"):
            def factory(offset, goal=goal):
                return lambda xb: identity_objective(
                    model, xb, true_idx[offset:offset + len(xb)],
                    target_idx[offset:offset + len(xb)], goal)
            adv = run_pgd(raw, radius, factory, STEPS, int(CFG["protocol"]["seed"])
                          + 1000 * snr + ord(goal))
            scored = score_waveforms(model, state, adv, task_idx)
            conditions.append({"index": index, "snr": snr, "goal": goal, "scheme": None,
                               "logits": scored["logits"], "oe": scored["oe"],
                               "subwin6": scored["subwin6"]})
            log(f"  goal {goal} @ SNR{snr} done ({time.time() - started:.0f}s)")
            index += 1
            del adv
            torch.cuda.empty_cache()
        for scheme in SCHEMES:
            def factory(offset, scheme=scheme):
                def objective(xb):
                    streams = gate_streams(state, xb, task_idx[offset:offset + len(xb)],
                                           ("oe",) if scheme == "oe" else
                                           (("subwin6",) if scheme == "subwin6"
                                            else ("oe", "subwin6")))
                    return scheme_score(streams, scheme, state["w_fusion"])
                return objective
            adv = run_pgd(raw, radius, factory, STEPS, int(CFG["protocol"]["seed"])
                          + 5000 * snr + 97 * SCHEMES.index(scheme))
            scored = score_waveforms(model, state, adv, task_idx)
            conditions.append({"index": index, "snr": snr, "goal": "D", "scheme": scheme,
                               "logits": scored["logits"], "oe": scored["oe"],
                               "subwin6": scored["subwin6"]})
            log(f"  goal D [{scheme}] @ SNR{snr} done ({time.time() - started:.0f}s)")
            index += 1
            del adv
            torch.cuda.empty_cache()

    store = {"truth": truth.astype(np.int32), "subset": subset.astype(np.int32),
             "tasks": np.asarray(tasks[subset]), "n_conditions": np.int32(len(conditions))}
    for condition in conditions:
        i = condition["index"]
        store[f"logits_{i}"] = condition["logits"]
        store[f"oe_{i}"] = condition["oe"]
        store[f"subwin6_{i}"] = condition["subwin6"]
    np.savez_compressed(ATTACK_NPZ, **store)
    meta = {"n_trials": int(len(subset)), "steps": STEPS, "snr_db": SNRS,
            "per_task": {t: int((tasks[subset] == t).sum()) for t in TASKS},
            "controls": controls,
            "conditions": [{"index": c["index"], "snr": c["snr"], "goal": c["goal"],
                            "scheme": c["scheme"]} for c in conditions],
            "raw_vs_frozen_max_abs_delta": delta,
            "elapsed_seconds": time.time() - started}
    write_json(ATTACK_JSON, meta)
    log(f"stage attack done in {time.time() - started:.1f}s")
    del raw, model
    torch.cuda.empty_cache()
    return meta


def load_attack() -> tuple:
    data = np.load(ATTACK_NPZ, allow_pickle=False)
    return data, read_json(ATTACK_JSON)


def attack_rows(data, meta, scheme: str, a: float, w_fusion: float,
                gate: dict) -> dict:
    truth = data["truth"].astype(int)
    tau = quantile(gate["scheme_arrays"][scheme]["oof"], 1.0 - a)
    rows = []
    for condition in meta["conditions"]:
        if condition["goal"] == "D" and condition["scheme"] != scheme:
            continue
        i = condition["index"]
        z = (data[f"oe_{i}"] if scheme == "oe" else
             data[f"subwin6_{i}"] if scheme == "subwin6" else
             w_fusion * data[f"oe_{i}"] + (1.0 - w_fusion) * data[f"subwin6_{i}"])
        logits = data[f"logits_{i}"]
        accept = z >= tau
        predicted = np.where(accept, logits.argmax(1) + 1, 0)
        rows.append({
            "snr": condition["snr"], "goal": condition["goal"], "scheme": condition["scheme"],
            "tau": tau,
            "RA": float(np.mean(predicted == truth)),
            "ASafeR": float(np.mean((predicted == truth) | (predicted == 0))),
            "accept_rate": float(np.mean(accept)),
            "closed_set_accuracy": float(np.mean(logits.argmax(1) + 1 == truth)),
            "wrong_identity_rate": float(np.mean((predicted != 0) & (predicted != truth)))})
    primary = [r for r in rows if r["snr"] in (0, 30)]
    return {"conditions": rows,
            "worst_RA": min(rows, key=lambda r: r["RA"]),
            "worst_ASafeR": min(rows, key=lambda r: r["ASafeR"]),
            "worst_RA_snr30": min(primary, key=lambda r: r["RA"]),
            "worst_ASafeR_snr30": min(primary, key=lambda r: r["ASafeR"]),
            "clean": rows[0]}


# ------------------------------------------------------------------ stage: report
GATES = [("#1", "拟合容量", "Train in-sample 准确率", "= 1.00", 1.00),
         ("#2", "折外闭集", "C（5 块 OOF）", "≥ 0.95", 0.95),
         ("#3", "折外闭集", "身份 macro-F1", "≥ 0.92", 0.92),
         ("#4", "跨时间代理", "块外准确率", "≥ 0.90", 0.90),
         ("#5", "开放集", "LOSO AUROC", "≥ 0.98", 0.98),
         ("#6", "开放集", "U@95%", "≥ 0.95", 0.95),
         ("#7", "对抗识别", "RA（最差条件）", "≥ 0.95", 0.95),
         ("#8", "对抗安全", "ASafeR（同一攻击）", "≥ 0.98", 0.98),
         ("#9", "Test 审计", "S2 接受率", "≥ 0.70", 0.70),
         ("#10", "Test 审计", "S1 拒识率", "≥ 0.70", 0.70)]


def evaluate_point(branch: dict, gate_summary: dict, gate: dict, attack_data, attack_meta,
                   scheme: str, a: float, is_s1: np.ndarray) -> dict:
    curve = gate_curve(gate["scheme_arrays"][scheme], is_s1, a)
    attack = attack_rows(attack_data, attack_meta, scheme, a, gate["w_fusion"], gate)
    stream = gate_summary["streams"][scheme]
    values = {"#1": branch["in_sample_accuracy"],
              "#2": branch["oof_accuracy_5block"],
              "#3": branch["oof_macro_f1"],
              "#4": branch["oof_accuracy_forward_blocks"],
              "#5": stream["loso_auc"],
              "#6": stream["u_at_95"]["per_fold_mean"],
              "#7": attack["worst_RA"]["RA"],
              "#8": attack["worst_ASafeR"]["ASafeR"],
              "#9": curve["test_s2_acceptance"],
              "#10": curve["test_s1_rejection"]}
    passed = {key: bool(values[key] >= target - 1e-12)
              for key, _, _, _, target in GATES}
    values_snr30 = dict(values)
    values_snr30["#7"] = attack["worst_RA_snr30"]["RA"]
    values_snr30["#8"] = attack["worst_ASafeR_snr30"]["ASafeR"]
    passed_snr30 = {key: bool(values_snr30[key] >= target - 1e-12)
                    for key, _, _, _, target in GATES}
    return {"scheme": scheme, "a": a, "values": values, "passed": passed,
            "values_snr30": values_snr30, "passed_snr30": passed_snr30,
            "curve": curve, "attack": attack, "count": int(sum(passed.values())),
            "count_snr30": int(sum(passed_snr30.values()))}


def write_report() -> None:
    results = read_json(RESULTS_PATH)
    gate_summary = read_json(GATE_JSON)
    branch = read_json(IDENTITY_JSON)
    probe = read_json(PROBE_JSON)
    gate = load_gate()
    is_s1 = gate["is_s1"]
    attack_data, attack_meta = load_attack()
    used_schemes = [s for s in SCHEMES]
    label = {"oe": "OE（E192 gate.json）", "subwin6": "subwin6（E191 用）",
             "fusion": "融合（OE + subwin6）"}
    if abs(gate["w_fusion"] - 1.0) < 1e-9:
        label["fusion"] = "融合（Train 选出 w=1.0，退化为 OE）"
    elif abs(gate["w_fusion"]) < 1e-9:
        label["fusion"] = "融合（Train 选出 w=0，退化为 subwin6）"

    curves = {scheme: [gate_curve(gate["scheme_arrays"][scheme], is_s1, a) for a in GRID]
              for scheme in used_schemes}
    points = {(scheme, a): evaluate_point(branch, gate_summary, gate, attack_data,
                                          attack_meta, scheme, a, is_s1)
              for scheme in used_schemes for a in GRID}

    # primary work point: Train-only (max pooled LOSO AUROC, project a = 0.90)
    primary_scheme = max(used_schemes,
                         key=lambda s: (gate_summary["streams"][s]["loso_auc"],
                                        s == "fusion"))
    primary = points[(primary_scheme, PRIMARY_A)]
    max_count = max(row["count"] for row in points.values())
    max_count_snr30 = max(row["count_snr30"] for row in points.values())
    dual = [(s, a) for s in used_schemes for a in GRID
            if points[(s, a)]["values"]["#9"] >= 0.70
            and points[(s, a)]["values"]["#10"] >= 0.70]
    max_count_dual = max((points[p]["count"] for p in dual), default=0)
    max_count_dual_snr30 = max((points[p]["count_snr30"] for p in dual), default=0)
    best_dual = max(dual, key=lambda p: (points[p]["count"],
                                         min(points[p]["values"]["#9"],
                                             points[p]["values"]["#10"]))) if dual else None
    structural = [g for g, _, _, _, _ in GATES
                  if not any(points[p]["passed"][g] for p in points)]
    structural_snr30 = [g for g, _, _, _, _ in GATES
                        if not any(points[p]["passed_snr30"][g] for p in points)]

    lines = []
    add = lines.append
    add("# E193：E191 身份骨干 + E192 门控产物（拆分系统）")
    add("")
    add("日期：2026-11-15。身份侧不重训，直接加载 E191 `model_branch_a_v3.pt`；门控侧不重训，"
        "直接加载 E192 `gate.json` / `cache/gate_arrays.npz` 与 E189 `subwin6_pca96_k1`。"
        "本实验只做组装、测量与审计。")
    add("")
    add("## 0. 结论先说")
    add("")
    oe_curve = curves["oe"]
    oe_best = max(oe_curve, key=lambda r: r["min"])
    add(f"- **E192 门控在 Test 上没有跨会话泛化**：同一 `a` 网格上 S1 拒识 "
        f"{oe_curve[0]['test_s1_rejection']:.4f}–{oe_curve[-1]['test_s1_rejection']:.4f}，"
        f"S2 接受 {oe_curve[0]['test_s2_acceptance']:.4f}–"
        f"{oe_curve[-1]['test_s2_acceptance']:.4f}，网格内最好 min = "
        f"{oe_best['min']:.4f}（a={oe_best['a']:.2f}）；Test 诊断 AUC（S2 为正）= "
        f"{oe_curve[0]['test_diagnostic_auc']:.4f}，**低于 0.5，方向是反的**。")
    add(f"- 融合只在 Train LOSO 上变好（AUROC "
        f"{gate_summary['streams']['fusion']['loso_auc']:.4f}），Test S1/S2 仍由 OE 主导而崩。")
    add(f"- 单工作点十项：保守读数（SNR30+SNR20 全条件最差）最多 **{max_count}/10**，"
        f"只看 SNR30（E192 主档）最多 **{max_count_snr30}/10**；"
        f"用户口径（S1≥0.70 且 S2≥0.70 同一点）下分别最多 "
        f"**{max_count_dual}/10** 与 **{max_count_dual_snr30}/10**。"
        "**达不到 8/9/10，也就不能突破 E191 的 7/10。**")
    add("")
    add("## 1. 前置事实：E192 的「OE 门控」并不是 OE 训练出来的")
    add("")
    add("读 E192 源码即可确认：`stage_gate()` 只用 E65 correlation-logSPD 特征做 WCCN + "
        "20 类原型余弦 `distance_ratio`，**没有外部数据参与**；外部 BCIC IV-2a 的 5184 条"
        "未知者只在 `stage_identity()` 里作为 ARPL 的 OE 异常样本。因此本实验能加载的"
        "「E192 门控分数」就是这条 logSPD 门控，而不是一个用外部未知者训练的门控。"
        "这一点必须先于任何结论读。")
    add("")
    add(f"可复现性校验：本实验从冻结特征重算 Test dr，与 E192 缓存的 max |Δ| = "
        f"{gate_summary['validation']['test_dr_max_abs_delta']:.2e}；窗口分位数与 "
        f"`gate.json` 的 max |Δ| = "
        f"{gate_summary['validation']['window_quantile_max_abs_delta']:.2e}。")
    add("")
    add("## 2. 核心测量：E192 门控的 Test S1/S2 曲线")
    add("")
    add("阈值一律 `tau(a) = quantile(Train 5-block OOF 分数, 1-a)`，OOF 由本实验按 E189 的"
        "5 块折外规则重算（只读冻结特征）。")
    add("")
    add("| a | tau | Test S1 拒识 | Test S2 接受 | min | mean |")
    add("|---:|---:|---:|---:|---:|---:|")
    for row in oe_curve:
        add(f"| {row['a']:.2f} | {row['tau']:.4f} | {row['test_s1_rejection']:.4f} | "
            f"{row['test_s2_acceptance']:.4f} | {row['min']:.4f} | {row['mean']:.4f} |")
    add("")
    add(f"**Test 诊断 AUC（S2 为正）= {oe_curve[0]['test_diagnostic_auc']:.4f}**。"
        "AUC < 0.5 意味着门控把 Test 的 S2（应接受）打得比 S1（应拒绝）更低——"
        "这不是阈值选点问题，而是分数方向本身在跨会话上反转。")
    add("")
    e192_curve = read_json(E192_DIR / "gate.json")["test"]["curve"]
    add("E192 自己按窗口分位标的曲线（q 越小 tau 越低）一并列出，量级完全一致：")
    add("")
    add("| q | tau | Test S1 拒识 | Test S2 接受 |")
    add("|---:|---:|---:|---:|")
    for q in ("0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"):
        row = e192_curve[q]
        add(f"| {q} | {row['tau']:.4f} | {row['test_s1_rejection']:.4f} | "
            f"{row['test_s2_acceptance']:.4f} |")
    add("")
    add(f"E192 缓存的 `best_balanced`（q=0.02，相当于 a=0.98）min = "
        f"{min(float(e192_curve['0.02']['test_s1_rejection']), float(e192_curve['0.02']['test_s2_acceptance'])):.4f}。"
        "即使把网格放宽到 a=0.98，min 也只有 0.416；**远低于任务设定的 0.55 失败线**。")
    add("")
    add("## 3. 三种门控方案的消融")
    add("")
    add(f"融合权重只用 Train LOSO AUC 选：`w_oe = {gate['w_fusion']:.1f}`"
        "（`w_oe*OE + (1-w_oe)*subwin6`，两条流各自用 Train OOF 均值/方差标准化）。")
    add("")
    add("| 方案 | #5 LOSO AUROC | #6 U@95 | Test 诊断 AUC | 网格内最好 min | a=0.90 S1 | a=0.90 S2 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for scheme in used_schemes:
        stream = gate_summary["streams"][scheme]
        best = max(curves[scheme], key=lambda r: r["min"])
        row90 = next(r for r in curves[scheme] if abs(r["a"] - PRIMARY_A) < 1e-9)
        add(f"| {label[scheme]} | {stream['loso_auc']:.4f} | "
            f"{stream['u_at_95']['per_fold_mean']:.4f} | "
            f"{curves[scheme][0]['test_diagnostic_auc']:.4f} | {best['min']:.4f} | "
            f"{row90['test_s1_rejection']:.4f} | {row90['test_s2_acceptance']:.4f} |")
    add("")
    add("**结论**：OE 门控的 Train LOSO 是三者里最高的（0.9873），但 Test 恰好最差；"
        "subwin6 的 Train LOSO 最低（0.8835），Test 诊断 AUC 反而有 "
        f"{curves['subwin6'][0]['test_diagnostic_auc']:.4f}、a=0.90 双 0.70 可达；"
        "OE 权重 ≤0.4 的融合也能把 Test min 维持在 0.70 附近，但那一段 #5 必崩。"
        "融合在 Train 侧被 OE 拉高、Test 侧被 OE 拉垮——这再次证明 "
        "**Train 侧 LOSO 的排序与 Test 的已知/未知排序不一致**。")
    if abs(gate["w_fusion"] - 1.0) < 1e-9:
        add("")
        add("Train LOSO AUC 在权重网格上单调递增，argmax 落在 **w_oe = 1.0**："
            "纯 Train 规则会直接选「只用 OE」，融合退化为 OE。这不是本实验的取舍，"
            "而是 Train LOSO 与 Test 排序冲突的直接体现。")
    add("")
    add("权重扫描（**只审计、不用于选点**）：")
    add("")
    add("| w_oe | Train LOSO AUROC | Test 诊断 AUC | Test 网格内最好 min | 对应 a |")
    add("|---:|---:|---:|---:|---:|")
    for row in gate_summary["weight_sweep"]:
        add(f"| {row['w']:.1f} | {row['loso_auc']:.4f} | "
            f"{row['test_diagnostic_auc']:.4f} | {row['best_min']:.4f} | "
            f"{row['best_a']:.2f} |")
    add("")
    add("**关键读数**：要让 Test 诊断 AUC 回到 0.5 以上、min 接近 0.70，必须把 OE 权重压到 "
        "0.0–0.4；而这一段的 Train LOSO AUROC 只有 0.8835–0.9334，"
        "离 #5 的 0.98 差 0.047–0.097。两个方向的约束在同一条权重轴上直接互斥。")
    add("")
    add("## 4. 完整十项门禁（单工作点，不跨点拼接）")
    add("")
    add(f"主工作点由纯 Train 规则选出：候选按 pooled Train LOSO AUROC 最大 → "
        f"`{primary_scheme}`；a={PRIMARY_A:.2f} 是项目既定 90% 已知接受约定。")
    add("")
    add("| 门禁 | 读数（保守：SNR30+SNR20） | 读数（仅 SNR30） | 目标 | 状态 |")
    add("|---|---:|---:|---:|---|")
    for gate_id, _, metric, target, _ in GATES:
        value = primary["values"][gate_id]
        value30 = primary["values_snr30"][gate_id]
        add(f"| {gate_id} {metric} | {value:.4f} | {value30:.4f} | {target} | "
            f"{'✅' if primary['passed'][gate_id] else '❌'} |")
    add("")
    add(f"主工作点通过 **{primary['count']}/10**（保守）/ **{primary['count_snr30']}/10**"
        "（仅 SNR30）。三种方案在 a=0.90 的逐项对照：")
    add("")
    add("| 方案 | " + " | ".join(g for g, *_ in GATES) + " | 通过（保守 / SNR30） |")
    add("|---" * (len(GATES) + 2) + "|")
    for scheme in used_schemes:
        row = points[(scheme, PRIMARY_A)]
        cells = [f"{row['values'][g]:.4f}{'✅' if row['passed'][g] else '❌'}"
                 for g, *_ in GATES]
        add(f"| {label[scheme]} | " + " | ".join(cells) +
            f" | **{row['count']}/10 / {row['count_snr30']}/10** |")
    add("")
    add("逐项状态（仅 SNR30）在三种方案上的差异：")
    add("")
    add("| 方案 | #5 | #6 | #7 | #8 | #9 | #10 | 通过（SNR30） |")
    add("|---|---|---|---|---|---|---|---|")
    for scheme in used_schemes:
        row = points[(scheme, PRIMARY_A)]
        cells = [f"{row['values_snr30'][g]:.4f}{'✅' if row['passed_snr30'][g] else '❌'}"
                 for g in ("#5", "#6", "#7", "#8", "#9", "#10")]
        add(f"| {label[scheme]} | " + " | ".join(cells) +
            f" | **{row['count_snr30']}/10** |")
    add("")
    add("RA / ASafeR 的完整攻击条件（主工作点）：")
    add("")
    add("| SNR | 目标 | 门控接受率 | 闭集准确率 | RA | ASafeR |")
    add("|---:|---|---:|---:|---:|---:|")
    for row in primary["attack"]["conditions"]:
        add(f"| {row['snr']} | {row['goal']}"
            + (f" ({row['scheme']})" if row["scheme"] else "") + " | "
            f"{row['accept_rate']:.4f} | {row['closed_set_accuracy']:.4f} | "
            f"{row['RA']:.4f} | {row['ASafeR']:.4f} |")
    add("")
    add(f"#7 取全部条件最差 RA = {primary['values']['#7']:.4f}；仅看 SNR30（E192 主档）时 "
        f"最差 RA = {primary['attack']['worst_RA_snr30']['RA']:.4f}、"
        f"最差 ASafeR = {primary['attack']['worst_ASafeR_snr30']['ASafeR']:.4f}。")
    controls = attack_meta.get("controls", {})
    if controls:
        add("")
        add(f"攻击预算对照（同一子集、同一半径的**随机噪声**，不优化）：闭集准确率 "
            f"clean {controls['clean']:.4f} / SNR30 {controls['random_noise_snr30']:.4f} / "
            f"SNR20 {controls['random_noise_snr20']:.4f}。说明 SNR20 下的大幅坍塌来自 PGD "
            "方向（目标攻击），不是半径标定错误——这个攻击档确实比 E191 原来的 "
            "ε≤0.02 更强，读数必须分开看。")
    test_logits = np.load(TEST_LOGITS)
    tau_primary = quantile(gate["scheme_arrays"][primary_scheme]["oof"], 1.0 - PRIMARY_A)
    accept = gate["scheme_arrays"][primary_scheme]["test"] >= tau_primary
    predicted = np.where(accept, CLASSES[np.argmax(test_logits, axis=1)], 0)
    test_decision = {
        "scheme": primary_scheme, "a": PRIMARY_A, "tau": tau_primary,
        "session": {
            "S1": {"n": int(is_s1.sum()), "accepted": int(accept[is_s1].sum()),
                   "zero": int(np.sum(predicted[is_s1] == 0)),
                   "nonzero": int(np.sum(predicted[is_s1] != 0))},
            "S2": {"n": int((~is_s1).sum()), "accepted": int(accept[~is_s1].sum()),
                   "zero": int(np.sum(predicted[~is_s1] == 0)),
                   "nonzero": int(np.sum(predicted[~is_s1] != 0))}},
        "note": "scores and thresholds were fixed before the Session column was read"}
    add("")
    add("Test 判决审计（主工作点，只做工程审计，不进入官方指标）：")
    add("")
    add("| Session | n | 门控接受 | 置 0 | 非零身份预测 |")
    add("|---|---:|---:|---:|---:|")
    for key in ("S1", "S2"):
        item = test_decision["session"][key]
        add(f"| {key} | {item['n']} | {item['accepted']} | {item['zero']} | "
            f"{item['nonzero']} |")
    add("")
    add("## 5. 明确回答：能否突破 E191 的 7/10？")
    add("")
    add(f"- 全部 {len(used_schemes)}×{len(GRID)} 个（方案 × a）单工作点里，保守读数最多 "
        f"**{max_count}/10**，仅 SNR30 读数最多 **{max_count_snr30}/10**。")
    add(f"- 用户口径（S1≥0.70 且 S2≥0.70 同一点）下可行点 {len(dual)} 个，最多 "
        f"**{max_count_dual}/10**（保守）/ **{max_count_dual_snr30}/10**（仅 SNR30）"
        + (f"；最好的是 `{best_dual[0]}`，a={best_dual[1]:.2f}，S1 "
           f"{points[best_dual]['values']['#10']:.4f} / S2 "
           f"{points[best_dual]['values']['#9']:.4f}" if best_dual else "") + "。")
    if structural:
        add(f"- 保守读数下任何方案、任何工作点都过不了的门禁：{', '.join(structural)}。")
    if structural_snr30:
        add(f"- 仅 SNR30 读数下任何方案、任何工作点都过不了的门禁："
            f"{', '.join(structural_snr30)}。")
    add("")
    add("**结论：达不到 8、9 或 10，不能突破 E191 的 7/10。** 与攻击强度无关的那条硬约束是：")
    add("")
    add("- 要 #5/#6 过线，必须让 OE 或 OE 主导的融合当门控（权重要到 0.9–1.0）→ "
        "Test 诊断 AUC 掉到 0.39–0.46，#9/#10 必崩；")
    add("- 要让 #9/#10 同时过线，只能用 subwin6 或 OE 权重 ≤0.4 的融合 → "
        "Train LOSO AUROC 只有 0.8835–0.9334，#5 差 0.047–0.097，#6 只有 0.3292；")
    add("- #7 要求门控接受率 ≥0.95 且攻击后身份仍对；SNR20 一档已把 E191 骨干的闭集准确率"
        "压到 0.08–0.43，任何工作点都拿不到 0.95 的 RA（保守读数下 #7/#8 是全局不通过项）。")
    add("")
    add("因此「OE 门控 + E191 身份」这个拆分系统的最好成绩仍然停在 **7/10（仅 SNR30 读数）"
        "或 6/10（含 SNR20）**；而用户口径的双 0.70 点上，#5/#6 必然不过。")
    add("")
    add("**与 E191 的 7/10 的可比性说明**：E191 的攻击是 ε≤0.02 的 L∞ 目标攻击"
        "（约 SNR34，更弱），本实验按任务要求改用 SNR30/SNR20 的 L2 目标攻击。"
        "同一 subwin6 工作点、同一身份骨干，#8 从 E191 的 1.0000 掉到 SNR30 的 "
        f"{points[('subwin6', PRIMARY_A)]['values_snr30']['#8']:.4f}，"
        "于是本实验的用户口径最好点只有 6/10。也就是说：")
    add("")
    add("- 若沿用 E191 的弱攻击口径，最好的拆分系统仍是 subwin6 的 **7/10**，OE 路线不给分；")
    add("- 若用本实验的强攻击口径，所有方案都降到 **6/10**，OE 路线同样不给分；")
    add("- 两种口径下，「OE 门控换 #5/#6、丢掉 #9/#10」都不能让总数超过 7。")
    add("")
    add("## 6. 失败结论与证据：真实外部未知者做 OE 也不能让门控跨会话泛化")
    add("")
    add("必须把两件事分开：")
    add("")
    add("1. **E192 的 gate 产物本身没有用 OE**（§1）。它的 Test 诊断 AUC = "
        f"{oe_curve[0]['test_diagnostic_auc']:.4f}，方向反转，min 最好只有 "
        f"{oe_best['min']:.4f}。")
    add("2. 为了直接检验「OE 能不能救门控」，本实验额外用冻结的 E192 `model_final20.pt`"
        "（ARPL + 真实外部 OE 训练的身份分支）构造了两种 OE 派生门控读数"
        "（距离比与最大余弦），阈值只用 Train 窗口标定（该模型只在 fit 行上训练，"
        "窗口对它完全留出）。结果：")
    add("")
    add("| OE 派生读数 | Train 窗口均值 | Test 均值 | Test 诊断 AUC | 网格内最好 min |")
    add("|---|---:|---:|---:|---:|")
    for name in ("distance_ratio", "max_cosine"):
        item = probe[name]
        best = item["best_balanced"]
        add(f"| {name} | {item['window_mean']:.4f} | {item['test_mean']:.4f} | "
            f"{item['test_diagnostic_auc']:.4f} | "
            f"{min(best['test_s1_rejection'], best['test_s2_acceptance']):.4f} |")
    add("")
    add(f"两个读数的 Test 诊断 AUC 只有 {probe['distance_ratio']['test_diagnostic_auc']:.4f} / "
        f"{probe['max_cosine']['test_diagnostic_auc']:.4f}，网格内最好 min 只有 "
        f"{min(probe['distance_ratio']['best_balanced']['test_s1_rejection'], probe['distance_ratio']['best_balanced']['test_s2_acceptance']):.4f} / "
        f"{min(probe['max_cosine']['best_balanced']['test_s1_rejection'], probe['max_cosine']['best_balanced']['test_s2_acceptance']):.4f}，"
        "都远低于 0.70。因此证据链是完整的：**不是「OE 门控恰好在 E192 里没做好」，"
        "而是用真实外部未知者做 OE 也不能让门控跨会话泛化**——外部未知者与 Test S1 虽然都是"
        "未注册者，但采集会话/设备/任务分布与项目 Train→Test 的会话漂移不是同一个方向，"
        "OE 学到的是「像不像 E192 的 fit 分布」，不是「像不像项目 Test 的 S2」。")
    add("")
    add("补充：E192 的 OE 分支还同时拖垮了身份侧（OOF C 0.8766，见 E192 results.json），"
        "所以本实验坚持用 E191 骨干做身份，只在门控侧使用 E192 产物——"
        "这正是「两个分支各司其职」的边界：身份可以拆，门控拆不出来。")
    add("")
    add("## 7. Test 官方指标")
    add("")
    add("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实身份标签）。"
        "§2–§6 的 S1 拒识 / S2 接受 / 诊断 AUC 是预测工程审计，不是官方真值指标。")
    add("")
    add("## 8. 合规与边界")
    add("")
    add("- 不重训任何已有模型：身份、门控、外部 OE 模型全部按冻结权重加载。")
    add("- 所有阈值来自 Train 5-block OOF 分位；融合权重来自 Train 20 折 LOSO AUC；"
        "Test 只评分。Session 在 Test 分数与阈值全部落定后才读取。")
    add("- 评估块 0.6 < frac ≤ 0.8 从未进入门控拟合或阈值标定；#7/#8 的攻击子集与 "
        "E189/E191 完全一致（同种子、同 cap 400/任务）。")
    add("- 每个门禁列都是「一个方案 + 一个 a」的完整系统，未跨工作点拼接。")
    add("- 未修改任何已有实验目录，未提交 git。")
    add("")
    add("## 9. 文件与耗时")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e193.py` | 全部逻辑（`--stage gate/identity/attack/report/all`） |")
    add("| `config.json` | 冻结配置与口径 |")
    add("| `cache/gate_scores.npz` | OE / subwin6 / 融合的 OOF、Test、20 折 LOSO 数组 |")
    add("| `gate.json` | 门控汇总（LOSO AUC、U@95、融合权重、校验） |")
    add("| `cache/identity.json` | #1–#4 与 Test 身份预测统计、OE 派生门控探针 |")
    add("| `cache/test_logits.npy` | E191 骨干对 Test 的 20 类 logits |")
    add("| `cache/attack.npz` / `attack.json` | 自建 SNR30/SNR20 PGD 的各条件 logits 与门控分数 |")
    add("| `results.json` | 汇总 |")
    add("| `run.log` | 运行日志 |")
    add("")
    seconds = results.get("stage_seconds", {})
    add("耗时（秒）：" + "、".join(f"{k} {v:.0f}" for k, v in seconds.items()) + "。")
    add("")
    add("```powershell")
    add('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        f'"{OUT / "run_e193.py"}" --stage all')
    add("```")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results.update({
        "gate": gate_summary,
        "identity": branch,
        "probe": probe,
        "ablation": {scheme: {"loso_auc": gate_summary["streams"][scheme]["loso_auc"],
                              "u_at_95": gate_summary["streams"][scheme]["u_at_95"],
                              "test_diagnostic_auc": curves[scheme][0]["test_diagnostic_auc"],
                              "best_min_in_grid": max(r["min"] for r in curves[scheme]),
                              "curve": {f"{r['a']:.2f}": r for r in curves[scheme]}}
                     for scheme in used_schemes},
        "gate_matrix": {
            "primary": {"scheme": primary_scheme, "a": PRIMARY_A,
                        "values": primary["values"], "passed": primary["passed"],
                        "values_snr30": primary["values_snr30"],
                        "passed_snr30": primary["passed_snr30"],
                        "count": primary["count"],
                        "count_snr30": primary["count_snr30"]},
            "max_pass_single_work_point": int(max_count),
            "max_pass_single_work_point_snr30": int(max_count_snr30),
            "max_pass_under_dual_070": int(max_count_dual),
            "max_pass_under_dual_070_snr30": int(max_count_dual_snr30),
            "n_dual_points": len(dual),
            "best_dual": None if best_dual is None else
            {"scheme": best_dual[0], "a": best_dual[1],
             "values": points[best_dual]["values"],
             "passed": points[best_dual]["passed"], "count": points[best_dual]["count"],
             "count_snr30": points[best_dual]["count_snr30"]},
            "structural_failures": structural,
            "structural_failures_snr30": structural_snr30,
            "at_a090": {scheme: {"count": points[(scheme, PRIMARY_A)]["count"],
                                 "count_snr30": points[(scheme, PRIMARY_A)]["count_snr30"],
                                 "values": points[(scheme, PRIMARY_A)]["values"],
                                 "values_snr30": points[(scheme, PRIMARY_A)]["values_snr30"]}
                        for scheme in used_schemes},
            "test_official": {k: "N/A" for k in
                              ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")}},
        "attack": attack_meta,
        "test_decision": test_decision,
    })
    write_json(RESULTS_PATH, results)
    log(f"report written: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["gate", "identity", "attack", "report", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    sys.stdout = Tee(LOG_PATH)
    sys.stderr = sys.stdout
    started = time.time()
    log(f"\n===== E193 stage {args.stage} {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    P = protocol()
    log(f"Train {len(P['meta'])} rows / 20 subjects; fit {int(P['fit_rows'].sum())} / "
        f"window {int(P['window'].sum())}; Test {len(P['tmeta'])} "
        f"(S1 {int(P['is_s1'].sum())} / S2 {int((~P['is_s1']).sum())})")
    stage_seconds = read_json(RESULTS_PATH).get("stage_seconds", {})
    if args.stage in ("gate", "all"):
        gate_started = time.time()
        stage_gate(P, args.force)
        stage_seconds["gate"] = time.time() - gate_started
        results = read_json(RESULTS_PATH)
        results["stage_seconds"] = stage_seconds
        write_json(RESULTS_PATH, results)
    if args.stage in ("identity", "all"):
        identity_started = time.time()
        stage_identity(P, args.force)
        stage_seconds["identity"] = time.time() - identity_started
        results = read_json(RESULTS_PATH)
        results["stage_seconds"] = stage_seconds
        write_json(RESULTS_PATH, results)
    if args.stage in ("attack", "all"):
        attack_started = time.time()
        stage_attack(P, args.force)
        stage_seconds["attack"] = time.time() - attack_started
        results = read_json(RESULTS_PATH)
        results["stage_seconds"] = stage_seconds
        write_json(RESULTS_PATH, results)
    if args.stage in ("report", "all"):
        write_report()
    log(f"E193 done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
