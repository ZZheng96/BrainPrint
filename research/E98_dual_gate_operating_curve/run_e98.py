"""E98: Train-only dual-gate operating curve for the 18-channel project Test.

Reuses the frozen E95 raw identity encoder and the E65 correlation-logSPD
prototypes. Both branches are scored on Train/session1 only; the accept rule is
chosen on Train-only known/calibration blocks. The project Test is read once,
after the rule is frozen, and Session is only used for the post-hoc audit.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e54 as E54
import run_e65
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
E95 = ROOT / "research" / "E95_e54_final20_candidate"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def raw_scores(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.scores(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def spd_scores(proto, feat, batch=1024):
    p = torch.as_tensor(proto, dtype=torch.float32, device=DEVICE)
    out = []
    for st in range(0, len(feat), batch):
        chunk = feat[st:st + batch]
        if isinstance(chunk, torch.Tensor):
            q = chunk.detach().to(device=DEVICE, dtype=torch.float32)
        else:
            q = torch.as_tensor(np.array(chunk, dtype=np.float32), device=DEVICE)
        out.append((-torch.cdist(q, p)).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_spd_prototypes(meta, feat, fit_mask):
    """Prototypes only for identities present in the fit block (pseudo-unknown excluded)."""
    subjects = meta.subject.to_numpy(int)
    rows = np.flatnonzero(fit_mask)
    present = np.unique(subjects[rows])
    proto = np.stack([
        np.asarray(feat[rows[subjects[rows] == c]], dtype=np.float32).mean(0)
        for c in present
    ])
    if not np.isfinite(proto).all():
        raise ValueError("non-finite SPD prototype")
    return present, proto


def choose_rule(known_raw, known_spd, unknown_raw, unknown_spd, known_labels):
    """Grid over raw threshold quantile and SPD gate quantile, Train-only."""
    kr, ks = known_raw.max(1), known_spd.max(1)
    ur, us = unknown_raw.max(1), unknown_spd.max(1)
    best = None
    for qr in CFG["rule"]["raw_quantiles"]:
        tr = float(np.quantile(kr, qr))
        for qs in CFG["rule"]["spd_quantiles"]:
            ts = float(np.quantile(ks, qs))
            accept_k = (kr >= tr) & (ks >= ts)
            accept_u = (ur >= tr) & (us >= ts)
            acc = float(np.mean(accept_k))
            if acc + 1e-12 < CFG["rule"]["known_acceptance_target"]:
                continue
            rej = float(np.mean(~accept_u))
            ident = float(np.mean(CLASSES[known_raw.argmax(1)][accept_k] == known_labels[accept_k])) if accept_k.any() else 0.0
            value = rej + 0.1 * ident + 0.01 * acc
            cand = (value, tr, ts, acc, rej, ident)
            if best is None or cand > best:
                best = cand
    if best is None:
        tr = float(np.quantile(kr, .05)); ts = float(np.quantile(ks, .05))
        best = (0.0, tr, ts, float(np.mean(kr >= tr)), float(np.mean(ur < tr)), 0.0)
    _, tr, ts, acc, rej, ident = best
    return {"raw_threshold": tr, "spd_threshold": ts, "known_acceptance": acc,
            "unknown_rejection": rej, "identity_accuracy": ident,
            "selection_source": "Train-only known/calibration blocks; no Test access"}


def apply_rule(raw_s, spd_s, rule):
    accept = (raw_s.max(1) >= rule["raw_threshold"]) & (spd_s.max(1) >= rule["spd_threshold"])
    return np.where(accept, CLASSES[raw_s.argmax(1)], 0)


def load_test(model, proto, rule):
    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"], usecols=["sample", "path", "session"])
    if len(order) != CFG["data"]["expected_test_rows"] or len(mapping) != len(order):
        raise ValueError("Test mapping mismatch")
    root = ROOT / CFG["data"]["train_root"]
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    labels = []
    for st in range(0, len(mapping), 64):
        paths = [root / p for p in mapping.path.iloc[st:st + 64]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"], dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            feat = run_e65.gpu_correlation_logspd(x, e36, e65cfg)
        labels.extend(apply_rule(raw_scores(model, x), spd_scores(proto, feat), rule).tolist())
    if len(labels) != len(order):
        raise ValueError("prediction count mismatch")
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(OUT / "predictions.csv", index=False)
    return mapping, np.asarray(labels, dtype=int)


def run(smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError("E98 requires CUDA")
    if not smoke and (OUT / "predictions.csv").exists():
        raise RuntimeError("E98 Test audit already performed; refuse rerun")
    e54cfg = E54.load_config()
    meta = E54.train_metadata(e54cfg)
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != 11304 or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    if not np.isfinite(feat).all():
        raise ValueError("Train features contain non-finite values")
    model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    ck = torch.load(E95 / "model_checkpoint.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    f = frac(meta); subjects = meta.subject.to_numpy(int)
    pseudo = np.asarray(CFG["calibration"]["pseudo_unknown_subjects"])
    known = np.setdiff1d(CLASSES, pseudo)
    fit = np.isin(subjects, known) & (f <= .6)
    cal_known = np.flatnonzero(np.isin(subjects, known) & (f > .6) & (f <= .8))
    cal_unknown = np.flatnonzero(np.isin(subjects, pseudo) & (f > .6) & (f <= .8))
    proto_classes, proto = build_spd_prototypes(meta, feat, fit)
    raw_all = raw_scores(model, raw)
    spd_all = spd_scores(proto, feat)
    rule = choose_rule(raw_all[cal_known], spd_all[cal_known], raw_all[cal_unknown], spd_all[cal_unknown], subjects[cal_known])
    pred = apply_rule(raw_all[cal_known], spd_all[cal_known], rule)
    # Continuous confidence for AUC: min of the two branches' rank-normalized scores.
    def rank_conf(known_s, unknown_s):
        both = np.r_[known_s, unknown_s]
        order = both.argsort().argsort().astype(float) / max(len(both) - 1, 1)
        return order[:len(known_s)], order[len(known_s):]
    rc, rcu = rank_conf(raw_all[cal_known].max(1), raw_all[cal_unknown].max(1))
    sc, scu = rank_conf(spd_all[cal_known].max(1), spd_all[cal_unknown].max(1))
    conf, conf_u = np.minimum(rc, sc), np.minimum(rcu, scu)
    auc = float(roc_auc_score(np.r_[np.ones(len(cal_known)), np.zeros(len(cal_unknown))], np.r_[conf, conf_u]))
    f1 = float(f1_score(subjects[cal_known], pred, labels=known, average="macro", zero_division=0))
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "spd_prototypes": proto,
                "spd_prototype_classes": proto_classes, "rule": rule,
                "source_checkpoint": str(E95 / "model_checkpoint.pt"), "config": CFG}, OUT / "model_checkpoint.pt")
    result = {"status": "complete_train_only" if smoke else "frozen_before_test", "experiment": CFG["experiment"],
              "train_only_metrics": {"AUC": auc, "F1": f1, "N_known": int(len(cal_known)),
                                     "N_pseudo_unknown": int(len(cal_unknown)), "rule": rule},
              "test_metrics": {k: "N/A" for k in ["AUC", "F1", "C", "RA", "ASafeR", "U", "Official Score"]},
              "protocol_checks": {"train_session1_only": True, "test_accessed": False,
                                  "e95_checkpoint_reused": True, "gpu": torch.cuda.get_device_name(0)}}
    if not smoke:
        mapping, labels = load_test(model, proto, rule)
        result["audit"] = {"N": int(len(labels)), "predictions_complete_before_session_audit": True,
                           "session_used_by_inference": False}
        sess = mapping.session.astype(str).str.casefold()
        for name, aliases in (("S1", ("session1", "s1", "1")), ("S2", ("session2", "s2", "2"))):
            m = sess.isin(aliases).to_numpy()
            result["audit"][name + "_N"] = int(m.sum())
            result["audit"][name + "_rejection_rate"] = float(np.mean(labels[m] == 0))
            result["audit"][name + "_acceptance_rate"] = float(np.mean(labels[m] != 0))
        result["protocol_checks"].update({"test_accessed": True, "test_read_once_after_freeze": True})
        result["status"] = "complete_one_shot_frozen_audit"
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = result.get("audit", {})
    (OUT / "report.md").write_text(
        "# E98 dual-gate operating curve\n\n"
        "复用冻结 E95 raw identity encoder 与 E65 correlation-logSPD 原型；接受规则仅由 Train/session1 已知/校准块选择。\n\n"
        f"## Train-only\n\nAUC={auc:.6f}，F1={f1:.6f}；raw 阈值={rule['raw_threshold']:.6f}，SPD 阈值={rule['spd_threshold']:.6f}，"
        f"known acceptance={rule['known_acceptance']:.6f}，pseudo-unknown rejection={rule['unknown_rejection']:.6f}。\n\n"
        "## Test\n\n真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n"
        f"S1 拒识率={audit.get('S1_rejection_rate', 'N/A')}；S2 接受率={audit.get('S2_acceptance_rate', 'N/A')}。"
        "predictions.csv 完整落盘后才读取 session 审计。\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.smoke == a.full: ap.error("choose --smoke or --full")
    run(a.smoke)
