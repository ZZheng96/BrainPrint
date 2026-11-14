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
import run_e54 as E54

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
E95 = ROOT / "research" / "E95_e54_final20_candidate"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def window_batch(x, windows):
    # E54's fixed-shape preprocessing is retained; each fixed crop is resampled to 1200.
    chunks = []
    for a, b in windows:
        crop = x[..., a:b]
        chunks.append(torch.nn.functional.interpolate(crop, size=1200, mode="linear", align_corners=False))
    return torch.stack(chunks, dim=1)


def mv_scores(model, raw, windows, batch=128):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            xb = window_batch(raw[st:st + batch], windows)
            embs = [model.embed(xb[:, j]) for j in range(xb.shape[1])]
            emb = torch.nn.functional.normalize(torch.stack(embs, 1).mean(1), dim=1)
            out.append((emb @ model.normalized_prototypes().T).cpu().numpy())
    return np.concatenate(out, axis=0)


def single_scores(model, raw, batch=128):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.scores(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def choose_threshold(known, unknown, labels):
    km = known.max(1); um = unknown.max(1)
    best = None
    for q in CFG["calibration"]["grid_quantiles"]:
        th = float(np.quantile(km, q)); kp = np.where(km >= th, CLASSES[known.argmax(1)], 0)
        acc = float(np.mean(kp != 0)); rej = float(np.mean(um < th))
        if acc + 1e-12 < CFG["calibration"]["known_acceptance_target"]: continue
        cand = (rej + .1 * float(np.mean(kp == labels)), th, acc, rej)
        if best is None or cand > best: best = cand
    if best is None:
        th = float(np.quantile(km, .05)); best = (0., th, float(np.mean(km >= th)), float(np.mean(um < th)))
    _, th, acc, rej = best
    return {"score": "max_cosine", "threshold": th, "known_acceptance": acc, "unknown_rejection": rej,
            "selection_source": "Train-only pseudo-unknown time calibration; no Test access"}


def apply(scores, params):
    m = scores.max(1)
    return np.where(m >= params["threshold"], CLASSES[scores.argmax(1)], 0), m


def load_test(model, params):
    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"], usecols=["sample", "path", "session"])
    labels = []
    root = ROOT / CFG["data"]["train_root"]
    for st in range(0, len(mapping), 128):
        paths = [root / p for p in mapping.path.iloc[st:st + 128]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"], dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        labels.extend(apply(mv_scores(model, x, CFG["windows"]), params)[0].tolist())
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(OUT / "predictions.csv", index=False)
    return mapping, np.asarray(labels, dtype=int)


def run(smoke=False):
    if not torch.cuda.is_available(): raise RuntimeError("E97 requires CUDA")
    e54cfg = E54.load_config(); meta = E54.train_metadata(e54cfg)
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    ck = torch.load(E95 / "model_checkpoint.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    f = frac(meta); subjects = meta.subject.to_numpy(int)
    pseudo = np.asarray(CFG["calibration"]["pseudo_unknown_subjects"])
    known = np.setdiff1d(CLASSES, pseudo)
    cal_known = np.flatnonzero(np.isin(subjects, known) & (f > .6) & (f <= .8))
    cal_unknown = np.flatnonzero(np.isin(subjects, pseudo) & (f > .6) & (f <= .8))
    s_mv = mv_scores(model, raw, CFG["windows"])
    s_single = single_scores(model, raw)
    params = choose_threshold(s_mv[cal_known], s_mv[cal_unknown], subjects[cal_known])
    pred, conf = apply(s_mv[cal_known], params)
    y_known = np.ones(len(cal_known)); y_unknown = np.zeros(len(cal_unknown))
    auc = float(roc_auc_score(np.r_[y_known, y_unknown], np.r_[conf, s_mv[cal_unknown].max(1)]))
    f1 = float(f1_score(subjects[cal_known], pred, labels=known, average="macro", zero_division=0))
    sm_params = choose_threshold(s_single[cal_known], s_single[cal_unknown], subjects[cal_known])
    sm_pred, sm_conf = apply(s_single[cal_known], sm_params)
    single_auc = float(roc_auc_score(np.r_[y_known, y_unknown], np.r_[sm_conf, s_single[cal_unknown].max(1)]))
    single_f1 = float(f1_score(subjects[cal_known], sm_pred, labels=known, average="macro", zero_division=0))
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "windows": CFG["windows"], "threshold": params,
                "source_checkpoint": str(E95 / "model_checkpoint.pt"), "config": CFG}, OUT / "model_checkpoint.pt")
    result = {"status": "complete_train_only" if smoke else "frozen_before_test", "experiment": CFG["experiment"],
              "train_only_metrics": {"AUC": auc, "F1": f1, "N_known": len(cal_known), "N_pseudo_unknown": len(cal_unknown), "threshold": params},
              "window_ablation": {"single": {"AUC": single_auc, "F1": single_f1, "threshold": sm_params}, "multiview": {"AUC": auc, "F1": f1}},
              "test_metrics": {k: "N/A" for k in ["AUC", "F1", "C", "RA", "ASafeR", "U", "Official Score"]},
              "protocol_checks": {"train_session1_only": True, "test_accessed": False, "test_read_once_after_freeze": False,
                                  "e95_checkpoint_reused": True, "gpu": torch.cuda.get_device_name(0), "windows": CFG["windows"]}}
    if not smoke:
        mapping, labels = load_test(model, params)
        result["audit"] = {"N": int(len(labels)), "predictions_complete_before_session_audit": True, "session_used_by_inference": False}
        sess = mapping.session.astype(str).str.casefold()
        for name, aliases in (("S1", ("session1", "s1", "1")), ("S2", ("session2", "s2", "2"))):
            m = sess.isin(aliases).to_numpy(); result["audit"][name + "_N"] = int(m.sum())
            result["audit"][name + "_rejection_rate"] = float(np.mean(labels[m] == 0)); result["audit"][name + "_acceptance_rate"] = float(np.mean(labels[m] != 0))
        result["protocol_checks"].update({"test_accessed": True, "test_read_once_after_freeze": True})
        result["status"] = "complete_one_shot_frozen_audit"
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = result.get("audit", {})
    report = f"# E97 multiview session invariant\n\n## Train-only calibration\nAUC={auc:.6f}，F1={f1:.6f}；阈值={params['threshold']:.6f}，known acceptance={params['known_acceptance']:.6f}，pseudo-unknown rejection={params['unknown_rejection']:.6f}。\n\n## Window ablation\nSingle AUC={single_auc:.6f}，F1={single_f1:.6f}；Multiview AUC={auc:.6f}，F1={f1:.6f}。\n\n## Test\nAUC/F1/C/RA/ASafeR/U/Official Score=N/A。S1拒识率={audit.get('S1_rejection_rate','N/A')}；S2接受率={audit.get('S2_acceptance_rate','N/A')}。predictions.csv完整落盘后才读取session审计。\n"
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true"); a = ap.parse_args()
    if a.smoke == a.full: ap.error("choose --smoke or --full")
    run(a.smoke)
