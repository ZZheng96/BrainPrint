"""One-shot descriptive Test S1/S2 audit for the frozen E251 Train-only deployment."""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E247_train_temporal_loso"))
import run_e247 as R
import run_e251 as E

DEV = R.DEV
meta = pd.read_csv(R.META_PATH)
feats = torch.as_tensor(np.array(np.load(R.FEATURE_PATH, mmap_mode="r"), dtype=np.float32, copy=True), device=DEV)
R.validate(meta, feats)
_, frac = R.temporal_blocks(meta)
subj = meta.subject.to_numpy()
tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))
known = np.arange(1, 21)
fit = np.isin(subj, known) & (frac <= .6)
inner = []
for held in known:
    remain = known[known != held]
    ifit = np.isin(subj, remain) & (frac <= .4)
    fitted, P, W = E.fit_base(feats, tasks, subj, ifit, remain)
    iu, _ = E.score_features(P, np.flatnonzero((subj == held) & (frac > .4) & (frac <= .6)), fitted, W, tasks)
    ik, _ = E.score_features(P, np.flatnonzero(np.isin(subj, remain) & (frac > .4) & (frac <= .6)), fitted, W, tasks)
    inner.append({"unknown_raw": iu, "known_raw": ik})
stats = E.candidate_stats(inner, subj, 0)
chosen = E.choose_lambda(stats)
fitted, P, W = E.fit_base(feats, tasks, subj, fit, known)
tmeta = pd.read_csv(ROOT / "BrainPrint_cache" / "test_metadata.csv")
tf = torch.as_tensor(np.load(ROOT / "research" / "E200_final_freeze" / "submission_frozen_work" / "features.npy").astype(np.float32), device=DEV)
assert tf.shape == (len(tmeta), feats.shape[1])
tt = R.M._task_indices([{"task": t} for t in tmeta.task], len(tmeta))
TP = E.project_all(fitted["space"], tf, tt)
ix = np.arange(len(tmeta)); raw, pred = E.score_features(TP, ix, fitted, W, tt)
score = E.reject_score(raw, chosen); threshold = stats[chosen]["threshold"]
sess = tmeta.session.astype(str).str.upper().str.replace("SESSION", "S", regex=False).to_numpy()
out = {"disclosure":"looked at Test once; descriptive only; no Test fitting, selection, or threshold changes","frozen_candidate":f"lambda_{chosen:.2f}","lambda":chosen,"threshold_P1_FAR10":threshold,"train_inner_stats":stats,"test":{},"official_test_AUC":"N/A (no identity truth)","official_test_F1":"N/A (no identity truth)","session_groups":["S1","S2"]}
for group in ("S1", "S2"):
    m = sess == group
    cnt = np.bincount(pred[m][score[m] >= threshold], minlength=21)[1:]
    out["test"][group] = {"n":int(m.sum()),"acceptance":float((score[m]>=threshold).mean()),"rejection":float((score[m]<threshold).mean()),"score_median":float(np.median(score[m])),"accepted_id_count_cv":float(cnt.std()/max(cnt.mean(),1e-12))}
for c in ("lambda_0.00", f"lambda_{chosen:.2f}"):
    pass
s1, s2 = score[sess == "S1"], score[sess == "S2"]
out["test"]["S1_vs_S2_score_AUC_descriptive"] = R.auc(s2, s1)
(OUT / "results" / "test_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
