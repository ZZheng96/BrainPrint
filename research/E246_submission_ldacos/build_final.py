"""E246 deploy candidate: LDA-cos (shrink 0.01, no centering) on the submission's frozen space.

Whitening = pooled within-class cov of the 20 Train ids (frac<=0.6) in the frozen PCA-96 space.
Threshold = median over j of the 90% quantile of j's max cosine when j is left out of stats/PCA/
prototypes/whitening (the 'honest' inner LOSO validated in run_e246.py). Test: descriptive only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import run_e246 as R  # noqa: E402

LAM = 0.01
ROOT, DEV, M = R.ROOT, R.DEV, R.M

tm = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
tm["frac"] = tm.trial / tm.groupby(["subject", "task"])["trial"].transform("max")  # = E200 bundle
feats = torch.as_tensor(np.load(ROOT / "BrainPrint_cache/e201_final_selftest/train_raw_scores_inputs.npy"), device=DEV)
tasks = M._task_indices([{"task": t} for t in tm.task], len(tm))
subj, frac = tm.subject.to_numpy(), tm.frac.to_numpy()
art = M.load_model(str(ROOT / "final/team_hymn"), str(DEV))
ids = np.arange(1, 21)
none_i, none_l = np.zeros(0, int), np.zeros(0, int)

space = R.Space(feats, tasks, art=art)
protos, valid, fm, eig = R.build(feats, tasks, subj, frac, space, ids, none_i, none_l)
W = R.whiten(eig, LAM)
# protos must equal submission prototypes (same rows, same space)
diff = float((protos[:, :20] - art["prototypes"][:, :20]).abs().max())

late, rng = frac > 0.8, np.random.default_rng(0)
qs, qe = [], []
for j in ids:
    kj = ids[ids != j]; sj = np.isin(subj, kj)
    sp = R.Space(feats, tasks, stat_rows=sj, pca_rows=sj & (frac <= 0.6))
    pj, vj, fmj, eigj = R.build(feats, tasks, subj, frac, sp, kj, none_i, none_l)
    rows = late
    res = R.score(feats, tasks, rows, sp, pj, vj, fmj, {"c": R.whiten(eigj, LAM), "e": None}, "none", rng)
    isj = subj[np.flatnonzero(rows)] == j
    qs.append(float(np.quantile(res["c"][0][isj], R.Q)))
    qe.append(float(np.quantile(res["e"][0][isj], R.Q)))
th, th_e = float(np.median(qs)), float(np.median(qe))  # th_e = Euclid at matched LOSO operating point

# Test descriptive
tmeta = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
tf = torch.as_tensor(np.load(ROOT / "research/E200_final_freeze/submission_frozen_work/features.npy"), device=DEV)
tt = M._task_indices([{"task": t} for t in tmeta.task], len(tmeta))
allr = np.ones(len(tmeta), bool)
res = R.score(tf, tt, allr, space, protos, valid, fm, {"c": W, "e": None}, "none", rng)
mx, am = res["c"]; emx, eam = res["e"]
sess = tmeta.session.astype(str).to_numpy()
desc = {}
for s in np.unique(sess):
    m = sess == s
    ea = emx[m] >= -art["tau"]; conf = emx[m] >= np.quantile(emx[m], 0.75)
    cnt = np.bincount(am[m][mx[m] >= th], minlength=24)[1:21]; ecnt = np.bincount(eam[m][ea], minlength=24)[1:21]
    desc[s] = {"n": int(m.sum()), "ldacos_accept": float((mx[m] >= th).mean()),
               "euclid_tau_accept": float(ea.mean()),
               "euclid_matched_loso_accept": float((emx[m] >= th_e).mean()),
               "nearest_agree_ldacos_vs_euclid": float((am[m] == eam[m]).mean()),
               "nearest_agree_on_euclid_top25pct_conf": float((am[m][conf] == eam[m][conf]).mean()),
               "accepted_id_count_cv_ldacos": float(cnt.std() / max(cnt.mean(), 1e-9)),
               "accepted_id_count_cv_euclid": float(ecnt.std() / max(ecnt.mean(), 1e-9))}
(OUT / "models").mkdir(exist_ok=True)
torch.save({"whiten": W.cpu(), "threshold": th, "shrink": LAM, "inner_q90": qs}, OUT / "models/ldacos_whiten.pt")
out = {"proto_maxabs_diff_vs_submission": diff, "threshold": th, "inner_q90": qs,
       "euclid_threshold_matched_loso": th_e, "inner_q90_euclid": qe, "test_descriptive": desc}
(OUT / "final_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
