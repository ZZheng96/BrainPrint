"""E249 one-shot Test audit (descriptive, discloses 'looked at Test').

Deployment = all 20 Train ids, Early (frac<=0.6) fit; D from frac<=0.3 vs (0.3,0.6] + cross-task.
Threshold + C2 alpha via the same inner LOSO as run_e249 (20 held-out folds, frac<=0.4 fit,
(0.4,0.6] calibration, P1 weighted FAR10). Test is scored once for C0 and the pre-selected C2.
Session is used ONLY to group the descriptive report, never in any decision.
"""
import json

import numpy as np
import pandas as pd
import torch

import run_e249 as E

R, DEV = E.R, E.DEV
ROOT, OUT = E.ROOT, E.OUT
CANDS = ["C0"] + [f"C2_a{a}" for a in E.ALPHAS]

meta = pd.read_csv(R.META_PATH)
feats = torch.as_tensor(np.load(R.FEATURE_PATH).astype(np.float32), device=DEV)
R.validate(meta, feats)
_, frac = R.temporal_blocks(meta)
subj = meta.subject.to_numpy()
tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))

plan = E.outer_plan(subj, frac, 0)  # outer=0 -> all 20 ids known
assert len(plan["known"]) == 20 and len(plan["inner"]) == 20
unk = {c: [] for c in CANDS}
kn = {c: [] for c in CANDS}
sel = {c: {e[0]: ([], []) for e in E.EPS} for c in CANDS}
for inn in plan["inner"]:
    fitted, Sw, Ds, P = E.fit_all(feats, tasks, subj, inn["remain"], inn["fit"], inn["d1"], inn["d2"])
    T = E.transforms(fitted, Sw, Ds)
    su = E.score_table(feats, tasks, P, inn["cal_unknown"], fitted, T)
    sk = E.score_table(feats, tasks, P, inn["cal_known"], fitted, T)
    for c in CANDS:
        for e in E.EPS:
            sel[c][e[0]][0].append(su[(c, e[0])][0])
            sel[c][e[0]][1].append(sk[(c, e[0])][0])
thr, sel_score = {}, {}
for c in CANDS:
    ka = {}
    for e in E.EPS:
        u, k = sel[c][e[0]]
        t = R.weighted_far_threshold(u, E.FAR)
        ka[e[0]] = float(np.mean([(v >= t).mean() for v in k]))
        if e[0] == "P1":
            thr[c] = t
    sel_score[c] = float(np.mean([ka[e] for e in E.P2_EPS]))
c2 = [c for c in CANDS if c.startswith("C2_")]
chosen = c2[int(np.argmax([sel_score[c] for c in c2]))]

fitted, Sw, Ds, _ = E.fit_all(feats, tasks, subj, plan["known"], plan["fit"], plan["d1"], plan["d2"])
T = E.transforms(fitted, Sw, Ds)
torch.save({"fitted": R.cpu_model(fitted), "Sw": Sw.cpu(), "D": {k: v.cpu() for k, v in Ds.items()},
            "thresholds": thr, "chosen": chosen, "inner_selection_score": sel_score},
           OUT / "results" / "deploy_model.pt")

tmeta = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
tf = torch.as_tensor(np.load(ROOT / "research/E200_final_freeze/submission_frozen_work/features.npy").astype(np.float32), device=DEV)
assert tf.shape == (len(tmeta), feats.shape[1])
tt = R.M._task_indices([{"task": t} for t in tmeta.task], len(tmeta))
TP = E.project_all(fitted["space"], tf, tt)
ar = np.arange(len(tmeta))
sess = tmeta.session.astype(str).to_numpy()
out = {"disclosure": "looked at Test (one-shot descriptive audit, no iteration, no selection by Test)",
       "chosen_C2": chosen, "inner_selection_score": sel_score, "thresholds_P1_FAR10": thr, "test": {}}
for c in ["C0", chosen]:
    mx, who = E.all_task_scores(TP, ar, T[(c, "all")], fitted)
    s, w = mx[ar, tt], who[ar, tt]
    acc = s >= thr[c]
    res = {}
    for g in np.unique(sess):
        m = sess == g
        cnt = np.bincount(w[m][acc[m]], minlength=21)[1:]
        res[g] = {"n": int(m.sum()), "accept": float(acc[m].mean()),
                  "score_median": float(np.median(s[m])),
                  "accepted_id_count_cv": float(cnt.std() / max(cnt.mean(), 1e-9))}
    # session-free ranking diagnostic: how well the score separates the two groups (descriptive only)
    res["S1_vs_S2_score_AUC_descriptive"] = R.auc(s[sess == "session2"], s[sess == "session1"])
    out["test"][c] = res
(OUT / "results" / "test_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
