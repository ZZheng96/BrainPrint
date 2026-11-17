"""E249 gate backfill: #1 in-sample acc, #3 macro-F1, #6 U@95 for C0 and C2_a0.5.

Train only. #6 uses the same inner LOSO as run_e249 (fit 19 ids frac<=0.4, calibration
(0.4,0.6] at 95% known acceptance, unknown = held-out id Late).
"""
import json

import numpy as np
import pandas as pd
import torch

import run_e249 as E

R = E.R
CANDS = ["C0", "C2_a0.5"]

meta = pd.read_csv(R.META_PATH)
feats = torch.as_tensor(np.load(R.FEATURE_PATH).astype(np.float32), device=E.DEV)
R.validate(meta, feats)
_, frac = R.temporal_blocks(meta)
subj, tasks = meta.subject.to_numpy(), R.M._task_indices([{"task": t} for t in meta.task], len(meta))

plan = E.outer_plan(subj, frac, 0)
fitted, Sw, Ds, P = E.fit_all(feats, tasks, subj, plan["known"], plan["fit"], plan["d1"], plan["d2"])
T = E.transforms(fitted, Sw, Ds)
early, late = plan["fit"], plan["late"]

out = {}
for c in CANDS:
    # #1 in-sample: 20-class closed-set accuracy on the fit block
    ix = np.flatnonzero(early)
    mx, who = E.all_task_scores(P, ix, T[(c, "all")], fitted)
    pred = who[np.arange(len(ix)), tasks[ix]]
    ins = float((pred == subj[ix]).mean())
    # #3 macro-F1 on the Late block (closed set, no rejection)
    ix = np.flatnonzero(late)
    mx, who = E.all_task_scores(P, ix, T[(c, "all")], fitted)
    pred, true = who[np.arange(len(ix)), tasks[ix]], subj[ix]
    f1s = []
    for s in range(1, 21):
        tp = int(((pred == s) & (true == s)).sum())
        fp = int(((pred == s) & (true != s)).sum())
        fn = int(((pred != s) & (true == s)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    out[c] = {"in_sample_acc": ins, "macro_f1_late": float(np.mean(f1s))}

uq = {c: [] for c in CANDS}
for inn in plan["inner"]:
    f2, Sw2, Ds2, P2 = E.fit_all(feats, tasks, subj, inn["remain"], inn["fit"], inn["d1"], inn["d2"])
    T2 = E.transforms(f2, Sw2, Ds2)
    sk = E.score_table(feats, tasks, P2, inn["cal_known"], f2, T2)
    su = E.score_table(feats, tasks, P2, inn["cal_unknown"], f2, T2)
    for c in CANDS:
        thr = float(np.quantile(sk[(c, "P1")][0], 0.95))
        uq[c].append(float((su[(c, "P1")][0] < thr).mean()))
for c in CANDS:
    out[c]["U_at_95"] = float(np.mean(uq[c]))
    out[c]["U_at_95_min"] = float(np.min(uq[c]))
print(json.dumps(out, indent=2))
(E.OUT / "results" / "gates_train.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
