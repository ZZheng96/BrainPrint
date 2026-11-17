"""One-shot descriptive Test S1/S2 audit for the pre-selected E250 candidate.

All fitting, calibration, thresholding, and candidate selection use Train only.
Test is read once for descriptive reporting; session is never used for decisions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
import run_e250 as E  # noqa: E402

R, DEV = E.R, E.DEV


def train_only_deployment(feats, tasks, subj, frac):
    plan = E.outer_plan(subj, frac, 0)
    E.validate_plan(subj, frac, 0, plan)
    inner = []
    for inn in plan["inner"]:
        fitted, P, W = E.fit_base(feats, tasks, subj, inn["fit"], inn["remain"])
        iu, _ = E.score_features(P, np.flatnonzero(inn["cal_unknown"]), fitted, W, tasks)
        ik, _ = E.score_features(P, np.flatnonzero(inn["cal_known"]), fitted, W, tasks)
        inner.append({c: {"unknown": iu, "known": ik} for c in E.CFGS})
    models, thresholds, selection = E.fit_calibrators(inner)
    scores = {c: float(selection[c]["inner_known_acceptance"]) for c in E.CFGS}
    c2 = "C2_margin_cohort_hubness"
    chosen = c2 if scores[c2] >= scores["C0"] else "C0"
    fitted, P, W = E.fit_base(feats, tasks, subj, plan["fit"], plan["known"])
    return plan, fitted, P, W, models, thresholds, selection, chosen


def main():
    meta = pd.read_csv(R.META_PATH)
    feats = torch.as_tensor(np.array(np.load(R.FEATURE_PATH, mmap_mode="r"), dtype=np.float32, copy=True), device=DEV)
    R.validate(meta, feats)
    _, frac = R.temporal_blocks(meta)
    subj = meta.subject.to_numpy()
    tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))
    plan, fitted, _, W, models, thresholds, selection, chosen = train_only_deployment(feats, tasks, subj, frac)

    tmeta = pd.read_csv(ROOT / "BrainPrint_cache" / "test_metadata.csv")
    tf = torch.as_tensor(np.load(ROOT / "research" / "E200_final_freeze" / "submission_frozen_work" / "features.npy").astype(np.float32), device=DEV)
    assert tf.shape[0] == len(tmeta) and tf.shape[1] == feats.shape[1]
    tt = R.M._task_indices([{"task": t} for t in tmeta.task], len(tmeta))
    TP = E.project_all(fitted["space"], tf, tt)
    ix = np.arange(len(tmeta))
    raw, pred = E.score_features(TP, ix, fitted, W, tt)
    sess = tmeta.session.astype(str).to_numpy()
    out = {
        "disclosure": "looked at Test (one-shot descriptive audit; no Test selection or fitting)",
        "selection_rule": "choose C2 if its Train inner known acceptance >= C0; otherwise C0",
        "chosen": chosen,
        "train_inner_known_acceptance": selection,
        "thresholds_P1_FAR10": thresholds,
        "test": {},
        "official_test_AUC": "N/A (no identity truth)",
        "official_test_F1": "N/A (no identity truth)",
        "attack_audit": "N/A; no attack rescore was run",
    }
    for c in ["C0", chosen]:
        score = E.apply_model(models[c], raw)
        accepted = score >= thresholds[c]
        by_session = {}
        for g in sorted(np.unique(sess)):
            m = sess == g
            cnt = np.bincount(pred[m][accepted[m]], minlength=21)[1:]
            by_session[g] = {
                "n": int(m.sum()),
                "acceptance": float(accepted[m].mean()),
                "rejection": float((~accepted[m]).mean()),
                "score_median": float(np.median(score[m])),
                "accepted_id_count_cv": float(cnt.std() / max(cnt.mean(), 1e-12)),
            }
        s1, s2 = score[sess == "session1"], score[sess == "session2"]
        by_session["S1_vs_S2_score_AUC_descriptive"] = R.auc(s2, s1)
        out["test"][c] = by_session
    (OUT / "results" / "test_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
