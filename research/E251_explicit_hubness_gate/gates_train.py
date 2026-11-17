"""E251 gates #1-#4 and #6 under the E195 definitions, lambda = 0 (selected model)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E247_train_temporal_loso"))
sys.path.insert(0, str(OUT))
import run_e247 as R  # noqa: E402
import run_e251 as E  # noqa: E402

LAM = 0.0


def closed_set(feats, tasks, subj, fit_rows, eval_rows, known):
    fitted, P, W = E.fit_base(feats, tasks, subj, fit_rows, known)
    ix = np.flatnonzero(eval_rows)
    raw, pred = E.score_features(P, ix, fitted, W, tasks)
    return pred, subj[ix]


def macro_f1(pred, true, n=20):
    scores = []
    for s in range(1, n + 1):
        tp = int(((pred == s) & (true == s)).sum())
        fp = int(((pred == s) & (true != s)).sum())
        fn = int(((pred != s) & (true == s)).sum())
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def main():
    meta = pd.read_csv(R.META_PATH)
    feats = torch.as_tensor(np.array(np.load(R.FEATURE_PATH, mmap_mode="r"), dtype=np.float32, copy=True), device=R.DEV)
    R.validate(meta, feats)
    _, frac = R.temporal_blocks(meta)
    subj = meta.subject.to_numpy()
    tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))
    known = np.arange(1, 21)
    fit_rows = np.isin(subj, known) & (frac <= .6)
    blocks = np.clip((frac * 5).astype(int), 0, 4)

    out = {}
    # #1 in-sample, #4 forward block-out (same fit, different evaluation rows)
    pred, true = closed_set(feats, tasks, subj, fit_rows, fit_rows, known)
    out["1_in_sample_accuracy"] = float((pred == true).mean())
    late = frac > .6
    pred4, true4 = closed_set(feats, tasks, subj, fit_rows, late, known)
    out["4_forward_block_out_accuracy"] = float((pred4 == true4).mean())

    # #2 / #3 five-block OOF closed-set
    oof_pred, oof_true = [], []
    per_block = {}
    for b in range(5):
        train_rows = blocks != b
        eval_rows = blocks == b
        pred, true = closed_set(feats, tasks, subj, train_rows, eval_rows, known)
        per_block[str(b)] = float((pred == true).mean())
        oof_pred.append(pred); oof_true.append(true)
    oof_pred, oof_true = np.concatenate(oof_pred), np.concatenate(oof_true)
    out["2_oof_5block_closed_set_accuracy"] = float((oof_pred == oof_true).mean())
    out["3_oof_5block_macro_f1"] = macro_f1(oof_pred, oof_true)
    out["2_per_block_accuracy"] = per_block

    # #6 U@95%: inner identity LOSO, threshold = 95th pct of calibration known scores
    us = []
    for held in known:
        remain = known[known != held]
        ifit = np.isin(subj, remain) & (frac <= .4)
        fitted, P, W = E.fit_base(feats, tasks, subj, ifit, remain)
        raw_k, _ = E.score_features(P, np.flatnonzero(np.isin(subj, remain) & (frac > .4) & (frac <= .6)), fitted, W, tasks)
        raw_u, _ = E.score_features(P, np.flatnonzero((subj == held) & (frac > .4) & (frac <= .6)), fitted, W, tasks)
        thr = float(np.quantile(raw_k[:, 0], 0.95))
        us.append(float((raw_u[:, 0] < thr).mean()))
    out["6_U_at_95_strict"] = float(np.mean(us))
    out["6_U_at_95_min"] = float(np.min(us))
    out["lambda"] = LAM
    out["protocol"] = {
        "fit": "all 20 identities, frac <= 0.6",
        "blocks": "block = clip(int(frac*5), 0, 4) per (subject, task) trial order",
        "1": "closed-set top-1 on the fit block",
        "4": "closed-set top-1 on frac > 0.6 with the frac <= 0.6 fit",
        "2_3": "leave-one-block-out, statistics and prototypes fitted on the other four blocks",
        "6": "inner LOSO: fit on 19 identities frac <= 0.4, threshold = quantile(known 0.4 < frac <= 0.6, 0.95), unknown = held identity, same window",
        "score": "strict LDA-cos top-1 cosine (lambda = 0)",
        "test_data": "not read",
    }
    (OUT / "results" / "gates_train.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
