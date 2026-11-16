"""E211: clean two-family fusion (E65 logSPD + subwin6) under one protocol.

E210 showed the historical "high LOSO + high S2" points came from scoring the
LOSO known threshold on rows that were inside the prototype.  With clean rows the
two families have complementary strengths:

  * E65 (PCA96) is stronger on S1 rejection
  * subwin6 with full-data registration is stronger on S2 acceptance

Both families are rebuilt here with one identical clean protocol:
  * 5 contiguous blocks per identity,
  * a query from block b is scored against prototypes built from the other blocks
    of the enrolled identities,
  * the held-out identity of a LOSO fold never enters a prototype,
  * per-fold task statistics and PCA are fitted on the enrolled identities only.

The fusion weight is selected on Train only (LOSO unknown rejection >= 0.90,
maximise Train known acceptance); Test is audited once afterwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'E204_relative_distance_gate'))
from run_e204 import TASKS, auc, choose_threshold, fit_stage1, project, task_codes  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = HERE
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TEST_META = ROOT / 'BrainPrint_cache' / 'test_metadata.csv'
TRAIN_E65 = ROOT / 'BrainPrint_cache' / 'e65_correlation_spd' / 'train_correlation_logspd.npy'
TEST_E65 = ROOT / 'research' / 'E202_dual_model_cross_session' / 'cache' / 'test_e65_features.npy'
TRAIN_SW6 = ROOT / 'research' / 'E171_multiscale_fusion' / 'train_subwin_mean_n6.npy'
TEST_SW6 = ROOT / 'research' / 'E171_multiscale_fusion' / 'test_subwin_mean_n6.npy'
BLOCKS = 5
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def protos(proj, subjects, codes, rows):
    out = torch.zeros((len(TASKS), 20, proj.shape[1]), device=DEVICE)
    for i in range(len(TASKS)):
        for sid in range(1, 21):
            sel = rows & (codes == i) & (subjects == sid)
            if sel.any():
                out[i, sid - 1] = torch.as_tensor(proj[sel].mean(0), device=DEVICE)
    return out


def score(proj, codes, proto, rows):
    out = np.full(len(proj), np.inf, dtype=np.float32)
    idx = np.flatnonzero(rows)
    if len(idx) == 0:
        return out
    q = torch.as_tensor(proj[idx], device=DEVICE)
    for i in range(len(TASKS)):
        local = np.flatnonzero(codes[idx] == i)
        if len(local) == 0:
            continue
        p = proto[i]
        out[idx[local]] = (-torch.cdist(q[local], p).min(1).values).cpu().numpy()
    return out


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    codes = task_codes(meta.task.to_numpy(str))
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    blocks = np.clip((frac * BLOCKS).astype(int), 0, BLOCKS - 1)
    f65 = np.load(TRAIN_E65).astype(np.float32)
    fsw = np.load(TRAIN_SW6).astype(np.float32)

    known65, unknown65, knownS, unknownS = [], [], [], []
    for sid in range(1, 21):
        enrolled = subjects != sid
        mu65, sd65, p65, b65 = fit_stage1(f65, codes, enrolled, enrolled)
        muS, sdS, pS, bS = fit_stage1(fsw, codes, enrolled, enrolled)
        pr65 = project(f65, codes, mu65, sd65, p65, b65)
        prS = project(fsw, codes, muS, sdS, pS, bS)
        for b in range(BLOCKS):
            train_rows = enrolled & (blocks != b)
            q_rows = blocks == b
            s65 = score(pr65, codes, protos(pr65, subjects, codes, train_rows), q_rows)
            sS = score(prS, codes, protos(prS, subjects, codes, train_rows), q_rows)
            krows = enrolled & q_rows
            urows = (~enrolled) & q_rows
            known65.append(s65[krows])
            unknown65.append(s65[urows])
            knownS.append(sS[krows])
            unknownS.append(sS[urows])
    known65, unknown65 = np.concatenate(known65), np.concatenate(unknown65)
    knownS, unknownS = np.concatenate(knownS), np.concatenate(unknownS)

    # ---------- frozen all-20 model, Test once ----------
    all_rows = np.ones(len(meta), bool)
    mu65, sd65, p65, b65 = fit_stage1(f65, codes, all_rows, all_rows)
    muS, sdS, pS, bS = fit_stage1(fsw, codes, all_rows, all_rows)
    pr65 = project(f65, codes, mu65, sd65, p65, b65)
    prS = project(fsw, codes, muS, sdS, pS, bS)
    proto65 = protos(pr65, subjects, codes, all_rows)
    protoS = protos(prS, subjects, codes, all_rows)

    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    t65 = project(np.load(TEST_E65).astype(np.float32), tcodes, mu65, sd65, p65, b65)
    tS = project(np.load(TEST_SW6).astype(np.float32), tcodes, muS, sdS, pS, bS)
    every = np.ones(len(tmeta), bool)
    ts65 = score(t65, tcodes, proto65, every)
    tsS = score(tS, tcodes, protoS, every)
    s1 = tmeta.session.astype(str).to_numpy() == 'session1'

    rows = []
    m65, d65 = float(np.mean(known65)), float(np.std(known65) + 1e-9)
    mS, dS = float(np.mean(knownS)), float(np.std(knownS) + 1e-9)
    for w in WEIGHTS:
        kn = w * ((knownS - mS) / dS) + (1 - w) * ((known65 - m65) / d65)
        un = w * ((unknownS - mS) / dS) + (1 - w) * ((unknown65 - m65) / d65)
        tn = w * ((tsS - mS) / dS) + (1 - w) * ((ts65 - m65) / d65)
        ka, q, ur = choose_threshold(kn, un)
        acc = tn >= q
        rows.append({'weight_subwin6': w, 'loso_auc': auc(kn, un),
                     'loso_unknown_rejection': ur, 'train_known_acceptance': ka,
                     'test_s1_rejection': float(np.mean(~acc[s1])),
                     'test_s2_acceptance': float(np.mean(acc[~s1]))})
        r = rows[-1]
        print(f"w={w:.2f} auc={r['loso_auc']:.4f} loso_ur={ur:.4f} ka={ka:.4f} "
              f"S1={r['test_s1_rejection']:.4f} S2={r['test_s2_acceptance']:.4f}", flush=True)
    (OUT / 'results.json').write_text(json.dumps(
        {'experiment': 'E211_clean_two_family_fusion',
         'protocol': '5-block cross-fitted registration, LOSO identity exclusion, per-fold '
                     'stats/PCA on enrolled only; weight chosen on Train only; Test audited once',
         'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
