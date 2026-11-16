"""E209: registration with much more per-identity data (block cross-fitting).

E200 registered each identity from frac<=0.6 of its trials only, because that is
what the OOF protocol required.  Deployment does not have that restriction: the
registered identities may use every available Trial of session 1.

Honest protocol that mimics "register with 80% of the data":
  * split every identity's trials into 5 contiguous blocks;
  * a query from block b is scored against prototypes built from the other blocks
    of the enrolled identities (so no query scores against its own row);
  * the held-out identity of a LOSO fold never enters any prototype;
  * the threshold is the Train-only operating point (LOSO UR >= 0.90);
  * deployment prototypes then use 100% of the rows, which is strictly more
    registration data than the calibration emulated, so the threshold stays
    conservative rather than optimistic.

Test is scored once and audited by session afterwards.
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
sys.path.insert(0, str(ROOT / 'research' / 'E200_final_freeze'))
sys.path.insert(0, str(ROOT / 'research' / 'E204_relative_distance_gate'))
from run_e204 import (  # noqa: E402
    TASKS, auc, choose_threshold, f32, fit_stage1, project, task_codes,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TEST_META = ROOT / 'BrainPrint_cache' / 'test_metadata.csv'
TRAIN_FEAT = ROOT / 'research' / 'E171_multiscale_fusion' / 'train_subwin_mean_n6.npy'
TEST_FEAT = ROOT / 'research' / 'E171_multiscale_fusion' / 'test_subwin_mean_n6.npy'
BLOCKS = 5


def make_prototypes(proj, subjects, codes, rows):
    proto = np.zeros((len(TASKS), 20, proj.shape[1]), dtype=np.float32)
    for i in range(len(TASKS)):
        for sid in range(1, 21):
            sel = rows & (codes == i) & (subjects == sid)
            if sel.any():
                proto[i, sid - 1] = proj[sel].mean(0)
    return proto


def nearest(proj, codes, proto, rows):
    out = np.full(len(proj), np.inf, dtype=np.float32)
    idx = np.flatnonzero(rows)
    if len(idx) == 0:
        return out
    q = torch.as_tensor(proj[idx], device=DEVICE)
    for i in range(len(TASKS)):
        local = np.flatnonzero(codes[idx] == i)
        if len(local) == 0:
            continue
        p = torch.as_tensor(proto[i], device=DEVICE)
        out[idx[local]] = torch.cdist(q[local], p).min(1).values.cpu().numpy()
    return out


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    codes = task_codes(meta.task.to_numpy(str))
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    blocks = np.clip((frac * BLOCKS).astype(int), 0, BLOCKS - 1)
    feat = f32(np.load(TRAIN_FEAT))
    all_rows = np.ones(len(meta), dtype=bool)

    known, unknown = [], []
    for sid in range(1, 21):
        enrolled = subjects != sid
        mu, sd, pmean, basis = fit_stage1(feat, codes, enrolled, enrolled)
        proj = project(feat, codes, mu, sd, pmean, basis)
        for b in range(BLOCKS):
            proto = make_prototypes(proj, subjects, codes, enrolled & (blocks != b))
            keep = blocks == b
            known.append(-nearest(proj, codes, proto, enrolled & keep)[enrolled & keep])
            unknown.append(-nearest(proj, codes, proto, (~enrolled) & keep)[(~enrolled) & keep])
    known = np.concatenate(known)
    unknown = np.concatenate(unknown)

    mu, sd, pmean, basis = fit_stage1(feat, codes, all_rows, all_rows)
    proj = project(feat, codes, mu, sd, pmean, basis)
    proto = make_prototypes(proj, subjects, codes, all_rows)
    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    tproj = project(f32(np.load(TEST_FEAT)), tcodes, mu, sd, pmean, basis)
    tdist = nearest(tproj, tcodes, proto, np.ones(len(tmeta), bool))
    tscores = -tdist
    session = tmeta.session.astype(str).to_numpy()
    s1 = session == 'session1'

    ka, q, ur = choose_threshold(known, unknown)
    accepted = tscores >= q
    grid = np.quantile(np.r_[known, unknown], np.linspace(0.001, 0.999, 400))
    rows = np.array([[np.mean(~(tscores[s1] >= v)), np.mean(tscores[~s1] >= v),
                      np.mean(unknown < v)] for v in grid])
    ok = rows[rows[:, 0] >= 0.70]
    ok2 = rows[rows[:, 1] >= 0.70]
    best = int(np.argmax(np.minimum(rows[:, 0], rows[:, 1])))

    result = {
        'experiment': 'E209_registration_block_crossfit',
        'protocol': '5-block cross-fitted registration on Train; LOSO identity exclusion; '
                    'Train-only threshold at LOSO UR>=0.90; Test audited once',
        'registration_rows': 'all Train rows (deployment); 4/5 blocks (calibration)',
        'loso_auc': auc(known, unknown),
        'operating_point': {'threshold': q, 'train_known_acceptance': ka,
                            'loso_unknown_rejection': ur},
        'test_s1_rejection': float(np.mean(~accepted[s1])),
        'test_s2_acceptance': float(np.mean(accepted[~s1])),
        'frontier_max_s2_at_s1_070': None if len(ok) == 0 else float(ok[:, 1].max()),
        'frontier_max_s1_at_s2_070': None if len(ok2) == 0 else float(ok2[:, 0].max()),
        'frontier_balanced': {'s1': float(rows[best, 0]), 's2': float(rows[best, 1]),
                              'loso_ur': float(rows[best, 2])},
        'test_supervised_metrics': 'N/A: no Test identity labels',
    }
    (HERE / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(CACHE / 'scores.npz', known=known, unknown=unknown,
                        test=tscores, test_session=session.astype('U16'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
