"""E206: temporal identity consistency gate + nearest-distance gate.

Signal: a registered identity should be matched to the *same* prototype by most of
the 6 sub-windows inside one trial; for an unregistered identity the votes scatter.
The first pilot showed consistency >= 5/6 rejects ~90% of held-out identities while
retaining ~67% of registered ones, far better separation than distance alone.

Strict protocol: every LOSO fold refits task stats, PCA and prototypes without the
held-out identity, and only then scores it.  The operating point is selected from
Train only (LOSO rejection >= 0.90, maximise Train known acceptance).  Test is
scored once and audited by session afterwards.
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
from build_bundle import SUBWIN  # noqa: E402
from run_e204 import TASKS, f32, fit_stage1, project, task_codes  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = HERE / 'cache'
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TEST_META = ROOT / 'BrainPrint_cache' / 'test_metadata.csv'
TRAIN_SUB = CACHE / 'train_subwin6.npy'
TEST_SUB = CACHE / 'test_subwin6.npy'
EPS = 1e-9


def make_prototypes(mean_proj, subjects, codes, rows):
    proto = np.zeros((len(TASKS), 20, mean_proj.shape[1]), dtype=np.float32)
    for i in range(len(TASKS)):
        for sid in range(1, 21):
            sel = rows & (codes == i) & (subjects == sid)
            if sel.any():
                proto[i, sid - 1] = mean_proj[sel].mean(0)
    return proto


def vote_stats(sub_features, codes, proto):
    """Majority vote share and minimum distance over sub-windows for the given rows."""
    n = len(sub_features)
    vote = np.zeros((n, 20), dtype=np.int16)
    dmin = np.full(n, np.inf, dtype=np.float32)
    if n == 0:
        return vote, dmin
    p = torch.as_tensor(proto, device=DEVICE)
    q = torch.as_tensor(sub_features.reshape(-1, sub_features.shape[-1]), device=DEVICE)
    for i in range(len(TASKS)):
        local = np.flatnonzero(codes == i)
        if len(local) == 0:
            continue
        pick = (local[:, None] * SUBWIN + np.arange(SUBWIN)[None, :]).ravel()
        d = torch.cdist(q[pick], p[i])  # (len(local)*SUBWIN, 20)
        nearest = d.argmin(1).view(-1, SUBWIN).cpu().numpy()
        values = d.min(1).values.view(-1, SUBWIN).min(1).values.cpu().numpy()
        for k in range(len(local)):
            vote[local[k]] = np.bincount(nearest[k], minlength=20)
        dmin[local] = values
    return vote, dmin


def consistency(vote):
    return vote.max(1) / float(SUBWIN)


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    codes = task_codes(meta.task.to_numpy(str))
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    fit = frac <= 0.6
    window = (frac > 0.6) & (frac <= 0.8)
    sub = np.load(TRAIN_SUB, mmap_mode='r')
    mean_feat = np.asarray(sub, dtype=np.float32).mean(1)
    ref = f32(np.load(ROOT / 'research' / 'E171_multiscale_fusion' / 'train_subwin_mean_n6.npy'))
    rebuild_max = float(np.max(np.abs(mean_feat - ref)))

    known_vote, known_d1, known_subj = [], [], []
    unknown_vote, unknown_d1, unknown_subj = [], [], []
    fold_rows = []
    for sid in range(1, 21):
        enrolled = subjects != sid
        mu, sd, pmean, basis = fit_stage1(mean_feat, codes, enrolled, enrolled & fit)
        mean_proj = project(mean_feat, codes, mu, sd, pmean, basis)
        proto = make_prototypes(mean_proj, subjects, codes, enrolled & fit)
        for mask, votes, dists, subs in (
                (enrolled & window, known_vote, known_d1, known_subj),
                (~enrolled, unknown_vote, unknown_d1, unknown_subj)):
            idx = np.flatnonzero(mask)
            feats = np.asarray(sub[idx], dtype=np.float32)
            proj = project(feats.reshape(-1, feats.shape[-1]),
                           np.repeat(codes[idx], SUBWIN), mu, sd, pmean, basis)
            proj = proj.reshape(len(idx), SUBWIN, -1)
            v, d = vote_stats(proj, codes[idx], proto)
            votes.append(v)
            dists.append(d)
            subs.append((subjects[idx], codes[idx]))
        fold_rows.append(int((~enrolled).sum()))
    kv = np.concatenate(known_vote)
    kd = np.concatenate(known_d1)
    uv = np.concatenate(unknown_vote)
    ud = np.concatenate(unknown_d1)
    kc = consistency(kv)
    uc = consistency(uv)

    # ---------- frozen all-20 model, Test scored once ----------
    mu, sd, pmean, basis = fit_stage1(mean_feat, codes, np.ones(len(meta), bool), fit)
    mean_proj = project(mean_feat, codes, mu, sd, pmean, basis)
    proto = make_prototypes(mean_proj, subjects, codes, fit)
    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    tmean = np.asarray(np.load(TEST_SUB, mmap_mode='r'), dtype=np.float32).mean(1)
    tproj_mean = project(tmean, tcodes, mu, sd, pmean, basis)
    p = torch.as_tensor(proto, device=DEVICE)
    q = torch.as_tensor(tproj_mean, device=DEVICE)
    tdist = np.full(len(tmeta), np.inf, dtype=np.float32)
    tident = np.zeros(len(tmeta), dtype=np.int32)
    for i in range(len(TASKS)):
        rows = np.flatnonzero(tcodes == i)
        d = torch.cdist(q[rows], p[i])
        tdist[rows] = d.min(1).values.cpu().numpy()
        tident[rows] = d.argmin(1).cpu().numpy() + 1
    tsub = np.load(TEST_SUB, mmap_mode='r')
    tvote = np.zeros((len(tmeta), 20), dtype=np.int16)
    td1sub = np.full(len(tmeta), np.inf, dtype=np.float32)
    for i in range(len(TASKS)):
        rows = np.flatnonzero(tcodes == i)
        feats = np.asarray(tsub[rows], dtype=np.float32)
        proj = project(feats.reshape(-1, feats.shape[-1]),
                       np.repeat(tcodes[rows], SUBWIN), mu, sd, pmean, basis)
        proj = proj.reshape(len(rows), SUBWIN, -1)
        v, d = vote_stats(proj, tcodes[rows], proto)
        tvote[rows] = v
        td1sub[rows] = d
    tcons = consistency(tvote)
    session = tmeta.session.astype(str).to_numpy()
    s1 = session == 'session1'

    # ---------- Train-only operating point ----------
    tau = float(np.quantile(kd, 0.90))
    grid = np.unique(kc)
    chosen = None
    for gamma in sorted(grid, reverse=True):
        ur = float(np.mean(uc < gamma))
        if ur < 0.90:
            continue
        ka = float(np.mean((kc >= gamma) & (kd <= tau)))
        if chosen is None or ka > chosen[1]:
            chosen = (float(gamma), ka, ur)
    gamma, ka, ur = chosen
    accept_test = (tcons >= gamma) & (tdist <= tau)
    # diagnostics: pure consistency gate
    pure = None
    for g in sorted(grid, reverse=True):
        u = float(np.mean(uc < g))
        if u < 0.90:
            continue
        k = float(np.mean(kc >= g))
        if pure is None or k > pure[1]:
            pure = (float(g), k, u)

    result = {
        'experiment': 'E206_subwindow_consistency_gate',
        'feature': 'subwin6 per-subwindow features, 6 votes per trial',
        'mean_feature_rebuild_max_abs': rebuild_max,
        'protocol': 'strict 20-fold LOSO (stats/PCA/prototypes exclude the held-out identity); '
                    'operating point from Train only; Test audited once after freezing',
        'train_rows': int(len(meta)), 'test_rows': int(len(tmeta)),
        'consistency_known': {'mean': float(kc.mean()),
                              'share_ge_4of6': float(np.mean(kc >= 4 / 6)),
                              'share_ge_5of6': float(np.mean(kc >= 5 / 6))},
        'consistency_unknown': {'mean': float(uc.mean()),
                                'share_ge_4of6': float(np.mean(uc >= 4 / 6)),
                                'share_ge_5of6': float(np.mean(uc >= 5 / 6))},
        'distance_tau_known90': tau,
        'operating_point': {'gamma': gamma, 'tau': tau,
                            'train_known_acceptance': ka,
                            'loso_unknown_rejection': ur},
        'pure_consistency_point': {'gamma': pure[0], 'train_known_acceptance': pure[1],
                                   'loso_unknown_rejection': pure[2]} if pure else None,
        'test_s1_rejection': float(np.mean(~accept_test[s1])),
        'test_s2_acceptance': float(np.mean(accept_test[~s1])),
        'test_s1_rows': int(s1.sum()), 'test_s2_rows': int((~s1).sum()),
        'test_identity_accuracy': 'N/A: no Test identity labels',
        'test_supervised_metrics': 'N/A: no Test identity labels',
    }
    (HERE / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(CACHE / 'gate_state.npz',
                        known_consistency=kc, unknown_consistency=uc,
                        known_d1=kd, unknown_d1=ud,
                        test_consistency=tcons, test_d1=tdist, test_identity=tident,
                        test_session=session.astype('U16'),
                        test_task=tmeta.task.to_numpy(str).astype('U16'),
                        tau=np.float64(tau), gamma=np.float64(gamma))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
