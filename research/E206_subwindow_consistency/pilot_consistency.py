"""E206: per-subwindow features of one trial -> temporal identity consistency.

Pilot: does the sub-window vote agreement separate a registered identity from a
held-out identity?  Runs on a small Train subset, no cache written.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'E200_final_freeze'))
sys.path.insert(0, str(ROOT / 'research' / 'E204_relative_distance_gate'))
from build_bundle import FEATURE, SUBWIN, load_mat  # noqa: E402
from build_bundle import logspd_subwindow  # noqa: E402
from run_e204 import TASKS, f32, fit_stage1, project, task_codes  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TRAIN_FEAT = ROOT / 'research' / 'E171_multiscale_fusion' / 'train_subwin_mean_n6.npy'
BASE = ROOT / 'Brainprint_Competition'
WINDOW_PER_TASK = 4
FIT_PER_TASK = 4


def subwindows(raw):
    length = FEATURE['subwindow_length']
    parts = [logspd_subwindow(raw[..., w * length:(w + 1) * length], length)
             for w in range(SUBWIN)]
    return torch.stack(parts, dim=1)


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy(str)
    codes = task_codes(tasks)
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    fit = frac <= 0.6
    window = (frac > 0.6) & (frac <= 0.8)

    chosen = []
    for sid in range(1, 21):
        for task in TASKS:
            sel_fit = np.flatnonzero((subjects == sid) & (tasks == task) & fit)[:FIT_PER_TASK]
            sel_win = np.flatnonzero((subjects == sid) & (tasks == task) & window)[:WINDOW_PER_TASK]
            chosen.extend(sel_fit.tolist())
            chosen.extend(sel_win.tolist())
    chosen = np.array(sorted(chosen))
    print(f'pilot trials {len(chosen)}')

    paths = [BASE / str(p) for p in meta['path'].to_numpy()[chosen]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        arrays = list(pool.map(load_mat, paths))
    raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        sub = subwindows(raw)
    sub = sub.cpu().numpy()
    mean_feat = sub.mean(1)
    ref = f32(np.load(TRAIN_FEAT))
    print('subwindow-mean vs E171 cache max|d|',
          float(np.max(np.abs(mean_feat - ref[chosen]))))

    # stage 1 + prototypes from the full Train set (pilot effect-size check only)
    feat = ref
    mu, sd, pmean, basis = fit_stage1(feat, codes, np.ones(len(meta), bool), fit)
    proj = project(feat, codes, mu, sd, pmean, basis)
    proto = np.zeros((len(TASKS), 20, proj.shape[1]), dtype=np.float32)
    for i in range(len(TASKS)):
        for sid in range(1, 21):
            sel = fit & (codes == i) & (subjects == sid)
            proto[i, sid - 1] = proj[sel].mean(0)

    subj_proj = project(sub.reshape(-1, sub.shape[-1]),
                        np.repeat(codes[chosen], SUBWIN),
                        mu, sd, pmean, basis).reshape(len(chosen), SUBWIN, -1)
    mean_proj = project(mean_feat, codes[chosen], mu, sd, pmean, basis)

    def vote_and_distance(proj_trials, trial_codes, banned=None):
        p = torch.as_tensor(proto, device=DEVICE)
        q = torch.as_tensor(proj_trials, device=DEVICE)
        vote = np.full((len(proj_trials), 20), 0.0, dtype=np.float32)
        dist = np.full((len(proj_trials), 20), np.nan, dtype=np.float32)
        for i in range(len(TASKS)):
            rows = np.flatnonzero(trial_codes == i)
            if len(rows) == 0:
                continue
            d = torch.cdist(q[rows], p[i])
            if banned is not None:
                for local, global_row in enumerate(rows):
                    if banned[global_row] is not None:
                        d[local, banned[global_row] - 1] = float('inf')
            nearest = d.argmin(2).cpu().numpy()
            dist[rows] = d.min(1).values.cpu().numpy()
            for local, global_row in enumerate(rows):
                vote[global_row] = np.bincount(nearest[local], minlength=20)
        return vote, dist

    tsid = subjects[chosen]
    # ---- known: enrolled window rows, prototypes exclude nobody
    known_rows = np.flatnonzero(window[chosen])
    vote_k, dist_k = vote_and_distance(subj_proj[known_rows], codes[chosen][known_rows])
    consistency_k = vote_k.max(1) / SUBWIN
    top_known = vote_k.argmax(1) + 1
    accuracy = float(np.mean(top_known == tsid[known_rows]))

    # ---- unknown: leave-one-subject-out
    cons_u, d1_u = [], []
    for sid in range(1, 21):
        rows = np.flatnonzero((tsid == sid) & window[chosen])
        if len(rows) == 0:
            continue
        banned = {r: sid for r in range(len(rows))}
        # rebuild prototypes without this identity
        localized = proto.copy()
        for i in range(len(TASKS)):
            p = torch.as_tensor(proto[i], device=DEVICE)
            q = torch.as_tensor(subj_proj[rows], device=DEVICE)
            d = torch.cdist(q, p)
            d[..., sid - 1] = float('inf')
            nearest = d.argmin(2).cpu().numpy()
            for local in range(len(rows)):
                counts = np.bincount(nearest[local], minlength=20)
                cons_u.append(counts.max() / SUBWIN)
            d1_u.extend(d.min(2).values.min(1).values.cpu().numpy().tolist())

    d1_k = dist_k[:, :].min(1)
    print(f'known: consistency mean {consistency_k.mean():.4f} median {np.median(consistency_k):.4f}, '
          f'identity acc {accuracy:.4f}, d1 mean {d1_k.mean():.2f}')
    print(f'unknown: consistency mean {np.mean(cons_u):.4f} median {np.median(cons_u):.4f}, '
          f'd1 mean {np.mean(d1_u):.2f}')
    for g in (0.5, 0.67, 0.83, 1.0):
        print(f'  consistency >= {g}: known {np.mean(consistency_k >= g):.4f}, '
              f'unknown {np.mean(np.asarray(cons_u) >= g):.4f}')
    d1_u = np.asarray(d1_u)
    tau = float(np.quantile(d1_k, 0.9))
    print(f'  d1 tau(known 90%) {tau:.3f}: known {np.mean(d1_k <= tau):.4f}, '
          f'unknown {np.mean(d1_u <= tau):.4f}')
    for g in (0.5, 0.67, 0.83):
        ka = np.mean((d1_k <= tau) & (consistency_k >= g))
        ur = np.mean((d1_u <= tau) & (np.asarray(cons_u) >= g))
        print(f'  joint d1<=tau & cons>={g}: known accept {ka:.4f}, unknown accept {ur:.4f}, '
              f'unknown rejection {1 - ur:.4f}')


if __name__ == '__main__':
    main()
