"""E205: radius-normalised nearest-prototype score on the frozen E200 subwin6 pipeline.

Motivation: E204 showed that a pure per-sample relative score improves Train LOSO
discrimination but degrades the Test S1/S2 frontier, while the raw nearest distance
keeps a usable Test frontier but poor LOSO rejection.  A registered identity owns a
within-class scatter radius in the same feature space, so d_i / r_i puts every
identity on a comparable scale and can absorb part of a global session shift.

Strict protocol: in every LOSO fold the task statistics, PCA basis, prototypes and
the per-identity radii are refit with the held-out identity removed.
Test is scored once and audited afterwards; thresholds come from Train only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'E204_relative_distance_gate'))
from run_e204 import (  # noqa: E402
    DEVICE, DIM, TASKS, TRAIN_FEAT, TRAIN_META, TEST_FEAT, TEST_META,
    auc, choose_threshold, distances, f32, fit_stage1, project, task_codes,
)

OUT = HERE
CACHE = OUT / 'cache'
CACHE.mkdir(exist_ok=True)
EPS = 1e-6


def prototypes_and_radii(proj, subjects, codes, rows):
    proto = np.zeros((len(TASKS), 20, proj.shape[1]), dtype=np.float32)
    radius = np.full((len(TASKS), 20), np.nan, dtype=np.float32)
    for i in range(len(TASKS)):
        for sid in range(1, 21):
            sel = rows & (codes == i) & (subjects == sid)
            if not sel.any():
                continue
            centre = proj[sel].mean(0)
            proto[i, sid - 1] = centre
            radius[i, sid - 1] = np.linalg.norm(proj[sel] - centre, axis=1).mean()
    task_median = np.nanmedian(radius, axis=1, keepdims=True)
    radius = np.where(np.isnan(radius), task_median, radius)
    return proto, radius


def radius_score(dist, radius, codes):
    ratio = dist / np.maximum(radius[codes], EPS)
    best = np.nanmin(ratio, axis=1)
    return -best


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    codes = task_codes(meta.task.to_numpy(str))
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    fit = frac <= 0.6
    window = (frac > 0.6) & (frac <= 0.8)
    feat = f32(np.load(TRAIN_FEAT))

    keys = ['raw', 'radius', 'raw_z_plus_radius']
    known = {k: [] for k in keys}
    unknown = {k: [] for k in keys}
    fold_radius = []
    for sid in range(1, 21):
        enrolled = subjects != sid
        mu, sd, pmean, basis = fit_stage1(feat, codes, enrolled, enrolled & fit)
        proj = project(feat, codes, mu, sd, pmean, basis)
        proto, radius = prototypes_and_radii(proj, subjects, codes, enrolled & fit)
        hold = ~enrolled
        d_hold = distances(proj, codes, proto, hold)[hold]
        d_known = distances(proj, codes, proto, enrolled & window)[enrolled & window]
        c_hold = codes[hold]
        c_known = codes[enrolled & window]
        fold_radius.append(radius)
        known['raw'].append(-np.nanmin(d_hold if False else d_known, axis=1))
        unknown['raw'].append(-np.nanmin(d_hold, axis=1))
        known['radius'].append(radius_score(d_known, radius, c_known))
        unknown['radius'].append(radius_score(d_hold, radius, c_hold))
    # fold arrays were appended in the loop for raw/radius only; build the combination
    known = {k: np.concatenate(v) for k, v in known.items() if v}
    unknown = {k: np.concatenate(v) for k, v in unknown.items() if v}
    raw_mu, raw_sd = known['raw'].mean(), known['raw'].std()
    rad_mu, rad_sd = known['radius'].mean(), known['radius'].std()
    for weight in (0.5,):
        known['raw_z_plus_radius'] = (weight * (known['raw'] - raw_mu) / raw_sd
                                      + (1 - weight) * (known['radius'] - rad_mu) / rad_sd)
        unknown['raw_z_plus_radius'] = (weight * (unknown['raw'] - raw_mu) / raw_sd
                                        + (1 - weight) * (unknown['radius'] - rad_mu) / rad_sd)

    # ---------- frozen all-20 model ----------
    mu, sd, pmean, basis = fit_stage1(feat, codes, np.ones(len(meta), bool), fit)
    proj = project(feat, codes, mu, sd, pmean, basis)
    proto, radius = prototypes_and_radii(proj, subjects, codes, fit)
    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    tproj = project(f32(np.load(TEST_FEAT)), tcodes, mu, sd, pmean, basis)
    tdist = distances(tproj, tcodes, proto, np.ones(len(tmeta), bool))
    tscores = {
        'raw': -np.nanmin(tdist, axis=1),
        'radius': radius_score(tdist, radius, tcodes),
    }
    tscores['raw_z_plus_radius'] = (0.5 * (tscores['raw'] - raw_mu) / raw_sd
                                    + 0.5 * (tscores['radius'] - rad_mu) / rad_sd)
    session = tmeta.session.astype(str).to_numpy()
    s1 = session == 'session1'

    results = {}
    for key in keys:
        ka, q, ur = choose_threshold(known[key], unknown[key])
        accepted = tscores[key] >= q
        results[key] = {
            'threshold': float(q),
            'train_known_acceptance': ka,
            'loso_unknown_rejection': ur,
            'loso_auc': auc(known[key], unknown[key]),
            'test_s1_rejection': float(np.mean(~accepted[s1])),
            'test_s2_acceptance': float(np.mean(accepted[~s1])),
        }
        # Frontier diagnostics: best achievable pair on Test (analysis only)
        p = np.quantile(np.r_[known[key], unknown[key]], np.linspace(0.001, 0.999, 400))
        rows = np.array([[np.mean(~(tscores[key][s1] >= v)), np.mean(tscores[key][~s1] >= v),
                          np.mean(unknown[key] < v), np.mean(known[key] >= v)] for v in p])
        ok = rows[rows[:, 0] >= 0.70]
        results[key]['frontier_max_s2_at_s1_070'] = None if len(ok) == 0 else float(ok[:, 1].max())
        ok2 = rows[rows[:, 1] >= 0.70]
        results[key]['frontier_max_s1_at_s2_070'] = None if len(ok2) == 0 else float(ok2[:, 0].max())
        best = np.argmax(np.minimum(rows[:, 0], rows[:, 1]))
        results[key]['frontier_balanced'] = {
            's1': float(rows[best, 0]), 's2': float(rows[best, 1]),
            'loso_ur': float(rows[best, 2]), 'train_ka': float(rows[best, 3])}

    summary = {
        'experiment': 'E205_radius_normalised_gate',
        'protocol': 'strict 20-fold LOSO; task stats/PCA/prototypes/radii exclude held-out '
                    'identity; Train-only threshold at LOSO rejection >= 0.90; Test audited once',
        'known_rows': int(len(known['raw'])), 'unknown_rows': int(len(unknown['raw'])),
        'test_s1_rows': int(s1.sum()), 'test_s2_rows': int((~s1).sum()),
        'candidates': results,
        'test_supervised_metrics': 'N/A: no Test identity labels',
    }
    (OUT / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(CACHE / 'scores.npz', test_dist=tdist, radius=radius,
                        test_session=session.astype('U16'),
                        test_task=tmeta.task.to_numpy(str).astype('U16'),
                        **{f'test_{k}': v for k, v in tscores.items()},
                        **{f'known_{k}': v for k, v in known.items()},
                        **{f'unknown_{k}': v for k, v in unknown.items()})
    for key, value in results.items():
        print(f"{key:20s} loso_ur={value['loso_unknown_rejection']:.4f} "
              f"ka={value['train_known_acceptance']:.4f} auc={value['loso_auc']:.4f} "
              f"S1={value['test_s1_rejection']:.4f} S2={value['test_s2_acceptance']:.4f} "
              f"frontier(S2@S1.7={value['frontier_max_s2_at_s1_070']}, "
              f"S1@S2.7={value['frontier_max_s1_at_s2_070']})")


if __name__ == '__main__':
    main()
