"""E207: registration-side variants that reuse the frozen subwin6 features.

All candidates change only the matching/aggregation step, never the feature
extractor, so they can be compared under one strict protocol:

  * mean-distance : identity = argmin over identities of the mean sub-window
                    distance; score = -that mean (robust to one bad sub-window)
  * median-distance
  * max-distance  : worst-case sub-window
  * min-distance  : best-case sub-window (E206 style)
  * mean-feature  : E200 baseline (nearest prototype of the averaged feature)
  * k prototypes  : min distance to k centres per identity (k = 2, 3, 4)

Every LOSO fold refits task stats, PCA and prototypes without the held-out
identity.  Operating point: LOSO rejection >= 0.90, maximise Train known
acceptance.  Test is scored once after the point is frozen.
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
CACHE.mkdir(exist_ok=True)
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TEST_META = ROOT / 'BrainPrint_cache' / 'test_metadata.csv'
TRAIN_SUB = ROOT / 'research' / 'E206_subwindow_consistency' / 'cache' / 'train_subwin6.npy'
TEST_SUB = ROOT / 'research' / 'E206_subwindow_consistency' / 'cache' / 'test_subwin6.npy'
KEYS = ['mean_dist', 'median_dist', 'max_dist', 'min_dist', 'mean_feature',
        'k2', 'k3', 'k4']


def mean_features(path, rows=None):
    arr = np.load(path, mmap_mode='r')
    if rows is None:
        return np.asarray(arr, dtype=np.float32).mean(1)
    return np.asarray(arr[rows], dtype=np.float32).mean(1)


def kmeans(x, k, seed, iters=15):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    centres = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centres).argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, x)
        count = torch.zeros(k, device=x.device).index_add_(
            0, assign, torch.ones(len(x), device=x.device))
        new = torch.where(count[:, None] > 0, sums / count.clamp(min=1)[:, None], centres)
        if torch.allclose(new, centres, atol=1e-6):
            break
        centres = new
    return centres


def build_protos(mean_proj, subjects, codes, rows, k=1):
    proto = torch.zeros((len(TASKS), 20, k, mean_proj.shape[1]), device=DEVICE)
    for i in range(len(TASKS)):
        for sid in range(1, 21):
            sel = rows & (codes == i) & (subjects == sid)
            if not sel.any():
                continue
            x = torch.as_tensor(mean_proj[sel], device=DEVICE)
            proto[i, sid - 1] = kmeans(x, min(k, len(x)), sid * 17 + i)
    return proto


def subwindow_proj(sub, idx, codes, mu, sd, pmean, basis):
    feats = np.asarray(sub[idx], dtype=np.float32)
    proj = project(feats.reshape(-1, feats.shape[-1]),
                   np.repeat(codes, SUBWIN), mu, sd, pmean, basis)
    return torch.as_tensor(proj.reshape(len(idx), SUBWIN, -1), device=DEVICE)


def subwindow_distances(proj, codes, proto1):
    n = len(codes)
    out = torch.full((n, SUBWIN, 20), float('inf'), device=DEVICE)
    for i in range(len(TASKS)):
        local = np.flatnonzero(codes == i)
        if len(local) == 0:
            continue
        d = torch.cdist(proj[local].reshape(-1, proj.shape[-1]), proto1[i])
        out[local] = d.reshape(len(local), SUBWIN, 20)
    return out


def candidates_from(dist):
    d_mean, _ = dist.mean(1).min(1)
    d_med, _ = dist.median(1).values.min(1)
    d_max, _ = dist.max(1).values.min(1)
    d_min, _ = dist.min(1).values.min(1)
    return {
        'mean_dist': (-d_mean).cpu().numpy(),
        'median_dist': (-d_med).cpu().numpy(),
        'max_dist': (-d_max).cpu().numpy(),
        'min_dist': (-d_min).cpu().numpy(),
    }


def nearest_from_mean(mean_proj, codes, protos):
    q = torch.as_tensor(mean_proj, device=DEVICE)
    vals = torch.full((len(q),), float('inf'), device=DEVICE)
    for i in range(len(TASKS)):
        local = np.flatnonzero(codes == i)
        if len(local) == 0:
            continue
        d = torch.cdist(q[local], protos[i].reshape(-1, q.shape[-1])).min(1).values
        vals[local] = d
    return (-vals).cpu().numpy()


def choose(known, unknown, target=0.90):
    grid = np.unique(np.concatenate([known, unknown]))
    best = None
    for q in grid:
        ur = float(np.mean(unknown < q))
        if ur < target:
            continue
        ka = float(np.mean(known >= q))
        if best is None or ka > best[0]:
            best = (ka, float(q), ur)
    if best is None:
        q = float(np.quantile(known, 1 - target))
        best = (float(np.mean(known >= q)), q, float(np.mean(unknown < q)))
    return best


def auc(known, unknown):
    values = np.concatenate([known, unknown])
    labels = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    sv = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    codes = task_codes(meta.task.to_numpy(str))
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    fit = frac <= 0.6
    window = (frac > 0.6) & (frac <= 0.8)
    mean_feat = mean_features(TRAIN_SUB)
    sub = np.load(TRAIN_SUB, mmap_mode='r')

    known = {k: [] for k in KEYS}
    unknown = {k: [] for k in KEYS}
    for sid in range(1, 21):
        enrolled = subjects != sid
        mu, sd, pmean, basis = fit_stage1(mean_feat, codes, enrolled, enrolled & fit)
        mean_proj = project(mean_feat, codes, mu, sd, pmean, basis)
        proto1 = build_protos(mean_proj, subjects, codes, enrolled & fit, 1)
        protos_k = {k: build_protos(mean_proj, subjects, codes, enrolled & fit, k)
                    for k in (2, 3, 4)}
        for mask, bucket in ((enrolled & window, known), (~enrolled, unknown)):
            idx = np.flatnonzero(mask)
            proj = subwindow_proj(sub, idx, codes[idx], mu, sd, pmean, basis)
            dist = subwindow_distances(proj, codes[idx], proto1)
            scores = candidates_from(dist)
            bucket['mean_dist'].append(scores['mean_dist'])
            bucket['median_dist'].append(scores['median_dist'])
            bucket['max_dist'].append(scores['max_dist'])
            bucket['min_dist'].append(scores['min_dist'])
            bucket['mean_feature'].append(nearest_from_mean(mean_proj[idx], codes[idx], proto1))
            for k, protos in protos_k.items():
                bucket[f'k{k}'].append(nearest_from_mean(mean_proj[idx], codes[idx], protos))
    known = {k: np.concatenate(v) for k, v in known.items()}
    unknown = {k: np.concatenate(v) for k, v in unknown.items()}

    # ---------- frozen all-20 model ----------
    mu, sd, pmean, basis = fit_stage1(mean_feat, codes, np.ones(len(meta), bool), fit)
    mean_proj = project(mean_feat, codes, mu, sd, pmean, basis)
    proto1 = build_protos(mean_proj, subjects, codes, fit, 1)
    protos_k = {k: build_protos(mean_proj, subjects, codes, fit, k) for k in (2, 3, 4)}
    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    tmean = mean_features(TEST_SUB)
    tmean_proj = project(tmean, tcodes, mu, sd, pmean, basis)
    tsub = np.load(TEST_SUB, mmap_mode='r')
    tproj = subwindow_proj(tsub, np.arange(len(tmeta)), tcodes, mu, sd, pmean, basis)
    tdist = subwindow_distances(tproj, tcodes, proto1)
    tscores = candidates_from(tdist)
    tscores['mean_feature'] = nearest_from_mean(tmean_proj, tcodes, proto1)
    for k, protos in protos_k.items():
        tscores[f'k{k}'] = nearest_from_mean(tmean_proj, tcodes, protos)
    session = tmeta.session.astype(str).to_numpy()
    s1 = session == 'session1'

    results = {}
    for key in KEYS:
        ka, q, ur = choose(known[key], unknown[key])
        accepted = tscores[key] >= q
        results[key] = {
            'threshold': q, 'train_known_acceptance': ka,
            'loso_unknown_rejection': ur,
            'loso_auc': auc(known[key], unknown[key]),
            'test_s1_rejection': float(np.mean(~accepted[s1])),
            'test_s2_acceptance': float(np.mean(accepted[~s1])),
        }
        p = np.quantile(np.r_[known[key], unknown[key]], np.linspace(0.001, 0.999, 400))
        rows = np.array([[np.mean(~(tscores[key][s1] >= v)), np.mean(tscores[key][~s1] >= v),
                          np.mean(unknown[key] < v)] for v in p])
        ok = rows[rows[:, 0] >= 0.70]
        results[key]['frontier_max_s2_at_s1_070'] = None if len(ok) == 0 else float(ok[:, 1].max())
        ok2 = rows[rows[:, 1] >= 0.70]
        results[key]['frontier_max_s1_at_s2_070'] = None if len(ok2) == 0 else float(ok2[:, 0].max())

    summary = {
        'experiment': 'E207_registration_aggregation',
        'protocol': 'strict 20-fold LOSO; operation point Train-only (LOSO UR>=0.90, max known '
                    'acceptance); Test audited once',
        'candidates': results,
        'test_supervised_metrics': 'N/A: no Test identity labels',
    }
    (HERE / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(CACHE / 'scores.npz',
                        test_session=session.astype('U16'),
                        test_task=tmeta.task.to_numpy(str).astype('U16'),
                        **{f'test_{k}': v for k, v in tscores.items()},
                        **{f'known_{k}': v for k, v in known.items()},
                        **{f'unknown_{k}': v for k, v in unknown.items()})
    for key, v in results.items():
        print(f"{key:14s} loso_ur={v['loso_unknown_rejection']:.4f} ka={v['train_known_acceptance']:.4f} "
              f"auc={v['loso_auc']:.4f} S1={v['test_s1_rejection']:.4f} S2={v['test_s2_acceptance']:.4f} "
              f"frontier(S2={v['frontier_max_s2_at_s1_070']}, S1={v['frontier_max_s1_at_s2_070']})")


if __name__ == '__main__':
    main()
