"""E204: per-sample relative-distance scoring on the frozen E200 subwin6 pipeline.

Train-only.  Every fitted quantity (task mean/sd, PCA, prototypes) is refit
inside each LOSO fold with the held-out identity removed, so the reported LOSO
numbers are clean.  Test is scored once, after all thresholds are frozen, and
only then audited by session.

Candidate scores (larger = more likely registered), all derived from the full
20-identity distance vector d of a sample inside its own task:
  raw          -d_min
  rel_mean     -(d_min - mean(d))
  z            -(d_min - mean(d)) / std(d)
  ratio        -d_min / mean(d)
  margin       -(d2 - d1) / d1
  z_plus_rel   z(train-OOF normalised raw) + z(per-sample relative term)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = OUT / 'cache'
CACHE.mkdir(exist_ok=True)
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TEST_META = ROOT / 'BrainPrint_cache' / 'test_metadata.csv'
TRAIN_FEAT = ROOT / 'research' / 'E171_multiscale_fusion' / 'train_subwin_mean_n6.npy'
TEST_FEAT = ROOT / 'research' / 'E171_multiscale_fusion' / 'test_subwin_mean_n6.npy'
TASKS = ['EC', 'EO', 'Imagery', 'SSVEP']
DIM = 96
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f32(x):
    return np.asarray(x, dtype=np.float32)


def task_codes(values):
    return np.array([TASKS.index(v) for v in values], dtype=np.int64)


def fit_stage1(feat, codes, stat_rows, pca_rows, dim=DIM):
    n_task = len(TASKS)
    mu = np.zeros((n_task, feat.shape[1]), dtype=np.float32)
    sd = np.zeros_like(mu)
    for i in range(n_task):
        sel = stat_rows & (codes == i)
        mu[i] = feat[sel].mean(0)
        sd[i] = feat[sel].std(0) + 1e-6
    z = (feat - mu[codes]) / sd[codes]
    x = torch.as_tensor(z[pca_rows], device=DEVICE)
    mean = x.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    basis = vh[:dim].T.contiguous()
    return mu, sd, mean.cpu().numpy()[0], basis.cpu().numpy()


def project(feat, codes, mu, sd, pmean, basis):
    z = (feat - mu[codes]) / sd[codes]
    return ((z - pmean) @ basis).astype(np.float32)


def prototypes(proj, subjects, codes, rows):
    n_task = len(TASKS)
    proto = np.zeros((n_task, 20, proj.shape[1]), dtype=np.float32)
    filled = np.zeros((n_task, 20), dtype=bool)
    for i in range(n_task):
        for sid in range(1, 21):
            sel = rows & (codes == i) & (subjects == sid)
            if sel.any():
                proto[i, sid - 1] = proj[sel].mean(0)
                filled[i, sid - 1] = True
    return proto, filled


def distances(proj, codes, proto, rows):
    out = np.full((len(proj), 20), np.nan, dtype=np.float32)
    idx = np.flatnonzero(rows)
    if len(idx) == 0:
        return out
    p = torch.as_tensor(proto, device=DEVICE)
    q = torch.as_tensor(proj[idx], device=DEVICE)
    c = torch.as_tensor(codes[idx], device=DEVICE)
    for i in range(len(TASKS)):
        sel = torch.nonzero(c == i).squeeze(1)
        if len(sel) == 0:
            continue
        d = torch.cdist(q[sel], p[i])
        out[idx[sel.cpu().numpy()]] = d.cpu().numpy()
    return out


def scores_from_dist(dist):
    ordered = np.sort(dist, axis=1)
    d1 = ordered[:, 0]
    d2 = ordered[:, 1]
    mean = np.nanmean(dist, axis=1)
    std = np.nanstd(dist, axis=1) + 1e-6
    return {
        'raw': -d1,
        'rel_mean': -(d1 - mean),
        'z': -(d1 - mean) / std,
        'ratio': -d1 / mean,
        'margin': -(d2 - d1) / d1,
    }


def choose_threshold(known, unknown, target=0.90):
    pooled = np.unique(np.concatenate([known, unknown]))
    best = None
    for q in pooled:
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
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
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
    feat = f32(np.load(TRAIN_FEAT))
    all_rows = np.ones(len(meta), dtype=bool)

    # ---------- clean 20-fold LOSO -------------------------------
    known_parts = {k: [] for k in ['raw', 'rel_mean', 'z', 'ratio', 'margin']}
    unknown_parts = {k: [] for k in known_parts}
    per_fold = []
    for sid in range(1, 21):
        enrolled = subjects != sid
        mu, sd, pmean, basis = fit_stage1(feat, codes, enrolled, enrolled & fit)
        proj = project(feat, codes, mu, sd, pmean, basis)
        proto, filled = prototypes(proj, subjects, codes, enrolled & fit)
        d_hold = distances(proj, codes, proto, ~enrolled)[~enrolled]
        d_known = distances(proj, codes, proto, enrolled & window)[enrolled & window]
        s_hold = scores_from_dist(d_hold)
        s_known = scores_from_dist(d_known)
        fold = {'held_out': sid, 'enrolled_cells': int(filled.sum())}
        for key in known_parts:
            known_parts[key].append(s_known[key])
            unknown_parts[key].append(s_hold[key])
        per_fold.append(fold)
    known = {k: np.concatenate(v) for k, v in known_parts.items()}
    unknown = {k: np.concatenate(v) for k, v in unknown_parts.items()}

    # ---------- frozen all-20 model, Test scored once -------------
    mu, sd, pmean, basis = fit_stage1(feat, codes, all_rows, fit)
    proj = project(feat, codes, mu, sd, pmean, basis)
    proto, filled = prototypes(proj, subjects, codes, fit)
    assert filled.all()
    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    tfeat = f32(np.load(TEST_FEAT))
    tproj = project(tfeat, tcodes, mu, sd, pmean, basis)
    tdist = distances(tproj, tcodes, proto, np.ones(len(tmeta), dtype=bool))
    tscores = scores_from_dist(tdist)
    session = tmeta.session.astype(str).to_numpy()
    s1 = session == 'session1'

    # ---------- fixed Train-only threshold per candidate ----------
    results = {}
    for key in known_parts:
        ka, q, ur = choose_threshold(known[key], unknown[key])
        accepted = tscores[key] >= q
        results[key] = {
            'threshold': q,
            'loso_unknown_rejection': ur,
            'train_known_acceptance': ka,
            'loso_auc': auc(known[key], unknown[key]),
            'test_s1_rejection': float(np.mean(~accepted[s1])),
            'test_s2_acceptance': float(np.mean(accepted[~s1])),
            'test_s1_s2_mean': float((np.mean(~accepted[s1]) + np.mean(accepted[~s1])) / 2),
        }
    # combined candidate: raw z-scored on LOSO-known plus relative term
    zr_known = (known['raw'] - known['raw'].mean()) / known['raw'].std()
    zr_unknown = (unknown['raw'] - known['raw'].mean()) / known['raw'].std()
    zz_known = (known['z'] - known['z'].mean()) / known['z'].std()
    zz_unknown = (unknown['z'] - known['z'].mean()) / known['z'].std()
    for w in [0.25, 0.5, 0.75]:
        kk = w * zr_known + (1 - w) * zz_known
        uu = w * zr_unknown + (1 - w) * zz_unknown
        ka, q, ur = choose_threshold(kk, uu)
        tr = (tscores['raw'] - known['raw'].mean()) / known['raw'].std()
        tz = (tscores['z'] - known['z'].mean()) / known['z'].std()
        accepted = w * tr + (1 - w) * tz >= q
        results[f'z_plus_rel_w{w}'] = {
            'threshold': q,
            'loso_unknown_rejection': ur,
            'train_known_acceptance': ka,
            'loso_auc': auc(kk, uu),
            'test_s1_rejection': float(np.mean(~accepted[s1])),
            'test_s2_acceptance': float(np.mean(accepted[~s1])),
            'test_s1_s2_mean': float((np.mean(~accepted[s1]) + np.mean(accepted[~s1])) / 2),
        }

    summary = {
        'experiment': 'E204_per_sample_relative_distance',
        'protocol': 'strict 20-fold LOSO (stats/PCA/prototypes exclude held-out identity); '
                    'threshold from Train known window and LOSO unknown at rejection>=0.90; '
                    'Test audited once after freezing',
        'device': str(DEVICE),
        'n_train': int(len(meta)), 'n_test': int(len(tmeta)),
        'test_s1_rows': int(s1.sum()), 'test_s2_rows': int((~s1).sum()),
        'candidates': results,
        'test_supervised_metrics': 'N/A: no Test identity labels',
    }
    (OUT / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(
        CACHE / 'distance_state.npz',
        test_dist=tdist, test_session=session.astype('U16'),
        test_task=tmeta.task.to_numpy(str).astype('U16'),
        **{f'test_{k}': v for k, v in tscores.items()},
        **{f'known_{k}': v for k, v in known.items()},
        **{f'unknown_{k}': v for k, v in unknown.items()},
    )
    for key, value in results.items():
        print(f"{key:16s} loso_ur={value['loso_unknown_rejection']:.4f} "
              f"ka={value['train_known_acceptance']:.4f} auc={value['loso_auc']:.4f} "
              f"S1={value['test_s1_rejection']:.4f} S2={value['test_s2_acceptance']:.4f}")


if __name__ == '__main__':
    main()
