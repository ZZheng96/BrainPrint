"""E210: is the historical "high LOSO + high S2" real?

E137/E139/E141 all compute LOSO rejection as
    mean(unknown < quantile(known, 1-a))
where ``known`` are window rows of the enrolled identities scored against
prototypes that were built from ``pm`` = *all* rows of those same identities,
i.e. the rows that define the threshold are inside the prototypes.  The known
scores are therefore in-sample and the quantile is too tight, which inflates the
reported LOSO rejection.

This script rebuilds E139's best configuration (E65 logSPD, PCA-96, k prototypes,
acceptance grid) with two protocols:
  * contaminated : exactly E139's rule
  * clean        : known window rows are scored against prototypes built from the
                   fit rows only, so no evaluated row enters its own prototype
The Test side is identical in both (5-block OOF threshold + deployed prototypes),
so the difference isolates the LOSO inflation.
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
from run_e204 import TASKS, task_codes  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = HERE
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TEST_META = ROOT / 'BrainPrint_cache' / 'test_metadata.csv'
TRAIN_E65 = ROOT / 'BrainPrint_cache' / 'e65_correlation_spd' / 'train_correlation_logspd.npy'
TEST_E65 = ROOT / 'research' / 'E202_dual_model_cross_session' / 'cache' / 'test_e65_features.npy'
KS = [1, 12, 32]
ACCEPT = [0.90, 0.96]


def kmeans(x, k, seed, iters=20):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    c = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.zeros(k, device=x.device).index_add_(0, a, torch.ones(len(x), device=x.device))
        new = torch.where(n[:, None] > 0, s / n.clamp(min=1)[:, None], c)
        if torch.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def protos(proj, subjects, codes, rows, k):
    out = torch.zeros((len(TASKS), 20, k, proj.shape[1]), device=DEVICE)
    for i in range(len(TASKS)):
        for sid in range(1, 21):
            sel = rows & (codes == i) & (subjects == sid)
            if not sel.any():
                continue
            x = torch.as_tensor(proj[sel], device=DEVICE)
            centres = kmeans(x, min(k, len(x)), sid * 17 + i)
            if len(centres) < k:  # too few rows for k centres: duplicate the last one
                pad = centres[-1:].repeat(k - len(centres), 1)
                centres = torch.cat([centres, pad], dim=0)
            out[i, sid - 1] = centres
    return out


def score(proj, codes, proto):
    out = np.full(len(proj), np.inf, dtype=np.float32)
    q = torch.as_tensor(proj, device=DEVICE)
    for i in range(len(TASKS)):
        local = np.flatnonzero(codes == i)
        if len(local) == 0:
            continue
        p = proto[i].reshape(-1, proj.shape[1])
        out[local] = (-torch.cdist(q[local], p).min(1).values).cpu().numpy()
    return out


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    codes = task_codes(meta.task.to_numpy(str))
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    fit = frac <= 0.6
    window = (frac > 0.6) & (frac <= 0.8)
    blocks = np.clip((frac * 5).astype(int), 0, 4)
    train = np.load(TRAIN_E65).astype(np.float32)
    test = np.load(TEST_E65).astype(np.float32)
    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    s1 = tmeta.session.astype(str).to_numpy() == 'session1'

    # task standardisation and PCA exactly as E139 (PCA fitted on fit rows)
    mu = np.stack([train[codes == i].mean(0) for i in range(len(TASKS))])
    sd = np.stack([train[codes == i].std(0) + 1e-6 for i in range(len(TASKS))])
    zt = (train - mu[codes]) / sd[codes]
    zq = (test - mu[tcodes]) / sd[tcodes]
    centre = zt[fit].mean(0)
    u, _, vt = torch.linalg.svd(torch.as_tensor(zt[fit] - centre, device=DEVICE),
                                full_matrices=False)
    basis = vt[:96].T.contiguous().cpu().numpy()
    trp = ((zt - centre) @ basis).astype(np.float32)
    tep = ((zq - centre) @ basis).astype(np.float32)

    rows = []
    for k in KS:
        deployed = protos(trp, subjects, codes, np.ones(len(trp), bool), k)
        te = score(tep, tcodes, deployed)
        oof = np.full(len(trp), np.nan, dtype=np.float32)
        for b in range(5):
            hold = blocks == b
            p = protos(trp, subjects, codes, ~hold, k)
            s = score(trp, codes, p)
            oof[hold] = s[hold]
        contaminated, clean = [], []
        for held in range(1, 21):
            enrolled = np.setdiff1d(np.arange(1, 21), [held])
            pm = np.isin(subjects, enrolled)
            ki = pm & window
            ui = (subjects == held) & window
            if ui.sum() == 0:
                continue
            p_all = protos(trp, subjects, codes, pm, k)           # E139 rule
            p_fit = protos(trp, subjects, codes, pm & fit, k)     # clean rule
            s_all = score(trp, codes, p_all)
            s_fit = score(trp, codes, p_fit)
            contaminated.append((s_all[ki], s_all[ui]))
            clean.append((s_fit[ki], s_fit[ui]))
        for a in ACCEPT:
            th = float(np.nanquantile(oof, 1 - a))
            s2 = float(np.mean(te[~s1] >= th))
            s1v = float(np.mean(te[s1] < th))
            loso_c = float(np.mean([np.mean(u < np.quantile(kn, 1 - a)) for kn, u in contaminated]))
            loso_k = float(np.mean([np.mean(u < np.quantile(kn, 1 - a)) for kn, u in clean]))
            rows.append({'k': k, 'acceptance': a, 'test_s1_rejection': s1v,
                         'test_s2_acceptance': s2,
                         'loso_rejection_E139_rule': loso_c,
                         'loso_rejection_clean': loso_k})
            print(f"k={k:2d} a={a:.2f} S1={s1v:.4f} S2={s2:.4f} "
                  f"LOSO(E139)={loso_c:.4f} LOSO(clean)={loso_k:.4f}", flush=True)
    (OUT / 'results.json').write_text(json.dumps(
        {'experiment': 'E210_loso_contamination_check',
         'feature': 'E65 logSPD 855 -> PCA96 (E139 best configuration)',
         'note': 'E137/E139/E141 build the LOSO known threshold from rows that are inside '
                 'the prototype, which inflates LOSO rejection',
         'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
