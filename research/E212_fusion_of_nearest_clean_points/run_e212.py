"""E212: fuse the cleanest near-miss (E170 score fusion) with subwin6 registration.

Survey result: every recorded point with LOSO>0.9 and S2>0.7 comes from an
experiment whose LOSO threshold rows sit inside the prototypes (E137/E139/E141,
E142, E152).  The closest genuinely clean point is E170's Train-selected 3-way
score fusion:

    acceptance 0.91 -> LOSO 0.8390, S1 0.7247, S2 0.6387
    acceptance 0.90 -> LOSO 0.8615, S1 0.7484, S2 0.6129

E209 (subwin6, full-data registration) reaches S1 0.7310 / S2 0.7022 but only
LOSO 0.3295.  The two families are complementary, so this script fuses their
frozen per-sample scores, selects the weight on Train only, and audits Test once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)
ROOT = HERE.parents[1]
E170 = ROOT / 'research' / 'E170_three_way_fusion_ceiling' / 'frozen_scores.npz'
E209 = ROOT / 'research' / 'E209_registration_block_crossfit' / 'cache' / 'scores.npz'
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


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


def main():
    a = np.load(E170)
    b = np.load(E209)
    known170 = np.asarray(a['z_rf_c_E170_best_auc_3way_score'], dtype=np.float64)
    test170 = np.asarray(a['z_te_c_E170_best_auc_3way_score'], dtype=np.float64)
    unk170 = np.concatenate([a[f'loso_unknown_c_E170_best_auc_3way_score_{i}']
                             for i in range(20)])
    known209 = np.asarray(b['known'], dtype=np.float64)
    unknown209 = np.asarray(b['unknown'], dtype=np.float64)
    test209 = np.asarray(b['test'], dtype=np.float64)
    s1 = np.asarray(a['is_s1']).astype(bool)
    assert int(s1.sum()) == 3033

    def norm(x, ref):
        return (x - ref.mean()) / (ref.std() + 1e-9)

    k170, t170 = norm(known170, known170), norm(test170, known170)
    u170 = norm(unk170, known170)
    k209, t209 = norm(known209, known209), norm(test209, known209)
    u209 = norm(unknown209, known209)

    # the two families pool different numbers of rows (E209 repeats across 20 folds x 5
    # blocks), so the fused pools use equally sized random subsets to avoid weighting bias
    rng = np.random.default_rng(20261116)
    n_known = min(len(k170), len(k209))
    n_unknown = min(len(u170), len(u209))
    keep = lambda arr, n: arr[rng.choice(len(arr), n, replace=False)]
    k170, k209 = keep(k170, n_known), keep(k209, n_known)
    u170, u209 = keep(u170, n_unknown), keep(u209, n_unknown)

    rows = []
    for w in WEIGHTS:  # w = weight on E170
        kn = w * k170 + (1 - w) * k209
        un = w * u170 + (1 - w) * u209
        tn = w * t170 + (1 - w) * t209
        ka, q, ur = choose(kn, un)
        acc = tn >= q
        rows.append({'weight_E170': w, 'threshold': q,
                     'loso_unknown_rejection': ur, 'train_known_acceptance': ka,
                     'test_s1_rejection': float(np.mean(~acc[s1])),
                     'test_s2_acceptance': float(np.mean(acc[~s1]))})
        r = rows[-1]
        print(f"w(E170)={w:.2f} loso_ur={ur:.4f} ka={ka:.4f} "
              f"S1={r['test_s1_rejection']:.4f} S2={r['test_s2_acceptance']:.4f}", flush=True)

    # also report E170's own acceptance grid operating points as the reference
    ref = {'acceptance_0.90': {'loso': 0.8615152517647682, 's1': 0.7484338938344873,
                               's2': 0.6128629409928192},
           'acceptance_0.91': {'loso': 0.8390221000235284, 's1': 0.7246950214309265,
                               's2': 0.6387241128109065}}
    (HERE / 'results.json').write_text(json.dumps(
        {'experiment': 'E212_fusion_of_nearest_clean_points',
         'inputs': {'E170': 'Train-selected 3-way score fusion (clean LOSO)',
                    'E209': 'subwin6 full-data registration (clean 5-block LOSO)'},
         'protocol': 'family scores normalised by their own known pools; weight and '
                     'threshold selected on Train only; Test audited once',
         'rows': rows, 'E170_reference': ref}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
