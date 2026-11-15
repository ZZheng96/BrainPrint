import json
import numpy as np

r = json.load(open('research/E108_loso_s1_boost/results.json'))
print('LOSO: 19 identities enrolled, 1 held out as unknown')
print('Threshold fixed so enrolled acceptance = 0.85')
print()
print('held  enrolled_acc  heldout_rej  N_heldout')
for f in sorted(r['folds'], key=lambda x: x['held_out']):
    s = f['signals']['base_ratio']
    print(f"{f['held_out']:4d}  {s['known_acceptance']:.3f}         {s['unknown_rejection']:.3f}        {f['N_unknown']}")
vals = [f['signals']['base_ratio']['unknown_rejection'] for f in r['folds']]
print()
print('mean heldout rejection:', round(np.mean(vals), 4))
print('folds where heldout rejected >= 0.80:', sum(1 for v in vals if v >= 0.80), '/20')
print('folds where heldout rejected >= 0.50:', sum(1 for v in vals if v >= 0.50), '/20')
