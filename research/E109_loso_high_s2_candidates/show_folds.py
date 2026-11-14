import json
r = json.load(open('research/E109_loso_high_s2_candidates/results.json'))
print('held  N_unk   E95    E97    E98    E104')
for f in r['folds']:
    s = f['signals']
    print(f"{f['held_out']:4d}  {f['N_unknown']:5d}  "
          f"{s['E95_raw_cosine']['unknown_rejection']:.3f}  "
          f"{s['E97_multiview']['unknown_rejection']:.3f}  "
          f"{s['E98_dual_gate']['unknown_rejection']:.3f}  "
          f"{s['E104_session_invariant']['unknown_rejection']:.3f}")
