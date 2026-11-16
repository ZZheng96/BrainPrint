"""E203: Train-only task-wise LOSO gate on frozen E200 subwin6 scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E189 = ROOT / 'research' / 'E189_balanced_gate_gates' / 'cache' / 'stream_scores.npz'
E200 = ROOT / 'research' / 'E200_final_freeze' / 'cache' / 'reference_test.npz'
META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
OUT = Path(__file__).resolve().parent
TASKS = ['EC', 'EO', 'Imagery', 'SSVEP']


def threshold(known, unknown, target=0.90):
    grid = np.unique(np.concatenate([known, unknown]))
    best = None
    for q in grid:
        ka = float(np.mean(known >= q))
        ur = float(np.mean(unknown < q))
        if ur >= target and (best is None or (ka, q) > (best[0], best[1])):
            best = (ka, q, ur)
    if best is None:
        q = float(np.quantile(known, 1 - target))
        best = (float(np.mean(known >= q)), q, float(np.mean(unknown < q)))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(E189)
    ref = np.load(E200)
    meta = pd.read_csv(META)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy(str)
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    fit = frac <= 0.6
    window = (frac > 0.6) & (frac <= 0.8)
    key = 'subwin6_pca96_k1__'
    oof = z[key + 'oof'].astype(float)
    test = z[key + 'test'].astype(float)
    # The cached score is negative distance; higher means more likely registered.
    thresholds = {}
    loso_summary = {}
    for task in TASKS:
        known = oof[window & (tasks == task)]
        unknown_parts = []
        held = z[key + 'loso_held'].astype(int)
        for i, sid in enumerate(held):
            arr = z[key + f'loso_{i}_unknown'].astype(float)
            # E189 stores the held-out unknown evaluation window (0.6<frac<=0.8)
            # in original metadata order.
            task_rows = np.flatnonzero((subjects == sid) & (window))
            ordered = tasks[task_rows]
            pos = np.flatnonzero(ordered == task)
            if len(pos) and pos.max() < len(arr):
                unknown_parts.append(arr[pos])
            elif len(pos):
                # Stored arrays may omit the first/last task rows; use the
                # deterministic task block counts from the window ordering.
                counts = [int(np.sum(ordered == name)) for name in TASKS]
                start = int(sum(counts[:TASKS.index(task)]))
                unknown_parts.append(arr[start:start + counts[TASKS.index(task)]])
        unknown = np.concatenate(unknown_parts)
        ka, q, ur = threshold(known, unknown)
        thresholds[task] = q
        loso_summary[task] = {'known_acceptance': ka, 'unknown_rejection': ur,
                              'known_rows': int(len(known)), 'unknown_rows': int(len(unknown))}
    ttask = ref['task'].astype(str)
    session = ref['session'].astype(str)
    accept = np.array([test[i] >= thresholds[ttask[i]] for i in range(len(test))])
    s1 = session == 'session1'
    result = {
        'experiment': 'E203_taskwise_loso_gate',
        'protocol': 'thresholds selected from Train OOF known and strict LOSO unknown, target LOSO rejection 0.90; Test only audited after decisions',
        'thresholds': thresholds,
        'loso_by_task': loso_summary,
        'loso_rejection_mean': float(np.mean([v['unknown_rejection'] for v in loso_summary.values()])),
        'test_s1_rejection': float(np.mean(~accept[s1])),
        'test_s2_acceptance': float(np.mean(accept[~s1])),
        'test_rows': int(len(test)), 's1_rows': int(s1.sum()), 's2_rows': int((~s1).sum()),
        'test_supervised_metrics': 'N/A: Test identities unavailable',
    }
    (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'predictions.npz', accept=accept, task=ttask, session=session,
                        score=test, thresholds=np.array([thresholds[t] for t in TASKS]),
                        sample=ref['sample'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
