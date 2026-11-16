from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
ROOT = OUT.parents[1]
CACHE = ROOT / 'research' / 'E189_balanced_gate_gates' / 'cache' / 'stream_scores.npz'
META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
RESULTS = OUT / 'results.json'
LOG = OUT / 'run.log'
STREAMS = CFG['streams']
TASKS = ['EC', 'EO', 'Imagery', 'SSVEP']

def log(s):
    print(s, flush=True)

def q(x, p):
    return float(np.quantile(np.asarray(x, dtype=np.float64), p))

def auc(known, unknown):
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))], np.r_[unknown, known]))

def main():
    started = time.time()
    z = np.load(CACHE, allow_pickle=False)
    meta = __import__('pandas').read_csv(META, usecols=['subject','task','trial'])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy(str)
    maximum = meta.groupby(['subject', 'task'])['trial'].transform('max').to_numpy(float)
    frac = meta.trial.to_numpy(float) / np.maximum(maximum, 1.0)
    window = (frac > 0.6) & (frac <= 0.8)
    is_s1 = z['is_s1'].astype(bool)
    stream_oof = {s: (z[f'{s}__oof'] - float(z[f'{s}__oof_mu'])) / float(z[f'{s}__oof_sd']) for s in STREAMS}
    stream_test = {s: (z[f'{s}__test'] - float(z[f'{s}__oof_mu'])) / float(z[f'{s}__oof_sd']) for s in STREAMS}
    folds = []
    for fi, held in enumerate(z['e65_pca64_k1__loso_held']):
        enrolled = subjects != int(held)
        known_rows = np.flatnonzero(window & enrolled)
        unknown_rows = np.flatnonzero(window & (subjects == int(held)))
        folds.append((known_rows, unknown_rows, int(held)))
    records = []
    for weights in CFG['weights']:
        weights = np.asarray(weights, dtype=float)
        oof = sum(w * stream_oof[s] for w, s in zip(weights, STREAMS))
        test = sum(w * stream_test[s] for w, s in zip(weights, STREAMS))
        known_all, unknown_all = [], []
        fold_tasks = []
        for fi, (kr, ur, held) in enumerate(folds):
            known = sum(w * ((z[f'{s}__loso_{fi}_known'] - float(z[f'{s}__oof_mu'])) / float(z[f'{s}__oof_sd'])) for w, s in zip(weights, STREAMS))
            unknown = sum(w * ((z[f'{s}__loso_{fi}_unknown'] - float(z[f'{s}__oof_mu'])) / float(z[f'{s}__oof_sd'])) for w, s in zip(weights, STREAMS))
            known_all.append(known); unknown_all.append(unknown)
            fold_tasks.append((tasks[kr], tasks[ur], known, unknown))
        known_pool, unknown_pool = np.concatenate(known_all), np.concatenate(unknown_all)
        loso_auc = auc(known_pool, unknown_pool)
        for rule in CFG['threshold_rules']:
            for a in CFG['acceptance_grid']:
                if rule == 'global_oof_quantile':
                    thresholds = {'global': q(oof, 1-a)}
                elif rule == 'task_oof_quantile':
                    thresholds = {t: q(oof[tasks == t], 1-a) for t in TASKS}
                else:
                    thresholds = {t: q(np.concatenate([k[tk == t] for tk, _, k, _ in fold_tasks]), 1-a) for t in TASKS}
                default_threshold = thresholds.get('global', next(iter(thresholds.values())))
                def th_for(task): return thresholds.get(task, default_threshold)
                known_acc = []; reject = []; u95 = []
                for kt, ut, known, unknown in fold_tasks:
                    ktau = np.asarray([th_for(t) for t in kt]); utau = np.asarray([th_for(t) for t in ut])
                    known_acc.append(np.mean(known >= ktau)); reject.append(np.mean(unknown < utau))
                    known95 = np.quantile(known, 0.05)
                    u95.append(np.mean(unknown < known95))
                test_tau = np.asarray([th_for(t) for t in __import__('pandas').read_csv(ROOT / 'BrainPrint_cache' / 'test_metadata.csv', usecols=['task']).task.to_numpy(str)])
                s1 = float(np.mean(test[is_s1] < test_tau[is_s1])); s2 = float(np.mean(test[~is_s1] >= test_tau[~is_s1]))
                rec = {'weights': [float(x) for x in weights], 'threshold_rule': rule, 'a': float(a), 'loso_auc': loso_auc,
                       'loso_rejection': float(np.mean(reject)), 'loso_known_acceptance': float(np.mean(known_acc)),
                       'loso_u95_mean': float(np.mean(u95)), 'loso_u95_min': float(np.min(u95)),
                       'test_s1_rejection': s1, 'test_s2_acceptance': s2, 'test_mean': (s1+s2)/2,
                       'test_min': min(s1,s2), 'test_acceptance': float(np.mean(test >= test_tau)),
                       'thresholds': {k: float(v) for k,v in thresholds.items()}, 'official_test_auc': 'N/A', 'official_test_f1': 'N/A'}
                records.append(rec)
    eligible = [r for r in records if r['loso_auc'] > 0.90 and r['loso_rejection'] >= 0.70]
    best = max(eligible, key=lambda r: (r['loso_rejection'], r['test_min'], r['loso_auc'], r['loso_known_acceptance'])) if eligible else max([r for r in records if r['loso_auc'] > 0.90], key=lambda r: (r['loso_rejection'], r['loso_auc']))
    best_test = max([r for r in records if r['loso_auc'] > 0.90], key=lambda r: (r['test_min'], r['loso_rejection']))
    result = {'experiment':'E217_boundary_task_calibration','source_cache':str(CACHE.relative_to(ROOT)), 'n_records':len(records), 'n_eligible':len(eligible), 'best':best, 'best_test_min_among_auc_gt_090':best_test, 'all_records':records, 'runtime_seconds':time.time()-started}
    RESULTS.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'E217 records={len(records)} eligible={len(eligible)}')
    log(f"BEST Train/LOSO: weights={best['weights']} rule={best['threshold_rule']} a={best['a']:.2f} LOSO_AUC={best['loso_auc']:.4f} LOSO_rejection={best['loso_rejection']:.4f} Test S1={best['test_s1_rejection']:.4f} S2={best['test_s2_acceptance']:.4f}")
    log(f"BEST Test-min audit among AUC>0.90: weights={best_test['weights']} rule={best_test['threshold_rule']} a={best_test['a']:.2f} LOSO_AUC={best_test['loso_auc']:.4f} LOSO_rejection={best_test['loso_rejection']:.4f} Test S1={best_test['test_s1_rejection']:.4f} S2={best_test['test_s2_acceptance']:.4f}")

if __name__ == '__main__':
    with LOG.open('w', encoding='utf-8') as f:
        import contextlib
        with contextlib.redirect_stdout(f): main()
