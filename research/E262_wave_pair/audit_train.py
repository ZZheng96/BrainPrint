"""E254 gate #1: frozen submission weights, cached Train features, no fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / 'final2/team_hymn'
CACHE = ROOT / 'BrainPrint_cache'
sys.path.insert(0, str(PKG))
import model as M


@torch.inference_mode()
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    art = M.load_model(str(PKG), dev)
    meta = pd.read_csv(CACHE / 'train_metadata.csv')
    x = np.load(CACHE / 'e253_subwindow/train_feat.npy', mmap_mode='r')
    tasks = M._task_indices(meta[['task']].to_dict('records'), len(meta))
    assert len(x) == len(meta) and (tasks >= 0).all()
    sim = torch.empty(len(meta), M.N_BASE, device=dev)
    for t in range(len(M.TASKS)):
        p = art['prototypes'][t, :M.N_BASE]
        p = M._unit(p @ art['whiten'])
        indices = np.flatnonzero(tasks == t)
        for start in range(0, len(indices), 256):
            rows = indices[start:start + 256]
            feat = torch.from_numpy(x[rows].astype(np.float32).mean(1)).to(dev)
            q = M._unit(M._project(art, feat, t) @ art['whiten'])
            sim[rows] = (q[:, None, None] * p).sum(-1).amax(-1)

    assert torch.isfinite(sim).all()
    best, second = sim.topk(2).values.T
    margin = best - second
    gate = best + art['beta'] * margin
    reject = (gate < art['tau']).cpu().numpy()
    pred = sim.argmax(1).cpu().numpy() + 1
    truth = meta.subject.to_numpy()
    ids = np.arange(1, M.N_BASE + 1)
    assert np.array_equal(np.unique(truth), ids)
    correct = pred == truth
    f1 = [2 * ((truth == c) & correct).sum() / ((truth == c).sum() + (pred == c).sum()) for c in ids]
    errors = []
    for i in np.flatnonzero(~correct):
        row = meta.iloc[i]
        errors.append(dict(sample=row.path, task=row.task, true_id=int(truth[i]),
                           closed_pred_id=int(pred[i]), s1=float(best[i]), s2=float(second[i]),
                           top1_minus_top2=float(margin[i]), top1_minus_true=float(best[i]-sim[i,truth[i]-1]),
                           gate_score=float(gate[i]), rejected=bool(reject[i]),
                           end_to_end_id=0 if reject[i] else int(pred[i])))
    report = dict(n_train=len(meta), open_set_auc=None, auc_reason='Train has no unregistered negatives.',
                  closed_set=dict(correct=int(correct.sum()), accuracy=float(correct.mean()), macro_f1=float(np.mean(f1))),
                  frozen_gate=dict(beta=art['beta'], tau=art['tau'], rejected=int(reject.sum()),
                                   accepted_wrong=int((~reject & ~correct).sum()), correct=int((~reject & correct).sum()),
                                   end_to_end_accuracy=float((~reject & correct).mean())), closed_set_errors=errors)
    text = json.dumps(report, indent=2, allow_nan=False)
    (HERE / 'train_errors.json').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
