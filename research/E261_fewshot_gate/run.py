"""E261: Train-only 20-shot enrollment, leaving the E254 old branch intact."""
import json
from pathlib import Path
import numpy as np
import torch
import e259_reference as ref

HERE = Path(__file__).resolve().parent
y, t, f = ref.y, ref.t, ref.f


def fewshot(q, people):
    tasks = []
    for i in range(4):
        protos = []
        for person in people:
            idx = torch.where((y == person) & (t == i) & (f <= .6))[0]
            idx = idx[ref.order[idx].argsort()[:5]]
            assert len(idx) == 5
            protos.append(q[idx].mean(0))
        tasks.append(torch.stack(protos))
    return torch.nn.functional.normalize(torch.stack(tasks), dim=-1)


def cosine(q, rows, proto):
    xx = torch.nn.functional.normalize(q[rows]); tt = t[rows]
    out = xx.new_empty(len(xx), proto.shape[1])
    for i in range(4):
        out[tt == i] = xx[tt == i] @ proto[i].T
    return out.cpu().numpy()


def decide(old_score, old_id, old_tau, new_scores=None, new=None, new_tau=None):
    if new_scores is None:
        return old_score, old_id, old_score >= old_tau
    j = new_scores.argmax(1)
    # Only the new candidate receives the offset. Keep the original old score scale.
    candidate = new_scores[np.arange(len(j)), j] + (old_tau - new_tau)
    wins = candidate > old_score
    score = np.where(wins, candidate, old_score)
    return score, np.where(wins, np.asarray(new)[j], old_id), score >= old_tau


def metrics(s, pred, accept, truth, old, new):
    o, n = np.isin(truth, old), np.isin(truth, new); k = o | n
    return dict(old_e2e=np.mean((pred[o] == truth[o]) & accept[o]),
                new_e2e=np.mean((pred[n] == truth[n]) & accept[n]),
                unknown=np.mean(~accept[~k]), new_closed=np.mean(pred[n] == truth[n]),
                AUC=ref.roc_auc_score(k, s), F1=ref.f1_score(truth[k], pred[k], 'macro'))


@torch.no_grad()
def fold(h):
    a = ref.ids[h:] + ref.ids[:h]; old, new, unknown = a[:14], a[14:17], a[17:]
    oldrows = torch.isin(y, ref.tens(old)); fit = oldrows & (f <= .6)
    cal = oldrows & (f > .6) & (f <= .8); ev = f > .8
    q, art = ref.project(fit); p, v = ref.templates(q, fit, old)
    cb, _ = ref.score(q, p, v, cal, old); tau = float(np.quantile(cb[0], .1))
    # All thresholds frozen before scoring the late segment. Each pseudo-identity
    # contributes its own genuine trials and 13 other old-identity impostors.
    pseudo = cosine(q, cal, fewshot(q, old)); cy = y[cal].cpu().numpy()
    ct = t[cal].cpu().numpy(); match = cy[:, None] == np.array(old)[None, :]
    thresholds = []
    for i in range(4):
        task = (ct == i)[:, None]
        thresholds.append(float(max(np.quantile(pseudo[task & match], .1),
                                    np.quantile(pseudo[task & ~match], .99))))
    proto = fewshot(q, new)
    ob, oid = ref.score(q, p, v, ev, old); os = ob[0]
    empty = decide(os, oid, tau)
    for actual, expected in zip(empty, (os, oid, os >= tau)):
        np.testing.assert_array_equal(actual, expected)
    s, pred, accept = decide(os, oid, tau, cosine(q, ev, proto), new,
                             np.asarray(thresholds)[t[ev].cpu().numpy()])
    truth = y[ev].cpu().numpy(); result = metrics(s, pred, accept, truth, old, new)
    # Paired E259 fixed-rule control (same projection, rows, trial order).
    en = fit.clone()
    for person in new:
        for i in range(4):
            ix = torch.where((y == person) & (t == i) & (f <= .6))[0]
            en[ix[ref.order[ix].argsort()[:5]]] = True
    bp, bv = ref.templates(q, en, old + new)
    bc, _ = ref.score(q, bp, bv, cal, old + new); bs, bi = ref.score(q, bp, bv, ev, old + new)
    control = metrics(bs[0], bi, bs[0] >= np.quantile(bc[0], .1), truth, old, new)
    state = {k: val.cpu() for k, val in art.items()}
    state.update(old=old, new=new, old_proto=p.cpu(), valid=v.cpu(), new_proto=proto.cpu(),
                 old_tau=tau, new_tau=thresholds,tasks=sorted(ref.M.task.unique().tolist()))
    return dict(fold=h+1, old=old, new=new, unknown_ids=unknown, no_new_equal_rows=len(os),
                old_tau=tau, new_tau=thresholds, E261=result, E259=control), state


if __name__ == '__main__':
    results, weights = [], []
    for h in range(20):
        row, state = fold(h); results.append(row); weights.append(state)
    torch.save(weights, HERE/'weights.pt')
    summary = {}
    for rule in ['E259','E261']:
        summary[rule] = {}
        for k in results[0][rule]:
            x = [r[rule][k] for r in results]; j = int(np.argmin(x))
            summary[rule][k] = dict(mean=float(np.mean(x)), worst=float(x[j]), fold=j+1)
    (HERE/'results.json').write_text(json.dumps(dict(summary=summary, folds=results), indent=2))
    print(json.dumps(summary)); print('equal rows:',sum(r['no_new_equal_rows'] for r in results))
