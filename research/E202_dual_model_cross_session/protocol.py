from __future__ import annotations
import numpy as np

def fit_mask(meta, subjects, enrolled, frac_max=0.6):
    return np.isin(subjects, enrolled) & (meta['frac'].to_numpy() <= frac_max)

def eval_mask(meta, subjects, ids, low=0.6, high=0.8):
    return np.isin(subjects, ids) & (meta['frac'].to_numpy() > low) & (meta['frac'].to_numpy() <= high)

def fit_projection(x, fit_rows, n_components=96):
    mu = x[fit_rows].mean(0)
    sd = x[fit_rows].std(0)
    sd[sd < 1e-8] = 1.0
    xs = (x - mu) / sd
    centered = xs[fit_rows]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n = min(n_components, x.shape[1], max(1, int(fit_rows.sum()) - 1))
    basis = vt[:n].T
    projected = (xs @ basis).astype(np.float32)
    return (mu, sd, basis), None, projected

def build_prototypes(z, subjects, rows, enrolled):
    proto=[]; ids=[]
    for sid in enrolled:
        take = rows & (subjects == sid)
        if not take.any():
            continue
        p = z[take].mean(0)
        p /= max(np.linalg.norm(p), 1e-8)
        proto.append(p); ids.append(int(sid))
    if not proto:
        raise ValueError('no enrollment prototypes')
    return np.stack(proto).astype(np.float32), np.asarray(ids, dtype=np.int64)

def sample_splits(subject_ids, k, seed=20261116):
    rng=np.random.default_rng(seed+k)
    ids=np.asarray(sorted(subject_ids),dtype=int)
    if k == len(ids):
        return [ids]
    reps=min(20, max(1, len(ids)))
    out=[]
    for _ in range(reps): out.append(np.sort(rng.choice(ids,size=k,replace=False)))
    unique=[]
    for a in out:
        if not any(np.array_equal(a,b) for b in unique): unique.append(a)
    return unique

def evaluate_registration(z, subjects, rows, enrolled, unknown_ids):
    proto, proto_ids = build_prototypes(z, subjects, rows, enrolled)
    s=z @ proto.T
    ix=np.argmax(s,axis=1); best=s[np.arange(len(z)),ix]
    pred=proto_ids[ix]
    known=np.isin(subjects,enrolled); unknown=np.isin(subjects,unknown_ids)
    return {'proto':proto,'proto_ids':proto_ids,'best':best,'pred':pred,'known':known,'unknown':unknown,'similarity':s}
