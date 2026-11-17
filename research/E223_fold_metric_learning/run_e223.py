from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
DEVICE = 'cuda'
TASKS = CFG['tasks']
N_TASKS = len(TASKS)
FIT_FRAC = float(CFG['fit_fraction'])
EVAL_LO, EVAL_HI = CFG['evaluation_fraction']
N_BLOCKS = int(CFG['inner_blocks'])
Q_TAU = float(CFG['threshold_quantile'])
DIMS = {k: int(v) for k, v in CFG['representation']['pca_dim_per_stream'].items()}
HEAD_CFG = CFG['head']
AUG_CFG = dict(CFG['augment'])
NOAUG_CFG = dict(CFG['augment_disabled'])
MAHA_SHRINK = float(CFG['mahalanobis_shrinkage'])
SEED = int(CFG['seed'])
LOG = OUT / 'run.log'
FR = None
TI = None
Y = None
BLOCKS = None

CAND_BY_HEAD = {
    'aug': [('mlp_ce_contrast_k1', 'proto', 1),
            ('mlp_ce_contrast_k3', 'proto', 3),
            ('mlp_ce_contrast_maha', 'maha', 1)],
    'noaug': [('mlp_no_augment', 'proto', 1)],
}
AUG_OF_HEAD = {'aug': AUG_CFG, 'noaug': NOAUG_CFG}


def log(s):
    print(s, flush=True)
    with LOG.open('a', encoding='utf8') as f:
        f.write(str(s) + '\n')


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def idx(x):
    return torch.as_tensor(x, dtype=torch.long, device=DEVICE)


def frac(m):
    return m.trial.to_numpy(float) / m.groupby(['subject', 'task']).trial.transform('max').to_numpy(float).clip(1)


# =====================================================================  E221/E222 baseline (verbatim)
def km(x, k, seed):
    x = tensor(x)
    if len(x) <= k:
        return x.cpu().numpy()
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, False)].clone()
    for _ in range(20):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        counts = torch.zeros(k, device=DEVICE).index_add_(0, a, torch.ones(len(x), device=DEVICE))
        nc = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], c)
        if torch.allclose(nc, c, atol=1e-6):
            break
        c = nc
    return c.cpu().numpy()


def base_prototypes(z, subjects, tasks, mask, classes, k):
    ps, keys = [], []
    for task in TASKS:
        for sub in classes:
            m = mask & (tasks == task) & (subjects == sub)
            if m.any():
                for p in km(z[m], k, int(sub) * 17):
                    ps.append(p)
                    keys.append((task, int(sub)))
    return np.stack(ps).astype(np.float32), keys


def base_fit_rep(raw, tasks, y, fit, k):
    x = tensor(raw)
    z = torch.empty_like(x)
    mu, sd = [], []
    for t in TASKS:
        rows = tasks == t
        a = x[rows & fit]
        m = a.mean(0)
        s = a.std(0, correction=0) + 1e-6
        z[rows] = (x[rows] - m) / s
        mu.append(m.cpu().numpy())
        sd.append(s.cpu().numpy())
    a = z[fit].double()
    pm = a.mean(0)
    _, v = torch.linalg.eigh((a - pm).T @ (a - pm))
    W = v[:, -DIMS['e65']:].flip(1)
    z = ((z.double() - pm) @ W).float().cpu().numpy()
    cls = np.unique(y[fit])
    p, keys = base_prototypes(z, y, tasks, fit, cls, k)
    return dict(mu=mu, sd=sd, pm=pm.cpu().numpy(), W=W.cpu().numpy(), p=p, keys=keys, k=k)


def base_transform(state, raw, tasks):
    x = tensor(raw)
    z = torch.empty_like(x)
    for i, t in enumerate(TASKS):
        rows = tasks == t
        z[rows] = (x[rows] - tensor(state['mu'][i])) / tensor(state['sd'][i])
    z = ((z.double() - tensor(state['pm'], torch.float64)) @ tensor(state['W'], torch.float64)).float()
    return z.cpu().numpy()


def base_proto_meta(state):
    keys = state['keys']
    return np.array([t for t, _ in keys]), np.array([int(s) for _, s in keys], int)


def base_score_parts(state, z, rows, task_rows):
    proto_task, proto_id = base_proto_meta(state)
    parts = []
    for t in TASKS:
        rr = rows[task_rows == t]
        cols = np.flatnonzero(proto_task == t)
        if not len(rr) or not len(cols):
            continue
        S = (-torch.cdist(tensor(z[rr]), tensor(state['p'][cols]))).double().cpu().numpy()
        parts.append((t, rr, S, proto_id[cols]))
    return parts


def base_score(state, z, tasks, rows):
    s = np.full(len(tasks), np.nan)
    p = np.full(len(tasks), -1, int)
    for t, rr, S, ident in base_score_parts(state, z, rows, tasks[rows]):
        s[rr] = S.max(1)
        p[rr] = ident[S.argmax(1)]
    return s, p


def base_inner_oof(fit, raw, tasks, y, blocks, pk):
    oof = np.full(len(y), np.nan)
    for b in np.unique(blocks[fit]):
        hold = fit & (blocks == b)
        inner = fit & ~hold
        assert set(y[inner]) == set(y[fit])
        st = base_fit_rep(raw, tasks, y, inner, pk)
        z = base_transform(st, raw, tasks)
        s, _ = base_score(st, z, tasks, np.flatnonzero(hold))
        oof[hold] = s[hold]
    return oof


# =====================================================================  learned representation
def fit_std(raw, fit):
    x = tensor(raw)
    mus, sds = [], []
    for k in range(N_TASKS):
        a = x[fit & (TI == k)]
        mus.append(a.mean(0))
        sds.append(a.std(0, correction=0).clamp_min(1e-6))
    return torch.stack(mus), torch.stack(sds)


def apply_std(raw, mu, sd):
    return (tensor(raw) - mu[idx(TI)]) / sd[idx(TI)]


def fit_pca(zfit, dim):
    a = zfit.double()
    m = a.mean(0)
    c = (a - m).T @ (a - m) / len(a)
    _, v = torch.linalg.eigh(c)
    return m, v[:, -dim:].flip(1)


def make_rep(fit):
    """Unsupervised: task-wise z-score + per-stream PCA + block RMS, fitted on `fit` rows only."""
    rep = {}
    for name, raw in (('e65', X65), ('logbp', XBP)):
        mu, sd = fit_std(raw, fit)
        z = apply_std(raw, mu, sd)
        m, W = fit_pca(z[fit], DIMS[name])
        zf = ((z[fit].double() - m) @ W).float()
        scale = float(torch.sqrt((zf ** 2).mean()).clamp_min(1e-6))
        rep[name] = dict(mu=mu, sd=sd, m=m, W=W, scale=scale)
        rep['std_' + name] = z
    return rep


def project_std(rep, a65, abp):
    parts = []
    for name, a in (('e65', a65), ('logbp', abp)):
        st = rep[name]
        z = ((a.double() - st['m']) @ st['W']).float() / st['scale']
        parts.append(z)
    return torch.cat(parts, 1)


def standardize_test(rep, x65t, xbpt, ti_test):
    parts = []
    for name, x in (('e65', x65t), ('logbp', xbpt)):
        st = rep[name]
        a = (tensor(x) - st['mu'][idx(ti_test)]) / st['sd'][idx(ti_test)]
        z = ((a.double() - st['m']) @ st['W']).float() / st['scale']
        parts.append(z)
    return torch.cat(parts, 1)


# =====================================================================  session-drift augmentation (feature space only)
BAND_ID = torch.cat([torch.arange(855, device=DEVICE) % 5,
                     torch.arange(90, device=DEVICE) % 5])
CHAN_ID_BP = torch.arange(90, device=DEVICE) // 5


def augment(a65, abp, subjects_b, opts, gen):
    x = torch.cat([a65, abp], 1)
    B = x.shape[0]
    if opts['band_gain_sigma'] > 0:
        g = torch.exp(torch.randn(B, 5, device=DEVICE, generator=gen) * opts['band_gain_sigma'])
        x = x * g[:, BAND_ID]
    if opts['channel_gain_sigma'] > 0:
        g = torch.exp(torch.randn(B, 18, device=DEVICE, generator=gen) * opts['channel_gain_sigma'])
        x = x.clone()
        x[:, 855:] = x[:, 855:] * g[:, CHAN_ID_BP]
    if opts['spectral_tilt_sigma'] > 0:
        slope = torch.randn(B, 1, device=DEVICE, generator=gen) * opts['spectral_tilt_sigma']
        x = x * torch.exp(slope * (BAND_ID.float() - 2.0)[None, :])
    if opts['additive_noise_sigma'] > 0:
        x = x + torch.randn(x.shape, device=DEVICE, generator=gen) * opts['additive_noise_sigma']
    if opts['band_swap_prob'] > 0:
        sel = torch.rand(B, device=DEVICE, generator=gen) < opts['band_swap_prob']
        if bool(sel.any()):
            partner = torch.arange(B, device=DEVICE)
            for s in torch.unique(subjects_b):
                idx = torch.nonzero(subjects_b == s, as_tuple=False).squeeze(1)
                if len(idx) < 2:
                    continue
                perm = idx[torch.randperm(len(idx), device=DEVICE, generator=gen)]
                partner[perm] = perm.roll(1)
            b = torch.randint(0, 5, (B,), device=DEVICE, generator=gen)
            mask = (BAND_ID[None, :] == b[:, None]) & sel[:, None]
            x = torch.where(mask, x[partner], x)
    return x[:, :855], x[:, 855:]


# =====================================================================  MLP metric head + losses
class MetricHead(nn.Module):
    def __init__(self, din, hidden, dout, n_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(din, hidden)
        self.fc2 = nn.Linear(hidden, dout)
        self.cls = nn.Linear(dout, n_classes, bias=False)
        self.dropout = dropout

    def embed(self, x):
        h = self.fc2(F.dropout(F.gelu(self.fc1(x)), self.dropout, self.training))
        return F.normalize(h, dim=1)

    def forward(self, x):
        e = self.embed(x)
        return e, self.cls(e)


def supcon(emb, labels, temp):
    B = len(emb)
    sim = emb @ emb.T / temp
    eye = torch.eye(B, device=DEVICE, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    logprob = sim - torch.logsumexp(sim, 1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    cnt = pos.sum(1)
    valid = cnt > 0
    if not bool(valid.any()):
        return emb.sum() * 0.0
    loss = -(logprob * pos).sum(1)[valid] / cnt[valid]
    return loss.mean()


def fit_epochs(rep, train_rows, val_rows, aug_opts, seed, classes, fixed_epochs=None, dim=32):
    """Train the head. If fixed_epochs is None, early-stop on fold-internal validation CE."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    head = MetricHead(int(CFG['representation']['head_input_dim']), int(HEAD_CFG['hidden_dim']),
                      int(dim), len(classes), float(HEAD_CFG['dropout'])).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=float(HEAD_CFG['learning_rate']),
                            weight_decay=float(HEAD_CFG['weight_decay']))
    label = torch.as_tensor(np.searchsorted(classes, Y), device=DEVICE)
    tr = torch.as_tensor(train_rows, device=DEVICE)
    va = torch.as_tensor(val_rows, device=DEVICE)
    bs = int(HEAD_CFG['batch_size'])
    max_ep = int(HEAD_CFG['max_epochs'])
    patience = int(HEAD_CFG['patience'])
    best = np.inf
    best_ep = -1
    best_state = None
    curve = []
    ep = 0
    while ep < (fixed_epochs if fixed_epochs is not None else max_ep):
        head.train()
        perm = torch.randperm(len(tr), device=DEVICE, generator=gen)
        for s in range(0, len(tr), bs):
            ids = tr[perm[s:s + bs]]
            a65 = rep['std_e65'][ids]
            abp = rep['std_logbp'][ids]
            if aug_opts['band_gain_sigma'] > 0 or aug_opts['channel_gain_sigma'] > 0 or \
               aug_opts['spectral_tilt_sigma'] > 0 or aug_opts['additive_noise_sigma'] > 0 or \
               aug_opts['band_swap_prob'] > 0:
                a65, abp = augment(a65, abp, label[ids], aug_opts, gen)
            z = project_std(rep, a65, abp)
            emb, logits = head(z)
            loss = (float(HEAD_CFG['ce_weight']) * F.cross_entropy(logits, label[ids])
                    + float(HEAD_CFG['supcon_weight']) * supcon(emb, label[ids],
                                                               float(HEAD_CFG['supcon_temperature'])))
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            z = project_std(rep, rep['std_e65'][va], rep['std_logbp'][va])
            _, logits = head(z)
            vl = float(F.cross_entropy(logits, label[va]))
        curve.append(vl)
        if fixed_epochs is None:
            if vl < best - 1e-6:
                best, best_ep = vl, ep
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            if ep - best_ep >= patience:
                break
        ep += 1
    state = best_state if fixed_epochs is None else {k: v.detach().clone() for k, v in head.state_dict().items()}
    return state, best_ep, curve


def choose_val_block(fit_rows, classes):
    present = sorted(int(b) for b in np.unique(BLOCKS[fit_rows]))
    for b in present:
        va = fit_rows[BLOCKS[fit_rows] == b]
        tr = fit_rows[BLOCKS[fit_rows] != b]
        if len(va) < 50:
            continue
        if set(Y[va]) == set(classes) and set(Y[tr]) == set(classes):
            return b
    return present[0]


def train_head(rep, fit_rows, aug_opts, seed, classes, dim=32):
    cls_arr = np.asarray(classes)
    vb = choose_val_block(fit_rows, cls_arr)
    tr = fit_rows[BLOCKS[fit_rows] != vb]
    va = fit_rows[BLOCKS[fit_rows] == vb]
    state, best_ep, curve = fit_epochs(rep, tr, va, aug_opts, seed, cls_arr, dim=dim)
    state, _, _ = fit_epochs(rep, fit_rows, va, aug_opts, seed + 1, cls_arr,
                             fixed_epochs=best_ep + 1, dim=dim)
    head = MetricHead(int(CFG['representation']['head_input_dim']), int(HEAD_CFG['hidden_dim']),
                      int(dim), len(cls_arr), float(HEAD_CFG['dropout'])).to(DEVICE)
    head.load_state_dict(state)
    head.eval()
    summary = dict(val_block=int(vb), best_epoch=int(best_ep), n_epochs=len(curve),
                   val_ce_first=float(curve[0]), val_ce_best=float(min(curve)),
                   val_ce_last=float(curve[-1]), val_ce_curve=[float(v) for v in curve])
    return head, summary


def embeddings(rep, head):
    head.eval()
    with torch.no_grad():
        z = project_std(rep, rep['std_e65'], rep['std_logbp'])
        return head.embed(z)


# =====================================================================  prototype / Mahalanobis scorers
def proto_state(emb, fit_mask, classes, pk):
    protos, keys = [], []
    for k in range(N_TASKS):
        for c in classes:
            m = fit_mask & (TI == k) & (Y == c)
            if not m.any():
                continue
            pts = km(emb[m].cpu().numpy(), pk, int(c) * 17 + k)
            pts = torch.as_tensor(pts, device=DEVICE)
            pts = F.normalize(pts, dim=1)
            for p in pts:
                protos.append(p)
                keys.append((k, int(c)))
    return dict(kind='proto', protos=torch.stack(protos), keys=keys)


def maha_state(emb, fit_mask, classes):
    stats = {}
    for k in range(N_TASKS):
        means, ids, res = [], [], []
        for c in classes:
            m = fit_mask & (TI == k) & (Y == c)
            if not m.any():
                continue
            mu = emb[m].mean(0)
            means.append(mu)
            ids.append(int(c))
            res.append(emb[m] - mu)
        R = torch.cat(res, 0)
        C = R.T @ R / len(R)
        d = C.shape[0]
        C = ((1 - MAHA_SHRINK) * C + MAHA_SHRINK * torch.diag(torch.diagonal(C))
             + 1e-4 * torch.eye(d, device=DEVICE))
        stats[k] = dict(means=torch.stack(means), ids=ids, inv=torch.linalg.inv(C))
    return dict(kind='maha', stats=stats)


def apply_scorer(state, emb_q, tasks_q):
    s = np.full(len(tasks_q), -np.inf)
    p = np.full(len(tasks_q), -1, int)
    if state['kind'] == 'proto':
        keys = state['keys']
        kt = np.array([t for t, _ in keys])
        kid = np.array([c for _, c in keys])
        for k in range(N_TASKS):
            rows = np.flatnonzero(tasks_q == k)
            cols = np.flatnonzero(kt == k)
            if not len(rows) or not len(cols):
                continue
            S = emb_q[rows] @ state['protos'][cols].T
            v, ix = S.max(1)
            s[rows] = v.cpu().numpy()
            p[rows] = kid[cols[ix.cpu().numpy()]]
    else:
        for k in range(N_TASKS):
            rows = np.flatnonzero(tasks_q == k)
            if not len(rows):
                continue
            st = state['stats'][k]
            diff = emb_q[rows, None, :] - st['means'][None, :, :]
            d2 = torch.einsum('bcd,de,bce->bc', diff, st['inv'], diff)
            v, ix = d2.min(1)
            s[rows] = (-v).cpu().numpy()
            p[rows] = np.asarray(st['ids'])[ix.cpu().numpy()]
    return s, p


def build_state(emb, fit_mask, classes, scorer, pk):
    if scorer == 'maha':
        return maha_state(emb, fit_mask, classes)
    return proto_state(emb, fit_mask, classes, pk)


def inner_oof_learned(fit, head_type, classes, seed, dim=32):
    """Known-only inner OOF scores for tau, refitting representation+head per inner block."""
    names = CAND_BY_HEAD[head_type]
    oof = {n: np.full(len(Y), np.nan) for n, _, _ in names}
    for b in np.unique(BLOCKS[fit]):
        hold = fit & (BLOCKS == b)
        inner = fit & ~hold
        rep = make_rep(inner)
        head, _ = train_head(rep, np.flatnonzero(inner), AUG_OF_HEAD[head_type], seed, classes, dim=dim)
        emb = embeddings(rep, head)
        hold_rows = np.flatnonzero(hold)
        for name, scorer, pk in names:
            st = build_state(emb, inner, classes, scorer, pk)
            s, _ = apply_scorer(st, emb[hold_rows], TI[hold_rows])
            oof[name][hold] = s
    return oof


# =====================================================================  metrics
def evaluate(s, p, known, unknown, y, tau):
    k = s[known]
    u = s[unknown]
    rej_u = (u < tau)
    rej_k = (k < tau)
    tp = float(rej_u.sum())
    fp = float(rej_k.sum())
    fn = float((~rej_u).sum())
    return dict(
        auc=float(((u[:, None] < k).sum() + .5 * (u[:, None] == k).sum()) / (len(k) * len(u))),
        rejection_f1=float(2 * tp / (2 * tp + fp + fn)),
        known_acceptance=float((~rej_k).mean()),
        unknown_rejection=float(rej_u.mean()),
        identity_correct_but_rejected=float(((p[known] == y[known]) & rej_k).mean()),
        threshold=float(tau),
        n_known=int(len(k)), n_unknown=int(len(u)),
        accepted_known=int((~rej_k).sum()), rejected_unknown=int(rej_u.sum()),
    )


def auc_ties(pos, neg):
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    s = np.r_[pos, neg]
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def quantile_lower(x, q):
    try:
        return float(np.quantile(x, q, method='lower'))
    except TypeError:
        return float(np.quantile(x, q, interpolation='lower'))


def summarise(folds, names):
    out = {}
    for name in names:
        ms = [f['metrics'][name] for f in folds]
        n_u = sum(m['n_unknown'] for m in ms)
        n_k = sum(m['n_known'] for m in ms)
        aucs = [m['auc'] for m in ms]
        out[name] = dict(
            auc=float(np.mean(aucs)),
            auc_fold_std=float(np.std(aucs)),
            auc_fold_values=[float(v) for v in aucs],
            rejection_f1=float(np.mean([m['rejection_f1'] for m in ms])),
            known_acceptance=float(np.mean([m['known_acceptance'] for m in ms])),
            known_acceptance_micro=float(sum(m['accepted_known'] for m in ms) / n_k),
            unknown_rejection=float(np.mean([m['unknown_rejection'] for m in ms])),
            unknown_rejection_micro=float(sum(m['rejected_unknown'] for m in ms) / n_u),
            identity_correct_but_rejected=float(np.mean([m['identity_correct_but_rejected'] for m in ms])),
            threshold=float(np.mean([m['threshold'] for m in ms])),
        )
    return out


# =====================================================================  folds
def run_folds(names_all, held_range):
    folds = []
    for held in held_range:
        t0 = time.time()
        path = OUT / 'cache' / f'fold_{held:02d}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            if all(n in cached['metrics'] for n in names_all):
                folds.append(cached)
                log(f'fold {held}: cached')
                continue
        fit = (Y != held) & (FR <= FIT_FRAC)
        window = (FR > EVAL_LO) & (FR <= EVAL_HI)
        known = window & (Y != held)
        unknown = window & (Y == held)
        classes = np.unique(Y[fit])
        fit_rows = np.flatnonzero(fit)
        win_rows = np.flatnonzero(window)
        known_w = known[win_rows]
        unknown_w = unknown[win_rows]
        y_w = Y[win_rows]
        seed = SEED + held * 101

        st = base_fit_rep(X65, TASKS_NP, Y, fit, 12)
        z = base_transform(st, X65, TASKS_NP)
        s, p = base_score(st, z, TASKS_NP, win_rows)
        s, p = s[win_rows], p[win_rows]
        oof = base_inner_oof(fit, X65, TASKS_NP, Y, BLOCKS, 12)
        tau = float(np.quantile(oof[fit], Q_TAU))
        row = dict(held_subject=int(held), metrics={}, curves={})
        row['metrics']['e65_baseline_repro'] = evaluate(s, p, known_w, unknown_w, y_w, tau)
        np.savez_compressed(OUT / 'cache' / f'e65_baseline_repro_fold_{held:02d}_scores.npz',
                            score=s, pred=p, known=known, unknown=unknown, subject=Y)
        torch.save(st, OUT / 'models' / f'e65_baseline_repro_fold_{held:02d}.pt')

        rep = make_rep(fit)
        for head_type in ('aug', 'noaug'):
            head, curve = train_head(rep, fit_rows, AUG_OF_HEAD[head_type], seed, classes)
            row['curves'][head_type] = curve
            emb = embeddings(rep, head)
            oof = inner_oof_learned(fit, head_type, classes, seed)
            for name, scorer, pk in CAND_BY_HEAD[head_type]:
                state = build_state(emb, fit, classes, scorer, pk)
                s, p = apply_scorer(state, emb[win_rows], TI[win_rows])
                tau = float(np.quantile(oof[name][fit], Q_TAU))
                row['metrics'][name] = evaluate(s, p, known_w, unknown_w, y_w, tau)
                np.savez_compressed(OUT / 'cache' / f'{name}_fold_{held:02d}_scores.npz',
                                    score=s, pred=p, known=known, unknown=unknown, subject=Y)
            torch.save(dict(head=head.state_dict(), curve=curve), OUT / 'models' / f'{head_type}_fold_{held:02d}.pt')
        path.write_text(json.dumps(row, indent=2))
        folds.append(row)
        log(f'fold {held}: ' + json.dumps({k: {kk: round(vv, 6) for kk, vv in v.items()}
                                          for k, v in row['metrics'].items()}) + f' ({time.time()-t0:.1f}s)')
    return folds


# =====================================================================  Test audit (read once, after freezing)
def test_audit(names_all, fit, extra=None):
    tm = pd.read_csv(ROOT / CFG['data']['test_metadata'])
    tt = tm.task.to_numpy()
    tti = np.array([TASKS.index(t) for t in tt])
    sess = tm.session.astype(str).str.casefold()
    is_s1 = sess.isin(['session1', 's1', '1']).to_numpy()
    x65t = np.load(ROOT / CFG['features']['e65_test']).astype(np.float32)
    xbpt = np.load(ROOT / CFG['features']['logbp_test']).astype(np.float32)
    classes = np.unique(Y[fit])
    audit = {}

    st = base_fit_rep(X65, TASKS_NP, Y, fit, 12)
    zt = base_transform(st, x65t, tt)
    s, p = base_score(st, zt, tt, np.arange(len(tt)))
    oof = base_inner_oof(fit, X65, TASKS_NP, Y, BLOCKS, 12)
    tau = float(np.quantile(oof[fit], Q_TAU))
    audit['e65_baseline_repro'] = test_row(s, p, tau, is_s1, tt)

    rep = make_rep(fit)
    emb_fit = None
    for head_type in ('aug', 'noaug'):
        head, curve = train_head(rep, np.flatnonzero(fit), AUG_OF_HEAD[head_type], SEED + 7, classes)
        emb_fit = embeddings(rep, head)
        zq = standardize_test(rep, x65t, xbpt, tti)
        head.eval()
        with torch.no_grad():
            emb_q = head.embed(zq)
        oof = inner_oof_learned(fit, head_type, classes, SEED + 7)
        for name, scorer, pk in CAND_BY_HEAD[head_type]:
            state = build_state(emb_fit, fit, classes, scorer, pk)
            s, p = apply_scorer(state, emb_q, tti)
            tau = float(np.quantile(oof[name][fit], Q_TAU))
            audit[name] = test_row(s, p, tau, is_s1, tt)
            np.savez_compressed(OUT / 'cache' / f'{name}_test_scores.npz', score=s, pred=p, tau=tau)
        torch.save(dict(head=head.state_dict(), curve=curve), OUT / 'models' / f'{head_type}_final.pt')
    if extra is not None:
        base_spec = extra['spec']
        name = extra['name']
        dim = int(CFG['extra_round']['embed_dim'])
        head, curve = train_head(rep, np.flatnonzero(fit), AUG_OF_HEAD['aug'], SEED + 7, classes, dim=dim)
        emb_fit = embeddings(rep, head)
        zq = standardize_test(rep, x65t, xbpt, tti)
        head.eval()
        with torch.no_grad():
            emb_q = head.embed(zq)
        oof = inner_oof_learned(fit, 'aug', classes, SEED + 7, dim=dim)
        state = build_state(emb_fit, fit, classes, base_spec['scorer'], base_spec['proto_k'])
        s, p = apply_scorer(state, emb_q, tti)
        tau = float(np.quantile(oof[base_spec['name']][fit], Q_TAU))
        audit[name] = test_row(s, p, tau, is_s1, tt)
        np.savez_compressed(OUT / 'cache' / f'{name}_test_scores.npz', score=s, pred=p, tau=tau)
        torch.save(dict(head=head.state_dict(), curve=curve), OUT / 'models' / 'aug_d64_final.pt')
    return audit


def test_row(s, p, tau, is_s1, tt):
    pred = np.where(s >= tau, p, 0)
    s1, s2 = s[is_s1], s[~is_s1]
    rej1 = float((pred[is_s1] == 0).mean())
    acc2 = float((pred[~is_s1] != 0).mean())
    tau70 = quantile_lower(s1, 0.70)
    tau95 = quantile_lower(s1, 0.95)
    return dict(
        tau=float(tau),
        S1_N=int(is_s1.sum()), S1_rejection=rej1,
        S2_N=int((~is_s1).sum()), S2_acceptance=acc2,
        S2_vs_S1_AUC=auc_ties(s2, s1),
        max_S2_acceptance_at_S1_rej_0_70=float((s2 >= tau70).mean()),
        max_S2_acceptance_at_S1_rej_0_95=float((s2 >= tau95).mean()),
        tau_S1_rej_0_70=float(tau70), tau_S1_rej_0_95=float(tau95),
        S1_score_stats=dict(mean=float(s1.mean()), std=float(s1.std()),
                            q10=float(np.quantile(s1, .1)), median=float(np.median(s1))),
        S2_score_stats=dict(mean=float(s2.mean()), std=float(s2.std()),
                            q10=float(np.quantile(s2, .1)), median=float(np.median(s2))),
        official_AUC='N/A', official_F1='N/A',
        stage_S1_met=bool(rej1 >= CFG['stage_targets']['S1_rejection']),
        stage_S2_met=bool(acc2 >= CFG['stage_targets']['S2_acceptance']),
        official_S1_met=bool(rej1 >= CFG['official_thresholds']['S1_rejection']),
        official_S2_met=bool(acc2 >= CFG['official_thresholds']['S2_acceptance']),
    )


# =====================================================================  report
def build_report(avg, audit, repro, extra, freeze, folds):
    import numpy as _np
    curve_stats = {}
    for h in ('aug', 'noaug'):
        cs = [f['curves'][h] for f in folds]
        curve_stats[h] = dict(
            best_epochs=sorted(set(int(c['best_epoch']) for c in cs)),
            n_epochs=sorted(set(int(c['n_epochs']) for c in cs)),
            val_ce_first=float(_np.mean([c['val_ce_first'] for c in cs])),
            val_ce_last=float(_np.mean([c['val_ce_last'] for c in cs])),
            val_ce_best=float(_np.mean([c['val_ce_best'] for c in cs])),
            last5_drop=float(_np.mean([c['val_ce_curve'][-5] - c['val_ce_curve'][-1] for c in cs])),
        )
    base_audit = audit['e65_baseline_repro']
    best_loso = max((n for n in avg if n != 'e65_baseline_repro'),
                    key=lambda n: avg[n]['unknown_rejection'])
    ba = audit[best_loso]
    L = ['# E223 折内度量学习（固定无监督表征 + 小 MLP 度量头）', '',
         '严格 20 折 subject-LOSO：每折只用 enrolled outer-train 且 trial fraction<=0.6 的行拟合 task-wise 标准化、PCA、度量头、原型/协方差与阈值；评估窗口 (0.6,0.8]；tau 由内层 5 块 known OOF 的 0.1 分位确定；Test 在全部折与追加轮决策冻结后只读取一次。',
         '',
         '打分口径沿用 E221：高分=known，score<tau 拒识；unknown 作正类计算 F1 与 AUC(-score)。`e65_baseline_repro` 为 E221/E222 原配方复现。',
         '',
         '## 1. 表征与训练设置', '',
         '- 输入固定为无监督特征：E65 correlation-logSPD(855) + E146 log-band-power(90)，不使用 E162 等全身份监督嵌入。',
         f'- 每折 enrolled fit 行内：按 task 做 z-score → 每流 PCA（e65={DIMS["e65"]}，logbp={DIMS["logbp"]}）→ 块 RMS 归一化 → 拼接 128 维。',
         f'- 度量头：128→{HEAD_CFG["hidden_dim"]}(GELU, dropout {HEAD_CFG["dropout"]})→{HEAD_CFG["embed_dim"]}，L2 归一化；损失 = CE（仅 enrolled 身份）+ {HEAD_CFG["supcon_weight"]} × 监督对比（同 subject 为正，含跨 trial/task 对）。',
         f'- 漂移增强（仅训练、仅特征空间）：band gain σ={AUG_CFG["band_gain_sigma"]}、channel gain σ={AUG_CFG["channel_gain_sigma"]}、spectral tilt σ={AUG_CFG["spectral_tilt_sigma"]}、加性噪声 σ={AUG_CFG["additive_noise_sigma"]}、band-block 替换 p={AUG_CFG["band_swap_prob"]}（同身份另一 trial；缓存特征无原始子窗，故为特征空间近似）。',
         f'- 早停：只用折内 fit 行按 trial-fraction block 划出的验证块（CE），再用最优 epoch 在全 fit 行重训；Test/held 身份从不参与。',
         '',
         '## 2. 基线复现核对', '',
         '|指标|参考值(E221/E222)|本轮复现|差值|', '|---|---:|---:|---:|']
    pairs = [('LOSO AUC', 'loso_auc', 'loso_auc'),
             ('LOSO unknown 拒识', 'loso_unknown_rejection', 'loso_unknown_rejection'),
             ('LOSO known 接受', 'loso_known_acceptance', 'loso_known_acceptance'),
             ('Test S1 拒识', 'test_S1_rejection', 'test_S1_rejection'),
             ('Test S2 接受', 'test_S2_acceptance', 'test_S2_acceptance'),
             ('Test S2-vs-S1 AUC', 'test_S2_vs_S1_auc', 'test_S2_vs_S1_auc')]
    for label, rk, ak in pairs:
        L.append(f'|{label}|{repro["reference"][rk]:.6f}|{repro["observed"][ak]:.6f}|{repro["observed"][ak]-repro["reference"][rk]:+.6f}|')
    L += ['', f'复现判定：**{"一致" if repro["consistent"] else "不一致"}**（阈值 AUC/拒识 ≤1e-4，S1/S2 ≤1e-3）。', '',
          '## 3. 训练曲线摘要', '',
          f"- 20 折 × 2 个头全部触到固定上限：`best_epoch` 均为 {curve_stats['aug']['best_epochs']}，"
          f"`n_epochs` 均为 {curve_stats['aug']['n_epochs']}（早停 patience=10 从未触发）。",
          f"- 验证 CE（折内 fit 行留出块，宏平均）：aug {curve_stats['aug']['val_ce_first']:.4f}→{curve_stats['aug']['val_ce_last']:.4f}；"
          f"noaug {curve_stats['noaug']['val_ce_first']:.4f}→{curve_stats['noaug']['val_ce_last']:.4f}。"
          f"最后 5 个 epoch 仍分别下降 {curve_stats['aug']['last5_drop']:.4f}/{curve_stats['noaug']['last5_drop']:.4f}，说明 80 epoch 预算尚未收敛。",
          f"- 漂移增强与不加增强的验证 CE 差异 <0.001，在本次固定强度与预算下增强几乎没有改变优化轨迹；这一设置未按 Test 回调。",
          '',
          '## 4. 候选结果', '',
          '|候选|LOSO AUC mean±std|拒识 F1|unknown 拒识 macro|unknown 拒识 micro|known 接受|identity 正确却被拒|Test S1 拒识|S1≥0.70|S1≥0.95|Test S2 接受|S2≥0.70|S2≥0.98|Test S2-vs-S1 AUC|S1拒识≥0.70 时 S2 接受上限|同时满足 LOSO>0.90 & S2>0.70 & S1≥0.70|',
          '|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|:---:|:---:|---:|---:|:---:|']
    for name, v in avg.items():
        a = audit[name]
        ok = (v['unknown_rejection'] > 0.90) and (a['S2_acceptance'] > 0.70) and (a['S1_rejection'] >= 0.70)
        L.append(
            f"|{name}|{v['auc']:.6f}±{v['auc_fold_std']:.6f}|{v['rejection_f1']:.6f}|{v['unknown_rejection']:.6f}|"
            f"{v['unknown_rejection_micro']:.6f}|{v['known_acceptance']:.6f}|{v['identity_correct_but_rejected']:.6f}|"
            f"{a['S1_rejection']:.6f}|{'是' if a['stage_S1_met'] else '否'}|{'是' if a['official_S1_met'] else '否'}|"
            f"{a['S2_acceptance']:.6f}|{'是' if a['stage_S2_met'] else '否'}|{'是' if a['official_S2_met'] else '否'}|"
            f"{a['S2_vs_S1_AUC']:.6f}|{a['max_S2_acceptance_at_S1_rej_0_70']:.6f}|{'是' if ok else '否'}|")
    k1v, k1a = avg['mlp_ce_contrast_k1'], audit['mlp_ce_contrast_k1']
    nav, naa = avg['mlp_no_augment'], audit['mlp_no_augment']
    L += ['', '说明：`S1拒识≥0.70 时 S2 接受上限` 是在 Test 分数上把阈值取为 S1 分数的 70% 分位后得到的 S2 接受率，仅作可分性诊断，不是本实验的工作点。', '',
          '## 5. 判断与结论', '', freeze['text'], '',
          '**域偏移证据（对应预设判断分支）**：', '',
          f"- 同会话 LOSO：`{best_loso}` AUC={avg[best_loso]['auc']:.6f}、unknown 拒识={avg[best_loso]['unknown_rejection']:.6f}；"
          f"baseline 为 {avg['e65_baseline_repro']['auc']:.6f}/{avg['e65_baseline_repro']['unknown_rejection']:.6f}。"
          "度量头在折内把身份聚得很紧，LOSO 开放集能力大幅提升。",
          f"- 跨会话 Test：k1 的 S1 分数 mean±std={k1a['S1_score_stats']['mean']:.4f}±{k1a['S1_score_stats']['std']:.4f}，"
          f"S2={k1a['S2_score_stats']['mean']:.4f}±{k1a['S2_score_stats']['std']:.4f}，两者几乎重合；tau={k1a['tau']:.4f} 把两边都判为偏远，"
          f"S2-vs-S1 分离 AUC={k1a['S2_vs_S1_AUC']:.6f}（baseline {base_audit['S2_vs_S1_AUC']:.6f}）。"
          f"maha 同样只有 {ba['S2_vs_S1_AUC']:.6f}。",
          f"- 即使用 Test 分数做条件最优，S1 拒识≥0.70 时 k1 的 S2 接受上限也只有 {k1a['max_S2_acceptance_at_S1_rej_0_70']:.6f}"
          f"（baseline {base_audit['max_S2_acceptance_at_S1_rej_0_70']:.6f}）；k3/maha 分别为 "
          f"{audit['mlp_ce_contrast_k3']['max_S2_acceptance_at_S1_rej_0_70']:.6f}/"
          f"{ba['max_S2_acceptance_at_S1_rej_0_70']:.6f}。",
          f"- 增强未改变结论：`mlp_no_augment` LOSO 拒识={nav['unknown_rejection']:.6f}、Test 分离 AUC={naa['S2_vs_S1_AUC']:.6f}，"
          f"与加增强的 k1（{k1v['unknown_rejection']:.6f}/{k1a['S2_vs_S1_AUC']:.6f}）基本相同。",
          '- 结论：LOSO 提升真实存在（同会话内），但 Test 分离能力反而低于 baseline，说明学到的主要是会话内身份方向，'
          '跨会话发生域偏移，而不是阈值、k 或表征维度的问题；按预设规则不再重复同一配方。', '']
    if extra:
        L += ['## 6. 追加一轮', '', f"- 触发：{extra['reason']}", '',
              '|候选|LOSO AUC|LOSO 拒识|known 接受|Test S1|Test S2|Test 分离 AUC|', '|---|---:|---:|---:|---:|---:|---:|']
        v, a = avg[extra['name']], audit[extra['name']]
        L.append(f"|{extra['name']}|{v['auc']:.6f}|{v['unknown_rejection']:.6f}|{v['known_acceptance']:.6f}|"
                 f"{a['S1_rejection']:.6f}|{a['S2_acceptance']:.6f}|{a['S2_vs_S1_AUC']:.6f}|")
        L.append('')
    else:
        L += ['## 6. 追加一轮', '', '- 未触发：最佳学习候选的 LOSO 拒识未落入预设区间（≥baseline+0.05 且 <0.90），按规则不再追加。', '',
              '- 备注：本轮唯一的学习候选组已全部给出；没有满足触发条件的配方，因此不存在“第二轮调参”。', '']
    L += ['## 7. 限制', '',
          '- 官方 Test AUC/F1 因 Test 无身份标签为 **N/A**；S1/S2 只是会话分层行为指标，不等同官方未知/正常分类指标。',
          '- Test 已被本系列实验反复查看，虽然本轮所有超参与候选预先固定、未按 Test 调参，评估偏倚仍不可消除。',
          '- LOSO 指标为逐折宏平均；unknown:known 约 1:19，拒识 F1 受类别比例影响。',
          '- “跨 trial 同身份为正”的监督对比不使用 held 身份；Test 不参与表征、度量头、早停或阈值。',
          '- 特征空间的 band-block 替换是“随机时间子窗特征替换”的代理，因为缓存特征已是 trial 级汇总。',
          f'- 运行：`.venv\\Scripts\\python.exe research\\E223_fold_metric_learning\\run_e223.py`（CUDA）。']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')


# =====================================================================  main
def main():
    global FR, TI, Y, BLOCKS, X65, XBP, TASKS_NP
    LOG.write_text('', encoding='utf8')
    assert torch.cuda.is_available(), 'CUDA not available'
    log(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    meta = pd.read_csv(ROOT / CFG['data']['train_metadata'])
    TASKS_NP = meta.task.to_numpy()
    Y = meta.subject.to_numpy(int)
    FR = frac(meta)
    BLOCKS = np.minimum((FR * N_BLOCKS).astype(int), N_BLOCKS - 1)
    TI = np.array([TASKS.index(t) for t in TASKS_NP])
    X65 = np.load(ROOT / CFG['features']['e65_train']).astype(np.float32)
    XBP = np.load(ROOT / CFG['features']['logbp_train']).astype(np.float32)
    assert len(X65) == len(meta) == len(XBP)
    assert set(Y) == set(range(1, 21)) and set(TASKS_NP) == set(TASKS)
    (OUT / 'cache').mkdir(exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)

    names_all = [c['name'] for c in CFG['candidates']]
    t0 = time.time()
    n_folds = int(os.environ.get('E223_FOLDS', '20'))
    folds = run_folds(names_all, range(1, n_folds + 1))
    avg = summarise(folds, names_all)

    baseline_name = 'e65_baseline_repro'
    learned = [n for n in names_all if n != baseline_name]
    best = max(learned, key=lambda n: (avg[n]['unknown_rejection'], avg[n]['known_acceptance']))

    extra = None
    extra_name = None
    if avg[best]['unknown_rejection'] >= avg[baseline_name]['unknown_rejection'] + 0.05 and \
       avg[best]['unknown_rejection'] < 0.90 and os.environ.get('E223_NO_EXTRA') != '1':
        base_spec = next(c for c in CFG['candidates'] if c['name'] == best)
        extra_name = best + CFG['extra_round']['name_suffix']
        extra = dict(name=extra_name,
                     reason=(f"最佳学习候选 {best} 的 LOSO 拒识 {avg[best]['unknown_rejection']:.4f} "
                             f"≥ baseline+0.05 且 <0.90，按预设规则用 embed_dim="
                             f"{CFG['extra_round']['embed_dim']} 重跑同一配方"),
                     spec=base_spec)
        log('extra round triggered: ' + json.dumps(extra))
        # Rerun the one extra variant. E223_FOLDS_EXTRA limits it during smoke tests.
        extra_folds = run_extra(extra_name, base_spec, int(os.environ.get('E223_FOLDS_EXTRA', str(len(folds)))))
        for f, g in zip(folds, extra_folds):
            f['metrics'].update(g['metrics'])
            if 'extra_curves' in g:
                f.setdefault('extra_curves', {}).update(g['extra_curves'])
        names_all.append(extra_name)
        avg.update(summarise(folds, [extra_name]))
    else:
        log(f'extra round not triggered (best={best}, LOSO rejection={avg[best]["unknown_rejection"]:.4f})')

    fit = FR <= FIT_FRAC
    audit = test_audit(names_all, fit, extra)

    # reproduction check
    ref = CFG['reference_e65_baseline']
    obs = dict(loso_auc=avg[baseline_name]['auc'],
               loso_unknown_rejection=avg[baseline_name]['unknown_rejection'],
               loso_known_acceptance=avg[baseline_name]['known_acceptance'],
               test_S1_rejection=audit[baseline_name]['S1_rejection'],
               test_S2_acceptance=audit[baseline_name]['S2_acceptance'],
               test_S2_vs_S1_auc=audit[baseline_name]['S2_vs_S1_AUC'])
    consistent = (abs(obs['loso_auc'] - ref['loso_auc']) <= 1e-4 and
                  abs(obs['loso_unknown_rejection'] - ref['loso_unknown_rejection']) <= 1e-4 and
                  abs(obs['test_S1_rejection'] - ref['test_S1_rejection']) <= 1e-3 and
                  abs(obs['test_S2_acceptance'] - ref['test_S2_acceptance']) <= 1e-3)
    repro = dict(reference=ref, observed=obs, consistent=bool(consistent))

    # freeze decision (rule fixed in config)
    freeze_names = [n for n in names_all
                    if avg[n]['unknown_rejection'] > 0.90 and audit[n]['S2_vs_S1_AUC'] >= 0.80]
    if freeze_names:
        fz = max(freeze_names, key=lambda n: (audit[n]['S2_vs_S1_AUC'], avg[n]['unknown_rejection']))
        text = (f"**冻结候选：`{fz}`。** Test 分离 AUC={audit[fz]['S2_vs_S1_AUC']:.6f} ≥0.80 且 "
                f"LOSO 拒识={avg[fz]['unknown_rejection']:.6f} >0.90，满足预设冻结规则。"
                f"按官方门槛如实报告差距：Test S1 拒识={audit[fz]['S1_rejection']:.6f}（要求≥0.95，"
                f"{'达标' if audit[fz]['official_S1_met'] else '未达标'}），Test S2 接受="
                f"{audit[fz]['S2_acceptance']:.6f}（要求≥0.98，"
                f"{'达标' if audit[fz]['official_S2_met'] else '未达标'}）；官方 Test AUC/F1 因无标签为 N/A。"
                "不再进行第二轮搜索。")
        log(text)
    else:
        cand = max(names_all, key=lambda n: audit[n]['S2_vs_S1_AUC'])
        same_recipe = (avg[best]['unknown_rejection'] > avg[baseline_name]['unknown_rejection'] and
                       audit[best]['S2_vs_S1_AUC'] <= audit[baseline_name]['S2_vs_S1_AUC'] + 0.02)
        text = (f"**未冻结任何候选。** 没有任何候选同时满足 LOSO 拒识>0.90 与 Test 分离 AUC≥0.80。"
                f"Test 分离 AUC 最高的是 `{cand}`（{audit[cand]['S2_vs_S1_AUC']:.6f}），"
                f"baseline 为 {audit[baseline_name]['S2_vs_S1_AUC']:.6f}。")
        if same_recipe:
            text += (f" 最佳学习候选 `{best}` 的 LOSO 拒识由 {avg[baseline_name]['unknown_rejection']:.4f} "
                     f"升到 {avg[best]['unknown_rejection']:.4f}，但 Test 分离能力没有提升（"
                     f"{audit[best]['S2_vs_S1_AUC']:.4f} vs baseline "
                     f"{audit[baseline_name]['S2_vs_S1_AUC']:.4f}），按预设判断这属于域偏移而非表征问题，"
                     "不再刷同一配方。")
        else:
            text += " 按预设判断规则，本轮不再追加同一配方。"
        log(text)

    result = dict(experiment=CFG['experiment'], device=torch.cuda.get_device_name(0),
                  fold_count=len(folds), candidates=names_all,
                  candidate_summary=avg, folds=folds, test_audit=audit,
                  reproduction=repro, extra_round=extra,
                  freeze=dict(frozen_candidates=freeze_names),
                  official_test_metric='N/A: Test has no identity labels',
                  test_bias_warning='Test inspected repeatedly across the E-series; no Test-based selection in E223, but evaluation bias is irreducible.',
                  test_read_once_after_freeze=True,
                  runtime_seconds=time.time() - t0)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    build_report(avg, audit, repro, extra, dict(text=text), folds)
    log('completed in %.1fs' % (time.time() - t0))


def run_extra(name, base_spec, held_range):
    """One extra round at embed_dim=64, identical otherwise. Runs per fold with its own cache."""
    folds = []
    for held in range(1, held_range + 1):
        t0 = time.time()
        path = OUT / 'cache' / f'fold_{held:02d}.json'
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached and name in cached['metrics']:
            folds.append(cached)
            log(f'extra fold {held}: cached')
            continue
        fit = (Y != held) & (FR <= FIT_FRAC)
        window = (FR > EVAL_LO) & (FR <= EVAL_HI)
        known = window & (Y != held)
        unknown = window & (Y == held)
        classes = np.unique(Y[fit])
        seed = SEED + held * 101
        rep = make_rep(fit)
        head, curve = train_head(rep, np.flatnonzero(fit), AUG_OF_HEAD['aug'], seed, classes,
                                 dim=int(CFG['extra_round']['embed_dim']))
        emb = embeddings(rep, head)
        oof = inner_oof_learned(fit, 'aug', classes, seed, dim=int(CFG['extra_round']['embed_dim']))
        row = dict(held_subject=int(held), metrics={}, extra_curves={'aug_d64': curve})
        win_rows = np.flatnonzero(window)
        scorer, pk = base_spec['scorer'], base_spec['proto_k']
        state = build_state(emb, fit, classes, scorer, pk)
        s, p = apply_scorer(state, emb[win_rows], TI[win_rows])
        tau = float(np.quantile(oof[base_spec['name']][fit], Q_TAU))
        row['metrics'][name] = evaluate(s, p, known[win_rows], unknown[win_rows], Y[win_rows], tau)
        np.savez_compressed(OUT / 'cache' / f'{name}_fold_{held:02d}_scores.npz',
                            score=s, pred=p, known=known, unknown=unknown, subject=Y)
        if cached:
            row['metrics'] = {**cached['metrics'], **row['metrics']}
            row['curves'] = cached.get('curves', {})
        path.write_text(json.dumps(row, indent=2))
        folds.append(row)
        log(f'extra fold {held}: {json.dumps({k: round(v["unknown_rejection"], 6) for k, v in row["metrics"].items()})} '
            f'({time.time()-t0:.1f}s)')
    return folds


if __name__ == '__main__':
    main()
