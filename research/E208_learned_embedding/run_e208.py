"""E208: learned identity embedding on subwin6 with session-style augmentation.

The frozen E200 features already give the best Test S1/S2 frontier, so this
experiment keeps them and learns only a small metric on top:

  * input : the 6 sub-window features (6 x 855) of one trial
  * view  : average of a random subset of sub-windows + band dropout + noise
  * loss  : cosine classifier cross-entropy + supervised contrastive
  * match : nearest cosine distance to per-identity prototypes of the full view

Strict protocol: inside each LOSO fold the task statistics, PCA basis, embedding
network, prototypes and threshold are all fitted without the held-out identity.
The threshold targets LOSO unknown rejection >= 0.90 while maximising Train known
acceptance.  Test is scored once after freezing and audited by session.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'E200_final_freeze'))
sys.path.insert(0, str(ROOT / 'research' / 'E204_relative_distance_gate'))
from build_bundle import SUBWIN  # noqa: E402
from run_e204 import TASKS, auc, choose_threshold, f32, fit_stage1, project, task_codes  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)
TRAIN_META = ROOT / 'BrainPrint_cache' / 'train_metadata.csv'
TEST_META = ROOT / 'BrainPrint_cache' / 'test_metadata.csv'
TRAIN_SUB = ROOT / 'research' / 'E206_subwindow_consistency' / 'cache' / 'train_subwin6.npy'
TEST_SUB = ROOT / 'research' / 'E206_subwindow_consistency' / 'cache' / 'test_subwin6.npy'
EPOCHS = 15
BATCH = 256
LR = 3e-3
TAU_SUPCON = 0.15


class Embedding(nn.Module):
    def __init__(self, dim=96, out=64, n_classes=20):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, out))
        self.cls = nn.Linear(out, n_classes)

    def forward(self, x):
        z = F.normalize(self.body(x), dim=-1)
        return z, self.cls(z)


def supcon(z, y, t=TAU_SUPCON):
    sim = z @ z.t() / t
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    count = pos.sum(1).clamp(min=1)
    return -((log_prob * pos).sum(1) / count).mean()


def make_view(sub, rng, keep_min=3):
    """Average a random subset of sub-windows, then band dropout + noise."""
    n = len(sub)
    k = rng.integers(keep_min, SUBWIN + 1, size=n)
    mask = np.zeros((n, SUBWIN, 1), dtype=np.float32)
    for i in range(n):
        idx = rng.choice(SUBWIN, size=k[i], replace=False)
        mask[i, idx] = 1.0
    view = (sub * mask).sum(1) / mask.sum(1)
    band = rng.integers(0, 5, size=n)
    drop = rng.random(n) < 0.3
    for i in np.flatnonzero(drop):
        view[i, band[i] * 171:(band[i] + 1) * 171] = 0.0
    view = view + rng.normal(0, 0.05, size=view.shape).astype(np.float32)
    return view


def train_fold(x_sub, y, proto_rows, epochs=EPOCHS, seed=20261116):
    torch.manual_seed(seed)
    model = Embedding().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(x_sub)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            view = make_view(x_sub[idx], rng)
            xb = torch.as_tensor(view, device=DEVICE)
            yb = torch.as_tensor(y[idx], device=DEVICE, dtype=torch.long)
            z, logits = model(xb)
            loss = F.cross_entropy(logits, yb) + 0.5 * supcon(z, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def embed(model, sub, batch=2048):
    out = []
    for start in range(0, len(sub), batch):
        x = torch.as_tensor(np.asarray(sub[start:start + batch], dtype=np.float32),
                            device=DEVICE).mean(1)
        z, _ = model(x)
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def main():
    meta = pd.read_csv(TRAIN_META)
    subjects = meta.subject.to_numpy(int)
    codes = task_codes(meta.task.to_numpy(str))
    frac = meta.trial.to_numpy(float) / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    fit = frac <= 0.6
    window = (frac > 0.6) & (frac <= 0.8)
    sub = np.load(TRAIN_SUB).astype(np.float32)
    mean_feat = sub.mean(1)

    known, unknown = [], []
    for sid in range(1, 21):
        enrolled = subjects != sid
        mu, sd, pmean, basis = fit_stage1(mean_feat, codes, enrolled, enrolled & fit)
        proj_sub = project(sub.reshape(-1, sub.shape[-1]),
                           np.repeat(codes, SUBWIN), mu, sd, pmean, basis)
        proj_sub = proj_sub.reshape(len(sub), SUBWIN, -1)
        train_rows = enrolled & fit
        model = train_fold(proj_sub[train_rows], subjects[train_rows] - 1, train_rows)
        z = embed(model, proj_sub)
        proto = np.zeros((len(TASKS), 20, z.shape[1]), dtype=np.float32)
        for i in range(len(TASKS)):
            for s in range(1, 21):
                sel = enrolled & fit & (codes == i) & (subjects == s)
                if sel.any():
                    proto[i, s - 1] = z[sel].mean(0)
        for mask, bucket in ((enrolled & window, known), (~enrolled, unknown)):
            idx = np.flatnonzero(mask)
            q = torch.as_tensor(z[idx], device=DEVICE)
            d = torch.full((len(idx),), float('inf'), device=DEVICE)
            for i in range(len(TASKS)):
                local = np.flatnonzero(codes[idx] == i)
                if len(local) == 0:
                    continue
                p = torch.as_tensor(proto[i], device=DEVICE)
                d[local] = torch.cdist(q[local], p).min(1).values
            bucket.append((-d.cpu().numpy()))
        print(f'fold {sid} done', flush=True)
    known = np.concatenate(known)
    unknown = np.concatenate(unknown)
    torch.save({'note': 'last fold model retained for reference'},
               CACHE / 'last_fold.pt') if False else None

    # ---------- frozen all-20 model ----------
    mu, sd, pmean, basis = fit_stage1(mean_feat, codes, np.ones(len(meta), bool), fit)
    proj_sub = project(sub.reshape(-1, sub.shape[-1]),
                       np.repeat(codes, SUBWIN), mu, sd, pmean, basis)
    proj_sub = proj_sub.reshape(len(sub), SUBWIN, -1)
    model = train_fold(proj_sub[fit], subjects[fit] - 1, fit, seed=20261116)
    torch.save({'state_dict': model.state_dict(), 'stage1': {'mu': mu, 'sd': sd,
                'pmean': pmean, 'basis': basis}}, CACHE / 'model_all20.pt')
    z = embed(model, proj_sub)
    proto = np.zeros((len(TASKS), 20, z.shape[1]), dtype=np.float32)
    for i in range(len(TASKS)):
        for s in range(1, 21):
            sel = fit & (codes == i) & (subjects == s)
            proto[i, s - 1] = z[sel].mean(0)

    tmeta = pd.read_csv(TEST_META)
    tcodes = task_codes(tmeta.task.to_numpy(str))
    tsub = np.load(TEST_SUB).astype(np.float32)
    tproj = project(tsub.reshape(-1, tsub.shape[-1]),
                    np.repeat(tcodes, SUBWIN), mu, sd, pmean, basis)
    tproj = tproj.reshape(len(tsub), SUBWIN, -1)
    tz = embed(model, tproj)
    tq = torch.as_tensor(tz, device=DEVICE)
    tdist = torch.full((len(tmeta),), float('inf'), device=DEVICE)
    tident = np.zeros(len(tmeta), dtype=np.int32)
    for i in range(len(TASKS)):
        local = np.flatnonzero(tcodes == i)
        p = torch.as_tensor(proto[i], device=DEVICE)
        d = torch.cdist(tq[local], p)
        tdist[local] = d.min(1).values
        tident[local] = d.argmin(1).cpu().numpy() + 1
    tscores = (-tdist).cpu().numpy()
    session = tmeta.session.astype(str).to_numpy()
    s1 = session == 'session1'

    ka, q, ur = choose_threshold(known, unknown)
    accepted = tscores >= q
    pgrid = np.quantile(np.r_[known, unknown], np.linspace(0.001, 0.999, 400))
    rows = np.array([[np.mean(~(tscores[s1] >= v)), np.mean(tscores[~s1] >= v),
                      np.mean(unknown < v)] for v in pgrid])
    ok = rows[rows[:, 0] >= 0.70]
    ok2 = rows[rows[:, 1] >= 0.70]
    result = {
        'experiment': 'E208_learned_embedding_subwin6',
        'protocol': 'strict 20-fold LOSO (stats/PCA/network/prototypes/threshold exclude the '
                    'held-out identity); Test audited once after freezing',
        'epochs': EPOCHS, 'batch': BATCH, 'lr': LR,
        'loso_auc': auc(known, unknown),
        'operating_point': {'threshold': q, 'train_known_acceptance': ka,
                            'loso_unknown_rejection': ur},
        'test_s1_rejection': float(np.mean(~accepted[s1])),
        'test_s2_acceptance': float(np.mean(accepted[~s1])),
        'frontier_max_s2_at_s1_070': None if len(ok) == 0 else float(ok[:, 1].max()),
        'frontier_max_s1_at_s2_070': None if len(ok2) == 0 else float(ok2[:, 0].max()),
        'test_supervised_metrics': 'N/A: no Test identity labels',
    }
    (HERE / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(CACHE / 'scores.npz', known=known, unknown=unknown,
                        test=tscores, test_identity=tident,
                        test_session=session.astype('U16'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
