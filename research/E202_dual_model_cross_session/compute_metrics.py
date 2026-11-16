"""E202 11-gate metric calculator (Train/LOSO side only).

Reuses the frozen per-fold checkpoints and gate bundles produced by run_e202.py.
No Test data is read here; gates 9/10 (Test S1/S2) stay N/A until the frozen audit.
Every statistic below is fitted on the fold's fit rows or its inner split only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_a import IdentityEmbedding, embed
from model_b import UnknownGate, rejection_features
from protocol import fit_mask, fit_projection, build_prototypes

BASE = ROOT.parents[1]
META = BASE / 'BrainPrint_cache' / 'train_metadata.csv'
E65 = BASE / 'BrainPrint_cache' / 'e65_correlation_spd' / 'train_correlation_logspd.npy'
E162 = ROOT.parent / 'E162_raw_eeg_contrastive_v2' / 'train_emb.npy'


def load_data():
    meta = pd.read_csv(META)
    meta['frac'] = (meta['trial'].to_numpy(dtype=float)
                    / meta.groupby(['subject', 'task'])['trial'].transform('max').to_numpy())
    subjects = meta.subject.to_numpy(int)
    x65 = np.load(E65, mmap_mode='r').astype(np.float32)
    x162 = np.load(E162, mmap_mode='r').astype(np.float32)
    return meta, subjects, x65, x162


def auc_score(y, p):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_f1(y_true, y_pred, classes):
    scores = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float('nan')


def feature_attack(model, x65p, x162, y, device, eps=0.5, steps=5):
    """Feature-space PGD proxy on the projected E65 branch (not raw-waveform PGD)."""
    model.eval()
    a = torch.as_tensor(x65p, dtype=torch.float32, device=device).clone().detach()
    b = torch.as_tensor(x162, dtype=torch.float32, device=device)
    t = torch.as_tensor(y, dtype=torch.long, device=device)
    sd = a.std(0, keepdim=True).clamp_min(1e-6)
    delta = torch.zeros_like(a, requires_grad=True)
    alpha = eps / steps
    for _ in range(steps):
        _, logits = model(a + delta, b)
        loss = torch.nn.functional.cross_entropy(logits, t)
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta += alpha * grad.sign()
            delta.clamp_(-eps * sd, eps * sd)
    with torch.no_grad():
        _, logits = model(a + delta, b)
    return (a + delta).detach().cpu().numpy(), logits.argmax(1).cpu().numpy()


def fold_metrics(meta, subjects, x65, x162, unknown_eval, k, device):
    ids = np.arange(1, 21)
    enrolled = np.array([s for s in ids if s not in set(unknown_eval)], dtype=int)
    fit = fit_mask(meta, subjects, enrolled)
    _, _, x65p = fit_projection(x65, fit, 96)
    ck = torch.load(ROOT / 'checkpoints' / ('model_a_k%d_%s.pt' % (k, '_'.join(map(str, unknown_eval)))), map_location=device)
    model = IdentityEmbedding(96, x162.shape[1], 20).to(device)
    model.load_state_dict(ck['state_dict'])
    z = embed(model, x65p, x162, device=device)
    proto, proto_ids = build_prototypes(z, subjects, fit, enrolled)
    feat, pred_idx, best = rejection_features(z, proto, x65, x162)
    pred_ids = proto_ids[pred_idx]

    gate_npz = np.load(ROOT / 'bundle' / ('gate_k%d_%s.npz' % (k, '_'.join(map(str, unknown_eval)))))
    gate = UnknownGate()
    gate.mean, gate.std, gate.w = gate_npz['mean'], gate_npz['std'], gate_npz['w']
    gate.b = float(gate_npz['b'][0]); gate.tau = float(gate_npz['tau'][0])
    p = gate.predict_known(feat)

    frac = meta['frac'].to_numpy()
    eval_rows = (frac > 0.6) & (frac <= 0.8)
    known_eval = eval_rows & np.isin(subjects, enrolled)
    unknown_rows = eval_rows & np.isin(subjects, unknown_eval)

    # 1) Train fit accuracy (supervised head, in-sample capacity check only)
    with torch.no_grad():
        a = torch.as_tensor(x65p[fit], dtype=torch.float32, device=device)
        b = torch.as_tensor(x162[fit], dtype=torch.float32, device=device)
        fit_acc = float((model(a, b)[1].argmax(1).cpu().numpy() == subjects[fit] - 1).mean())

    # 2) C / closed-set accuracy and 3) identity macro-F1 on held-out evaluation rows
    C = float((pred_ids[known_eval] == subjects[known_eval]).mean())
    id_f1 = macro_f1(subjects[known_eval], pred_ids[known_eval], proto_ids)

    # 4) continuous-time-block out-of-fold accuracy inside the evaluation window
    block_acc = []
    sub_frac = frac[known_eval]
    sub_ok = (pred_ids[known_eval] == subjects[known_eval])
    edges = np.linspace(0.6, 0.8, 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sub_frac > lo) & (sub_frac <= hi)
        if sel.sum() > 0:
            block_acc.append(float(sub_ok[sel].mean()))

    # 5) pseudo-unknown AUROC
    rows5 = known_eval | unknown_rows
    y5 = np.isin(subjects[rows5], enrolled).astype(int)
    auroc = auc_score(y5, p[rows5])

    # 6) U@95: threshold from gate-train known 95th percentile, evaluated on held-out unknown
    inner_unknown = enrolled[:max(1, len(enrolled) // 5)]
    inner_registered = np.array([s for s in enrolled if s not in set(inner_unknown)], dtype=int)
    inner_fit = fit_mask(meta, subjects, inner_registered)
    inner_proto, _ = build_prototypes(z, subjects, inner_fit, inner_registered)
    inner_feat, _, _ = rejection_features(z, inner_proto, x65, x162)
    gate_train_known = eval_rows & np.isin(subjects, inner_registered) & ~inner_fit
    tau95 = float(np.quantile(gate.predict_known(inner_feat[gate_train_known]), 0.05)) if gate_train_known.any() else 0.5
    u95 = float(np.mean(p[unknown_rows] < tau95))
    known_accept95 = float(np.mean(p[known_eval] >= tau95))

    # 7/8) feature-space PGD proxy -> RA (registered identity accuracy) and ASafeR (safe rejection)
    x65_atk, _ = feature_attack(model, x65p[known_eval], x162[known_eval], subjects[known_eval] - 1, device)
    z_atk = embed(model, x65_atk, x162[known_eval], device=device)
    feat_atk, pred_atk_idx, _ = rejection_features(z_atk, proto, x65[known_eval], x162[known_eval])
    p_atk = gate.predict_known(feat_atk)
    RA = float((proto_ids[pred_atk_idx] == subjects[known_eval]).mean())
    # ASafeR: adversarial sample must not be silently accepted as the WRONG identity
    ASafeR = float(np.mean((p_atk < tau95) | (proto_ids[pred_atk_idx] == subjects[known_eval])))

    return {
        'k': int(k), 'unknown_eval': unknown_eval.tolist(), 'enrolled': enrolled.tolist(),
        'train_fit_accuracy': fit_acc, 'C': C, 'identity_macro_f1': id_f1,
        'block_accuracy_mean': float(np.mean(block_acc)) if block_acc else float('nan'),
        'block_accuracy_min': float(np.min(block_acc)) if block_acc else float('nan'),
        'pseudo_unknown_auroc': auroc,
        'U_at_95': u95, 'known_accept_at_95': known_accept95,
        'RA_feature_pgd_proxy': RA, 'ASafeR_feature_pgd_proxy': ASafeR,
        'known_accept_at_tauB': float(np.mean(p[known_eval] >= gate.tau)),
        'unknown_rejection_at_tauB': float(np.mean(p[unknown_rows] < gate.tau)),
        'n_known_eval': int(known_eval.sum()), 'n_unknown_eval': int(unknown_rows.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ks', type=int, nargs='+', default=[2, 5, 10])
    ap.add_argument('--per-k', type=int, default=2)
    args = ap.parse_args()
    device = args.device
    meta, subjects, x65, x162 = load_data()
    plan = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))['results']
    out = []
    for k in args.ks:
        folds = [r['unknown_eval'] for r in plan if r['k'] == k][:args.per_k]
        for unknown in folds:
            print('metrics fold', k, unknown, flush=True)
            out.append(fold_metrics(meta, subjects, x65, x162, np.array(unknown), k, device))
    keys = ['train_fit_accuracy', 'C', 'identity_macro_f1', 'block_accuracy_mean', 'block_accuracy_min',
            'pseudo_unknown_auroc', 'U_at_95', 'RA_feature_pgd_proxy', 'ASafeR_feature_pgd_proxy',
            'known_accept_at_tauB', 'unknown_rejection_at_tauB']
    summary = {}
    for k in args.ks:
        rows = [r for r in out if r['k'] == k]
        if not rows:
            continue
        summary[str(k)] = {key: float(np.mean([r[key] for r in rows])) for key in keys}
        summary[str(k)]['folds'] = len(rows)
    payload = {'experiment': 'E202 metric audit', 'scope': 'Train-only; Test S1/S2 not read', 'per_fold': out, 'summary': summary}
    (ROOT / 'metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
