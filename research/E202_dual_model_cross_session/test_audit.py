"""E202 frozen Test audit: register 20 subjects, then score the official Test split once.

Frozen once from Train only:
  - SVD projection fitted on all-20 fit rows (frac <= 0.6)
  - Model A trained on all-20 fit rows
  - 20 identity prototypes from all-20 fit rows
  - Gate trained on an inner split (16 registered vs 4 held-out unknown), using the
    inner prototype feature space, exactly as in the fold experiments

The Test block is read only for features and the session column is used only AFTER all
labels are written, to report S1 rejection / S2 acceptance. No Test statistic, label or
threshold tuning is involved.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_a import train_model, embed
from model_b import UnknownGate, rejection_features
from protocol import fit_mask, fit_projection, build_prototypes

BASE = ROOT.parents[1]
TRAIN_META = BASE / 'BrainPrint_cache' / 'train_metadata.csv'
TRAIN_E65 = BASE / 'BrainPrint_cache' / 'e65_correlation_spd' / 'train_correlation_logspd.npy'
TRAIN_E162 = ROOT.parent / 'E162_raw_eeg_contrastive_v2' / 'train_emb.npy'
TEST_META = BASE / 'BrainPrint_cache' / 'test_metadata.csv'
TEST_E65 = ROOT / 'cache' / 'test_e65_features.npy'
TEST_E162 = ROOT.parent / 'E162_raw_eeg_contrastive_v2' / 'test_emb.npy'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = pd.read_csv(TRAIN_META)
    meta['frac'] = (meta['trial'].to_numpy(dtype=float)
                    / meta.groupby(['subject', 'task'])['trial'].transform('max').to_numpy())
    subjects = meta.subject.to_numpy(int)
    frac = meta['frac'].to_numpy()
    x65 = np.load(TRAIN_E65, mmap_mode='r').astype(np.float32)
    x162 = np.load(TRAIN_E162, mmap_mode='r').astype(np.float32)

    enrolled = np.arange(1, 21)
    fit = fit_mask(meta, subjects, enrolled)
    proj, _, x65p = fit_projection(x65, fit, 96)
    mu, sd, basis = proj
    model, _, _ = train_model(x65p[fit], x162[fit], subjects[fit] - 1, 20,
                              device=device, epochs=5, seed=20261116)
    torch.save({'state_dict': model.state_dict(), 'protocol': 'all-20 fit rows, frozen Test audit'},
               ROOT / 'checkpoints' / 'model_a_frozen20.pt')
    z = embed(model, x65p, x162, device=device)
    proto, proto_ids = build_prototypes(z, subjects, fit, enrolled)
    feat, pred_idx, _ = rejection_features(z, proto, x65, x162)

    inner_unknown = enrolled[:4]
    inner_registered = np.array([s for s in enrolled if s not in set(inner_unknown)], dtype=int)
    inner_fit = fit_mask(meta, subjects, inner_registered)
    inner_proto, _ = build_prototypes(z, subjects, inner_fit, inner_registered)
    inner_feat, _, _ = rejection_features(z, inner_proto, x65, x162)
    window = (frac > 0.6) & (frac <= 0.8)
    gate_known = window & np.isin(subjects, inner_registered) & ~inner_fit
    gate_unknown = window & np.isin(subjects, inner_unknown)
    train_rows = gate_known | gate_unknown
    labels = np.isin(subjects[train_rows], inner_registered).astype(int)
    gate = UnknownGate().fit(inner_feat[train_rows], labels)
    np.savez(ROOT / 'bundle' / 'frozen_bundle_e202.npz',
             proj_mu=mu, proj_sd=sd, proj_basis=basis,
             proto=proto, proto_ids=proto_ids,
             gate_mean=gate.mean, gate_std=gate.std, gate_w=gate.w,
             gate_b=np.asarray([gate.b]), gate_tau=np.asarray([gate.tau]))

    # ---- Test inference (no labels available) ----
    tmeta = pd.read_csv(TEST_META)
    t65 = np.load(TEST_E65).astype(np.float32)
    t162 = np.load(TEST_E162).astype(np.float32)
    t65p = ((t65 - mu) / sd) @ basis
    t65p = t65p.astype(np.float32)
    tz = embed(model, t65p, t162, device=device)
    tfeat, tpred_idx, _ = rejection_features(tz, proto, t65, t162)
    tp = gate.predict_known(tfeat)
    identities = proto_ids[tpred_idx]
    pred = np.where(tp >= gate.tau, identities, 0)
    pd.DataFrame({'sample': tmeta['sample'], 'label': pred}).to_csv(
        ROOT / 'submission_e202.csv', index=False)

    # ---- Session audit after labels are written ----
    session = tmeta['session'].astype(str).to_numpy()
    s1 = session == 'session1'
    s2 = session == 'session2'
    audit = {
        'experiment': 'E202 frozen Test audit',
        'n_rows': int(len(pred)),
        'unknown_rate_overall': float(np.mean(pred == 0)),
        'S1_rejection_rate': float(np.mean(pred[s1] == 0)),
        'S2_accept_rate': float(np.mean(pred[s2] != 0)),
        'S1_rows': int(s1.sum()), 'S2_rows': int(s2.sum()),
        'known_accept_train': float(np.mean(gate.predict_known(feat[window & np.isin(subjects, enrolled)]) >= gate.tau)),
        'unknown_rejection_train_inner': float(np.mean(gate.predict_known(inner_feat[gate_unknown]) < gate.tau)),
        'gate_tau': float(gate.tau),
        'test_auc_f1_C_RA_ASafeR_U_score': 'N/A (no Test labels)',
    }
    (ROOT / 'test_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
