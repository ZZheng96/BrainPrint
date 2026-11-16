"""Strict leave-one-subject-out (20 folds) for the E202 dual-model scheme.

Each fold: 19 subjects enrolled, 1 subject held out as unknown. The held-out subject
never enters the SVD/PCA fit, the identity prototypes, or the gate's training rows.
The gate's unknown class comes from an inner split (first 3 enrolled subjects) whose
prototype is built without them.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_a import train_model, embed
from model_b import UnknownGate, rejection_features
from protocol import fit_mask, fit_projection, build_prototypes

BASE = ROOT.parents[1]
META = BASE / 'BrainPrint_cache' / 'train_metadata.csv'
E65 = BASE / 'BrainPrint_cache' / 'e65_correlation_spd' / 'train_correlation_logspd.npy'
E162 = ROOT.parent / 'E162_raw_eeg_contrastive_v2' / 'train_emb.npy'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = __import__('pandas').read_csv(META)
    meta['frac'] = (meta['trial'].to_numpy(dtype=float)
                    / meta.groupby(['subject', 'task'])['trial'].transform('max').to_numpy())
    subjects = meta.subject.to_numpy(int)
    frac = meta['frac'].to_numpy()
    x65 = np.load(E65, mmap_mode='r').astype(np.float32)
    x162 = np.load(E162, mmap_mode='r').astype(np.float32)
    rows = []
    for held in range(1, 21):
        enrolled = np.array([s for s in range(1, 21) if s != held], dtype=int)
        fit = fit_mask(meta, subjects, enrolled)
        _, _, x65p = fit_projection(x65, fit, 96)
        model, _, _ = train_model(x65p[fit], x162[fit], subjects[fit] - 1, 20,
                                 device=device, epochs=5, seed=20261116 + held)
        torch.save({'state_dict': model.state_dict(), 'held_out': held},
                   ROOT / 'checkpoints' / ('model_a_loso1_%d.pt' % held))
        z = embed(model, x65p, x162, device=device)
        proto, proto_ids = build_prototypes(z, subjects, fit, enrolled)
        feat, pred_idx, _ = rejection_features(z, proto, x65, x162)

        inner_unknown = enrolled[:3]
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
        p = gate.predict_known(feat)
        eval_rows = window & (subjects == held)
        known_rows = window & np.isin(subjects, enrolled)
        rows.append({
            'held_out': held,
            'loso_rejection': float(np.mean(p[eval_rows] < gate.tau)),
            'known_accept': float(np.mean(p[known_rows] >= gate.tau)),
            'registration_id_accuracy': float(np.mean(proto_ids[pred_idx[known_rows]] == subjects[known_rows])),
            'tau_b': float(gate.tau),
        })
        print('LOSO', held, rows[-1], flush=True)
    values = np.array([r['loso_rejection'] for r in rows])
    payload = {
        'experiment': 'E202 strict leave-one-subject-out (20 folds)',
        'protocol': 'held-out subject excluded from SVD/prototypes/gate; gate unknown from inner split',
        'mean_loso_rejection': float(values.mean()),
        'min_loso_rejection': float(values.min()),
        'std_loso_rejection': float(values.std()),
        'mean_registration_id_accuracy': float(np.mean([r['registration_id_accuracy'] for r in rows])),
        'mean_known_accept': float(np.mean([r['known_accept'] for r in rows])),
        'folds': rows,
    }
    (ROOT / 'loso20.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'folds'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
