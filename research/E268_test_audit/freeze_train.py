"""Freeze the complete E266 verifier and E267 guarded gate on Train only."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
e267 = runpy.run_path(str(H.parent / 'E267_guarded_gate' / 'run.py'))
e266 = e267['e266']
e264 = e267['e264']
r = e267['r']
D = r.D
RECIPE = dict(
    stage='E268 full-Train freeze; no late-segment evaluation and no Test access',
    fit='all people 1..20; r.f <= .6',
    calibration='all people 1..20; .6 < r.f <= .8; strict E267 inner folds; early60 art/prototypes',
    deployment='projection and prototypes refit on all 11304 Train rows after threshold freeze',
    network=dict(e266['RECIPE'], held=0,
                 evaluation='not evaluated in E268; Test not accessed'),
    gate=dict(e267['RECIPE'], network='E266 train_verifier full-Train fit, held=0',
              late='not evaluated in E268; Test not accessed',
              evaluation='not evaluated in E268; Test not accessed'),
    feature_source='final2/team_hymn/weights.pt: feature dictionary only, read weights_only=True',
)


def save_json(name, obj):
    (H / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    if D != 'cuda':
        raise RuntimeError('E268 requires CUDA')
    fit = r.f <= .6
    cal = (r.f > .6) & (r.f <= .8)
    people = list(range(1, 21))
    clean = r.tens(np.load(r.CACHE / 'e253_subwindow' / 'train_feat.npy'))
    augmented = r.tens(np.load(r.CACHE / 'e264_subwindow_verifier' / 'train_aug.npy'))
    if clean.shape != (11304, 6, 855) or augmented.shape != clean.shape:
        raise ValueError('Unexpected Train feature cache shape')

    print('E268 training full E266 network on', torch.cuda.get_device_name(), flush=True)
    model = e266['train_verifier'](clean, augmented, fit, people, held=0)
    print('E266 network trained; projecting early60 calibration artifacts', flush=True)
    calibration_q, calibration_art = r.project(fit)
    calibration_q_sw = e264['project_subwindows'](clean, calibration_art)
    if not torch.allclose(calibration_q_sw.mean(1), calibration_q, atol=2e-5, rtol=2e-5):
        raise ValueError('Calibration subwindow mean projection mismatch')
    calibration_p, calibration_valid = e264['package_prototypes'](
        calibration_q_sw.mean(1), fit, people, calibration_art)

    rows = e267['inner_calibration'](clean, fit, cal, people, model)
    picked, diagnostics = e267['choose'](rows)
    inner_folds = []
    for inner, (registered, (base, candidate, known)) in enumerate(
            zip(e266['inner_people'](people), rows)):
        rates = e267['grid_rates']([(base, candidate, known)], picked['delta'], picked['veto'])
        inner_folds.append(dict(
            inner=inner,
            registered=registered,
            n_known=int(known.sum()),
            n_unknown=int((~known).sum()),
            base_tau=float(torch.quantile(base[known], .1)),
            candidate_tau=float(torch.quantile(candidate[known], .1)),
            veto_tau=(float(torch.quantile(candidate[known], picked['veto']))
                      if picked['veto'] else -1e30),
            known_accept=rates['known_accept_mean'],
            unknown_reject=rates['unknown_reject_mean'],
            base_known_accept=rates['base_known_accept_mean'],
            base_unknown_reject=rates['base_unknown_reject_mean'],
        ))
    del rows

    base, candidate, _ = e266['infer'](
        calibration_q_sw, calibration_p, calibration_valid, cal, people, model)
    base_tau = float(torch.quantile(base, .1))
    candidate_tau = float(torch.quantile(candidate, .1))
    veto_tau = (float(torch.quantile(candidate, picked['veto']))
                if picked['veto'] else -1e30)

    print('Thresholds frozen; fitting full-Train deployment artifacts', flush=True)
    full = torch.ones_like(fit)
    assert int(full.sum()) == 11304
    q, art = r.project(full)
    q_sw = e264['project_subwindows'](clean, art)
    if not torch.allclose(q_sw.mean(1), q, atol=2e-5, rtol=2e-5):
        raise ValueError('Deployment subwindow mean projection mismatch')
    p, valid = e264['package_prototypes'](q_sw.mean(1), full, people, art)
    calibration = dict(
        recipe=RECIPE,
        train_samples=int(fit.sum()),
        calibration_samples=int(cal.sum()),
        deployment_samples=int(full.sum()),
        people=people,
        picked=picked,
        diagnostics=diagnostics,
        inner_folds=inner_folds,
        outer=dict(
            base_quantile=.1,
            candidate_quantile=.1,
            veto_quantile=picked['veto'],
            base_tau=base_tau,
            candidate_tau=candidate_tau,
            veto_tau=veto_tau,
        ),
        test_accessed=False,
    )
    save_json('calibration.json', calibration)

    source = torch.load(ROOT / 'final2' / 'team_hymn' / 'weights.pt',
                        map_location='cpu', weights_only=True)
    feature = source['feature']
    del source
    bundle = dict(
        feature=feature,
        projection={key: value.detach().cpu() for key, value in art.items()},
        prototypes=p.detach().cpu(),
        valid=valid.detach().cpu(),
        calibration_projection={key: value.detach().cpu()
                                for key, value in calibration_art.items()},
        calibration_prototypes=calibration_p.detach().cpu(),
        calibration_valid=calibration_valid.detach().cpu(),
        model={key: value.detach().cpu() for key, value in model.state_dict().items()},
        base_tau=base_tau,
        candidate_tau=candidate_tau,
        veto_tau=veto_tau,
        delta=float(picked['delta']),
        veto=float(picked['veto']),
        people=people,
        tasks=sorted(r.M.task.unique().tolist()),
        recipe=RECIPE,
        calibration=calibration,
    )
    weights_path = H / 'weights.pt'
    torch.save(bundle, weights_path)
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    loaded = torch.load(weights_path, map_location='cpu', weights_only=True)
    required = {'feature', 'projection', 'prototypes', 'valid', 'calibration_projection',
                'calibration_prototypes', 'calibration_valid', 'model', 'base_tau',
                'candidate_tau', 'veto_tau', 'delta', 'veto', 'people', 'tasks',
                'recipe', 'calibration'}
    assert required <= loaded.keys()
    assert loaded['feature'] == feature and loaded['calibration'] == calibration
    assert loaded['model'].keys() == model.state_dict().keys()
    assert all(torch.isfinite(value).all() for value in loaded['model'].values())
    summary = dict(
        train_samples=int(fit.sum()),
        calibration_samples=int(cal.sum()),
        deployment_samples=int(full.sum()),
        selected_parameters=dict(
            delta=float(picked['delta']),
            veto=float(picked['veto']),
            base_tau=base_tau,
            candidate_tau=candidate_tau,
            veto_tau=veto_tau,
        ),
        weights_sha256=weights_sha256,
        weights_only_load_verified=True,
        gpu=torch.cuda.get_device_name(),
        test_accessed=False,
    )
    save_json('freeze_summary.json', summary)
    print('E268_FINAL ' + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
