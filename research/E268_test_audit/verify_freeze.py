"""Independently verify the Train-only E268 freeze without reading Test data."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parent
e267 = runpy.run_path(str(H.parent / 'E267_guarded_gate' / 'run.py'))
e266 = e267['e266']
e264 = e267['e264']
r = e267['r']
D = r.D


def assert_close(name, actual, expected):
    if actual.dtype == torch.bool or expected.dtype == torch.bool:
        assert torch.equal(actual, expected), f'{name} mismatch'
        return
    if not torch.allclose(actual, expected, atol=2e-5, rtol=2e-5):
        diff = float((actual - expected).abs().max())
        raise AssertionError(f'{name} mismatch; max abs diff={diff}')


def assert_projection(name, actual, expected):
    assert actual.keys() == expected.keys(), f'{name} keys mismatch'
    for key in actual:
        assert_close(f'{name}.{key}', actual[key], expected[key])


@torch.inference_mode()
def main():
    if D != 'cuda':
        raise RuntimeError('E268 verification requires CUDA')
    weights_path = H / 'weights.pt'
    bundle = torch.load(weights_path, map_location=D, weights_only=True)
    calibration = json.loads((H / 'calibration.json').read_text(encoding='utf-8'))
    summary = json.loads((H / 'freeze_summary.json').read_text(encoding='utf-8'))
    assert bundle['calibration'] == calibration
    sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    assert sha256 == summary['weights_sha256']

    model = e266['NestedVerifier']().to(D)
    model.load_state_dict(bundle['model'], strict=True)
    model.eval()
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())

    fit = r.f <= .6
    cal = (r.f > .6) & (r.f <= .8)
    full = torch.ones_like(fit)
    people = list(range(1, 21))
    assert bundle['people'] == people
    assert int(fit.sum()) == calibration['train_samples'] == summary['train_samples']
    assert int(cal.sum()) == calibration['calibration_samples'] == summary['calibration_samples']
    assert int(full.sum()) == 11304
    assert calibration['deployment_samples'] == summary['deployment_samples'] == 11304

    clean = r.tens(np.load(r.CACHE / 'e253_subwindow' / 'train_feat.npy'))
    assert clean.shape == (11304, 6, 855)

    q_full, art_full = r.project(full)
    q_full_sw = e264['project_subwindows'](clean, art_full)
    assert_close('full subwindow mean', q_full_sw.mean(1), q_full)
    p_full, valid_full = e264['package_prototypes'](
        q_full_sw.mean(1), full, people, art_full)
    assert_projection('deployment projection', bundle['projection'], art_full)
    assert_close('deployment prototypes', bundle['prototypes'], p_full)
    assert_close('deployment valid', bundle['valid'], valid_full)
    del q_full, q_full_sw, p_full, valid_full, art_full

    q_cal, art_cal = r.project(fit)
    q_cal_sw = e264['project_subwindows'](clean, art_cal)
    assert_close('calibration subwindow mean', q_cal_sw.mean(1), q_cal)
    p_cal, valid_cal = e264['package_prototypes'](
        q_cal_sw.mean(1), fit, people, art_cal)
    assert_projection('calibration projection', bundle['calibration_projection'], art_cal)
    assert_close('calibration prototypes', bundle['calibration_prototypes'], p_cal)
    assert_close('calibration valid', bundle['calibration_valid'], valid_cal)
    del q_cal, q_cal_sw, p_cal, valid_cal, art_cal

    saved_cal_q_sw = e264['project_subwindows'](clean, bundle['calibration_projection'])
    base, candidate, _ = e266['infer'](
        saved_cal_q_sw, bundle['calibration_prototypes'], bundle['calibration_valid'],
        cal, people, model)
    veto = float(bundle['veto'])
    reproduced = dict(
        base_tau=float(torch.quantile(base, .1)),
        candidate_tau=float(torch.quantile(candidate, .1)),
        veto_tau=float(torch.quantile(candidate, veto)) if veto else -1e30,
    )
    threshold_diffs = {}
    for key, value in reproduced.items():
        threshold_diffs[key] = abs(value - float(bundle[key]))
        assert np.isclose(value, bundle[key], atol=1e-6, rtol=1e-7), key
        assert bundle[key] == calibration['outer'][key]
        assert bundle[key] == summary['selected_parameters'][key]
    assert bundle['delta'] == calibration['picked']['delta'] == summary['selected_parameters']['delta']
    assert veto == calibration['picked']['veto'] == summary['selected_parameters']['veto']

    result = dict(
        status='ok',
        train_samples=int(fit.sum()),
        calibration_samples=int(cal.sum()),
        deployment_samples=int(full.sum()),
        delta=float(bundle['delta']),
        veto=veto,
        thresholds=reproduced,
        threshold_max_abs_diff=max(threshold_diffs.values()),
        weights_sha256=sha256,
        gpu=torch.cuda.get_device_name(),
        weights_only_load_verified=True,
        model_load_state_dict_verified=True,
        full_train_deployment_artifacts_verified=True,
        early60_calibration_artifacts_verified=True,
        mid_thresholds_reproduced=True,
        test_accessed=False,
        auc_f1='not computed; freeze verification only',
    )
    print('E268_VERIFY ' + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
