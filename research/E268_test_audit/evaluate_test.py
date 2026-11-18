"""E268：全 Train 权重冻结后，按 E254 历史口径一次测量 Test #9/#10。"""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
e267 = runpy.run_path(str(H.parent / 'E267_guarded_gate' / 'run.py'))
e266 = e267['e266']
e264 = e266['e264']


def project_subwindows(x, art, tasks):
    z = (x - art['mu'][tasks][:, None]) / art['sd'][tasks][:, None]
    return ((z - art['center']) @ art['basis']) @ art['white']


@torch.inference_mode()
def main():
    weight_path = H / 'weights.pt'
    expected = json.loads((H / 'freeze_summary.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == expected['weights_sha256']
    b = torch.load(weight_path, map_location='cpu', weights_only=True)
    assert b['recipe']['test_accessed'] is False if 'test_accessed' in b['recipe'] else True
    required = ['projection', 'prototypes', 'valid', 'model', 'base_tau',
                'candidate_tau', 'veto_tau', 'delta', 'veto']
    assert all(key in b for key in required)
    assert 'calibration_projection' in b and 'calibration_prototypes' in b
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art = {key: value.to(device) for key, value in b['projection'].items()}
    p = b['prototypes'].to(device)
    valid = b['valid'].to(device)
    model = e266['NestedVerifier']().to(device)
    model.load_state_dict(b['model']); model.eval()

    meta = pd.read_csv(ROOT / 'BrainPrint_cache/test_metadata.csv')
    labels = pd.read_csv(ROOT / 'Brainprint_Competition/test.csv')
    meta = meta.merge(labels, left_on='sample', right_on='Sample', how='left', validate='one_to_one')
    assert len(meta) == 22251 and not meta['Label'].isna().any()
    x = torch.from_numpy(np.load(ROOT / 'BrainPrint_cache/e253_subwindow/test_feat.npy')).to(device)
    assert x.shape == (22251, 6, 855)
    task_names = sorted(e266['r'].M.task.unique().tolist())
    task_map = {name: i for i, name in enumerate(task_names)}
    tasks = torch.as_tensor(meta['task'].map(task_map).to_numpy(), device=device)
    assert int((tasks >= 0).all()) and b['people'] == list(range(1, 21))

    scores = x.new_full((len(x), 20), -torch.inf)
    for task in range(4):
        take = tasks == task
        if not take.any():
            continue
        q_sw = project_subwindows(x[take], art, tasks[take])
        query = normalize(q_sw.mean(1), dim=-1)
        sim = (query[:, None, None] * p[task][None]).sum(-1).masked_fill(~valid[task][None], -torch.inf).amax(-1)
        scores[take] = sim
    claimed = scores.argmax(1)
    top = scores.topk(2, dim=1).values
    base = top[:, 0] + .5 * (top[:, 0] - top[:, 1])
    assert torch.allclose(base, e264['candidate_gate'](scores, claimed))

    # 分批仅用于控制显存；每行评分不依赖批次或其他查询。
    candidates = []
    for start in range(0, len(x), 512):
        stop = min(start + 512, len(x))
        q_sw = project_subwindows(x[start:stop], art, tasks[start:stop])
        proof = e264['evidence'](q_sw, tasks[start:stop], p, claimed[start:stop])
        cohort = e266['cohort_features'](scores[start:stop], claimed[start:stop])
        candidates.append(e266['verifier_scores'](base[start:stop], proof, cohort, model))
    candidate = torch.cat(candidates)
    accept = e267['guarded_accept'](
        base, candidate, float(b['base_tau']), float(b['candidate_tau']),
        float(b['veto_tau']), float(b['delta'])).cpu().numpy()
    base_accept = (base >= float(b['base_tau'])).cpu().numpy()
    pred = (claimed + 1).cpu().numpy()
    label = meta['Label'].to_numpy()
    registered = label > 0
    unknown = label == 0
    assert int(registered.sum()) == 19218 and int(unknown.sum()) == 3033

    def rates(accepted):
        correct = (pred == label) & accepted
        return dict(
            registered_accept=float(accepted[registered].mean()),
            registered_top1_given_accept=float((pred[registered & accepted] == label[registered & accepted]).mean()),
            registered_e2e=float(correct[registered].mean()),
            unregistered_reject=float((~accepted[unknown]).mean()),
            accepted_registered=int(accepted[registered].sum()),
            rejected_unknown=int((~accepted[unknown]).sum()),
        )

    result = dict(
        protocol='E254 audit_e254 Test labels; frozen E266/E267 full-Train deployment',
        test_rows=len(meta), registered_rows=int(registered.sum()), unknown_rows=int(unknown.sum()),
        weights_sha256=expected['weights_sha256'],
        delta=float(b['delta']), veto=float(b['veto']),
        base_tau=float(b['base_tau']), candidate_tau=float(b['candidate_tau']), veto_tau=float(b['veto_tau']),
        e267=rates(accept), same_refit_base=rates(base_accept),
        final2_historical=dict(registered_accept=0.22416484545738372,
                               registered_top1_given_accept=0.930594243268338,
                               unregistered_reject=0.5918232772832179),
        selection='one frozen candidate; no post-Test threshold or model changes',
    )
    (H / 'test_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(H / 'test_decisions.npz', sample=meta['sample'].to_numpy(), label=label,
                        prediction=pred, accept=accept, base_accept=base_accept,
                        base=base.cpu().numpy(), candidate=candidate.cpu().numpy())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
