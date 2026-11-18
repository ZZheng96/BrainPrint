"""E267：复用 E266 深度权重，仅以中段选择恢复接受和拒识保护。"""
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parent
PREV = H.parent / 'E266_strict_nested'
e266 = runpy.run_path(str(PREV / 'run.py'))
r = e266['r']
e264 = e266['e264']
D = r.D
CORE = e266['CORE']
DELTA_GRID = [0., .005, .01, .02, .04, .08]
VETO_GRID = [0., .025, .05, .075, .10]
RECIPE = dict(
    network='E266 heldXX.pt, no retraining', candidate_quantile=.10,
    delta_grid=DELTA_GRID, veto_quantile_grid=VETO_GRID,
    rule='(base_accept & candidate>=veto) | (candidate_accept & base>=base_tau-delta)',
    calibration='strict five inner folds, clean mid only; mean and worst inner rates must not decrease',
    selection='maximize mean known accept, then mean unknown reject, then smaller delta and veto',
    late='evaluate once on clean and independent stress; Test not accessed',
)


def save_json(name, obj):
    (H / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def guarded_accept(base, candidate, base_tau, candidate_tau, veto_tau, delta):
    rescued = (candidate >= candidate_tau) & (base >= base_tau - delta)
    retained = (base >= base_tau) & (candidate >= veto_tau)
    return retained | rescued


def grid_rates(rows, delta, veto):
    out = []
    for base, candidate, known in rows:
        base_tau = torch.quantile(base[known], .1)
        candidate_tau = torch.quantile(candidate[known], .1)
        veto_tau = torch.quantile(candidate[known], veto) if veto else torch.tensor(-torch.inf, device=D)
        accept = guarded_accept(base, candidate, base_tau, candidate_tau, veto_tau, delta)
        base_accept = base >= base_tau
        out.append(dict(
            known_accept=float(accept[known].float().mean()),
            unknown_reject=float((~accept[~known]).float().mean()),
            base_known_accept=float(base_accept[known].float().mean()),
            base_unknown_reject=float((~base_accept[~known]).float().mean()),
        ))
    summary = dict(delta=delta, veto=veto)
    for key in ['known_accept', 'unknown_reject', 'base_known_accept', 'base_unknown_reject']:
        vals = [row[key] for row in out]
        summary[key + '_mean'] = float(np.mean(vals))
        summary[key + '_worst'] = float(min(vals))
    summary['eligible'] = bool(
        summary['known_accept_mean'] >= summary['base_known_accept_mean'] - 1e-12 and
        summary['unknown_reject_mean'] >= summary['base_unknown_reject_mean'] - 1e-12 and
        summary['known_accept_worst'] >= summary['base_known_accept_worst'] - 1e-12 and
        summary['unknown_reject_worst'] >= summary['base_unknown_reject_worst'] - 1e-12
    )
    return summary


def choose(rows):
    diagnostics = [grid_rates(rows, delta, veto) for delta in DELTA_GRID for veto in VETO_GRID]
    eligible = [d for d in diagnostics if d['eligible']]
    assert eligible, 'delta=0,veto=0 must reproduce the package'
    picked = max(eligible, key=lambda d: (
        d['known_accept_mean'], d['unknown_reject_mean'], -d['delta'], -d['veto']))
    return picked, diagnostics


@torch.inference_mode()
def inner_calibration(clean, fit, cal, people, model):
    rows = []
    for registered in e266['inner_people'](people):
        enrolled = torch.isin(r.y, r.tens(registered))
        inner_fit = fit & enrolled
        _, art = r.project(inner_fit)
        q = e264['project_subwindows'](clean, art)
        p, valid = e264['package_prototypes'](q.mean(1), inner_fit, registered, art)
        base, candidate, _ = e266['infer'](q, p, valid, cal, registered, model)
        known = enrolled[cal]
        assert not bool(torch.isin(r.y[cal][~known], r.tens(registered)).any())
        rows.append((base, candidate, known))
        del q, p, valid, art
    return rows


def checkpoint_path(held):
    return H / 'weights' / f'held{held:02d}.json'


def load_checkpoint(held):
    path = checkpoint_path(held)
    if not path.exists() or not (PREV / 'weights' / f'held{held:02d}.pt').exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        assert obj['recipe'] == RECIPE and obj['fold']['held'] == held
        assert obj['threshold']['held'] == held
        assert set(obj['fold']) == {f'{v}_{a}' for v in ['clean', 'stress'] for a in ['package', 'guarded']} | {'held'}
        return obj
    except (AssertionError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


@torch.inference_mode()
def run_fold(clean, stress, held):
    b = torch.load(PREV / 'weights' / f'held{held:02d}.pt', map_location=D, weights_only=True)
    assert b['held'] == held and b['recipe'] == e266['RECIPE']
    model = e266['NestedVerifier']().to(D)
    model.load_state_dict(b['model']); model.eval()
    people = b['people']
    fit = (r.f <= .6) & (r.y != held)
    cal = (r.f > .6) & (r.f <= .8) & (r.y != held)
    late = r.f > .8
    rows = inner_calibration(clean, fit, cal, people, model)
    picked, diagnostics = choose(rows)
    del rows
    q = e264['project_subwindows'](clean, b['projection'])
    s = e264['project_subwindows'](stress, b['projection'])
    p, valid = b['prototypes'], b['valid']
    cal_base, cal_candidate, _ = e266['infer'](q, p, valid, cal, people, model)
    base_tau = float(torch.quantile(cal_base, .1))
    candidate_tau = float(torch.quantile(cal_candidate, .1))
    veto_tau = float(torch.quantile(cal_candidate, picked['veto'])) if picked['veto'] else -1e30
    threshold = dict(held=held, **picked, diagnostics=diagnostics,
                     base_tau=base_tau, candidate_tau=candidate_tau, veto_tau=veto_tau)
    row = dict(held=held)
    truth = r.y[late].cpu().numpy()
    for view, projected in [('clean', q), ('stress', s)]:
        base, candidate, pred = e266['infer'](projected, p, valid, late, people, model)
        package_accept = base >= base_tau
        guarded = guarded_accept(base, candidate, base_tau, candidate_tau, veto_tau, picked['delta'])
        row[f'{view}_package'] = e266['metrics'](base, package_accept.float(), pred, truth, held, .5)
        row[f'{view}_guarded'] = e266['metrics'](base, guarded.float(), pred, truth, held, .5)
    assert np.allclose(base_tau, b['base_tau']) and np.allclose(candidate_tau, b['candidate_tau'])
    obj = dict(recipe=RECIPE, fold=row, threshold=threshold)
    save_json(str(Path('weights') / f'held{held:02d}.json'), obj)
    return obj


def summarize(folds):
    return {arm: {key: dict(mean=float(np.mean(vals)), worst=float((max if key == 'accepted_misrank' else min)(vals)))
                  for key in CORE for vals in [[fold[arm][key] for fold in folds]]}
            for arm in folds[0] if arm != 'held'}


def report(obj):
    s = obj['summary']
    def stat(arm, key):
        return f"{s[arm][key]['mean']:.5f} / {s[arm][key]['worst']:.5f}"
    def delta(view, key):
        return 100 * (s[f'{view}_guarded'][key]['mean'] - s[f'{view}_package'][key]['mean'])
    lines = [
        '# E267：E266 深度门控的恢复接受与拒识保护', '',
        '**联合判定：' + ('通过' if obj['passed'] else '不通过；保留 final2，不读取 Test') + '。**', '',
        '仅复用 E266 已训练权重。delta/veto 均由严格内层中段网格选择；晚段不用于选择。表中为20折均值/最差折。', '',
        '|视图与规则|' + '|'.join(CORE) + '|', '|---|' + '---:|' * len(CORE)]
    for arm in s:
        lines.append('|' + arm + '|' + '|'.join(stat(arm, key) for key in CORE) + '|')
    lines += ['', '候选减同折现包（百分点）：', '', '|视图|' + '|'.join(CORE) + '|', '|---|' + '---:|' * len(CORE)]
    for view in ['clean', 'stress']:
        lines.append('|' + view + '|' + '|'.join(f'{delta(view, key):+.4f}' for key in CORE) + '|')
    lines += ['', f"delta 分布：{dict(Counter(t['delta'] for t in obj['thresholds']))}；veto 分布：{dict(Counter(t['veto'] for t in obj['thresholds']))}。", '',
              '## 11 项指标', '', '|#|指标|数值与证据性质|', '|---:|---|---|',
              '|1|Train 样本内 acc|0.9996461429582448（现包历史值；E267未重算）|',
              '|2|OOF acc|0.9988499646142959（现包历史值；E267未重算）|',
              '|3|OOF macro-F1|0.9988253655342916（现包历史值；E267未重算）|',
              '|4|forward acc|0.9982509838216004（现包历史值；E267未重算）|',
              f"|5|strict LOSO AUROC|{s['clean_guarded']['AUROC']['mean']:.12f}（E267 Train晚段20折均值）|",
              f"|6|U@95|{s['clean_guarded']['U95']['mean']:.12f}（E267 Train晚段20折均值）|",
              '|7|RA|N/A|', '|8|ASafeR|N/A|',
              '|9|Test S2|0.22416484545738372（现包历史值；E267未审计）|',
              '|10|Test S1|0.5918232772832179（现包历史值；E267未审计）|',
              f"|11|strict LOSO rej@90|{s['clean_guarded']['R90']['mean']:.12f}（E267 Train晚段20折均值）|", '',
              '## 冻结规则', '',
              '- 对每个 inner fold，base/candidate 阈值分别为已知中段分的 P10；veto 是候选已知分的 v 分位，v=0 表示不否决。最终接受为 `(base接受 且 candidate>=veto) 或 (candidate接受 且 base>=base_tau-delta)`。',
              '- delta∈[0,.005,.01,.02,.04,.08]，veto∈[0,.025,.05,.075,.10]。仅当 inner known accept 与 unknown reject 的均值和最差均不低于 base 时合格；依次最大化 known accept 均值、unknown reject 均值，再取较小 delta/veto。delta=0,veto=0 精确复现 base，保证有可行解。',
              '- 深度网络和各折 outer/inner 投影完全复用 E266；stress 不参与校准。AUROC/U@95/rej@90 仍用 base 连续分，F1仍用 base argmax。', '',
              f"- ROC完全一致：{obj['roc_identical']}；F1完全一致：{obj['f1_identical']}；两视图均值/最差不降：{obj['rates_ok']}；known accept均值严格提升：{obj['improvement_ok']}；错排接受不增：{obj['safety_ok']}。Test未读取：{not obj['test_accessed']}。", '',
              '复跑：`..\\.venv\\Scripts\\python.exe -B research/E267_guarded_gate/run.py`。']
    text = '\n'.join(lines) + '\n'
    (H / 'REPORT.md').write_text(text, encoding='utf-8')
    print(text)


@torch.inference_mode()
def main():
    (H / 'weights').mkdir(exist_ok=True)
    cache = e264['r'].CACHE / 'e264_subwindow_verifier'
    clean = r.tens(np.load(e264['r'].CACHE / 'e253_subwindow' / 'train_feat.npy'))
    stress = r.tens(np.load(cache / 'stress.npy'))
    saved = [load_checkpoint(h) for h in r.ids]
    if all(obj is not None for obj in saved):
        folds = [obj['fold'] for obj in saved]; thresholds = [obj['threshold'] for obj in saved]
    else:
        folds, thresholds = [], []
        for h, obj in zip(r.ids, saved):
            if obj is None:
                obj = run_fold(clean, stress, h)
            folds.append(obj['fold']); thresholds.append(obj['threshold'])
            print('held', h, 'delta', obj['threshold']['delta'], 'veto', obj['threshold']['veto'], flush=True)
    folds.sort(key=lambda x: x['held']); thresholds.sort(key=lambda x: x['held'])
    summary = summarize(folds)
    def no_worse(metric, greater=False):
        return all((f[f'{v}_guarded'][metric] >= f[f'{v}_package'][metric] - 1e-12) if not greater else
                   (f[f'{v}_guarded'][metric] <= f[f'{v}_package'][metric] + 1e-12)
                   for f in folds for v in ['clean', 'stress'])
    roc_identical = all(no_worse(k) and no_worse(k, True) for k in ['AUROC', 'U95', 'R90'])
    f1_identical = no_worse('F1') and no_worse('F1', True)
    rates_ok = all(no_worse(k) for k in ['old_e2e', 'known_accept', 'unknown_reject'])
    improvement_ok = all(summary[f'{v}_guarded']['known_accept']['mean'] > summary[f'{v}_package']['known_accept']['mean'] for v in ['clean', 'stress'])
    safety_ok = no_worse('accepted_misrank', True)
    obj = dict(recipe=RECIPE, folds=folds, thresholds=thresholds, summary=summary,
               roc_identical=roc_identical, f1_identical=f1_identical, rates_ok=rates_ok,
               improvement_ok=improvement_ok, safety_ok=safety_ok,
               passed=roc_identical and f1_identical and rates_ok and improvement_ok and safety_ok,
               test_accessed=False)
    save_json('results.json', obj); report(obj)


if __name__ == '__main__':
    main()
