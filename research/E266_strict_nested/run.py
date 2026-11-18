"""E266 strict nested Train-only verifier with inner-specific projections."""
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

H = Path(__file__).resolve().parent
PREV = H.parent / 'E264_subwindow_verifier'
e264 = runpy.run_path(str(PREV / 'run.py'))
r = e264['r']
D = r.D
CORE = ['AUROC', 'U95', 'R90', 'F1', 'old_e2e', 'known_accept', 'unknown_reject', 'accepted_misrank']
Q_GRID = [.05, .10, .15, .20, .25, .30]
RECIPE = dict(
    seed=26600, epochs=24, batch_size=512, lr=.002, weight_decay=1e-4,
    encoder='4-32-32 shared DeepSets', pooling='mean+min+max',
    global_features='E264 window-max mean/std/min/max + per-prototype mean/std',
    cohort_features='s1,s2,margin,gate,mean,std,s1-mean,(s1-mean)/std',
    cohort_mlp='8-16-16', fusion='108-48 and cohort-16; 64-1 zero-initialized residual',
    logit='12*(base_gate-0.33)+residual', loss='BCEWithLogitsLoss balanced pos_weight',
    inner_enrollment_folds=5, quantile_grid=Q_GRID,
    inner_projection='refit on each inner registered identities early 60% only',
    train_views=['clean', 'train_aug'], calibration='inner clean mid with inner_art; outer clean mid only',
    roc_score='original base gate in both arms', ranking='original base argmax in both arms',
    evaluation='late clean and independent stress; no Test access',
)


def save_json(name, obj):
    (H / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def inner_people(people):
    return [[person for j, person in enumerate(people) if j % 5 != inner] for inner in range(5)]


def parity_halves(fit, people):
    parity = torch.full_like(r.y, -1)
    for person in people:
        for task in range(4):
            ix = torch.where(fit & (r.y == person) & (r.t == task))[0]
            ix = ix[r.order[ix].argsort()]
            parity[ix] = torch.arange(len(ix), device=D) % 2
    return parity


def cohort_features(scores, claimed):
    top = scores.topk(2, dim=1).values
    s1, s2 = top[:, 0], top[:, 1]
    margin = s1 - s2
    gate = e264['candidate_gate'](scores, claimed)
    mean = scores.mean(1)
    std = scores.std(1, unbiased=False).clamp_min(1e-6)
    centered = s1 - mean
    return torch.stack([s1, s2, margin, gate, mean, std, centered, centered / std], dim=1)


def examples(q_sw, rows, people, p, valid):
    scores = e264['identity_scores'](q_sw.mean(1), p, valid, rows)
    claimed = scores.argmax(1)
    pred = r.tens(people)[claimed]
    labels = (pred == r.y[rows]).float()
    gate = e264['candidate_gate'](scores, claimed)
    proof = e264['evidence'](q_sw[rows], r.t[rows], p, claimed)
    cohort = cohort_features(scores, claimed)
    return gate, proof, cohort, labels, pred


class NestedVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.deep = nn.Sequential(nn.Linear(108, 48), nn.SiLU())
        self.cohort = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU())
        self.residual = nn.Linear(64, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x, cohort):
        h = self.encoder(x)
        pooled = torch.cat([h.mean(1), h.amin(1), h.amax(1)], dim=1)
        window_max = x.amax(-1)
        stats = torch.stack([
            window_max.mean(1), window_max.std(1, unbiased=False),
            window_max.amin(1), window_max.amax(1),
        ], dim=1)
        proto_stats = torch.cat([x.mean(1), x.std(1, unbiased=False)], dim=1)
        deep = self.deep(torch.cat([pooled, stats, proto_stats], dim=1))
        return self.residual(torch.cat([deep, self.cohort(cohort)], dim=1)).squeeze(1)


def verifier_scores(gate, proof, cohort, model):
    return 12 * (gate - .33) + model(proof, cohort)


def train_verifier(clean, augmented, fit, people, held):
    parity = parity_halves(fit, people)
    gates, proofs, cohorts, labels = [], [], [], []
    for registered in inner_people(people):
        enrolled = torch.isin(r.y, r.tens(registered))
        inner_fit = fit & enrolled
        _, inner_art = r.project(inner_fit)
        inner_clean = e264['project_subwindows'](clean, inner_art)
        inner_augmented = e264['project_subwindows'](augmented, inner_art)
        for half in (0, 1):
            template_rows = inner_fit & (parity != half)
            query_rows = fit & (parity == half)
            p, valid = e264['package_prototypes'](
                inner_clean.mean(1), template_rows, registered, inner_art)
            for q_sw in (inner_clean, inner_augmented):
                a, b, c, d, _ = examples(q_sw, query_rows, registered, p, valid)
                gates.append(a)
                proofs.append(b)
                cohorts.append(c)
                labels.append(d)
        del p, valid, q_sw, inner_clean, inner_augmented, inner_art
        torch.cuda.empty_cache()
    gates = torch.cat(gates)
    proofs = torch.cat(proofs)
    cohorts = torch.cat(cohorts)
    labels = torch.cat(labels)
    pos_weight = ((len(labels) - labels.sum()) / labels.sum()).detach()
    torch.manual_seed(RECIPE['seed'] + held)
    model = NestedVerifier().to(D)
    opt = torch.optim.AdamW(
        model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    for _ in range(RECIPE['epochs']):
        order = torch.randperm(len(labels), device=D)
        for ix in order.split(RECIPE['batch_size']):
            logits = verifier_scores(gates[ix], proofs[ix], cohorts[ix], model)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels[ix], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()


@torch.inference_mode()
def infer(q_sw, p, valid, rows, people, model):
    gate, proof, cohort, _, pred = examples(q_sw, rows, people, p, valid)
    candidate = verifier_scores(gate, proof, cohort, model)
    return gate, candidate, pred


def metrics(base, decision, pred, truth, held, tau):
    base_np = base.cpu().numpy()
    pred_np = pred.cpu().numpy()
    row = e264['e262']['metrics'](base_np, pred_np, truth, held, 0.)
    known = truth != held
    correct = pred_np == truth
    accept = decision.cpu().numpy() >= tau
    out = dict(
        AUROC=row['AUC'], U95=row['U95'], R90=row['R90'], F1=row['F1'],
        old_e2e=float((correct & accept)[known].mean()),
        known_accept=float(accept[known].mean()),
        unknown_reject=float((~accept)[~known].mean()),
        accepted_misrank=float((accept & ~correct)[known].mean()),
    )
    return {k: out[k] for k in CORE}


@torch.inference_mode()
def calibrate(clean, fit, cal, people, model):
    inner_rows = []
    per_q = {q: dict(known_accept=[], unknown_reject=[]) for q in Q_GRID}
    base_known_rates, base_unknown_rates = [], []
    for inner, registered in enumerate(inner_people(people)):
        enrolled = torch.isin(r.y, r.tens(registered))
        inner_fit = fit & enrolled
        _, inner_art = r.project(inner_fit)
        inner_clean = e264['project_subwindows'](clean, inner_art)
        p, valid = e264['package_prototypes'](
            inner_clean.mean(1), inner_fit, registered, inner_art)
        base, candidate, _ = infer(inner_clean, p, valid, cal, registered, model)
        known = enrolled[cal]
        unknown = ~known
        base_tau = float(torch.quantile(base[known], .10))
        base_known = float((base[known] >= base_tau).float().mean())
        base_unknown = float((base[unknown] < base_tau).float().mean())
        base_known_rates.append(base_known)
        base_unknown_rates.append(base_unknown)
        candidates = []
        for q in Q_GRID:
            tau = float(torch.quantile(candidate[known], q))
            known_accept = float((candidate[known] >= tau).float().mean())
            unknown_reject = float((candidate[unknown] < tau).float().mean())
            per_q[q]['known_accept'].append(known_accept)
            per_q[q]['unknown_reject'].append(unknown_reject)
            candidates.append(dict(
                q=q, tau=tau, known_accept=known_accept, unknown_reject=unknown_reject))
        inner_rows.append(dict(
            inner=inner, registered=registered, n_known=int(known.sum()), n_unknown=int(unknown.sum()),
            base_tau=base_tau, base_known_accept=base_known, base_unknown_reject=base_unknown,
            candidates=candidates,
        ))
        del p, valid, base, candidate, inner_clean, inner_art
        torch.cuda.empty_cache()
    base_known = float(np.mean(base_known_rates))
    base_unknown = float(np.mean(base_unknown_rates))
    diagnostics = []
    for q in Q_GRID:
        known_accept = float(np.mean(per_q[q]['known_accept']))
        unknown_reject = float(np.mean(per_q[q]['unknown_reject']))
        diagnostics.append(dict(
            q=q, known_accept=known_accept, unknown_reject=unknown_reject,
            base_known_accept=base_known, base_unknown_reject=base_unknown,
            eligible=known_accept >= base_known - 1e-12 and unknown_reject >= base_unknown - 1e-12,
        ))
    eligible = [row for row in diagnostics if row['eligible']]
    if eligible:
        selected = max(eligible, key=lambda row: (
            row['known_accept'], row['unknown_reject'], -row['q']))
        fallback = False
    else:
        selected = max(diagnostics, key=lambda row: (
            min(row['known_accept'] - row['base_known_accept'],
                row['unknown_reject'] - row['base_unknown_reject']), -row['q']))
        fallback = True
    return selected['q'], fallback, diagnostics, inner_rows


def summarize(folds):
    summary = {}
    for arm in [k for k in folds[0] if k != 'held']:
        summary[arm] = {}
        for key in CORE:
            values = np.array([fold[arm][key] for fold in folds])
            worst = values.max() if key == 'accepted_misrank' else values.min()
            summary[arm][key] = dict(mean=float(values.mean()), worst=float(worst))
    return summary


def no_worse(summary, candidate, baseline, keys):
    return all(
        summary[candidate][key][part] >= summary[baseline][key][part] - 1e-12
        for key in keys for part in ('mean', 'worst'))


def report(results):
    folds, summary = results['folds'], results['summary']
    lines = [
        '# E266：严格内层独立重拟合的六子窗 DeepSets 门控', '',
        '**联合判定：' + ('通过。' if results['passed'] else '不通过；保留 final2。') +
        '本轮不读取 Test，不修改 E264 或 final2。**', '',
        '表中为 20 个身份外折等权均值 / 最差折；accepted_misrank 的最差为最大值，其余为最小值。'
        '两臂 AUROC/U@95/rej@90 均使用原 base gate，F1 使用原 argmax；深度分数仅决定冻结阈值下的接受/拒识。', '',
        '|视图与规则|' + '|'.join(CORE) + '|',
        '|---|' + '---:|' * len(CORE),
    ]
    for arm, values in summary.items():
        lines.append('|' + arm + '|' + '|'.join(
            f"{values[k]['mean']:.5f} / {values[k]['worst']:.5f}" for k in CORE) + '|')
    lines += ['', '候选减同折现包（百分点；accepted_misrank 越低越好）：', '',
              '|视图|' + '|'.join(CORE) + '|', '|---|' + '---:|' * len(CORE)]
    for view in ('clean', 'stress'):
        lines.append('|' + view + '|' + '|'.join(
            f"{100 * (summary[view + '_nested'][k]['mean'] - summary[view + '_package'][k]['mean']):+.4f}"
            for k in CORE) + '|')
    q_text = '，'.join(f"{q}: {n} 折" for q, n in sorted(results['q_distribution'].items()))
    lines += ['', '选中 q 分布：' + q_text + '。', '', '## 11 项指标', '',
              '|#|指标|数值与证据性质|', '|---:|---|---|']
    historical = {
        1: ('Train 样本内 acc', '0.9996461429582448（现包历史值；E266未重算）'),
        2: ('OOF acc', '0.9988499646142959（现包历史值；E266未重算）'),
        3: ('OOF macro-F1', '0.9988253655342916（现包历史值；E266未重算）'),
        4: ('forward acc', '0.9982509838216004（现包历史值；E266未重算）'),
    }
    for i in range(1, 5):
        lines.append(f'|{i}|{historical[i][0]}|{historical[i][1]}|')
    lines += [
        f"|5|strict LOSO AUROC|{summary['clean_nested']['AUROC']['mean']:.12f}（E266 Train 晚段20折均值）|",
        f"|6|U@95|{summary['clean_nested']['U95']['mean']:.12f}（E266 Train 晚段20折均值）|",
        '|7|RA|N/A|', '|8|ASafeR|N/A|',
        '|9|Test S2|0.22416484545738372（现包历史值；E266未审计）|',
        '|10|Test S1|0.5918232772832179（现包历史值；E266未审计）|',
        f"|11|strict LOSO rej@90|{summary['clean_nested']['R90']['mean']:.12f}（E266 Train 晚段20折均值）|",
        '', '## 冻结方法与防泄漏', '',
        '- 每折留出一个身份；其余19人的前60%拟合任务标准化、PCA-256、类内白化及深度门控。中段(60%,80%]只做阈值选择，所有身份末20%只评估一次；held 身份不参与任何拟合、原型、网络或阈值。',
        '- 19人按位置模5构造5个 inner enrollment folds，每次约15人注册。每个 inner fold 只用 registered 身份的早60%重新拟合任务标准化、PCA-256和类内白化，并以其 inner_art 分别投影 clean/train_aug；inner unknown 不参与 inner 变换、原型、网络拟合或阈值。inner_art 使用后即释放且不写入 outer 权重。每个 inner fold 再按身份×任务的时间奇偶分半，查询使用另一半 clean 原型；clean 与 train_aug 都训练。label 为原 top1 是否等于真实身份，inner unknown 和已知错排均为0。',
        '- 每个查询仅评估原 base 第一名：6×4子窗证据经共享4→32→32 DeepSets、mean/min/max池化和E264统计；8维 cohort 为 s1、s2、margin、gate、mean、std、s1−mean、(s1−mean)/std，经8→16→16 MLP后融合。残差 logit=12×(base_gate−0.33)+residual，末层零初始化，BCE pos_weight 按全体训练标签平衡。',
        '- 每个 inner fold 重新使用该 inner 自己的 inner_art，投影 clean，并用早段完整 inner registered 原型和中段 clean 查询校准；绝不复用 outer_art。base tau 为 known base 分 P10；候选 q 网格固定为[.05,.10,.15,.20,.25,.30]，先要求 inner known accept 与 unknown reject 均值均不低于 base，再依次最大化 known accept、unknown reject、较小 q；无合格项时最大化两项差值的较小者，再取较小 q。outer 候选 tau 为19人中段 clean candidate 分的选中 q 分位。stress 从未训练或校准。',
        '- U@95/rej@90 是晚段 base 连续分经验 ROC 上已知接受率达到95%/90%时的未知拒识，不是冻结阈值拒识率。old_e2e 要求已知身份排对且接受；accepted_misrank 是已知错排且接受的比例。逐折权重在 weights/held01.pt 至 held20.pt，q、inner 校准诊断及逐折结果在 results.json。',
        '', '## 验收条件', '',
        f"- 两臂 AUROC/U@95/rej@90 与 package 逐折完全一致：{results['roc_identical']}",
        f"- 两臂 F1/argmax 与 package 完全一致：{results['f1_identical']}",
        f"- clean/stress old_e2e、known_accept、unknown_reject 均值与最差均不降：{results['rates_ok']}",
        f"- clean/stress accepted_misrank 均值不增：{results['safety_ok']}",
        f"- 联合 passed：{results['passed']}",
        '', '复跑：`..\\.venv\\Scripts\\python.exe -B research/E266_strict_nested/run.py`。',
    ]
    text = '\n'.join(lines) + '\n'
    (H / 'REPORT.md').write_text(text, encoding='utf-8')
    print(text, flush=True)


def load_checkpoint(held):
    stem = H / 'weights' / f'held{held:02d}'
    try:
        bundle = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
        record = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
        fold, threshold = record['fold'], record['threshold']
        valid = (
            bundle['held'] == held and fold['held'] == held and threshold['held'] == held
            and bundle['recipe'] == RECIPE
            and bundle['selected_q'] == threshold['q']
            and bundle['people'] == [i for i in r.ids if i != held]
            and isinstance(bundle['projection'], dict)
            and isinstance(bundle['prototypes'], torch.Tensor)
            and isinstance(bundle['model'], dict)
            and bundle['model'].keys() == NestedVerifier().state_dict().keys()
            and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all()
                    for v in bundle['model'].values())
            and bundle['candidate_tau'] == threshold['candidate']
            and bundle['base_tau'] == threshold['package']
            and len(threshold['q_diagnostics']) == len(Q_GRID)
            and len(threshold['inner_calibration']) == 5
        )
        for view in ('clean', 'stress'):
            package, nested = fold[view + '_package'], fold[view + '_nested']
            valid = valid and all(
                package[k] == nested[k] and np.isfinite(nested[k])
                for k in ('AUROC', 'U95', 'R90', 'F1'))
            valid = valid and all(np.isfinite(nested[k]) for k in CORE)
        return (fold, threshold) if valid else None
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError, EOFError):
        return None


def main():
    if D != 'cuda':
        raise RuntimeError('E266 requires CUDA')
    cache = r.CACHE / 'e264_subwindow_verifier'
    clean = r.tens(np.load(r.CACHE / 'e253_subwindow' / 'train_feat.npy'))
    augmented = r.tens(np.load(cache / 'train_aug.npy'))
    stress = r.tens(np.load(cache / 'stress.npy'))
    if clean.shape != (11304, 6, 855) or augmented.shape != clean.shape or stress.shape != clean.shape:
        raise ValueError('Unexpected Train feature cache shape')
    weights = H / 'weights'
    weights.mkdir(exist_ok=True)
    folds, thresholds = [], []
    for held in r.ids:
        cached = load_checkpoint(held)
        if cached is not None:
            folds.append(cached[0])
            thresholds.append(cached[1])
            print('held', held, 'loaded valid checkpoint', flush=True)
            continue
        fit = (r.f <= .6) & (r.y != held)
        cal = (r.f > .6) & (r.f <= .8) & (r.y != held)
        late = r.f > .8
        people = [i for i in r.ids if i != held]
        q_mean, art = r.project(fit)
        clean_q = e264['project_subwindows'](clean, art)
        aug_q = e264['project_subwindows'](augmented, art)
        stress_q = e264['project_subwindows'](stress, art)
        if not torch.allclose(clean_q.mean(1), q_mean, atol=2e-5, rtol=2e-5):
            raise ValueError(f'Subwindow mean projection mismatch in fold {held}')
        model = train_verifier(clean, augmented, fit, people, held)
        q, fallback, q_diagnostics, inner_diagnostics = calibrate(
            clean, fit, cal, people, model)
        p, valid = e264['package_prototypes'](clean_q.mean(1), fit, people, art)
        base_cal, cand_cal, _ = infer(clean_q, p, valid, cal, people, model)
        base_tau = float(torch.quantile(base_cal, .10))
        cand_tau = float(torch.quantile(cand_cal, q))
        truth = r.y[late].cpu().numpy()
        row = {'held': held}
        for view, q_sw in [('clean', clean_q), ('stress', stress_q)]:
            base, candidate, pred = infer(q_sw, p, valid, late, people, model)
            package = metrics(base, base, pred, truth, held, base_tau)
            nested = metrics(base, candidate, pred, truth, held, cand_tau)
            if any(package[k] != nested[k] for k in ('AUROC', 'U95', 'R90', 'F1')):
                raise AssertionError('Candidate changed frozen base ROC or F1')
            row[view + '_package'] = package
            row[view + '_nested'] = nested
        bundle = dict(
            held=held, people=people, projection=art, prototypes=p, valid=valid,
            model=model.state_dict(), base_tau=base_tau, candidate_tau=cand_tau,
            selected_q=q, fallback=fallback, recipe=RECIPE,
            tasks=sorted(r.M.task.unique().tolist()),
        )
        torch.save(bundle, weights / f'held{held:02d}.pt')
        thresholds.append(dict(
            held=held, q=q, fallback=fallback, package=base_tau, candidate=cand_tau,
            q_diagnostics=q_diagnostics, inner_calibration=inner_diagnostics))
        folds.append(row)
        checkpoint = weights / f'held{held:02d}.json'
        checkpoint.write_text(json.dumps(dict(
            fold=row, threshold=thresholds[-1]), indent=2, ensure_ascii=False,
            allow_nan=False), encoding='utf-8')
        print('held', held, 'q', q, 'clean known/unknown',
              row['clean_nested']['known_accept'], row['clean_nested']['unknown_reject'], flush=True)
        del clean_q, aug_q, stress_q, model
        torch.cuda.empty_cache()
    if len(folds) != 20 or len(thresholds) != 20:
        raise RuntimeError('E266 requires exactly 20 complete folds')
    summary = summarize(folds)
    roc_identical = all(
        row[view + '_nested'][key] == row[view + '_package'][key]
        for row in folds for view in ('clean', 'stress') for key in ('AUROC', 'U95', 'R90'))
    f1_identical = all(
        row[view + '_nested']['F1'] == row[view + '_package']['F1']
        for row in folds for view in ('clean', 'stress'))
    rates_ok = all(no_worse(
        summary, view + '_nested', view + '_package',
        ['old_e2e', 'known_accept', 'unknown_reject']) for view in ('clean', 'stress'))
    safety_ok = all(
        summary[view + '_nested']['accepted_misrank']['mean']
        <= summary[view + '_package']['accepted_misrank']['mean'] + 1e-12
        for view in ('clean', 'stress'))
    q_distribution = dict(sorted(Counter(f"{row['q']:.2f}" for row in thresholds).items()))
    results = dict(
        recipe=RECIPE, gpu=torch.cuda.get_device_name(), folds=folds, thresholds=thresholds,
        summary=summary, q_distribution=q_distribution, roc_identical=roc_identical,
        f1_identical=f1_identical, ranking_identical=True, rates_ok=rates_ok,
        safety_ok=safety_ok,
        passed=roc_identical and f1_identical and rates_ok and safety_ok,
        test_accessed=False,
    )
    save_json('results.json', results)
    report(results)
    concise = dict(
        passed=results['passed'], q_distribution=q_distribution,
        rates_ok=rates_ok, safety_ok=safety_ok,
        clean={arm: {k: summary['clean_' + arm][k] for k in ('old_e2e', 'known_accept', 'unknown_reject', 'accepted_misrank')}
               for arm in ('package', 'nested')},
        stress={arm: {k: summary['stress_' + arm][k] for k in ('old_e2e', 'known_accept', 'unknown_reject', 'accepted_misrank')}
                for arm in ('package', 'nested')},
    )
    print('E266_FINAL ' + json.dumps(concise, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
