"""Train-only six-subwindow DeepSets verifier; identity ranking stays frozen."""
import json
import runpy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.functional import normalize

H = Path(__file__).resolve().parent
PREV = H.parent / 'E262_wave_pair'
e262 = runpy.run_path(str(PREV / 'run.py'))
r = e262['ref']
D = r.D
CORE = ['AUROC', 'U95', 'R90', 'F1', 'old_e2e', 'known_accept', 'unknown_reject', 'accepted_misrank']
RECIPE = dict(
    seed=26400, epochs=24, batch_size=512, lr=.002, weight_decay=1e-4,
    encoder='4-32-32 shared DeepSets', pooling='mean+min+max',
    global_features='window-max mean/std/min/max + per-prototype mean/std',
    head='108-48-1 residual', logit='12*(candidate_gate-0.33)+residual',
    train_views=['clean', 'train_aug'], calibration='clean mid-segment P10 only',
    evaluation='late clean and independent stress; no Test access',
)


def save_json(name, obj):
    (H / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def project_subwindows(x, art):
    z = (x - art['mu'][r.t][:, None]) / art['sd'][r.t][:, None]
    return ((z - art['center']) @ art['basis']) @ art['white']


def package_prototypes(q_mean, rows, people, art):
    p, valid = r.templates(q_mean, rows, people)
    return normalize(p @ art['white'], dim=-1), valid


def identity_scores(q_mean, p, valid, rows):
    """Current final2 rule: queries are whitened once, stored prototypes twice."""
    ix = torch.where(rows)[0]
    out = q_mean.new_full((len(ix), p.shape[1]), -torch.inf)
    tasks = r.t[ix]
    query = normalize(q_mean[ix], dim=-1)
    for task in range(4):
        take = tasks == task
        if not take.any():
            continue
        sim = (query[take] @ p[task].flatten(0, 1).T).view(-1, p.shape[1], 4)
        out[take] = sim.masked_fill(~valid[task][None], -torch.inf).amax(-1)
    return out


def candidate_gate(scores, claimed):
    own = scores.gather(1, claimed[:, None]).squeeze(1)
    other = scores.clone()
    other.scatter_(1, claimed[:, None], -torch.inf)
    runner_up = other.amax(1)
    return own + .5 * (own - runner_up)


def evidence(q_sw, tasks, p, claimed):
    """Six non-overlapping windows by four claimed-identity prototype matches."""
    out = q_sw.new_empty((len(q_sw), 6, 4))
    for task in range(4):
        take = tasks == task
        if not take.any():
            continue
        query = normalize(q_sw[take], dim=-1)
        proto = p[task][claimed[take]]
        out[take] = torch.einsum('nwd,npd->nwp', query, proto)
    return out


class SubwindowVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(108, 48), nn.SiLU(), nn.Linear(48, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h = self.encoder(x)
        pooled = torch.cat([h.mean(1), h.amin(1), h.amax(1)], dim=1)
        window_max = x.amax(-1)
        stats = torch.stack([
            window_max.mean(1), window_max.std(1, unbiased=False),
            window_max.amin(1), window_max.amax(1),
        ], dim=1)
        proto_stats = torch.cat([x.mean(1), x.std(1, unbiased=False)], dim=1)
        return self.head(torch.cat([pooled, stats, proto_stats], dim=1)).squeeze(1)


def verifier_scores(gate, proof, model):
    return 12 * (gate - .33) + model(proof)


def make_pairs(q_mean, q_sw, rows, true_people, p, valid):
    """Genuine and hardest-impostor claims; neither query is in its templates."""
    s = identity_scores(q_mean, p, valid, rows)
    truth = torch.searchsorted(true_people, r.y[rows])
    masked = s.clone()
    masked.scatter_(1, truth[:, None], -torch.inf)
    impostor = masked.argmax(1)
    n = len(truth)
    sel = torch.arange(n, device=D).repeat(2)
    claimed = torch.cat([truth, impostor])
    labels = torch.cat([torch.ones(n, device=D), torch.zeros(n, device=D)])
    proof = evidence(q_sw[rows][sel], r.t[rows][sel], p, claimed)
    gate = candidate_gate(s[sel], claimed)
    return proof, gate, labels


def train_verifier(clean, augmented, fit, people, art, held):
    parity = torch.full_like(r.y, -1)
    for person in people:
        for task in range(4):
            ix = torch.where(fit & (r.y == person) & (r.t == task))[0]
            ix = ix[r.order[ix].argsort()]
            parity[ix] = torch.arange(len(ix), device=D) % 2
    proofs, gates, labels = [], [], []
    true_people = r.tens(people)
    for half in (0, 1):
        template_rows = fit & (parity != half)
        query_rows = fit & (parity == half)
        p, valid = package_prototypes(clean.mean(1), template_rows, people, art)
        for q_sw in (clean, augmented):
            a, b, c = make_pairs(q_sw.mean(1), q_sw, query_rows, true_people, p, valid)
            proofs.append(a)
            gates.append(b)
            labels.append(c)
    proofs, gates, labels = torch.cat(proofs), torch.cat(gates), torch.cat(labels)
    torch.manual_seed(RECIPE['seed'] + held)
    model = SubwindowVerifier().to(D)
    opt = torch.optim.AdamW(
        model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    for _ in range(RECIPE['epochs']):
        order = torch.randperm(len(labels), device=D)
        for ix in order.split(RECIPE['batch_size']):
            logits = verifier_scores(gates[ix], proofs[ix], model)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.inference_mode()
def infer(q_sw, p, valid, rows, people, model=None):
    q_mean = q_sw.mean(1)
    s = identity_scores(q_mean, p, valid, rows)
    claimed = s.argmax(1)
    base = candidate_gate(s, claimed)
    pred = r.tens(people)[claimed]
    if model is None:
        return base, pred
    proof = evidence(q_sw[rows], r.t[rows], p, claimed)
    return verifier_scores(base, proof, model), pred


def metrics(raw, pred, truth, held, tau):
    margin = raw.cpu().numpy() - tau
    pred = pred.cpu().numpy()
    row = e262['metrics'](margin, pred, truth, held, 0.)
    known = truth != held
    accept = margin >= 0
    row['AUROC'] = row.pop('AUC')
    row['known_accept'] = float(accept[known].mean())
    row['accepted_misrank'] = float((accept & (pred != truth))[known].mean())
    return {k: row[k] for k in CORE}


def summarize(folds):
    arms = [k for k in folds[0] if k != 'held']
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for key in CORE:
            values = np.array([fold[arm][key] for fold in folds])
            worst = values.max() if key == 'accepted_misrank' else values.min()
            summary[arm][key] = dict(mean=float(values.mean()), worst=float(worst))
    return summary


def no_worse(summary, candidate, baseline, keys, mean_and_worst=True):
    parts = ('mean', 'worst') if mean_and_worst else ('mean',)
    return all(
        summary[candidate][key][part] >= summary[baseline][key][part] - 1e-12
        for key in keys for part in parts
    )


def report(results):
    folds, summary = results['folds'], results['summary']
    lines = [
        '# E264：六子窗 DeepSets 第一名验证器', '',
        '**联合判定：' + ('通过，可冻结全 Train 模型后做一次历史 Test 审计' if results['passed']
                       else '不通过；保留 final2，不读取 Test') + '。**', '',
        '当前为 Train 严格 20 身份外折，不是初赛 S1/S2 实测，不能据此声称第 9/10 项已达到 0.5。表中为 20 折等权均值 / 最差折；AUC 指 AUROC，F1 指已知身份闭集 macro-F1。', '',
        '|视图与规则|' + '|'.join(CORE) + '|',
        '|---|' + '---:|' * len(CORE),
    ]
    for arm, values in summary.items():
        lines.append('|' + arm + '|' + '|'.join(
            f"{values[k]['mean']:.5f} / {values[k]['worst']:.5f}" for k in CORE) + '|')
    lines += [
        '', '## 11 项指标总表', '',
        '#1–4 是现包历史全 Train/OOF 审计，E264 未训练全 Train 候选，故不重算、不冒充候选结果；#9/10 同样仅列现包历史实测。', '',
        '|序号|指标|现包历史或同折对照|E264|口径|',
        '|---:|---|---:|---:|---|',
        '|1|Train 样本内闭集 acc|0.999646|未重算|现包历史全 Train|',
        '|2|5 折 OOF 闭集 acc|0.998850|未重算|现包历史|',
        '|3|5 折 OOF macro-F1|0.998825|未重算|现包历史|',
        '|4|前向块外 acc|0.998251|未重算|现包历史|',
        f"|5|严格 LOSO AUROC|{summary['clean_package']['AUROC']['mean']:.6f}|{summary['clean_verifier']['AUROC']['mean']:.6f}|20 折均值|",
        f"|6|严格 LOSO U@95|{summary['clean_package']['U95']['mean']:.6f}|{summary['clean_verifier']['U95']['mean']:.6f}|20 折均值|",
        '|7|RA|N/A|N/A|本轮无对抗样本|',
        '|8|ASafeR|N/A|N/A|本轮无对抗样本|',
        '|9|Test S2 已注册接受率|0.224165|未审计|现包历史实测，非 E264 结果|',
        '|10|Test S1 未注册拒识率|0.591823|未审计|现包历史实测，非 E264 结果|',
        f"|11|严格 LOSO 未知拒识@90|{summary['clean_package']['R90']['mean']:.6f}|{summary['clean_verifier']['R90']['mean']:.6f}|20 折均值|",
        '', '候选减同折现包（百分点；accepted_misrank 越低越好）：', '',
        '|视图|' + '|'.join(CORE) + '|', '|---|' + '---:|' * len(CORE)]
    for view in ('clean', 'stress'):
        lines.append('|' + view + '|' + '|'.join(
            f"{100 * (summary[view + '_verifier'][k]['mean'] - summary[view + '_package'][k]['mean']):+.4f}"
            for k in CORE) + '|')
    lines += [
        '', '## 冻结方法与防泄漏', '',
        '- 仅使用 Train。逐折留一个身份作未知，其余 19 人按身份×任务的 trial/max(trial) 前 60% 拟合任务标准化、PCA-256、类内白化、注册原型与验证器；中段 (60%,80%] 仅以 clean 连续分的 P10 标定阈值；所有身份末 20% 只评估一次。',
        '- 身份排名始终由现包余弦规则产生，验证器只给原第一名一个接受分数，因此闭集 top1 与 F1 必须完全一致。原型按现包规则再次白化，查询仅白化一次。',
        '- 训练查询按身份×任务奇偶分半，查询只用另一半干净样本构造的四原型，查询及其增强副本均不进入自身模板。每个查询提供本人候选正例和最难非本人候选负例；clean 与 train_aug 均参与训练。stress 是独立固定扰动，未参与训练或校准。',
        '- 共享 4→32→32 编码器处理每个不重叠子窗与候选身份四原型的余弦，跨六窗做 mean/min/max 池化，并拼接窗口最大值及逐原型均值/标准差；108→48→1 头输出残差。最终连续分为 12×(候选 gate−0.33)+残差。',
        '- U@95/rej@90 是晚段经验 ROC 上已知接受率至少达到 95%/90% 时的未知拒识，不是冻结阈值拒识率。old_e2e 要求已知身份排对且接受；accepted_misrank 是已知错排且接受的比例。',
        '- 同一试次的不同视图与不同折不独立；模拟 stress 不等于真实跨设备或 Test。权重在 weights/held01.pt 至 held20.pt，逐折阈值与结果在 results.json。',
        '', '## 验收条件', '',
        f"- clean AUROC/U@95/rej@90 均值与最差均不降：{results['clean_roc_ok']}",
        f"- clean/stress old_e2e、known_accept、unknown_reject 均值与最差均不降：{results['rates_ok']}",
        f"- 两视图 accepted_misrank 均值不增：{results['safety_ok']}",
        f"- 闭集 F1 逐折完全一致：{results['f1_identical']}",
        '', '复跑：先生成六子窗扰动缓存，再运行训练评估：',
        '', '```powershell',
        '.\\.venv\\Scripts\\python.exe -B research/E264_subwindow_verifier/build_views.py',
        '.\\.venv\\Scripts\\python.exe -B research/E264_subwindow_verifier/run.py',
        '```',
    ]
    text = '\n'.join(lines) + '\n'
    (H / 'REPORT.md').write_text(text, encoding='utf-8')
    print(text)


def main():
    if D != 'cuda':
        raise RuntimeError('E264 requires CUDA')
    cache = r.CACHE / 'e264_subwindow_verifier'
    clean = r.tens(np.load(r.CACHE / 'e253_subwindow' / 'train_feat.npy'))
    augmented = r.tens(np.load(cache / 'train_aug.npy'))
    stress = r.tens(np.load(cache / 'stress.npy'))
    weights = H / 'weights'
    weights.mkdir(exist_ok=True)
    folds, thresholds = [], []
    for held in r.ids:
        fit = (r.f <= .6) & (r.y != held)
        cal = (r.f > .6) & (r.f <= .8) & (r.y != held)
        late = r.f > .8
        people = [i for i in r.ids if i != held]
        q_mean, art = r.project(fit)
        clean_q = project_subwindows(clean, art)
        aug_q = project_subwindows(augmented, art)
        stress_q = project_subwindows(stress, art)
        if not torch.allclose(clean_q.mean(1), q_mean, atol=2e-5, rtol=2e-5):
            raise ValueError(f'Subwindow mean projection mismatch in fold {held}')
        model = train_verifier(clean_q, aug_q, fit, people, art, held)
        p, valid = package_prototypes(clean_q.mean(1), fit, people, art)
        base_cal, _ = infer(clean_q, p, valid, cal, people)
        cand_cal, _ = infer(clean_q, p, valid, cal, people, model)
        base_tau = float(torch.quantile(base_cal, .1))
        cand_tau = float(torch.quantile(cand_cal, .1))
        truth = r.y[late].cpu().numpy()
        row = {'held': held}
        for view, q_sw in [('clean', clean_q), ('stress', stress_q)]:
            base, base_pred = infer(q_sw, p, valid, late, people)
            candidate, cand_pred = infer(q_sw, p, valid, late, people, model)
            if not torch.equal(base_pred, cand_pred):
                raise AssertionError('Verifier changed identity ranking')
            row[view + '_package'] = metrics(base, base_pred, truth, held, base_tau)
            row[view + '_verifier'] = metrics(candidate, cand_pred, truth, held, cand_tau)
        bundle = dict(
            held=held, people=people, projection=art, prototypes=p, valid=valid,
            model=model.state_dict(), base_tau=base_tau, candidate_tau=cand_tau,
            recipe=RECIPE, tasks=sorted(r.M.task.unique().tolist()),
        )
        torch.save(bundle, weights / f'held{held:02d}.pt')
        thresholds.append(dict(held=held, package=base_tau, verifier=cand_tau))
        folds.append(row)
        print('held', held, 'clean AUC', row['clean_verifier']['AUROC'], flush=True)
        del clean_q, aug_q, stress_q, model
        torch.cuda.empty_cache()
    summary = summarize(folds)
    f1_identical = all(
        row[view + '_package']['F1'] == row[view + '_verifier']['F1']
        for row in folds for view in ('clean', 'stress'))
    clean_roc_ok = no_worse(summary, 'clean_verifier', 'clean_package', ['AUROC', 'U95', 'R90'])
    rates_ok = all(no_worse(
        summary, view + '_verifier', view + '_package',
        ['old_e2e', 'known_accept', 'unknown_reject']) for view in ('clean', 'stress'))
    safety_ok = all(
        summary[view + '_verifier']['accepted_misrank']['mean']
        <= summary[view + '_package']['accepted_misrank']['mean'] + 1e-12
        for view in ('clean', 'stress'))
    results = dict(
        recipe=RECIPE, gpu=torch.cuda.get_device_name(), folds=folds, thresholds=thresholds,
        summary=summary, clean_roc_ok=clean_roc_ok, rates_ok=rates_ok,
        safety_ok=safety_ok, f1_identical=f1_identical,
        passed=clean_roc_ok and rates_ok and safety_ok and f1_identical,
        test_accessed=False,
    )
    save_json('results.json', results)
    report(results)


if __name__ == '__main__':
    main()
