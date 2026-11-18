"""复用 E262 的 LOSO 权重；唯一干预是门控阈值。"""
import json
import runpy
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

H = Path(__file__).resolve().parent
PREV = H.parent / 'E262_wave_pair'
e262 = runpy.run_path(str(PREV / 'run.py'))
r = e262['ref']
METRICS = ['AUROC', 'U95', 'R90', 'F1', 'old_e2e', 'known_accept', 'unknown_reject']


def save(name, obj):
    (H / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


@torch.inference_mode()
def main():
    cache = r.CACHE / 'e262_wave_pair'
    aug, stress = [r.tens(np.load(cache / (s + '.npy'))) for s in ['train_aug', 'stress']]
    late = r.f > .8
    truth = r.y[late].cpu().numpy()
    late_task = r.t[late].cpu().numpy()
    folds, thresholds = [], []
    for h in r.ids:
        b = torch.load(PREV / 'weights' / f'held{h:02}.pt', map_location=r.D, weights_only=True)
        art, old, people = b['projection'], b['E254'], b['people']
        assert people == [i for i in r.ids if i != h] and b['tasks'] == sorted(r.M.task.unique())
        p = normalize(old['p'] @ art['white'], dim=-1)
        def score(x, mask):
            q = (((x - art['mu'][r.t]) / art['sd'][r.t] - art['center']) @ art['basis']) @ art['white']
            s, pred = r.score(q, p, old['v'], mask, people)
            return s[0].astype(np.float64), pred
        cal = (r.f > .6) & (r.f <= .8) & (r.y != h)
        ct = r.t[cal].cpu().numpy()
        c, a = score(r.X, cal)[0], score(aug, cal)[0]
        global_tau = float(np.quantile(c, .1))
        tau = np.array([np.quantile(np.r_[c[ct == i], a[ct == i]], .1) for i in range(4)])
        thresholds.append(dict(held=h, global_clean_P10=global_tau, task_mix_P10=dict(zip(b['tasks'], tau)), n_per_view=dict(zip(b['tasks'], np.bincount(ct).tolist()))))
        row = {'held': h}
        for view, x in [('clean', r.X), ('stress', stress)]:
            raw, pred = score(x, late)
            for arm, margin in [('package', raw - global_tau), ('task', raw - tau[late_task])]:
                m = e262['metrics'](margin, pred, truth, h, 0.)
                m['AUROC'] = m['AUC']; m['known_accept'] = float((margin[truth != h] >= 0).mean())
                row[f'{view}_{arm}'] = {k: m[k] for k in METRICS}
        folds.append(row)
    summary = {arm: {k: {'mean': float(np.mean(v)), 'worst': min(v)} for k in METRICS for v in [[f[arm][k] for f in folds]]} for arm in folds[0] if arm != 'held'}
    def delta(view, k):
        return summary[view+'_task'][k]['mean'] - summary[view+'_package'][k]['mean']
    roc_ok = all(delta('clean', k) >= 0 for k in METRICS[:3])
    rates_ok = all(delta('stress', k) > 0 for k in ['old_e2e','unknown_reject'])
    save('thresholds.json', thresholds)
    save('results.json', dict(folds=folds, summary=summary, clean_roc_ok=roc_ok, stress_rates_ok=rates_ok, passed=roc_ok and rates_ok))
    lines = ['# E263：仅改逐任务拒识阈值', '', '联合判定：' + ('通过' if roc_ok and rates_ok else '不通过') + '。三项以 clean LOSO 均值不下降判断，两率以 stress 老用户端到端和未知拒识均值严格提升判断。', '', '表中为 20 折等权均值 / 最差折（最小值）。', '', '|视图与规则|'+'|'.join(METRICS)+'|', '|---|'+'---:|'*7]
    for arm, vals in summary.items():
        lines.append('|'+arm+'|'+'|'.join(f"{v['mean']:.5f} / {v['worst']:.5f}" for v in vals.values())+'|')
    lines += ['', '候选减现包（百分点）：', '', '|视图|'+'|'.join(METRICS)+'|', '|---|'+'---:|'*7]
    for view in ['clean','stress']:
        lines.append('|'+view+'|'+'|'.join(f'{100*delta(view,k):+.4f}' for k in METRICS)+'|')
    lines += ['', '## 方法与解释', '', '- 数据限 Train（20 身份、4 任务）。按身份×任务的 trial/max(trial) 切段。逐折留 1 身份为未知；复用 E262 仅用其余 19 人前 60% 拟合的投影与 E254 原型。按 package_metric 对原型再白化并归一化，查询只白化一次；同一视图两臂共用一次身份打分和预测。', '- 同折 19 人中段 (60%,80%]：对照取 clean 全局 P10；候选逐任务等量拼接 clean 和 train_aug 的原 gate 后取 P10，不是先平均两个视图。连续分均为原 gate 减相应阈值，≥0 接受。没有拟合新模型或筛选阈值。', '- 所有身份末 20% 分别作 clean 和独立扰动 stress 查询；stress 不参与标定。正式 U95/R90 从晚段已知连续分降序取首次达到 95%/90% 接受率的分数，未知低于它即拒识；不是冻结阈值下的拒识率。F1 是已知闭集 macro-F1，old_e2e 还要求已知身份排对且接受。', '- 同一原始试次的不同视图及不同折并非独立样本；结果不是 Test 或新设备实测。不读取 Test，不改 final2 或 git。', '- 复跑：在仓库根目录运行 `.\\.venv\\Scripts\\python.exe -B research/E263_task_gate/evaluate.py`。逐折门限见 thresholds.json，逐折结果见 results.json。']
    report = '\n'.join(lines)+'\n'
    (H/'REPORT.md').write_text(report,encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
