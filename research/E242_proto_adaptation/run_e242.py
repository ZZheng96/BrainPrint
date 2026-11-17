"""Train-only, chronological holdout prototype adaptation using immutable E241 models."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'research/E241_robust_prototype_aug'))
import run_e241 as P
E = P.E
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
SOURCE = ROOT / 'research/E241_robust_prototype_aug/models'
NAMES = CFG['candidates']


def load(tag, fit, query):
    path = SOURCE / f'{tag}_{CFG["source_fingerprint"]}.npz'
    with np.load(path, allow_pickle=True) as z:
        assert np.array_equal(z['fit_ids'], np.flatnonzero(fit)), tag
        assert np.array_equal(z['query_ids'], np.flatnonzero(query)), tag
        return {k: z[k] for k in ['state', 'baseline_prototypes', 'baseline_scores', 'prototype_tasks', 'prototype_subjects']}


def transform(features, tasks, state):
    st = state[0]['e65']
    z = torch.as_tensor(features, device=E.DEVICE, dtype=torch.float64).clone()
    for task in E.TASKS:
        rows = torch.as_tensor(tasks == task, device=E.DEVICE)
        mu, sd = st['standardization'][task]
        # Match E215 float32 task standardization before float64 projection.
        z[rows] = ((z[rows].float() - torch.as_tensor(mu, device=E.DEVICE)) / torch.as_tensor(sd, device=E.DEVICE)).double()
    mu, basis = st['pca']
    return ((z - torch.as_tensor(mu, device=E.DEVICE)) @ torch.as_tensor(basis, device=E.DEVICE)).float().cpu().numpy()


def adapt(x, tasks, ps, keys, gate, rule):
    updated = ps.copy()
    kt = np.array([k[0] for k in keys])
    selected = 0
    for task in E.TASKS:
        rows = np.flatnonzero(tasks == task)
        cols = np.flatnonzero(kt == task)
        if not len(rows):
            continue
        d = torch.cdist(torch.as_tensor(x[rows], device=E.DEVICE), torch.as_tensor(ps[cols], device=E.DEVICE))
        distances, nearest = d.min(1)
        keep = (-distances >= gate).cpu().numpy()
        assignment = cols[nearest.cpu().numpy()]
        selected += int(keep.sum())
        for col in np.unique(assignment[keep]):
            values = x[rows[keep & (assignment == col)]]
            if rule == 'query_ema':
                for value in values:
                    updated[col] = (1 - CFG['ema_alpha']) * updated[col] + CFG['ema_alpha'] * value
            else:
                # One original center pseudo-observation; no reclustering/encoder updates.
                updated[col] = (ps[col] + values.sum(0)) / (1 + len(values))
    return updated, selected


def run_window(tag, model, history_ids, query_ids, features, meta, gate):
    assert not np.intersect1d(history_ids, query_ids).size
    # Trial order is only consumed on Train; no Test subject/label access exists.
    history_ids = history_ids[np.argsort(meta.trial.to_numpy()[history_ids], kind='stable')]
    ids = np.r_[history_ids, query_ids]
    tasks = meta.task.to_numpy()[ids]
    x = transform(features[ids], tasks, model['state'])
    nh = len(history_ids)
    ps = model['baseline_prototypes']
    keys = list(zip(model['prototype_tasks'].tolist(), model['prototype_subjects'].tolist()))
    result, saved = {}, dict(state=model['state'], fit_model=str(SOURCE / f'{tag}_{CFG["source_fingerprint"]}.npz'), history_ids=history_ids, query_ids=query_ids, confidence_threshold=gate, prototype_tasks=model['prototype_tasks'], prototype_subjects=model['prototype_subjects'])
    for name in NAMES:
        updated, count = (ps.copy(), 0) if name == 'baseline' else adapt(x[:nh], tasks[:nh], ps, keys, gate, name)
        result[name] = E.scores(x[nh:], tasks[nh:], updated, keys)
        saved[name + '_prototypes'] = updated
        saved[name + '_scores'] = result[name]
        saved[name + '_selected'] = count
    np.savez_compressed(OUT / 'models' / f'{tag}.npz', **saved)
    np.savez_compressed(OUT / 'cache' / f'{tag}.npz', row_ids=ids, representation=x)
    return result


def inner(tag, outer_fit, features, meta, fr, blocks):
    subjects = meta.subject.to_numpy()
    groups = np.array_split(np.unique(subjects[outer_fit]), CFG['inner_subject_groups'])
    windows = []
    for g, held in enumerate(groups):
        for b in np.unique(blocks[outer_fit]):
            fit = outer_fit & ~np.isin(subjects, held) & (blocks != b)
            query = outer_fit & (blocks == b)
            model = load(f'{tag}_inner_g{g}_b{b}', fit, query)
            ids = np.flatnonzero(query)
            midpoint = (b + .5) / E.CFG['blocks']
            h = fr[ids] <= midpoint
            # Boundary-only block b3 (frac=0.6) has no second half: exclude explicitly.
            if not h.any() or h.all():
                continue
            unknown = np.isin(subjects[ids], held)
            windows.append(dict(g=g, b=int(b), model=model, history=ids[h], query=ids[~h], unknown=unknown[~h], gate_ref=model['baseline_scores'][h & ~unknown]))
    refs = {n: [[], []] for n in NAMES}
    gates = []
    for w in windows:
        pool = np.concatenate([v['gate_ref'] for v in windows if v['g'] != w['g'] and v['b'] != w['b']])
        gate = float(np.quantile(pool, CFG['confidence_known_quantile']))
        gates.append(gate)
        scored = run_window(f'{tag}_inner_g{w["g"]}_b{w["b"]}', w['model'], w['history'], w['query'], features, meta, gate)
        for n in NAMES:
            refs[n][0].extend(scored[n][~w['unknown']].tolist())
            refs[n][1].extend(scored[n][w['unknown']].tolist())
    refs = {n: tuple(np.array(a) for a in values) for n, values in refs.items()}
    gate = float(np.quantile(np.concatenate([w['gate_ref'] for w in windows]), CFG['confidence_known_quantile']))
    thresholds = {n: P.calibrated(*refs[n])[0] for n in NAMES}
    np.savez_compressed(OUT / 'cache' / f'{tag}_calibration.npz', **{n + suffix: refs[n][j] for n in NAMES for j, suffix in enumerate(['_known', '_unknown'])}, crossfit_gates=gates, confidence_threshold=gate)
    return thresholds, gate, refs


def main():
    started = time.time()
    torch.set_num_threads(4)
    for name in ['models', 'cache']:
        (OUT / name).mkdir(exist_ok=True)
    meta = pd.read_csv(ROOT / E.CFG['data']['train_metadata'])
    features = np.load(ROOT / E.CFG['data']['e65_train']).astype(np.float32)
    fr = E.frac(meta)
    blocks = np.floor(fr * E.CFG['blocks']).astype(int)
    subjects = meta.subject.to_numpy()
    eligible = fr <= E.FIT_FRACTION
    history = (fr > .6) & (fr <= .7)
    query = (fr > .7) & (fr <= .8)
    results = {n: {'folds': []} for n in NAMES}
    freeze = {}
    for held in E.CLASSES:
        tag = f'outer_{held:02d}'
        fit = eligible & (subjects != held)
        thresholds, gate, refs = inner(tag, fit, features, meta, fr, blocks)
        freeze[tag] = dict(thresholds=thresholds, confidence_threshold=gate)
        model = load(tag, fit, history | query)
        scored = run_window(tag, model, np.flatnonzero(history), np.flatnonzero(query), features, meta, gate)
        unknown = subjects[query] == held
        for n in NAMES:
            row = P.metrics(scored[n][~unknown], scored[n][unknown], thresholds[n])
            row.update(subject=int(held), inner_simulation=P.metrics(*refs[n], thresholds[n]))
            results[n]['folds'].append(row)
        print(f'{tag} complete', flush=True)
    thresholds, gate, refs = inner('final', eligible, features, meta, fr, blocks)
    freeze['final'] = dict(thresholds=thresholds, confidence_threshold=gate)
    (OUT / 'frozen_thresholds.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    # Final Train-only model is reconstructed without opening any Test file.
    E.project = P.project_gpu
    x, _, state = E.make_representation(P.SPEC, {'e65': (features, features[:0])}, meta, meta.iloc[:0], eligible)
    ps, keys = E.prototypes(x, subjects, meta.task.to_numpy(), eligible, E.CLASSES, P.SPEC['k'])
    np.savez_compressed(OUT / 'models/final_train_only.npz', state=np.array([state], object), prototypes=ps, fit_ids=np.flatnonzero(eligible), prototype_tasks=[k[0] for k in keys], prototype_subjects=[k[1] for k in keys], confidence_threshold=gate, **{n + '_threshold': thresholds[n] for n in NAMES})
    for n, r in results.items():
        rows = r['folds']
        for metric in ['AUC', 'unknown_positive_F1']:
            r[metric] = float(np.mean([v[metric] for v in rows]))
        for metric, weight in [('unknown_rejection', 'n_unknown'), ('known_acceptance', 'n_known')]:
            r[metric] = float(np.average([v[metric] for v in rows], weights=[v[weight] for v in rows]))
        r['final_train_simulated_adaptation'] = P.metrics(*refs[n], thresholds[n])
    payload = dict(fold_count=20, device=str(E.DEVICE), runtime_seconds=time.time()-started, test_read=False, official_Test_AUC=None, official_Test_F1=None, results=results, status='completed_train_only_minimal', blockers=['Test chronology/stream grouping without identity labels not established; Test and S2 intentionally not evaluated.', 'Train-simulated adaptation implemented as inner OOF chronological adaptation, not an additional meta-trained encoder candidate.'])
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    lines = ['# E242 Train-only 高置信原型自适应', '', f'完整20折，设备 {E.DEVICE}，耗时 {payload["runtime_seconds"]:.1f}s。Test任何文件/标签均未读取。', '', '## 严格协议与范围', '- 阅读赛题说明：20注册+5未知，Test含对抗样本；会话不是身份/未知标签。', '- 只读复用E241 af0d5c05ca9e的fit-only标准化、PCA96/k12原型及共同E65特征缓存，不使用增强视图。断言fit/query行ID与E241完全一致。', '- 外层fit<=0.6；当前窗口前半(0.6,0.7]只适应，后半(0.7,0.8]只评估。baseline也只评估同一后半，不能直接对比E241完整窗口指标。', '- 内层两组pseudo-unknown及时间块沿用E241，所有pseudo-unknown从表征/原型拟合排除。每个块前半适应后半查询；只有边界无后半的块3排除。base表征沿用leave-block-out，可使用其他较晚fit块；适应严格先于查询，不宣称全流程前向链训练。', '- train-simulated adaptation实际执行在Train inner OOF时间块中。此最小版本不另训练自适应编码器，不把同一机制伪称独立第三模型。', '- 高置信门槛=known baseline inner OOF history分数90%分位数；每个inner窗口用另身份组且另时间块计算门槛；outer/final用汇总inner分数冻结。拒识阈值由各候选adapted inner unknown后半分数90%分位数冻结。', '- EMA alpha=0.1，按trial稳定顺序更新；center_mean将当前中心作一个伪观测与高置信样本均值融合。两者仅更新最近身份-task簇中心，不改表征、不重新聚类，分配与筛选均用冻结初始中心。EMA不滚动重筛。', '- 所有适应函数只接收特征、task、冻结原型与门槛，不接收query身份；Train身份仅用于规定LOSO/时间分块和离线指标。', '- inner目标0.90不保证outer拒识0.90，两组身份库大小差异存在校准偏移；未按outer结果回调参数。', '', '|候选|LOSO AUC|unknown-positive F1|未知拒识micro|已知接受micro|', '|---|---:|---:|---:|---:|']
    for n, r in results.items():
        lines.append(f'|{n}|{r["AUC"]:.6f}|{r["unknown_positive_F1"]:.6f}|{r["unknown_rejection"]:.6f}|{r["known_acceptance"]:.6f}|')
    lines += ['', 'AUC与unknown-positive F1均为逐折均值；正类=外层held身份，AUC使用负置信分数。', '', '## 最终Train内层模拟（校准集，不是独立外层表现）', '|候选|AUC|unknown-positive F1|', '|---|---:|---:|']
    for n, r in results.items():
        v = r['final_train_simulated_adaptation']
        lines.append(f'|{n}|{v["AUC"]:.6f}|{v["unknown_positive_F1"]:.6f}|')
    lines += ['', '## 结论与阻塞', '本次为最小可验证Train-only版本，不宣称满足LOSO拒识约0.90约束或提升S2。Test严格时间顺序及无身份标签的流分组尚未确认，因此宁可不读取Test，也不使用Test身份字段构造前后半窗口。Test AUC/F1=N/A，S1/S2=N/A；S2未来也仅可描述性。未计算独立train-simulated元训练候选。', '', '## 产物与复现', '`models`保存每内外折冻结/自适应原型、行ID、分数、表征状态和最终Train模型；`cache`保存投影特征与OOF校准分数；`frozen_thresholds.json`保存全部门槛。源E241文件保持只读。', '运行：`.venv-py311\\Scripts\\python.exe research\\E242_proto_adaptation\\run_e242.py`。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({n: {k: v for k, v in r.items() if k != 'folds'} for n, r in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
