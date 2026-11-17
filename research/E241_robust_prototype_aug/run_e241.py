"""Train-only waveform prototype augmentation; no Test-driven selection."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'research/E65_correlation_spd'))
sys.path.insert(0, str(ROOT / 'research/E215_strict_loso_e170_rebuild'))
import run_e215 as E
import run_e65 as F
DEVICE = E.DEVICE
SPEC = E.CFG['configs'][CFG['representation']]
NAMES = CFG['candidates']
CACHE = OUT / 'cache'
MODELS = OUT / 'models'


def log(msg):
    print(msg, flush=True)
    with (OUT / 'run.log').open('a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')


def project_gpu(tr, te, fit, dim):
    z = torch.as_tensor(tr, device=DEVICE, dtype=torch.float64)
    selected = z[torch.as_tensor(fit, device=DEVICE)]
    mu = selected.mean(0)
    _, _, vh = torch.linalg.svd(selected - mu, full_matrices=False)
    basis = vh[:min(dim, len(selected) - 1)].T
    test = torch.as_tensor(te, device=DEVICE, dtype=torch.float64)
    return ((z - mu) @ basis).float().cpu().numpy(), ((test - mu) @ basis).float().cpu().numpy(), (mu.cpu().numpy(), basis.cpu().numpy())


def perturb(raw, ids, view):
    # Per-row seeds make augmentation independent of batch size and fold order.
    rngs = [np.random.default_rng(CFG['seed'] + int(i) * 31 + view) for i in ids]
    name = CFG['views'][view]
    x = raw.clone()
    if name == 'channel_gain':
        db = np.stack([r.uniform(-CFG['channel_gain_db'], CFG['channel_gain_db'], 18) for r in rngs])
        x *= torch.as_tensor(10 ** (db / 20), device=DEVICE, dtype=x.dtype)[:, :, None]
    elif name == 'global_gain':
        db = np.array([r.uniform(-CFG['global_gain_db'], CFG['global_gain_db']) for r in rngs])
        x *= torch.as_tensor(10 ** (db / 20), device=DEVICE, dtype=x.dtype)[:, None, None]
    elif name.startswith('tilt'):
        freq = torch.fft.rfftfreq(x.shape[-1], d=1 / 300, device=DEVICE)
        alpha = CFG['tilt_exponent'] * (1 if name == 'tilt_up' else -1)
        weight = (freq.clamp_min(1) / 10).pow(alpha)
        x = torch.fft.irfft(torch.fft.rfft(x) * weight, n=x.shape[-1])
    else:
        channels = torch.tensor([r.integers(18) for r in rngs], device=DEVICE)
        x[torch.arange(len(x), device=DEVICE), channels] = x.mean(1)
        x -= CFG['reference_fraction'] * x.mean(1, keepdim=True)
    return x


class Augmentation:
    def __init__(self, meta, original, fingerprint):
        self.meta, self.original = meta, original
        self.raw = np.load(ROOT / CFG['raw_train'], mmap_mode='r')
        assert self.raw.shape == (len(meta), 18, 1200)
        self.e36 = F.load_e36_config()
        self.e65 = F.load_config()
        self.path = CACHE / f'waveform_features_{fingerprint}.npz'
        self.values = np.zeros((len(meta), len(CFG['views']), original.shape[1]), np.float32)
        self.done = np.zeros(len(meta), bool)
        if self.path.exists():
            with np.load(self.path) as z:
                self.values, self.done = z['features'], z['done']
        self.parity = None

    def ensure(self, fit):
        ids = np.flatnonzero(fit & ~self.done)
        for start in range(0, len(ids), CFG['batch_size']):
            rows = ids[start:start + CFG['batch_size']]
            raw = torch.as_tensor(np.array(self.raw[rows]), device=DEVICE)
            with torch.no_grad():
                if self.parity is None:
                    rebuilt = F.gpu_correlation_logspd(raw, self.e36, self.e65).cpu().numpy()
                    self.parity = float(np.max(np.abs(rebuilt - self.original[rows])))
                    if not np.allclose(rebuilt, self.original[rows], atol=2e-4, rtol=2e-4):
                        raise RuntimeError(f'Raw/cache E65 parity failed: max_abs={self.parity}')
                    log(f'Raw/cache parity max_abs={self.parity}')
                for v in range(len(CFG['views'])):
                    self.values[rows, v] = F.gpu_correlation_logspd(perturb(raw, rows, v), self.e36, self.e65).cpu().numpy()
            self.done[rows] = True
        if len(ids):
            np.savez_compressed(self.path, features=self.values, done=self.done)
            log(f'Augmented raw fit rows cached: {self.done.sum()}')


def prototypes_all(x, state, fit, subjects, tasks, aug):
    aug.ensure(fit)
    ps, keys = E.prototypes(x, subjects, tasks, fit, np.unique(subjects[fit]), SPEC['k'])
    outputs = {'baseline': ps}
    transformed = np.empty((fit.sum(), len(CFG['views']), x.shape[1]), np.float32)
    rows = np.flatnonzero(fit)
    st = state['e65']
    mu, basis = st['pca']
    for task in E.TASKS:
        mask = tasks[rows] == task
        center, sd = st['standardization'][task]
        a = (aug.values[rows[mask]] - center) / sd
        transformed[mask] = ((a.astype(np.float64) - mu) @ basis).astype(np.float32)
    combined = np.concatenate([x[rows, None], transformed], 1)
    means, trimmed = ps.copy(), ps.copy()
    for task, sub in sorted(set(keys)):
        rr = np.flatnonzero((tasks[rows] == task) & (subjects[rows] == sub))
        cols = np.array([i for i, key in enumerate(keys) if key == (task, sub)])
        assignment = torch.cdist(torch.as_tensor(x[rows[rr]], device=DEVICE), torch.as_tensor(ps[cols], device=DEVICE)).argmin(1).cpu().numpy()
        for j, col in enumerate(cols):
            vals = combined[rr[assignment == j]].reshape(-1, x.shape[1])
            if len(vals):
                means[col] = vals.mean(0)
                cut = int(len(vals) * CFG['trim_fraction'])
                ordered = np.sort(vals, axis=0)
                trimmed[col] = ordered[cut:len(vals)-cut].mean(0)
    outputs['aug_mean'], outputs['aug_trimmed'] = means, trimmed
    return outputs, keys


def build_model(tag, fit, query, raw, meta, aug, fingerprint):
    path = MODELS / f'{tag}_{fingerprint}.npz'
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            assert np.array_equal(z['fit_ids'], np.flatnonzero(fit))
            assert np.array_equal(z['query_ids'], np.flatnonzero(query))
            return {n: z[n + '_scores'] for n in NAMES}
    x, _, state = E.make_representation(SPEC, raw, meta, meta.iloc[:0], fit)
    all_ps, keys = prototypes_all(x, state, fit, meta.subject.to_numpy(), meta.task.to_numpy(), aug)
    saved = dict(fit_ids=np.flatnonzero(fit), query_ids=np.flatnonzero(query), state=np.array([state], object), prototype_tasks=np.array([k[0] for k in keys]), prototype_subjects=np.array([k[1] for k in keys]))
    result = {}
    for name, ps in all_ps.items():
        result[name] = E.scores(x[query], meta.task.to_numpy()[query], ps, keys)
        saved[name + '_scores'], saved[name + '_prototypes'] = result[name], ps
    np.savez_compressed(path, **saved)
    return result


def inner(tag, outer_fit, raw, meta, blocks, aug, fingerprint):
    subjects = meta.subject.to_numpy()
    classes = np.unique(subjects[outer_fit])
    # Two disjoint pseudo-unknown identity groups cover every outer-fit identity.
    groups = np.array_split(classes, CFG['inner_subject_groups'])
    known = {n: [] for n in NAMES}
    unknown = {n: [] for n in NAMES}
    for g, held in enumerate(groups):
        for b in np.unique(blocks[outer_fit]):
            fit = outer_fit & ~np.isin(subjects, held) & (blocks != b)
            query = outer_fit & (blocks == b)
            assert not np.any(fit & query)
            result = build_model(f'{tag}_inner_g{g}_b{b}', fit, query, raw, meta, aug, fingerprint)
            is_unknown = np.isin(subjects[query], held)
            for n in NAMES:
                known[n].extend(result[n][~is_unknown].tolist())
                unknown[n].extend(result[n][is_unknown].tolist())
    return {n: (np.array(known[n]), np.array(unknown[n])) for n in NAMES}


def calibrated(known, unknown):
    threshold = float(np.nextafter(np.quantile(unknown, CFG['target_rejection'], method='higher'), np.inf))
    q = float(np.mean(known < threshold))
    return threshold, q


def metrics(sk, su, threshold):
    y = np.r_[np.zeros(len(sk)), np.ones(len(su))]
    scores = np.r_[sk, su]
    return {'AUC': E.auc_ties(y, -scores), 'unknown_positive_F1': E.binary_f1(y, scores < threshold), 'unknown_rejection': float(np.mean(su < threshold)), 'known_acceptance': float(np.mean(sk >= threshold)), 'n_unknown': len(su), 'n_known': len(sk), 'threshold': threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=20)
    parser.add_argument('--no-test', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.folds <= 20:
        raise ValueError('--folds must be 1..20')
    CACHE.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    started = time.time()
    torch.manual_seed(CFG['seed']); torch.set_num_threads(4)
    E.project = project_gpu
    fingerprint = hashlib.sha256((OUT / 'config.json').read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    meta = pd.read_csv(ROOT / E.CFG['data']['train_metadata'])
    features = np.load(ROOT / E.CFG['data']['e65_train']).astype(np.float32)
    raw = {'e65': (features, np.empty((0, features.shape[1]), np.float32))}
    aug = Augmentation(meta, features, fingerprint)
    subjects = meta.subject.to_numpy(int); fr = E.frac(meta)
    blocks = np.floor(fr * E.CFG['blocks']).astype(int)
    eligible = fr <= E.FIT_FRACTION
    evaluation = (fr > .6) & (fr <= .8)
    results = {n: {'folds': []} for n in NAMES}
    for held in E.CLASSES[:args.folds]:
        tag = f'outer_{held:02d}'
        fit = eligible & (subjects != held)
        refs = inner(tag, fit, raw, meta, blocks, aug, fingerprint)
        scored = build_model(tag, fit, evaluation, raw, meta, aug, fingerprint)
        is_unknown = subjects[evaluation] == held
        for n in NAMES:
            threshold, q = calibrated(*refs[n])
            row = metrics(scored[n][~is_unknown], scored[n][is_unknown], threshold)
            row.update(subject=int(held), inner_known_quantile=q, inner_unknown_rejection=float(np.mean(refs[n][1] < threshold)))
            results[n]['folds'].append(row)
        log(f'{tag} complete: ' + str({n: round(results[n]['folds'][-1]['unknown_rejection'], 4) for n in NAMES}))
    for n in NAMES:
        rows = results[n]['folds']
        results[n]['AUC'] = float(np.mean([r['AUC'] for r in rows]))
        results[n]['unknown_positive_F1'] = float(np.mean([r['unknown_positive_F1'] for r in rows]))
        for key, weight in [('unknown_rejection', 'n_unknown'), ('known_acceptance', 'n_known')]:
            results[n][key] = float(np.average([r[key] for r in rows], weights=[r[weight] for r in rows]))
    if args.folds == 20 and not args.no_test:
        refs = inner('final', eligible, raw, meta, blocks, aug, fingerprint)
        thresholds = {n: calibrated(*refs[n])[0] for n in NAMES}
        # Freeze all thresholds before opening Test metadata/features.
        (OUT / 'frozen_thresholds.json').write_text(json.dumps(thresholds, indent=2), encoding='utf-8')
        tm = pd.read_csv(ROOT / E.CFG['data']['test_metadata'])
        te = np.load(ROOT / E.CFG['data']['e65_test']).astype(np.float32)
        x, xt, state = E.make_representation(SPEC, {'e65': (features, te)}, meta, tm, eligible)
        all_ps, keys = prototypes_all(x, state, eligible, subjects, meta.task.to_numpy(), aug)
        s1 = tm.session.astype(str).str.lower().isin(['session1', 's1', '1']).to_numpy()
        for n, ps in all_ps.items():
            score = E.scores(xt, tm.task.to_numpy(), ps, keys)
            results[n]['test'] = {'S1_rejection': float(np.mean(score[s1] < thresholds[n])), 'S2_acceptance': float(np.mean(score[~s1] >= thresholds[n])), 'official_AUC': None, 'official_F1': None}
            np.savez_compressed(MODELS / f'final_{n}_{fingerprint}.npz', prototypes=ps, prototype_tasks=np.array([k[0] for k in keys]), prototype_subjects=np.array([k[1] for k in keys]), state=np.array([state], object), threshold=thresholds[n], fit_ids=np.flatnonzero(eligible), test_scores=score, inner_known=refs[n][0], inner_unknown=refs[n][1])
    payload = {'device': str(DEVICE), 'fold_count': args.folds, 'fingerprint': fingerprint, 'runtime_seconds': time.time()-started, 'results': results}
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    lines = ['# E241 原始EEG鲁棒原型增强', '', f'- 实际完成 {args.folds}/20 outer folds，设备 {DEVICE}，耗时 {payload["runtime_seconds"]:.1f}s。', '- 完整阅读并复用E215特征路径、任务标准化、PCA/k12原型、GPU距离、分折与指标。PCA改为CUDA float64 SVD，数学目标不变。', '- 只在当前fit行上生成确定性原始EEG扰动，再调用E65 GPU提取器；样本级缓存不含跨样本统计。原始聚类分配固定，aug_mean/aug_trimmed以原始及五个增强视图更新每个簇中心；trimmed逐维去掉两端15%。', '- inner覆盖所有非空时间块，另划分两个身份组，pseudo-unknown身份完全不参与相应内层标准化/PCA/原型。所有outer held身份完全不参与其阈值标定。', '- 每个候选独立用inner unknown OOF的90%分位数设阈值（strict < 的边界采用nextafter）；同时记录对应known OOF分位数。90%是inner目标，不保证outer实测恰为90%；不得依据outer或Test回调阈值。', '- 两个inner身份组的注册库比outer小，存在校准迁移偏差；不应将inner=0.90冒充完整LOSO=0.90。', '- E215 acceptance=0.90不是unknown rejection=0.90。这里baseline也重新标定，不直接引用旧结果。', '- AUC为每outer fold unknown-positive AUC均值（使用负距离置信度）；F1为unknown-positive二分类F1逐折均值。Test无真实身份标签，官方AUC/F1均N/A。S1/S2只作会话描述，非官方准确率。', '- 三个候选与扰动强度预先固定，未依据Test选型；历史反复查看Test的研究偏差仍存在。', '', '|候选|LOSO AUC|unknown-positive F1|LOSO拒识micro|known接受micro|Test S1拒识|Test S2接受|', '|---|---:|---:|---:|---:|---:|---:|']
    for n, r in results.items():
        t = r.get('test', {})
        lines.append(f'|{n}|{r["AUC"]:.6f}|{r["unknown_positive_F1"]:.6f}|{r["unknown_rejection"]:.6f}|{r["known_acceptance"]:.6f}|{t.get("S1_rejection", "N/A")}|{t.get("S2_acceptance", "N/A")}|')
    lines += ['', '## 结论', '仅当完整20折实测拒识约0.90且S2高于同协议baseline时，才可宣称满足约束的提升。未满足者为负结果；部分折不代表完整LOSO。', '', '## 复现与产物', '命令：`.venv-py311\\Scripts\\python.exe research\\E241_robust_prototype_aug\\run_e241.py`；可行性阶段使用`--folds 1 --no-test`。', 'config.json固定扰动与协议；cache保存原始波形重提特征；models保存每内外折fit/query行ID、标准化/PCA状态、三种原型与分数，完整运行保存最终模型/阈值/OOF分数。文件名以脚本+config哈希隔离，既有实验只读。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
