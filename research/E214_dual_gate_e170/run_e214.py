"""E214: identity-excluded nested score-level gate, frozen reliability branch."""
from pathlib import Path
import json
import time
import itertools
import numpy as np
import pandas as pd
import torch
def roc_auc_score(y, score):
    y = np.asarray(y, bool)
    ranks = pd.Series(score).rank(method='average').to_numpy()
    n = y.sum()
    return (ranks[y].sum() - n * (n + 1) / 2) / (n * (~y).sum())


def f1_score(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp = np.sum(y & pred)
    return 2 * tp / max(int(y.sum() + pred.sum()), 1)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REGS = [0.001, 0.01, 0.1, 1.0]
ACCEPTANCE = np.round(np.arange(0.80, 1.0001, 0.005), 3)
TAGS = ['a_E143_E65_PCA96_k12', 'd_E65_PCA64_k12']


def features(s):
    return np.column_stack([s, np.abs(s[:, 0] - s[:, 1])]).astype(np.float32)


def main():
    start = time.time()
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(ROOT / 'research/E170_three_way_fusion_ceiling/frozen_scores.npz')
    meta = pd.read_csv(ROOT / 'BrainPrint_cache/train_metadata.csv')
    subjects = meta.subject.to_numpy(int) - 1
    frac = meta.trial.to_numpy() / meta.groupby(['subject', 'task']).trial.transform('max').to_numpy()
    window = (frac > .6) & (frac <= .8)
    train_s = np.column_stack([z[f'z_rf_{t}'] for t in TAGS])
    us, ks, ug = [], [], []
    for i in range(20):
        u = np.column_stack([z[f'loso_unknown_{t}_{i}'] for t in TAGS])
        j = (i + 1) % 20
        k = np.column_stack([z[f'loso_known_{t}_{j}'] for t in TAGS])
        ids = subjects[window & (subjects != j)]
        assert len(ids) == len(k) and len(u) == np.sum(window & (subjects == i))
        us.append(u)
        ks.append(k[ids == i])
        ug.extend([i] * len(u))
    x = features(np.vstack([train_s, np.concatenate(us)]))
    y = np.r_[np.ones(len(train_s)), np.zeros(len(ug))].astype(np.float32)
    groups = np.r_[subjects, ug]
    exclusions = [()] + [(i,) for i in range(20)] + list(itertools.combinations(range(20), 2))
    specs = [(ex, reg) for ex in exclusions for reg in REGS]
    lookup = {(ex, reg): n for n, (ex, reg) in enumerate(specs)}
    tx = torch.as_tensor(x, device=device)
    ty = torch.as_tensor(y[:, None], device=device)
    mask = np.stack([~np.isin(groups, ex) for ex, _ in specs], axis=1)
    weights = torch.as_tensor(mask, dtype=torch.float32, device=device)
    weights *= ty / (weights * ty).sum(0) + (1 - ty) / (weights * (1 - ty)).sum(0)
    weights *= .5
    mu = weights.T @ tx
    sd = ((weights.T @ tx.square() - mu.square()).clamp_min(1e-8)).sqrt()
    w = torch.zeros_like(mu)
    b = torch.zeros(len(specs), device=device)
    reg = torch.tensor([r for _, r in specs], device=device)
    for step in range(600):
        logits = tx @ (w / sd).T + b - (mu * w / sd).sum(1)
        delta = (logits.sigmoid() - ty) * weights
        w -= .1 * ((delta.T @ tx - delta.sum(0)[:, None] * mu) / sd + reg[:, None] * w)
        b -= .1 * delta.sum(0)
        if step % 200 == 0:
            print(f'GPU batched logistic step {step}/600, models={len(specs)}', flush=True)
    models = [a.detach().cpu().numpy() for a in (w, b, mu, sd)]
    del weights, tx, ty, delta, logits

    def predict(s, ex, r):
        idx = lookup[(tuple(sorted(ex)), r)]
        ww, bb, mm, ss = (a[idx] for a in models)
        v = (features(s) - mm) / ss @ ww + bb
        return 1 / (1 + np.exp(-np.clip(v, -40, 40)))

    def validation(ex, r):
        pp, kk, uu = [], [], []
        for j in range(20):
            if j in ex:
                continue
            held = tuple(sorted((*ex, j)))
            pp.append(predict(train_s[subjects == j], held, r))
            kk.append(predict(ks[j], held, r))
            uu.append(predict(us[j], held, r))
        return np.concatenate(pp), np.concatenate(kk), np.concatenate(uu)

    def choose(ex):
        candidates = []
        for r in REGS:
            p, k, u = validation(ex, r)
            curve = []
            for acc in ACCEPTANCE:
                th = float(np.quantile(p, 1 - acc))
                curve.append((float(np.mean(u < th)), float(np.mean(k >= th)), float(acc), th))
            eligible = [row for row in curve if row[1] >= .95]
            best = max(eligible or [curve[-1]], key=lambda row: (row[0], -row[2]))
            auc = float(roc_auc_score(np.r_[np.ones(len(k)), np.zeros(len(u))], np.r_[k, u]))
            candidates.append({'reg': r, 'auc': auc, 'rejection': best[0], 'known_acceptance': best[1], 'acceptance': best[2], 'threshold': best[3]})
        best = max(candidates, key=lambda row: (row['rejection'], row['auc'], row['reg']))
        return best, candidates

    selected, candidates = choose(())
    nested_p, nested_k, nested_u, outer = [], [], [], []
    npred, kpred, upred = [], [], []
    outer_models = []
    for i in range(20):
        choice, _ = choose((i,))
        r, th = choice['reg'], choice['threshold']
        p = predict(train_s[subjects == i], (i,), r)
        k = predict(ks[i], (i,), r)
        u = predict(us[i], (i,), r)
        nested_p.append(p); nested_k.append(k); nested_u.append(u)
        npred.extend(p >= th); kpred.extend(k >= th); upred.extend(u >= th)
        outer_models.append(lookup[((i,), r)])
        outer.append({'subject': i + 1, **choice, 'outer_rejection': float(np.mean(u < th)), 'outer_known_acceptance': float(np.mean(k >= th))})
    pp, kk, uu = map(np.concatenate, (nested_p, nested_k, nested_u))
    yy = np.r_[np.ones(len(pp)), np.zeros(len(uu))]
    train_auc = float(roc_auc_score(yy, np.r_[pp, uu]))
    train_f1 = float(f1_score(yy, np.r_[npred, upred]))
    loso_auc = float(roc_auc_score(np.r_[np.ones(len(kk)), np.zeros(len(uu))], np.r_[kk, uu]))
    r = selected['reg']
    # Freeze all model/threshold decisions before accessing Test score arrays.
    final_idx = lookup[((), r)]
    final_p, final_k, final_u = validation((), r)
    thresholds = [float(np.quantile(final_p, 1 - a)) for a in ACCEPTANCE]
    np.savez_compressed(OUT / 'model.npz', weights=models[0][final_idx], bias=models[1][final_idx],
                        feature_mean=models[2][final_idx], feature_scale=models[3][final_idx],
                        regularization=r, threshold=selected['threshold'], tags=TAGS,
                        reliability_weights=np.array([.5, .5]),
                        all_weights=models[0], all_bias=models[1], all_means=models[2], all_scales=models[3],
                        exclusion_masks=np.array([[i in ex for i in range(20)] for ex, _ in specs]),
                        all_regularization=np.array([r for _, r in specs]), outer_model_indices=outer_models)
    test_s = np.column_stack([z[f'z_te_{t}'] for t in TAGS])
    test_p = predict(test_s, (), r)
    is_s1 = z['is_s1']
    reliability = test_s.mean(1)
    accepted = test_p >= selected['threshold']
    curves = []
    for a, th in zip(ACCEPTANCE, thresholds):
        outer_rej, outer_acc = [], []
        for i in range(20):
            cal, _, _ = validation((i,), outer[i]['reg'])
            oth = float(np.quantile(cal, 1 - a))
            outer_rej.append(float(np.mean(nested_u[i] < oth)))
            outer_acc.append(float(np.mean(nested_k[i] >= oth)))
        curves.append({'acceptance': float(a), 'threshold': th,
                       'LOSO_rejection_nested_macro': float(np.mean(outer_rej)),
                       'LOSO_known_acceptance_nested_macro': float(np.mean(outer_acc)),
                       'Test_S1_rejection': float(np.mean(test_p[is_s1] < th)),
                       'Test_S2_acceptance': float(np.mean(test_p[~is_s1] >= th))})
    baseline = np.load(ROOT / 'research/E213_e170_adaptive_cross_session_gate/adaptive_scores.npz')
    bm = np.load(ROOT / 'research/E213_e170_adaptive_cross_session_gate/model.npz')
    assert np.array_equal(is_s1, baseline['is_s1'])
    baseline_rows = []
    for a in ACCEPTANCE:
        th = float(np.quantile(baseline['train_oof_gate'], 1 - a))
        baseline_rows.append({'acceptance': float(a), 'Test_S1_rejection': float(np.mean(baseline['test_gate'][is_s1] < th)), 'Test_S2_acceptance': float(np.mean(baseline['test_gate'][~is_s1] >= th))})
    frozen = {'acceptance': selected['acceptance'], 'threshold': selected['threshold'],
              'LOSO_rejection_nested_macro': float(np.mean([o['outer_rejection'] for o in outer])),
              'LOSO_known_acceptance_nested': float(np.mean(kpred)),
              'Test_S1_rejection': float(np.mean(~accepted[is_s1])),
              'Test_S2_acceptance': float(np.mean(accepted[~is_s1]))}
    limitations = [
        '严格性仅限新增 gate 的按样本身份排除及嵌套选参；冻结 E170 的 PCA/标准化和其他身份原型可能包含外层身份，缓存不足以实现端到端 strict LOSO。',
        'E213 将 LOSO unknown 用于拟合又用于评估；其正则选择及 AUC/拒识不是独立留出估计，不直接比较为公平提升。',
        'E170 c 流按 Test 诊断 AUC 选择；b 流也有历史选型风险。新增 gate 只用预定 a/d 两个 E65 流，不输入 E213 概率或其模型。E213 缓存与模型仅作冻结对照。',
        'Gate B 是独立的两流平均最近原型相似度可靠性分数，接受后输出，不改变 Gate A 决策；不是身份概率，缓存无类别分数/预测，身份 F1=N/A。',
        'Train F1 是 known/unknown 二分类 F1，不是 E170 的多身份 macro F1。S2>=0.70 仅事后目标核验，不能作为选点约束。',
        'Train session1 的身份留出不能模拟跨会话偏移；提高 Train known 接受率不能保证 Test S2。']
    result = {'experiment': 'E214_dual_gate_e170', 'device': device, 'selected': selected,
              'selection_rule': 'nested subject exclusion; maximize inner LOSO unknown rejection subject to inner known acceptance >=0.95; ties AUC then larger regularization; thresholds from inner cross-fitted Train OOF known',
              'train_oof_auc': train_auc, 'train_oof_f1': train_f1, 'loso_auc': loso_auc,
              'strict_end_to_end_loso': False, 'strict_score_level_subject_crossfit': True,
              'frozen_point': frozen, 'curve': curves, 'regularization_candidates': candidates,
              'outer_folds': outer, 'baseline_e213_curve': baseline_rows,
              'baseline_e213_regularization': float(bm['regularization']),
              'test_official': {'AUC': 'N/A', 'F1': 'N/A'}, 'identity_f1': 'N/A',
              'limitations': limitations, 'runtime_seconds': time.time() - start}
    np.savez_compressed(OUT / 'dual_scores.npz', train_oof_gate_nested=pp,
                        train_oof_subjects=np.concatenate([subjects[subjects == i] for i in range(20)]),
                        train_oof_reliability=np.concatenate([train_s[subjects == i].mean(1) for i in range(20)]),
                        loso_known_gate_nested=kk, loso_unknown_gate_nested=uu,
                        test_gate=test_p, test_reliability=reliability,
                        test_accepted_reliability=np.where(accepted, reliability, np.nan),
                        test_accepted=accepted, is_s1=is_s1,
                        e213_test_gate=baseline['test_gate'], thresholds=thresholds, acceptance=ACCEPTANCE)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# E214 双 gate / 双评分实验', '',
             '## 结果', '', f'- Train 嵌套 OOF gate AUC：{train_auc:.6f}；二分类 F1：{train_f1:.6f}。',
             f'- LOSO 嵌套 gate AUC：{loso_auc:.6f}；拒识：{frozen["LOSO_rejection_nested_macro"]:.6f}。',
             f'- 冻结工作点 acceptance={selected["acceptance"]:.3f}，正则={r}，阈值={selected["threshold"]:.6f}。',
             f'- Test S1 拒识：{frozen["Test_S1_rejection"]:.6f}；S2 接受：{frozen["Test_S2_acceptance"]:.6f}。',
             f'- S2>=0.70 事后目标：{"达到" if frozen["Test_S2_acceptance"] >= .7 else "未达到；未据此重新选点"}。',
             '- 官方 Test AUC/F1：**N/A**；身份分类 F1：N/A。', '',
             '## 协议与边界', '',
             'Train 20 个身份；EC/EO/Imagery/SSVEP。E170 五块 OOF 为正类，0.6<trial/max_trial<=0.8 的每身份 LOSO unknown 为负类。外层排除该身份全部正负训练行，内层再次按身份留出，拟合均值/尺度只用相应训练行。LOSO known 取同一外层身份在相邻留出折的 known 分数；每个身份恰评估一次，避免把其他折重复的 known 行混入评估。',
             '四个正则与所有单/双身份排除模型在 GPU 上批量拟合；类别平衡 logistic，600 步，学习率0.1。内层最大化拒识且 known 接受>=0.95；阈值来自内层 OOF 正样本分位数。最终部署模型全 Train 重训，阈值来自单身份交叉拟合。外层最终指标评估的是完整选参算法，曲线每折也使用仅内层校准的阈值。',
             'Gate A 负责接受；Gate B 使用独立平均相似度在接受样本上输出可靠性，不把身份/可靠性与开放集概率混成一个分数。', '',
             *['- ' + s for s in limitations], '',
             '## 完整预定阈值曲线（Test 仅冻结后审计）', '',
             '|OOF接受分位|阈值|嵌套LOSO拒识|嵌套known接受|Test S1拒识|Test S2接受|',
             '|---:|---:|---:|---:|---:|---:|']
    for row in curves:
        lines.append('| ' + ' | '.join(f'{row[k]:.6f}' for k in ['acceptance', 'threshold', 'LOSO_rejection_nested_macro', 'LOSO_known_acceptance_nested_macro', 'Test_S1_rejection', 'Test_S2_acceptance']) + ' |')
    lines += ['', '## 产物与运行', '',
              '`run_e214.py`、`results.json`、`model.npz`（最终及所有嵌套模型）、`dual_scores.npz`（双评分和交叉拟合缓存）。',
              '', r'运行：`.venv\Scripts\python.exe research\E214_dual_gate_e170\run_e214.py`',
              f'', f'运行耗时：{result["runtime_seconds"]:.1f} 秒；设备：{device}。未修改 E170/E213，未提交 git。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'train_auc': train_auc, 'train_f1': train_f1, 'loso_auc': loso_auc, **frozen}, indent=2), flush=True)


if __name__ == '__main__':
    main()
