"""Train-only paired replay of the E279 decision loop on CUDA."""
from pathlib import Path
import inspect
import runpy

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
E279 = H.parent / 'E279_relative_margin'
METRICS = ['old', 'new', 'unknown', 'false_reject', 'auc', 'f1']
KEY = ['seed', 'scene', 'shift', 'method']


def replay():
    # Execute only the original setup and evaluation, not E279's output writer.
    path = E279 / 'run.py'
    body = path.read_text(encoding='utf8').split('d=pd.DataFrame(out);', 1)
    assert len(body) == 2
    setup, loop = body[0].split('with torch.inference_mode():', 1)
    ns = {'__file__': str(path)}
    exec(compile(setup, str(path), 'exec'), ns)
    g = ns['g']
    original = g['score']
    src = inspect.getsource(original)
    assert src.count('.amax(2)') == 1
    env = original.__globals__.copy()
    exec(src.replace('.amax(2)', '.topk(2, dim=2).values.mean(2)'), env)
    frames = []
    for pool, fn in [('max', original), ('top2', env['score'])]:
        ns['out'] = []
        g['score'] = fn
        exec(compile('with torch.inference_mode():' + loop, str(path), 'exec'), ns)
        frames.append(pd.DataFrame(ns['out']).assign(pool=pool))
    ref = pd.read_json(E279 / 'results.json').set_index(KEY).sort_index()
    base = frames[0].set_index(KEY).sort_index()
    assert ref.index.equals(base.index)
    err = np.abs(ref[METRICS]-base[METRICS]).to_numpy().max()
    assert err < 1e-8, err
    return pd.concat(frames, ignore_index=True), err


def report(d, err):
    d.to_json(H/'results.json', orient='records', indent=2, double_precision=15)
    lines = ['# E280：四原型 top2 均值与 E279 最大值配对实验', '',
             '仅在每个任务、每个身份内部，把四原型最大余弦换成最高两个余弦的等权均值。跨任务最大值不变，随后按各方案自己的身份分数重新计算赢家、身份分差、标准差和全部门控；未调阈值。', '',
             '只读 Train 缓存，GPU 推理；复用 E279 的 286、312、346 三个冻结折、晚段查询、任务掩码和注册权重。每折 14 个旧身份、3 个模拟新人、3 个模拟未知。AUC 为已知对未知的连续接受余量 ROC AUC；F1 为含未知类的宏 F1。场景内三折等权，总体再对四场景等权。stress 为模拟扰动，不是隐藏测试成绩。', '',
             f'最大值基线与 E279 已存逐折结果在六项指标上的最大绝对误差为 {err:.2g}。', '',
             '## 主比较：相对分差门控，原分数不偏移', '', '|场景|E279 AUC|E280 AUC|Δ AUC|E279 F1|E280 F1|Δ F1|','|---|---:|---:|---:|---:|---:|---:|']
    main = d.query("method=='relative' and shift==0")
    for scene, z in [*main.groupby('scene', sort=False), ('四场景均值', main)]:
        p = z.groupby('pool')[['auc','f1']].mean(); a,b = p.loc['max'],p.loc['top2']
        lines.append(f'|{scene}|{a.auc:.6f}|{b.auc:.6f}|{b.auc-a.auc:+.6f}|{a.f1:.6f}|{b.f1:.6f}|{b.f1-a.f1:+.6f}|')
    lines += ['', '## 全部门控与偏移对照（三折均值）', '', '偏移量沿用 E279，仅从绝对余量中扣除。四项等权为完整 clean 旧人、完整 stress 旧人、完整 clean 新人、完整 clean 未知拒识四者均值。', '', '|偏移量|门控|原型聚合|旧人 clean|旧人 stress|新人 clean|未知拒识 clean|四项等权|AUC|宏 F1|', '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for (shift,gate,pool), z in d.groupby(['shift','method','pool']):
        means = z.groupby('scene')[METRICS].mean(); c=means.loc['clean_complete']; st=means.loc['stress_complete']; v=[c.old,st.old,c['new'],c.unknown]
        lines.append(f'|{shift}|{gate}|{pool}|'+ '|'.join(f'{n:.6f}' for n in [*v,np.mean(v),z.auc.mean(),z.f1.mean()])+'|')
    lines += ['', '逐折、逐场景完整指标见 `results.json`。复现：在项目根目录运行 `.\\.venv-py311\\Scripts\\python.exe -B research\\E280_proto_consensus\\run.py`。']
    (H/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8'); print('\n'.join(lines))


if __name__ == '__main__':
    report(*replay())
