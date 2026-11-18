"""Render fold-matched results. Run after run.py."""
from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent
obj=json.loads((H/'results.json').read_text()); rows=obj['folds']; summ=obj['summary']
text=['# E262：原始波形配对训练能否改善老用户匹配？','', '20 个身份外折等权。每格为均值 / 最差折；错排、排对被拒取最大为最差，其他指标取最小。F1 为已知身份闭集 macro-F1。老用户端到端要求身份正确且被接受。','', '|方案、晚段查询|AUROC|U@95|rej@90|闭集 F1|老用户端到端|已知错排|已知排对被拒|未知拒识|','|---|'+'---:|'*8]
for arm,metrics in summ.items():text.append('|'+arm+'|'+'|'.join(f'{a:.5f} / {b:.5f}' for a,b in metrics.values())+'|')
text+=['','## 配对差值','', '|晚段视图|配对模型相对|老用户端到端差（百分点）|改善折数 / 20|未知拒识差（百分点）|','|---|---|---:|---:|---:|']
for view in ['clean','stress']:
    for base in ['E254','clean_net']:
        def diff(m):return np.array([r['paired_net_'+view][m]-r[base+'_'+view][m] for r in rows])
        d=diff('old_e2e'); text.append(f'|{view}|{base}|{100*d.mean():+.3f}|{sum(d>0)}|{100*diff("unknown_reject").mean():+.3f}|')
text+=['','## 方法与边界','','- Train 的每个身份 × 任务按 trial/max(trial) 切分。仅 19 个老用户前 60% 拟合变换、深度网络和注册原型；仅其中段 60%–80% 标定各自第 10 百分位阈值。所有身份末 20% 只评估。重复折共用试次，不作独立重复。','- 三组均沿用固定 E254 投影、四原型和门控。clean_net、paired_net 使用相同初始化、采样、160 次优化及 256→96→256 残差网络。训练将输出拉向正确身份的任务中心和同一试次的干净方向；唯一实验差异是第二视图为原样或波形扰动。不是端到端波形 CNN。','- 注册模板与阈值仅使用干净数据；stress 使用另一固定扰动条件，未参与训练或阈值标定。正式 ROC 工作点不使用冻结阈值：从晚段已知分数降序取首个接受率达到目标的分数，并计算未知拒识。','- 时间块来自同一可见 session；模拟变化不是实际跨设备或初赛 S2 成绩。未评估新人，不据此推断新人注册改善。','- 复跑顺序：views.py → run.py → report.py。模型保存在 weights/held01.pt 至 held20.pt；逐折数值见 results.json。']
text+=['','## 提交决策与实现差异','','E262 不替换 final2：干净三项 ROC 均值过线，但冻结门槛下扰动老用户端到端从 82.34% 降至 81.86%；未知拒识则提高。未读取 Test。完整波形映射与配置见 INPUT.md。','', '额外审计发现历史研究基线和现包老身份打分并不相同：freeze_e254.py 已用白化后的向量建原型，而 final2 的 model.py 在打分前再次白化这些已存原型，查询仍只白化一次。已用第一任务、第一身份、第一原型重算核对：已存原型与一次白化均值的相对误差 2.66e-7，与未白化均值的相对误差 0.681。不能直接将 E254_clean 的分数标为现包评分。package_metric.py 按现包老用户规则逐折重拟合，干净 AUROC/U@95/rej@90 为 0.98858/0.95465/0.98814（最差 0.95382/0.63710/0.87903），闭集 F1 为 0.99454，老用户端到端 86.66%。尚不能从这次比较推断二次白化对 Test 的因果影响。','', '现包 Train 样本内审计：4 条闭集错例全是 Imagery，均被拒；另有 196 条身份排对但被拒，端到端 11104/11304，详见 train_errors.json。']
(H/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
