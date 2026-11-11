# Session-blind threshold sweep v3

## 推荐与边界

推荐候选：`test_candidate_accept_0p995.csv`。这是预先固定的 Train-only 折中（目标已知接受率99.5%），不是根据 Test 总拒识或 Session 审计选出的最优阈值。正式提交未修改。

## v2 严格外层验证基准（沿用，非本轮阈值的验证结果）

- AUC：0.965632；身份 macro-F1：0.936207。
- 检测 macro-F1：0.936935；未知 F1：0.941038；U：0.915591。
- v2 外层 subject 五折、已知末20% trial 留出；内层伪未知仅使用外层注册用户。此轮不重新训练外层，不将基准冒充新阈值成绩。

## 推理与阈值协议

- 固定 v2 置信指标 `margin`；复用 v2 五个连续 trial 块 OOF，不使用 Test 分布选择阈值。
- Train/Test 原始 full/window 特征先取同一 COV_COLUMNS，再把相邻窗口重排为 (trial, 2, 1026)，直接复用 v2 load_features。全量 TaskSpecificModel(alpha=100) 在 CUDA 上计算 fused scores。
- 推理前读取的 Train 元数据仅 subject/task，Test 仅 sample/task；模型接口只有 EEG 特征及 task，sample 只对齐提交顺序。确认推理不读取 session。
- 所有六个候选 CSV 写出后才首次读取 session，Session1/Session2/task 仅作事后审计，不参与阈值、候选选择或后处理。
- 目标阈值沿用 v2：取最接近目标的可达接受率，保留相同分数；q01=OOF 第1百分位，q001=第0.1百分位，采用 linear 插值。接受条件为 confidence >= threshold。
- 目标1.0对应 OOF 最小置信度，不代表 Test 全接受。已知-only OOF 无未知标签，无法计算检测 AUC/U。

## 阈值敏感性

|候选|阈值|Train OOF known acceptance|OOF身份 macro-F1|Test拒识/总数|Test总拒识率|
|---|---:|---:|---:|---:|---:|
|accept_0p99|0.3894340396|0.990004|0.947332|11849/22251|53.251539%|
|accept_0p995|0.2533782572|0.994958|0.949748|9086/22251|40.834120%|
|accept_0p999|0.05299857259|0.999027|0.951305|2524/22251|11.343310%|
|accept_1p0|0.01519984007|1.000000|0.998847|767/22251|3.447036%|
|q01|0.3894429517|0.989915|0.947291|11849/22251|53.251539%|
|q001|0.05649296084|0.998938|0.951351|2670/22251|11.999461%|

## Session1/Session2 与 task 事后拒识审计

下表只是无标签分组拒识统计，不是准确率、AUC、F1 或未知拒识率 U；不从 Session 推断真实身份。

|候选|审计维度|组|拒识/总数|拒识率|
|---|---|---|---:|---:|
|accept_0p99|session|session1|1533/3033|50.544016%|
|accept_0p99|session|session2|10316/19218|53.678843%|
|accept_0p99|task|EC|368/1972|18.661258%|
|accept_0p99|task|EO|1419/2312|61.375433%|
|accept_0p99|task|Imagery|7100/11207|63.353261%|
|accept_0p99|task|SSVEP|2962/6760|43.816568%|
|accept_0p99|session_task|session1/EC|62/292|21.232877%|
|accept_0p99|session_task|session1/EO|153/291|52.577320%|
|accept_0p99|session_task|session1/Imagery|825/1459|56.545579%|
|accept_0p99|session_task|session1/SSVEP|493/991|49.747730%|
|accept_0p99|session_task|session2/EC|306/1680|18.214286%|
|accept_0p99|session_task|session2/EO|1266/2021|62.642256%|
|accept_0p99|session_task|session2/Imagery|6275/9748|64.372179%|
|accept_0p99|session_task|session2/SSVEP|2469/5769|42.797712%|
|accept_0p995|session|session1|1289/3033|42.499176%|
|accept_0p995|session|session2|7797/19218|40.571339%|
|accept_0p995|task|EC|258/1972|13.083164%|
|accept_0p995|task|EO|1106/2312|47.837370%|
|accept_0p995|task|Imagery|5542/11207|49.451236%|
|accept_0p995|task|SSVEP|2180/6760|32.248521%|
|accept_0p995|session_task|session1/EC|46/292|15.753425%|
|accept_0p995|session_task|session1/EO|125/291|42.955326%|
|accept_0p995|session_task|session1/Imagery|727/1459|49.828650%|
|accept_0p995|session_task|session1/SSVEP|391/991|39.455096%|
|accept_0p995|session_task|session2/EC|212/1680|12.619048%|
|accept_0p995|session_task|session2/EO|981/2021|48.540327%|
|accept_0p995|session_task|session2/Imagery|4815/9748|49.394748%|
|accept_0p995|session_task|session2/SSVEP|1789/5769|31.010574%|
|accept_0p999|session|session1|409/3033|13.484998%|
|accept_0p999|session|session2|2115/19218|11.005308%|
|accept_0p999|task|EC|58/1972|2.941176%|
|accept_0p999|task|EO|283/2312|12.240484%|
|accept_0p999|task|Imagery|1640/11207|14.633711%|
|accept_0p999|task|SSVEP|543/6760|8.032544%|
|accept_0p999|session_task|session1/EC|14/292|4.794521%|
|accept_0p999|session_task|session1/EO|42/291|14.432990%|
|accept_0p999|session_task|session1/Imagery|242/1459|16.586703%|
|accept_0p999|session_task|session1/SSVEP|111/991|11.200807%|
|accept_0p999|session_task|session2/EC|44/1680|2.619048%|
|accept_0p999|session_task|session2/EO|241/2021|11.924790%|
|accept_0p999|session_task|session2/Imagery|1398/9748|14.341403%|
|accept_0p999|session_task|session2/SSVEP|432/5769|7.488300%|
|accept_1p0|session|session1|139/3033|4.582921%|
|accept_1p0|session|session2|628/19218|3.267770%|
|accept_1p0|task|EC|18/1972|0.912779%|
|accept_1p0|task|EO|88/2312|3.806228%|
|accept_1p0|task|Imagery|495/11207|4.416882%|
|accept_1p0|task|SSVEP|166/6760|2.455621%|
|accept_1p0|session_task|session1/EC|5/292|1.712329%|
|accept_1p0|session_task|session1/EO|13/291|4.467354%|
|accept_1p0|session_task|session1/Imagery|87/1459|5.962988%|
|accept_1p0|session_task|session1/SSVEP|34/991|3.430878%|
|accept_1p0|session_task|session2/EC|13/1680|0.773810%|
|accept_1p0|session_task|session2/EO|75/2021|3.711034%|
|accept_1p0|session_task|session2/Imagery|408/9748|4.185474%|
|accept_1p0|session_task|session2/SSVEP|132/5769|2.288092%|
|q01|session|session1|1533/3033|50.544016%|
|q01|session|session2|10316/19218|53.678843%|
|q01|task|EC|368/1972|18.661258%|
|q01|task|EO|1419/2312|61.375433%|
|q01|task|Imagery|7100/11207|63.353261%|
|q01|task|SSVEP|2962/6760|43.816568%|
|q01|session_task|session1/EC|62/292|21.232877%|
|q01|session_task|session1/EO|153/291|52.577320%|
|q01|session_task|session1/Imagery|825/1459|56.545579%|
|q01|session_task|session1/SSVEP|493/991|49.747730%|
|q01|session_task|session2/EC|306/1680|18.214286%|
|q01|session_task|session2/EO|1266/2021|62.642256%|
|q01|session_task|session2/Imagery|6275/9748|64.372179%|
|q01|session_task|session2/SSVEP|2469/5769|42.797712%|
|q001|session|session1|431/3033|14.210353%|
|q001|session|session2|2239/19218|11.650536%|
|q001|task|EC|69/1972|3.498986%|
|q001|task|EO|304/2312|13.148789%|
|q001|task|Imagery|1723/11207|15.374320%|
|q001|task|SSVEP|574/6760|8.491124%|
|q001|session_task|session1/EC|14/292|4.794521%|
|q001|session_task|session1/EO|44/291|15.120275%|
|q001|session_task|session1/Imagery|258/1459|17.683345%|
|q001|session_task|session1/SSVEP|115/991|11.604440%|
|q001|session_task|session2/EC|55/1680|3.273810%|
|q001|session_task|session2/EO|260/2021|12.864918%|
|q001|session_task|session2/Imagery|1465/9748|15.028724%|
|q001|session_task|session2/SSVEP|459/5769|7.956318%|

## 局限与复现

Test 无真实标签，Test AUC/F1 均不可计算。降低阈值可能减少已知误拒，也可能增加未知误接受；此轮无法估计这种权衡的真实收益。不能把 Test 审计当验证指标。
全量拟合与 OOF 折模型样本量不同，分数分布仍可能偏移；v2 外层基准不证明新候选更好。未重新评估对抗鲁棒性或跨 Session 泛化。

运行：`py -3 research/E16_session_blind_threshold_sweep_v3/run.py --output-dir <新的空目录>`。依赖现有 numpy/pandas/scikit-learn/PyTorch CUDA；metrics.json 保留来源哈希、v2 外层各折及完整审计。现有输出一律不覆盖。
