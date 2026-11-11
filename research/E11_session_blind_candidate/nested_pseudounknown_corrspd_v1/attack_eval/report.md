# nested_pseudounknown_corrspd_v1 特征空间 PGD 攻击评估

## 结论

在固定预注册配置 **PGD 10 步、epsilon=0.05、步长 0.01** 下，对外层已知评估样本执行了两种特征表示的严格特征空间攻击。攻击预算按各方法训练特征标准差单位约束，实际观测最大标准化 L∞ 均为 0.0500000045。计算优先使用 GPU，未修改正式提交或原候选。

|方法|已知样本数|伪未知样本数|C|clean F1|adv F1|RA|ASafeR|A=0.7RA+0.3ASafeR|U|S_proxy|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|logcov|10687|617|0.094507|0.163764|0.162620|0.093852|0.999532|0.365556|0.949757|27.016142|
|corrSPD|10687|617|0.094320|0.163404|0.161448|0.093197|0.998971|0.364929|0.949757|26.978518|

公式：`S_proxy=100*C^0.40*A^0.35*U^0.25`。这里的 U 是每个外层折中留出一个身份产生的伪未知样本拒识率，不是 Test 或真实未知标签；因此 S_proxy 仅为内部协议代理分，不能解释为官方 Test 成绩。

原实验报告中的非攻击 blocked 结果仍保留在上级目录；本报告的 C/F1 是本次外层攻击评估协议下的 clean 结果，并非部署 Test 指标。

## 协议与命令

- 实际命令：`py -3.11 research/E11_session_blind_candidate/nested_pseudounknown_corrspd_v1/attack_eval/run_attack_eval.py`
- GPU：`NVIDIA GeForce RTX 3080 Laptop GPU`
- PyTorch：`2.7.1+cu118`
- 训练样本总数：11304；外层已知攻击样本总数：10687；伪未知样本总数：617。
- 每个外层折重新拟合 `TaskSpecificModel(alpha=100)`；阈值在该折冻结为伪未知样本 `max_logit` 的 95% 分位数，未用攻击结果调参。
- 预测摘要：`prediction_summary.csv`；完整数值：`attack_metrics.json`。

## 严格性限制

1. `src/open_set_improvement/core.py::attack_batch` 的实现固定对模型 **fused** 分数求梯度，并同时扰动 `x` 与 two-window 特征；本实验沿用该实现，因此不是对 raw EEG 波形、滤波器输出或时域采样的官方攻击。
2. 对 logcov，攻击输入是原始 log-cov 特征；对 corrSPD，攻击输入是“恢复 SPD → 相关矩阵归一化 → lambda=1e-3 shrinkage → log-vectorize”后的 corrSPD 特征。corrSPD 的攻击在该变换后的特征坐标中是严格可执行的，但没有通过 corrSPD 的非线性映射反传到原始 EEG，因此不能冒充 raw EEG 官方攻击。
3. 现有攻击实现不能攻击本候选部署脚本中的 `full_none` 分数，而是攻击 `fused` 分数；故结果应标为“fused 特征空间 PGD 内部代理”，不是官方攻击协议复现。
4. U 使用留身份伪未知样本，真实 Test 没有标签；Test 的 C、U、AUC、F1、RA、ASafeR 与官方分均保持 `null`，没有补值。
5. `prediction_summary.csv` 只保存外层已知样本 clean/adv 预测及伪未知 clean 预测摘要，不含原始 EEG 或敏感数据。
