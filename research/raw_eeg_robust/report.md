# raw EEG Robust 实验报告

## 结论

固定五折、嵌套阈值、三种攻击的验收已完成。候选 `adversarial_ridge` **未通过冻结验收门槛**，因此不进入 adaptive attack，也不替换正式提交。

主要原因是：

- `untargeted_snr20` 的 RA=0.4477，未达到 0.60；
- unknown-F1 相比 direct 下降 0.0126，超过允许的 0.01；
- unknown AUROC 相比 direct 下降 0.0077，超过允许的 0.005；
- `targeted_snr20` 的 ASafeR=0.7761，未达到 0.90。

## 实验协议

- 数据：Train Session1；按 subject 固定五折外层留出。
- 阈值：每个外层折内独立进行 inner holdout 选择。
- 候选：clean 全量训练 + 当前拟合折内 raw-PGD 增强，仅重拟合 Ridge 的 full/window/low 组件。
- 训练攻击：SNR=20 dB、untargeted、10 步、每 subject/task 取 8 条。
- 验收攻击：untargeted SNR20、untargeted SNR30、targeted SNR20，均为 10 步。
- 对抗样本特征：重新使用既有 SciPy 原始特征提取器计算，未使用测试标签或 outer 信息。

## 五折汇总指标

| 攻击 | 方法 | C | RA | ASafeR | A | U | 官方分数 | unknown AUROC（AUC） | unknown F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted SNR20 | direct | 0.9228 | 0.0000 | 0.4374 | 0.1312 | 0.9567 | 47.00 | 0.9803 | 0.9472 |
| untargeted SNR20 | adversarial_ridge | 0.9089 | 0.4477 | 0.9687 | 0.6040 | 0.9425 | 79.47 | 0.9726 | 0.9346 |
| untargeted SNR30 | direct | 0.9228 | 0.0103 | 0.7868 | 0.2433 | 0.9567 | 58.38 | 0.9803 | 0.9472 |
| untargeted SNR30 | adversarial_ridge | 0.9089 | 0.9222 | 0.9999 | 0.9455 | 0.9425 | 92.99 | 0.9726 | 0.9346 |
| targeted SNR20 | direct | 0.9228 | 0.0000 | 0.0051 | 0.0015 | 0.9567 | 9.29 | 0.9803 | 0.9472 |
| targeted SNR20 | adversarial_ridge | 0.9089 | 0.2632 | 0.7761 | 0.4171 | 0.9425 | 69.82 | 0.9726 | 0.9346 |

说明：AUC 指 unknown AUROC；F1 同时报告 unknown F1。clean macro-F1（含 reject）为 direct 0.8988、adversarial_ridge 0.8901；对抗 macro-F1（含 reject）在 untargeted SNR20 下分别为 0.0000 与 0.5467。

## 固定验收结果

- 官方分数增益：+32.4744，达标。
- A 增益：+0.4728，达标。
- ASafeR（untargeted SNR20）：0.9687，达标。
- C、U：均满足绝对值与下降限制，达标。
- 五折中候选不差于 direct：5/5，达标；最差折分数差：+30.3376，达标。
- 未通过：RA 绝对门槛、unknown F1 下降限制、unknown AUROC 下降限制、targeted SNR20 ASafeR 门槛。

## 产物与后续

- 完整结果：`research/raw_eeg_robust/results.json`
- 配对候选门禁：`research/raw_eeg_robust/pairing_feasibility.json`；结构覆盖上界 0.4292，低于 0.65，已在 outer 前停止。
- 本轮候选保留为研究结果，不替换正式提交；后续若继续，应优先改善 targeted SNR20 与 clean/unknown 保真度，而不是直接扩大训练增强规模。
