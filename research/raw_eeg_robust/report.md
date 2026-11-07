# raw EEG Robust 实验报告：混合增强 + clean:adv=2:1

## 结论

固定五折、嵌套阈值、三种攻击验收已完成。本轮在上一轮 targeted 混合增强基础上，将 Ridge 拟合中的 clean:adversarial 有效权重调整为 `2:1`。未知类指标有所恢复，但攻击恢复能力明显下降，整体**未通过冻结验收**，不进入 adaptive attack，也不替换正式提交。

## 与上一轮的变化

- unknown F1：`0.9333 → 0.9362`，向 direct 的 `0.9472` 回升；
- unknown AUROC（AUC）：`0.9718 → 0.9738`，向 direct 的 `0.9803` 回升；
- untargeted SNR20 RA：`0.4565 → 0.2743`，明显恶化；
- targeted SNR20 ASafeR：`0.9189 → 0.8195`，重新低于 `0.90` 门槛；
- untargeted SNR20 官方分数：`79.64 → 73.06`。

结论是：简单降低对抗样本权重能够缓解未知类分布偏移，但代价是鲁棒恢复能力下降，不能作为当前最终方案。

## 实验协议

- 数据：Train Session1；按 subject 固定五折外层留出。
- 阈值：每个外层折内独立进行 inner holdout 选择。
- 候选：clean 全量训练 + 当前拟合折内 raw-PGD 增强，仅重拟合 Ridge 的 full/window/low 组件。
- 训练增强：每个 subject/task 总计 8 条，其中 6 条 untargeted SNR20、2 条 targeted SNR20。
- Ridge 权重：clean:adversarial 有效权重为 `2:1`，通过重复 clean 训练行实现。
- 验收攻击：untargeted SNR20、untargeted SNR30、targeted SNR20，均为 10 步。
- 对抗样本特征：重新使用既有 SciPy 原始特征提取器计算，未使用测试标签或 outer 信息。

## 五折汇总指标

| 攻击 | 方法 | C | RA | ASafeR | A | U | 官方分数 | unknown AUROC（AUC） | unknown F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted SNR20 | direct | 0.9228 | 0.0000 | 0.4374 | 0.1312 | 0.9567 | 47.00 | 0.9803 | 0.9472 |
| untargeted SNR20 | adversarial_ridge | 0.9099 | 0.2743 | 0.9391 | 0.4738 | 0.9449 | 73.06 | 0.9738 | 0.9362 |
| untargeted SNR30 | direct | 0.9228 | 0.0103 | 0.7868 | 0.2433 | 0.9567 | 58.38 | 0.9803 | 0.9472 |
| untargeted SNR30 | adversarial_ridge | 0.9099 | 0.8237 | 0.9997 | 0.8765 | 0.9449 | 90.65 | 0.9738 | 0.9362 |
| targeted SNR20 | direct | 0.9228 | 0.0000 | 0.0051 | 0.0015 | 0.9567 | 9.29 | 0.9803 | 0.9472 |
| targeted SNR20 | adversarial_ridge | 0.9099 | 0.3395 | 0.8195 | 0.4835 | 0.9449 | 73.61 | 0.9738 | 0.9362 |

说明：AUC 指 unknown AUROC；F1 报告 unknown F1。候选相对 direct 的五折官方分数差为 `[+27.8795, +28.5806, +25.8910, +23.4982, +24.4599]`，5/5 折不差于 direct。

## 冻结验收结果

- 官方分数增益：`+26.0618`，达标。
- A 增益：`+0.3425`，达标。
- C、U：满足绝对值与下降限制，达标。
- 五折不差于 direct：`5/5`，达标；最差折提升 `+23.4982`，达标。
- 未通过：untargeted SNR20 RA=`0.2743`；targeted SNR20 ASafeR=`0.8195`；unknown F1 下降约 `0.0110`；unknown AUC 下降约 `0.0064`。

## 产物与后续

- 完整结果：[results.json](file:///d:/WORK/BCIC2026/BrainPrint/research/raw_eeg_robust/results.json)
- 配对候选门禁：[pairing_feasibility.json](file:///d:/WORK/BCIC2026/BrainPrint/research/raw_eeg_robust/pairing_feasibility.json)；结构覆盖上界 `0.4292`，低于 `0.65`，已在 outer 前停止。
- 本轮候选保留为研究结果，不替换正式提交。简单全局降权不可行；后续应改为仅对 hard untargeted 样本增强，或按样本风险自适应加权，以同时保留 RA 与 unknown 指标。
