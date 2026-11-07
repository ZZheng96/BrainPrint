# raw EEG Robust 实验报告：targeted 混合增强

## 结论

固定五折、嵌套阈值、三种攻击验收已完成。`adversarial_ridge` 相比上一轮纯 untargeted 训练进一步改善，尤其 targeted SNR20 已通过对应门槛；但整体仍**未通过冻结验收**，不进入 adaptive attack，也不替换正式提交。

剩余未通过项：

- untargeted SNR20 的 RA=`0.4565`，低于 `0.60`；
- unknown-F1 从 `0.9472` 降至 `0.9333`，下降约 `0.0139`，超过允许的 `0.01`；
- unknown AUROC（AUC）从 `0.9803` 降至 `0.9718`，下降约 `0.0085`，超过允许的 `0.005`。

## 相比上一轮的变化

- targeted SNR20 ASafeR：`0.7761 → 0.9189`，从未达标变为达标；
- untargeted SNR20 官方分数：`79.47 → 79.64`；
- untargeted SNR20 A：`0.6040 → 0.6096`；
- RA：`0.4477 → 0.4565`，仍未达到 `0.60`；
- 未知类保真度略有进一步下降，说明 targeted 样本覆盖有效但带来了额外分布偏移。

## 实验协议

- 数据：Train Session1；按 subject 固定五折外层留出。
- 阈值：每个外层折内独立进行 inner holdout 选择。
- 候选：clean 全量训练 + 当前拟合折内 raw-PGD 增强，仅重拟合 Ridge 的 full/window/low 组件。
- 训练增强：每个 subject/task 总计 8 条，其中 6 条 untargeted SNR20、2 条 targeted SNR20；总增强量与上一轮相同。
- 验收攻击：untargeted SNR20、untargeted SNR30、targeted SNR20，均为 10 步。
- 对抗样本特征：重新使用既有 SciPy 原始特征提取器计算，未使用测试标签或 outer 信息。

## 五折汇总指标

| 攻击 | 方法 | C | RA | ASafeR | A | U | 官方分数 | unknown AUROC（AUC） | unknown F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted SNR20 | direct | 0.9228 | 0.0000 | 0.4374 | 0.1312 | 0.9567 | 47.00 | 0.9803 | 0.9472 |
| untargeted SNR20 | adversarial_ridge | 0.9071 | 0.4565 | 0.9671 | 0.6096 | 0.9413 | 79.64 | 0.9718 | 0.9333 |
| untargeted SNR30 | direct | 0.9228 | 0.0103 | 0.7868 | 0.2433 | 0.9567 | 58.38 | 0.9803 | 0.9472 |
| untargeted SNR30 | adversarial_ridge | 0.9071 | 0.9277 | 0.9999 | 0.9494 | 0.9413 | 93.02 | 0.9718 | 0.9333 |
| targeted SNR20 | direct | 0.9228 | 0.0000 | 0.0051 | 0.0015 | 0.9567 | 9.29 | 0.9803 | 0.9472 |
| targeted SNR20 | adversarial_ridge | 0.9071 | 0.5140 | 0.9189 | 0.6355 | 0.9413 | 80.82 | 0.9718 | 0.9333 |

说明：AUC 指 unknown AUROC；F1 报告 unknown F1。候选相对 direct 的五折官方分数差为 `[+34.3840, +34.8874, +32.9308, +30.3895, +30.6064]`，5/5 折不差于 direct。

## 冻结验收结果

- 官方分数增益：`+32.6396`，达标。
- A 增益：`+0.4784`，达标。
- untargeted SNR20 ASafeR：`0.9671`，达标。
- C、U：满足绝对值与下降限制，达标。
- 五折不差于 direct：`5/5`，达标；最差折仍提升 `+30.3895`，达标。
- untargeted SNR30 ASafeR：`0.9999`，达标。
- targeted SNR20 ASafeR：`0.9189`，达标。
- 未通过：untargeted SNR20 RA、unknown F1 下降限制、unknown AUROC 下降限制。

## 产物与后续

- 完整结果：[results.json](file:///d:/WORK/BCIC2026/BrainPrint/research/raw_eeg_robust/results.json)
- 配对候选门禁：[pairing_feasibility.json](file:///d:/WORK/BCIC2026/BrainPrint/research/raw_eeg_robust/pairing_feasibility.json)；结构覆盖上界 `0.4292`，低于 `0.65`，已在 outer 前停止。
- 本轮候选保留为研究结果，不替换正式提交。下一步若继续，应优先降低 unknown 分布偏移并提高 untargeted SNR20 的 RA，而不是继续增加 targeted 比例。
