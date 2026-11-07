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

## 官方分数情景换算（修订版：不作为官方预测）

第一次提交提供了唯一真实锚点：官方 `69.17`。本地 direct 在 untargeted SNR20 协议下的公式分为 `46.9996`，但它不是第一次正式提交在官方 Test 上的完整复算，因此不能直接拟合一个“官方校准系数”。

上一版把 `A=0.7*RA+0.3*ASafeR` 乘以 `3.00792`，会让 robust 候选的 A 截断到 1，所有候选被推到约 94–95 分，已经失去排序意义。本版改为只迁移本地相对增益的一部分：

```text
S_base = 69.17
R_i = S_local_i / S_local_direct
S_scenario(i, lambda) = S_base * R_i^lambda
lambda ∈ {0.15, 0.25, 0.35}
```

其中 `lambda=0.25` 是中性情景，表示本地人工攻击分数相对提升只有约四分之一能够迁移到官方 Test；`0.15–0.35` 仅用于敏感性范围。这个方法不修改官方公式，也不声称识别了真实的 C、A、U；它只是把不同候选的本地相对排序压缩到更合理的区间，并避免饱和。

| 实验版本 | 攻击 | 本地公式分 | λ=0.15 | λ=0.25（中性） | λ=0.35 |
|---|---|---:|---:|---:|---:|
| direct | untargeted SNR20 | 47.00 | 69.17 | 69.17 | 69.17 |
| raw robust 首轮 1:1 | untargeted SNR20 | 79.47 | 74.84 | 78.88 | 83.13 |
| raw robust 混合 targeted 1:1 | untargeted SNR20 | 79.64 | 74.86 | 78.97 | 83.24 |
| raw robust 混合 targeted 1:1 | targeted SNR20 | 80.82 | 75.03 | 79.21 | 83.62 |
| raw robust 2:1 | untargeted SNR20 | 73.06 | 73.90 | 77.24 | 80.72 |
| raw robust 2:1 | targeted SNR20 | 73.61 | 73.98 | 77.38 | 80.93 |

更合理的结论是：在该情景模型下，当前 raw robust 1:1 候选的中性估计约 `79` 分，敏感性范围约 `75–83`；2:1 候选约 `77` 分，范围约 `74–81`。这不是官方成绩置信区间，也不能证明第二次 Test 候选一定高于 `69.17`。候选筛选仍必须同时约束 `C`、`U`、RA、ASafeR、unknown AUC/F1 和五折稳定性。

## 产物与后续

- 完整结果：[results.json](file:///d:/WORK/BCIC2026/BrainPrint/research/raw_eeg_robust/results.json)
- 配对候选门禁：[pairing_feasibility.json](file:///d:/WORK/BCIC2026/BrainPrint/research/raw_eeg_robust/pairing_feasibility.json)；结构覆盖上界 `0.4292`，低于 `0.65`，已在 outer 前停止。
- 本轮候选保留为研究结果，不替换正式提交。简单全局降权不可行；后续应改为仅对 hard untargeted 样本增强，或按样本风险自适应加权，以同时保留 RA 与 unknown 指标。
