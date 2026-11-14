# E72–E86 外部实验总结与 Test 指标问题

日期：2026-11-14
范围：本对话推进的 E72–E86，以及此前被引用的 E73 外部基线。

## 1. 结论摘要

1. 本批实验**没有优化项目 Test 指标**。E74–E84 全部在外部 BCIC IV-2a（22 通道、9 名被试）上评估，与项目 Test（18 通道、20 注册者 + 5 名未注册者）不兼容；本批没有产出任何项目 Test 候选。
2. 外部代理指标确有提升：detection AUC 约 0.76 → 0.976，EER 0.144 → 0.069，unknown-F1 0.779 → 0.902。
3. 但外部 known/unknown 置信分布重叠未消除，**没有任何一组同时满足 known acceptance ≥95% 与 unknown FAR ≤2%**；E83 对已落盘 operating point 的可行性审计为 0。
4. 项目 Test 现有三份候选（E64/E42，非本批）S1 拒识率仅 58–59%（要求 ≥95%），S2 拒识率 70–85%（要求 ≤2%），全部未通过。
5. 存在**指标语义错配**：E77/E78/E79/E81/E84 把外部 `known_acceptance` 命名为 S1、`unknown_false_acceptance` 命名为 S2。数值未算错，但命名与项目定义在对象和方向上都不对应。
6. 另有口径不一致：E82 的 `unknown_F1` 正类取反（`pos_label=1`），E75 的 `C` 为不含拒识的 argmax 准确率。

## 2. 各实验尝试与结果

| 实验 | 尝试 | 关键结果（外部 outer test） | 判定 |
|---|---|---|---|
| E72 | Train-only nested 样本—模板配对协议，5 outer × 2 inner | 数据隔离通过，无外部/项目 Test 预测 | 协议基线 |
| E73 | 接入 BCIC IV-2a 双 Session，native22 与 native22_pair | AUC 0.9141 / 0.9243 | 基线 |
| E74 | 多尺度 native22 编码器 + cross-session SupCon | AUC 0.9226、unknown-F1 0.8185、EER 0.1444 | 检测大幅提升 |
| E75 | 图像化 topomap + 冻结 DINOv2 | AUC 0.4640、unknown-FAR 0.9421 | 失败，排除 |
| E76 | Mahalanobis open-set head | AUC 0.9764、unknown-F1 0.8919、EER 0.0692、DIR@FPIR1 0.7674 | 综合最强 |
| E77 | cosine 0.5 + Mahalanobis 0.5 固定融合 | AUC 0.9682、FAR 0.1227 | 无增益 |
| E78 | 逐身份阈值 q05 | AUC 0.9764、FAR 0.0729、known acc 0.9036 | FAR 最低，但接受率受损 |
| E79 | sample–template pair head | q02 known acc 0.9557、FAR 0.2755、id-F1 0.9764 | 检测回落，校准失败 |
| E80 | robust classwise Mahalanobis 与 pair 的 min 融合 | id-F1 0.9814、known acc 0.9644、FAR 0.2569 | 干净指标最好 |
| E81 | EVT / 尾部分位校准 | FAR 0.2060、known acc 0.9401 | 未突破 |
| E82 | 留被试 pseudo-unknown 标定 FPIR 阈值 | FPIR1%：FAR 0.0938、known acc 0.6936 | 过严 |
| E83 | 门禁可行性只读审计 | 同时满足门禁的可行率 0 | 证明瓶颈 |
| E84 | hard-negative pair head | DIR@FPIR1 0.5408、AUC 0.9332、FAR 0.1944 | 局部改善 |
| E85 | 项目 Test CSV 统一审计（E64/E42 候选） | S1 58–59%、S2 70–85% 拒识 | 未通过 |
| E86 | 补 E73–E84 outer test 逐样本 CSV 与比对 | 11 个实验全部补齐，误差 ≤1.1e-16 | 可审计 |

### 2.1 外部 outer test 主表

| 实验 / 方法 | AUC | id-F1 | unknown-F1 | known acc | unknown FAR | EER |
|---|---:|---:|---:|---:|---:|---:|
| E73 native22_pair | 0.9243 | 0.9635 | 0.7736 | 0.9349 | 0.2870 | 0.1331 |
| E74 wccn:margin | 0.9226 | 0.9584 | 0.8185 | 0.9253 | 0.2315 | 0.1444 |
| E75 image | 0.4640 | 0.4993 | 0.1017 | 0.9514 | 0.9421 | 0.5382 |
| E76 mahalanobis | 0.9764 | 0.9524 | 0.8919 | 0.9175 | 0.1042 | 0.0692 |
| E77 fusion_q05 | 0.9682 | 0.9506 | 0.8745 | 0.9115 | 0.1227 | 0.0842 |
| E78 per_identity_q05 | 0.9764 | 0.9451 | 0.9018 | 0.9036 | 0.0729 | 0.0692 |
| E79 pair_q05 | 0.9185 | 0.9565 | 0.8291 | 0.9210 | 0.2025 | 0.1380 |
| E80 min_fusion_q02 | 0.9336 | 0.9814 | 0.8149 | 0.9644 | 0.2569 | 0.1244 |
| E81 evt_q02 | 0.9336 | 0.9671 | 0.8347 | 0.9401 | 0.2060 | 0.1244 |
| E82 fpir01 | 0.9185 | 0.7909 | 0.7942 | 0.6936 | 0.0938 | 0.1380 |
| E84 pair_q05 | 0.9332 | 0.9553 | 0.8336 | 0.9193 | 0.1944 | 0.1357 |

## 3. 对项目 Test 指标的影响：没有做到

### 3.1 指标对象错位

- 本批全部指标来自外部 BCIC IV-2a 的 run5 outer query，不是项目 Test。
- 外部模型输入 22 通道、输出 4 名 enrolled 身份；项目 Test 为 18 通道、20 名注册者 + 5 名未注册者。
- 因此 E74–E84 的 checkpoint 不能生成合规的项目 Test CSV。E85 的项目 Test 候选仍来自更早的 E64/E42。

### 3.2 方向性错配

项目 Test 现状：S2（注册者 Session2）已被拒识 70–85%，`S2 ≤2%` 的门槛要求几乎全部接受。本批外部实验的主攻方向却是"提高 known acceptance"，并且 known acceptance 越高，unknown FAR 往往也越高（E80：0.9644 / 0.2569）。把这一方向直接迁移到项目 Test，只会让 S2 更差。

### 3.3 可行性已被审计否定

E83 只读审计 E76/E79/E82 已落盘的 operating point：没有任何一个点同时满足 known acceptance ≥95% 与 unknown FAR ≤2%，可行率为 0。E82 用 pseudo-unknown 选阈值后 FAR 能到 0.0938，但 known acceptance 跌到 0.6936，说明这是分布重叠问题，不是阈值位置问题。

## 4. 指标问题清单

### 4.1 S1/S2 命名错配（主要问题）

涉及：E77、E78、E79、E81、E84 的 `config.json`、`results.json`、`report.md`。

| 外部实验字段 | 实际含义 | 项目真实定义 |
|---|---|---|
| `S1_known_acceptance_ge_95` | 已注册者**接受**率 ≥95% | S1 = Session1 **拒识**率 ≥95% |
| `S2_unknown_false_acceptance_le_2` | 未注册者**误接受**率 ≤2% | S2 = Session2 **拒识**率 ≤2% |

对象不同（known subject vs Session1）、方向不同（接受 vs 拒识）。数值本身正确，错在命名和门禁映射，容易让读者误以为项目 Test 已在改善。

### 4.2 F1 正类方向相反

E82 原脚本 `unknown_F1` 使用 `pos_label=1`（即 known 接受类的 F1），E86 统一 schema 使用 `pos_label=0`。同名不同义：0.8012 vs 0.8021。

### 4.3 C 口径不同

E75 的 `C` 是 argmax 直接准确率（不含拒识），E86 统一 schema 的 C 含拒识。E75 被拒样本的 argmax 身份无法从 CSV 恢复，故两者不可逐位比较。

### 4.4 覆盖不完整但已说明

- E72：Train-only 协议，无外部 test，无项目 Test。
- E83：只读可行性审计，不产生预测。
- E75：仅有 `T_to_E` 一个方向，3 组，不是完整 3 折 × 2 方向。

### 4.5 数值可信性

E86 从落盘 CSV 重读并重算，与各实验 `results.json` 逐项对比，最大绝对误差 ≤1.1e-16；逐样本标签、阈值、拒识标记均一致。上述问题属于命名与口径，不是计算错误。

## 5. 后续要求

1. 任何实验的最终评价只看项目 Test 的 S1 拒识率 ≥95% 与 S2 拒识率 ≤2%；外部 pseudo-unknown 指标只能作为代理，必须标注"外部代理，非项目 Test"。
2. 禁止把外部 `known_acceptance` / `unknown_false_acceptance` 命名为 S1/S2。
3. 每批实验必须明确回答"对项目 Test 指标做了什么"；没有生成项目 Test CSV 的实验，不得声称优化了 Test 指标。
4. 下一步应优先解决 Test S2 的过度拒识（当前 70–85%），而不是继续抬高外部 known acceptance。
5. 统一指标口径：`unknown-F1` 固定 `pos_label=0`；`C` 必须含拒识；S1/S2 仅用于 CSV 写出后的 Session 分组审计。

## 6. 产物索引

- 逐样本 CSV 与指标：各实验 `outputs/outer_test_predictions.csv`、`outputs/outer_test_metrics.json`
- 汇总：[all_experiments_metrics.json](file:///D:/WORK/BCIC2026/BrainPrint/research/E86_external_outer_csv/outputs/all_experiments_metrics.json)
- 跨实验比对：[cross_experiment_comparison.json](file:///D:/WORK/BCIC2026/BrainPrint/research/E86_external_outer_csv/outputs/cross_experiment_comparison.json)
- 技术报告：[E86 report.md](file:///D:/WORK/BCIC2026/BrainPrint/research/E86_external_outer_csv/report.md)
- 项目 Test 审计：[E85 results.json](file:///D:/WORK/BCIC2026/BrainPrint/research/E85_test_csv_audit/outputs/results.json)
