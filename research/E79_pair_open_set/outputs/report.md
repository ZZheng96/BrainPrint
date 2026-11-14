# E79 冻结 E74 表征 + 通用样本—模板匹配头

## 协议

- 仅改变匹配头；E74 multiscale native22 encoder、三外层折和 run 隔离全部冻结。
- 匹配头仅使用 enrolled 的 run0–2 跨 Session same/different 对；不使用两类 unknown 训练或调参。
- 所有折的匹配头及 run4 已知样本 q05/q02 阈值先落盘，再一次性评估 run5 外层已知/未知样本。

## 主结果（3 外层折 × 2 方向的均值）

| 指标 | 数值 |
|---|---:|
| Pair AUC | 0.989778 |
| Detection AUC | 0.918523 |
| Detection EER | 0.138021 |
| C | 0.921007 |
| Identity macro-F1 | 0.956474 |
| Unknown-F1 | 0.829146 |
| Known acceptance | 0.921007 |
| Unknown false acceptance/rejection | 0.202546 / 0.797454 |
| U@95 | 0.747685 |
| 冻结 q02 的已知接受率 | 0.955729 |
| 冻结 q02 的未知拒识率 | 0.724537 |
| DIR@FPIR1% | 0.468750 |
| q02 working known acceptance | 0.955729 |
| q02 working unknown rejection | 0.724537 |

## 阶段门禁

- 未通过：`S1_known_acceptance_ge_95`
- 未通过：`S2_unknown_false_acceptance_le_02`

**总体结论：未通过 E79 阶段出口。**

Official Score=N/A；Test AUC/F1=N/A（未访问竞赛 Train/Test）。

模型、逐折数值与训练历史见 models/ 和 results.json。
