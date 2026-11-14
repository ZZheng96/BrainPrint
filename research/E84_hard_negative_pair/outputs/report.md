# E84 冻结 E74 表征 + hard-negative pair head

## 协议

- 仅改变匹配头；E74 multiscale native22 encoder、三外层折和 run 隔离全部冻结。
- 匹配头仅使用 enrolled 的 run0–2 跨 Session same/different 对；每个 query 以最高 wrong-identity score 做 hard-negative mining；不使用两类 unknown 训练或调参。
- 所有折的匹配头及 run4 已知样本 q05/q02 阈值先落盘，再一次性评估 run5 外层已知/未知样本。

## 主结果（3 外层折 × 2 方向的均值）

| 指标 | 数值 |
|---|---:|
| Pair AUC | 0.991614 |
| Detection AUC | 0.933202 |
| Detection EER | 0.135706 |
| C | 0.919271 |
| Identity macro-F1 | 0.955317 |
| Unknown-F1 | 0.833643 |
| Known acceptance | 0.919271 |
| Unknown false acceptance/rejection | 0.194444 / 0.805556 |
| U@95 | 0.753472 |
| 冻结 q02 的已知接受率 | 0.955729 |
| 冻结 q02 的未知拒识率 | 0.730324 |
| DIR@FPIR1% | 0.540799 |
| q02 working known acceptance | 0.955729 |
| q02 working unknown rejection | 0.730324 |

## 相对 E79

- E79 基线：Pair AUC 0.989778，Detection AUC 0.918523，Detection EER 0.138021，known acceptance 0.921007，unknown false acceptance 0.202546。
- E84 变化：Pair AUC +0.001836，Detection AUC +0.014679，Detection EER -0.002315，known acceptance -0.001736，unknown false acceptance -0.008102。
- 结论：Pair/Detection AUC 与未知误受率改善，但 known acceptance 略降；S1/S2 代理门禁仍均未通过，不能宣称解决 E79 问题。

## 阶段门禁

- 未通过：`S1_known_acceptance_ge_95`
- 未通过：`S2_unknown_false_acceptance_le_02`

**总体结论：未通过 E84 阶段出口。**

Official Score=N/A；Test AUC/F1=N/A（未访问竞赛 Train/Test）。

模型、逐折数值与训练历史见 models/ 和 results.json。
