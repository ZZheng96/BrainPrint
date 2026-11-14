# E73c native22 冻结表征 + 通用样本—模板匹配头

## 协议

- 仅改变匹配头；E73b native22 SupCon 编码器、三外层折和 run 隔离全部冻结。
- 匹配头仅使用 enrolled 的 run0–2 跨 Session same/different 对；不使用两类 unknown 训练或调参。
- 所有折的匹配头及 run4 已知样本 q05/q02 阈值先落盘，再一次性评估 run5 外层已知/未知样本。

## 主结果（3 外层折 × 2 方向的均值）

| 指标 | 数值 |
|---|---:|
| Pair AUC | 0.990170 |
| Detection AUC | 0.924310 |
| Detection EER | 0.133102 |
| C | 0.933160 |
| Identity macro-F1 | 0.963516 |
| Unknown-F1 | 0.773650 |
| U@95 | 0.678241 |
| 冻结 q02 的已知接受率 | 0.972222 |
| 冻结 q02 的未知拒识率 | 0.549769 |
| DIR@FPIR1% | 0.457465 |

## 阶段门禁

- 通过：`pair_AUROC`
- 未通过：`detection_AUROC`
- 未通过：`detection_EER`
- 未通过：`working_98_known_acceptance`
- 未通过：`working_98_unknown_rejection`

**总体结论：未通过 E73 阶段出口。**

Official Score=N/A；Test AUC/F1=N/A（未访问竞赛 Train/Test）。

模型、逐折数值与训练历史见 models/ 和 results.json。
