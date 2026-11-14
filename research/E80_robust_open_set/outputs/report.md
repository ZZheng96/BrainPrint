# E80 robust open-set score

## 预注册协议

- 冻结 E74 encoder；pair head 与 E79 固定。classwise Mahalanobis 仅用 enrolled run0–2 拟合，run3 建模板。
- 每类距离用训练集 robust median/MAD 标准化；最终 confidence 为 pair sigmoid 与 Mahalanobis sigmoid 的固定 min。
- q02 仅由 enrolled run4 known confidence 标定；calibration unknown 仅诊断；run5 outer 只评估一次。

## 主结果（3 折 × 2 方向均值）

| 指标 | 数值 |
|---|---:|
| Detection AUROC | 0.933642 |
| Identity macro-F1 | 0.981365 |
| Unknown-F1 | 0.814862 |
| q02 known acceptance | 0.964410 |
| q02 unknown false acceptance | 0.256944 |
| DIR@FPIR1% | 0.494792 |
| Pair AUROC | 0.991658 |

结果未设置通过门槛；E80 仅报告预注册指标。

Official Score=N/A；Test AUC/F1=N/A（未访问竞赛 Train/Test）。
