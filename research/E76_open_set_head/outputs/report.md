# E76 结构性 unknown-aware/open-set 评估

- 冻结 E74 encoder；头部为 enrolled run0–2 协方差 + run3 prototype Mahalanobis 距离。
- run4 仅用于 known calibration；calibration unknown 仅诊断，不参与阈值/模型选择。run5 为 outer evaluation。

| 指标 | 数值 |
|---|---:|
| Detection AUC | 0.976364 |
| Identity macro-F1 | 0.952444 |
| Unknown-F1 | 0.891897 |
| Test AUC | N/A |
| Test F1 | N/A |

E74 encoder 未重复训练；未访问 Test。
