# E78 per-identity calibration

冻结 E74 encoder，复用 E76 Mahalanobis score。每折/方向仅用 enrolled run0–2 训练，run3 建 prototype；每个身份阈值仅由 enrolled run4 known 的该身份 score 计算 q05，q02 作为预注册报告；calibration unknown 不参与阈值。run5 为 outer 评估。

| 指标 | q05 三折两方向均值 | q02 三折两方向均值 |
|---|---:|---:|
| AUROC | 0.976364 | 0.976364 |
| Identity macro-F1 | 0.945056 | 0.970367 |
| Unknown-F1 | 0.901845 | 0.873091 |
| S1 known acceptance | 0.903646 | 0.946181 |
| S2 unknown false acceptance | 0.072917 | 0.166667 |

S1 门禁：失败；S2 门禁：失败。失败如实记录，未扫阈值。Test AUC/F1=N/A，未访问 Test。逐折阈值和结果见 results.json，head checkpoint 见 outputs/models。
