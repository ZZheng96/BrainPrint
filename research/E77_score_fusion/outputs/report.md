# E77 EEG-only score fusion

冻结 E74 encoder，复用 E76 Mahalanobis head；未使用 E75，未访问 competition Test。融合权重预注册为 cosine 0.5 / Mahalanobis 0.5；阈值仅由 inner known run4 的 q05/q02 决定，calibration unknown 仅诊断。

| 指标 | 三折两方向 q05 均值 |
|---|---:|
| AUROC | 0.968232 |
| Identity macro-F1 | 0.950586 |
| Unknown-F1 | 0.874473 |
| S1 known acceptance | 0.911458 (门禁 失败) |
| S2 unknown false acceptance | 0.122685 (门禁 失败) |

q02、逐折结果和 calibration unknown 诊断见 results.json。若门禁失败，结论为未达成，不扫阈值、不伪造。
