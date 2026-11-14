# E82 严格 nested 留被试 pseudo-unknown 标定

- 冻结 E74 encoder，复用 E79 pair score；拟合仅 enrolled run0–3，inner known 为 run4。
- calibration_unknown 为 roles calibration_unknown 的 run5，仅用于固定 FPIR=1%/2% 分位阈值；outer unknown run5 仅最终一次评估。

| target FPIR | AUROC | F1 | known acceptance | unknown false acceptance | DIR@FPIR1 |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.918523 | 0.769978 | 0.693576 | 0.093750 | 0.470486 |
| 0.02 | 0.918523 | 0.801187 | 0.743056 | 0.115741 | 0.470486 |

Test AUC/F1=N/A；未访问竞赛 Test。
