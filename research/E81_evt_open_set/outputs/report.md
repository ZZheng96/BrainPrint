# E81 预注册 EVT/尾部校准开放集

## 协议

- 冻结 E74 encoder；pair head 复用 E79，表示拟合和 pair head 训练只使用 enrolled run0–2；run3 建模板。
- run4 enrolled known 按身份拟合固定 q02 下尾分位数；calibration unknown 仅诊断，run5 outer 只评估，不扫描阈值。

## 主结果（3 折 × 2 方向均值）

| 指标 | 数值 |
|---|---:|
| Pair AUROC | 0.991658 |
| Detection AUROC | 0.933642 |
| Identity macro-F1 | 0.967070 |
| Unknown-F1 | 0.834686 |
| q02 known acceptance | 0.940104 |
| q02 unknown false acceptance | 0.206019 |
| q02 unknown rejection | 0.793981 |
| DIR@FPIR1% | 0.494792 |

## S1/S2 代理门禁

- S1 known acceptance ≥ 0.95：未通过
- S2 unknown false acceptance ≤ 0.02：未通过

无竞赛 Train/Test 访问；Official Score、Test AUC/F1=N/A。
