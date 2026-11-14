# E106 LOSO-selected mean fusion frozen audit

E105 留一身份对比中，SPD distance ratio 与 E95 raw distance ratio 的均值融合在 17/20 折有可行点、平均 unknown rejection 0.945，为最稳健候选。本实验冻结该融合并做一次项目 Test 审计。

## Train-only

AUC=0.941550，F1=0.918431；阈值=0.047668，known acceptance=0.900591，unknown rejection=0.813636，目标达成=True。

## 项目 Test

真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。

| 项目 | 结果 | 目标 | 是否达标 |
|---|---:|---:|---|
| S1 拒识率 | 0.534454 | ≥0.80 | False |
| S2 接受率 | 0.517640 | ≥0.85 | False |

predictions.csv 完整落盘后才读取 session 审计。
