# E102 task-conditioned distance-ratio gate

任务条件化 correlation-logSPD 原型 + distance ratio（次近/最近身份距离）。原型与阈值仅由 Train/session1 确定，Test 在冻结后一次性读取。

## Train-only

AUC=0.916003，F1=0.903332；阈值=1.058701，known acceptance=0.850081，unknown rejection=0.843182，目标达成=True。

## 项目 Test

真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。

| 项目 | 结果 | 目标 | 是否达标 |
|---|---:|---:|---|
| S1 拒识率 | 0.842730 | ≥0.80 | True |
| S2 接受率 | 0.287335 | ≥0.85 | False |

predictions.csv 完整落盘后才读取 session 审计。
