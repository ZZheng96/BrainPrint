# E104 session-invariant pretraining + 18-channel transfer

外部 BCIC IV-2a 双 Session 做跨 Session SupCon 预训练（22→18 IDW 投影），冻结编码器后在项目 Train/session1 上建任务条件化原型，用 distance ratio 打分；阈值由 Train 已知块 + 外部未知确定，Test 冻结后一次性读取。

## Train + 外部

AUC=0.832017，F1=0.314600；阈值=1.070292，known acceptance=0.657106，external unknown rejection=0.898148，目标达成=False。

## 项目 Test

真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。

| 项目 | 结果 | 目标 | 是否达标 |
|---|---:|---:|---|
| S1 拒识率 | 0.426311 | ≥0.80 | False |
| S2 接受率 | 0.614996 | ≥0.85 | False |

predictions.csv 完整落盘后才读取 session 审计。
