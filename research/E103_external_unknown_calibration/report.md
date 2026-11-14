# E103 external-unknown calibrated distance-ratio gate

未知代理改为真实未注册者：BCIC IV-2a 9 名被试经 E88 IDW 投影到 18 通道。原型来自项目 Train/session1，阈值由 Train 已知块 + 外部未知共同确定，Test 冻结后一次性读取。

## Train + 外部

AUC=0.807899，F1=0.744713；阈值=1.109893，known acceptance=0.629726，external unknown rejection=0.885995，目标达成=False。

## 项目 Test

真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。

| 项目 | 结果 | 目标 | 是否达标 |
|---|---:|---:|---|
| S1 拒识率 | 0.929113 | ≥0.80 | True |
| S2 接受率 | 0.125820 | ≥0.85 | False |

predictions.csv 完整落盘后才读取 session 审计。
