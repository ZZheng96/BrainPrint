# E130 Per-identity thresholds from out-of-fold scores

身份之间的紧致程度差异很大，单一全局阈值会系统性过度拒绝松散的identity。本实验用 5 折 leave-block-out 的 Train 分数为每个身份（或原型/任务）单独标定阈值。全部只用 Train。

| 变体 | 阈值类型 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 |
|---|---|---:|---:|---:|
| G_global | global threshold | 0.7655 | 0.7933 | 0.5872 |
| P1_per_identity | identity | 0.7138 | 0.6109 | 0.7214 |
| P2_per_prototype | prototype | 0.6955 | 0.6930 | 0.5852 |
| P3_per_task | task | 0.7638 | 0.7794 | 0.5939 |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
