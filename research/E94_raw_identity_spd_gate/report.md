# E94 raw identity SPD gate

Train/session1 内层伪未知校准 raw confidence 与 correlation-logSPD confidence；冻结模型、参数后完成唯一一次 Test 推理。Session 仅在完整 predictions.csv 落盘后审计。

## Train-only 指标

| AUC | F1 | C | U |
|---:|---:|---:|---:|
| 0.385275 | 0.609315 | 0.704669 | 0.096536 |

## Test 审计

Test 无真实标签：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。

| 项目 | 结果 | 目标 | 是否达标 |
|---|---:|---:|---|
| S1 拒识率 | 0.727003 | >=0.95 | 未通过 |
| S2 接受率 | 0.531481 | >=0.98 | 未通过 |

拒识规则：spd；阈值和候选均未使用 Test session。
