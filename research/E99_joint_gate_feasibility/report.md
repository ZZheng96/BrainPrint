# E99 Train-only joint gate feasibility audit

未训练新模型，未读取项目 Test。未知代理为留身份 pseudo-unknown。

目标：known acceptance ≥ 0.98，unknown rejection ≥ 0.95。

| 分支 | AUC | 可行工作点数 | 最佳联合点 known acc | 最佳联合点 unknown rej | U@95 时 known acc |
|---|---:|---:|---:|---:|---:|
| raw | 0.631989 | 0 | 0.596991 | 0.647727 | 0.077915 |
| spd | 0.640526 | 0 | 0.382590 | 0.838636 | 0.260613 |
| min_fusion | 0.630104 | 0 | 0.581408 | 0.645455 | 0.133799 |

**结论**：Train-only 代理下不存在同时满足 S1/S2 目标的工作点；瓶颈是分布重叠。

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。
