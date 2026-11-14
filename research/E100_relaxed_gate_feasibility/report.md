# E100 relaxed-gate feasibility audit

未训练新模型，未读取项目 Test。未知代理为留身份 pseudo-unknown。

放宽目标：known acceptance ≥ 0.85，unknown rejection ≥ 0.8。

| 分支 | AUC | 可行工作点数 | 最佳 known acc | 最佳 unknown rej |
|---|---:|---:|---:|---:|
| raw | 0.631989 | 0 | 0.596991 | 0.647727 |
| spd | 0.640526 | 0 | 0.383665 | 0.838636 |
| min_fusion | 0.630104 | 0 | 0.581408 | 0.645455 |
| mean_fusion | 0.669036 | 0 | 0.598066 | 0.704545 |

**结论**：放宽目标下仍不存在可行工作点。

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。
