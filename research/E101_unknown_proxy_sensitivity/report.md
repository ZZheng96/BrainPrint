# E101 unknown-proxy sensitivity audit

未训练新模型，未读取项目 Test。比较三种未知代理在同一冻结 E95 编码器上的表现。

目标：known acceptance ≥ 0.85，unknown rejection ≥ 0.8。

| 未知代理 | N | 分支 | AUC | 可行点数 | 最佳 known acc | 最佳 unknown rej |
|---|---:|---|---:|---:|---:|---:|
| held_identity | 440 | raw | 0.606750 | 0 | 0.545415 | 0.652273 |
| held_identity | 440 | spd | 0.394085 | 0 | 0.000435 | 1.000000 |
| held_identity | 440 | mean_fusion | 0.507214 | 0 | 0.622338 | 0.431818 |
| late_block | 2308 | raw | 0.524869 | 0 | 0.734898 | 0.309792 |
| late_block | 2308 | spd | 0.525982 | 0 | 0.549761 | 0.504766 |
| late_block | 2308 | mean_fusion | 0.532050 | 0 | 0.596697 | 0.474003 |
| held_identity_late | 440 | raw | 0.646204 | 0 | 0.607997 | 0.629545 |
| held_identity_late | 440 | spd | 0.456189 | 0 | 0.959583 | 0.047727 |
| held_identity_late | 440 | mean_fusion | 0.566325 | 0 | 0.596697 | 0.554545 |

**是否存在可行工作点**：否。

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。
