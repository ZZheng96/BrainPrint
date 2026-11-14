# E132 Three submission candidates on the PCA-96 score family

三个工作点覆盖三种优先级，全部阈值来自 out-of-fold Train 分数。

| 候选 | 优先级 | Test S1 拒识 | Test S2 接受 | 整体拒识 |
|---|---|---:|---:|---:|
| cand_S1 | S1 优先 | 0.7933 | 0.5872 | 0.4647 |
| cand_bal | 均衡 | 0.6729 | 0.6647 | 0.3813 |
| cand_S2 | S2 优先 | 0.5107 | 0.8518 | 0.1976 |

LOSO（全局分数）：拒识 0.6261，AUC 0.8579，9/20 折 ≥0.80。

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
