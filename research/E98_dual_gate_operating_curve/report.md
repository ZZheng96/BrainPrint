# E98 dual-gate operating curve

复用冻结 E95 raw identity encoder 与 E65 correlation-logSPD 原型；接受规则仅由 Train/session1 已知/校准块选择。

## Train-only

AUC=0.630104，F1=0.929320；raw 阈值=0.054329，SPD 阈值=-19.572369，known acceptance=0.950027，pseudo-unknown rejection=0.075000。

## Test

真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。

S1 拒识率=0.4516979887899769；S2 接受率=0.7686023519617026。predictions.csv 完整落盘后才读取 session 审计。
