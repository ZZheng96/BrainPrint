# E95 E54 final20 candidate

Train/session1 only for training and calibration; the final 20-class model is frozen before the sole Test read.

## Train-only calibration

AUC=0.598731，F1=0.876676。阈值=0.333006，known acceptance=0.950027，pseudo-unknown rejection=0.095455。

## Test

真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。

S1 拒识率=0.154303；S2 接受率=0.941357。predictions.csv 已完整落盘后才读取 session 做审计。
