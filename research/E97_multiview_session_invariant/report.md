# E97 multiview session invariant

## Train-only calibration
AUC=0.609809，F1=0.236013；阈值=0.297117，known acceptance=0.950027，pseudo-unknown rejection=0.054545。

## Window ablation
Single AUC=0.631989，F1=0.927660；Multiview AUC=0.609809，F1=0.236013。

## Test
AUC/F1/C/RA/ASafeR/U/Official Score=N/A。S1拒识率=0.10121991427629409；S2接受率=0.9689874076386721。predictions.csv完整落盘后才读取session审计。
