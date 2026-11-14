# E108 LOSO search for an S1-boosting signal

固定 S2 工作点（known acceptance = 0.85），在留一身份协议下比较各辅助信号能否提高 unknown rejection。
未访问项目 Test。

目标：known acceptance ≥ 0.85，unknown rejection ≥ 0.8。

| 信号 | AUC 均值 | known acc 均值 | unknown rej 均值 | 同时达标的折数 |
|---|---:|---:|---:|---:|
| and_base_cal | — | 0.7853 | 0.9011 | 0/20 |
| base_ratio | 0.9144 | 0.8499 | 0.8310 | 4/20 |
| cal_ratio | 0.8419 | 0.8499 | 0.5568 | 1/20 |
| neg_d1 | 0.7701 | 0.8499 | 0.4551 | 2/20 |
| neg_hb | 0.5009 | 0.8499 | 0.1602 | 0/20 |
| neg_z_d1 | 0.7803 | 0.8499 | 0.4455 | 2/20 |
| or_base_cal | — | 0.9145 | 0.4866 | 1/20 |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。
