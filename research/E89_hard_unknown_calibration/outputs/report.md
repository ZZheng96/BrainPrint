# E89 分层 hard-unknown calibration

基于 E88 外部嵌入；仅使用 E73 派生数据，未访问项目 Test。每个 outer fold 先用 calibration known/unknown 决策阈值，再对 outer query 只评估一次。

| 策略 | AUC | known acceptance | unknown FAR | identity macro-F1 | unknown-F1(pos_label=0) | EER |
|---|---:|---:|---:|---:|---:|---:|
| q05_known_only | 0.905129 | 0.924479 | 0.310185 | 0.959585 | 0.760630 | 0.169850 |
| tail_q01_known_only | 0.905129 | 0.979167 | 0.491898 | 0.989312 | 0.651957 | 0.169850 |
| known_unknown_far10 | 0.905129 | 0.792535 | 0.146991 | 0.877967 | 0.802638 | 0.169850 |
| known_unknown_youden | 0.905129 | 0.862847 | 0.204861 | 0.923699 | 0.801492 | 0.169850 |
| per_identity_q05 | 0.905129 | 0.921875 | 0.278935 | 0.957809 | 0.783505 | 0.169850 |

## Trade-off
- `q05_known_only` 保持 E88 基线的 known coverage，但不使用 calibration unknown，通常 unknown FAR 较高。
- `known_unknown_far10` 以 calibration unknown 的 90% 分位控制约 10% FAR，偏安全但可能降低 known acceptance。
- `known_unknown_youden` 在 calibration known/unknown 上最大化 Youden 指数，折中 FAR 与 acceptance；不使用 outer unknown。
- `per_identity_q05` 为每个 enrolled identity 单独用 calibration known 估计 q05，适合身份间分数尺度不同；`tail_q01_known_only` 更保守。

项目 Test AUC/F1：N/A（明确未访问）。
