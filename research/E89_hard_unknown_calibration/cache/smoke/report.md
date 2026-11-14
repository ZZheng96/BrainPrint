# E89 分层 hard-unknown calibration

基于 E88 外部嵌入；仅使用 E73 派生数据，未访问项目 Test。每个 outer fold 先用 calibration known/unknown 决策阈值，再对 outer query 只评估一次。

| 策略 | AUC | known acceptance | unknown FAR | identity macro-F1 | unknown-F1(pos_label=0) | EER |
|---|---:|---:|---:|---:|---:|---:|
| smoke | N/A（未评估 outer query） | - | - | - | - | - |

## Trade-off
- `q05_known_only` 保持 E88 基线的 known coverage，但不使用 calibration unknown，通常 unknown FAR 较高。
- `known_unknown_far10` 以 calibration unknown 的 90% 分位控制约 10% FAR，偏安全但可能降低 known acceptance。
- `known_unknown_youden` 在 calibration known/unknown 上最大化 Youden 指数，折中 FAR 与 acceptance；不使用 outer unknown。
- `per_identity_q05` 为每个 enrolled identity 单独用 calibration known 估计 q05，适合身份间分数尺度不同；`tail_q01_known_only` 更保守。

项目 Test AUC/F1：N/A（明确未访问）。
