# E152 跨会话鲁棒评分

最佳：**trial_center_unit_norm / B_E65logspd / k=1**，W=0.7500。
LOSO 拒识=1.0000，Test S1 拒识=0.0000，Test S2 接受=1.0000，诊断 AUC=0.4156。

Test 官方 AUC/F1：N/A（无真实标签）；Test session 仅作诊断，不参与任何变换、阈值或原型。严格沿用 E143 的 OOF 阈值和 LOSO 排除被评分块。

| transform | view | k | AUC | LOSO | S1拒识 | S2接受 | W |
|---|---|---:|---:|---:|---:|---:|---:|
| trial_center_unit_norm | B_E65logspd | 1 | 0.4156 | 1.0000 | 0.0000 | 1.0000 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 1 | 0.4156 | 1.0000 | 0.0000 | 1.0000 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 3 | 0.3788 | 1.0000 | 0.0000 | 1.0000 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 3 | 0.3788 | 1.0000 | 0.0000 | 1.0000 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 3 | 0.3788 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9999 | 0.7500 |
| trial_center_unit_norm | B_E65logspd | 1 | 0.4156 | 1.0000 | 0.0000 | 0.9999 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 1 | 0.4156 | 1.0000 | 0.0000 | 0.9999 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9999 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9999 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9999 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9999 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 3 | 0.3788 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| baseline | B_E65logspd | 3 | 0.2414 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 3 | 0.3788 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 3 | 0.3788 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 3 | 0.3788 | 1.0000 | 0.0000 | 0.9998 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 1 | 0.4156 | 1.0000 | 0.0000 | 0.9997 | 0.7499 |
| trial_center_unit_norm | B_E65logspd | 6 | 0.3738 | 1.0000 | 0.0000 | 0.9997 | 0.7498 |
| trial_center_unit_norm | B_E65logspd | 12 | 0.3775 | 1.0000 | 0.0000 | 0.9997 | 0.7498 |
| nuisance_pca1 | B_E65logspd | 3 | 0.2351 | 1.0000 | 0.0000 | 0.9997 | 0.7498 |
