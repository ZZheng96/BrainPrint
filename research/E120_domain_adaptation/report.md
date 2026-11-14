# E120 Domain adaptation on correlation-logSPD (diagnostic only)

E116/E119 已用 14 个分数证明 Test S1-vs-S2 AUC 上限约 0.68–0.70。本实验测试唯一剩下的机械杠杆：改变特征几何（而非决策规则）的域适应。所有变换只在 Train 拟合块上拟合，未用 Test 统计。

| 变体 | 诊断 AUC | S1≥0.80 时 S2 接受 | 最佳均衡点 |
|---|---:|---:|---:|
| D0_baseline_max | 0.6854 | 0.4230 | 0.6548 |
| D0b_baseline_ratio | 0.5798 | 0.3333 | 0.5585 |
| D1_pertask_z_max | 0.6417 | 0.3769 | 0.6135 |
| D2_wccn_max | 0.5879 | 0.2870 | 0.5625 |
| D3_wccn_pertask_z_max | 0.5805 | 0.2869 | 0.5519 |
| D4_pca64_wccn_max | 0.7614 | 0.5360 | 0.6841 |
| D5_pca64_wccn_ratio | 0.4354 | 0.0644 | 0.4672 |
| D6_pertask_center_renorm_max | 0.5487 | 0.2724 | 0.5414 |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
