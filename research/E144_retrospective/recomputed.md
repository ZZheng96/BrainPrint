# 新加权目标重算：全部已存实验

目标：`W = 0.25×LOSO 拒识 + 0.25×Test S1 拒识 + 0.5×Test S2 接受`。
每行是该实验在所有已存工作点里 W 最高的一个。

| 实验 | 配置 | 接受率 | LOSO | S1 拒识 | S2 接受 | W | 诊断 AUC | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E142_cdf_fusion | rawz:B_plus_cleanA | 0.95 | 0.9475 | 0.5987 | 0.8022 | **0.7877** | nan | **LOSO 被污染，W 偏高** |
| E139_feature_fusion | C_plus_cov | 0.96 | 0.9724 | 0.5997 | 0.7888 | **0.7875** | nan | **LOSO 被污染，W 偏高** |
| E141_weighted_image_route | score:B_E65logspd | 0.97 | 0.9481 | 0.5124 | 0.8290 | **0.7796** | nan | **LOSO 被污染，W 偏高** |
| E135_rank_calibration | - | nan | 0.8301 | 0.3963 | 0.9175 | **0.7653** | 0.7822 | **犯规：用了 Test 侧统计** |
| E138_scale_free_score | d1 | 0.95 | 0.8244 | 0.6159 | 0.7597 | **0.7399** | nan | **LOSO 被污染，W 偏高** |
| E137_multi_view | greedy | 0.90 | 0.9902 | 0.5740 | 0.6133 | **0.6977** | nan | **LOSO 被污染，W 偏高** |
| E145_fusion_recheck_fixed_loso | B_only | 0.90 | 0.7936 | 0.7979 | 0.5960 | **0.6958** | nan | 干净 |
| E143_k_sweep_nearest | B_only | 0.90 | 0.7936 | 0.7979 | 0.5960 | **0.6958** | nan | 干净 |
| E149_image_route_audit | B_only | 0.90 | 0.7936 | 0.7979 | 0.5960 | **0.6958** | nan | 干净 |
| E147_phase_features_eval | B_only | 0.90 | 0.7936 | 0.7979 | 0.5960 | **0.6958** | nan | 干净 |
| E148_logbandpower_probe | B_plus_logbp | 0.85 | 0.8282 | 0.8675 | 0.4809 | **0.6644** | nan | 干净 |
| E140_metric_learning | PCA128_raw_control | 0.90 | 0.7138 | 0.8190 | 0.5458 | **0.6561** | nan | **LOSO 被污染，W 偏高** |
| E123_band_pca | - | nan | 0.6261 | 0.7916 | 0.5823 | **0.6456** | nan | 干净 |
| E136_clean_three_way | P96_k3 | 0.95 | 0.4819 | 0.7043 | 0.6930 | **0.6431** | 0.7822 | 干净 |
| E134_steep_loso | P96 | 0.85 | 0.4790 | 0.7956 | 0.5769 | **0.6071** | nan | 干净 |
| E125_band_fusion | - | nan | 0.3831 | 0.7221 | 0.6403 | **0.5965** | 0.7458 | 干净 |

## 无法补算的实验

这些实验当时没有在同一工作点同时记录 LOSO 拒识与 Test S1/S2，因此无法事后拼出 W（拼出来的数不是同一个阈值下的三个率，没有意义）。

| 实验 | 原因 |
|---|---|
| E05_asnorm_openset | 未存 LOSO/S1/S2 三者，无法补算 |
| E07_raw_eeg_defense | 未存 LOSO/S1/S2 三者，无法补算 |
| E08_raw_eeg_anchor | 未存 LOSO/S1/S2 三者，无法补算 |
| E09_raw_eeg_robust | 未存 LOSO/S1/S2 三者，无法补算 |
| E100_relaxed_gate_feasibility | 未存 LOSO/S1/S2 三者，无法补算 |
| E101_unknown_proxy_sensitivity | 未存 LOSO/S1/S2 三者，无法补算 |
| E102_task_distance_ratio | 未存 LOSO/S1/S2 三者，无法补算 |
| E103_external_unknown_calibration | 未存 LOSO/S1/S2 三者，无法补算 |
| E104_session_invariant_pretrain | 未存 LOSO/S1/S2 三者，无法补算 |
| E105_loso_candidate_comparison | 未存 LOSO/S1/S2 三者，无法补算 |
| E106_loso_fusion_frozen | 未存 LOSO/S1/S2 三者，无法补算 |
| E107_proxy_test_gap | 未存 LOSO/S1/S2 三者，无法补算 |
| E108_loso_s1_boost | 未存 LOSO/S1/S2 三者，无法补算 |
| E109_loso_high_s2_candidates | 未存 LOSO/S1/S2 三者，无法补算 |
| E110_loso_rejection_boost | 未存 LOSO/S1/S2 三者，无法补算 |
| E111_external_joint_embedding | 未存 LOSO/S1/S2 三者，无法补算 |
| E112_open_set_classifier | 未存 LOSO/S1/S2 三者，无法补算 |
| E113_loso_finetune | 未存 LOSO/S1/S2 三者，无法补算 |
| E114_finetune_ablation | 未存 LOSO/S1/S2 三者，无法补算 |
| E115_v3_frozen_audit | 未存 LOSO/S1/S2 三者，无法补算 |
| E116_test_separability | 未存 LOSO/S1/S2 三者，无法补算 |
| E117_dual_target_loso | 未存 LOSO/S1/S2 三者，无法补算 |
| E118_c7_frozen_audit | 未存 LOSO/S1/S2 三者，无法补算 |
| E119_score_battery | 未存 LOSO/S1/S2 三者，无法补算 |
| E120_domain_adaptation | 未存 LOSO/S1/S2 三者，无法补算 |
| E121_pca_sweep | 未存 LOSO/S1/S2 三者，无法补算 |
| E122_pca96_deploy | 未存 LOSO/S1/S2 三者，无法补算 |
| E124_cluster_structure | 未存 LOSO/S1/S2 三者，无法补算 |
| E126_lda_reduction | 未存 LOSO/S1/S2 三者，无法补算 |
| E127_cross_task | 未存 LOSO/S1/S2 三者，无法补算 |
| E128_distance_forms | 未存 LOSO/S1/S2 三者，无法补算 |
| E129_more_prototype_data | 未存 LOSO/S1/S2 三者，无法补算 |
| E130_per_identity_thresholds | 未存 LOSO/S1/S2 三者，无法补算 |
| E131_fusion_thresholds | 未存 LOSO/S1/S2 三者，无法补算 |
| E132_submission_candidates | 未存 LOSO/S1/S2 三者，无法补算 |
| E133_mean_objective | 未存 LOSO/S1/S2 三者，无法补算 |
| E36_pure_eeg | 未存 LOSO/S1/S2 三者，无法补算 |
| E38_image_pure_fusion | 未存 LOSO/S1/S2 三者，无法补算 |
| E39_ecdf_transfer | 未存 LOSO/S1/S2 三者，无法补算 |
| E40_attack_rejection | 未存 LOSO/S1/S2 三者，无法补算 |
| E42_submission_candidates | 未存 LOSO/S1/S2 三者，无法补算 |
| E45_raw_cnn_at | 未存 LOSO/S1/S2 三者，无法补算 |
| E47_dual_model_system | 未存 LOSO/S1/S2 三者，无法补算 |
| E49_three_region | 未存 LOSO/S1/S2 三者，无法补算 |
| E52_frozen_submission | 未存 LOSO/S1/S2 三者，无法补算 |
| E53_raw_eeg_open_robust | 未存 LOSO/S1/S2 三者，无法补算 |
| E54_outlier_exposure | 未存 LOSO/S1/S2 三者，无法补算 |
| E55_online_adversarial | 未存 LOSO/S1/S2 三者，无法补算 |
| E58_oe_three_region | 未存 LOSO/S1/S2 三者，无法补算 |
| E59_online_three_region | 未存 LOSO/S1/S2 三者，无法补算 |
| E64_e58_test_audit | 未存 LOSO/S1/S2 三者，无法补算 |
| E65_correlation_spd | 未存 LOSO/S1/S2 三者，无法补算 |
| E66_neuroshield_frozen | 未存 LOSO/S1/S2 三者，无法补算 |
| E70_image_adversarial_training | 未存 LOSO/S1/S2 三者，无法补算 |
| E72_pair_protocol | 未存 LOSO/S1/S2 三者，无法补算 |
| E73_external_multisession | 未存 LOSO/S1/S2 三者，无法补算 |
| E90_project18_identity_template | 未存 LOSO/S1/S2 三者，无法补算 |
| E91_hybrid_identity_unknown | 未存 LOSO/S1/S2 三者，无法补算 |
| E92_frozen_project_candidate | 未存 LOSO/S1/S2 三者，无法补算 |
| E93_frozen_audit_only | 未存 LOSO/S1/S2 三者，无法补算 |
| E94_raw_identity_spd_gate | 未存 LOSO/S1/S2 三者，无法补算 |
| E94_robust_raw_template | 未存 LOSO/S1/S2 三者，无法补算 |
| E95_e54_final20_candidate | 未存 LOSO/S1/S2 三者，无法补算 |
| E96_domain_robust_open_set | 未存 LOSO/S1/S2 三者，无法补算 |
| E97_multiview_session_invariant | 未存 LOSO/S1/S2 三者，无法补算 |
| E98_dual_gate_operating_curve | 未存 LOSO/S1/S2 三者，无法补算 |
| E99_joint_gate_feasibility | 未存 LOSO/S1/S2 三者，无法补算 |
