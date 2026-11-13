# research 目录索引

本目录按实验创建时间编号：`E00` 为第一轮协方差 Ridge 基线（2026-10-29），`E01`–`E19` 按目录创建时间递增。
每个实验目录内保留该实验的报告（`report.md` 等）、指标文件与候选 CSV。

> 说明：所有 AUC/F1 均为 **Train Session1 内部验证**结果，Test 无公开标签，Test AUC/F1 不可计算。
> 官方评分公式：`Score = 100 × C^0.40 × (0.7×RA + 0.3×ASafeR)^0.35 × U^0.25`。

## 根级文档

| 文件 | 内容 |
|---|---|
| [brainprint_research_and_experiments.md](brainprint_research_and_experiments.md) | 研究假设、文献依据与四轮迭代主报告（历史数字主记录） |
| [E20-E35_DINO路线总结.md](E20-E35_DINO路线总结.md) | **EEG 图像化 + DINO 路线总结**：目标、E20–E35 全部尝试、困难、当前最优配置与指标 |
| [E36-E59_实验总结.md](E36-E59_实验总结.md) | **E36–E59 总结**：两条并行路线归属、10 项完整门禁、训练侧进展、已排除方向、跨会话瓶颈 |
| [E00-E19既往实验经验总结.md](E00-E19既往实验经验总结.md) | E00–E19 的经验与负结果 |
| [对话交接_实验历程与下一步.md](对话交接_实验历程与下一步.md) | 对话交接：实验历程、官方反馈与下一步方向 |
| [问题梳理.md](问题梳理.md) | 赛题结构、Session 元数据捷径与开放集问题梳理 |

## 实验目录

| 编号 | 目录 | 主题 | 关键指标（内部验证） |
|---|---|---|---|
| E00 | [E00_cov_ridge_baseline](E00_cov_ridge_baseline) | 第一轮：对数协方差 Ridge + z-score 对齐 + 图融合 + Sinkhorn | 块保留 5 折正常识别率 99.94%；前推 99.73%；伪未知 AUROC 0.9975 |
| E01 | [E01_brainprint_project_report](E01_brainprint_project_report) | 项目自包含 HTML 报告（ECharts/Mermaid） | 汇总 E00 结果 |
| E02 | [E02_open_set_improvement](E02_open_set_improvement) | 纯 EEG 嵌套留被试开集拒识 | base：AUROC 0.9935，macro-F1 0.9556，unknown-F1 0.9680，C=0.9321，U=0.9901 |
| E03 | [E03_open_set_calibration](E03_open_set_calibration) | 目标域 z-score 校准 | base_target_zscore：AUROC 0.9960，macro-F1 0.9653，unknown-F1 0.9746，C=0.9424 |
| E04 | [E04_pure_eeg_brainprint](E04_pure_eeg_brainprint) | 纯 EEG 脑纹方案设计（仅 plan） | 无指标 |
| E05 | [E05_asnorm_openset](E05_asnorm_openset) | ASNorm 归一化开集 | AUROC 0.9751，macro-F1 0.9262 |
| E06 | [E06_ea_riemannian_openset](E06_ea_riemannian_openset) | EA / Riemannian 对齐与轻量分类器对比 | EA 与收缩 LDA 均未超过 Ridge，路线停止 |
| E07 | [E07_raw_eeg_defense](E07_raw_eeg_defense) | 原始 EEG 对抗防御（特征空间 PGD） | 见 report.md 表格（C/RA/ASafeR/A/U/AUROC/F1） |
| E08 | [E08_raw_eeg_anchor](E08_raw_eeg_anchor) | 原始 EEG anchor 防御 | 冻结验收 `passed=false` |
| E09 | [E09_raw_eeg_robust](E09_raw_eeg_robust) | raw-PGD 对抗增强 Ridge（1:1 / 2:1） | 1:1 混合 targeted：unknown AUC 0.9718，unknown F1 0.9333；2:1：AUC 0.9738，F1 0.9362 |
| E10 | [E10_second_submission](E10_second_submission) | 第二次提交候选与冻结评估 | unknown AUC 0.9803，unknown F1 0.9472；Test AUC/F1 不可计算 |
| E11 | [E11_session_blind_candidate](E11_session_blind_candidate) | 无 Session 依赖候选（伪真值 v1–v4、五候选、嵌套伪未知） | 五候选 Test 拒识率 79.2%–95.4%；Test AUC/F1 不可计算 |
| E12 | [E12_unified_full_none_vs_raw_pgd](E12_unified_full_none_vs_raw_pgd) | full/none 与 raw-PGD 统一对比 | 见 report.md 表格（含 unknown AUROC/F1） |
| E13 | [E13_session_blind_shift_calibration](E13_session_blind_shift_calibration) | Session 盲偏移校准 | 见 report.md（unknown AUC/F1）；Test AUC/F1 不可计算 |
| E14 | [E14_session_blind_effective_solution](E14_session_blind_effective_solution) | 身份分支 + robust 分支融合方案 | 见 report.md（unknown AUC/F1） |
| E15 | [E15_session_blind_calibrated_v2](E15_session_blind_calibrated_v2) | 校准 v2 | AUC 0.965632，身份 macro-F1 0.936207，检测 macro-F1 0.936935 |
| E16 | [E16_session_blind_threshold_sweep_v3](E16_session_blind_threshold_sweep_v3) | 阈值扫描 v3 | AUC 0.965632，身份 macro-F1 0.936207，unknown F1 0.941038，U 0.915591 |
| E17 | [E17_enrollment_template_session_robust_v1](E17_enrollment_template_session_robust_v1) | 注册模板 Session 鲁棒 v1 | 见 report.md 表格；Test AUC/F1 不可计算 |
| E18 | [E18_enrollment_template_session_robust_v1_scaled](E18_enrollment_template_session_robust_v1_scaled) | 注册模板 v1 缩放版 | prototype_only AUC 0.734395 / F1 0.522782；prototype_margin AUC 0.751237 / F1 0.4712 |
| E19 | [E19_ridge_session_robust_v1](E19_ridge_session_robust_v1) | Ridge Session 鲁棒 v1 | AUC 0.993997，身份 macro-F1 0.925737，unknown-F1 0.951435；最终候选 base，Test 拒识 17,619/22,251 |
| E20 | [E20_eeg_image_dinov2](E20_eeg_image_dinov2) | EEG 图像化 + 冻结 DINOv2（纯 EEG，无 Session） | E20D 严格嵌套：covariance_only C=0.957/U=0.975/AUROC=0.993；covariance5_image C=0.929/U=0.432/AUROC=0.910；covariance5_fused 与纯协方差持平，无互补增益 |
| E21 | [E21_scale_invariant](E21_scale_invariant) | 跨会话尺度不变特征与拒识权衡 | cov_detrend_zband 闭集 10/9,232（0.9989）、C=0.9635/U=0.976/AUROC=0.994；Test S2 两阶段接受率仅 0.10–0.28，AUROC(S2 vs S1)=0.29–0.43 → 纯 EEG 下「S2 拒识≈0」与「U>0」互斥 |
| E22 | [E22_cross_session_viability](E22_cross_session_viability) | 跨会话身份可恢复性验证 | 相关矩阵对通道增益/频带缩放严格不变（保持率 1.000），但 Test 分数比仍 0.54、AUROC 0.29；域随机化对 Test 零改善 → 偏移在相关结构本身，判定不可恢复 |
| E23 | [E23_spatial_reference](E23_spatial_reference) | 空间参考/通道混合的跨会话对齐 | CAR/Cz/拉普拉斯只把闭集错误 27→18、AUROC(S2 vs S1) 0.317→0.43，分数比仍 0.50–0.55；另发现 Test S2 双峰（7,985 低范数 / 11,233 高范数），高范数组 AUROC 仍 0.358 |
| E24 | [E24_norm_gate_robustness](E24_norm_gate_robustness) | 扰动检测（范数门控）与候选 | 噪声注入 SNR30 dB 使范数 59.7→31.7，与 Low 组 34.1 吻合；门控在 clean 子集拒识 0.5%、低范数组拒识 98.5%；嵌套 C=0.9854/F1=0.9324/U=0.0102/AUROC=0.9929；**结论已被 E25A 推翻，候选不建议提交** |
| E25 | [E25_robust_recognition](E25_robust_recognition) | RA 导向的对抗增广识别 | 同会话 SNR30 扰动下 RA 0.9896→0.9972（ASafeR 1.0000），但对 Test 低范数组接受率只从 0.041 到 0.055 → 低范数组不是对抗样本，跨会话识别才是唯一瓶颈 |
| E26 | [E26_final_candidates](E26_final_candidates) | 最终候选装配（距离拒识 + 对抗增广） | 距离型新颖度 AUROC(S2 vs S1)=0.758；reject5% 时 S2 接受 0.845 / S1 拒识 0.467，代理分约 34.6；五个候选已校验格式 |
| E27 | [E27_unknown_rejection](E27_unknown_rejection) | 提升 S1 未注册拒识率 | 五信号秩融合反而把 AUROC 从 0.758 拉到 0.467（内层伪未知与真实 S1 分布不同）；只用 dist 时 S1 拒识 0.47–0.76 对应 S2 接受 0.85–0.60，八个工作点候选已生成 |
| E28 | [E28_dino_openset](E28_dino_openset) | DINO 特征上的开集拒识改进 | cov5 上 S1 距离中位数 54.27 vs S2 20.94（差距 33.3，原始协方差仅 3.1）；reject1% 时 S2 接受 0.952 / S1 拒识 0.435，同时优于 E27 的 0.965/0.207 与 0.845/0.467 |
| E29 | [E29_dino_calibration](E29_dino_calibration) | DINO 开集标定改进 | 修正 E28 的伪未知构造错误：正确留出被试下内层 AUROC 0.6369（非 0.45–0.51），与 Test 0.7065 仅差 0.07；q01 时 S2 接受 0.955 / S1 拒识 0.430 |
| E30 | [E30_dino_multiview](E30_dino_multiview) | DINO 多视图距离融合 | cov5 类内白化（WCCN）距离最优：Test AUROC 0.7243，加 PCA 残差 0.7253；q01 时 S2 接受 0.938 / S1 拒识 0.486（E29 为 0.952/0.435）；topomap5 融合反而降到 0.66–0.68 |
| E31 | [E31_task_conditional](E31_task_conditional) | DINO 任务条件化类中心 | 任务特异模型 + 任务内 WCCN：Test AUROC 0.7327（内层 0.7119），q01 时 S2 接受 0.917 / S1 拒识 0.509；工作点与 E30 持平 |
| E32 | [E32_distance_ratio](E32_distance_ratio) | 距离比率类开集判据 | dist+ratio 的 Test AUROC 0.7403（最高），但 S2 接受 0.894 时 S1 拒识 0.506，工作点未改善；gap 方向相反（内层 AUROC 0.081） |
| E33 | [E33_arcface_head](E33_arcface_head) | ArcFace 身份头微调 DINO 特征 | **负结果**：ArcFace 闭集 0.8957/C 0.8773/Test AUROC 0.6321，普通 CE 0.8410/0.8129/0.6522，均劣于线性探针（0.9294/0.7327）；20 类数据量不足以训练非线性投影头 |
| E34 | [E34_covariance_hr](E34_covariance_hr) | 高分辨率协方差图（224×224） | **负结果**：Test AUROC 0.7223 < covariance5 的 0.7327，内层 0.6400 < 0.7119；分辨率不是瓶颈，DINO 特征对跨会话漂移敏感 |
| E35 | [E35_per_task_analysis](E35_per_task_analysis) | 逐任务身份证据拆解 | SSVEP 最强（Test AUROC 0.7528）、EC 最弱（0.7022）；内层与 Test 逐任务吻合（差 0.006–0.020）；任务内阈值与全局阈值等价（同 S2 接受率下 S1 拒识差 ≤0.01） |
| E36 | [E36_pure_eeg](E36_pure_eeg) | 纯 EEG 五频带 logSPD + WCCN 原型主线，含 E36C 六攻击嵌套评估 | 内层 clean：direct C 0.9992、身份 macro-F1 0.9992、unknown AUROC 0.9962、U@95 0.9927；E36C 外层六攻击宏 C 0.9395 / RA 0.0609 / ASafeR 0.6818 / A 0.2472 / U 1.0 / Score 59.78；RA 与 ASafeR 门禁未过 |
| E37 | [E37_spd_adversarial](E37_spd_adversarial) | 残差 SPD + ARPL 开空间损失 + 原始波形对抗训练 | **负结果/未完成**：SMOKE 单折 clean 0.99–1.00，但 PGD20/SNR20 攻击后准确率 0.03（FGSM30 0.39）；并证伪「放大 logits 尺度可增强对抗训练」——L2 归一化 PGD 下尺度不变 |
| E38 | [E38_image_pure_fusion](E38_image_pure_fusion) | DINO 图像分支与纯 EEG WCCN 的分数级 ECDF 融合 | **负结果**：内层选出图像权重 0.0；外层 direct C 0.9998、AUROC 0.9978，但 q05 工作点 U 仅 0.6774，U 门禁未过；图像分支无增益 |
| E39 | [E39_ecdf_transfer](E39_ecdf_transfer) | 类数不变 ECDF 分数校准与阈值跨类数迁移 | **8/8 门禁通过**：阈值 0.023106；K8 内层 C 0.9774 / U 0.9688 / AUROC 0.9964；K16 外层 C 0.9773 / U 0.9937 / AUROC 0.9981；K20 全注册 direct C 0.9996 |
| E40 | [E40_attack_rejection](E40_attack_rejection) | 双门控（distance_ratio + max_cosine）攻击感知拒识，冻结 E36 模型 | 机制成立：C 0.9500、身份 F1 0.9983（direct）、U 0.9902、检测器 AUROC 0.9008 / F1 0.8690；ASafeR 0.6062→**0.7082**、Score 56.39→**59.14**（同协议对照 +2.75）；但对照 E36C 外层（Score 59.78）差 −0.64，未达采纳判据；目标 PGD 检测 AUROC 仅 0.76–0.85 是短板 |
| E41 | [E41_supervised_detector](E41_supervised_detector) | fit 折自训练（clean + 5 类对抗增广）监督攻击检测器 | **负结果**：单折 detAUROC 0.8976 vs E40 无监督 0.8926，ASafeR 0.7324 vs 0.7480；8 维逐 trial 统计量信息量不足，未扩展至完整内层 |
| E42 | [E42_submission_candidates](E42_submission_candidates) | 一次 Test 推理产出两份冻结候选（E36C 单门控 / E40 双门控） | 校准块 N=2,301，q=0.0305、t_u=0.081456、t_mc=0.548043，联合接受率 0.9505；整体拒识 0.8093 / 0.8155（与 E19 的 79.2% 同量级，管线复现既有提交行为）；A/B 分歧 0.62%；Test 指标全部 N/A |
| E43 | [E43_noise_robust_features](E43_noise_robust_features) | 每频带归一化空间相关特征（抗各向同性噪声假设） | **负结果**：攻击下准确率反低于 logSPD（FGSM30 0.148 vs 0.227，PGD20 0.000 vs 0.090，目标攻击 100% 命中）；相关结构比含特征值的 logSPD 更易被推动 |
| E44 | [E44_transfer_robustness](E44_transfer_robustness) | 迁移攻击 vs 白盒攻击强度对比 | **迁移不比白盒弱**：从 shrinkage=0.001 替代模型迁移，PGD20 直接准确率 0.035（白盒 0.082），Ridge 迁移 0.000；「官方预生成攻击更弱」的捷径不存在，E36C 的 RA≈0.06 是合理估计 |
| E45 | [E45_raw_cnn_at](E45_raw_cnn_at) | 原始波形 CNN + PGD7 对抗训练 | **突破 RA 边界**：单折 PGD20/SNR20 直接准确率 0.551（logSPD 0.082）、宏 RA 0.748、目标误认证 0.922→0.207；但 softmax 置信度的伪未知 U 仅 0.0156 |
| E46 | [E46_embedding_rejection](E46_embedding_rejection) | CNN 嵌入空间原型/Mahalanobis 拒识 | **负结果**：嵌入 max_cosine U=0.301、Mahalanobis 0.129，均远低于 0.95；CNN 学到跨被试可迁移特征，未知被试落在原型附近，CNN 不适合承担拒识 |
| E47 | [E47_dual_model_system](E47_dual_model_system) | **双模型分工：CNN 身份 + logSPD 拒识** | **当前最优**：C 0.9148、RA **0.5772**、ASafeR **0.9076**、A 0.6763、U 0.9965、F1 0.9358（direct 0.9522）、**Score 84.08**（E36C 59.78，+24.30）；RA/ASafeR 首次从不可达变为可优化 |
| E48 | [E48_cnn_clean_tuning](E48_cnn_clean_tuning) | CNN clean 准确率容量调参 | **负结果**：width 128/80 epoch、clean 权重 0.7 等配置均未把 clean 直接准确率推过 0.965（原 0.9537）；单靠 CNN 无法满足 C≥0.95，导向 E49 |
| E49 | [E49_three_region](E49_three_region) | **三段式决策：clean 用 logSPD、受扰动用 CNN、未知拒识** | **当前最优**：C **0.9719** ✅、F1 direct 0.9871 ✅、U **0.9820** ✅、RA **0.7313**、ASafeR **0.9607**、A 0.8001、**Score 91.03**（E47 84.08、E36C 59.78）；SNR30 下 RA 0.905–0.941、ASafeR 0.977–0.992 已达目标量级，宏均值被 SNR20 拖低 |
| E50 | [E50_ensemble_at](E50_ensemble_at) | 3 种子 CNN 集成 + SNR20 加重对抗训练 | **无增益**：同折上 3 集成/Single+0.65/Single+0.50 的宏 RA 0.8132/0.8086/0.8073、Score 93.40/93.28/93.19，差异小于折间波动；不采用 |
| E51 | [E51_tta](E51_tta) | 测试时增强（5 个时间平移视图平均 logits） | **小幅有效**：C 0.9727→0.9766、RA 0.7773→0.7826、ASafeR 0.9323→0.9375、Score 92.23→**92.58**（+0.35，零训练成本）；增益集中在目标攻击 |
| E52 | [E52_frozen_submission](E52_frozen_submission) | E49 系统的全 Train 固化与 Test 提交 | 校准块 logSPD direct 0.9996、CNN(TTA5) 0.9796，τ=1.0796/1.1071；**Test 72.5% 判为拒识**（Train 校准块仅 2%），跨会话域偏移严重，CNN 区只覆盖 5.9% 样本；E49 的 91.03 不能外推为 Test 预期 |
| E53 | [E53_raw_eeg_open_robust](E53_raw_eeg_open_robust) | 原始 EEG 轻量 EEGNet + ARPL 开空间 + 对抗训练（单模型端到端） | main Score 67.52（C 0.9567 / RA 0.7696 / ASafeR 0.8780 / U 0.3316）；untargeted_at 70.52；内层门禁未过 |
| E54 | [E54_outlier_exposure](E54_outlier_exposure) | 非类别化配对 Outlier Exposure（单模型端到端） | **oe 71.97**（C 0.9618 / RA 0.7824 / ASafeR 0.8901 / U 0.4180）；control 66.82；OE 显著优于 control，但 U 仍低 |
| E55 | [E55_online_adversarial](E55_online_adversarial) | 每批次在线原始波形 PGD7（替代缓存 PGD3） | 仅 smoke 完成（覆盖/梯度/显存通过），inner 未跑 |
| E56 | [E56_cross_session_calibration](E56_cross_session_calibration) | 跨会话分数偏移诊断 | **偏移不可用后处理修复**：Test dr 中位 1.033 vs Train 1.240，q95 3.65 vs 1.77；秩映射后拒识率仍 0.7231（非单调平移，分布形状不同） |
| E57 | [E57_self_normalized_score](E57_self_normalized_score) | 逐样本自归一化分数（未执行） | 因 E56 已证明偏移非尺度问题而中止，未产生结果 |
| E58 | [E58_oe_three_region](E58_oe_three_region) | **OE-EEGNet 身份 + logSPD 三段式** | **当前最优**：C **0.9714** ✅、F1 direct 0.9849 ✅、AUROC 0.9984 ✅、U **0.9875** ✅、RA **0.9371**、ASafeR **0.9758**、A 0.9487、**Score 96.74**（E49 91.03，+5.71；五折 96.55–96.83 极稳定）；SNR30 三档全部达标，仅 SNR20 差 0.02–0.04 |
| E59 | [E59_online_three_region](E59_online_three_region) | E58 + E55 每批次在线 PGD7 | **无增益**：Score 96.65（E58 96.74，−0.09）、RA 0.9362（−0.0009）、ASafeR 持平；未达 RA≥0.95/ASafeR≥0.98 判据，保留 E58 |
| E60 | [E60_test_audit_diagnosis](E60_test_audit_diagnosis) | Test S1/S2 拒识门禁可达性诊断 | **不可达**：S1/S2 的 dr 分布几乎重合（中位 1.0371 vs 1.0329），AUROC(S2 vs S1)=**0.4159**；S2 拒识=2% 时 S1 拒识仅 0.0201，S1 拒识=95% 时 S2 拒识 0.9978；两项门禁互斥，是特征问题而非阈值问题 |

## 其他

| 目录 | 内容 |
|---|---|
| [_trash](_trash) | 已从根级移出的无引用/可重建 json、csv（确认无用后可整体删除） |

## 官方提交记录

| 提交 | 文件 | 官方成绩 |
|---|---|---|
| 第 1 次 | `commit/answer/赞美诗_第1次.csv` | 69.17/100，排名 8/39 |
| 第 2 次 | `commit/answer/赞美诗_第2次.csv` | 67.94/100 |
| 第 3 次 | `commit/answer/赞美诗_第3次.csv`、`赞美诗_第3次_rawPGD.csv` | 待官方反馈 |
