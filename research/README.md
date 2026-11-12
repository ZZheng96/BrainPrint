# research 目录索引

本目录按实验创建时间编号：`E00` 为第一轮协方差 Ridge 基线（2026-10-29），`E01`–`E19` 按目录创建时间递增。
每个实验目录内保留该实验的报告（`report.md` 等）、指标文件与候选 CSV。

> 说明：所有 AUC/F1 均为 **Train Session1 内部验证**结果，Test 无公开标签，Test AUC/F1 不可计算。
> 官方评分公式：`Score = 100 × C^0.40 × (0.7×RA + 0.3×ASafeR)^0.35 × U^0.25`。

## 根级文档

| 文件 | 内容 |
|---|---|
| [brainprint_research_and_experiments.md](brainprint_research_and_experiments.md) | 研究假设、文献依据与四轮迭代主报告（历史数字主记录） |
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
| E21 | [E21_scale_invariant](E21_scale_invariant) | 跨会话尺度不变特征与拒识权衡 | cov_detrend_zband 闭集 10/9,232（0.9989）、C=0.9635/U=0.976/AUROC=0.994；Test S2 两阶段接受率仅 0.10–0.24，AUROC(S2 vs S1)=0.40 → 纯 EEG 下「S2 拒识≈0」与「U>0」互斥 |

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
