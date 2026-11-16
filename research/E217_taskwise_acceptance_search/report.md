# E217 taskwise acceptance search

- 独立实验；复用 E215 分数/模型缓存，不修改 E215/E216。
- 任务级 acceptance 候选：`[0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]`；组合数：4096/config。
- 选点只使用 outer-fold inner OOF known 分数校准阈值及严格 LOSO；筛选 LOSO rejection micro 或 macro ≥ 0.90 且 known acceptance micro ≥ 0.80。字典序：rejection micro → rejection macro → rejection F1 → known acceptance。
- all-20 final_inner_oof 仅用于冻结最终 task threshold；Test S1/S2 仅审计。官方 Test AUC/F1：N/A。

## E65_PCA96_k12
- **无组合同时满足** LOSO rejection micro 或 macro ≥ 0.90 且 known acceptance micro ≥ 0.80；4096 个候选均未入选。
- LOSO AUC fold mean：**0.910173**；identity macro-F1：**0.987757**；final-inner-OOF identity macro-F1：**0.987288**；官方 Test AUC/F1：**N/A / N/A**。

## E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6
- **无组合同时满足** LOSO rejection micro 或 macro ≥ 0.90 且 known acceptance micro ≥ 0.80；4096 个候选均未入选。
- LOSO AUC fold mean：**0.904961**；identity macro-F1：**0.985737**；final-inner-OOF identity macro-F1：**0.985491**；官方 Test AUC/F1：**N/A / N/A**。

运行耗时：612.84s；完整候选与结果见 `results.json`。
