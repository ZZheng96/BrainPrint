# 冻结验收实验报告

- 仅使用训练 features/metadata 与已有任务特异分数；未使用 Test 标签，未修改正式 CSV。
- pair gate 协议：同 task、session2、相似度 > 0.99；blocked OOF 在同 fold 内构图；组件 purity 与 margin 只来自 direct prediction。

| 协议 | 方案 | C/accuracy | F1 macro | unknown AUC | unknown F1 |
|---|---|---:|---:|---:|---:|
|blocked_oof (5 trial-block folds)|first_submission_equivalent|1.000000|1.000000|N/A|N/A|
|blocked_oof (5 trial-block folds)|second_pair_gate|1.000000|1.000000|N/A|N/A|
|reverse (first 20% holdout)|first_submission_equivalent|0.054343|0.005154|N/A|N/A|
|reverse (first 20% holdout)|second_pair_gate|0.054343|0.005154|N/A|N/A|
|middle (40%-60% holdout)|first_submission_equivalent|0.054940|0.005208|N/A|N/A|
|middle (40%-60% holdout)|second_pair_gate|0.054940|0.005208|N/A|N/A|

RA/ASafeR：N/A（本次缓存没有可复用的 raw EEG attack 数据）。

JSON：`D:\WORK\BCIC2026\BrainPrint\research\second_submission\freeze_evaluation.json`
