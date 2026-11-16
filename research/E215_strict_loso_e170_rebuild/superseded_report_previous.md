# E215 strict LOSO E170 rebuild（修正版）

- 设备：`cuda`。完整20折；outer held subject 完全不参与任何内层拟合或校准。
- 修复：prototype 最近邻使用 `argmin`；每个非空 inner query block 排除该 block 后重新拟合 task standardization、PCA、prototype。
- 内层实际块数/样本范围、每折模型状态、行 ID、分数和最终部署模型已保存。
- ties 使用平均秩 AUC；LOSO AUC 报告 per-fold AUC 均值，不混合不同 outer fold 的异尺度分数。
- 旧 `results.json/report.md` 已保存为 `superseded_results_previous.json` / `superseded_report_previous.md`；旧结果不可用于结论。
- 官方 Test AUC/F1：N/A（Test 无公开身份标签）。

## E65_PCA96_k12

- Strict LOSO per-fold AUC mean：**0.910173**；开放集拒识 F1（acceptance=0.90）：**0.346727**；identity macro-F1（per-subject identity_macro_f1，非官方F1）：**0.987757**；final inner OOF：**0.987288**。
- acceptance=0.90：LOSO rejection micro/macro=0.558887/0.563519；known acceptance micro/macro=0.918388/0.918387。
- 冻结全量 Train 后 Test S1 rejection=0.826574；S2 acceptance=0.550734。
- inner query blocks：[4]；每 outer fold query 样本数范围：[6323, 6431]。

## E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6

- Strict LOSO per-fold AUC mean：**0.904961**；开放集拒识 F1（acceptance=0.90）：**0.335898**；identity macro-F1（per-subject identity_macro_f1，非官方F1）：**0.985737**；final inner OOF：**0.985491**。
- acceptance=0.90：LOSO rejection micro/macro=0.619296/0.624289；known acceptance micro/macro=0.896750/0.896752。
- 冻结全量 Train 后 Test S1 rejection=0.820310；S2 acceptance=0.554844。
- inner query blocks：[4]；每 outer fold query 样本数范围：[6323, 6431]。

## 固定 acceptance 诊断

- 0.50/0.60/0.70/0.75/0.80 仅作为预先固定的 Train 阈值诊断，不从 Test 选点；不代表表征改善。

|config|acceptance|LOSO rejection F1|known acceptance|S1 rejection|S2 acceptance|
|---|---:|---:|---:|---:|---:|
|E65_PCA96_k12|0.50|0.169068|0.484183|1.0|0.04074305338744927|
|E65_PCA96_k12|0.60|0.207627|0.604268|0.9993405868776789|0.08091372671453845|
|E65_PCA96_k12|0.70|0.264426|0.721494|0.9891196834817013|0.1626079716932043|
|E65_PCA96_k12|0.75|0.301063|0.779775|0.9772502472799208|0.23040899157040276|
|E65_PCA96_k12|0.80|0.337823|0.831469|0.962743158588856|0.3110105109792902|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.50|0.152696|0.416226|1.0|0.02336351337287959|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.60|0.186295|0.541275|1.0|0.05416796753044021|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.70|0.234681|0.662504|0.9967029343883943|0.12191695285669685|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.75|0.269198|0.724948|0.9904385097263435|0.18180872099073786|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.80|0.308621|0.788056|0.9729640619848335|0.2773441565199292|

完整 acceptance grid 及 per-subject/task 明细见 `results.json`。
运行耗时：8.9s。
