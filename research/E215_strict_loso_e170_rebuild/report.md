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

## 固定 acceptance 结果

- acceptance 全部预先固定于 Train/LOSO 参考分数；Test 仅用于冻结模型后的 S1/S2 描述，不据 Test 挑工作点。
- LOSO rejection F1 为拒识二分类F1；identity_macro_f1 是身份识别宏F1，非官方Test F1。

|config|acceptance|LOSO rejection micro|LOSO rejection macro|known acceptance micro|known acceptance macro|rejection F1|S1 rejection|S2 acceptance|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E65_PCA96_k12|0.85|0.740113|0.742041|0.874448|0.874450|0.354311|0.911638641608968|0.41695285669684673|
|E65_PCA96_k12|0.88|0.647545|0.650998|0.900867|0.900867|0.358556|0.8644906033630069|0.49370381933603913|
|E65_PCA96_k12|0.90|0.558887|0.563519|0.918388|0.918387|0.346727|0.8265743488295417|0.5507336871682798|
|E65_PCA96_k12|0.92|0.471969|0.477852|0.937327|0.937326|0.336185|0.7665677546983185|0.6156728067436777|
|E65_PCA96_k12|0.94|0.355932|0.363130|0.957181|0.957183|0.299561|0.6927134849983515|0.6870121760849204|
|E65_PCA96_k12|0.96|0.240765|0.247941|0.974130|0.974132|0.242095|0.6188592152983844|0.7600166510563013|
|E65_PCA96_k12|0.50|0.997392|0.997183|0.484183|0.484192|0.169068|1.0|0.04074305338744927|
|E65_PCA96_k12|0.60|0.987397|0.987097|0.604268|0.604275|0.207627|0.9993405868776789|0.08091372671453845|
|E65_PCA96_k12|0.70|0.960017|0.959530|0.721494|0.721498|0.264426|0.9891196834817013|0.1626079716932043|
|E65_PCA96_k12|0.75|0.921773|0.921750|0.779775|0.779781|0.301063|0.9772502472799208|0.23040899157040276|
|E65_PCA96_k12|0.80|0.859192|0.859342|0.831469|0.831468|0.337823|0.962743158588856|0.3110105109792902|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.85|0.807040|0.810518|0.841922|0.841924|0.332976|0.9221892515661062|0.4047247372255177|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.88|0.700565|0.705566|0.872916|0.872920|0.334324|0.8644906033630069|0.49354771568321365|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.90|0.619296|0.624289|0.896750|0.896752|0.335898|0.8203099241674909|0.5548444166926839|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.92|0.524989|0.529802|0.917038|0.917038|0.321954|0.7649192218925157|0.6144760120720159|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.94|0.426336|0.429458|0.935108|0.935108|0.298795|0.7082096933728981|0.6748360911645332|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.96|0.297262|0.300185|0.957433|0.957433|0.252728|0.6379821958456974|0.7572588198563847|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.50|0.999565|0.999580|0.416226|0.416234|0.152696|1.0|0.02336351337287959|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.60|0.997827|0.997852|0.541275|0.541288|0.186295|1.0|0.05416796753044021|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.70|0.986093|0.986027|0.662504|0.662507|0.234681|0.9967029343883943|0.12191695285669685|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.75|0.969144|0.969324|0.724948|0.724957|0.269198|0.9904385097263435|0.18180872099073786|
|E65_PCA64_logbp_PCA32_blocknorm_jointPCA96_k6|0.80|0.919600|0.920632|0.788056|0.788064|0.308621|0.9729640619848335|0.2773441565199292|

完整 per-subject/task 明细与模型/阈值引用行见 `results.json`；官方 Test AUC/F1 未完成（无公开标签）。
实际运行：`.venv\Scripts\python.exe research\E215_strict_loso_e170_rebuild\run_e215.py`（CUDA，完整20折；先以 `E215_FOLDS=1` 验证后再运行完整实验）。
未完成项：无法在无Test身份标签时计算官方Test AUC/F1；本报告不宣称11项全达标。
运行耗时：8.9s。
