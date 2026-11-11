# 前两次提交方案对齐 raw-PGD 冻结验收

同一五折 subject holdout、同一 TaskSpecificModel、同一攻击缓存；未使用 Test 标签或 Test waveform_pairs。

|攻击|方案|C|RA|ASafeR|U|官方分|unknown AUC|unknown F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|untargeted_snr20|first_submission_equivalent|0.922769|0.000000|0.440858|0.956653|47.176165|0.980257|0.947288|
|untargeted_snr20|second_pair_gate|0.922769|0.000000|0.439450|0.956653|47.123371|0.980257|0.947288|
|untargeted_snr30|first_submission_equivalent|0.922769|0.009965|0.788237|0.956653|58.407289|0.980257|0.947288|
|untargeted_snr30|second_pair_gate|0.922769|0.010074|0.787587|0.956653|58.397282|0.980257|0.947288|
|targeted_snr20|first_submission_equivalent|0.922769|0.000000|0.007149|0.956653|11.148206|0.980257|0.947288|
|targeted_snr20|second_pair_gate|0.922769|0.000000|0.007149|0.956653|11.148206|0.980257|0.947288|

验收：未通过

逐门禁结果：
- score_gain_at_least_2: 未通过
- A_gain_at_least_0.05: 未通过
- RA_at_least_0.60_and_gain_0.03: 未通过
- ASafeR_at_least_0.90: 未通过
- C_at_least_0.90_and_drop_at_most_0.02: 通过
- U_at_least_0.85_and_drop_at_most_0.02: 通过
- unknown_f1_drop_at_most_0.01: 通过
- AUROC_drop_at_most_0.005: 通过
- four_of_five_folds_not_worse: 未通过
- worst_fold_score_drop_at_most_1: 通过
- untargeted_snr30_ASafeR_at_least_0.90: 未通过
- untargeted_snr30_A_drop_at_most_0.01: 通过
- targeted_snr20_ASafeR_at_least_0.90: 未通过
- targeted_snr20_A_drop_at_most_0.01: 通过

JSON：`D:\WORK\BCIC2026\BrainPrint\research\E10_second_submission\aligned_evaluation.json`
攻击缓存完全同一：`True`
