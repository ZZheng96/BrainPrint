# E107 proxy-vs-Test gap audit

汇总各候选的留一身份（LOSO）代理 AUC 与项目 Test 事后诊断 AUC。

| 候选 | LOSO 代理 AUC | Test 诊断 AUC | 差距 |
|---|---:|---:|---:|
| A_spd_ratio | 0.9144 | 0.5856 | 0.3288 |
| B_raw_cosine | 0.9353 | nan | nan |
| C_raw_ratio | 0.8641 | nan | nan |
| D_inv_ratio | 0.5564 | 0.5305 | 0.0259 |
| E_mean_fusion | 0.9289 | 0.5308 | 0.3981 |

## 冻结 Test 审计结果

| 候选 | S1 拒识率 | S2 接受率 |
|---|---:|---:|
| E102_spd_ratio | 0.8427 | 0.2873 |
| E103_external_cal | 0.9291 | 0.1258 |
| E104_inv_ratio | 0.4263 | 0.6150 |
| E106_mean_fusion | 0.5345 | 0.5176 |

## 结论

LOSO 代理 AUC 普遍在 0.86–0.94，但对应候选在 Test 上的真实区分度只有 0.53–0.59。
代理高估约 0.30–0.40 AUC。这说明留一身份代理无法预测跨 Session 泛化，
任何仅凭 LOSO 代理判断的候选都不可信。

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
