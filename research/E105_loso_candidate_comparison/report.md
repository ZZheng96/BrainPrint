# E105 leave-one-identity-out candidate comparison

19 个身份注册，留出 1 个身份作为未注册者。未访问项目 Test。

目标：known acceptance ≥ 0.85，unknown rejection ≥ 0.8。

| 候选 | AUC 均值 | AUC 标准差 | 最佳 known acc | 最佳 unknown rej | 有可行点的折数 |
|---|---:|---:|---:|---:|---:|
| A_spd_ratio | 0.9144 | 0.0320 | 0.7994 | 0.9391 | 16/20 |
| B_raw_cosine | 0.9353 | 0.0495 | 0.8623 | 0.8989 | 16/20 |
| C_raw_ratio | 0.8641 | 0.0413 | 0.7224 | 0.9200 | 2/20 |
| D_inv_ratio | 0.5564 | 0.0595 | 0.4029 | 0.7305 | 0/20 |
| E_mean_fusion | 0.9289 | 0.0268 | 0.8209 | 0.9454 | 17/20 |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。
