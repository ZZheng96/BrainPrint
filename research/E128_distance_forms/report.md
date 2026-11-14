# E128 Distance-form search in the PCA-96 space

E127 发现度量方式影响巨大：最近原型欧氏距离 AUC 0.766，余弦只有 0.539。本实验在 PCA-96 空间里系统搜索距离形态。全部只用 Train 拟合块。

| 变体 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |
|---|---:|---:|---:|---:|---:|
| neg_d1 | 0.7658 | 0.7916 | 0.5823 | 0.6261 | 0.8579 |
| ratio_d2_d1 | 0.6108 | 0.8216 | 0.3546 | 0.8427 | 0.9189 |
| margin_ratio | 0.6108 | 0.8216 | 0.3546 | 0.8427 | 0.9189 |
| neg_d1_rel | 0.4960 | 0.6142 | 0.4183 | 0.7131 | 0.8818 |
| neg_d1_mahal | 0.7490 | 0.7870 | 0.5737 | nan | nan |
| fuse_d1_ratio | 0.7341 | 0.8345 | 0.4451 | nan | nan |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
