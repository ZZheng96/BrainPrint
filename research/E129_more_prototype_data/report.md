# E129 More prototype data plus out-of-fold threshold

E122 k1 只用前 60% 的 Train trial 建原型，剩余 40% 未使用。本实验把原型数据扩到 80%/90%/100%，并用 5 折 out-of-fold 分数标定阈值。打分仍是 PCA-96 空间的最近原型欧氏距离。

| 变体 | 每身份原型 trial | 标定集 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |
|---|---:|---|---:|---:|---:|---:|---:|
| E0_ref_fit60_thr6080 | 335 | 0.6-0.8 block | 0.7658 | 0.7916 | 0.5823 | 0.6261 | 0.8579 |
| E1_fit80_thrOOF | 450 | 5-fold out-of-fold | 0.7631 | 0.7906 | 0.5772 | 0.6261 | 0.8579 |
| E2_fit80_thr6080 | 450 | 0.6-0.8 block | 0.7631 | 0.8994 | 0.4170 | nan | nan |
| E3_fit90_thrOOF | 508 | 5-fold out-of-fold | 0.7643 | 0.7903 | 0.5834 | nan | nan |
| E4_fit100_thrOOF | 565 | 5-fold out-of-fold | 0.7655 | 0.7933 | 0.5872 | nan | nan |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
