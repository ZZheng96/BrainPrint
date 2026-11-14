# E126 Supervised reduction (LDA) and per-task PCA

PCA 是无监督的（保留最大方差），但身份信号不一定在最大方差子空间。多类 LDA 直接最大化类间/类内散度比，应得到更可分的空间。全部只用 Train 拟合块，未用 Test。

| 变体 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |
|---|---:|---:|---:|---:|---:|
| REF_PCA96 | 0.7658 | 0.7916 | 0.5823 | 0.6261 | 0.8579 |
| LDA19 | 0.7199 | 0.9914 | 0.0733 | nan | nan |
| LDA32 | 0.7199 | 0.9914 | 0.0733 | nan | nan |
| LDA48 | 0.7199 | 0.9914 | 0.0733 | nan | nan |
| LDA64 | 0.7199 | 0.9914 | 0.0733 | 1.0000 | 0.9986 |
| LDA96 | 0.7199 | 0.9914 | 0.0733 | nan | nan |
| LDA128 | 0.7199 | 0.9914 | 0.0733 | nan | nan |
| PCA96_LDA19 | 0.7264 | 0.9799 | 0.1064 | nan | nan |
| PCA96_LDA32 | 0.7264 | 0.9799 | 0.1064 | nan | nan |
| PCA96_LDA48 | 0.7264 | 0.9799 | 0.1064 | nan | nan |
| PCA96_LDA64 | 0.7264 | 0.9799 | 0.1064 | nan | nan |
| PERTASK_PCA96 | 0.7560 | 0.7679 | 0.5971 | 0.6583 | 0.8671 |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
