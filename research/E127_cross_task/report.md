# E127 Cross-task identity aggregation

当前打分只用同一任务的 proto。注册者在同一 Session 里做过 4 个任务，其 trial 也应接近本人在其他任务的原型；未注册者没有这种跨任务一致性。全部只用 Train 拟合块。

| 变体 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |
|---|---:|---:|---:|---:|---:|
| pertask_pca96/own_task_max | 0.5389 | 0.6436 | 0.3779 | 0.9374 | 0.9522 |
| pertask_pca96/cross_task_max | 0.5092 | 0.5714 | 0.4277 | 0.5708 | 0.8345 |
| pertask_pca96/relative_fit | 0.5266 | 0.6205 | 0.3898 | nan | nan |
| pertask_pca96/both_task_gap | 0.5598 | 0.5806 | 0.5159 | 0.9466 | 0.9543 |
| pertask_pca96/fuse_own_cross | 0.5226 | 0.5984 | 0.4056 | nan | nan |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
