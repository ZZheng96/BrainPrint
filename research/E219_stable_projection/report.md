# E219 训练折内稳定联合投影实验

输入仅为 E65(855)+E146 logbp(90)，每个 outer fold 只使用 outer-train 的 fit_fraction≤0.6。投影候选和 lambda 网格预先固定；Test 仅在候选冻结后读取。

|候选|AUC|rejection F1|known acceptance|LOSO unknown rejection|
|---|---:|---:|---:|---:|
|fisher_l0.1_d32|0.8939|0.9231|0.8714|0.6902|
|fisher_l0.1_d64|0.8877|0.9225|0.8711|0.6723|
|fisher_l1.0_d32|0.8979|0.9220|0.8683|0.7153|
|fisher_l1.0_d64|0.8923|0.9213|0.8677|0.6991|
|fisher_l10.0_d32|0.9080|0.9204|0.8624|0.7798|
|fisher_l10.0_d64|0.9038|0.9179|0.8584|0.7732|
|fisher_l100.0_d32|0.9093|0.9211|0.8636|0.7835|
|fisher_l100.0_d64|0.9042|0.9169|0.8562|0.7846|
|pca_whiten_l0_d32|0.7837|0.9317|0.9090|0.1992|
|pca_whiten_l0_d64|0.7929|0.9333|0.9120|0.1992|

冻结候选：`fisher_l100.0_d64`（仅按 Train LOSO 预先定义的 unknown rejection、再按 acceptance 选择）。

Test S1 rejection=0.871744，S2 acceptance=0.434957。官方 Test 无身份标签，因此 AUC/F1=N/A。

达标条件：LOSO unknown rejection>0.90、S2>0.70、S1≥0.70；本轮冻结结果是否达标：False。

稳定性正则采用与 E218 同等能量随机对照的特征变化；完整 PGD 成本过高，已明确限制，未冒充 PGD。
