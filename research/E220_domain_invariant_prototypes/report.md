# E220 domain-invariant prototypes

严格修正 E219：每个 outer LOSO fold 的 robust task-wise center/scale 和原型只来自 enrolled outer_train 的 fit_fraction≤0.6；held identity 不参与。阈值仅由 enrolled train 的 inner OOF query block 分数确定。

|候选|LOSO AUC|拒识F1|unknown拒识|known接受|identity-correct-but-rejected|
|---|---:|---:|---:|---:|---:|
|baseline_fixed_oof|0.744688|0.094319|0.086307|0.985772|0.013085|
|task_query_norm|0.949018|0.147350|0.088723|0.997232|0.000755|
|robust_mahalanobis|0.674735|0.084048|0.074190|0.995198|0.002858|
|branch_rank_fusion|0.716773|0.099296|0.111905|0.961391|0.025688|
|branch_rank_fusion_w025|0.721526|0.092891|0.112642|0.950961|0.034403|
|branch_rank_fusion_w075|0.687462|0.100815|0.112520|0.966745|0.020380|

冻结候选：`branch_rank_fusion_w025`；Test S1 rejection=0.000000，S2 acceptance=1.000000。官方 Test AUC/F1=N/A（无身份标签）。

达标判定：LOSO unknown rejection>0.90、S2>0.70、S1≥0.70；本轮：False。

E65 与 E146 logbp 独立 robust 标准化；branch_rank_fusion 分别计算距离后固定权重 rank/min-max fusion；其余候选覆盖 raw distance、task-wise query normalization、特征值白化 Mahalanobis、多原型距离。
