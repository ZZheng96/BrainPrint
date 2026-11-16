# E218 稳定特征成分诊断实验

- 设备：`cuda`；诊断子集：160 行；特征维度 945（E65 855 + E146 log-band-power 90）。
- 严格 outer LOSO：每折成分筛选仅使用 outer-train fit rows；held identity 未参与筛选或阈值。
- PGD：eps=0.005/0.01/0.02；随机噪声按配对扰动幅度生成；PGD 为固定外训质心的特征距离最大化诊断。

|方案|AUC|F1|known acceptance|unknown rejection|身份正确但拒绝|
|---|---:|---:|---:|---:|---:|
|all|0.672533|0.295723|0.174342|0.937500|0.622368|
|stable_same|0.644737|0.513169|0.349342|0.850000|0.373684|
|stable_low|0.592434|0.592456|0.430263|0.737500|0.329605|
|random_same|0.598355|0.566233|0.401316|0.712500|0.355263|

- 官方 Test AUC/F1：N/A（无身份标签，未用 Test 选型）。
- 冻结全 Train Test 的 S1 rejection、S2 acceptance：N/A；本诊断未将无标签 Test 作为评估样本。
- 真实限制：原始数据仅提供 Train 注册身份标签，未知身份采用严格 LOSO held identity 代理；完整 PGD 受计算量限制采用每个 subject-task 两行诊断子集。
- 产物：`results.json` 保存每折指标和成分账本；`cache/feature_pairs.npz` 保存配对特征；`models/e218_last_fold_state.npz` 保存最近折筛选状态。
