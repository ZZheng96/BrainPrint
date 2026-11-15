# E155 task-conditioned calibration

设备：cuda；E65 raw correlation-logSPD；LOSO k 扫描：[1, 3, 6, 12]。

Train-only proxy 选中：**task_identity, k=12, 接受率=0.800**；proxy=0.8632。
Train OOF AUC=0.9112；Train OOF F1=0.7662。
LOSO rejection=0.9140；Test S1 rejection=0.9753；Test S2 acceptance=0.2868；**W=0.6157**；Test AUC/F1=N/A。

阈值规则预注册为 global、task 分位数、task+identity shrinkage（identity 向 task 收缩 0.5），仅由 Train OOF known 与留身份 unknown proxy 扫描；Test 分数未参与参数选择。

Test 元数据没有 identity 列，因此 task_identity 阈值在实际部署时退化为 task 阈值。若 Test task 组成或各 task 接受率与 Train 不同，task-conditioned 阈值会改变总体 operating point，不能公平解释为全局阈值的无条件改进。
