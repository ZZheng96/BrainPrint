# calibrated pseudo ground truth v3 — synthetic calibration scenarios

## 结论与边界

本目录生成 **16 份 synthetic calibration scenarios**，不是官方真实 Test 标签，也不声称是真实官方预测。每份场景包含 `pseudo_answer.csv` 与 `pseudo_groups.csv`；共同正式观测行保持硬约束，攻击类别只在两次正式提交不一致的不可观测行随机分配。

固定条件代理为 `ASafeR=0.95`、`U=1.0`，严格使用：`Score=100*C^0.40*(0.7*RA+0.3*ASafeR)^0.35*U^0.25`。每个场景的正式提交锚点都通过隐含 RA 微调精确复现目标。

## 锚点复现

- 第一次正式提交目标：69.17；最大绝对误差：1.4210854715202e-14；严格通过 `<1e-6`：True。
- 第二次正式提交目标：67.94；最大绝对误差：1.4210854715202e-14；严格通过 `<1e-6`：True。

## 候选场景结果

`scenario_metrics.csv` 给出每个场景的 C、RA、ASafeR、U、分数/范围、AUC 和相对伪答案 F1。`candidate_mean_std_summary.csv` 给出场景条件平均分、标准差和范围；其中 logcov/corrSPD 没有攻击样本标签，不能隐藏补值，使用明确的 `RA∈[0,1]` 条件范围。AUC 均为 null（没有概率分数），F1 是相对场景伪答案的审计指标，不是官方 F1。

场景平均只代表在这些合成条件下的模型平均，不解决真实标签、攻击分组、RA、ASafeR、U 的不可识别性。

## 完整性

运行时核验两次正式 CSV SHA-256 未改变；输入与运行后摘要见 `assumptions.json`。每个场景参数（seed、类别计数、硬约束规则）见 `scenario_metrics.csv` 和场景 `pseudo_groups.csv`。

## AUC/F1

AUC：null（无连续概率输出）；F1：各场景逐方法记录于 `scenario_metrics.csv`，汇总均值记录于 `candidate_mean_std_summary.csv`。
