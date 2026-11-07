# 重建更合理的伪答案 v4 计划

## Context

现有 v3 伪答案可以精确复现两次官方总分，但其核心类别分配和随机翻转缺少数据证据，导致候选方法的伪答案 F1/条件分可能严重偏离真实 Test。赛题规则明确：标签 0 是未知身份；对抗样本仍保留原始身份标签 1–20，预测 0 只是安全拒识，不能把对抗样本本身当成独立标签。因此需要最后重建一套“证据优先、假设显式”的伪答案，而不是继续随机生成并用隐含 RA 闭环拟合。

## Recommended approach

1. 在新目录 `research/session_blind_candidate/calibrated_pseudo_ground_truth_v4/` 建立可审计产物，不覆盖 v1/v2/v3。
2. 读取两次正式提交、五个无 Session 候选、logcov/corrSPD 原始候选和 Test 元数据，仅使用 Sample 对齐、任务、标签一致性、模型共识、置信度/拒识状态等已有证据；保留两次正式提交完全相同的行作为高置信观测标签。
3. 对正式提交不一致的行不再随机翻转标签。按证据优先级确定代理标签：正式提交共识 > 多候选身份共识 > 高置信候选身份；若证据只支持拒识，则标签设为 0；无法区分的行保留为“不可识别集合”，分别生成少量可解释情景，而不是伪装成确定真值。
4. 按赛题语义单独记录三类假设：`clean_known`、`adversarial_known`、`unknown`。对抗样本标签必须是其原始身份 1–20，不能生成“对抗类别标签”；无法从提交 CSV 识别攻击身份的行标记为 `category_unobserved`，不把随机类别当作事实。
5. 生成三类结果：主情景（最小额外假设）、保守情景（不确定行偏向拒识/unknown）、宽松情景（不确定行偏向已知身份），并保存每个样本的 `evidence_source`、`confidence`、`ambiguity_set` 和规则原因。
6. 使用官方公式计算候选的 C、F1 及在显式 `RA/ASafeR/U` 假设下的分数范围。两次官方提交仅做锚点检查：报告它们在每个情景下的 C 和需要的隐含 RA，但不把反解 RA 当成真实标签证据。固定代理值只作为敏感性分析，不作为唯一结论。
7. 输出 `pseudo_answer.csv`、`pseudo_groups.csv`、`scenario_metrics.csv`、`candidate_mean_std_summary.csv`、`assumptions.json`、`report.md`；报告必须明确 AUC 在没有连续 Test 真值/概率输出时不可计算，F1 是伪答案审计指标，不是官方 F1，并写明当前 AUC/F1 状态。

## Critical files and reuse

- 复用 [v3/run.py](file:///d:/WORK/BCIC2026/BrainPrint/research/session_blind_candidate/calibrated_pseudo_ground_truth_v3/run.py) 的 CSV 对齐、SHA-256 和官方公式校验结构，但不复用其随机类别分配。
- 复用 [five_method_eval/run.py](file:///d:/WORK/BCIC2026/BrainPrint/research/session_blind_candidate/calibrated_pseudo_ground_truth_v3/five_method_eval/run.py) 的候选逐场景评估结构，并扩展为 v4 多情景。
- 依据 [赛题说明.md](file:///d:/WORK/BCIC2026/BrainPrint/赛题说明.md) 的标签和对抗评分语义编写假设。

## Verification

- 核验所有输入 Sample 顺序、22,251 行、Label 范围和正式提交 SHA-256 不变。
- 核验共同正式标签全部被保留；不确定行的标签来源可逐行追溯。
- 核验每个情景的官方公式计算、C/F1、类别计数和敏感性范围。
- 运行 v4 脚本并检查报告；不再声称重建得到真实官方 Test 标签或真实官方分数。
