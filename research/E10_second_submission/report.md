# 第二次提交候选实验报告：保守配对门控

## 结论

已生成一个独立、格式正确的第二次提交候选：

- [candidate_conservative_pair_gate.csv](file:///d:/WORK/BCIC2026/BrainPrint/research/E10_second_submission/candidate_conservative_pair_gate.csv)
- [candidate_conservative_pair_gate.json](file:///d:/WORK/BCIC2026/BrainPrint/research/E10_second_submission/candidate_conservative_pair_gate.json)

候选没有覆盖第一次正式提交。第一次提交 `commit/answer/赞美诗_第1次.csv` 仍是唯一具有官方成绩的保护版本，成绩为 69.17/100。

本候选复用第一次任务特异链路，仅对 Session2 的高相似波形配对结果做保守门控：当直接预测与配对融合预测不一致时，只保留低直接置信度（top-1/top-2 margin < 0.10）、组件至少 2 行且组件直接预测纯度至少 0.75 的变化；其他不满足条件的配对变化回退到直接预测。Session1 全部继承标签 0。

## 生成链路

- 使用已有 `BrainPrint_cache` 特征、分数和 `waveform_pairs.csv`。
- 先运行 `commit/code/predict_task_specific.py`，固定参数：`--pair-fusion --balanced-ec-imagery --guarded-ec --tag second_submission_baseline`。
- 再由 `conservative_pair_gate.py` 生成门控候选。
- 没有重跑 PGD、没有使用 Test 真实标签、没有修改正式答案。

## CSV 验收

| 检查项 | 结果 |
|---|---:|
| 行数 | 22,251 |
| Sample 顺序 | 通过 |
| Sample 唯一 | 通过 |
| Label 非空且范围 0–20 | 通过 |
| Session1 行数 | 3,033 |
| Session1 改变行数 | 0 |
| 相对第一次提交改变行数 | 961 |
| 相对第一次提交改变比例 | 4.3189%（低于 5% 上限） |
| 保留的配对门控行 | 102 |
| 回退到直接预测的弱配对行 | 1,193 |

候选标签 0 共 3,033 行，与第一次提交保持一致。注册身份标签的计数发生少量变化，没有出现标签全部坍缩为单一类别的情况。

## AUC 与 F1

Test 没有公开真实标签，因此本实验的 **Test AUC 不可计算，Test F1 不可计算**，不能把候选 CSV 的本地统计称为官方成绩。

作为方法参考，既有内部嵌套验证记录为：

- unknown AUROC（AUC）：0.9803；
- unknown F1：0.9472。

这些 AUC/F1 是内部验证参考，不是 Test 真实成绩，也不能保证第二次提交分数提升。

## 风险与提交建议

该候选相对第一次提交只改变 4.3189% 的行，且所有 Session1 标签保持不变，属于低风险、可解释候选；但由于 Test 标签未知，无法证明它优于 69.17。若必须进行第二次官方尝试，建议使用该候选；若提交风险必须由验证结果确认，则应继续保留第一次提交作为回退版本。
