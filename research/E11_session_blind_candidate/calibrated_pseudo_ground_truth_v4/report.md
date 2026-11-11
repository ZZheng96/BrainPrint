# calibrated pseudo ground truth v4 — evidence-first scenarios

## 结论与边界

本目录生成 3 个证据优先情景，不是真实 Test 标签，也不声称恢复真实官方分。v4 不再对官方冲突行随机翻转标签，也不把对抗样本当成独立标签；依据赛题规则，对抗样本若为已知身份，其真实标签仍应为 1–20。由于类别来源和攻击身份不可观测，`category_hypothesis` 只表示假设，不是真值。

## 情景规则

- `main_evidence_consensus`：保留官方共识；冲突行使用五个候选的多数标签。
- `conservative_reject_ambiguous`：候选中至少两个拒识时设为 0，其余使用候选多数。
- `liberal_known_consensus`：冲突行优先采用非零候选的多数；没有非零证据才设为 0。

三种规则均不使用 session；逐行证据、置信等级、候选集合和原因在每个情景的 `pseudo_groups.csv` 中。

## 评估

`scenario_metrics.csv` 使用官方公式，并对未观测的 RA 显式报告 RA=0 与 RA=1 条件分；同时对 `ASafeR=(0.8, 0.95, 1.0)`、`U=(0.8, 1.0)` 做敏感性分析。官方提交的 69.17/67.94 仅用于计算锚点所需的隐含 RA，记录在 `anchor_implied_ra.csv`，不把隐含 RA 当作真实标签证据。

AUC：不可计算（Test 无真实标签，候选 CSV 无连续概率输出）。F1：是相对伪答案的审计 F1，不是官方 Test F1。`candidate_mean_std_summary.csv` 给出当前条件分和 F1 汇总，不能替代官方成绩。

## 完整性

输入 Sample 均已对齐；正式提交 SHA-256 运行前后保持不变；样本数为 22251。标签 0/1–20 和对抗样本语义依据 `赛题说明.md`。
