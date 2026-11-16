# E203 任务分层 LOSO Gate 报告

## 目的
在不读取 Test 分布用于调参的前提下，利用 Train OOF 已知样本和严格 LOSO 未知样本，为四个 task 分别校准拒识阈值，检验能否修复 E202 的跨会话 S2 接受率问题。

## 方法
- 输入：E200 `subwin6_pca96_k1` 的冻结 Train OOF / LOSO / Test 分数；分数越大越像已注册身份。
- 每个 task 的阈值只用 Train：known 使用 `0.6 < frac <= 0.8` 的 OOF 样本，unknown 使用其他留出身份的 LOSO 未知样本。
- 阈值选择要求 LOSO unknown rejection 至少 0.90，并在满足条件的阈值中尽量保留已知接受率。
- Test session 只在所有阈值冻结后读取并审计。

## 结果
| 指标 | E203 |
|---|---:|
| Train/LOSO unknown rejection mean | 0.9029 |
| Test S1 rejection | 0.9331 |
| Test S2 acceptance | 0.3646 |
| Test AUC | N/A（无 Test 身份标签） |
| Test F1 | N/A（无 Test 身份标签） |

task 阈值：EC=-16.3885，EO=-16.5533，Imagery=-16.99997，SSVEP=-17.1577。

## 结论
E203 达到 S1≥0.70 和 LOSO>0.90，但 S2 仅 0.3646，未达到 0.70。任务分层阈值只是把 gate 推向更严格的一侧，不能消除 Train/LOSO 与 Test S2 的排序错位。因此该候选不作为新冻结方案。

历史对照显示，E200 原始全局阈值的 S1=0.7135、S2=0.7069，是当前最接近三项最低约束的单一冻结基线；E202 的模型 A+B 虽有 LOSO=0.9847，却把 S2 降至0.2176。后续必须改进跨会话表征或使用已证明的 E200 子窗口骨干，而不是继续单独提高 gate 阈值。
