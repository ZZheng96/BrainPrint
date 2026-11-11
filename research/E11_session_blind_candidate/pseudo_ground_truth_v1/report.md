# pseudo ground truth v1

- 生成时间：Python运行；样本数 **22251**。
- 输入：两次正式提交、`candidate_logcov.csv`、`candidate_corrspd.csv`；正式提交文件未修改。
- 两次正式标签共同的 **{int(x.same_official.sum())}** 行直接保留；不同的 **{int((~x.same_official).sum())}** 行按候选一致性、拒识状态与确定性并列规则构造。
- 这不是真实标签恢复：共同标签仅表示两次提交相同，discordant 行为低置信伪标签；不得用于声称官方真值。

## 候选相对伪答案
### logcov
- accepted accuracy（伪答案非0子集，n=19218）：**0.513113**
- overall accuracy：**0.495393**
- macro-F1：**0.481772**
- 伪真值0且预测0：0.383119；伪真值0误接受数：1871；非0被拒识率：0.373452
- confusion matrix：见 `metrics.json`，标签顺序 0–20。

### corrspd
- accepted accuracy（伪答案非0子集，n=19218）：**0.472890**
- overall accuracy：**0.457642**
- macro-F1：**0.446458**
- 伪真值0且预测0：0.361029；伪真值0误接受数：1938；非0被拒识率：0.440056
- confusion matrix：见 `metrics.json`，标签顺序 0–20。

## 官方分数边界
赛题官方分数需要真实样本分组与真实标签：Clean Accuracy、RA、ASafeR、U。当前只有伪答案，且没有可核验的真实 Clean/Adversarial/Unknown 分组以及真实 RA/ASafeR/U，因此不计算、不声称官方分。
若强行把伪答案的 0 当未知、非0 当已知并把候选0视作拒识，只能得到“clean 伪答案条件分/上界”，但这依赖未经验证的分组假设，不能替代官方分；本次仅报告通用 accepted/overall/macro-F1 与拒识统计。

## 限制
- 任务元数据仅用于核验分布和审计；规则没有把 task/session 当作真实标签。
- 伪答案会继承两次正式提交的系统性错误；discordant 行尤其不确定。
- 当前候选无概率分数，无法计算可靠 AUC。
