# 决赛方案综合总结与 11 项指标决策计划

## Context

项目已有 E191、E247、E248、E249、E250、E251 等多轮实验。现有结果分别优化了闭集识别、严格 LOSO、拒识分数、漂移校准和 hubness，但没有一个研究候选在统一、可比、合规的 11 项指标下形成完整证据。与此同时，E200 冻结的 `final/team_hymn` 仍是唯一已完成提交接口、注册流程和 Test CSV 审计的正式方案，并且其冻结审计中 S1 拒识率和 S2 接受率达到约 0.70。

本任务的目标是把历史经验整理成正式 Markdown 文档，并建立一个不混用实验口径的综合决策表，最终给出适合决赛的整体方案。核心原则是：优先保证真实决赛场景和提交合规性；Train-only 指标用于判断泛化风险；Test S1/S2 只作为冻结后的工程审计，不用于回溯调参；任何官方 Test AUC/F1 等无真值指标均标为 N/A。

## Recommended approach

### 1. 新建综合总结文档

创建一个新的研究文档目录，避免覆盖 E191/E200/E249/E250/E251：

- `research/E252_final_decision/REPORT.md`
- `research/E252_final_decision/results/`

`REPORT.md` 统一整理：

1. 数据集和决赛接口边界；
2. E248–E251 的理论经验与已证伪方向；
3. E191/E200 正式冻结方案的优势和局限；
4. 每轮实验的模型、协议、AUC、F1、S1/S2 和攻击结果；
5. 不同实验之间不可直接比较的指标；
6. 11 项指标定义、目标、证据侧、来源实验和可比性；
7. Test 官方指标的 N/A 说明；
8. 最终方案选择、替代方案和不采用原因；
9. 决赛提交前的接口、压缩包、注册和复现检查；
10. 后续若继续研究，必须满足的预注册和不泄漏要求。

### 2. 建立统一 11 项指标矩阵

在 `results/` 保存机器可读的统一表，例如 `metrics_matrix.csv` 和 `decision.json`。每项指标至少记录：

- 指标编号与定义；
- 目标值；
- 候选方案；
- 数值或 N/A；
- Train/Test audit 数据侧；
- 具体协议；
- 来源实验和文件；
- 是否可与其他候选直接比较；
- 是否通过；
- 是否存在 Test 选择污染；
- 备注。

禁止把 E191 的攻击/门控数字、E249/E251 的 strict LOSO 数字、E200 的 Test 审计数字直接拼成某一个候选的“11 项成绩”。每个候选必须标明哪些指标是同一工作点真实测得，哪些只是历史参考。

### 3. 统一候选分层，而不是只按单项最高值选型

决策表分为三层：

- **正式可提交候选**：E200/E191 冻结的 `final/team_hymn`，重点验证接口、注册新用户、CSV、模型 reload 和现有 S1/S2 审计；
- **Train-only 研究候选**：E249 C2、E250 C2、E251 lambda=0，作为严格 LOSO、AUC、F1 和未知拒识的研究参考；
- **不可采用候选**：违反 Test-only 规则、口径不一致、攻击结果未重测或无法提供完整证据的方案。

最终推荐规则：先按决赛真实场景和硬约束筛选，再在可提交候选中比较完整 11 项证据；不能因为某个研究候选的 Train AUC 更高就替换 Test S2 已达到门槛的冻结方案。

### 4. 对正式方案做一次统一证据核验

只读既有产物并补充必要的离线审计，不重新查看 Test 进行调参：

- 核对 E200/E191 的冻结模型、阈值、报告和 `final/team_hymn` 接口一致；
- 核对 11 项中已有指标的定义和来源；
- 对缺失或口径不一致的指标标为 `N/A/不可比`，不从其他实验借数；
- 运行提交包 self-test、reload、预测形状、NaN、label 范围和新用户 enroll 检查；
- 如需要补测，补测只使用既定 Train 协议，并保存模型与报告；
- 不修改 `final/`，除非完整统一证据显示新候选在决赛约束下整体优于冻结方案。

### 5. 输出最终决策

在报告中明确给出：

- 推荐提交：`final/team_hymn` / `subwin6_pca96_k1 @ a=0.90`；
- 选择理由：真实 Test S1/S2 冻结审计、完整提交接口、注册流程、可复现性和合规性；
- E249/E250/E251 不替换正式方案的证据；
- 当前 11 项中哪些已经通过、哪些无法由同一工作点证明；
- 不把 Test 审计写成官方 AUC/F1；
- 决赛前必须执行的最终 checklist。

如用户随后批准该计划，再执行文档创建、统一矩阵生成和正式方案核验；未经批准不运行新实验、不修改 `final/`、不提交 Git。

## Critical files to reuse

- `research/E248_metric_backfill/REPORT.md`
- `research/E249_drift_nap/REPORT.md`
- `research/E250_score_split_calibration/REPORT.md`
- `research/E251_explicit_hubness_gate/REPORT.md`
- `research/E191_fix_gate1_gate7/report.md`
- `research/E200_final_freeze/report.md`
- `final/team_hymn/model.py`
- `final/team_hymn/README.md`
- `final/技术方案说明.md`
- `project_memory.md`
- `research/研发目标与量化验收指标.md`
- `BrainPrint_cache/e201_final_selftest/final_rules.txt`

## Verification

完成执行后应验证：

1. `REPORT.md` 明确包含 AUC、F1 和完整 11 项指标表；
2. `metrics_matrix.csv` 中每个数字都能追溯到同一来源文件和协议；
3. Test 官方 AUC/F1/C/RA/ASafeR/U/Score 均按无真值规则标为 N/A；
4. `final/team_hymn` 的 `load_model`、`predict_batch`、`enroll` 通过接口和形状检查；
5. 提交包不包含数据、标签、日志缓存或 `__pycache__`；
6. 报告明确说明正式方案是否替换，且默认不替换；
7. 结果写入新的 E252 目录，旧实验和 `final/` 不被覆盖。
