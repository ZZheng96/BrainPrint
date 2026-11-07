# 无 Session 依赖的 EEG-only 候选计划

## Context
当前正式链路依赖 Session1→0、Session2 统计对齐或 Session2 后处理，不能作为可迁移的最终方案。用户要求继续改进，并明确最终推理不得读取或依赖 session。目标是在不覆盖第一次提交和现有未提交改动的前提下，建立严格 session-blind 的纯 EEG 候选，并评估是否存在支持官方分数超过 75 的证据；若证据不足，必须明确停止，不把内部代理分当作官方成绩。

## 推荐实现
1. 在 `src/session_blind_candidate/` 建立独立入口和核心逻辑，在 `research/session_blind_candidate/` 保存结果与候选；不修改旧实验目录和正式提交。
2. 使用项目内缓存的 EEG 特征，训练元数据只读取 `subject/task`，测试元数据只读取 `sample/task`；禁止读取 `session`、`trial`、路径和质量字段。使用 GPU 的 `TaskSpecificModel/TorchRidge`，不复用含 Session2 统计的旧 zscore/grouped/guarded 分数缓存。
3. 复用 [core.py](file:///d:/WORK/BCIC2026/BrainPrint/src/open_set_improvement/core.py) 的纯 EEG 模型与置信度计算，以及 [core.py](file:///d:/WORK/BCIC2026/BrainPrint/src/open_set_calibration/core.py) 的任务内目标对齐和提交构建逻辑。先比较四个候选：`full_none`、`base_none`、`full_target_zscore`、`base_target_zscore`；不使用 Session 规则、Sinkhorn、配对或旧正式链路后处理。
4. 使用固定五折留被试嵌套验证，只在内层选择阈值与配置，外层仅验收。报告 AUC、macro-F1、unknown-F1、C、U、每折结果、最差折和阈值来源；Test 无标签时明确 AUC/F1/C/U/RA/ASafeR/官方分不可计算。
5. 生成独立 CSV 和 manifest，检查 Sample 唯一、模板顺序、行数、标签范围 0–20、拒识比例、实际读取字段和 `session_used=false`。不复制到 `commit/answer/`，除非后续候选通过门槛且用户明确要求。
6. 新增回归测试：删除或扰动伪 session 列不改变分数与标签；输入行重排仍按 Sample 恢复模板顺序；生产加载器使用白名单；任务统计互不串扰。

## 验收与决策
- session-invariance 和列删除测试全部通过。
- 候选在五折外层相对 `full_none/base_none` 不出现 AUC、macro-F1、unknown-F1 的实质下降；C/U 不因极端拒识换取，最差折不崩溃。
- Test 候选拒识比例仅作风险诊断，不作为真实 U；不能据此调阈值。
- 只有内部外层指标稳定、无明显 Test 分布异常且严格 session-blind 时，才进入人工提交审查。没有官方多点校准时，不宣称预估超过 75；若证据不足，保留第一次 69.17 保护版本。

## 验证
1. 运行 `pytest -q test/test_session_blind_candidate.py`。
2. 先执行 GPU smoke，再执行五折嵌套验证和候选生成。
3. 检查 report、results、manifest、CSV 格式和工作树，确保未覆盖用户已有文件。
4. 必要时运行相关测试，不直接使用 `git add .`、`git add -A`，不自动提交或推送。

## 关键文件
- `src/open_set_improvement/core.py`
- `src/open_set_calibration/core.py`
- `src/session_blind_candidate/run.py`
- `src/session_blind_candidate/invariance.py`
- `test/test_session_blind_candidate.py`
- `research/session_blind_candidate/report.md`
