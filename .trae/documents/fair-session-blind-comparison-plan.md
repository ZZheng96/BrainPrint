# 公平补做无 Session 方案对比计划

## Context

当前只能证明 Riemannian recenter + Ridge 在 Train 内部伪未知验证中表现较好，但它尚未生成 Test 候选，不能与已有 `session_blind_candidate` 和两次提交方案公平比较。此前直接给出官方分数区间缺乏实际 Test 证据，需改为基于统一协议和真实数据的可复现实验。

Test 没有真实 Label，因此补实验可以得到真实的 Train 内部 C/U/AUC/F1、无标签 Test 拒识与预测分布、历史提交差异，但不能得到 Test AUC/F1/C/U/RA/ASafeR 或官方成绩。第一次提交的 69.17 作为唯一官方锚点；第二次方案若无官方回执，只标记为候选，不伪造成绩。

## 推荐实施方案

1. **冻结输入和保护版本**
   - 固定 `BrainPrint_cache`、Train/Test 元数据、特征缓存、Sample 顺序、类别映射、随机种子和代码版本。
   - 保留第一次提交、第二次候选及所有用户已有修改，不覆盖正式答案。
   - 新实验全部写入独立目录：`research/ea_riemannian_openset/fair_comparison/<run_id>/`。

2. **统一 Train 内部验证协议**
   - 对 `full_none`、`full_target_zscore`、`full_ea`、`full_riemannian_recenter` 使用相同的 5 折留被试和 forward/reverse 时间划分。
   - 统一 Ridge `alpha=100`、full 4 秒协方差输入、`max_logit` 置信度和已知置信度 5% 分位拒识规则。
   - 每个对齐方法使用自身变换后的合法 OOF/内层分数校准阈值，禁止复用普通 Ridge 阈值。
   - 保存逐折 C、U、unknown AUROC、macro-F1、unknown-F1、detection-macro-F1、阈值、最差折和逐样本预测。

3. **补齐 Riemannian Test candidate**
   - 扩展现有候选生成入口，复用 `SPDRecenter`、`fit_recentered_model`、`predict_recentered_full_scores`，不重写 SPD 算法。
   - Train 只读取 `subject/task`，Test 只读取 `sample/task`；不读取 session、trial、path、fs，不使用 `Session1 → 0`。
   - Riemannian 源参考只用 Train；如使用 Test 无标签目标参考，按 task 使用整个 Test 批次，并在报告中标记为 transductive 对齐。
   - 为每个方法生成 CSV、连续分数、阈值来源、manifest 和 Test 拒识/标签分布诊断。

4. **公平比较两次提交方案**
   - 第一次提交作为历史官方基线：保留真实成绩 69.17，不把其 Session 规则移除后冒充原方案。
   - 第二次候选单列为历史 Session-dependent 候选，记录 Session2 对齐、`Session1 → 0`、配对门控和改动比例；无官方回执则官方分未知。
   - 与无 Session 候选比较时，明确它们不是同一输入条件，不将 Session-dependent 方法的结果归因于 Riemannian 或 EEG-only 改进。
   - 比较 Test 上可真实观测的拒识率、各任务拒识率、预测身份计数、两两一致率、改变行数和 CSV 合法性；不把这些指标当成 Test 性能。

5. **评分与报告口径**
   - 只有同时具备 C、RA、ASafeR、U 才按官方公式计算内部模拟攻击协议分。
   - 无对抗指标的 Riemannian/开放集结果只报告 C/U/AUC/F1，不硬算官方分；如给出 `RA=ASafeR=1` 仅能作为理论上限，不能作为预估分。
   - 报告明确写出 Test AUC/F1/官方分不可计算，禁止用 69.17 做比例映射或输出无依据预测区间。
   - 每份实验报告使用 Markdown，并包含当前 AUC 和 F1；同时列出数据、阈值、GPU、随机种子、产物路径和限制。

## 关键文件

- `src/ea_riemannian_openset/run.py`
- `src/ea_riemannian_openset/core.py`
- `src/session_blind_candidate/run.py`
- `src/open_set_improvement/core.py`
- `research/ea_riemannian_openset/report.md`
- `research/session_blind_candidate/report.md`
- `research/second_submission/report.md`
- `.trae/rules/实验.md`

## 验证方式

- CUDA 不可用时直接停止，不回退 CPU。
- 运行现有回归测试：
  `py -m pytest -q test/test_ea_riemannian_openset.py test/test_ea_riemannian_e2.py test/test_session_blind_candidate.py test/test_open_set_improvement.py`
- 运行统一 forward/reverse 内部验证，核对每个方法的逐折指标和阈值来源。
- 生成 Riemannian 与 baseline 的 Test candidates，核对 22,251 行、Sample 顺序、Label 范围、无 Session 元数据读取和 manifest。
- 生成最终 Markdown 报告，明确哪些结果是 Train 内部指标、哪些是 Test 无标签诊断、哪些是唯一官方成绩。
