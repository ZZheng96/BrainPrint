# Session-blind 跨会话校准方案

## Context

第一次提交官方成绩为 69.17，但依赖 Session 规则和配对融合；第二次为 67.94；第三次 raw-PGD 候选为 26.10。现有纯 EEG 候选在 Test 上大量拒识，尤其是已注册的 Session2，说明训练 Session1 的固定 OOF 阈值不能直接迁移到 Test。新方案必须不读取 Test `session`、Test 标签或 Session 分组规则，优先恢复跨会话已注册用户，同时保留未知身份拒识能力。

## 推荐方案

实现“身份模型 + 单样本稳健化 + 漂移/质量诊断 + 条件拒识”的 Session-blind 管线：

1. 保留 `TaskSpecificModel` 的 `full` 和双窗口 `base` 分支作为身份模型，继续按 `task` 路由；不把 Sinkhorn、Test 配对传播或目标批次 z-score 作为主路径。
2. 新增训练期拟合、逐样本应用的稳健特征变换；保留 direct 分支和 robust 分支，不使用 Test 全量统计。
3. 计算 margin、energy、full/base 身份一致性、双窗口一致性、训练域稳健距离和特征质量等诊断量。漂移量只用于调整拒识阈值，不单独把样本判为未知。
4. 用嵌套留被试验证拟合 2–3 个质量/漂移区间的条件阈值；只有在多个身份证据一致时允许适度放宽阈值，高漂移且分支不一致时仍拒识。
5. raw-PGD 只作为第二阶段训练增强消融，不直接用于 Test 端阈值或规则。

## 代码改动

- 新增 `src/session_blind_candidate/shift_calibration.py`：实现稳健变换、shift/quality 特征、direct/robust 一致性、条件阈值拟合与应用、参数序列化。
- 修改 `src/session_blind_candidate/run.py`：保持元数据白名单（Train 仅 `subject/task`，Test 仅 `sample/task`），加入嵌套 OOF 校准、外层评估、全量训练推理、候选 CSV 和 manifest 输出；不使用 `target_zscore` 主路径。
- 最小扩展 `src/open_set_improvement/core.py`：提供逐样本分数诊断、full/base 一致性和可选稳健输入接口，保持现有 API 与 GPU Ridge 计算。
- 若已有测试目录，则补充 `test/test_session_blind_candidate.py`；否则只在现有测试结构中添加对应测试，不为一次性脚本额外堆积文件。
- 新建独立实验目录 `research/session_blind_shift_calibration/`，保存 `report.md`、`metrics.json`、`candidate_manifest.json`、候选 CSV 和冻结参数，不覆盖第一次提交或已有候选。

## 验证协议

- 固定 5 折留被试；内层只用训练折拟合模型、稳健变换和条件阈值，外层冻结评估。
- Session 字段仅可作为离线审计变量，不能进入推理、校准或模型选择；报告明确记录 `session_used_for_inference/calibration/model_selection=false`。
- Train 内部报告 C、RA、ASafeR、A、U、unknown AUC、unknown F1、macro-F1、接受率及官方公式代理分；所有结果标记为内部代理而非 Test 官方成绩。
- Test 只报告样本数、任务分布、标签计数、拒识率、各分支一致率、条件放宽触发率、Sample 顺序和标签合法性；不计算或声称 Test AUC/F1。

## 通过门槛与回退

- 相对当前纯 EEG baseline：外层 C、U、unknown AUC/F1、macro-F1 不得出现明显下降；至少 4/5 折代理分不下降，最差折下降不超过 1 分。
- 跨域代理数据上，已知接受率应提升，且未知误接受不能明显增加；不能以单纯全局放宽阈值换取低拒识率。
- 若条件放宽损害未知检测，关闭放宽，仅保留稳健分支；若稳健化损害身份 C，回退 `full_none/base_none`；任何候选未通过冻结门槛则不替换第一次提交。

## 执行顺序

1. 先读取实验规则和现有入口/测试结构，确认复用接口。
2. 在独立目录实现 direct/base 的 OOF 诊断与稳健校准，不加入 PGD。
3. 运行 GPU 五折实验，生成 AUC/F1、官方代理指标和报告。
4. 仅在第一阶段通过门槛后生成无 Session Test 候选 CSV。
5. 核验 CSV、Session-blind 白名单和候选 manifest；未通过则保留第一次 69.17 作为正式保护版本。
