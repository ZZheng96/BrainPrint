# Enrollment 模板与跨会话稳健身份模型实施计划

## Context

当前 Session2 高拒识不是阈值单独造成的问题，而是训练仅含 Session1，现有 Ridge/手工融合分数在跨会话后置信度塌缩。历史提交的 Session2=0 拒识率依赖 `session` 后处理，不能作为模型能力。本次改进目标是实际改进身份表征与分类器：不读取 Test `session` 做训练、阈值、模型选择或后处理，使用 EEG 特征构建注册身份模板，并通过训练集内部验证选择稳定的决策规则。

## Recommended approach

1. 新建研究目录 `research/enrollment_template_session_robust_v1`，不覆盖历史实验和正式提交。
2. 复用 `BrainPrint_cache/train_features.npy`、`train_window2_features.npy`、对应 metadata 以及 `COV_COLUMNS`；窗口按 trial 重排，使用全局训练 StandardScaler。
3. 按 `(subject, task)` 构建 enrollment 表征：两窗口平均、窗口一致性、稳健中心/medoid 或少量多模板；不混合不同 task 的模板。
4. 以模板余弦相似度、类内标准化距离、top-1/top-2 margin 和两窗口一致性形成身份分数；现有 `TorchRidge`/`TorchNeighbor` 仅作候选生成或对照，不以当前 `fused` 作为最终主决策。
5. 用严格的 subject 外层五折开放集评估：外层留出身份作为伪未知，注册身份后段 trial 作为已知评估；所有模板、候选、阈值只在外层训练集及内层校准中产生。补充 forward/reverse/middle 时间切分和训练内部特征扰动消融，但不把它们称为真实 Session2 验证。
6. 逐项比较 prototype-only、prototype+class-distance、prototype+window-consistency，并保留冻结 Ridge/base 基线。只有在外层 AUC、身份 macro-F1、unknown-F1/U 和已知接受率未发生不可接受退化时才生成 Test 候选。
7. 全量 Train 构建最终模板后，只用 Test EEG 特征、window 特征和 task 推理；Sample 仅用于输出顺序。候选生成完成后才按 Session 做审计，Test AUC/F1 记为不可计算。

## Critical files

- `src/open_set_improvement/core.py`：复用 `COV_COLUMNS`、`TorchRidge`、`TorchNeighbor` 及 GPU 推理接口。
- `src/open_set_improvement/run_nested.py`：参考严格外层/内层 subject holdout 结构，避免使用测试样本校准。
- `src/session_blind_candidate/run.py`：参考 Session 不参与推理的边界和候选清单记录方式。
- `research/enrollment_template_session_robust_v1/run.py`：新实验主脚本。
- `research/enrollment_template_session_robust_v1/report.md`：记录实际 AUC、F1、C、U、known acceptance 及限制。

## Verification

- 使用 CUDA 运行外层五折与消融，确认没有静默 CPU fallback。
- 对每折记录 AUC、identity macro-F1、detection macro-F1、unknown-F1、U、C、已知接受率和按 task 指标。
- 检查改变/删除 Test `session` 后，冻结推理输出完全一致；Session 只能出现在生成后审计。
- 验证所有 CSV 与 `Brainprint_Competition/test.csv` 的 Sample 顺序一致，标签范围为 0–20。
- Test 报告只写无标签分布审计，不伪造 Test AUC/F1；正式提交文件不覆盖，候选单独保存。
