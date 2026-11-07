# Session-blind EEG-only 候选报告

## 目标与约束

本实验最终推理不读取 `session`，不使用 `Session1 → 0`，不按 session 设置阈值、类别规则或后处理。训练仅读取 `subject/task`，测试仅读取 `sample/task`。候选使用项目内 EEG 特征缓存和 GPU 纯 EEG Ridge 模型，不复用可能含 Session2 统计的旧分数缓存。

## 最新结果：五方法独立 blocked-OOF 阈值（2026-11-07）

复现命令（仓库根目录）：`py -3.11 research/session_blind_candidate/generate_five.py`。

实际使用 CUDA / NVIDIA GeForce RTX 3080 Laptop GPU、PyTorch 2.7.1+cu118。新入口从既有训练 OOF 表仅读取 `subject/task/fold`，复用五折分配而不复用旧分数；每折重新训练，分别生成五方法的 held-out 分数，各自以 `max_logit` 的 5% 分位设阈值。验证折的对齐参考只使用该折无标签特征；最终使用全部训练样本拟合，Test 参考仅按 task 从完整无标签批次估计。Riemannian 直接复用 `fit_recentered_model/predict_recentered_full_scores`。全流程不读取 `session/trial/path/fs` 元数据列。

| 方法 | 独立 OOF 阈值 | Test 拒识数量 | Test 拒识率 | Test AUC | Test F1 |
|---|---:|---:|---:|---|---|
| full_none | 0.2271697313 | 17,629 | 79.2279% | 不可计算 | 不可计算 |
| base_none | 0.2039187118 | 17,645 | 79.2998% | 不可计算 | 不可计算 |
| full_target_zscore | 0.2228513718 | 21,111 | 94.8766% | 不可计算 | 不可计算 |
| base_target_zscore | 0.2029163465 | 21,218 | 95.3575% | 不可计算 | 不可计算 |
| full_riemannian_recenter | 0.2432883948 | 18,300 | 82.2435% | 不可计算 | 不可计算 |

五份 CSV 位于 `five_candidates/candidate_*.csv`，生成流程已逐份回读核验：22,251 行、Sample 唯一、与官方 test.csv 的 Sample 顺序完全一致、Label 为 0–20 整数。`five_candidates/manifest.json` 记录阈值、完整标签计数、软件版本、参考诊断和 SHA256；`blocked_oof_scores.npz` 保存五方法 OOF 分数、fold 和类别顺序。两份 `赞美诗_第1次.csv` / `赞美诗_第2次.csv` 前后 SHA256 一致，未修改，也未提交 git。

当前 Test AUC/F1 因无标签不可计算；本次仅做阈值校准，没有生成内部开集 AUC/F1 或预估官方分数。拒识率只是输出统计，不代表未知身份拒识准确率。

以下保留旧四候选记录，其中共用阈值问题已由上述独立入口修正；旧入口与旧产物不作为本次五候选结果。

## 历史候选

比较 `full_none`、`base_none`、`full_target_zscore`、`base_target_zscore`。目标对齐仅按 task 使用当前无标签测试批次特征统计；它不读取 session，但仍需单独评估分布迁移风险。

## 结果

四个候选均生成 22,251 行，且入口记录 `session_used=false`、`session_columns_read=false`。使用训练 blocked-OOF `max_logit` 的 5% 分位阈值 `0.2270835549`，当前候选拒识率如下：

| 候选 | 拒识数量 | 拒识率 | Test AUC | Test F1 |
|---|---:|---:|---:|---:|
| full_none | 17,629 | 79.23% | 不可计算 | 不可计算 |
| base_none | 17,870 | 80.31% | 不可计算 | 不可计算 |
| full_target_zscore | 21,138 | 95.00% | 不可计算 | 不可计算 |
| base_target_zscore | 21,343 | 95.92% | 不可计算 | 不可计算 |

Test AUC、F1 和官方分数不可计算，因为 Test 没有标签和官方反馈。实际读取列为训练 `subject/task`、测试 `sample/task`；没有读取 `session`。

内部 AUC/F1 若有结果，必须明确属于训练集内部外层验证，不是 Test 成绩，也不是官方分数预测。本轮候选生成入口未将 Test 标签用于任何决策。

## 验收判断

严格 session-invariance 测试通过后，仍需用五折留被试嵌套验证确认 AUC、macro-F1、unknown-F1、C/U 和最差折稳定性。本候选若只有高内部 AUC/F1、但 Test 拒识比例异常，不能据此宣称预估超过 75，也不应自动替换第一次提交。

第一次官方成绩 69.17 仍是唯一真实锚点。由于本实验与正式提交链路不同，且没有第二个官方反馈点，任何 `>75` 只能作为待验证假设，不能写成可靠预测。
