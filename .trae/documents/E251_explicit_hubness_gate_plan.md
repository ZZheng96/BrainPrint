# E251 显式 Hubness 拒识门控实验计划

## Context

E249 的 Train-only 漂移正则化和 E250 的 margin/cohort/hubness logistic 校准均未解决真实 Test S2：E250 C2 的 Test S2 接受率约 0.442，S1/S2 分数分离没有改善。当前瓶颈更像少数注册原型吸附未知样本，而不是身份排序不足。本轮只验证一个可归因改动：保持 top-1 身份排序不变，显式惩罚预测原型的 hubness，仅改变拒识分数。

## 推荐方案

新建 `research/E251_explicit_hubness_gate`，复用 E250/E247 的严格嵌套 identity LOSO、GPU 特征空间、身份等权 FAR10 阈值、模型保存/重载和指标计算。

拒识分数固定为：`s_reject = top1_cosine - lambda * standardized_predicted_prototype_hubness`。固定候选 `lambda ∈ {0, 0.02, 0.05, 0.10}`，只在 inner LOSO 中按预先固定规则选择；身份预测始终是未经校正的 top-1 cosine。hubness 图、标准化参数和阈值均只使用当前 inner/outer fit 身份，不能读取 Test。

选择规则：优先要求 unknown rejection 不低于 C0，在满足者中最大化 known acceptance；无候选满足时选 C0。候选保留还需满足：AUC/F1 有稳定正向趋势、已知接受下降不超过 1 个百分点、未知拒识不下降、最差身份不恶化，且 reload 一致。

## 严格评估流程

1. `--validate-only`：断言 outer/inner held-out identity 不进入拟合、原型、hubness、校准或阈值；session/Test 不参与。
2. `--outer-subject 1`：完成单折、保存模型/划分/结果，验证 top-1 身份排序不变和 reload 分数一致。
3. `--all-folds --resume`：完成 20 折严格嵌套 LOSO，输出 AUC、拒识 F1、known acceptance、unknown rejection、最差身份和身份级配对 CI。
4. 对 C0 与预先固定候选按 E195 口径补齐 #1–#8：#7/#8 必须使用修正后的攻击方向、SNR30 主口径和 SNR20 参考口径；不得借用旧模型攻击数字。
5. 冻结 Train-only 候选、阈值和模型后，只做一次 Test S1/S2 描述性审计；#9/#10 标注为 session-based audit，Test 官方 AUC/F1/C/RA/ASafeR/U/Score 写 N/A。
6. 生成 `REPORT.md`，固定列出全部 #1–#11、目标、数值、PASS/FAIL/NOT_EVALUATED、来源与限制；任何缺失指标不得省略。

## 关键文件

- 新增：`research/E251_explicit_hubness_gate/run_e251.py`
- 新增：`research/E251_explicit_hubness_gate/config.json`
- 新增：`research/E251_explicit_hubness_gate/gates_attack.py`
- 新增：`research/E251_explicit_hubness_gate/audit_test.py`
- 新增：`research/E251_explicit_hubness_gate/REPORT.md`
- 复用：`research/E250_score_split_calibration/run_e250.py`
- 复用：`research/E247_train_temporal_loso/run_e247.py`
- 参考门禁：`research/E195_eleven_gates_loso90/run_e195.py`

不修改 `final/`、E249 或既有正式提交包；模型和结果保存到 E251 独立目录，便于索引和复现。

## 失败判据

若只改善 Train 平均值但身份级区间跨 0、未知拒识下降、已知接受下降超过 1 个百分点、最差身份恶化、攻击指标低于 C0、严格 LOSO 或 reload 断言失败，则判定 E251 失败，保留现有正式方案，不把 Test S2 结果用于二次调参或选择。

## Verification

```powershell
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\run_e251.py --validate-only
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\run_e251.py --outer-subject 1
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\run_e251.py --all-folds --resume
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\gates_attack.py
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\audit_test.py
```

最终报告必须包含 11 项指标；其中 #9/#10 明确是一次性 Test 审计，#7/#8 报告逐攻击条件和最差值。