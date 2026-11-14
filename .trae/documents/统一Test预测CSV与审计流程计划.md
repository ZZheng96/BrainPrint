# 统一 Test 预测 CSV 与审计流程计划

## Summary
为每个真正兼容 BrainPrint 项目 Test 的冻结实验生成完整 `Sample,Label` CSV，并在 CSV 写出后统一执行 Test 工程审计与候选比较。Test 没有真实身份/攻击标签，因此真实 Test AUC、F1、C、RA、ASafeR、U、Official Score 均记录为 `N/A`；S1/S2 只表示写出后的 Session 分组拒识审计。

## Current State Analysis

- 项目 Test 模板为 `Brainprint_Competition/test.csv`，字段为 `Sample,Label`，Sample 顺序必须保留，Label 合法范围为 `0..20`。
- `BrainPrint_cache/test_metadata.csv` 提供 Test 的 `Sample`、`path`、`session`、`task` 等字段，共 22251 行。Session 只能在预测 CSV 写出后读取并审计。
- `research/E64_e58_test_audit/run_e64.py` 已有冻结后 GPU 推理、CSV 写出和写出后 Session 审计的完整链路；E52、E42 也有可复用实现。
- `research/E36_pure_eeg/run_e36e_inference.py` 已定义 18 通道、1200 点、Session-blind 的冻结候选校验和推理边界，但当前模板候选不能直接作为冻结候选。
- `research/E42_submission_candidates/run_e42.py` 可一次推理生成两个候选 CSV，适合验证多候选比较。
- `vis/schemes.py` 已有标签一致率、分歧行、标签转移矩阵、Session/task 分组比较的参考逻辑，但其旧代理指标不能作为 Test 真值指标。
- 外部 E74–E84 使用 22 通道 BCIC IV-2a，不接入项目 Test CSV 流程。

## Proposed Changes

### 1. 新增统一 Test CSV writer/audit 模块

建议新建一个实验专用子目录，例如 `research/E85_test_csv_audit/`，避免修改历史实验和其他会话目录。新增模块负责：

- 读取并校验官方 Test 模板与 Test metadata：
  - 行数一致；
  - `Sample` 唯一且顺序一致；
  - metadata 的路径均位于 `Test/`；
  - 不读取模板中的 Label 作为推理输入或真值。
- 接收已冻结候选输出的 `labels`，只允许 `0..20`、非空、长度等于 Test 行数。
- 生成 `outputs/prediction_<variant>.csv`，严格字段顺序 `Sample,Label`。
- CSV 写出后重新读取 CSV，校验文件行数、Sample 顺序、Sample 唯一性和 Label 范围；只有校验通过后才加载 Session/task 做审计。
- 计算每个候选的：
  - overall `n`、拒识数、拒识率、接受率；
  - S1 `n`、拒识率、接受率、是否满足拒识率 `>=0.95`；
  - S2 `n`、拒识率、接受率、是否满足拒识率 `<=0.02`；
  - task 分组和 Session×task 分组拒识/接受率；
  - Label 0..20 预测分布。
- 在结果 JSON 中固定写入：
  - `test_metrics.AUC/F1/C/RA/ASafeR/U/Official_Score = N/A（无Test真实标签）`；
  - `session_used_by_inference = false`；
  - `test_labels_used = false`；
  - S1/S2 的定义说明。
- 保存 CSV、metadata、模型、配置和代码来源的 SHA256，以及生成时间和推理规则，便于复现和防止误把不同冻结候选混用。

### 2. 接入兼容项目 Test 的冻结候选

优先接入已有 18 通道候选，不复用外部 22 通道模型：

- E64：从 `research/E64_e58_test_audit/run_e64.py` 复用冻结模型和 GPU 推理逻辑，生成一个正式候选 CSV；模型和来源指纹必须通过既有 manifest 校验。
- E42/E36E：仅在 `frozen_candidate.json` 存在且通过其完整冻结哈希、Train 校准和候选门禁校验后接入；否则只保留为不可运行候选，不用模板文件冒充冻结模型。
- 如需批量比较，使用 E42 的双候选结构生成独立 CSV，但所有候选必须来自 Train-only 冻结规则，不能用 Test S1/S2 反向调阈值或选模型。
- 每个候选的 CSV 和审计结果放在对应实验 `outputs/` 下，不覆盖历史 CSV。

### 3. 新增跨候选比较

在 E85 中加入统一比较函数，输入两个已经通过完整性校验的 CSV，输出：

- 逐行一致率和分歧行数；
- A 标签到 B 标签的 `21×21` 转移矩阵；
- `A=0,B!=0`、`A!=0,B=0`、两者均非零但身份不同的数量；
- overall、Session、task、Session×task 分组的一致率与拒识率差异；
- 每个候选 Label 分布差异。

比较结果只用于候选行为审计和报告，不把无真值的 Test 代理量命名为 AUC/F1/C/RA/ASafeR/U。

### 4. 报告与实验产物

在 `research/E85_test_csv_audit/outputs/` 保存：

- 每个候选的 `prediction_<variant>.csv`；
- `results.json`，包含完整性校验、S1/S2 审计、分组统计、哈希和 Test 真值指标 N/A 声明；
- `comparison.json`，包含候选逐行比较；
- `report.md`，明确列出 AUC/F1 为 N/A 及原因，并分别报告 S1 拒识率与 S2 拒识率/接受率；
- `config.json`，记录候选来源、推理规则、数据路径和禁止使用 Session/Sample/标签先验的约束。

不修改 `.trae/rules/实验.md`、E71 目录、原始数据、缓存和其他会话改动。

## Assumptions & Decisions

- S1 定义为 Session1 拒识率，门限为 `>=95%`；S2 定义为 Session2 拒识率，门限为 `<=2%`，并同时报告 S2 接受率 `>=98%`。
- Session/task 仅在 CSV 已写出并重新读取后用于离线审计，绝不进入样本预测、阈值拟合或候选选择。
- 官方 `test.csv` 中现存 Label 不是真实 Test 标签，不能用于计算任何 Test 真值指标。
- Test 真值指标统一写 `N/A`，不使用旧 `vis/schemes.py` 中基于参考方案的 proxy C/RA/ASafeR/U/Score 充当真实结果。
- 只有项目 Train 训练、18 通道输入、冻结校验通过的候选可生成正式项目 Test CSV；E74–E84 外部模型不接入。
- 训练好的模型不重复训练；E85 只负责加载冻结模型、推理、写 CSV、审计和比较。

## Verification Steps

1. 对每个接入候选运行其冻结校验和 Test 推理，确认使用 CUDA、推理函数参数不含 Session/Sample/标签。
2. 检查每个 CSV：22251 行、字段为 `Sample,Label`、Sample 与官方模板顺序完全一致、Label 全在 `0..20`。
3. 删除或篡改一行 CSV 做负向测试，确认完整性校验失败；恢复后再运行审计。
4. 确认 Session 审计发生在 CSV 写出/重读之后，结果包含 S1/S2 拒识率、接受率、门禁布尔值、task 与 Session×task 统计。
5. 确认 `results.json` 和 `report.md` 将 Test AUC/F1/C/RA/ASafeR/U/Official Score 全部写为 N/A，并明确无真实标签原因。
6. 对至少两个候选运行比较，验证一致率、分歧行数、21×21 转移矩阵及分组差异与手工抽样一致。
7. 运行 Python 编译检查和已有相关单测；只提交本任务新增/修改文件，排除规则文件、E71、缓存和原始数据。
