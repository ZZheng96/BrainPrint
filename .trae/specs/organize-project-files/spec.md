# 项目文件整理 Spec

## Why
项目根目录与 `research/` 目录已积累大量实验产物（322 个文件、约 112MB），其中包含重复文件、可重建的中间产物、命名混乱的根级散落文件，导致索引困难、检索成本高，违反 `.trae/rules/实验.md` 中"做好文件管理，要便于索引，避免垃圾文件积累"的要求。

## What Changes
- 清理根目录散落文件：`赞美诗_第3次.csv` 与 `commit/answer/赞美诗_第3次.csv` 内容完全相同（MD5 一致），删除根目录副本。
- 清理可重建的缓存与工具产物：`__pycache__/`、`.pytest_cache/`、`.screenshots/`（空目录）、`.trae-html-share-packages/`（已 gitignore 的打包产物）。
- 整理 `research/` 根级散落文件：将 18 个 `submission_*.csv`、68 个 `*.json` 诊断/基线结果按类别归入子目录，保留 3 个核心 md 文档在根级作为索引入口。
- 整理 `research/` 内的重复/冗余产物：`calibrated_pseudo_ground_truth_v3/scenarios/scenario_XX/` 中 16 份 `pseudo_answer.csv` 存在重复（4 组 MD5 相同），`v4/scenarios/main_evidence_consensus/` 与 `v4/pseudo_answer.csv` 完全一致。
- 新增 `research/README.md` 作为实验索引（目录 → 方法 → 结论 → 报告链接），并新增根目录 `README.md` 说明项目结构与复现入口。
- 更新 `.gitignore`，覆盖新增的缓存/中间产物模式。
- **BREAKING**：`research/` 根级文件路径变更，所有引用旧路径的脚本（如 `research/session_blind_candidate/calibrated_pseudo_ground_truth_v3/run.py` 等）需同步更新路径常量。

## Impact
- Affected specs: 无（首次建立 spec）
- Affected code:
  - `research/` 根级 89 个文件（18 csv + 68 json + 3 md）
  - `research/session_blind_candidate/calibrated_pseudo_ground_truth_v3/`、`v4/` 场景目录
  - 引用 `research/*.csv`、`research/*.json` 的脚本：`src/*.py`、`research/**/run.py`
  - `.gitignore`、根目录 `README.md`、`research/README.md`

## ADDED Requirements

### Requirement: 根目录整洁
项目根目录 SHALL 只保留：数据集目录、源码目录、实验目录、提交包目录、赛题说明文档、启动脚本、`.gitignore`、`README.md`。

#### Scenario: 根目录无重复提交文件
- **WHEN** 检查项目根目录
- **THEN** 不存在 `赞美诗_第3次.csv`（已存在于 `commit/answer/`）

#### Scenario: 根目录无缓存产物
- **WHEN** 检查项目根目录
- **THEN** 不存在 `__pycache__/`、`.pytest_cache/`、`.screenshots/`、`.trae-html-share-packages/`

### Requirement: research 目录可索引
`research/` SHALL 提供 `README.md` 索引，列出每个子目录的实验方法、关键结论（含 AUC/F1）、对应报告路径。

#### Scenario: 通过索引定位实验
- **WHEN** 开发者需要查找某个实验的报告
- **THEN** 可在 `research/README.md` 中按方法名或目录名定位到 `report.md` 路径

### Requirement: research 根级文件分类归档
`research/` 根级散落的 `submission_*.csv` 与诊断 `*.json` SHALL 归入语义化子目录，不再平铺在根级。

#### Scenario: 根级只保留索引文档
- **WHEN** 列出 `research/` 根级文件
- **THEN** 仅包含 `README.md`、`brainprint_research_and_experiments.md`、`对话交接_实验历程与下一步.md`、`问题梳理.md` 及子目录

### Requirement: 重复产物去重
内容完全相同的文件 SHALL 只保留一份，其余删除或改为引用。

#### Scenario: 场景伪答案去重
- **WHEN** 检查 `calibrated_pseudo_ground_truth_v3/scenarios/`
- **THEN** 不存在 MD5 完全相同的 `pseudo_answer.csv` 副本

### Requirement: 路径引用一致性
文件移动后，所有引用旧路径的脚本 SHALL 更新为新路径，且脚本可正常运行。

#### Scenario: 脚本路径可解析
- **WHEN** 运行任一 `research/**/run.py`
- **THEN** 其引用的输入文件路径均存在，不出现 `FileNotFoundError`

## MODIFIED Requirements

### Requirement: .gitignore 覆盖范围
`.gitignore` SHALL 额外忽略 `__pycache__/`（已有）、`.pytest_cache/`、`.screenshots/`、`*.pyc`（已有），并保持 `BrainPrint_cache/`、`.trae-html-share-packages/` 忽略。

## REMOVED Requirements
### Requirement: 无
**Reason**: 首次建立 spec，无历史需求。
**Migration**: 不适用。
