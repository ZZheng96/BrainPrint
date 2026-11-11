# Tasks

- [ ] Task 1: 清理根目录与缓存产物
  - [ ] SubTask 1.1: 删除根目录 `赞美诗_第3次.csv`（与 `commit/answer/赞美诗_第3次.csv` MD5 一致）
  - [ ] SubTask 1.2: 删除所有 `__pycache__/`、`.pytest_cache/`、空目录 `.screenshots/`、`.trae-html-share-packages/`
  - [ ] SubTask 1.3: 更新 `.gitignore`，补充 `.pytest_cache/`、`.screenshots/`

- [ ] Task 2: 归档 `research/` 根级散落文件
  - [ ] SubTask 2.1: 新建 `research/archive/submissions/`，移入 18 个 `submission_*.csv`
  - [ ] SubTask 2.2: 新建 `research/archive/diagnostics/`，移入 `diagnostics_*.json`、`validation_*.json`、`baseline_*.json`
  - [ ] SubTask 2.3: 新建 `research/archive/train_analysis/`，移入 `train_*.json`、`test_pair_analysis.json`、`waveform_pair_analysis.json`、`data_audit.json`
  - [ ] SubTask 2.4: 确认 `research/` 根级仅剩 3 个 md 文档 + 子目录

- [ ] Task 3: 去重 `research/` 内重复产物
  - [ ] SubTask 3.1: 删除 `calibrated_pseudo_ground_truth_v3/scenarios/` 中 MD5 重复的 `pseudo_answer.csv`（保留每组一份，其余改为软引用说明）
  - [ ] SubTask 3.2: 删除 `calibrated_pseudo_ground_truth_v4/scenarios/main_evidence_consensus/`（与 `v4/pseudo_answer.csv` 完全一致）
  - [ ] SubTask 3.3: 检查 `research/session_blind_candidate/candidates/` 与 `five_candidates/` 的重复 csv，保留 `five_candidates/`

- [ ] Task 4: 更新路径引用
  - [ ] SubTask 4.1: 全仓搜索 `research/submission_`、`research/diagnostics_`、`research/train_`、`research/baseline_`、`research/validation_` 引用
  - [ ] SubTask 4.2: 更新 `src/*.py`、`research/**/run.py` 中的路径常量
  - [ ] SubTask 4.3: 更新 `research/*.md` 中的相对路径链接

- [ ] Task 5: 建立索引文档
  - [ ] SubTask 5.1: 编写 `research/README.md`：目录 → 方法 → 关键结论（AUC/F1）→ 报告链接
  - [ ] SubTask 5.2: 编写根目录 `README.md`：项目结构、数据准备、复现入口、提交包说明

- [ ] Task 6: 验证与提交
  - [ ] SubTask 6.1: 运行 `py -3 -m pytest test/ -q` 确认测试通过
  - [ ] SubTask 6.2: 抽查 2 个 `research/**/run.py` 的路径解析（不实际跑完整实验，仅 dry-run 路径检查）
  - [ ] SubTask 6.3: `git add` 变更并提交，推送到 origin/main

# Task Dependencies
- Task 4 依赖 Task 2、Task 3（路径确定后才能更新引用）
- Task 5 依赖 Task 2、Task 3（索引需反映最终结构）
- Task 6 依赖 Task 1–5
