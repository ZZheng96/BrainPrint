# 第二次提交候选方案

## Context

第一次正式提交 `commit/answer/赞美诗_第1次.csv` 已获得官方 69.17 分，是当前唯一真实基线。现有 raw EEG robust 方向的五折验收未通过，不能直接转成正式 Test 预测；纯 EEG 开集和固定阈值候选又出现明显过度拒识。因此本阶段目标是：不覆盖第一次提交、不继续跑长 PGD 实验，尽快产出一个格式正确、变化范围受控、可解释的第二次候选 CSV，并在提交前保留第一次答案作为保护版本。

已确认的数据集说明位于 [赛题说明.md](file:///d:/WORK/BCIC2026/BrainPrint/赛题说明.md)：Test 同时包含已注册正常样本、对抗样本和未注册样本，标签 0 表示未知身份；Session 字段不是官方标签保证。当前工作区已有用户未提交文件，必须保持不变。

## 推荐实现

1. 新建独立实验目录 `research/second_submission/`，所有候选 CSV、诊断 JSON 和报告只写入该目录；不修改 `commit/answer/赞美诗_第1次.csv`。
2. 复用现有缓存和 [predict_task_specific.py](file:///d:/WORK/BCIC2026/BrainPrint/src/predict_task_specific.py) 的第一次任务特异链路，固定配置：`--pair-fusion --balanced-ec-imagery --guarded-ec --reject-session1`。先生成一个 baseline snapshot，确认可复现且格式正确。
3. 实现一个独立的保守门控脚本，仅在 Session2 处理已有“直接预测”和高相似波形组件融合预测不一致的行：
   - 只接受同任务组件；
   - 仅当组件标签一致、原始分数置信度较低、支持强度满足固定门槛时改变标签；
   - Session1 的 3,033 行始终继承第一次答案的 0；
   - 与第一次答案的总变化比例设硬上限（不超过 5%），超限则候选无效；
   - 不新增 Test 标签分布调参，不用官方未知标签反推阈值，不继续加入 PGD 样本。
4. 对 baseline snapshot 和门控候选执行统一验收：行数 22,251、Sample 顺序、整数标签范围 0–20、无缺失/重复、Session1 全部为 0、按 task/session 的标签分布、变更行数、配对一致率及异常类别集中度。
5. 生成 Markdown 实验报告，明确 Test 无真实标签，因此 Test AUC/F1 不可计算；报告既有训练/嵌套验证参考 AUC/F1（如当前基线参考 unknown AUC 0.9803、F1 0.9472），并记录候选相对第一次答案的所有变化和限制。

## 关键文件

- 复用：[src/predict_task_specific.py](file:///d:/WORK/BCIC2026/BrainPrint/src/predict_task_specific.py)
- 分数融合：[src/evaluate_task_specific_ensemble.py](file:///d:/WORK/BCIC2026/BrainPrint/src/evaluate_task_specific_ensemble.py)
- 缓存：`BrainPrint_cache/test_metadata.csv`、`scores_ridge_cov_zscore.npy`、`scores_window2_zscore.npy`、`waveform_pairs.csv`
- 保护基线：[赞美诗_第1次.csv](file:///d:/WORK/BCIC2026/BrainPrint/commit/answer/赞美诗_第1次.csv)
- 新目录：`research/second_submission/`

## 验证

- 先用 `py -3 -B` 执行候选生成和格式检查，避免修改正式提交。
- 对候选执行独立 CSV 校验脚本，确认 Sample 与官方模板完全一致。
- 检查 Session1 标签未发生变化，候选变化行数不超过 5%。
- 检查报告中同时出现 AUC 和 F1，并明确这些是内部参考指标而非官方 Test 成绩。
- 只有候选通过全部格式、变化上限、标签分布和可解释性门槛后，才将其复制为用户指定的第二次提交 CSV；否则保留第一次 69.17 版本，不提交风险候选。
