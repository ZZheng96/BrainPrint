# raw-PGD Test CSV 生成计划

## Context

当前 `full_none_global` 的提交 CSV 不是 raw-PGD 方案。`src/raw_eeg_robust/run.py` 目前只有训练集五折 raw-PGD 验收流程，没有全量训练后 Test 推理和 CSV 输出入口。用户需要真正对应 `adversarial_ridge` raw-PGD 方案的 Test CSV。

## 推荐方案

1. 在 `src/raw_eeg_robust/run.py` 增加独立的最终 Test 推理分支，保持现有 raw-PGD 训练协议：全量 Train、每个 subject/task 分组采样、10 步 raw EEG PGD、SNR20、untargeted 与 targeted 混合增强、重新拟合 adversarial Ridge。
2. 在全量 Train 内使用预先固定的独立校准流程选择 operating point，禁止读取 Test 标签、Test Session 或利用 Test 分布反向调参。
3. Test 阶段只读取 `sample`、`task`、Test EEG 特征和窗口特征；用 robust Ridge 直接预测，不在 Test 上生成或使用 Session 特定规则。
4. 按 `Brainprint_Competition/test.csv` 的原始 Sample 顺序生成 `Sample,Label`，输出研究候选 `research/submission_raw_pgd_adversarial_ridge.csv`，再复制为队名格式 `commit/answer/赞美诗_第3次_rawPGD.csv`（若提交系统只接受“队名+第几次”，则使用 `赞美诗_第3次.csv` 前需明确覆盖风险）。
5. 对输出执行行数、Sample 唯一性/顺序、Label 范围、缺失值和类别计数核验；同时明确该 CSV 只代表 raw-PGD 方案，不能把训练集代理分数当作官方 Test 分数。

## 关键文件

- `src/raw_eeg_robust/run.py`：新增全量 robust 重训和 Test 推理入口。
- `src/raw_eeg_robust/adversarial.py`：复用现有 `fit_adversarial_ridge` 和攻击特征生成逻辑，不改变训练算法。
- `src/open_set_improvement/core.py`：复用 `TaskSpecificModel`、特征列和缓存定位。
- `src/open_set_calibration/core.py`：复用提交构造与格式校验。
- `research/raw_eeg_robust/report.md`：记录当前 raw-PGD 尚未通过冻结验收的事实和生成结果边界。

## 验证

- 运行 raw-PGD Test 推理命令，优先使用 CUDA。
- 检查生成 CSV 为 22,251 条数据、两列 `Sample,Label`、Sample 与官方模板顺序完全一致、Label 为 0–20 整数且无缺失。
- 用代码审查确认 Test 阶段只读取 `sample/task`，没有读取或使用 `session`、Test 标签或官方反馈。
- 将 CSV 复制到 `commit/answer` 后再复核哈希和文件大小。
- 报告中明确：raw-PGD 本地 RA/ASafeR/代理分来自 Train 内部攻击验证，不等于官方 Test 分数，且当前实验尚未通过原冻结门槛。
