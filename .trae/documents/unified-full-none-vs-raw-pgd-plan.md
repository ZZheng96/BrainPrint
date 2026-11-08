# full_none_global 与 raw-PGD 统一对抗评估计划

## Context

用户要求补齐缺失指标，不能再把 full_none_global 的 clean OOF 指标与 raw-PGD 的对抗指标分开报告。目标是在同一数据划分、同一阈值选择、同一攻击生成和同一官方评分函数下，计算两个方案的 clean C、untargeted RA、ASafeR、A、U、unknown AUC/F1 与官方公式分。

## 推荐方案

1. 复用 `src/raw_eeg_robust/run.py` 的五折 outer/inner 划分、伪未知构造、raw EEG PGD 攻击和 `src/metrics.py` 官方公式。
2. 保留当前 raw-PGD `adversarial_ridge` 作为一个分支；新增 `full_none_global` 分支，使用其实际的完整协方差特征、无 target z-score、无 Session 规则和全局 operating point，在每个相同 outer fold 上重新拟合。
3. 两个分支共享完全相同的 fold、clean/unknown 评估集、PGD 样本、阈值选择、攻击 SNR/步数和拒识逻辑，避免拼接来自不同实验的指标。
4. 对每个 fold 和总体均输出 clean C、clean F1、untargeted SNR20 RA/ASafeR/A、targeted SNR20 ASafeR、U、unknown AUROC/F1、官方 Score，并记录样本数、阈值和随机种子。
5. 生成一份 Markdown 报告和 JSON/CSV 明细，明确结论以统一评估为准；不使用 Test 标签、Test Session 或官方反馈回调参数。

## 关键文件

- `src/raw_eeg_robust/run.py`：统一实验入口与两个模型分支。
- `src/raw_eeg_robust/adversarial.py`：复用 raw-PGD 增强与 adversarial Ridge。
- `src/raw_eeg_defense/run.py`、`src/metrics.py`：复用指标和官方公式。
- `src/open_set_improvement/core.py`：复用 full_none_global 特征与任务路由。
- 输出到新目录 `research/unified_full_none_vs_raw_pgd/`，避免覆盖已有报告和提交 CSV。

## 验证

- 确认两个方法使用相同 outer fold、攻击样本、阈值选择与指标函数。
- 检查报告包含每个指标的分子/分母和总体聚合方式。
- 校验没有读取 Test 标签或 Session 作为预测规则。
- 运行结果后给出“clean 优势”和“对抗优势”分开结论，不把代理分数宣称为官方 Test 分数。
