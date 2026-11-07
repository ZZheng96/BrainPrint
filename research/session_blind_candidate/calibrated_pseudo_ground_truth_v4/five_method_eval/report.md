# v4 主伪答案下的五个无 Session 方案评估

## 结论与边界

本次使用 v4 的 `pseudo_answer.csv`（main_evidence_consensus）评估五个无 Session 方案。结果是相对伪答案的条件估计，不是真实 Test 官方分。伪答案依据官方共同标签和候选证据构造，不再随机翻转标签；但 Test 的真实标签、clean/adversarial/unknown 分组和候选 RA/ASafeR/U 仍不可观测。

## 结果

| 方法 | C 代理 | 相对伪答案总体准确率 | 宏 F1 | RA=0 条件分范围 | RA=1 条件分范围 | AUC |
|---|---:|---:|---:|---:|---:|---|
| full_none | 0.1709 | 0.2617 | 0.1697 | 28.31–32.37 | 45.65–49.33 | null |
| base_none | 0.1689 | 0.2603 | 0.1640 | 28.18–32.21 | 45.44–49.09 | null |
| full_target_zscore | 0.0587 | 0.2246 | 0.0899 | 18.46–21.11 | 29.77–32.17 | null |
| base_target_zscore | 0.0526 | 0.2192 | 0.0818 | 17.67–20.20 | 28.49–30.78 | null |
| full_riemannian_recenter | 0.1409 | 0.2384 | 0.1446 | 26.21–29.97 | 42.27–45.67 | null |

`score_RA0`/`score_RA1` 只表示在固定 ASafeR/U 敏感性设置下的上下界；候选没有真实攻击版本标签，因此不能选定真实 RA。AUC 不可计算；F1 是相对 v4 伪答案的审计指标，不是官方 Test F1。

## 文件

- `candidate_metrics.csv`：每个方法在 ASafeR/U 敏感性组合下的条件结果。
- `candidate_mean_std_summary.csv`：方法汇总。
- `assumptions.json`：输入哈希和假设。
