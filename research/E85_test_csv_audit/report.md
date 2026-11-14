# E85 统一项目 Test CSV 与后置审计

本实验汇总真正兼容 BrainPrint 项目 Test 的冻结候选 CSV。所有 CSV 均按 `Sample,Label` 校验，并在写出后读取 Session/task 做离线工程审计。

## Test 指标边界

项目 Test 没有真实身份、攻击状态或注册状态标签，因此 Test AUC、F1、C、RA、ASafeR、U、Official Score 均为 `N/A（无Test真实标签）`。官方模板中的 Label 不作为真值使用。

- **S1**：Session1 拒识率，门限 `>=95%`。
- **S2**：Session2 拒识率，门限 `<=2%`，等价接受率 `>=98%`。
- Session/task 只在 CSV 写出并重新读取后审计，未用于推理。

## 候选来源

- `E64_e58_test_audit/e64_predictions.csv`：18 通道项目候选。
- `E42_submission_candidates/candidate_e36c_rule.csv`：18 通道 E36C 单门控候选。
- `E42_submission_candidates/candidate_two_gate_rule.csv`：18 通道双门控候选。

外部 E74–E84 为 22 通道 BCIC IV-2a，不接入项目 Test。

## 运行

```powershell
python research/E85_test_csv_audit/audit.py `
  --csv research/E64_e58_test_audit/e64_predictions.csv `
  research/E42_submission_candidates/candidate_e36c_rule.csv `
  research/E42_submission_candidates/candidate_two_gate_rule.csv `
  --compare research/E42_submission_candidates/candidate_e36c_rule.csv `
  research/E42_submission_candidates/candidate_two_gate_rule.csv
```

结果写入 `outputs/results.json` 与 `outputs/comparison.json`。

## 本次审计结果

| 候选 | 总拒识率 | S1 拒识率 | S2 拒识率 | S2 接受率 |
|---|---:|---:|---:|---:|
| E64 | 68.6261% | 58.0284% | 70.2987% | 29.7013% |
| E42 E36C | 80.9312% | 59.2153% | 84.3584% | 15.6416% |
| E42 双门控 | 81.5514% | 59.2812% | 85.0661% | 14.9339% |

三份 CSV 均为 22251 行，Sample 顺序与官方模板一致，Label 均在 0..20。E36C 与双门控 CSV 一致率为 99.3798%，有 138 行由接受变为拒识。当前候选均未同时满足 S1/S2；这只是 Test 无标签工程审计，不是有真值的分类性能结论。
