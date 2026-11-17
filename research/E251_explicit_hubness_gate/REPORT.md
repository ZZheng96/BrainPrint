# E251 显式 Hubness 拒识门控实验报告

## 结论

E251 **失败**：主判定 SNR30 下 PGD 后 RA=0.0000（目标 A/B/C/D 均为 0），未达到 RA≥0.95；ASafeR=1.0000 是全量拒识造成的，不能视为通过。SNR20 仅作参考，结果与 SNR30 相同。Test 官方 AUC/F1 为 **N/A**（Test 没有身份真值）。

## 固定协议与模型

- Train 11,304 行、20 个身份、855 维；使用 `cuda:0`。
- full Train deployment：20 个身份，`frac <= 0.6` 拟合；lambda 与 P1 FAR10 threshold 由 Train-only 内层 LOSO 固定。
- 选定 `lambda=0.00`，threshold=`0.3594579100608826`；未使用 Test 选择或调参。
- 攻击子集：Train `0.6 < frac <= 0.8`，每 task 最多 400 条，实际 1,227 条；20 步，扰动半径为去均值 waveform L2 / `10^(SNR/20)`。
- 优化方向：A 最大化最大错误身份 logit−真值 logit；B 最大化错误身份 logsumexp−真值 logit；C 最大化固定目标身份 logit−真值 logit；D 最小化拒识 gate。主判定 SNR30，SNR20 仅参考。

## 运行结果

| 条件 | RA | ASafeR | 接受率 | 闭集准确率 | 错身份率 |
|---|---:|---:|---:|---:|---:|
| clean | 0.957620 | 0.998370 | 0.959250 | 0.991850 | 0.001630 |
| SNR30 A/B/C/D（各） | 0.000000 | 1.000000 | 0.000000 | 0.047270 | 0.000000 |
| SNR20 A/B/C/D（各） | 0.000000 | 1.000000 | 0.000000 | 0.047270 | 0.000000 |

- `worst_RA`：SNR30 / goal A，RA=0.000000，ASafeR=1.000000。
- `worst_ASafeR`：SNR30 / goal A，ASafeR=1.000000（并列条件）。
- 由于攻击后全部拒识，ASafeR 虽为 1，但 RA 门禁明确失败。

## Test 一次性描述性审计

冻结上述 Train-only lambda/model/threshold；Test 仅读取一次，session 只分为 S1/S2，不参与决策。

| session | n | acceptance | rejection | score median | accepted identity count CV |
|---|---:|---:|---:|---:|---:|
| S1 | 3,033 | 0.452028 | 0.547972 | 0.325374 | 3.918427 |
| S2 | 19,218 | 0.441045 | 0.558955 | 0.336337 | 1.070334 |

S1 vs S2 score AUC（描述性）= **0.462531**。官方 Test AUC：**N/A**；官方 Test F1：**N/A**；没有身份真值，未进行官方身份指标计算。

## 十一项指标对照

| # | 指标 | 目标/口径 | E251 | 判定 |
|---:|---|---|---:|---|
| 1 | in-sample 准确率 | 1.00 | N/A | NOT_EVALUATED |
| 2 | C 闭集（块外，含拒识） | ≥0.95 | N/A | NOT_EVALUATED |
| 3 | macro-F1（Late） | ≥0.92 | N/A | NOT_EVALUATED |
| 4 | 块外闭集准确率（不含拒识） | ≥0.90 | N/A | NOT_EVALUATED |
| 5 | pseudo-unknown AUROC | ≥0.98 | 0.970837（严格 outer identity LOSO，lambda=0） | 未通过（Train代理） |
| 6 | U@95% | ≥0.95 | N/A | NOT_EVALUATED |
| 7 | RA | ≥0.95 | 0.000000（SNR30 worst） | 未通过 |
| 8 | ASafeR | ≥0.98 | 1.000000（SNR30 worst） | 通过但由全拒识造成 |
| 9 | Test S1 拒识 | ≥0.70 | 0.547972 | 未通过 |
| 10 | Test S2 接受 | ≥0.70 | 0.441045 | 未通过 |
| 11 | 严格 LOSO unknown rejection @ a=0.90 | >0.90 | 0.917569（严格 outer identity LOSO，lambda=0） | 通过 |

#7/#8 使用 `gates_attack.py` 的白盒 PGD；#9/#10 使用 `audit_test.py` 的一次性 Test 描述审计。缺少与 E251 口径一致的现有计算时，指标按要求标记 N/A/NOT_EVALUATED，未回填其他实验数值。

## 产物与运行命令

- `gates_attack.py` → `results/gates_attack.json`
- `audit_test.py` → `results/test_audit.json`
- 运行：
  - `.venv\\Scripts\\python.exe audit_test.py`
  - `.venv\\Scripts\\python.exe gates_attack.py`
- 未修改旧实验，未提交 git。模型与结果均保留在 E251 目录及既有 `results/outer_XX` 中。
