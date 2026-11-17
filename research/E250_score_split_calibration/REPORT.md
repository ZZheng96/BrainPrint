# E250：Score split calibration（严格嵌套 Train-only + 一次性 Test 审计）

## 结论

- 严格嵌套的 Train 评估中，预先固定规则选择 **C2_margin_cohort_hubness**：其内层已知接受率高于 C0（0.957250 vs 0.955807）。Test 未参与候选、阈值或模型选择。
- C2 相比 C0 的提升很小：AUC **0.971098**、拒识 F1 **0.540760**；C0 为 AUC **0.970837**、拒识 F1 **0.529394**。
- 一次性 Test 描述性审计显示，C2 的 S1 接受率 **0.451039**、S2 接受率 **0.442190**，没有显示出预期的 S2 接受优势。Test 没有身份真值，因此官方意义的 Test AUC/F1 均为 **N/A**。
- 本实验没有进行攻击重测；攻击相关门禁为 **N/A**，不把 E249 或 E191 的攻击结果冒充为 E250 结果。

## 协议与静态审计

E250 的外层为严格身份级 LOSO：外层留出身份不进入拟合；外层 fit 使用已知身份 `frac <= 0.6`，Late 评估使用 `frac > 0.8`。每个外层折内再对已知身份做 inner LOSO：inner fit 为其余身份 `frac <= 0.4`，`0.4 < frac <= 0.6` 只用于伪未知/已知校准与阈值估计。身份预测始终是严格 LDA-cos top-1，校准只改变拒识分数，不改变身份排序。

静态审计发现并修正了一处阈值实现问题：原实现将 20 个 inner held identity 的伪未知分数直接拼接后计算 FAR10，实际按样本数而不是按身份等权。现改为保留每个 held identity 的向量，再调用 E247 的 tie-safe `weighted_far_threshold`，使每个伪未知身份等权；阈值、校准模型及原型均仍只来自对应 inner/outer Train 数据。重载检查保留并通过：保存模型后重新加载，校准分数与 top-1 身份逐项一致。

## Train 全折结果（20 个外层身份折，Late，均值）

| 候选 | AUC | 拒识 F1 | 已知接受率 | 未知拒识率 | 已知身份识别准确率 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.970837 | 0.529394 | 0.919270 | 0.917569 | 0.916808 |
| C1_margin | 0.970884 | 0.537630 | 0.922239 | 0.915811 | 0.920583 |
| **C2_margin_cohort_hubness** | **0.971098** | **0.540760** | **0.923663** | 0.913003 | **0.921845** |

全折范围（用于识别折间不稳定性）：C2 的 AUC 为 0.861174–0.990529，拒识 F1 为 0.260062–0.660494，已知接受率为 0.910297–0.952162，未知拒识率为 0.352941–1.000000。Subject 07 对未知拒识相关指标仍是明显困难折，说明均值提升不能解释为所有身份均稳定改善。

## Test S1/S2 一次性描述性审计（看过 Test）

部署模型使用全部 20 个 Train 身份的 Early 数据拟合；阈值和候选选择沿用 Train inner LOSO。Test 特征按 E249 audit 的加载方式读取：`BrainPrint_cache/test_metadata.csv` 与 `research/E200_final_freeze/submission_frozen_work/features.npy`。`session` 仅用于审计分组，未用于判决、拟合或选择。

| 候选 | 阈值（P1 FAR10） | S1 n | S1 接受 | S1 拒识 | S2 n | S2 接受 | S2 拒识 | S1 vs S2 分数 AUC（描述性） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.359458 | 3,033 | 0.452028 | 0.547972 | 19,218 | 0.441045 | 0.558955 | 0.462531 |
| **C2_margin_cohort_hubness** | **0.856693** | 3,033 | **0.451039** | **0.548961** | 19,218 | **0.442190** | **0.557810** | **0.462357** |

`S1 vs S2 分数 AUC` 仅表示两组无真值分数的分离度：这里按 S2 为正类、S1 为负类计算，不能解释为身份识别 AUC。Test 官方 AUC/F1：**N/A（无身份真值）**。接受身份计数 CV 也仅作描述：C0 为 S1 3.918、S2 1.070；C2 为 S1 3.928、S2 1.081，提示接受样本仍集中于少数注册原型。

## 十一项门禁复用边界

E249 的 11-gate 表包含 E249 自身的候选定义、攻击重测和一次性 Test 指标。E250 没有完成同一攻击协议，也没有身份真值 Test，因此不可靠地把 E249 数字复制到 E250。可直接从 E250 复用的只有本报告的 Train 开放集 AUC/F1、已知接受/未知拒识和识别准确率；其余门禁记录如下：

| 门禁 | E250 状态 | 说明 |
|---:|---|---|
| #1–#4 | N/A | 未按 E249 gate 脚本重算相同定义，避免口径混用 |
| #5–#6 | N/A | 未完成 E249 同口径门禁聚合 |
| #7–#8 | N/A | 本实验无攻击补测 |
| #9–#10 | N/A（官方门禁） | Test 无身份真值；本报告只给 S1/S2 描述性接受/拒识 |
| #11 | N/A | 未另行生成 E249 同口径 LOSO gate 聚合 |

## 与 E249 / E191 的可比性

- 与 E249 的共同点是 E247 严格身份级 LDA-cos、Train-only 阈值估计、外层留出身份、Test 只做一次描述性审计；因此可以比较总体方向和 Test S1/S2 描述，但不能把 E250 的 C2 校准结果与 E249 的 NAP/WCCN 候选当作同一模型。
- E250 的 C0 与 E249 C0 在基础身份空间上相近，但 E250 的评分校准特征、内层分数和阈值流程不同；本报告只将它们作为相邻实验的参考，不宣称逐数复现。
- E191 是冻结部署方案，包含与 E250 不同的特征/模型/门控和正式提交结果。E191 的 Test S1/S2、攻击和 11-gate 数字不能移植到 E250；E250 也没有修改 `final/` 或 E191 冻结产物。
- E249/E191 的 Test 结论均属于“看过 Test”的描述性审计。E250 同样明确披露 Test 已查看，不能把本次 Test 分组结果当作新的 Train-only 选择证据。

## 复现与产物

已运行：

```powershell
& .\.venv\Scripts\python.exe research\E250_score_split_calibration\run_e250.py --validate-only
& .\.venv\Scripts\python.exe research\E250_score_split_calibration\run_e250.py --all-folds
& .\.venv\Scripts\python.exe research\E250_score_split_calibration\audit_test.py
```

主要产物：

- `run_e250.py`：严格嵌套 Train 训练、校准、阈值与重载检查；
- `audit_test.py`：一次性 Test S1/S2 描述性审计；
- `results/per_fold.csv`、`results/summary.csv`、`results/selection.json`；
- `results/test_audit.json`；
- `results/outer_01`–`outer_20/model.pt`：各外层折模型，避免重复训练。

计算设备为 `cuda:0`。未修改 `final/`、E249 或旧实验，未提交 git。
