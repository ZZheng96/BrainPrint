# E86 外部 outer test CSV 与全对话指标进展

## 1. 目的与边界

本实验回答用户提出的问题：**从本对话开始到 E84 的每个实验，是否都给出了 test 上的 CSV，并从 CSV 比对计算 test 指标？** 对缺失者补做。

两类 test 必须区分：

- **项目 Test（BrainPrint，18 通道）**：E64/E42 候选已有 `Sample,Label` CSV，由 E85 统一审计。因无真实标签，Test AUC/F1/C/RA/ASafeR/U 均为 `N/A`。
- **外部 outer test（BCIC IV-2a，22 通道）**：E73–E84 使用的 held-out outer query（run5），有真实身份与 known/unknown 标签，因此可从 CSV 重算 AUC/F1 等指标。

外部实验的 22 通道模型不兼容项目 18 通道 Test，二者结果不可互相代替。

## 2. 覆盖核查

| 实验 | test CSV | test 类型 | 说明 |
|---|---|---|---|
| E72 | 无 | 项目 Train-only nested 协议 | 无 Test 访问、无外部 test；只做 5 outer × 2 inner 的 Train 内协议审计 |
| E73 | 有 | 外部 outer | `native22` 与 `native22_pair` 两路线 |
| E74 | 有 | 外部 outer | 主方法 `wccn:margin` |
| E75 | 有 | 外部 outer | 仅 `T_to_E` 方向，3 组 |
| E76 | 有 | 外部 outer | `mahalanobis` |
| E77 | 有 | 外部 outer | `fusion_q05` |
| E78 | 有 | 外部 outer | `per_identity_q05` |
| E79 | 有 | 外部 outer | `pair_q02` 与 `pair_q05` |
| E80 | 有 | 外部 outer | `min_fusion_q02` |
| E81 | 有 | 外部 outer | `evt_q02` |
| E82 | 有 | 外部 outer | `fpir01` 与 `fpir02` |
| E83 | 无 | 只读可行性审计 | 不产生预测，只审计 E76/E79/E82 已落盘 operating point |
| E84 | 有 | 外部 outer | `pair_q02` 与 `pair_q05` |
| E85 | 有 | **项目 Test** | 审计 E64/E42 三份候选 CSV，S1/S2 工程门禁 |

E72 与 E83 没有可导出的预测：前者是 Train 内协议，后者只做只读门禁审计。其余实验全部补齐。

## 3. 每个实验的 outer test CSV

所有 CSV 均写入各实验 `outputs/outer_test_predictions.csv`，列固定为：

`experiment, outer_fold, direction, method, role, sample_id, session, run, task, artifact, true_identity, pred_label, accepted, confidence, threshold`

- `role`：`enrolled`（run5 known query）或 `outer_unknown`（run5 held-out subject query）。
- `pred_label`：0 表示拒识，其余为预测身份。
- 每个 CSV 由 `research/E86_external_outer_csv/export_e7x.py`/`export_e8x.py` 生成：只加载已有 checkpoint，不重训。

### 校验

从落盘 CSV 重新读取并用 `common.metrics_from_frame` 重算，与该实验 `results.json` 中同折同方向的 outer 指标逐项对比：

| 实验 | CSV 行数 | 分组 | max abs diff |
|---|---:|---:|---:|
| E73 | 4032 | 12 | 5.6e-17 |
| E74 | 2016 | 6 | 1.1e-16 |
| E75 | 1008 | 3 | 4.2e-17 |
| E76 | 2016 | 6 | 0.0 |
| E77 | 2016 | 6 | 0.0 |
| E78 | 2016 | 6 | 0.0 |
| E79 | 4032 | 12 | 5.6e-17 |
| E80 | 2016 | 6 | 0.0 |
| E81 | 2016 | 6 | 0.0 |
| E82 | 4032 | 12 | 0.0 |
| E84 | 4032 | 12 | 1.1e-16 |

### 指标（CSV 重算，折/方向均值）

| 实验 / 方法 | AUC | identity macro-F1 | unknown-F1 | known acc | unknown FAR | EER | DIR@FPIR1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E73 native22 | 0.9141 | 0.9578 | 0.7792 | 0.9219 | 0.2789 | 0.1534 | 0.4583 |
| E73 native22_pair | 0.9243 | 0.9635 | 0.7736 | 0.9349 | 0.2870 | 0.1331 | 0.4575 |
| E74 wccn:margin | 0.9226 | 0.9584 | 0.8185 | 0.9253 | 0.2315 | 0.1444 | 0.3759 |
| E75 cosine_prototype_margin | 0.4640 | 0.4993 | 0.1017 | 0.9514 | 0.9421 | 0.5382 | 0.0295 |
| E76 mahalanobis | **0.9764** | 0.9524 | 0.8919 | 0.9175 | 0.1042 | **0.0692** | **0.7674** |
| E77 fusion_q05 | 0.9682 | 0.9506 | 0.8745 | 0.9115 | 0.1227 | 0.0842 | 0.6814 |
| E78 per_identity_q05 | **0.9764** | 0.9451 | **0.9018** | 0.9036 | **0.0729** | **0.0692** | 0.7613 |
| E79 pair_q02 | 0.9185 | **0.9764** | 0.8016 | **0.9557** | 0.2755 | 0.1380 | 0.4688 |
| E79 pair_q05 | 0.9185 | 0.9565 | 0.8291 | 0.9210 | 0.2025 | 0.1380 | 0.4679 |
| E80 min_fusion_q02 | 0.9336 | **0.9814** | 0.8149 | **0.9644** | 0.2569 | 0.1244 | 0.4948 |
| E81 evt_q02 | 0.9336 | 0.9671 | 0.8347 | 0.9401 | 0.2060 | 0.1244 | 0.4905 |
| E82 fpir01 | 0.9185 | 0.7909 | 0.7942 | 0.6936 | **0.0938** | 0.1380 | 0.4705 |
| E82 fpir02 | 0.9185 | 0.8304 | 0.8021 | 0.7431 | 0.1157 | 0.1380 | 0.4705 |
| E84 pair_q02 | 0.9332 | 0.9763 | 0.8040 | **0.9557** | 0.2697 | 0.1357 | **0.5408** |
| E84 pair_q05 | 0.9332 | 0.9553 | 0.8336 | 0.9193 | 0.1944 | 0.1357 | 0.5391 |

E75 的 C 在原报告是 argmax 直接准确率（不含拒识），CSV 中 0 表示拒识，无法恢复被拒样本的 argmax 身份，故统一 schema 下 C 与 unknown-F1 不能与原口径逐位比较；其身份准确率已用缓存特征确定性重放验证为 diff=0。

## 4. 跨实验 CSV 比对

`cross_experiment_comparison.json` 对 11 个实验两两比较（55 对，53 对样本键匹配成功）。同一 fold/direction 下样本 ID 可对齐，因此可计算逐行一致率、分歧方向与标签转移矩阵。

代表性结果：

- E79 ↔ E84：一致率 0.9916，仅 17/2016 行不同（4 行 E79 拒识而 E84 接受，13 行相反），无身份分歧。
- E76 ↔ E78：一致率 0.9499，72 行由接受变拒识，说明逐身份阈值主要收紧接受。
- E77 ↔ E80：一致率 0.9082，181 行由拒识变接受。
- E75 与其他实验一致率仅 0.23–0.33，大量身份分歧，与其 Detection AUC≈0.46 一致。
- E73 与其他路线有少量身份分歧（2–9 行），源于不同预训练编码器。

## 5. 全对话指标改进进展

1. **E73 跨 Session 预训练**：pair AUC 0.9765，但 detection AUC 仅约 0.76，检测是短板。
2. **E74 多尺度 native22 编码器**：detection AUC 0.9226、unknown-F1 0.8185、EER 0.1444，检测能力大幅提升。
3. **E75 图像化 DINOv2**：detection AUC 0.4640、unknown-FAR 0.9421，路线不可用，已排除。
4. **E76 Mahalanobis open-set head**：AUC 0.9764、unknown-F1 0.8919、EER 0.0692、DIR@FPIR1 0.7674，为当前综合最强。
5. **E77 固定融合**：AUC 0.9682，低于 E76，说明等权融合无增益。
6. **E78 逐身份阈值**：unknown-FAR 降至 0.0729，但 known acceptance 降到 0.9036。
7. **E79 pair head**：q02 下 known acceptance 0.9557、identity-F1 0.9764，但 unknown-FAR 0.2755。
8. **E80 稳健 min 融合**：identity-F1 0.9814（最高），unknown-FAR 仍 0.2569。
9. **E81 EVT 尾校准**：unknown-FAR 0.2060，known acceptance 0.9401，未突破。
10. **E82 pseudo-unknown 标定**：FAR 可压到 0.0938（FPIR1%），但 known acceptance 跌到 0.6936。
11. **E84 hard-negative pair**：DIR@FPIR1 提升到 0.5408，AUC 0.9332，identity-F1 0.9763，但 unknown-FAR 仍 0.2697。
12. **E83 可行性审计**：E76/E79/E82 已落盘 operating point 同时满足 known≥95% 与 FAR≤2% 的可行率为 0。

**结论**：检测与排序指标显著改善（detection AUC 0.76→0.976，EER 0.14→0.069），但 known/unknown 置信分布重叠没有消除，外部 pseudo-unknown 上没有任何一组同时达到 `known acceptance ≥95%` 与 `unknown FAR ≤2%`。

## 6. 项目 Test 现状（E85）

| 候选 | 总拒识率 | S1 拒识率 | S2 拒识率 | S2 接受率 |
|---|---:|---:|---:|---:|
| E64 | 68.6261% | 58.0284% | 70.2987% | 29.7013% |
| E42 E36C | 80.9312% | 59.2153% | 84.3584% | 15.6416% |
| E42 双门控 | 81.5514% | 59.2812% | 85.0661% | 14.9339% |

S1 要求拒识率 ≥95%，S2 要求拒识率 ≤2%。三份候选均未通过。项目 Test 无真实标签，AUC/F1/C/RA/ASafeR/U/Official Score 仍为 `N/A`。

## 7. 产物

- 各实验 `outputs/outer_test_predictions.csv`、`outputs/outer_test_metrics.json`
- `outputs/all_experiments_metrics.json`
- `outputs/cross_experiment_comparison.json`
- 生成脚本：`export_e73.py` … `export_e84.py`、`common.py`、`summarize.py`
