# E230 提交产物：E228 冻结模型的 Test 逐样本预测

## 0. 摘要

- 本提交只加载 `research/E228_constrained_maxS2/models/deploy_frozen.pt` 做前向推理；**没有重新训练、没有改阈值、没有改特征/块权重/PCA/原型**。
- 冻结候选：`e226_auc2W`（表征 `e226_all_blocks_v2`，打分 `plain`，来源 `E227 frozen deploy_auc2_lofo.pt (block-weighted representation)`）。
- 阈值：`tau = -24.138366796`（`A* = 0.804985444`）；`score < tau` → 标签 0，否则输出最近原型身份 1–20。
- 输出：`test.csv`（22251 行，与 `test_metadata.csv` 原始顺序逐行一致，表头 `Sample,Label`）与 `test_predictions_detailed.csv`（含 score、pred_identity、rejected、label）。
- 一致性：重载模型复算分数与冻结缓存最大绝对差 **0.000e+00**，身份预测逐行一致 **True**；S1 拒识 0.963732、S2 接受 0.304298，与 E228 记录值一致（Δ 均 < 1e-4）。
- 主要风险：Test 无身份标签；S2 接受仅 0.3043（拒识率 0.6957），远高于官方 Session2 拒识 ≤2% 的验收要求（等价于接受率应 ≥98%）；Test 自 E65 起被反复查看，S1/S2 带评估偏倚。

## 1. 冻结模型与不变项

| 项 | 值 |
|---|---|
| 模型文件 | `research/E228_constrained_maxS2/models/deploy_frozen.pt` |
| 候选 | `e226_auc2W` |
| 表征 | `e226_all_blocks_v2`（10 个特征块顺序拼接，1569 维） |
| 打分规则 | `score = -min Euclidean distance to same-task prototypes`；identity = 最近原型所属身份 |
| 门控 | 全局单一阈值 `tau = -24.138366796`；不做按身份/按 task 阈值 |
| PCA | 96 维（fit 行中心化协方差特征分解） |
| 原型 | 每 (task, identity) 独立 k-means，k = 12；20 身份 × 4 task = 960 个原型 |
| 标定 | `A* = 0.804985444`；阈值来自 Train fit 行 inner-OOF known 分数的 (1−A*) 分位 |
| 输入 | 18 通道、300 Hz、1200 点/trial；Test 22251 条 |

## 2. 特征块与部署权重

按以下顺序拼接特征，然后在 PCA 之前对每个维度乘 `sqrt(w_b)`（`w_b` 为部署模型内置块权重，取自 E227 auc2_lofo 的 20 折留一折均值，归一化到均值 1）：

| 特征块 | 维度 | 部署权重 |
|---|---:|---:|
| `corr_logspd` | 855 | 1.8724 |
| `bandpower_gainfree` | 90 | 1.3899 |
| `spectral_shape` | 90 | 1.1489 |
| `spectral_slope` | 18 | 0.9140 |
| `pac` | 18 | 0.5668 |
| `trial_energy` | 1 | 0.5910 |
| `hjorth` | 180 | 0.8124 |
| `spec_entropy` | 180 | 0.6582 |
| `power_ratio` | 112 | 1.3086 |
| `graph_metrics` | 25 | 0.7378 |

## 3. 推理流程

1. 按冻结特征定义读取 Test 的 10 个特征块，按上表顺序拼接成 `x`（22251 × 1569）。
2. 按 `task`（EC / EO / Imagery / SSVEP）使用 Train fit 行的均值/标准差做逐维 z-score（标准差为总体标准差 + 1e-6）。
3. 逐维乘 `sqrt(w_b)`（块权重在模型 `state['w']` 中）。
4. 减去 fit 行均值 `pm`，乘 PCA 投影矩阵 `W`（1569 × 96），得到 96 维表示 `z`。
5. 在同一 task 的 20 组 × 12 个原型上计算欧氏距离，取最小距离 `d_min` 及其原型身份。
6. `score = -d_min`；若 `score < tau` 则输出 0（拒识/未注册），否则输出最近原型身份（1–20）。
7. 按 `test_metadata.csv` 原始顺序写入 `test.csv` 与 `test_predictions_detailed.csv`。

## 4. 预测分布

- 拒识（0）：**16293 / 22251 = 0.7322**
- 接受（1–20）：**5958 / 22251 = 0.2678**
- Session1（3033 行）拒识 **0.9637**；Session2（19218 行）接受 **0.3043**（拒识 0.6957）。
- 未被预测到的身份：**5, 8, 19**。

| task | S1 行数 | S1 拒识 | S2 行数 | S2 接受 | 全部行拒识数 |
|---|---:|---:|---:|---:|---:|
| EC | 292 | 0.9829 | 1680 | 0.2893 | 1481 |
| EO | 291 | 0.9863 | 2021 | 0.1672 | 1970 |
| Imagery | 1459 | 0.9767 | 9748 | 0.3086 | 8165 |
| SSVEP | 991 | 0.9324 | 5769 | 0.3495 | 4677 |


逐身份预测数见 `metrics_summary.md` 第 3 节。

## 5. 一致性校验

| 校验 | 结果 |
|---|---|
| 重载冻结模型复算 Test 分数 vs `research/E227_joint_frontier/cache/test_v1_auc2_lofo.npz` | 最大绝对差 **0.000e+00**；身份预测逐行一致 **True** |
| S1 拒识 vs E228 的 0.963732 | **0.963732**，Δ = +0.000e+00（通过 1e-4） |
| S2 接受 vs E228 的 0.304298 | **0.304298**，Δ = +0.000e+00（通过 1e-4） |
| 分离 AUC vs E228 的 0.791840 | **0.791840**，Δ = +0.000e+00 |
| 行数/顺序 | 22251 行；`sample` 与 `test_metadata.csv` 逐行一致 |
| LOSO 拒识复算 | **0.9000**（E228 记录 0.900043；2301 个未知评估样本） |
| 拟合准确率复算 | **0.9991**（6695 个 fit 行） |

E228 `cache/` 只包含 `e226_all_blocks_v2/plain` 等候选的逐样本 Test 分数，不含冻结 auc2W 的逐样本分数；后者由 E228 `build_frozen()` 从 E227 的 `test_v1_auc2_lofo.npz` 读取，因此本报告以该文件作为逐行比对基准。以 E228 的 unweighted plain 缓存作负对照时最大绝对差为 13.728691，与预期一致（候选不同，不应相等）。

## 6. 指标可用性

详细表见 `metrics_summary.md`。要点：

- 可计算并已给出：拟合准确率 0.9991；真正留块准确率 0.9784；伪未知 LOSO AUROC 0.9096；U@95 0.3724；S1 拒识 0.9637；S2 接受 0.3043；LOSO 拒识 0.9000。
- **N/A**：官方 Test C、身份 macro-F1（需 Test 真实身份标签）；RA、ASafeR（需官方对抗样本及原始身份标签）；官方 Test AUROC/F1（无标签）。官方未提供这些标签，本提交不以代理值冒充。

## 7. 限制与风险

1. **Test 无身份标签**：无法计算官方 C、身份 macro-F1、RA、ASafeR、U 与最终 Score；S1/S2 仅是会话分层的结构审计。
2. **Test 已被反复查看**：自 E65 起 Test 的 S1/S2 行为被多次读取，候选/工作点的比较含评估偏倚；本提交不是无偏泛化估计。
3. **S2 接受率仅 0.3043**：对应 Session2 拒识率 0.6957。官方研发验收目标是 Session2 拒识 ≤2%（即接受 ≥98%），当前差距约 67.6 个百分点。以当前门控直接提交，注册者正常样本会被大量拒识为 0，C/RA 会被严重压低。
4. **LOSO 与 Test 未知集难度不一致**：LOSO 的未知是同一采集批次内留出的已注册身份；Test S1 是 5 名未注册者且存在跨 Session/跨域漂移。LOSO 拒识 0.9000 与 Test S1 拒识 0.9637 不能互相换算，LOSO 0.90 的约束不代表 Test 未知拒识达标。
5. **S2 混合正常与对抗样本**：官方未给攻击标签，无法把 S2 接受率拆成 C 与 RA/ASafeR；拒识已知样本虽可能提高 ASafeR，但会直接降低 C/RA。
6. **单一全局阈值**：tau 在 Train fit 行上标定，未使用 Test 统计量；但 Train 只有 session1，跨会话漂移不可观测，阈值对 S2 偏严。
7. **标签 0 的语义**：模型对已知但分数低于 tau 的样本也输出 0；在官方评分中这会同时影响 C、RA 与 ASafeR，不能解释为"只拒未注册者"。

## 8. 交付文件与复现

| 文件 | 内容 |
|---|---|
| `test.csv` | 22251 行，表头 `Sample,Label`，原始顺序，Label ∈ 0–20 整数 |
| `test_predictions_detailed.csv` | sample / path / task / session / score / pred_identity / rejected / label |
| `metrics_summary.md` | 十一项指标、N/A 原因、一致性校验、标签分布 |
| `report.md` | 本文件 |
| `run_e230.py` | 推理、复算校验、产物生成 |
| `config.json` | 冻结模型、阈值、特征路径、块顺序/权重、锚点与容差 |
| `run.log` | 本次运行日志 |

复现：

```text
.venv\Scripts\python.exe research/E230_submission_artifacts/run_e230.py
```

本次运行耗时 0.9s，设备 NVIDIA GeForce RTX 3080 Laptop GPU。未修改任何历史实验，未写入 `research/E229_raw_robust_encoder/`，未提交 git。
