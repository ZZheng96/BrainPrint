# E174 历史实验 W 指标考古补算

日期：2026-11-15
范围：`research/E*` 全部 177 个实验目录（含 `_trash`、`commit/` 下的提交文件一并审计，但不计入实验数）
性质：只读补算。未修改任何已有实验目录，未提交 git。
Test 官方 AUC / F1：**N/A**（无真实标签，全项目一致）。

```
W = 0.25 × LOSO 拒识率 + 0.25 × Test S1 拒识率 + 0.5 × Test S2 接受率
```

---

## 0. 结论先行

1. **0.71 不是历史最高。** 历史上有实验超过 0.71，但幅度很小，且都依赖乐观口径。
2. **按 pre-registered 固定接受率 a=0.90、Train-only、无污染、非退化**：
   最高 **W = 0.7171**，来自 **E170_three_way_fusion_ceiling**
   （`score 0.4*E65(64) + 0.1*logbp(48) + 0.5*E162(128) k=1`，S1 0.6106 / S2 0.7813 / LOSO 0.6950，
   与 E170 报告第 13 行「固定工作点（接受率 0.90，不用 Test 选点）三路族最高 W 0.7171」一致）。
   次高 E163 = 0.7164、E168 = 0.7156。三者都只比 0.71 高 0.006–0.007，属同一噪声量级。
3. **按 0.80–0.96 接受率网格取最大 W（工作点由 Test 选出）**：
   clean 最高 **W = 0.7617**，来自 **E168_subwindow_covariance**
   （`subwin_concat_n4 PCA-256 k=3`，@a=0.80，S1 0.5813 / S2 0.7851 / LOSO 0.8953）。
   E168 自己的报告明确写了这一点「是用 **Test 标签选出来的工作点**……属于分数分布形状/阈值效应，
   不能当作方案提升」；该配置 Train OOF AUC 只有 0.7350。
4. **计入违规/污染实验**，最高 **W = 0.7877**（E142_cdf_fusion，LOSO 污染），其次 0.7875（E139）、0.7796（E141）。
   这些数不能引用。
5. 老实验（E00–E120）**没有一个能补算出可信的 W > 0.71**；它们大多只有 Test CSV、没有同一工作点的 LOSO，
   只能给出 `W(LOSO=0)` 与「达到 0.71 所需的 LOSO」作为边界（见 §5）。

---

## 1. 数据来源与补算方法

### 1.1 输入

| 来源 | 用途 | 数量 |
|---|---|---:|
| `research/**/results.json` 等 | 已公布的三元组（best / ref@0.90 / curve / audit 块） | 18,415 条去重记录 |
| `research/**/report.md` | Markdown 表格里的同工作点三元组 | 21 个实验、566 行（去重后并入上表） |
| `research/**`、`commit/**` 的 22251 行预测 CSV | 补算 Test S1 / S2 | 251 个文件全部对齐 |
| `BrainPrint_cache/test_metadata.csv` | sample→session 映射（session1 = 3,033，session2 = 19,218） | 22,251 行 |

### 1.2 CSV 审计规则

- 文件大小 150 KB–700 KB 的全量 CSV 先全部读入，再按列名（`Sample/Label` 或 `sample/label`）识别；
  不含这两列的 OOF / 外部预测文件（如 `outer_test_predictions.csv`）记为 `not_test_csv`，共 17 个。
- 行数必须为 22,251 且 sample 集合与元数据**完全一致**；本次 **0 个 misaligned**（无一例偏差）。
- `S1 拒识率 = P(label==0 | session1)`，`S2 接受率 = P(label!=0 | session2)`。
- `label==0 条数 == session1 条数（3,033）` 即判定为「按 Session 直接置 0」，标 `session_zeroed`。

### 1.3 配对 LOSO 的硬约束

只有以下情形才算「同一工作点」：

1. 同一 JSON 节点内同时给出 `best_W / best_acceptance / loso_at_best / s1_at_best / s2_at_best`；
2. 同一节点给出 `W_at_ref / ref_acceptance / loso_at_ref / …`（E168 的 a=0.90 参考点）；
3. 同一节点给出 `W_at_090 / loso_at_090 / s1_at_090 / s2_at_090`；
4. `curve[]` 一行内联动给出 `acceptance_rate + LOSO_rejection + Test_S1_rejection + Test_S2_acceptance`；
5. one-shot frozen audit 布局（`train_only_metrics.rule` 与 `audit` 同文件、同一部署阈值）；
6. report.md 同一行同时给出 S1 拒识、S2 接受、LOSO 拒识三列。

其余一律 `LOSO = NA`，不做借用、不做插值、不跨实验借值。

### 1.4 有效性标签

| 标签 | 判定 | 涉及实验 |
|---|---|---|
| `clean` | Train-only 协议、LOSO 无原型污染、未用 Test 统计 | 其余全部 |
| `loso_contaminated` | 原型集合曾包含被评分块 | E119、E133、E136、E137、E138、E139、E140、E141、E142 |
| `test_statistic` | 使用 Test 侧统计量 | E135（Test 逐任务经验 CDF）、E142（CDF 融合）；以及任何 `session_zeroed` 提交 |
| `no_loso` | 缺同一工作点的 LOSO | E132（只有全局 LOSO 0.6261，未与三个工作点对齐）、E130、E131 等 |

> E142 同时命中 `loso_contaminated` 与 Test 侧 CDF，按「污染」优先展示，但性质上是双重无效。

---

## 2. (a) 主排序表

### 2.1 clean（可引用）

排序依据：每个实验取可算 W 的最优工作点。`口径` 列：`a=0.90` = 预注册固定接受率；`grid-best` = 在 0.80–0.96 网格上挑出的最大 W（工作点由 Test 选出）。

| # | 实验 | 方法/配置 | 接受率 | 口径 | S1 拒识 | S2 接受 | LOSO | W | 备注 |
|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 1 | E168 | subwin_concat_n4 PCA-256 k=3 | 0.80 | grid-best | 0.5813 | 0.7851 | 0.8953 | 0.7617 | 网格最优（Test 选点） |
| 2 | E152 | session_robust_scoring | 0.95 | other | 0.0000 | 1.0000 | 1.0000 | 0.7500 | 退化（S1=0/S2=1 的恒定分数） |
| 3 | E170 | score 0.5*E65(64) + 0.1*logbp(32) + 0.4*E162(64) k=1 | 0.88 | grid-best | 0.6775 | 0.7210 | 0.7530 | 0.7181 | 网格最优（Test 选点） |
| 4 | E163 | score 0.50*E65(64) + 0.50*E162(64) k=1 | 0.90 | a=0.90 | 0.6030 | 0.7697 | 0.7231 | 0.7164 |  |
| 5 | E169 | spectral_shape | 0.90 | other | 0.8180 | 0.5966 | 0.7975 | 0.7022 | oracle 阈值（Test 选点） |
| 6 | E171 | multiscale_fusion | 0.95 | other | 0.7916 | 0.6283 | 0.7569 | 0.7013 | oracle 阈值（Test 选点） |
| 7 | E157 | per_sample_calibration | 0.90 | a=0.90 | 0.7969 | 0.5945 | 0.8147 | 0.7002 |  |
| 8 | E167 | raw concat 96+32 -> PCA-96 k=12 | 0.90 | a=0.90 | 0.8098 | 0.5817 | 0.8257 | 0.6997 |  |
| 9 | E159 | ceiling_analysis | 0.89 | other | 0.8140 | 0.5756 | 0.8230 | 0.6971 | oracle 阈值（Test 选点） |
| 10 | E150 | s2_diagnosis | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6959 |  |
| 11 | E158 | bandwise_prototype | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6959 |  |
| 12 | E165 | identity | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6959 |  |
| 13 | E143 | k_sweep_nearest | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 14 | E145 | fusion_recheck_fixed_loso | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 15 | E147 | phase_features_eval | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 16 | E149 | image_route_audit | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 17 | E172 | E65_logspd PCA-96 k=12 | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 18 | E161 | raw_eeg_contrastive | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 19 | E162 | raw_eeg_contrastive_v2 | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 20 | E164 | raw_eeg_highrank | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 21 | E166 | tfmap_ssl | 0.90 | a=0.90 | 0.7979 | 0.5960 | 0.7936 | 0.6958 |  |
| 22 | E153 | dual_gate_scoring |  | other | 0.7807 | 0.5200 | 0.8774 | 0.6746 | 网格外工作点，仅记录 |
| 23 | E148 | logbandpower_probe | 0.85 | other | 0.8675 | 0.4809 | 0.8282 | 0.6644 |  |
| 24 | E126 | PERTASK_PCA96 |  | other | 0.7679 | 0.5971 | 0.6583 | 0.6551 | 网格最优（Test 选点） |
| 25 | E122 | pca96_deploy | 0.85 | other | 0.7916 | 0.5823 | 0.6261 | 0.6456 |  |
| 26 | E123 | band_pca | 0.85 | other | 0.7916 | 0.5823 | 0.6261 | 0.6456 |  |
| 27 | E128 | neg_d1 |  | other | 0.7916 | 0.5823 | 0.6261 | 0.6456 | 网格最优（Test 选点） |
| 28 | E129 | E0_ref_fit60_thr6080 |  | other | 0.7916 | 0.5823 | 0.6261 | 0.6456 | 网格最优（Test 选点） |
| 29 | E127 | pertask_pca96/both_task_gap |  | other | 0.5806 | 0.5159 | 0.9466 | 0.6398 | 网格最优（Test 选点） |
| 30 | E104 | session_invariant_pretrain | 0.66 | other | 0.4263 | 0.6150 | 0.8981 | 0.6386 | 网格外工作点，仅记录 |
| 31 | E109 | E98_dual_gate |  | other | 0.4517 | 0.7686 | 0.4844 | 0.6183 | 网格最优（Test 选点） |
| 32 | E155 | task_conditional_calibration | 0.80 | other | 0.9753 | 0.2868 | 0.9140 | 0.6157 |  |
| 33 | E134 | P96 | 0.85 | other | 0.7956 | 0.5769 | 0.4790 | 0.6071 | 网格最优（Test 选点） |
| 34 | E115 | v3_frozen_audit |  | other | 0.5783 | 0.4668 | 0.8863 | 0.5996 | 网格外工作点，仅记录 |
| 35 | E125 | band_fusion |  | other | 0.7221 | 0.6403 | 0.3831 | 0.5965 | 网格外工作点，仅记录 |
| 36 | E106 | loso_fusion_frozen | 0.90 | other | 0.5345 | 0.5176 | 0.8136 | 0.5958 |  |
| 37 | E102 | task_distance_ratio | 0.85 | other | 0.8427 | 0.2873 | 0.8432 | 0.5651 |  |
| 38 | E151 | identity_mahalanobis | 0.80 | other | 0.5509 | 0.6262 | 0.4429 | 0.5616 |  |
| 39 | E154 | robust_identity_embedding | 0.95 | other | 0.5602 | 0.3561 | 0.9470 | 0.5548 |  |
| 40 | E118 | c7_frozen_audit | 0.94 | other | 0.7362 | 0.3278 | 0.8090 | 0.5502 |  |
| 41 | E156 | invariant_feature_probe |  | other | 0.6459 | 0.7229 | 0.1001 | 0.5479 | 网格外工作点，仅记录 |
| 42 | E95 | e54_final20_candidate | 0.95 | other | 0.1543 | 0.9414 | 0.0955 | 0.5331 |  |
| 43 | E97 | multiview_session_invariant | 0.95 | other | 0.1012 | 0.9690 | 0.0545 | 0.5234 |  |
| 44 | E103 | external_unknown_calibration | 0.63 | other | 0.9291 | 0.1258 | 0.8860 | 0.5167 | 网格外工作点，仅记录 |
| 45 | E98 | dual_gate_operating_curve | 0.95 | other | 0.4517 | 0.7686 | 0.0750 | 0.5160 |  |

> 第 10–21 行是同一基线（E65 PCA-96 k=12，a=0.90）在不同实验里的复现，W 都是 0.6958/0.6959，不是 12 个独立方案。
> 表中 E152 的 S1=0、S2=1 说明分数恒定（全判为已注册），W=0.75 是恒等式的产物，不具判别意义。

### 2.2 invalid（污染或违规，不计入）

| 实验 | 方法/配置 | a | S1 | S2 | LOSO | W | 标记 |
|---|---|---:|---:|---:|---:|---:|---|
| E142 | cdf_fusion | 0.95 | 0.5987 | 0.8022 | 0.9475 | 0.7877 | loso_contaminated |
| E139 | feature_fusion | 0.96 | 0.5997 | 0.7888 | 0.9724 | 0.7875 | loso_contaminated |
| E141 | weighted_image_route | 0.97 | 0.5124 | 0.8290 | 0.9481 | 0.7796 | loso_contaminated |
| E135 | k3_TASK_RANK | 0.87 | 0.3963 | 0.9175 | 0.8301 | 0.7653 | test_statistic |
| E138 | d1 | 0.95 | 0.6159 | 0.7597 | 0.8244 | 0.7399 | loso_contaminated |
| E132 | submission_candidates |  | 0.5107 | 0.8518 | 0.6261 | 0.7101 | no_loso |
| E137 | multi_view | 0.90 | 0.5740 | 0.6133 | 0.9902 | 0.6977 | loso_contaminated |
| E140 | PCA128_raw_control | 0.90 | 0.8190 | 0.5458 | 0.7138 | 0.6561 | loso_contaminated |
| E136 | clean_three_way | 0.95 | 0.7043 | 0.6930 | 0.4819 | 0.6431 | loso_contaminated |

说明：

- E142/E139/E141 的 LOSO 高达 0.95–0.97，是「被评分块进了原型集合」造成的虚高；同一实验的 Test S1 只有 0.51–0.60，
  两者相差 0.35 左右甚至更多（0.349 / 0.373 / 0.436），本身就是污染的指纹。
- E135 的 0.7653 来自 Test 逐任务经验 CDF 校准（转导式推断），用户已判定犯规；其 RAW 版本（=E122 k3）只有 0.6286。
- E132 的三个候选只共享一个全局 LOSO 0.6261，与 cand_S2（S2 接受 0.8518）不是同一工作点，故 W 不计入；
  若强行代入得 0.7101，仅作参考。

---

## 3. (b) 三口径对比：0.71 是不是最高？

| 口径 | 定义 | 最大值 | 来源 | S1 / S2 / LOSO | 是否可信 |
|---|---|---:|---|---|---|
| **A** 固定 a=0.90，全部记录 | 预注册工作点，不挑接受率 | **0.7593** | E142_cdf_fusion | 0.7023 / 0.6689 / 0.9971 | 否（LOSO 污染） |
| **A** 固定 a=0.90，clean | 同上，排除污染/违规 | 0.7491 | E152_session_robust_scoring | 0.0000 / 0.9981 / 1.0000 | 否（退化：分数恒定） |
| **A** 固定 a=0.90，clean 且非退化 | 同上 + 排除退化 | **0.7171** | **E170_three_way_fusion_ceiling** | 0.6106 / 0.7813 / 0.6950 | 是（含配置级 Test 选择偏差） |
| **B** 0.80–0.96 网格取最大 W，全部 | 工作点由 Test 选出 | **0.7877** | E142_cdf_fusion | 0.5987 / 0.8022 / 0.9475 | 否（LOSO 污染） |
| **B** 同上，clean | | **0.7617** | **E168_subwindow_covariance** | 0.5813 / 0.7851 / 0.8953 | 有条件：协议 clean，但工作点由 Test 选出 |
| **C** clean 且非退化 | 不限接受率，只看有效记录 | **0.7617** | **E168_subwindow_covariance** | 0.5813 / 0.7851 / 0.8953 | 同上 |

### 口径 A 前 8 名（a=0.90，clean 非退化）

| 实验 | 配置 | S1 | S2 | LOSO | W |
|---|---|---:|---:|---:|---:|
| E170 | score 0.4*E65(64) + 0.1*logbp(48) + 0.5*E162(128) k=1 | 0.6106 | 0.7813 | 0.6950 | 0.7171 |
| E163 | score 0.50*E65(64) + 0.50*E162(64) k=1 | 0.6030 | 0.7697 | 0.7231 | 0.7164 |
| E168 | subwin_concat_plus_std_n4 PCA-128 k=12 | 0.6202 | 0.7449 | 0.7523 | 0.7156 |
| E169 | raw concat 96+32 -> PCA-96 k=12 | 0.8160 | 0.5977 | 0.7919 | 0.7008 |
| E157 | per_sample_calibration | 0.7969 | 0.5945 | 0.8147 | 0.7002 |
| E167 | raw concat 96+32 -> PCA-96 k=12 | 0.8098 | 0.5817 | 0.8257 | 0.6997 |
| E171 | E65 PCA-128 k=12 | 0.7428 | 0.6350 | 0.7727 | 0.6964 |
| E150 | s2_diagnosis | 0.7979 | 0.5960 | 0.7936 | 0.6959 |

### 口径 B 前 10 名（0.80–0.96 网格）

| 实验 | 配置 | a | S1 | S2 | LOSO | W | 有效性 |
|---|---|---:|---:|---:|---:|---:|---|
| E142 | cdf_fusion | 0.95 | 0.5987 | 0.8022 | 0.9475 | 0.7877 | loso_contaminated |
| E139 | feature_fusion | 0.96 | 0.5997 | 0.7888 | 0.9724 | 0.7875 | loso_contaminated |
| E141 | weighted_image_route | 0.96 | 0.5763 | 0.7866 | 0.9687 | 0.7795 | loso_contaminated |
| E135 | k3_TASK_RANK | 0.875 | 0.3963 | 0.9175 | 0.8301 | 0.7653 | test_statistic |
| E168 | subwin_concat_n4 PCA-256 k=3 | 0.800 | 0.5813 | 0.7851 | 0.8953 | 0.7617 | clean |
| E152 | session_robust_scoring | 0.950 | 0.0000 | 1.0000 | 1.0000 | 0.7500 | clean（退化） |
| E138 | d1 | 0.950 | 0.6159 | 0.7597 | 0.8244 | 0.7399 | loso_contaminated |
| E170 | score 0.5*E65(64) + 0.1*logbp(32) + 0.4*E162(64) k=1 | 0.880 | 0.6775 | 0.7210 | 0.7530 | 0.7181 | clean |
| E163 | score 0.50*E65(64) + 0.50*E162(64) k=1 | 0.900 | 0.6030 | 0.7697 | 0.7231 | 0.7164 | clean |
| E169 | spectral_shape | 0.899 | 0.8180 | 0.5966 | 0.7975 | 0.7022 | clean（oracle 阈值） |

**回答用户的问题**：

- 「0.71 是最高吗？」——**不是**。
- 「真实最高值多少？」——
  - 只看可信（clean、非退化、固定 a=0.90）：**0.7171 / E170**；
  - 放宽到「工作点由 Test 在 0.80–0.96 网格上选出」：**0.7617 / E168**；
  - 若把污染实验也算进来：0.7877 / E142（无效）。
- 「可信吗？」——**0.7617 不可信**：它是 Test 选点的产物，E168 报告自己写明该配置 Train OOF AUC 只有 0.7350（同变体其它配置最高 0.9448），
  W 的抬升来自阈值位置，不是表征提升；同实验固定 a=0.90 时最好只有 0.7156。
  **0.7171 相对可信**，但仍带「在数十个配置里按 Test W 挑最优」的模型选择偏差；若完全禁止 Test 侧参与任何选择，天花板回到 E143 的 **0.6958**。

---

## 4. CSV 补算明细

### 4.1 与已有记录同点匹配、从而补上 LOSO 的 CSV（18 个）

| 实验 | CSV | S1 | S2 | LOSO（同点） | W | 有效性 |
|---|---|---:|---:|---:|---:|---|
| E102 | predictions.csv | 0.8427 | 0.2873 | 0.8432 | 0.5651 | clean |
| E103 | predictions.csv | 0.9291 | 0.1258 | 0.8860 | 0.5167 | clean |
| E104 | predictions.csv | 0.4263 | 0.6150 | 0.8981 | 0.6386 | clean |
| E106 | predictions.csv | 0.5345 | 0.5176 | 0.8136 | 0.5958 | clean |
| E115 | predictions.csv | 0.5783 | 0.4668 | 0.8863 | 0.5996 | clean |
| E118 | predictions.csv | 0.7362 | 0.3278 | 0.8090 | 0.5502 | clean |
| E122 | predictions_k1.csv | 0.7916 | 0.5823 | 0.6261 | 0.6456 | clean |
| E122 | predictions_k3.csv | 0.9245 | 0.3799 | 0.8301 | 0.6286 | clean |
| E132 | cand_bal.csv | 0.6729 | 0.6647 | 0.6261（全局） | (0.6571) | no_loso |
| E132 | cand_S1.csv | 0.7933 | 0.5872 | 0.6261（全局） | (0.6484) | no_loso |
| E132 | cand_S2.csv | 0.5107 | 0.8518 | 0.6261（全局） | (0.7101) | no_loso |
| E135 | predictions_k1_RAW.csv | 0.7916 | 0.5823 | 0.6261 | 0.6456 | test_statistic |
| E135 | predictions_k1_TASK_RANK.csv | 0.4069 | 0.9099 | 0.8301 | 0.7132 | test_statistic |
| E135 | predictions_k3_RAW.csv | 0.9245 | 0.3799 | 0.8301 | 0.6286 | test_statistic |
| E135 | predictions_k3_TASK_RANK.csv | 0.3963 | 0.9175 | 0.8301 | 0.7653 | test_statistic |
| E95 | predictions.csv | 0.1543 | 0.9414 | 0.0955 | 0.5331 | clean |
| E97 | predictions.csv | 0.1012 | 0.9690 | 0.0545 | 0.5234 | clean |
| E98 | predictions.csv | 0.4517 | 0.7686 | 0.0750 | 0.5160 | clean |

括号里的 E132 数值只作参考：LOSO 是全局值，不是该工作点的值。

### 4.2 按 Session 置 0 的候选（必须不计入）

审计发现 **22 个** CSV 满足 `label==0 条数 == 3,033 == session1 条数`，即把 session1 整段判为未知：

| 位置 | 文件 | S1 | S2 | 处置 |
|---|---|---:|---:|---|
| `commit/answer/` | `赞美诗_第1次.csv` | 1.0000 | 1.0000 | **不计入**（session 规则） |
| `commit/answer/` | `赞美诗_第2次.csv` | 1.0000 | 1.0000 | **不计入**（session 规则） |
| `commit/answer/` | `submission_task_specific_zscore_grouped_guarded.csv` | 1.0000 | 1.0000 | 不计入 |
| `commit/review/results/` | `submission_task_specific_zscore_grouped_guarded.csv` | 1.0000 | 1.0000 | 不计入 |
| `research/E00_cov_ridge_baseline/` | sinkhorn×2、task_specific×3 共 5 个 | 1.0000 | 1.0000 | 不计入 |
| `research/E10_second_submission/` | 2 个 | 1.0000 | 1.0000 | 不计入 |
| `research/_trash/` | 11 个 | 1.0000 | 1.0000 | 不计入 |

这类提交若代入 W 会得到 `0.75 + 0.25×LOSO`，分母都来自 Session 字段而不是 EEG；按用户新约束（决赛不允许使用 session 字段），
它们只作为参考，不进入任何排序。`赞美诗_第3次.csv`（S1 0.5796 / S2 0.1742，`label=0` 17,629 条）是唯一 session-blind 的正式提交，等价于
`submission_session_blind_full_none_global.csv`，其 W 因缺同点 LOSO 记为 NA。

---

## 5. (c) 无法补算清单

177 个实验目录的覆盖情况：

| 状态 | 数量 | 说明 |
|---|---:|---|
| 已算出 W（可进入主排序表） | **54** | 45 clean + 9 invalid |
| 只有 Test CSV、无同点 LOSO | 32 | 已算出 S1/S2，W = NA |
| 既无三元组、也无可用 CSV | 90 | 下方逐条列原因 |
| 本任务目录 E174 | 1 | 无实验 |

### 5.1 有 Test CSV、但缺同一工作点 LOSO（32）

E00_cov_ridge_baseline、E02_open_set_improvement、E03_open_set_calibration、E05_asnorm_openset、
E09_raw_eeg_robust、E10_second_submission、E11_session_blind_candidate、E14_session_blind_effective_solution、
E15_session_blind_calibrated_v2、E16_session_blind_threshold_sweep_v3、E17_enrollment_template_session_robust_v1、
E18_enrollment_template_session_robust_v1_scaled、E19_ridge_session_robust_v1、E20_eeg_image_dinov2、
E24_norm_gate_robustness、E26_final_candidates、E27_unknown_rejection、E28_dino_openset、E29_dino_calibration、
E30_dino_multiview、E31_task_conditional、E32_distance_ratio、E33_arcface_head、E34_covariance_hr、
E35_per_task_analysis、E42_submission_candidates、E52_frozen_submission、E64_e58_test_audit、
E65_correlation_spd、E93_frozen_audit_only、E94_raw_identity_spd_gate、E96_domain_robust_open_set。

这些实验的 Test S1/S2 已算好，但它们的 report/results 只记录了 Train 外层代理（C/RA 代理、Train 拒识率），
没有在同一阈值上算 LOSO，因此 `LOSO = NA`。为了给出边界，下表用 `W(LOSO=0)` 与「达到 0.71 所需的 LOSO」表示：

| 实验 | CSV 数 | 该 CSV 的 S1 | 该 CSV 的 S2 | W(LOSO=0) | 达 0.71 所需 LOSO | 备注 |
|---|---:|---:|---:|---:|---:|---|
| E00_cov_ridge_baseline | 5 | 1.0000 | 1.0000 | 0.7500 | -0.160 | 含 Session 置 0 候选 |
| E02_open_set_improvement | 1 | 0.5931 | 0.1469 | 0.2217 | +1.953 |  |
| E03_open_set_calibration | 1 | 0.9476 | 0.0582 | 0.2660 | +1.776 |  |
| E05_asnorm_openset | 1 | 0.5862 | 0.3267 | 0.3099 | +1.600 |  |
| E09_raw_eeg_robust | 1 | 0.5691 | 0.2610 | 0.2728 | +1.749 |  |
| E10_second_submission | 2 | 1.0000 | 1.0000 | 0.7500 | -0.160 | 含 Session 置 0 候选 |
| E11_session_blind_candidate | 23 | 0.3831 | 0.6265 | 0.4091 | +1.204 |  |
| E14_session_blind_effective_solution | 2 | 0.2413 | 0.7741 | 0.4474 | +1.051 |  |
| E15_session_blind_calibrated_v2 | 1 | 0.5424 | 0.3526 | 0.3119 | +1.592 |  |
| E16_session_blind_threshold_sweep_v3 | 6 | 0.0458 | 0.9673 | 0.4951 | +0.860 |  |
| E17_enrollment_template_session_robust_v1 | 1 | 0.5028 | 0.4928 | 0.3721 | +1.352 |  |
| E18_enrollment_template_session_robust_v1_scaled | 1 | 0.0600 | 0.9504 | 0.4902 | +0.879 |  |
| E19_ridge_session_robust_v1 | 1 | 0.5908 | 0.1764 | 0.2359 | +1.896 |  |
| E20_eeg_image_dinov2 | 1 | 0.2404 | 0.6523 | 0.3862 | +1.295 |  |
| E24_norm_gate_robustness | 1 | 0.0712 | 0.5772 | 0.3064 | +1.614 |  |
| E26_final_candidates | 8 | 0.3093 | 0.9300 | 0.5423 | +0.671 |  |
| E27_unknown_rejection | 8 | 0.3093 | 0.9300 | 0.5423 | +0.671 |  |
| E28_dino_openset | 12 | 0.4352 | 0.9520 | 0.5848 | +0.501 |  |
| E29_dino_calibration | 8 | 0.4296 | 0.9548 | 0.5848 | +0.501 |  |
| E30_dino_multiview | 48 | 0.4804 | 0.9423 | 0.5913 | +0.475 |  |
| E31_task_conditional | 20 | 0.5087 | 0.9166 | 0.5855 | +0.498 |  |
| E32_distance_ratio | 35 | 0.5087 | 0.9166 | 0.5855 | +0.498 |  |
| E33_arcface_head | 5 | 0.2694 | 0.9256 | 0.5301 | +0.719 |  |
| E34_covariance_hr | 5 | 0.4557 | 0.9487 | 0.5883 | +0.487 |  |
| E35_per_task_analysis | 10 | 0.4510 | 0.9565 | 0.5910 | +0.476 |  |
| E42_submission_candidates | 2 | 0.5922 | 0.1564 | 0.2262 | +1.935 |  |
| E52_frozen_submission | 1 | 0.5875 | 0.2533 | 0.2735 | +1.746 |  |
| E64_e58_test_audit | 1 | 0.5803 | 0.2970 | 0.2936 | +1.666 |  |
| E65_correlation_spd | 1 | 0.4764 | 0.3938 | 0.3160 | +1.576 |  |
| E93_frozen_audit_only | 1 | 0.9993 | 0.0240 | 0.2618 | +1.793 |  |
| E94_raw_identity_spd_gate | 1 | 0.7270 | 0.5315 | 0.4475 | +1.050 |  |
| E96_domain_robust_open_set | 1 | 0.4069 | 0.8768 | 0.5401 | +0.680 |  |

（每个实验取「0.25×S1 + 0.5×S2」最大的那个 CSV。）

关键读数：只有 7 个实验（E28、E29、E30、E31、E32、E34、E35）在 LOSO ≥ 0.51 时就能过 0.71；
E26/E27 需要 LOSO ≥ 0.671，E96 需要 0.680。E00/E10 的 0.75 来自 Session 置 0，不计入；
其余 22 个实验即使 LOSO = 1.0 也到不了 0.71。
由于这些实验都没有记录同点 LOSO，一律 **不能** 宣布它们超过 0.71。

### 5.2 既无三元组、也无可用 CSV（90）

**只有 Train 侧 LOSO / 伪未知拒识，没有 Test S1/S2（13）**：
E99_joint_gate_feasibility、E100_relaxed_gate_feasibility、E101_unknown_proxy_sensitivity、E105_loso_candidate_comparison、
E108_loso_s1_boost、E110_loso_rejection_boost、E111_external_joint_embedding、E112_open_set_classifier、
E113_loso_finetune、E114_finetune_ablation、E117_dual_target_loso、E133_mean_objective、E144_retrospective。

**只有 Test 审计、没有 LOSO（1）**：E107_proxy_test_gap。

**LOSO 与 Test S1/S2 未在同一工作点对齐（2）**：
E130_per_identity_thresholds、E131_fusion_thresholds（两者都在多个阈值上分别记录，但编号不一一对应，无法确定配对）。

**无 LOSO、无 Test S1/S2、也无可用逐样本预测 CSV（74）**：
E01_brainprint_project_report、E04_pure_eeg_brainprint、E06_ea_riemannian_openset、E07_raw_eeg_defense、E08_raw_eeg_anchor、
E12_unified_full_none_vs_raw_pgd、E13_session_blind_shift_calibration、E21_scale_invariant、E22_cross_session_viability、
E23_spatial_reference、E25_robust_recognition、E28_plan_aggregation_whitening、E36_pure_eeg、E37_spd_adversarial、
E38_image_pure_fusion、E39_ecdf_transfer、E40_attack_rejection、E41_supervised_detector、E43_noise_robust_features、
E44_transfer_robustness、E45_raw_cnn_at、E46_embedding_rejection、E47_dual_model_system、E48_cnn_clean_tuning、
E49_three_region、E50_ensemble_at、E51_tta、E53_raw_eeg_open_robust、E54_outlier_exposure、E55_online_adversarial、
E56_cross_session_calibration、E57_self_normalized_score、E58_oe_three_region、E59_online_three_region、
E60_test_audit_diagnosis、E61_feature_screen、E62_multiwindow_image、E63_adaptive_fusion、E66_neuroshield_frozen、
E67_test_root_cause、E68_acquisition_invariance、E69_image_adversarial、E70_image_adversarial_training、
E71_covmix_geometry、E72_pair_protocol、E73_external_multisession、E74_multiscale_native22、E75_image_template、
E76_open_set_head、E77_score_fusion、E78_identity_calibration、E79_pair_open_set、E80_robust_open_set、E81_evt_open_set、
E82_unknown_calibration、E83_gate_feasibility、E84_hard_negative_pair、E85_test_csv_audit、E86_external_outer_csv、
E87_channel_mask_hard_unknown、E88_spatial_channel_projection、E89_hard_unknown_calibration、E90_project18_identity_template、
E91_hybrid_identity_unknown、E92_frozen_project_candidate、E94_robust_raw_template、E116_test_separability、
E119_score_battery、E120_domain_adaptation、E121_pca_sweep、E124_cluster_structure、E146_phase_features、
E160_principle_review、E173_first_submission_audit。

其中 E119（原型污染）、E160/E173（复审/审计文档）本来就没有可引用的候选；其余多为：
早期只报 Train 内部指标（E04–E08、E21–E25）、外部数据实验只产出 OOF 预测（E73–E92）、
分析类实验（E60、E67、E85、E86、E116、E144、E160、E173）不产出提交。

---

## 6. (d) 结论

1. **0.71 不是历史最高。** 历史上有 4 个 clean 实验超过它：E170（0.7171@a=0.90）、E163（0.7164@a=0.90）、
   E168（0.7156@a=0.90 / 0.7617@grid-best）、E152（0.75，但退化不可用）。
2. **可信的最高值：**
   - 预注册 a=0.90、clean、非退化：**W = 0.7171**（E170，三路分数融合 `0.4*E65(64)+0.1*logbp(48)+0.5*E162(128) k=1`）。
   - 允许 Test 在 0.80–0.96 网格选工作点：**W = 0.7617**（E168，`subwin_concat_n4 PCA-256 k=3`）。
3. **0.7617 不可作为方案成绩。** E168 自己写明这是 Test 选点 + 阈值位置放大的结果
   （该配置 Train OOF AUC 仅 0.7350，同变体其它配置可达 0.9448）；固定 a=0.90 时 E168 只有 0.7156。
4. **污染/违规的上限更高但无效**：E142 0.7877、E139 0.7875、E141 0.7796（LOSO 污染），E135 0.7653（Test CDF）。
   把这些数当历史最好会严重误导——它们的 LOSO 比 Test S1 高 0.35 以上。
5. **老实验（E00–E120）没有可确认的 W > 0.71 证据**：32 个能算 S1/S2 的实验全部缺同点 LOSO，
   其中只有 7 个在 LOSO ≥ 0.51 的假设下可能过线，但这是假设不是测量。剩余 90 个实验根本没有 Test 逐样本记录。
6. **对「0.71 是否最高」的直接回答**：不是最高；真实最高是 **0.7617（E168，clean 但 Test 选点）**，
   严肃口径下是 **0.7171（E170，clean，a=0.90）**；两者都只比 0.71 高 0.007–0.05，
   且都含 Test 侧参与模型/工作点选择的偏差——如果完全排除 Test 选择，天花板仍是 **0.6958**。

---

## 7. 复现

```powershell
.\.venv\Scripts\python.exe research\E174_recompute_all_w\audit_csv_w.py        # CSV → S1/S2（csv_audit.json）
.\.venv\Scripts\python.exe research\E174_recompute_all_w\extract_pairs.py      # JSON 三元组（json_pairs.json）
.\.venv\Scripts\python.exe research\E174_recompute_all_w\extract_reported.py   # best/ref/090/curve 家族（reported_w.json）
.\.venv\Scripts\python.exe research\E174_recompute_all_w\extract_report_tables.py  # report.md 表格（report_tables.json）
.\.venv\Scripts\python.exe research\E174_recompute_all_w\build_master.py       # 合并 + 打标（master_table.json）
.\.venv\Scripts\python.exe research\E174_recompute_all_w\find_gaps.py          # 无法补算清单（gaps.json）
.\.venv\Scripts\python.exe research\E174_recompute_all_w\make_report_tables.py # 本报告表格
```

产物：`csv_audit.json`、`json_pairs.json`、`reported_w.json`、`report_tables.json`、`master_table.json`、`gaps.json`、`all_records.json`（上一轮扫描）。
所有脚本只读已有实验，不写任何实验目录。

## 8. 已知局限

- 60 余个实验的 Test 数字只以自然语言散落在 report.md 正文（非表格），本报告未做 NLP 抽取，可能仍有少量同点三元组未被识别。
- `report.md` 表格抽取按列名启发式匹配，已人工核对进入前 20 名的每一行；20 名以外的行未逐条人工复核。
- E169/E171/E159 的「最优 W」是扫满阈值得到的 oracle 值（`oracle_point`），标注为 Test 选点；
  E171 报告里 0.8033 的「最松口径」是把 LOSO 与 Test 工作点分别取各自最优的**不可达上界**，未计入任何排序。
- 本报告不涉及模型训练，无 GPU 使用；Test 官方 AUC/F1 一律 N/A。
