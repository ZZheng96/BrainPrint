# E72：配对协议正确性审计与模板匹配基线

## 结论

- 已完成 **5 个外层留被试折 × 每折 2 个内层标定折**；正式比较仅使用纯 EEG 协方差与相关 logSPD 缓存。外层 known 连续块累计 N=9232，outer unknown 全 Trial 累计 N=11304。
- 预注册出口为 C≥0.95、AUROC≥0.98、U@95≥0.95。满足三项的候选：**e36_covariance/wccn/margin, e65_correlation/wccn/margin**。这里只作冻结候选的描述性确认，不依据外层重新选模或调阈值。
- Test AUC、F1、C、RA、ASafeR、A、U、Score：**N/A**；无公开真值，且本实验未读取 Test 数据、索引或标签，也未生成 Test 预测。
- 官方 Score：**N/A**，因为 clean-only 实验没有 RA、ASafeR、A。表中 `100*C^0.40*U^0.25` 仅为明确标注的非官方 C-U proxy，不能当作官方 Score。

## 实际执行命令

```powershell
.venv/Scripts/python.exe research/E72_pair_protocol/test_pair_protocol.py
.venv/Scripts/python.exe research/E72_pair_protocol/run_e72.py --stage audit
.venv/Scripts/python.exe research/E72_pair_protocol/run_e72.py --stage smoke
.venv/Scripts/python.exe research/E72_pair_protocol/run_e72.py --stage full
```

完整验证耗时：2.57 秒。逐折模型已保存至 `research\E72_pair_protocol\outer_templates.pt`。

## 数据、配对与隔离口径

- 仅 Train/session1：20 人、11,304 trials、18 通道、1,200 点、300 Hz，任务 EC/EO/Imagery/SSVEP。任务和时间块只构造划分，不作为推理输入。
- RNG42 将 20 人分成五组，每组 4 人 outer unknown；其余 16 人以 RNG=100+outer 分成两组 8 人。每内折 enrolled=其余 8 人，8 名留出者全部只作 calibration unknown；**没有 OE 角色**。
- inner fit：enrolled 且 trial≤0.60×subject/task 内最大 trial；inner known query：0.60–0.80；inner unknown：8 名留出者的 ≤0.80 数据。
- outer 模板：16 名 enrolled 的 ≤0.80 数据；outer known query：**trial>0.80×最大值的连续末端文件块**；outer unknown：4 人全部 trials，仅最终评估。
- 配对为查询样本对全部注册模板：每个 known query 恰有一个 same pair，其余为 different；unknown query 的所有模板对均为 different。没有物化 trial×trial 的 N² 矩阵。
- 每个候选阈值只取对应 outer group 内全部 inner-known confidence 的 q05（linear）；未扫描阈值，未使用 inner/outer unknown 调阈值。
- 全部已执行折的数据隔离断言通过：**True**。

## 输入正确性与哈希审计

| 缓存 | shape | dtype | finite | 重算最大绝对差 | 行对齐 | SHA256 |
|---|---|---|---|---:|---|---|
| e36_covariance | [11304, 855] | float32 | True | 4.77e-07 | True | `fdc68441c1099ce57128cb892211eb7bb3c9c5dbde9beeed2789b519e8564f27` |
| e65_correlation | [11304, 855] | float32 | True | 4.77e-07 | True | `bc9a1c32bac175498d95f2f2d7641be13b7fe880550fcc792fe675fdef4b43f6` |

- Train metadata SHA256：`f3be516e93082b78a9f483f9edb92f6330b1be39d32dff999a0a60b9bb92b545`；两个正式缓存均用抽样 Train 原始波形在 GPU 逐 Trial 重算并核对行对齐。
- 图像直接编码审计：shape=[3, 224, 224]、dtype=float32、finite=True、范围=[0, 1]，通过=True。
- 旧图像缓存为 `uint8`，抽样范围 [0, 255]，不能直接当作 float32 [0,1] 输入。历史 E20 提取源码在 DINO 前有 `/255.0`；E72 未修改旧缓存。
- 共享图像编码器频带为 [[1.0, 8.0], [8.0, 13.0], [13.0, 30.0], [30.0, 45.0], [45.0, 90.0]]，其中 [[30.0, 45.0], [45.0, 90.0]] 超出官方 1–40 Hz；因此旧 DINO 仅用于 smoke，不进入正式结论。

## 内层标定结果（10 inner runs 均值）

C/U/F1 使用同一 pooled inner-known q05；inner U@95 则按各 inner run 自己的 known q05 计算，两者不必完全相等。内层结果只用于协议检查，不用于从外层结果反向选模。

| 特征 / 模板空间 / 置信度 | direct C | C | 身份 macro-F1 | AUROC | U@95 | 工作点 U | unknown-F1 | 非官方 C-U proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e36_covariance/cosine/max_cosine | 0.911287 | 0.876917 | 0.897692 | 0.857697 | 0.339400 | 0.296176 | 0.447607 | 69.506207 |
| e36_covariance/wccn/max_cosine | 0.999241 | 0.948994 | 0.972296 | 0.932029 | 0.636825 | 0.597436 | 0.717483 | 84.652107 |
| e36_covariance/wccn/margin | 0.999241 | 0.949695 | 0.973265 | 0.994831 | 0.983481 | 0.981189 | 0.984031 | 97.483483 |
| e36_covariance/wccn/distance_ratio | 0.999241 | 0.949440 | 0.973241 | 0.996009 | 0.990789 | 0.992696 | 0.989947 | 97.761978 |
| e65_correlation/cosine/max_cosine | 0.938892 | 0.896189 | 0.915454 | 0.820122 | 0.335241 | 0.297276 | 0.447932 | 70.132204 |
| e65_correlation/wccn/max_cosine | 0.998905 | 0.948820 | 0.970382 | 0.838295 | 0.416635 | 0.346526 | 0.485557 | 73.137857 |
| e65_correlation/wccn/margin | 0.998905 | 0.949752 | 0.973022 | 0.991266 | 0.966317 | 0.947607 | 0.966108 | 96.604557 |
| e65_correlation/wccn/distance_ratio | 0.998905 | 0.949400 | 0.972856 | 0.993568 | 0.983676 | 0.983445 | 0.985238 | 97.525875 |

## 外层五折均值（正式结果）

AUROC 以 known=1、unknown=0，confidence 越大越像 known。`U@95` 严格复用 `src.metrics.open_set_metrics`：在相应 outer known confidence 的 q05 下计算 unknown 拒识率；C、U、身份 macro-F1 和 unknown-F1 则使用内层冻结 q05 工作阈值。direct C 即连续时间块不拒识身份准确率。

| 特征 / 模板空间 / 置信度 | direct C | C | 身份 macro-F1 | AUROC | U@95 | 工作点 U | unknown-F1 | 非官方 C-U proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e36_covariance/cosine/max_cosine | 0.877203 | 0.853573 | 0.868635 | 0.840040 | 0.256272 | 0.211781 | 0.313419 | 58.658690 |
| e36_covariance/wccn/max_cosine | 0.999781 | 0.916833 | 0.953028 | 0.935287 | 0.657418 | 0.737022 | 0.813225 | 89.290845 |
| e36_covariance/wccn/margin | 0.999781 | 0.959792 | 0.978431 | 0.997519 | 0.997877 | 0.994803 | 0.981251 | 98.242550 |
| e36_covariance/wccn/distance_ratio | 0.999781 | 0.942656 | 0.968890 | 0.997878 | 0.998620 | 0.998899 | 0.976613 | 97.637473 |
| e65_correlation/cosine/max_cosine | 0.913535 | 0.883190 | 0.895471 | 0.788622 | 0.277027 | 0.244637 | 0.363498 | 64.519391 |
| e65_correlation/wccn/max_cosine | 0.999016 | 0.941882 | 0.966688 | 0.813700 | 0.360992 | 0.388215 | 0.537414 | 76.813997 |
| e65_correlation/wccn/margin | 0.999016 | 0.958270 | 0.977737 | 0.993833 | 0.980130 | 0.972043 | 0.968918 | 97.605105 |
| e65_correlation/wccn/distance_ratio | 0.999016 | 0.932914 | 0.963442 | 0.995357 | 0.990918 | 0.994614 | 0.970661 | 97.126224 |

### 三项出口门禁

| 候选 | C≥0.95 | AUROC≥0.98 | U@95≥0.95 | 全部 |
|---|---|---|---|---|
| e36_covariance/cosine/max_cosine | 未通过 | 未通过 | 未通过 | 未通过 |
| e36_covariance/wccn/max_cosine | 未通过 | 未通过 | 未通过 | 未通过 |
| e36_covariance/wccn/margin | 通过 | 通过 | 通过 | 通过 |
| e36_covariance/wccn/distance_ratio | 未通过 | 通过 | 通过 | 未通过 |
| e65_correlation/cosine/max_cosine | 未通过 | 未通过 | 未通过 | 未通过 |
| e65_correlation/wccn/max_cosine | 未通过 | 未通过 | 未通过 | 未通过 |
| e65_correlation/wccn/margin | 通过 | 通过 | 通过 | 通过 |
| e65_correlation/wccn/distance_ratio | 未通过 | 通过 | 通过 | 未通过 |

### WCCN + distance_ratio 逐外层折

| 折 | outer unknown | 特征 | C | 身份 macro-F1 | AUROC | U@95 | U |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 16,10,15,8 | e36_covariance | 0.939297 | 0.966431 | 0.998481 | 0.997167 | 0.998111 |
| 1 | 16,10,15,8 | e65_correlation | 0.946219 | 0.970327 | 0.993736 | 0.980170 | 0.986780 |
| 2 | 13,11,7,20 | e36_covariance | 0.929399 | 0.961953 | 0.998287 | 1.000000 | 1.000000 |
| 2 | 13,11,7,20 | e65_correlation | 0.933811 | 0.964135 | 0.994780 | 0.994641 | 0.997115 |
| 3 | 4,1,17,6 | e36_covariance | 0.943417 | 0.969357 | 0.997673 | 1.000000 | 1.000000 |
| 3 | 4,1,17,6 | e65_correlation | 0.929815 | 0.961440 | 0.996591 | 0.997434 | 0.999145 |
| 4 | 12,19,3,5 | e36_covariance | 0.943305 | 0.969161 | 0.997077 | 0.997294 | 0.997745 |
| 4 | 12,19,3,5 | e65_correlation | 0.909827 | 0.951165 | 0.996397 | 0.993685 | 0.998647 |
| 5 | 18,2,14,9 | e36_covariance | 0.957861 | 0.977548 | 0.997872 | 0.998639 | 0.998639 |
| 5 | 18,2,14,9 | e65_correlation | 0.944895 | 0.970144 | 0.995281 | 0.988662 | 0.991383 |

## Smoke（非正式证据）

- smoke 仅执行第一个 outer group、第一个 inner split，状态 `complete_smoke`；配对接口审计：{'feature': 'e36_covariance', 'N_pairs': 7600, 'N_same': 950, 'N_different': 6650, 'one_same_template_per_known_query': True, 'trial_by_trial_N_squared_materialized': False}。
- 下表旧 DINO 数字只证明流程可运行，**不得作为新结论或与正式纯 EEG 结果比较**。

| 特征 / 模板空间 / 置信度 | direct C | C | 身份 macro-F1 | AUROC | U@95 | 工作点 U | unknown-F1 | 非官方 C-U proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_dinov2_covariance5/cosine/max_cosine | 0.589989 | 0.566028 | 0.568431 | 0.581530 | 0.058546 | 0.057602 | 0.104586 | 39.016072 |
| legacy_dinov2_covariance5/wccn/max_cosine | 0.972843 | 0.927583 | 0.947499 | 0.619002 | 0.064684 | 0.063739 | 0.115139 | 48.757674 |
| legacy_dinov2_covariance5/wccn/margin | 0.972843 | 0.894036 | 0.938431 | 0.947554 | 0.630784 | 0.881020 | 0.893037 | 92.637849 |
| legacy_dinov2_covariance5/wccn/distance_ratio | 0.972843 | 0.914271 | 0.948139 | 0.933309 | 0.499528 | 0.699717 | 0.790189 | 88.238973 |

## 证据边界与问题

- 所有 unknown 都是 Train 留被试 pseudo-unknown；连续末端文件块只是同 Session 时间隔离代理，不是真实跨天/跨 Session 证据，不能推出官方 Test 门禁可达。
- Cosine/WCCN 决定模板空间，max cosine、top1−top2 margin、distance_ratio 决定置信度；所有候选使用完全相同的被试折、时间块和 q05 规则。
- 完整逐折阈值、样本数、身份/未知/联合 F1、AUPR、fold std、pooled 指标、拟合准确率及角色 ID 见 `results.json`；缓存与图像细节见 `input_manifest.json`。
- 阶段出口由两路 WCCN+margin 达到；其余未通过候选不再扫阈值。Train 出口通过不等于跨 Session 问题解决，后续仍需真实跨 Session 监督与独立验证。
