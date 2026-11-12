# E36 纯 EEG 五频带基线、嵌套开放集与原始波形攻击结果

日期：2026-11-12。仅使用 Train Session1；未读取、推理或调参使用 Test。

## 冻结方法与复现

- 五个频带：1–8、8–13、13–20、20–30、30–40 Hz；逐 Trial FFT 隔离频带，GPU 协方差、0.05 固定收缩、logSPD、上三角向量化（非对角乘 sqrt(2)）。
- 四候选：raw Ridge、detrend+zband Ridge、detrend+zband 原型、detrend+zband WCCN 原型。Ridge alpha=100；WCCN 仅用训练折。
- 外层 RNG42 五折留被试，每折 4 人 unknown；内层 RNG=100+外层折号（折号从 1），16 人两折。known 折外评估为 subject/task 内后 20% 连续文件块代理。
- 内层对每种分数搜索 101 个阈值，约束 known acceptance≥0.95，最大化 `100*C^0.40*U^0.25`。这是缺少 A 项的选择代理，不是官方分数；同时冻结 q05 工作点。
- AUC 指 unknown-detection AUROC，较大 confidence 表示 known；unknown AUPR 使用 `-confidence`。身份 F1 与 unknown 类 F1 分开报告。

实际命令：

```powershell
& "d:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "d:\WORK\BCIC2026\BrainPrint\research\E36_pure_eeg\run_e36.py" --stage features --rebuild-features
& "d:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "d:\WORK\BCIC2026\BrainPrint\research\E36_pure_eeg\run_e36.py" --stage nested
```

配置：`D:\WORK\BCIC2026\BrainPrint\research\E36_pure_eeg\config.json`；完整逐折数值：`D:\WORK\BCIC2026\BrainPrint\research\E36_pure_eeg\results.json`；特征证据：`D:\WORK\BCIC2026\BrainPrint\research\E36_pure_eeg\feature_build.json`。

## 特征与逐样本不变性

- 特征形状：[11304, 855]，raw/zband 均 finite=True。
- 缓存重建最大绝对差：0；单条/批次：0；置换/批次：0。

| 候选 | 单条/批次最大分数差 | 置换/批次最大分数差 | 标签/分数容差检查 |
|---|---:|---:|---|
| raw_cov5_ridge | 5.96e-07 | 0 | 通过 |
| detrend_zband_cov5_ridge | 1.07e-06 | 0 | 通过 |
| detrend_zband_cov5_prototype | 3.58e-07 | 1.79e-07 | 通过 |
| detrend_zband_cov5_wccn_prototype | 2.98e-07 | 1.19e-07 | 通过 |

## 外层五折均值与 pooled 指标

direct C 是不拒识的连续块身份准确率；C/U/unknown F1 是内层所选工作点；U@inner95 使用内层已知置信度 q05。joint 21 类 macro-F1 在跨折 pooled 的固定标签 0–20 上计算。

| 候选 | Train fit | direct C | C | 身份F1 direct | 身份F1+拒识 | AUROC | unknown AUPR | U@inner95 | 工作点U | unknown F1 | joint 21类F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_cov5_ridge | 0.9999 | 0.9997 | 0.9620 | 0.9997 | 0.9799 | 0.9973 | 0.9977 | 0.9964 | 0.9889 | 0.9792 | 0.9737 |
| detrend_zband_cov5_ridge | 0.9998 | 0.9996 | 0.9637 | 0.9996 | 0.9808 | 0.9967 | 0.9966 | 0.9956 | 0.9818 | 0.9760 | 0.9710 |
| detrend_zband_cov5_prototype | 0.9188 | 0.8772 | 0.8525 | 0.8774 | 0.8752 | 0.8539 | 0.8242 | 0.3136 | 0.3117 | 0.4344 | 0.6513 |
| detrend_zband_cov5_wccn_prototype | 0.9999 | 0.9998 | 0.9726 | 0.9998 | 0.9855 | 0.9979 | 0.9975 | 0.9989 | 0.9945 | 0.9862 | 0.9825 |

Pooled AUROC/AUPR 与逐折均值可能不同；完整 JSON 同时保存两者、fold std、EER、known AUPR、known acceptance 和逐折阈值。

## 每折工作点结果

| 候选 | 折 | unknown subjects | 分数/阈值来源 | direct C | C | U | AUROC | unknown F1 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| raw_cov5_ridge | 1 | 16,10,15,8 | margin / inner | 1.0000 | 0.9675 | 0.9967 | 0.9990 | 0.9841 |
| detrend_zband_cov5_ridge | 1 | 16,10,15,8 | margin / inner | 1.0000 | 0.9633 | 0.9986 | 0.9990 | 0.9833 |
| detrend_zband_cov5_prototype | 1 | 16,10,15,8 | max_cosine / inner | 0.8749 | 0.8562 | 0.1931 | 0.8307 | 0.3167 |
| detrend_zband_cov5_wccn_prototype | 1 | 16,10,15,8 | distance_ratio / inner | 1.0000 | 0.9712 | 0.9939 | 0.9985 | 0.9843 |
| raw_cov5_ridge | 2 | 13,11,7,20 | max / inner | 0.9994 | 0.9487 | 0.9930 | 0.9977 | 0.9777 |
| detrend_zband_cov5_ridge | 2 | 13,11,7,20 | max / inner | 0.9994 | 0.9653 | 0.9967 | 0.9981 | 0.9855 |
| detrend_zband_cov5_prototype | 2 | 13,11,7,20 | distance_ratio / inner | 0.8858 | 0.8571 | 0.4786 | 0.8974 | 0.6234 |
| detrend_zband_cov5_wccn_prototype | 2 | 13,11,7,20 | distance_ratio / inner | 0.9994 | 0.9713 | 0.9951 | 0.9983 | 0.9869 |
| raw_cov5_ridge | 3 | 4,1,17,6 | margin / inner | 0.9995 | 0.9592 | 0.9983 | 0.9983 | 0.9834 |
| detrend_zband_cov5_ridge | 3 | 4,1,17,6 | margin / inner | 0.9995 | 0.9548 | 1.0000 | 0.9982 | 0.9826 |
| detrend_zband_cov5_prototype | 3 | 4,1,17,6 | max_cosine / inner | 0.8727 | 0.8498 | 0.0620 | 0.7800 | 0.1135 |
| detrend_zband_cov5_wccn_prototype | 3 | 4,1,17,6 | distance_ratio / inner | 0.9995 | 0.9695 | 0.9996 | 0.9977 | 0.9880 |
| raw_cov5_ridge | 4 | 12,19,3,5 | max / inner | 0.9995 | 0.9606 | 0.9621 | 0.9929 | 0.9645 |
| detrend_zband_cov5_ridge | 4 | 12,19,3,5 | max / inner | 0.9995 | 0.9725 | 0.9211 | 0.9912 | 0.9476 |
| detrend_zband_cov5_prototype | 4 | 12,19,3,5 | distance_ratio / inner | 0.8726 | 0.8461 | 0.3947 | 0.8775 | 0.5423 |
| detrend_zband_cov5_wccn_prototype | 4 | 12,19,3,5 | distance_ratio / inner | 1.0000 | 0.9725 | 0.9910 | 0.9971 | 0.9841 |
| raw_cov5_ridge | 5 | 18,2,14,9 | margin / inner | 1.0000 | 0.9741 | 0.9946 | 0.9985 | 0.9865 |
| detrend_zband_cov5_ridge | 5 | 18,2,14,9 | margin / inner | 0.9995 | 0.9627 | 0.9927 | 0.9969 | 0.9810 |
| detrend_zband_cov5_prototype | 5 | 18,2,14,9 | distance_ratio / inner | 0.8801 | 0.8531 | 0.4299 | 0.8840 | 0.5761 |
| detrend_zband_cov5_wccn_prototype | 5 | 18,2,14,9 | distance_ratio / inner | 1.0000 | 0.9784 | 0.9932 | 0.9979 | 0.9876 |

## 量化门禁（描述性比较，不据外层重新选模）

| 候选 | Train fit=100% | direct C≥.95 | 身份F1 direct≥.92 | 连续块≥.90 | AUROC≥.98 | U@inner95≥.95 |
|---|---|---|---|---|---|---|
| raw_cov5_ridge | 未通过 | 通过 | 通过 | 通过 | 通过 | 通过 |
| detrend_zband_cov5_ridge | 未通过 | 通过 | 通过 | 通过 | 通过 | 通过 |
| detrend_zband_cov5_prototype | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 |
| detrend_zband_cov5_wccn_prototype | 未通过 | 通过 | 通过 | 通过 | 通过 | 通过 |

## E36A/B 证据边界

- 所有 unknown 都是 Train 留被试 pseudo-unknown，不等于官方 5 名新被试；连续文件块顺序只是代理，不是真实跨天或跨 Session 证据。
- 内层与外层注册类别数不同，置信度尺度迁移仍是限制；四候选的外层结果只作冻结协议报告，不用于回头调参。
- 上表是此前已完成且已查看的 E36A/B 开发结果；其阈值使用 C/U 代理。E36C 下方才使用包含六攻击 A 项的完整本地 Score 冻结阈值。
- Train fit 100% 若出现，仅是容量/流程检查，不是泛化证据。

# E36C 原始 EEG 波形攻击

## 预注册协议与复现

- 仅使用 Train session1；最终模型为 detrend+zband 五频带 WCCN 原型。内层预注册 shrinkage={0.0001, 0.001, 0.01, 0.05}，按内层 clean 指标和全 Train 拟合约束选择 0.05；未使用外层选参。
- 固定分层子集覆盖每折、每身份、每任务：按 trial/path 排序，每组最多取前 8 条（不足 8 条则全取），不依据结果挑样本。外层第 1–4 折每攻击 known N=509，第 5 折 N=512；每折 unknown N=128。
- 六攻击为非目标 FGSM、非目标 PGD、目标 PGD（目标为排序后的下一注册身份），各含 SNR30/20 dB；PGD=20 步。L2 半径为每通道时间去均值后整条干净波形范数除以 `10^(SNR/20)`，PGD 使用随机起点、L2 归一化梯度及逐步投影；FGSM 到达半径边界。
- 攻击微分链严格为冻结的 `gpu_logspd -> detrend_zband -> WCCN`，使用 GPU float64 特征反向计算。所有梯度/分数有限；外层实际 SNR 不低于指定值，PGD20 折均值为 20.02–20.05 dB。
- 每折仅用对应 inner clean、六攻击与 pseudo-unknown，在 clean known acceptance≥0.95 下联合选择 confidence 与阈值，最大化 `100*C^0.40*A^0.35*U^0.25`；outer 只应用冻结规则。保留 inner clean q05 的 U@95。
- 表中 C 来自同子集 clean known，RA/ASafeR/A/身份 F1 来自 attacked known，U 来自 clean pseudo-unknown；AUROC/unknown F1 使用 attacked known 对 clean pseudo-unknown。Score 为本地完整 Score，不是官方 Test 分数。

实际命令：

```powershell
& "d:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "d:\WORK\BCIC2026\BrainPrint\research\E36_pure_eeg\run_attacks.py" --stage inner
& "d:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "d:\WORK\BCIC2026\BrainPrint\research\E36_pure_eeg\run_attacks.py" --stage outer
```

冻结配置：`attack_config.json`；逐折、逐攻击完整数值与诊断：`attack_results.json`；实现：`run_attacks.py`。

## 内层选择、严格拟合与唯一一次改进

- 四种 shrinkage 的全 Train 拟合均为 0.9999115（各 1/11304 条错误）；所选 0.05 经 float64 复核仍有 1 条错误，因此严格 100% 未达成，并非浮点精度造成。选择 shrinkage=0.05 是内层 C/U 代理最高，不是外层选优。
- 最终 baseline 内层 clean：direct C=0.999238，身份 macro-F1=0.999233，unknown AUROC=0.996177，U@95=0.992717，四项门禁全部通过。
- 内层六攻击宏：C=0.965893，RA=0.038739，ASafeR=0.594015，A=0.205321，U=0.993750，Score=56.559；目标攻击接受后误认证率宏均值=0.826687。
- 因鲁棒性及严格拟合门禁未达，触发预注册的唯一一次单变量改进：fit 内原始 PGD20/SNR20 + FGSM/SNR30 对抗增广。增广后内层 RA=0.084065、ASafeR=0.576932、Score=55.736；clean AUROC=0.975046、U@95=0.882474，且全 Train 拟合降至 0.997169。按冻结规则保留 baseline，不进行第二轮调参。

## 冻结外层 clean 全量结果

known N=9232、unknown N=11304（五折合计）。direct C=0.999781，direct 身份 macro-F1=0.999782，unknown AUROC=0.997878；攻击感知阈值下 C=0.949872、身份 macro-F1=0.972922、unknown F1=0.979312、U=0.998452；q05 的 U@95=0.998993。

## 冻结外层逐攻击与宏汇总

以下数值为五折均值；每攻击 known/adversarial N=2548、unknown N=640。宏行 adversarial N=15288，unknown 评价累计 N=3840。目标误认证为接受后 `pred==target`，括号内为不拒识 direct 误认证。

| 攻击 | C | RA | ASafeR | A | 身份 macro-F1 | unknown AUROC | unknown F1 | U | 完整本地 Score | 目标误认证 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FGSM SNR30 | 0.939540 | 0.066710 | 0.999607 | 0.346579 | 0.079495 | 0.521199 | 0.350103 | 1.000000 | 67.294 | N/A |
| FGSM SNR20 | 0.939540 | 0.042374 | 0.999607 | 0.329544 | 0.045755 | 0.414103 | 0.344249 | 1.000000 | 66.118 | N/A |
| PGD SNR30 | 0.939540 | 0.066322 | 0.966250 | 0.336300 | 0.077695 | 0.583810 | 0.358488 | 1.000000 | 66.589 | N/A |
| PGD SNR20 | 0.939540 | 0.035315 | 0.639868 | 0.216681 | 0.040902 | 0.759944 | 0.457876 | 1.000000 | 57.035 | N/A |
| 目标PGD SNR30 | 0.939540 | 0.120879 | 0.445140 | 0.218157 | 0.119617 | 0.928609 | 0.609848 | 1.000000 | 57.189 | 0.554860（direct 0.749540） |
| 目标PGD SNR20 | 0.939540 | 0.033743 | 0.040414 | 0.035745 | 0.025133 | 0.998924 | 0.986938 | 1.000000 | 26.303 | 0.959586（direct 0.962332） |
| **宏平均** | **0.939540** | **0.060891** | **0.681814** | **0.247168** | **0.064766** | **0.701098** | **0.517917** | **1.000000** | **59.784** | **0.757223（direct 0.855936）** |

宏 Score 由宏 C/A/U 重新计算；六攻击 Score 的算术均值为 56.754。系统并非全拒识：宏 direct RA=0.122402，阈值后 RA=0.060891。

## 达标判定与证据边界

| 门禁 | 结果 |
|---|---|
| 全 Train 严格拟合=100% | **未通过**：0.9999115，1/11304 错误 |
| 内层 direct C≥0.95、身份 F1≥0.92、AUROC≥0.98、U@95≥0.95 | **全部通过** |
| 内层 RA≥0.95、ASafeR≥0.98 | **未通过**：0.038739 / 0.594015 |
| 外层 RA≥0.95、ASafeR≥0.98（描述性确认） | **未通过**：0.060891 / 0.681814 |
| 不能全拒识 | **通过**：all_rejection=False |

- 外层基线结果此前已经查看，因此本次外层属于冻结规则的确认性/限制性证据，不能称为 untouched validation；未据外层调整 shrinkage、攻击、增广或阈值。
- unknown 仍为 Train 留被试 pseudo-unknown，文件后 20% 为连续块代理；不是跨 Session 或官方新被试证据。
- Test AUC、F1、C、RA、ASafeR、A、U、Score：N/A；Test 未读取、未推理、未用于调参。
