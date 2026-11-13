# E65：逐通道相关矩阵 logSPD / WCCN（E64 拒识表征修正）

日期：2026-11-13。E64 的 S1 拒识/S2 接受失败只用于确定表征问题；未用于任何 Test 阈值或变体选择。

## 结论与当前 AUC / F1

Correlation inner 五外层折均值：**direct C=0.9888，threshold C=0.9734，direct身份F1=0.9888，含拒识身份F1=0.9841，clean AUROC=0.9951，unknown AUPR=0.9879，U@95=0.9891，工作点U=0.9641**。

CNN-only 攻击宏均值：RA=0.9336，ASafeR=0.9769。这不是全链路自适应攻击证据，不能据此宣称全部门禁通过。

## 冻结方法

- 唯一主变量：每个 trial、每个 FFT 频带内，18 个通道分别去均值并除以自身样本标准差（ddof=1），再计算 18×18 相关矩阵。禁止跨 trial、跨 Test 批次统计。
- 后续严格沿用 E36：0.05 固定 SPD 收缩、特征值 floor=1e-12、logm、上三角（非对角乘 sqrt(2)）、detrend+zband；WCCN shrinkage=0.05、eigenvalue floor=1e-6。
- 对照为同折 E36 原始协方差 logSPD/WCCN；两方案共享 E54 OE-EEGNet 身份分数、样本、攻击波形和角色，只各自重拟合 WCCN 并从 clean known dr 计算 q02/q05。
- 十个 inner run 为 5 outer groups × 2 inner folds；每 run 的 outer/enrolled/OE-only/calibration 为 4/8/4/4 互斥被试。known 为 60–80% 连续块，每被试/任务排序取前8条；unknown 仅 calibration 四人。
- 三段式：dr≥tau_high 用 logSPD 身份；中间区用 OE-EEGNet；dr<tau_low 拒识。inner 分位数 method=linear；U@95 是 dr<clean-known-q05 的 unknown 比例，工作点U来自实际 q02 拒识。
- 攻击复现 E58 的 CNN-only 白盒 `raw_attack(..., steps=20)` 调用（含保留历史名称的六个攻击）；只攻击 OE-EEGNet，再冻结波形送入两种 logSPD。**不是对 logSPD+路由+EEGNet 的全链路自适应攻击**；E65 未新增对抗训练。

## Train inner 五外层折均值

| 表征 | direct C | threshold C | direct身份F1 | 含拒识身份F1 | 连续块 | AUROC | unknown AUPR | U@95 | 工作点U | unknown F1 | joint F1 | RA | ASafeR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e36_covariance | 0.9841 | 0.9710 | 0.9842 | 0.9818 | 0.9841 | 0.9982 | 0.9958 | 0.9984 | 0.9875 | 0.9723 | 0.9778 | 0.9396 | 0.9771 |
| e65_correlation | 0.9888 | 0.9734 | 0.9888 | 0.9841 | 0.9888 | 0.9951 | 0.9879 | 0.9891 | 0.9641 | 0.9600 | 0.9733 | 0.9336 | 0.9769 |

Pooled clean 结果（跨 run 固定1–20/0–20标签；每 run 阈值不变）见 results.json。逐折比较：

| outer | 表征 | direct C | threshold C | direct F1 | 含拒识F1 | AUROC | AUPR | U@95 | U | RA | ASafeR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | e36_covariance | 0.9880 | 0.9721 | 0.9883 | 0.9828 | 0.9981 | 0.9962 | 1.0000 | 0.9766 | 0.9496 | 0.9775 |
| 1 | e65_correlation | 0.9920 | 0.9742 | 0.9922 | 0.9849 | 0.9961 | 0.9901 | 0.9922 | 0.9844 | 0.9328 | 0.9808 |
| 2 | e36_covariance | 0.9900 | 0.9741 | 0.9903 | 0.9848 | 0.9978 | 0.9952 | 0.9961 | 0.9922 | 0.9324 | 0.9736 |
| 2 | e65_correlation | 0.9860 | 0.9721 | 0.9863 | 0.9827 | 0.9926 | 0.9838 | 0.9766 | 0.9336 | 0.9383 | 0.9781 |
| 3 | e36_covariance | 0.9819 | 0.9700 | 0.9820 | 0.9809 | 0.9990 | 0.9979 | 1.0000 | 0.9922 | 0.9356 | 0.9806 |
| 3 | e65_correlation | 0.9859 | 0.9721 | 0.9859 | 0.9828 | 0.9963 | 0.9885 | 1.0000 | 0.9922 | 0.9331 | 0.9739 |
| 4 | e36_covariance | 0.9762 | 0.9702 | 0.9762 | 0.9808 | 0.9989 | 0.9979 | 0.9961 | 0.9883 | 0.9437 | 0.9820 |
| 4 | e65_correlation | 0.9880 | 0.9721 | 0.9874 | 0.9826 | 0.9967 | 0.9927 | 0.9922 | 0.9688 | 0.9285 | 0.9794 |
| 5 | e36_covariance | 0.9844 | 0.9688 | 0.9843 | 0.9798 | 0.9971 | 0.9918 | 1.0000 | 0.9883 | 0.9365 | 0.9717 |
| 5 | e65_correlation | 0.9922 | 0.9766 | 0.9922 | 0.9878 | 0.9936 | 0.9845 | 0.9844 | 0.9414 | 0.9352 | 0.9723 |

### 分攻击（五外层折均值）

| 表征 | 攻击 | RA | ASafeR | direct RA | 拒识率 | N/run |
|---|---|---:|---:|---:|---:|---:|
| e36_covariance | untargeted_fgsm_snr30 | 0.9567 | 0.9841 | 0.9675 | 0.0274 | 252 |
| e36_covariance | untargeted_fgsm_snr20 | 0.9108 | 0.9608 | 0.9227 | 0.0500 | 252 |
| e36_covariance | untargeted_pgd_snr30 | 0.9583 | 0.9845 | 0.9679 | 0.0262 | 252 |
| e36_covariance | untargeted_pgd_snr20 | 0.9143 | 0.9668 | 0.9254 | 0.0524 | 252 |
| e36_covariance | targeted_pgd_snr30 | 0.9639 | 0.9889 | 0.9770 | 0.0250 | 252 |
| e36_covariance | targeted_pgd_snr20 | 0.9334 | 0.9774 | 0.9524 | 0.0440 | 252 |
| e65_correlation | untargeted_fgsm_snr30 | 0.9552 | 0.9841 | 0.9691 | 0.0289 | 252 |
| e65_correlation | untargeted_fgsm_snr20 | 0.8986 | 0.9592 | 0.9133 | 0.0606 | 252 |
| e65_correlation | untargeted_pgd_snr30 | 0.9556 | 0.9838 | 0.9691 | 0.0282 | 252 |
| e65_correlation | untargeted_pgd_snr20 | 0.9004 | 0.9610 | 0.9139 | 0.0606 | 252 |
| e65_correlation | targeted_pgd_snr30 | 0.9674 | 0.9940 | 0.9841 | 0.0266 | 252 |
| e65_correlation | targeted_pgd_snr20 | 0.9242 | 0.9794 | 0.9512 | 0.0552 | 252 |

## 连续块 dr 漂移（十个 inner run 均值的分布统计）

漂移为当前块统计减去同 run 的 fit≤60% 统计；不是跨 Session 漂移。完整 q95/std/N 见 JSON。

| 表征 | 块 | dr mean | q02 | q05 | median | mean漂移 | q02漂移 | q05漂移 | median漂移 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e36_covariance | fit≤60% | 1.5280 | 1.2153 | 1.2472 | 1.3983 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| e36_covariance | 60–70% | 1.4215 | 1.0946 | 1.1285 | 1.2747 | -0.1065 | -0.1206 | -0.1187 | -0.1237 |
| e36_covariance | 70–80% | 1.3974 | 1.0878 | 1.1203 | 1.2728 | -0.1306 | -0.1275 | -0.1269 | -0.1255 |
| e65_correlation | fit≤60% | 1.3868 | 1.1656 | 1.1898 | 1.3131 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| e65_correlation | 60–70% | 1.2648 | 1.0667 | 1.0914 | 1.2069 | -0.1219 | -0.0988 | -0.0984 | -0.1062 |
| e65_correlation | 70–80% | 1.2555 | 1.0607 | 1.0864 | 1.2023 | -0.1312 | -0.1049 | -0.1034 | -0.1108 |

## 预注册 Test 门禁

| 门禁（correlation五折clean均值） | 实测 | 最低值 | 结果 |
|---|---:|---:|---|
| clean_direct_C | 0.9888 | 0.9500 | 通过 |
| clean_identity_macro_F1_direct | 0.9888 | 0.9200 | 通过 |
| clean_AUROC | 0.9951 | 0.9800 | 通过 |
| clean_U_at_95 | 0.9891 | 0.9500 | 通过 |

四项 clean 门禁全部通过：**True**。RA/ASafeR 不阻止工程审计，但由于攻击不是全链路自适应，不宣称 Train 全部门禁通过。

## 获准后的 final20 Train 冻结

保持 E64 十 backbone/20 类 CNN ensemble 不变，仅替换 correlation-logSPD/WCCN 与 dr；fit≤60% 拟合，60–70% clean known 以 method=higher 得到 tau_low=1.035048309、tau_high=1.058471022，70–80% sanity。

| 块 | N | direct C | threshold C | direct F1 | 含拒识F1 | 拒识率 | dr median | median相对fit漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit | 6695 | 0.9975 | 0.9970 | 0.9974 | 0.9974 | 0.0007 | 1.2378 | 0.0000 |
| calibration | 1162 | 0.9707 | 0.9639 | 0.9698 | 0.9726 | 0.0207 | 1.1786 | -0.0592 |
| sanity | 1139 | 0.9658 | 0.9596 | 0.9640 | 0.9684 | 0.0237 | 1.1738 | -0.0640 |

## Test 执行状态与工程审计

已按门禁执行一次冻结 Test；`D:\WORK\BCIC2026\BrainPrint\research\E65_correlation_spd\e65_predictions.csv` 共 22251 行。Session 仅在全部预测和 CSV 落盘后读取。

| Session | N | 拒识率 | 接受率 | 目标 | 判定 |
|---|---:|---:|---:|---|---|
| S1 | 3033 | 0.4764 | 0.5236 | 拒识≥95% | 未达到 |
| S2 | 19218 | 0.6062 | 0.3938 | 接受≥98% | 未达到 |

审计后未调整阈值、模型或变体。

**Test AUC/F1/C/RA/ASafeR/U/Score=N/A（无真实标签）**。S1拒识/S2接受若存在仅为预测工程审计，不是官方真值指标。

## 证据边界与复现

- 所有 AUC/F1/C/U/RA/ASafeR 数值均来自 Train Session1 的 pseudo-unknown 与连续块代理，不是跨 Session 或官方新被试证据。fit 块 dr 为拟合内诊断，calibration/sanity 为连续块迁移诊断。
- Correlation 与 covariance 的差值是同折配对描述，不涉及 Test 调参。E64 的历史 Test 结果只用于提出表征问题。
- 特征与 final20（若执行）均检查单条/批次/置换一致性；GPU batch 仅加速，不使用批次统计。

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E65_correlation_spd\run_e65.py" --stage inner
# 仅当上述结果四项 clean 门禁全部通过：
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E65_correlation_spd\run_e65.py" --stage test
```

配置：`D:\WORK\BCIC2026\BrainPrint\research\E65_correlation_spd\config.json`；结果：`D:\WORK\BCIC2026\BrainPrint\research\E65_correlation_spd\results.json`；缓存：`D:\WORK\BCIC2026\BrainPrint\BrainPrint_cache\e65_correlation_spd`。断点绑定配置和代码 SHA256。
