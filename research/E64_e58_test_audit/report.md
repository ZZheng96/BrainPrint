# E64：E58最佳方案20类冻结迁移与Test工程审计

日期：2026-11-13。主方案严格为E54 OE-EEGNet通用backbone迁移 + E36 logSPD/WCCN三段式，不是E59。

> 本次是用户明确授权的一次性冻结Test工程审计；**不声称Train全门禁已通过**。Test结果未用于调阈值、选变体或改集合划分。

## 冻结方法与复现

- 加载E54十个inner `oe` EEGNet并冻结encoder，丢弃原八类 prototypes/radii；每个encoder仅用全部20名Train被试的≤60%连续块embedding建立20个归一化均值原型。没有对20名已注册者同时做OE或继续训练。
- 推理身份分数为十个20类cosine向量的固定等权算术平均。logSPD/WCCN也仅用20名被试≤60%块拟合。
- 60–70% clean known dr独立校准：`tau_low=q0.02`、`tau_high=q0.05`，NumPy `method=higher`；70–80%仅作final20已知迁移sanity，>80%不使用。
- 冻结阈值：`tau_low=1.069218409`，`tau_high=1.094469850`。规则：dr≥tau_high→logSPD身份；中间区→EEGNet ensemble身份；dr<tau_low→0。
- Test逐条读取原始18×1200 MAT；GPU batch仅为加速，所有预处理、logSPD和embedding均为trial-local，无批次z-score、ECDF、分布匹配、阈值调整、身份配额、Session/任务/数量/配对/文件顺序决策。Sample顺序只用于对齐路径和写CSV。

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E64_e58_test_audit\run_e64.py" --stage train
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E64_e58_test_audit\run_e64.py" --stage test
# 或依次自动执行：
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E64_e58_test_audit\run_e64.py" --stage all
```

配置：`D:\WORK\BCIC2026\BrainPrint\research\E64_e58_test_audit\config.json`；代码：`D:\WORK\BCIC2026\BrainPrint\research\E64_e58_test_audit\run_e64.py`；模型：`D:\WORK\BCIC2026\BrainPrint\BrainPrint_cache\e64_e58_test_audit\frozen_model.pt`；冻结清单：`D:\WORK\BCIC2026\BrainPrint\BrainPrint_cache\e64_e58_test_audit\frozen_manifest.json`。
冻结模型SHA256：`791f42f8a84e60d5f6dba87e5bdd36d03a55e8ecbf208b8798b0bbac18895c5c`；E64代码SHA256：`b2cdfad7d907eae3ae58f07d1054fed6536997d476797ddb1d254c40ee75f847`。

## E58 inner参考口径（只读参考，E64未重跑inner）

E58为5 outer groups × 2 inner runs、每run八类注册身份、60–80% clean known与pseudo-unknown、六种白盒攻击宏均值；与E64 final20全20类连续块口径不同。

| C | RA | ASafeR | U | Score | direct身份F1 | clean阈值身份F1 | pseudo-unknown AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9714 | 0.9371 | 0.9758 | 0.9875 | 96.74 | 0.9849 | 0.9821 | 0.9984（E58报告表列值） |

## E64 final20 Train fit、校准与sanity

direct C/F1保留三段路由但不拒识；threshold C/F1包含dr<tau_low拒识。F1固定为1–20类macro-F1，0不计作身份类。

| Train块 | N | direct C | threshold C | direct身份F1 | threshold身份F1 | EEGNet-only C | logSPD-only C | 拒识率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit ≤60% | 6695 | 0.9969 | 0.9955 | 0.9968 | 0.9967 | 0.9240 | 0.9999 | 0.0024 |
| calibration 60–70% | 1162 | 0.9759 | 0.9664 | 0.9751 | 0.9757 | 0.8606 | 0.9983 | 0.0207 |
| final20 sanity 70–80% | 1139 | 0.9605 | 0.9473 | 0.9593 | 0.9622 | 0.8560 | 0.9991 | 0.0360 |

- **Train final20 AUC：N/A**。全部20名Train被试均为注册者，本阶段没有pseudo-unknown；上表AUC不能由身份C/F1替代。E58参考AUROC不属于E64新评估。
- Train数值不变性：N=8，单条/批次与重排分数容差、标签一致均passed=True；未使用跨样本统计。
- Train fit只是拟合/容量诊断；60–70%用于阈值，70–80%为sanity，二者都不是真实跨Session证据。

## Test冻结预测与Session离线审计

预测文件：`D:\WORK\BCIC2026\BrainPrint\research\E64_e58_test_audit\e64_predictions.csv`，共22251行，表头`Sample,Label`。全部预测和CSV写出后才读取Test Session。

| 项目 | N | 拒识率 | 接受率 | 目标 | 审计判定 |
|---|---:|---:|---:|---|---|
| 整体Test | 22251 | 0.6863 | 0.3137 | — | 描述性 |
| S1 | 3033 | 0.5803 | 0.4197 | 拒识≥95% | 未达到 |
| S2 | 19218 | 0.7030 | 0.2970 | 接受≥98% | 未达到 |

非0身份分布（N；占全部Test比例）：

| 身份 | N | 比例 |
|---:|---:|---:|
| 1 | 29 | 0.0013 |
| 2 | 4 | 0.0002 |
| 3 | 569 | 0.0256 |
| 4 | 12 | 0.0005 |
| 5 | 2722 | 0.1223 |
| 6 | 396 | 0.0178 |
| 7 | 404 | 0.0182 |
| 8 | 1 | 0.0000 |
| 9 | 35 | 0.0016 |
| 10 | 104 | 0.0047 |
| 11 | 67 | 0.0030 |
| 12 | 30 | 0.0013 |
| 13 | 813 | 0.0365 |
| 14 | 268 | 0.0120 |
| 15 | 587 | 0.0264 |
| 16 | 177 | 0.0080 |
| 17 | 469 | 0.0211 |
| 18 | 105 | 0.0047 |
| 19 | 137 | 0.0062 |
| 20 | 52 | 0.0023 |

决策路由计数：拒识=15270，EEGNet中间区=1411，logSPD高dr区=5570。

**Test AUC、F1、C、RA、ASafeR、U、Score：全部N/A（无真实标签）。** S1拒识/S2接受只是对预测0/非0的工程审计，不等于U或clean C；非0身份分布也不提供身份正确性。即使目标未达到，本实验也不会据Test回头调整。

## 证据边界

- E64只使用Train Session1连续块拟合/校准/sanity；不是跨天、跨Session或官方新被试真值评估。E58 inner数值仅是参考，不能外推为Test得分。
- 十个backbone来自E54各自八类inner OE训练；E64迁移阶段没有全20类OE。新原型与WCCN严格使用≤60%块。
- Test无真值，因此不报告或暗示官方AUC/F1/C/RA/ASafeR/U/Score。Session审计完成后未修改阈值、模型、代码路线或预测。
- 本次审计获得用户对Train门禁未全通过状态的明确豁免，仅授权工程运行，不构成门禁通过声明。
