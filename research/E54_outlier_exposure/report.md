# E54 非类别化配对 Outlier Exposure 纯 EEG 实验

日期：2026-11-13。仅使用 Train Session1；Test 未读取。

## 实际执行与复现

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E54_outlier_exposure\run_e54.py" --stage smoke
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E54_outlier_exposure\run_e54.py" --stage inner
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E54_outlier_exposure\run_e54.py" --stage outer  # 仅合规变体全部门禁通过时执行
```

状态：`complete_inner_gates_failed`。实现：`D:\WORK\BCIC2026\BrainPrint\research\E54_outlier_exposure\run_e54.py`；配置：`D:\WORK\BCIC2026\BrainPrint\research\E54_outlier_exposure\config.json`；完整结果：`D:\WORK\BCIC2026\BrainPrint\research\E54_outlier_exposure\results.json`。

- 保留E53原始波形预处理、EEGNet、ARPL及混合目标/非目标PGD对抗训练。输入始终为单条1×18×1200 EEG。
- OE主方案固定 λu=0.1、m=0.2、T=0.05：L=λu·mean softplus((r_u-r_y+m)/T)，r=max_k z·mu_k；未知不设第21类或共享原型，r_u分支对注册原型参数detach。
- 唯一消融为oe_task_grl：四任务GRL CE，λtask=0.05；control为相同划分下无OE的E53共享控制。
- 实际训练攻击路径：`pgd3_waveform_cache_refresh5_compute_fallback`；训练缓存按当前模型刷新。正式评估始终对当前折/变体重新生成原始波形FGSM/PGD20，未复用E53攻击输出。
- 阈值只用对应inner known、六攻击和calibration unknown，在clean known acceptance≥0.95下最大化完整代理Score；OE-only不进入阈值或未知评估。
- Train fit为10个inner fit-known训练准确率均值（同时保存最小折），不是全20类OE模型；E54未执行最终20类迁移。

## 十 inner 折汇总（五个outer-group内各两折）

AUC/AUROC为known=1、unknown=0的检测指标（置信度越大越像known）；AUPR为unknown类。clean F1在连续块全量已知上计算；攻击F1/AUC/AUPR为六攻击宏均值。N为每折六攻击对抗样本合计的折均值。

| 变体 | Train fit | direct C | 阈值C | clean F1 direct/拒识 | 连续块 | 攻击F1 direct/拒识 | clean AUROC/AUPR | 攻击AUROC/AUPR | U@95 | 工作点U | unknown F1 | joint F1 | direct RA | 阈值RA | ASafeR | A | Score | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0.9878 | 0.9471 | 0.9012 | 0.9469/0.9403 | 0.9471 | 0.8089/0.7836 | 0.8204/0.8762 | 0.7279/0.5668 | 0.3050 | 0.3375 | 0.4334 | 0.6637 | 0.8126 | 0.7518 | 0.8583 | 0.7837 | 66.8229 | 3024 |
| oe | 0.9940 | 0.9618 | 0.9203 | 0.9619/0.9475 | 0.9618 | 0.8487/0.8178 | 0.8687/0.9107 | 0.7729/0.6230 | 0.4209 | 0.4180 | 0.5070 | 0.7078 | 0.8509 | 0.7824 | 0.8901 | 0.8147 | 71.9656 | 3024 |
| oe_task_grl | 0.9881 | 0.9531 | 0.9103 | 0.9526/0.9441 | 0.9531 | 0.8330/0.8086 | 0.8819/0.9140 | 0.7769/0.6135 | 0.3888 | 0.3945 | 0.4852 | 0.6965 | 0.8367 | 0.7740 | 0.8845 | 0.8072 | 70.3377 | 3024 |

## control 逐攻击（inner折均值）

| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted_fgsm_snr30 | 0.9012 | 0.9111 | 0.8789 | 0.9380 | 0.8967 | 0.9096/0.9015 | 0.8087 | 0.6704 | 0.4593 | 0.3375 | 0.7598 | 70.0267 | 504 |
| untargeted_fgsm_snr20 | 0.9012 | 0.8717 | 0.8288 | 0.9023 | 0.8509 | 0.8688/0.8540 | 0.7754 | 0.6216 | 0.4510 | 0.3375 | 0.7211 | 68.7756 | 504 |
| untargeted_pgd_snr30 | 0.9012 | 0.7769 | 0.7161 | 0.8345 | 0.7517 | 0.7703/0.7495 | 0.6984 | 0.5344 | 0.4263 | 0.3375 | 0.6364 | 65.8560 | 504 |
| untargeted_pgd_snr20 | 0.9012 | 0.6154 | 0.5411 | 0.6866 | 0.5847 | 0.6084/0.5691 | 0.6574 | 0.5004 | 0.4142 | 0.3375 | 0.4921 | 60.3310 | 504 |
| targeted_pgd_snr30 | 0.9012 | 0.8881 | 0.8114 | 0.9210 | 0.8443 | 0.8862/0.8518 | 0.7367 | 0.5599 | 0.4305 | 0.3375 | 0.7167 | 68.5757 | 504 |
| targeted_pgd_snr20 | 0.9012 | 0.8127 | 0.7344 | 0.8671 | 0.7742 | 0.8100/0.7756 | 0.6908 | 0.5140 | 0.4193 | 0.3375 | 0.6559 | 66.5299 | 504 |

## oe 逐攻击（inner折均值）

| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted_fgsm_snr30 | 0.9203 | 0.9374 | 0.8949 | 0.9556 | 0.9131 | 0.9370/0.9193 | 0.8357 | 0.7083 | 0.5360 | 0.4180 | 0.7943 | 74.8690 | 504 |
| untargeted_fgsm_snr20 | 0.9203 | 0.8962 | 0.8350 | 0.9239 | 0.8617 | 0.8937/0.8666 | 0.7995 | 0.6613 | 0.5173 | 0.4180 | 0.7495 | 73.3793 | 504 |
| untargeted_pgd_snr30 | 0.9203 | 0.8384 | 0.7713 | 0.8867 | 0.8059 | 0.8364/0.8097 | 0.7646 | 0.6166 | 0.5026 | 0.4180 | 0.7010 | 71.7056 | 504 |
| untargeted_pgd_snr20 | 0.9203 | 0.6597 | 0.5907 | 0.7220 | 0.6301 | 0.6524/0.6190 | 0.7264 | 0.5850 | 0.4935 | 0.4180 | 0.5434 | 65.8088 | 504 |
| targeted_pgd_snr30 | 0.9203 | 0.9236 | 0.8513 | 0.9517 | 0.8814 | 0.9235/0.8913 | 0.7845 | 0.6212 | 0.5107 | 0.4180 | 0.7682 | 73.9585 | 504 |
| targeted_pgd_snr20 | 0.9203 | 0.8502 | 0.7509 | 0.9009 | 0.7959 | 0.8490/0.8010 | 0.7268 | 0.5457 | 0.4818 | 0.4180 | 0.6900 | 71.3777 | 504 |

## oe_task_grl 逐攻击（inner折均值）

| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted_fgsm_snr30 | 0.9103 | 0.9258 | 0.8889 | 0.9476 | 0.9065 | 0.9253/0.9126 | 0.8508 | 0.7106 | 0.5142 | 0.3945 | 0.7839 | 73.2189 | 504 |
| untargeted_fgsm_snr20 | 0.9103 | 0.8786 | 0.8254 | 0.9167 | 0.8528 | 0.8761/0.8575 | 0.8053 | 0.6411 | 0.4946 | 0.3945 | 0.7372 | 71.6847 | 504 |
| untargeted_pgd_snr30 | 0.9103 | 0.8248 | 0.7617 | 0.8776 | 0.7965 | 0.8199/0.7965 | 0.7651 | 0.6019 | 0.4832 | 0.3945 | 0.6867 | 70.0418 | 504 |
| untargeted_pgd_snr20 | 0.9103 | 0.6486 | 0.5732 | 0.7159 | 0.6160 | 0.6364/0.5970 | 0.7135 | 0.5629 | 0.4700 | 0.3945 | 0.5250 | 64.0096 | 504 |
| targeted_pgd_snr30 | 0.9103 | 0.9100 | 0.8548 | 0.9497 | 0.8833 | 0.9100/0.8928 | 0.7975 | 0.6242 | 0.4920 | 0.3945 | 0.7648 | 72.5671 | 504 |
| targeted_pgd_snr20 | 0.9103 | 0.8322 | 0.7402 | 0.8993 | 0.7879 | 0.8303/0.7951 | 0.7291 | 0.5400 | 0.4574 | 0.3945 | 0.6816 | 69.7474 | 504 |

## 数据角色与隔离

每折四个角色为outer unknown 4人、registered/fit-known 8人、OE-only 4人、calibration/eval unknown 4人，被试集合两两互斥且并集为20人。fit-known/OE-only仅取≤60%连续块，known validation为60–80%，calibration unknown取≤80%；outer封存。

| outer | inner | outer unknown ID | registered ID | OE-only ID | calibration/eval unknown ID | OE split seed | model seed |
|---:|---:|---|---|---|---|---:|---:|
| 1 | 1 | 8,10,15,16 | 3,5,6,7,11,13,14,19 | 1,2,9,18 | 4,12,17,20 | 54101 | 20262114 |
| 1 | 2 | 8,10,15,16 | 1,2,4,9,12,17,18,20 | 3,7,13,19 | 5,6,11,14 | 54102 | 20262115 |
| 2 | 1 | 7,11,13,20 | 4,5,6,8,12,15,16,19 | 1,3,9,18 | 2,10,14,17 | 54201 | 20263114 |
| 2 | 2 | 7,11,13,20 | 1,2,3,9,10,14,17,18 | 6,12,16,19 | 4,5,8,15 | 54202 | 20263115 |
| 3 | 1 | 1,4,6,17 | 3,5,9,10,15,16,18,20 | 2,11,12,19 | 7,8,13,14 | 54301 | 20264114 |
| 3 | 2 | 1,4,6,17 | 2,7,8,11,12,13,14,19 | 5,9,16,20 | 3,10,15,18 | 54302 | 20264115 |
| 4 | 1 | 3,5,12,19 | 1,4,6,11,13,16,17,20 | 7,10,14,15 | 2,8,9,18 | 54401 | 20265114 |
| 4 | 2 | 3,5,12,19 | 2,7,8,9,10,14,15,18 | 6,11,13,17 | 1,4,16,20 | 54402 | 20265115 |
| 5 | 1 | 2,9,14,18 | 1,5,6,8,10,13,17,19 | 7,12,16,20 | 3,4,11,15 | 54501 | 20266114 |
| 5 | 2 | 2,9,14,18 | 3,4,7,11,12,15,16,20 | 1,10,17,19 | 5,6,8,13 | 54502 | 20266115 |

- 全部已执行折的数据隔离断言通过：True；OE-only没有评分数组，也未参与原型初始化、阈值或unknown指标。

## 门禁与消融定位

| 变体 | Train fit | direct/阈值C | 身份F1 | 连续块 | AUROC | U@95 | direct RA | 阈值RA | ASafeR | 非全拒识 | 全部 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 未通过 | 未通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |
| oe | 未通过 | 通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |
| oe_task_grl | 未通过 | 通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |

- OE主方案 all_rejection=False；direct RA<0.95时，不宣称鲁棒门禁达标。
- 仅执行预注册三变体，未搜索OE超参数，未据outer/Test调参。

### OE相对共享控制的增量

- ΔAUROC=+0.0483；ΔU@95=+0.1159；Δdirect C=+0.0147；Δ阈值C=+0.0191；Δdirect RA=+0.0383；Δ阈值RA=+0.0306；ΔASafeR=+0.0319。

### task GRL采纳判定

- adopted=False。AUROC_strictly_improves=True；U_at_95_strictly_improves=False；direct_C_not_worse=False；threshold_C_not_worse=False；direct_RA_not_worse=False；threshold_RA_not_worse=False。
- best_inner_variant_by_score=oe；selected_variant_for_outer=None。

## SMOKE 数值、OE梯度与不变性

- outer1/inner1参数量：2992；在线PGD7 batch64实测 0.121s（仅速度基准）。
- 单样本/批次最大分数差 1.19e-07；置换差 0；标签一致=True。
- 方向导数有限差分相对误差 0.0552；所有正式攻击梯度与分数均要求finite。
- OE loss=0.0735686，known/unknown r均值=0.5859/0.2581；未知波形梯度L2=0.0527489，projection梯度L2=0.31789；finite=True，nonzero=True，r_u原型梯度L2=0（zero=True）。

## Outer 与 Test

- Outer：N/A: no eligible E54 variant passed every inner gate; outer was not run。
- Test AUC、F1、C、RA、ASafeR、A、U、Score：N/A（无真实标签，且未读取 Test）。

## 证据边界与20类迁移

- unknown指标只来自Train的calibration/eval pseudo-unknown，不等于官方新被试；OE-only仅参与训练损失，不能混入未知评估。连续文件块是顺序假设下的时间隔离代理，不是真实跨天/跨Session证据。
- Train fit仅为fit-known容量/训练流程检查，不是泛化证据。SMOKE只用于速度、梯度、隔离和数值检查，不用于模型选择。
- 本地攻击覆盖FGSM、非目标/目标PGD20×SNR30/20，不能保证覆盖官方全部扰动；Score是Train内代理，不是官方Test得分。
- E54没有在全部20名已注册者上做OE，也未伪造最终20类结果。若inner门禁通过，下一冻结实验应冻结通用backbone、用全Train建立20原型并以20类OOF ECDF校准；具体迁移不在本实验完成。
- 若门禁失败，本报告即为负结果归档；不通过削弱攻击、全拒识或outer/Test调参补救。
