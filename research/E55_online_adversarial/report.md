# E55 全 fit/OE 当前批次在线原始波形 PGD7 纯 EEG 实验

日期：2026-11-13。仅使用 Train Session1；Test 未读取。

## 实际执行与复现

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py" --stage smoke
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py" --stage inner
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py" --stage outer  # 仅变体全部门禁通过时执行
```

状态：`complete_inner_gates_failed`。实现：`D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py`；配置：`D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\config.json`；完整结果：`D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\results.json`。

- E55唯一主要改动：每个当前fit mini-batch及配对OE mini-batch均在GPU上对当前模型执行真实原始波形PGD7；没有调用E54的PGD3波形缓存或refresh路径。
- fit的clean/adv身份+ARPL损失各0.5；OE保持λu=0.1、m=0.2、T=0.05及r_u原型detach，clean/adv配对OE项各0.5。OE攻击只用当前模型伪标签，不设未知类或新原型。
- 主变体目标攻击比例0.5；唯一消融untargeted_online_pgd7比例0。早停轻量PGD3不变，正式评估在恢复早停最佳完整模型后重新生成FGSM/PGD20。
- 阈值、E36六攻击宏指标、互斥8 known/4 OE-only/4 calibration unknown、SNR20/30、40 epoch上限与早停协议均与E54一致。

## 十 inner 折汇总及E54同划分OE基线

AUC/AUROC为known=1、unknown=0的检测指标；AUPR为unknown类。clean F1在连续块全量已知上计算；攻击F1/AUC/AUPR为六攻击宏均值。N为每折六攻击对抗样本合计的折均值。

| 变体 | Train fit | direct C | 阈值C | clean F1 direct/拒识 | 连续块 | 攻击F1 direct/拒识 | clean AUROC/AUPR | 攻击AUROC/AUPR | U@95 | 工作点U | unknown F1 | joint F1 | direct RA | 阈值RA | ASafeR | A | Score | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E54 OE（PGD3缓存，参考） | 0.9940 | 0.9618 | 0.9203 | 0.9619/0.9475 | 0.9618 | 0.8487/0.8178 | 0.8687/0.9107 | 0.7729/0.6230 | 0.4209 | 0.4180 | 0.5070 | 0.7078 | 0.8509 | 0.7824 | 0.8901 | 0.8147 | 71.9656 | 3024 |
| main_online_pgd7 | 0.9898 | 0.9570 | 0.9148 | 0.9565/0.9484 | 0.9570 | 0.8582/0.8371 | 0.8631/0.9038 | 0.7752/0.5928 | 0.3491 | 0.3664 | 0.4595 | 0.7112 | 0.8608 | 0.8002 | 0.9079 | 0.8325 | 70.0193 | 3024 |
| untargeted_online_pgd7 | 0.9789 | 0.9314 | 0.8883 | 0.9306/0.9229 | 0.9314 | 0.8209/0.8035 | 0.7954/0.8666 | 0.7084/0.5538 | 0.3056 | 0.3117 | 0.4093 | 0.6789 | 0.8272 | 0.7790 | 0.8677 | 0.8056 | 65.3468 | 3024 |

## main_online_pgd7 逐攻击（inner折均值）

| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted_fgsm_snr30 | 0.9148 | 0.9271 | 0.8957 | 0.9561 | 0.9138 | 0.9257/0.9201 | 0.8454 | 0.6909 | 0.4869 | 0.3664 | 0.7809 | 72.3320 | 504 |
| untargeted_fgsm_snr20 | 0.9148 | 0.8854 | 0.8402 | 0.9232 | 0.8651 | 0.8830/0.8706 | 0.8051 | 0.6291 | 0.4729 | 0.3664 | 0.7393 | 70.9603 | 504 |
| untargeted_pgd_snr30 | 0.9148 | 0.8570 | 0.8022 | 0.9027 | 0.8324 | 0.8545/0.8387 | 0.7700 | 0.5835 | 0.4628 | 0.3664 | 0.7119 | 70.0237 | 504 |
| untargeted_pgd_snr20 | 0.9148 | 0.7106 | 0.6284 | 0.7828 | 0.6747 | 0.7043/0.6658 | 0.7008 | 0.5213 | 0.4349 | 0.3664 | 0.5716 | 65.0720 | 504 |
| targeted_pgd_snr30 | 0.9148 | 0.9243 | 0.8604 | 0.9612 | 0.8907 | 0.9239/0.9014 | 0.7969 | 0.6103 | 0.4626 | 0.3664 | 0.7635 | 71.6779 | 504 |
| targeted_pgd_snr20 | 0.9148 | 0.8605 | 0.7739 | 0.9216 | 0.8182 | 0.8581/0.8259 | 0.7328 | 0.5220 | 0.4369 | 0.3664 | 0.6999 | 69.5949 | 504 |

## untargeted_online_pgd7 逐攻击（inner折均值）

| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted_fgsm_snr30 | 0.8883 | 0.9023 | 0.8692 | 0.9256 | 0.8862 | 0.9000/0.8897 | 0.7646 | 0.6288 | 0.4268 | 0.3117 | 0.7497 | 67.5367 | 504 |
| untargeted_fgsm_snr20 | 0.8883 | 0.8543 | 0.8125 | 0.8900 | 0.8357 | 0.8495/0.8372 | 0.7294 | 0.5760 | 0.4149 | 0.3117 | 0.7059 | 66.1778 | 504 |
| untargeted_pgd_snr30 | 0.8883 | 0.8104 | 0.7746 | 0.8493 | 0.7970 | 0.8030/0.7938 | 0.7126 | 0.5611 | 0.4164 | 0.3117 | 0.6724 | 65.1185 | 504 |
| untargeted_pgd_snr20 | 0.8883 | 0.6849 | 0.6320 | 0.7374 | 0.6636 | 0.6687/0.6450 | 0.6633 | 0.5161 | 0.4004 | 0.3117 | 0.5514 | 61.0959 | 504 |
| targeted_pgd_snr30 | 0.8883 | 0.8911 | 0.8390 | 0.9272 | 0.8655 | 0.8895/0.8711 | 0.7179 | 0.5527 | 0.4093 | 0.3117 | 0.7327 | 66.9787 | 504 |
| targeted_pgd_snr20 | 0.8883 | 0.8204 | 0.7465 | 0.8769 | 0.7856 | 0.8147/0.7844 | 0.6625 | 0.4880 | 0.3882 | 0.3117 | 0.6610 | 64.7803 | 504 |

## 数据角色与隔离

每折outer unknown 4人、registered/fit-known 8人、OE-only 4人、calibration/eval unknown 4人两两互斥。划分种子与E54相同；fit/OE仅取≤60%，known为60–80%，calibration unknown≤80%；outer封存。

| outer | inner | outer unknown ID | registered ID | OE-only ID | calibration/eval unknown ID | OE split seed | model seed |
|---:|---:|---|---|---|---|---:|---:|
| 1 | 1 | 16,10,15,8 | 3,5,6,7,11,13,14,19 | 1,2,9,18 | 4,12,17,20 | 54101 | 20262114 |
| 1 | 2 | 16,10,15,8 | 1,2,4,9,12,17,18,20 | 3,7,13,19 | 5,6,11,14 | 54102 | 20262115 |
| 2 | 1 | 13,11,7,20 | 4,5,6,8,12,15,16,19 | 1,3,9,18 | 2,10,14,17 | 54201 | 20263114 |
| 2 | 2 | 13,11,7,20 | 1,2,3,9,10,14,17,18 | 6,12,16,19 | 4,5,8,15 | 54202 | 20263115 |
| 3 | 1 | 4,1,17,6 | 3,5,9,10,15,16,18,20 | 2,11,12,19 | 7,8,13,14 | 54301 | 20264114 |
| 3 | 2 | 4,1,17,6 | 2,7,8,11,12,13,14,19 | 5,9,16,20 | 3,10,15,18 | 54302 | 20264115 |
| 4 | 1 | 12,19,3,5 | 1,4,6,11,13,16,17,20 | 7,10,14,15 | 2,8,9,18 | 54401 | 20265114 |
| 4 | 2 | 12,19,3,5 | 2,7,8,9,10,14,15,18 | 6,11,13,17 | 1,4,16,20 | 54402 | 20265115 |
| 5 | 1 | 18,2,14,9 | 1,5,6,8,10,13,17,19 | 7,12,16,20 | 3,4,11,15 | 54501 | 20266114 |
| 5 | 2 | 18,2,14,9 | 3,4,7,11,12,15,16,20 | 1,10,17,19 | 5,6,8,13 | 54502 | 20266115 |

- 全部已执行折的数据隔离断言通过：True；OE-only没有评分数组，也不参与原型初始化、阈值或unknown评估。

## 门禁

| 变体 | Train fit | direct/阈值C | 身份F1 | 连续块 | AUROC | U@95 | direct RA | 阈值RA | ASafeR | 非全拒识 | 全部 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main_online_pgd7 | 未通过 | 通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |
| untargeted_online_pgd7 | 未通过 | 未通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |

- best_inner_variant_by_score=main_online_pgd7；selected_variant_for_outer=None。
- direct RA未达0.95时不宣称鲁棒达标；任何变体未通过全部门禁均不触发outer。仅比较两个预注册变体，无额外网格。

## SMOKE：逐epoch在线PGD7覆盖、显存与吞吐

| epoch | fit PGD7 batches | OE PGD7 batches | 实际PGD步数 | 梯度finite | GPU峰值allocated/reserved MiB | fit+OE samples/s | 训练秒数 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 43/43 | 43/43 | 602 | True | 1780.5/1878.0 | 598.8 | 9.13 |
| 2 | 43/43 | 43/43 | 602 | True | 1781.7/1878.0 | 608.8 | 8.98 |

- smoke覆盖断言=True；参数量=2992。
- 单样本/批次最大分数差=8.94e-08；置换差=0；有限差分相对误差=0.181。
- OE梯度finite/nonzero=True/True；r_u原型梯度L2=0。

## Outer 与 Test

- Outer：N/A: no E55 variant passed every inner gate; outer was not run。
- Test AUC、F1、C、RA、ASafeR、A、U、Score：N/A（无真实标签，且未读取Test）。

## 证据边界

- unknown指标只来自Train calibration/eval pseudo-unknown，不等于官方新被试；OE-only只参与训练，不进入未知评估。连续文件块是时间隔离代理，不是跨天/跨Session证据。
- Train fit是10个inner fit-known准确率均值及严格逐折100%门禁，不是全20类OE模型或泛化证据。SMOKE只检查攻击覆盖、梯度、显存/吞吐和数值，不用于选模。
- 正式攻击对保存信息所指纹标识的当前完整模型生成；复用 artifact 时同时校验E55配置、脚本、E54基础脚本、模型与攻击指纹，任一变化即重建。
- 本地攻击覆盖FGSM、非目标/目标PGD20×SNR30/20，不保证覆盖全部可能扰动；Score是Train内代理，不是官方Test得分。若门禁失败，本报告为负结果，不削弱攻击或借助outer/Test调参。
