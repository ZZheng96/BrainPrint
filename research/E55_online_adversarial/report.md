# E55 全 fit/OE 当前批次在线原始波形 PGD7 纯 EEG 实验

日期：2026-11-13。仅使用 Train Session1；Test 未读取。

## 实际执行与复现

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py" --stage smoke
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py" --stage inner
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py" --stage outer  # 仅变体全部门禁通过时执行
```

状态：`smoke_complete`。实现：`D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\run_e55.py`；配置：`D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\config.json`；完整结果：`D:\WORK\BCIC2026\BrainPrint\research\E55_online_adversarial\results.json`。

- E55唯一主要改动：每个当前fit mini-batch及配对OE mini-batch均在GPU上对当前模型执行真实原始波形PGD7；没有调用E54的PGD3波形缓存或refresh路径。
- fit的clean/adv身份+ARPL损失各0.5；OE保持λu=0.1、m=0.2、T=0.05及r_u原型detach，clean/adv配对OE项各0.5。OE攻击只用当前模型伪标签，不设未知类或新原型。
- 主变体目标攻击比例0.5；唯一消融untargeted_online_pgd7比例0。早停轻量PGD3不变，正式评估在恢复早停最佳完整模型后重新生成FGSM/PGD20。
- 阈值、E36六攻击宏指标、互斥8 known/4 OE-only/4 calibration unknown、SNR20/30、40 epoch上限与早停协议均与E54一致。

## 十 inner 折汇总及E54同划分OE基线

AUC/AUROC为known=1、unknown=0的检测指标；AUPR为unknown类。clean F1在连续块全量已知上计算；攻击F1/AUC/AUPR为六攻击宏均值。N为每折六攻击对抗样本合计的折均值。

| 变体 | Train fit | direct C | 阈值C | clean F1 direct/拒识 | 连续块 | 攻击F1 direct/拒识 | clean AUROC/AUPR | 攻击AUROC/AUPR | U@95 | 工作点U | unknown F1 | joint F1 | direct RA | 阈值RA | ASafeR | A | Score | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E54 OE（PGD3缓存，参考） | 0.9940 | 0.9618 | 0.9203 | 0.9619/0.9475 | 0.9618 | 0.8487/0.8178 | 0.8687/0.9107 | 0.7729/0.6230 | 0.4209 | 0.4180 | 0.5070 | 0.7078 | 0.8509 | 0.7824 | 0.8901 | 0.8147 | 71.9656 | 3024 |

## 门禁

| 变体 | Train fit | direct/阈值C | 身份F1 | 连续块 | AUROC | U@95 | direct RA | 阈值RA | ASafeR | 非全拒识 | 全部 |
|---|---|---|---|---|---|---|---|---|---|---|---|

## SMOKE：逐epoch在线PGD7覆盖、显存与吞吐

| epoch | fit PGD7 batches | OE PGD7 batches | 实际PGD步数 | 梯度finite | GPU峰值allocated/reserved MiB | fit+OE samples/s | 训练秒数 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 43/43 | 43/43 | 602 | True | 1780.5/1878.0 | 598.8 | 9.13 |
| 2 | 43/43 | 43/43 | 602 | True | 1781.7/1878.0 | 608.8 | 8.98 |

- smoke覆盖断言=True；参数量=2992。
- 单样本/批次最大分数差=8.94e-08；置换差=0；有限差分相对误差=0.181。
- OE梯度finite/nonzero=True/True；r_u原型梯度L2=0。

## Outer 与 Test

- Outer：N/A: gated until a preregistered E55 variant passes every inner gate。
- Test AUC、F1、C、RA、ASafeR、A、U、Score：N/A（无真实标签，且未读取Test）。

## 证据边界

- unknown指标只来自Train calibration/eval pseudo-unknown，不等于官方新被试；OE-only只参与训练，不进入未知评估。连续文件块是时间隔离代理，不是跨天/跨Session证据。
- Train fit是10个inner fit-known准确率均值及严格逐折100%门禁，不是全20类OE模型或泛化证据。SMOKE只检查攻击覆盖、梯度、显存/吞吐和数值，不用于选模。
- 正式攻击对保存信息所指纹标识的当前完整模型生成；复用 artifact 时同时校验E55配置、脚本、E54基础脚本、模型与攻击指纹，任一变化即重建。
- 本地攻击覆盖FGSM、非目标/目标PGD20×SNR30/20，不保证覆盖全部可能扰动；Score是Train内代理，不是官方Test得分。若门禁失败，本报告为负结果，不削弱攻击或借助outer/Test调参。
