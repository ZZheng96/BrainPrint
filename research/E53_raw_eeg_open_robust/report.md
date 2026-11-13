# E53 原始 EEG 轻量开放集与对抗鲁棒实验

日期：2026-11-13。仅使用 Train Session1；Test 未读取。

## 实际执行与复现

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E53_raw_eeg_open_robust\run_e53.py" --stage smoke
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E53_raw_eeg_open_robust\run_e53.py" --stage inner
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E53_raw_eeg_open_robust\run_e53.py" --stage outer  # 仅主方案全部门禁通过时执行
```

状态：`complete_inner_gates_failed`。实现：`D:\WORK\BCIC2026\BrainPrint\research\E53_raw_eeg_open_robust\run_e53.py`；配置：`D:\WORK\BCIC2026\BrainPrint\research\E53_raw_eeg_open_robust\config.json`；完整结果：`D:\WORK\BCIC2026\BrainPrint\research\E53_raw_eeg_open_robust\results.json`。

- 输入为单条 1×18×1200；逐通道线性去趋势、median/1.4826×MAD（RMS 回退）均逐样本可微。
- 8 时间卷积(k=65)→depthwise 空间卷积(D=2)→16 separable(k=15)，BN/ELU/池化/GAP，64 维 L2 embedding；归一化可训练身份原型+cosine logits。
- 主方案双侧 ARPL/reciprocal-point 开空间损失，不设第 21 类；两个消融仅去 ARPL、或目标 AT 改全非目标 AT。
- 实际训练攻击路径：`pgd3_waveform_cache_refresh5_compute_fallback`；clean/adv 损失各 0.5，标签始终为真实身份。正式评估始终原始波形 FGSM/PGD20。
- 阈值仅用对应 inner clean、六攻击和 pseudo-unknown，在 clean known acceptance≥0.95 下最大化完整本地代理 Score。direct RA 与阈值 RA 分开判定。

## 十 inner 折汇总（五个 outer-group 内各两折）

AUC 为 known=1、unknown=0 的检测 AUROC（置信度越大越像 known）；AUPR 为 unknown 类。clean F1 在连续块全量已知上计算；对抗 F1/AUC 为六攻击宏均值。N 为每折六攻击对抗样本合计的折均值。

| 变体 | Train fit | direct C | 阈值C | clean F1 direct/拒识 | 连续块 | 攻击F1 direct/拒识 | clean AUROC/AUPR | U@95 | 工作点U | unknown F1 | joint F1 | direct RA | 阈值RA | ASafeR | A | Score | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.9609 | 0.9567 | 0.9100 | 0.9567/0.9472 | 0.9567 | 0.8309/0.8042 | 0.8238/0.9375 | 0.3057 | 0.3316 | 0.4599 | 0.6032 | 0.8340 | 0.7696 | 0.8780 | 0.8021 | 67.5151 | 3024 |
| no_arpl | 0.9566 | 0.9461 | 0.8982 | 0.9454/0.9420 | 0.9461 | 0.8045/0.7835 | 0.8197/0.9335 | 0.2328 | 0.2449 | 0.3609 | 0.5690 | 0.8086 | 0.7521 | 0.8572 | 0.7836 | 61.7103 | 3024 |
| untargeted_at | 0.9611 | 0.9627 | 0.9159 | 0.9628/0.9500 | 0.9627 | 0.8508/0.8192 | 0.8182/0.9373 | 0.3372 | 0.3812 | 0.5128 | 0.6290 | 0.8527 | 0.7845 | 0.8888 | 0.8158 | 70.5172 | 3024 |

## 主方案逐攻击（inner 折均值）

| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untargeted_fgsm_snr30 | 0.9100 | 0.9242 | 0.8890 | 0.9493 | 0.9071 | 0.9233/0.9130 | 0.8021 | 0.7858 | 0.4751 | 0.3316 | 0.6823 | 70.4808 | 504 |
| untargeted_fgsm_snr20 | 0.9100 | 0.8834 | 0.8361 | 0.9152 | 0.8598 | 0.8808/0.8647 | 0.7652 | 0.7427 | 0.4691 | 0.3316 | 0.6472 | 69.1775 | 504 |
| untargeted_pgd_snr30 | 0.9100 | 0.8050 | 0.7347 | 0.8519 | 0.7699 | 0.8007/0.7712 | 0.7009 | 0.6858 | 0.4570 | 0.3316 | 0.5800 | 66.5515 | 504 |
| untargeted_pgd_snr20 | 0.9100 | 0.6518 | 0.5675 | 0.7250 | 0.6147 | 0.6451/0.6015 | 0.6462 | 0.6412 | 0.4453 | 0.3316 | 0.4595 | 61.5168 | 504 |
| targeted_pgd_snr30 | 0.9100 | 0.9041 | 0.8370 | 0.9362 | 0.8668 | 0.9028/0.8755 | 0.7396 | 0.7130 | 0.4623 | 0.3316 | 0.6526 | 69.3668 | 504 |
| targeted_pgd_snr20 | 0.9100 | 0.8355 | 0.7536 | 0.8903 | 0.7946 | 0.8330/0.7996 | 0.6935 | 0.6737 | 0.4507 | 0.3316 | 0.5974 | 67.2747 | 504 |

## 门禁与消融定位

| 变体 | Train fit | direct/阈值C | 身份F1 | 连续块 | AUROC | U@95 | direct RA | 阈值RA | ASafeR | 非全拒识 | 全部 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 未通过 | 通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |
| no_arpl | 未通过 | 未通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |
| untargeted_at | 未通过 | 通过/未通过 | 通过 | 通过 | 未通过 | 未通过 | 未通过 | 未通过 | 未通过 | 通过 | 未通过 |

- 主方案 all_rejection=False。
- ARPL 定位：main 相对 no_arpl 的 direct C +0.0106、AUROC +0.0042、U@95 +0.0729、direct RA +0.0254、ASafeR +0.0207、Score +5.8047，说明 ARPL 有正向贡献但不足以达到门禁。
- 目标 AT 定位：untargeted_at 相对 main 的 direct RA +0.0187、ASafeR +0.0108、U@95 +0.0316、Score +3.0021；在本实现中混合目标 AT 未带来收益，但仍未达到鲁棒门禁。
- 两个消融只用于上述定位；未做网格搜索，未据 outer/Test 调参。direct RA<0.95 时，即使阈值 RA 或 ASafeR 较高，也不宣称鲁棒门禁达标。

## SMOKE 数值与不变性

- 参数量：2992；在线 PGD7 batch64 实测 0.074s（仅速度基准）。
- 单样本/批次最大分数差 5.96e-08；置换差 0；标签一致=True。
- 方向导数 analytic=0.00062520、numeric=0.00053644，绝对误差 8.88e-05、相对误差 0.142；保留小导数量级下有限差分误差较大的数值限制。所有正式攻击梯度与分数均为 finite。

## Outer 与 Test

- Outer：N/A: at least one preregistered main inner gate failed; outer was not run。
- Test AUC、F1、C、RA、ASafeR、A、U、Score：N/A（无真实标签，且未读取 Test）。

## 证据边界

- unknown 全部是 Train 留被试 pseudo-unknown，不等于官方新被试；连续文件块是时间隔离代理，不是真实跨天/跨 Session 证据。
- Train fit=100% 仅为容量与训练流程检查，不是泛化证据。SMOKE 只用于速度、梯度和数值检查，不用于模型选择。
- 本地攻击覆盖 FGSM、非目标/目标 PGD20×SNR30/20，不能保证覆盖官方全部扰动；Score 是 Train 内代理，不是官方 Test 得分。
- 若门禁失败，本报告即为负结果归档；不通过削弱攻击、全拒识或外层/Test 调参补救。
