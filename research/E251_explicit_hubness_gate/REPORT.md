# E251 显式 Hubness 拒识门控实验报告

## 结论

- **“保严格 LOSO、提 Test S2”仍未解决。** 显式 hubness 惩罚在 Train-only 内层 LOSO 中被判为不合格，最终选中 `lambda = 0.00`，即不启用 hubness 惩罚。Test S2 接受率仍是 **0.4410**（目标 ≥0.70）。
- 本轮修正了初版（commit `9dca865`）的两个缺陷，并补齐了原先 `N/A` 的 #1–#4、#6：
  1. 攻击梯度 NaN：`torch.linalg.eigh` 在协方差矩阵出现重复特征值（实测最小特征值间距 = 0）时反向传播返回 NaN，经归一化后污染整个扰动，导致所有攻击条件都“全量拒识、RA=0、ASafeR=1”的假结果。已用 Daleckii–Krein 形式的 SPD 矩阵对数反向传播修复。
  2. 攻击目标 D 方向写反：初版最小化拒识 gate，与 E195 的“gate evasion”（最大化 gate）相反。已改正。
- **11 项中通过 5 项（#2 #3 #4 #6 #11），未通过 6 项（#1 #5 #7 #8 #9 #10）。** 修正后 #7/#8 是真实攻击读数（0.0350 / 0.0416），不是初版的伪通过。
- Test 官方 AUC/F1/C/RA/ASafeR/U/Score：**N/A**（Test 无身份真值）。

## 1. 协议

- 严格嵌套 identity LOSO，20 个外层折：
  - outer fit：其余 19 个身份，`frac <= 0.6`；
  - outer 评估：`frac > 0.8`（Late），outer 身份作为 pseudo-unknown；
  - inner fit：外层剩余身份中再留一人，其余 18 人 `frac <= 0.4`；
  - inner 校准：留出身份（unknown）与其余身份（known）的 `0.4 < frac <= 0.6`。
- 拒识分数：`s = top1_cosine − lambda * z(predicted_prototype_hubness)`。身份判决始终是未经校正的 strict LDA-cos top-1；每折逐样本断言 λ 不改变身份，reload 前后一致。
- 阈值：内层 identity-balanced weighted FAR10（每个 pseudo-unknown 身份等权，tie-safe）。
- λ 选择规则（预先固定，只看内层）：unknown rejection 不低于 C0、known acceptance 下降 ≤1 个百分点、AUC 与拒识 F1 不降、最差身份不恶化；满足者中取 known acceptance 最大。
- Test 只做一次性描述性审计，session 只用于事后分组。

## 2. λ 网格结果（20 折外层 LOSO，Late）

| 候选 | AUC | 拒识 F1 | known acceptance | unknown rejection | known id accuracy |
|---|---:|---:|---:|---:|---:|
| lambda_0.00 | 0.970837 | 0.529394 | 0.919270 | 0.917569 | 0.916808 |
| lambda_0.02 | 0.971065 | 0.529452 | 0.919365 | 0.917270 | 0.916949 |
| lambda_0.05 | 0.969695 | 0.516136 | 0.914302 | 0.919908 | 0.912000 |
| lambda_0.10 | 0.963173 | 0.464554 | 0.891840 | 0.926998 | 0.889884 |

判读：

- `lambda=0.02` 的 AUC/F1 只提高约 `+0.0002`，但 unknown rejection 降低 `0.0003`，内层 AUC 也从 0.981960 降到 0.981658，不满足预注册条件。
- `lambda>=0.05` 用 known acceptance 与 F1 明显下降换取 unknown rejection，代价不可接受。
- 因此内层选择 `lambda=0.00`。**在这套“原型 hubness”定义下，显式惩罚没有可用增益。**

## 3. 攻击评估（#7 / #8）

协议：Train `0.6 < frac <= 0.8`，每 task 至多 400 条（实际 1,227 条），20 步 PGD，L2 半径 = 去均值波形范数 / `10^(SNR/20)`，目标 A/B/C 攻身份头、D 攻 gate（最大化 gate，E195 口径）。阈值与 λ 用同一 Train-only 工作点。

| SNR | 目标 | RA | ASafeR | 接受率 | 攻击后闭集准确率 |
|---|---|---:|---:|---:|---:|
| clean | — | 0.9576 | 0.9984 | 0.9593 | 0.9919 |
| 30 | A | 0.0407 | 0.0416 | 0.9992 | 0.0407 |
| 30 | B | 0.0350 | 0.9617 | 0.0733 | 0.1051 |
| 30 | C | 0.0831 | 0.1826 | 0.9006 | 0.0994 |
| 30 | D | 0.9919 | 0.9919 | 1.0000 | 0.9919 |
| 20 | A | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 20 | B | 0.0000 | 0.9544 | 0.0456 | 0.0000 |
| 20 | C | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 20 | D | 0.9894 | 0.9894 | 1.0000 | 0.9894 |

- #7 主判定（SNR30 最差）= **0.0350**（目标 B）；#8 主判定（SNR30 最差）= **0.0416**（目标 A）。把 SNR20 计入最差则为 0.0000 / 0.0000。
- 目标 D 只开 gate、不改身份，因此 RA 接近 clean；这说明门控与身份头确实解耦，但也说明这条管线没有任何对抗鲁棒性。

**修复验证（不是数值巧合）**：

- 同一半径的随机噪声几乎不影响模型（256 条诊断样本：clean 0.9883 / SNR30 0.9883 / SNR20 0.9805），说明攻击利用的是真实梯度方向，而非“特征管线对任意扰动都敏感”。
- 攻击后特征仍有界（64 条诊断样本：|feature| 2.84 → 2.95），通道标准差不塌缩（最小 2.69，ε=1e-12），排除“把方差压到数值下限”的退化利用。
- 扰动范数逐步裁剪，最终 `‖δ‖/radius <= 1.0000001`。

## 4. Test 一次性描述性审计

冻结 `lambda=0.00`、full-Train 模型与阈值 `0.3594579100608826` 后只读一次 Test。

| session | n | acceptance | rejection | 分数中位数 | 被接受身份计数 CV |
|---|---:|---:|---:|---:|---:|
| S1 | 3,033 | 0.452028 | 0.547972 | 0.325374 | 3.918427 |
| S2 | 19,218 | 0.441045 | 0.558955 | 0.336337 | 1.070334 |

S1 vs S2 分数描述性 AUC = **0.462531**（S1 分数反而略高于 S2）。Test 官方 AUC/F1 = **N/A**。

## 5. 十一项指标（同一工作点，不跨点拼接）

| # | 指标 | 目标 | E251（lambda=0.00） | 判定 | 来源 |
|---:|---|---:|---:|---|---|
| 1 | Train in-sample 准确率 | =1.00 | 0.998816 | 未通过 | `gates_train.py` |
| 2 | 5 块 OOF 闭集准确率 | ≥0.95 | 0.996461 | 通过 | `gates_train.py` |
| 3 | 5 块 OOF 身份 macro-F1 | ≥0.92 | 0.996408 | 通过 | `gates_train.py` |
| 4 | 前向块外闭集准确率（`frac>0.6`） | ≥0.90 | 0.986365 | 通过 | `gates_train.py` |
| 5 | pseudo-unknown AUROC（strict LOSO） | ≥0.98 | 0.970837 | 未通过 | `summary.csv` |
| 6 | U@95%（strict） | ≥0.95 | 1.000000 | 通过（仅 Train 代理） | `gates_train.py` |
| 7 | RA（SNR30 最差目标） | ≥0.95 | 0.035045 | 未通过 | `gates_attack.py` |
| 8 | ASafeR（SNR30 同批攻击最差） | ≥0.98 | 0.041565 | 未通过 | `gates_attack.py` |
| 9 | Test S2 接受率 | ≥0.70 | 0.441045 | 未通过 | `audit_test.py` |
| 10 | Test S1 拒识率 | ≥0.70 | 0.547972 | 未通过 | `audit_test.py` |
| 11 | 严格 LOSO unknown rejection @ a=0.90 | >0.90 | 0.917569 | 通过 | `summary.csv` |

口径说明：

- #2/#3 按 E195 config 的定义（`block = clip(int(frac*5),0,4)`，留一 block 拟合，pooled 评估），#4 用 `frac<=0.6` 拟合、`frac>0.6` 评估。
- E249 报告里的 #2 实际是 known identification accuracy（0.9263）、#4 是 Late 闭集准确率（0.9873），与本表的 #2/#4 不是同一口径，不能逐格比较。
- #6 的 1.0000 只是同会话伪未知结果；同一阈值下 Test S1 只拒 0.548，不能宣称可迁移。
- #5/#11 均为 strict 口径（留出身份不进入 z 统计、PCA、原型与 hubness）。

## 6. 与 E249 / E250 的关系

- E251 的 `lambda=0.00` 就是 E247/E249 的严格 LDA-cos 基线：AUC、known/unknown acceptance 与 E249 C0 完全一致（0.970837 / 0.919270 / 0.917569），说明主流程复现无误。
- E249 的 C2（WCCN+漂移）在 Train 代理上更好（AUC 0.9779），但 Test S2 反而更低（0.429），本次 hubness 方向也没有改变这一结论。
- E249 的 #7/#8 由另一套攻击脚本、另一套模型（C2）产生；本次修复的 NaN 缺陷属于 E251 自己的实现。要对齐比较，需要在同一脚本下重跑 C2。

## 7. 结论与下一步

1. 显式 hubness 惩罚方向**证伪**：λ>0 只能用 known acceptance 换 unknown rejection，内层选择规则直接判其不合格。
2. S2 缺口不是“少数原型吸附”这一单一机制能解释的；S1 被接受样本的身份分布确实极不均匀（CV 3.92），但惩罚 hub 并不改善 S2。
3. 下一步不应继续在同一 Train-only 代理上微调评分函数。可行方向需要真实跨会话信息（官方 S2 注册样本，若规则允许）或对会话不敏感的表征，并且必须先预注册协议再看 Test。

## 8. 产物与复现

- `run_e251.py` → `results/summary.csv`、`results/per_fold.csv`、`results/outer_01..20`
- `gates_train.py` → `results/gates_train.json`（#1–#4、#6）
- `gates_attack.py` → `results/gates_attack.json`（#7/#8，含逐条件）
- `audit_test.py` → `results/test_audit.json`（#9/#10，一次性）

```powershell
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\run_e251.py --validate-only
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\run_e251.py --all-folds --resume
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\gates_train.py
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\gates_attack.py
& .\.venv\Scripts\python.exe research\E251_explicit_hubness_gate\audit_test.py
```

未修改 `final/`、E249、E250。
