# E111 External identity+session-invariant embedding (LOSO)

外部 BCIC IV-2a 双 Session 上训练 `SupCon(跨 Session) + λ·SupCon(同身份)`，用 E54 EEGNet 骨干，冻结后在项目 Train/session1 上做 LOSO。未访问项目 Test。

## 结果

| 信号 | LOSO AUC | known acc | 留出者拒识均值 | 中位数 | ≥0.80 折数 | ≥0.50 折数 | 相对 B4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4_E102_spd_ratio | 0.9144 | 0.8499 | 0.8310 | 0.8630 | 16/20 | 19/20 | +0.0000 |
| E111_external_joint_ratio | 0.5781 | 0.8499 | 0.1919 | 0.1843 | 0/20 | 0/20 | -0.6390 |

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（未访问 Test）。

## 结论

**外部预训练嵌入在 LOSO 上严重失败**：留出者拒识 0.1919，比 B4 低 0.639，AUC 仅 0.5781。

原因分析：

1. **外部数据只有 9 名被试**，而项目有 20 名。9 个身份学到的判别边界无法覆盖 20 个身份的开放集结构。
2. **22→18 IDW 投影损失信息**。BCIC IV-2a 的 22 通道经 IDW 投影到项目 18 通道后，空间分布与项目原生 18 通道不一致（E103 已发现外部 ratio 中位数 1.049 vs 项目 1.140）。
3. **跨 Session SupCon 会压缩身份间距离**。即使加了同身份 SupCon，跨 Session 正例对仍主导梯度（cross loss 3.91 vs id loss 3.87，两者几乎相等但 cross 项覆盖更多样本对），导致嵌入空间被"会话不变"目标主导，身份可分性下降。
4. **EEGNet 骨干在外部数据上过拟合**。30 epoch 后 loss 仍在下降（10.81 → 7.78），没有收敛迹象，但外部只有 5184 条 trial。

**结论**：外部数据（9 被试、22 通道、运动想象范式）与项目数据（20 被试、18 通道、静息+SSVEP+想象）的域差异过大，无法通过预训练迁移。这条路走不通。

