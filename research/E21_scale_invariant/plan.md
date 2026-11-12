# E21：跨会话尺度不变特征

## 动机

E20I 证明 Test 上的分类分数整体只有训练的一半，Session1 与 Session2 分布几乎相同，
说明训练（仅 Session1）学到的决策边界无法迁移到 Test。E20 的路线（图像化 + 冻结视觉骨干）
在开集拒识上显著弱于纯协方差，且骨干升级（DINOv3）无效，判定瓶颈在表征的跨会话稳定性。

## 硬约束（不可违反）

- 纯 EEG：不读 session、不用文件顺序、不用样本数量、不用正常/对抗分组、不用近重复配对。
- 阈值只能由内层伪未知产生，外层一次性评估。
- 任何变换必须逐样本可归纳（inductive），禁止批次对齐（EA、target z-score、BatchNorm 统计对齐）。
- 目标：Train 闭集 100% 区分；Test Session2 拒识率接近 0；同时优化 C / RA / ASafeR / U。

## 步骤

### E21A 偏移性质量化（只做诊断，不做决策）

对比 Train 与 Test（S1/S2）的协方差特征统计：

1. 全局尺度：逐样本 L2 范数、逐频带对角线（log 方差）均值。
2. 方向漂移：逐维均值向量的余弦相似度、最佳标量拟合残差。
3. 漂移分布：对角 vs 非对角维度的偏移幅度。
4. 判断属于「全局幅度缩放」还是「方向性漂移」。

### E21B 尺度不变表示

依 E21A 结论选择：

- 若为全局尺度漂移：逐频带扣对角线均值（determinant normalization）、log 协方差除以 trace、相关矩阵。
- 若为方向漂移：需要更强的结构对齐（通道间相对关系、频带归一化），仍须逐样本。

在 E21B 用严格嵌套协议（外层 5 折留被试 + 内层伪未知阈值）评估 C / U / AUROC，
并与 E20D 的 covariance_only 对照。

### E21C Test 审计

用 E21B 最优表示生成 Test 候选，写出后才读 session 做无标签审计，统计 Session2 拒识率。

### E21D 对抗鲁棒性

复用 `src/raw_eeg_defense/gpu_attack.py` 的 `raw_pgd` 与 `src/raw_eeg_defense/run.py` 的
`competition_metrics`，补齐 RA / ASafeR。

## 记录规范

每步写 `report.md`，明确 AUC 与 F1（无标签则写 N/A 及原因）；及时 commit + push。
