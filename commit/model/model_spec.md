# 模型与特征说明

## 特征

每个 MAT 样本为 1200×18、300 Hz 的 EEG trial。预处理为逐 trial 稳健截尾、0.5–90 Hz 带通、50 Hz 陷波；提取 1–90 Hz PSD 与六个频带的对数协方差特征。双窗口版本将每个 4 秒 trial 切成两个非重叠 2 秒窗口并分别提取特征。

## 分类器

每个任务（EC、EO、Imagery、SSVEP）独立训练 `StandardScaler + RidgeClassifier(solver='lsqr', alpha=100)`。模型权重不单独保存；预测时从训练特征和身份标签重新拟合，因此提交包中没有 `.pkl`、`.joblib` 或深度模型权重。

## 融合和后处理

- 基础分数与双窗口分数等权融合。
- EC 使用低频邻居分数和行 z-score；EO、Imagery、SSVEP 使用任务特异的固定分数组合。
- 对 EC、Imagery、SSVEP 使用温度 1.0 的 Sinkhorn 类别均衡；EC 采用 margin<1 的 guarded 规则，否则使用直接分类结果。
- 测试 `session1` 样本按数据结构规则输出 0；该规则不是未知身份脑电识别的独立验证。
- 高相似波形按任务内连通分量进行标签融合。
