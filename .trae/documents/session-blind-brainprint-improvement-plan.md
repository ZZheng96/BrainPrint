# 无 Session 依赖跨会话脑纹识别改进计划

## Context

现有候选主要是 log-covariance + 任务特异 Ridge + max-logit 拒识。v4 伪答案评估已证明 Test 真值不可由伪答案可靠恢复，因此后续以有标签训练数据的跨会话验证为主，不再继续调伪答案。目标是提升同一身份跨会话稳定性、对抗稳定性和未知/攻击样本的正确拒识。

## 改进对象与目标

1. **身份表征**：从直接分类分数扩展为稳定的身份 embedding/模板距离，减少对单一线性分类头的依赖。
2. **跨会话稳定性**：以会话留一或批次留一的外层验证评估，优先处理功率、协方差和通道漂移；推理不读取 session/trial/path/fs。
3. **拒识机制**：联合最近身份距离、第一二名间隔、多窗口一致性和信号质量，而不是只用 max-logit。
4. **对抗稳定性**：先做特征级、再做原始 EEG 波形级攻击，区分保持原身份、正确拒识和错误认证。

## 借鉴人脸识别

借鉴身份 embedding、注册模板、质量感知、距离阈值与 margin 拒识、抗冒认训练；不直接照搬大网络。EEG 先以现有 log-cov/SPD 特征做小型度量学习基线，控制模型复杂度和过拟合风险。

## 执行顺序

1. 建立新实验目录 `research/session_blind_candidate/brainprint_embedding_v1/`，复用现有缓存和 GPU Ridge/Neighbor 组件。
2. 实现严格 session/采集批次分组的外层评估；内层只用训练折选择归一化、模板聚合、距离/间隔阈值和拒识阈值。
3. 先跑可解释基线：log-cov 归一化、类模板/类内 top-k 相似度、双窗口聚合、质量分；输出闭集识别、拒识、误认证和官方代理分项。
4. 在基线有效后加入小型 embedding 模型与 supervised contrastive/ArcFace 风格损失；采用会话增强、通道 dropout、频带扰动，避免直接强域对抗。
5. 接入 feature PGD 和 raw EEG PGD；明确记录攻击空间，不能把特征空间结果称为 raw EEG 官方攻击结果。
6. 仅当跨会话 clean、RA、ASafeR、U 和错误认证率同时改善，才生成新的 Test CSV；不以训练集准确率或拒识率单独替换现有提交。

## 借鉴与复用

- [session_blind_candidate/run.py](file:///d:/WORK/BCIC2026/BrainPrint/src/session_blind_candidate/run.py)：无 Session 读取约束、缓存加载和候选生成结构。
- [open_set_improvement/core.py](file:///d:/WORK/BCIC2026/BrainPrint/src/open_set_improvement/core.py)：TorchRidge、TorchNeighbor、任务特异分数与 GPU 执行。
- [eeg_features.py](file:///d:/WORK/BCIC2026/BrainPrint/src/eeg_features.py)：log-cov、功率尺度归一化和质量字段。
- [raw_eeg_defense/gpu_attack.py](file:///d:/WORK/BCIC2026/BrainPrint/src/raw_eeg_defense/gpu_attack.py)：原始 EEG GPU 攻击组件。

## 报告与验收

每个实验目录必须有 `report.md`，明确当前 AUC 和 F1。报告分离观测结果、代理指标和假设；AUC 只有存在连续分数与标签时计算。所有外层验证必须记录折级结果和最差折，不允许把伪答案结果当官方 Test 真值。
