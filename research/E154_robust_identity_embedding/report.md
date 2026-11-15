# E154 鲁棒身份一致性 embedding

设备：cuda；特征：E65 855维，经 Train-only PCA-96。

## 最佳结果

最佳方案：**augmentation_consistency**；LOSO拒识=0.9470；Test S1拒识=0.5602；Test S2接受=0.3561；W=0.5548。
Train OOF AUC=0.9861；Train OOF F1=0.9920。Test AUC：N/A（无真实标签）；Test F1：N/A（无真实标签）。

阈值来自五块 Train OOF；每个 LOSO 评分同时排除被评分身份和被评分块。Test 仅评分，Session 仅在评分后作诊断，不进入训练、阈值或选择。

|方案|LOSO拒识|S1拒识|S2接受|W|Train OOF AUC|Train OOF F1|
|---|---:|---:|---:|---:|---:|---:|
|baseline_control|0.9448|0.5605|0.3551|0.5539|0.9858|0.9919|
|augmentation_consistency|0.9470|0.5602|0.3561|0.5548|0.9861|0.9920|

增强方案为同 trial 幅值缩放、特征噪声和特征遮挡，使用分类损失加增强前后 embedding 一致性损失；baseline 为分类控制。模型已保存为 model_baseline_control.pt 与 model_augmentation_consistency.pt。

证据边界：Train 内时间块与留身份代理评估；Test 无身份真值，Session 结果仅诊断。
