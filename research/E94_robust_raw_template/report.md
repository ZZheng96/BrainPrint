# E94 Train-only 原始EEG鲁棒身份+拒识

仅访问 Train/session1 的 18 通道原始 EEG；项目 Test 目录、test.csv、test_metadata.csv 均未访问。项目 Test 指标：N/A。

## 当前结果

| AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.699405 | 0.567347 | 0.480298 | 0.156752 | 0.731967 | 0.350905 | 0.464856 | 0.843248 |

## 方法与边界

EEGNet1d 在 GPU 训练；训练增强包含通道 dropout、循环时间平移、幅度扰动和低强度原始波形 PGD。outer 留出被试，inner 再留 pseudo-unknown 被试；阈值由 inner known/pseudo-unknown confidence 校准。C 为已知身份拒识后的正确率，U 为未知拒识率。

模型 checkpoint：`model_checkpoint.pt`；结构化结果：`results.json`。
