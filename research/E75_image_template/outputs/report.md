# E75 冻结 EEG 图像模板实验

## 协议

- 数据：E73 派生的 BCIC IV-2a 外部数据；训练 run 0–2，模板 run 3，标定 run 4，评估 run 5。
- 三折 enrolled / calibration-unknown / outer-unknown 被试隔离；unknown 不参与训练、模型选择或阈值选择。
- 图像：现有 `topomap5` 编码器 + 冻结 DINOv2 ViT-B/14；未访问竞赛 Test。

## 三折均值

| 指标 | 数值 |
|---|---:|
| Detection AUC | 0.464012 |
| Unknown F1 | 0.715853 |
| Identity macro-F1 | 0.337257 |
| Known identity accuracy | 0.531250 |
| Unknown rejection | 0.057870 |

Official Score=N/A；Test AUC/F1=N/A（竞赛 Test 未访问）。
