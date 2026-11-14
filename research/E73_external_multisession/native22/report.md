# E73 外部跨 Session 身份验证实验

## 协议

- 数据：公开 BCIC IV-2a，9 名被试，T/E 双 Session；未读取竞赛 Train/Test。
- 模型：E54 EEGNet 编码器，单损失 cross-session-only SupCon，固定训练轮数。
- 隔离：run 0–2 训练、run 3 注册模板、run 4 内层标定、run 5 最终查询；三外层折的未知被试完全留出。
- 方向：T→E 与 E→T；主方法在任何外层评估前固定为 WCCN + top1/top2 margin。

## 主结果（3 外层折 × 2 方向的均值）

| 指标 | 数值 |
|---|---:|
| Pair AUC | 0.986889 |
| Detection AUC | 0.914144 |
| Detection EER | 0.153356 |
| C | 0.921875 |
| Identity macro-F1 | 0.957787 |
| Unknown-F1 | 0.779227 |
| U@95 | 0.659722 |
| U@98（描述性） | 0.510417 |
| 冻结 q02 的已知接受率 | 0.961806 |
| 冻结 q02 的未知拒识率 | 0.590278 |
| DIR@FPIR1% | 0.458333 |

## 阶段门禁

- 通过：`pair_AUROC`
- 未通过：`detection_AUROC`
- 未通过：`detection_EER`
- 未通过：`working_98_known_acceptance`
- 未通过：`working_98_unknown_rejection`

**总体结论：未通过 E73 阶段出口。**

Official Score=N/A（无对抗评估）；Test AUC/F1=N/A（未访问 Test）。

逐折数值、所有预声明评分方法、训练历史和模型哈希见 results.json 与 models/。
