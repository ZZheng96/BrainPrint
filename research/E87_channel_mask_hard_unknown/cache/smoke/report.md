# E87 通道掩码 + 分层 hard-unknown

- 外部数据：E73 派生 BCIC IV-2a；22 通道源表征显式投影为 18 通道，未使用项目 Test。
- 18 通道映射见 `config.json`；缺失通道为零掩码，不复用旧 22 通道 checkpoint。
- hard-unknown：按被试 outer 留出，固定 run=5；阈值仅由 enrolled known run=4 q05 固定。
- unknown-F1 使用 `pos_label=0`。

| 指标 | 结果 |
|---|---:|
| Outer AUC/F1/EER | N/A（smoke 未做 outer 评估） |
| 项目 Test AUC/F1 | N/A（未访问 Test） |
