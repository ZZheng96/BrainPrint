# E88 标准电极二维空间投影 + 分层 hard-unknown

- 外部数据：E73 派生 BCIC IV-2a；22 通道源表征显式投影为 18 通道，未使用项目 Test。
- 18 通道由标准 10-20/10-10 二维坐标计算 nearest 或 IDW 插值，保留 22 通道信号信息；不访问项目 Test。
- hard-unknown：按被试 outer 留出，固定 run=5；阈值仅由 enrolled known run=4 q05 固定。
- unknown-F1 使用 `pos_label=0`。

| 指标 | 结果 |
|---|---:|
| AUC | 0.905129 |
| known acceptance | 0.924479 |
| unknown FAR | 0.310185 |
| identity macro-F1 | 0.959585 |
| unknown-F1 | 0.760630 |
| EER | 0.169850 |
| 项目 Test AUC/F1 | N/A（未访问 Test） |
