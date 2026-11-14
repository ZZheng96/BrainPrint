# E96 domain robust open set

- 冻结 E95 raw identity encoder；E73 22 通道源数据按 IDW(k=4,p=2)投影为 18 通道。外部 BCIC IV-2a 只作为独立域 unknown/校准负例，未读取项目 Test 进行调参。

## Train / 外部

| 数据 | AUC | F1 | known acceptance | unknown FAR | unknown rejection |
|---|---:|---:|---:|---:|---:|
| Train pseudo-unknown calibration | 0.651829 | 0.874598 | 0.950027 | 0.940909 | 0.059091 |
| E73 external | 0.859713 | 0.436651 | 0.950027 | 0.862076 | 0.137924 |

## 项目 Test

真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。S1/S2 仅在模型和参数冻结后一次性读取并做工程审计：S1 拒识率=0.40685789647213977，S2 接受率=0.8767821833697576。
