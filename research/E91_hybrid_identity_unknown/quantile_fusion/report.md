# E91 quantile_fusion

## 结果

仅使用 Train/session1 和 E65 已生成的 855 维 correlation-logSPD 特征；未访问项目 Test、Test 索引或 Test 标签。

| AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.761304 | 0.530067 | 0.459172 | 0.109690 | 0.763757 | 0.311932 | 0.453646 | 0.890310 |

## 协议与泄漏审计

outer 为 5 折 subject 留出，known query 为每个 subject/task 的 >80% 时间块，unknown 为 outer 留出 subject；模板仅用 outer fit≤60%。每个 outer 内仅用两个 inner calibration 的 known 与留出 unknown 选择融合权重/分位数阈值，outer query 不参与调参。

E90 身份分数复用 task/time multi-template；E65 分支使用同一 logSPD 特征的身份原型负欧氏距离/rank。未发现评价标签或特征跨 outer 泄漏；E61 的 Test 频带筛查结果未用于本实验。

Test AUC/F1/C/U=N/A。结果仅代表 Train session1 的 pseudo-unknown/时间块证据边界。
