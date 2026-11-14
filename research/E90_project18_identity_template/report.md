# E90 project18 身份模板（Train-only）

## 结论与当前 AUC / F1

仅使用 Train/session1 的 18 通道、E65 855 维相关 logSPD 特征；未读取项目 Test、Test 索引或 Test 标签。

| 模板 | AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_centroid | 0.788622 | 0.892507 | 0.953336 | 0.734207 | 0.383593 | 0.290449 | 0.878349 | 0.265793 |
| task_time_multi_template | 0.865880 | 0.966866 | 0.964040 | 0.683755 | 0.445393 | 0.208698 | 0.952218 | 0.316245 |

## Inner calibration summary

| 模板 | AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_centroid | 0.787732 | 0.911577 | 0.950863 | 0.714393 | 0.394855 | 0.278607 | 0.893666 | 0.285607 |
| task_time_multi_template | 0.817526 | 0.963393 | 0.952011 | 0.664412 | 0.444315 | 0.256167 | 0.942090 | 0.335588 |

## 协议

- outer：5 折留出被试（每折 4 人）；outer fit 为注册被试各 subject/task 的前 80%，outer known query 为 >80% 的末端时间块，outer unknown 为留出被试全部 Train trials。
- inner：每个 outer 的 16 名已知被试再分 8/8；模型只在 enrolled 的 ≤60% 时间块拟合，calibration known 为 60–70%，calibration unknown 为留出 8 人 ≤80%；阈值仅由 inner calibration known q05 产生，outer query 不参与标定。
- single centroid：每个身份一个归一化均值模板；task/time-stratified multi-template：每个身份按任务和时间分层建立多个归一化均值模板，查询取该身份模板最高 cosine。
- 报告 unknown FAR、unknown-F1 使用 pos_label=0，EER 为 known/unknown confidence 等错误率交点；Test 指标全部 N/A。

## 实际命令

```powershell
.venv/Scripts/python.exe research/E90_project18_identity_template/run_e90.py --smoke
.venv/Scripts/python.exe research/E90_project18_identity_template/run_e90.py --full
```

模型/模板：`identity_templates.pt`；逐样本 Train 外层预测：`sample_predictions.csv`；结果：`results.json`。
