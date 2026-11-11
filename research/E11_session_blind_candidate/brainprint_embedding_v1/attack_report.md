# 模板融合基线攻击评估记录

## 本轮执行

运行了已有 raw EEG PGD + 对抗 Ridge 流程的 smoke 验证：

```powershell
$env:PYTHONPATH='.'
py -3.11 -m src.raw_eeg_robust.run --smoke --workers 1 --batch-size 32
```

此前直接执行脚本会因项目包路径缺少 `PYTHONPATH` 而失败，模块方式运行已解决该问题。

## 冒烟结果

| 方法 | C | RA | ASafeR | A | U | 官方公式分 | unknown AUC | unknown F1 | clean F1 | adv F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 0.976563 | 0.007813 | 0.742188 | 0.228125 | 0.937500 | 58.11 | 0.980713 | 0.923077 | 0.929412 | 0.013072 |
| adversarial_ridge | 0.984375 | 0.062500 | 0.890625 | 0.310938 | 0.906250 | 64.42 | 0.978271 | 0.920635 | 0.933333 | 0.091503 |

## 解释

在同一组 SNR20 untargeted raw EEG PGD 冒烟攻击下，对抗增强 Ridge：

- 官方公式分提高 `+6.31`；
- RA 从 `0.0078` 提高到 `0.0625`；
- ASafeR 从 `0.7422` 提高到 `0.8906`；
- clean F1 略升；
- adversarial F1 明显升高，但仍然较低；
- unknown AUC/F1 略有下降，说明鲁棒增强存在开放集性能代价。

这支持“对抗增强应与拒识联合优化”的方向，但不能据此宣称最终方案已具备对抗稳定性。

## 严格边界

- 这是 smoke/单外层折结果，不是完整五折验收。
- 使用的是现有 raw waveform PGD 流程，评估模型是对抗增强 Ridge，不是深度 embedding 模型。
- 当前攻击评估使用训练集 Session1 的外层 subject 划分；没有有标签 Session2。
- unknown 是折内伪未知身份，不是真实 Test unknown。
- AUC 在此报告中指 unknown AUROC；F1 同时列 clean/adv/unknown，不能替代官方 Test F1。
- 正式提交文件未修改。

## 下一步

1. 完成非 smoke 五折 raw EEG PGD 验收；
2. 把模板融合的 margin、窗口一致性和质量分接入攻击后拒识；
3. 分别报告错误认证、正确身份保持和安全拒识；
4. 只有在五折 RA、ASafeR、unknown AUC/F1 同时满足门禁后，才生成新的 Test CSV。
