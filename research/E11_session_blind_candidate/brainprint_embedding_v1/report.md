# BrainPrint 无 Session 跨会话改进：模板融合基线 v1

## 目的

在不读取 Session 字段的推理链路中，先验证“基础 log-cov 双窗口分数 + 身份近邻校准”的可行性，为后续身份 embedding、模板匹配和联合拒识提供基线。

## 数据与协议

- 数据：`BrainPrint_cache/train_features.npy` 与 `train_metadata.csv`。
- 任务：EC、EO、Imagery、SSVEP。
- 特征：已有 cov 特征；近邻分支使用 `cov_1_8` 与 `cov_1_13` 两个频带视角。
- 基础分数：full cov Ridge 分数与 2 秒窗口分数各 0.5 融合。
- 近邻分支：按身份计算 top-3 近邻证据，使用嵌套折 Logistic 校准，再以 0.05 倍行内 z-score 加入基础分数。
- 未使用：`session`、`path`、`fs`；训练阶段只使用身份、任务和 trial 切分。
- `blocked_oof`：按 trial block 五折；不是严格的 session 留一，因为当前训练集实际只有 session1，无法从现有有标签数据估计真正的 session2 留一性能。
- `forward`：每个身份/任务后 20% trial 留出，作为时间外推压力测试。

## 结果

| 协议 | 样本数 | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|---:|
| blocked_oof | 11,304 | 0.999646 | 0.999651 | 0.999998 |
| forward | 2,308 | 0.998267 | 0.998229 | 0.999990 |

AUC 使用融合连续分数经 softmax 后计算；F1 为身份标签的 macro-F1。结果中的 AUC/F1 是训练集内部验证指标，不是 Test 指标。

## 解读

1. 近邻证据以较小权重加入基础分数后，blocked OOF 与 forward 均达到接近 100% 的内部识别性能；说明当前训练数据中存在很强的身份可分性。
2. 不能据此证明跨 Session 稳定。训练集只有 session1，现有验证只能证明 trial block/时间外推稳定，不能替代 session2 有标签验证。
3. 不能据此证明对抗稳定。当前结果尚未加入 PGD 或 raw EEG 攻击。
4. 下一步应把融合连续分数转成模板距离、top-1/top-2 margin、窗口一致性和质量分联合拒识，并在相同 blocked/forward 协议下报告拒识率、错误认证率、RA、ASafeR 和 U。

## 改动与可复现入口

- 修正 [evaluate_calibrated_neighbors.py](file:///d:/WORK/BCIC2026/BrainPrint/src/evaluate_calibrated_neighbors.py) 的缓存根目录，使其正确读取工作区内 `BrainPrint_cache`。
- 运行：`py -3.11 src/evaluate_calibrated_neighbors.py --protocols blocked_oof forward`
- 输出：`research/E11_session_blind_candidate/train_calibrated_neighbor_comparison.json`，以及 `BrainPrint_cache/train_*_scores_subject_calibrated_neighbors.npy`。

## 联合拒识实验 v1

在同一份有标签内部验证上，将 full cov、2 秒窗口和近邻身份证据融合，计算 top-1/top-2 margin、full/window 一致性和行内质量分，并分别用 5% 分位阈值拒识。结果如下：

| 协议 | 连续 Accuracy | 连续 Macro-F1 | 连续 Macro-AUC | 5% 阈值接受率 | 接受样本准确率 | 严格联合拒识接受率 |
|---|---:|---:|---:|---:|---:|---:|
| blocked OOF | 0.999646 | 0.999651 | 0.999998 | 94.993% | 1.000000 | 87.880% |
| forward | 0.998267 | 0.998229 | 0.999990 | 94.974% | 1.000000 | 87.868% |

5% 阈值下，三种单一/联合质量判据的结果几乎相同，且接受样本准确率为 100%；但这只是已知身份验证集上的阈值内插。严格联合规则将接受率降至约 87.9%，没有证据显示它优于简单 5% 分位拒识。训练数据只有 session1，因此这些结果不能证明跨 Session 或 Test 性能。

逐项结果和阈值保存在 `joint_rejection_metrics.json`。运行入口为 `run_joint_rejection.py`。

## 当前边界

- Test 没有标签，AUC/F1 无法计算。
- 没有真实攻击样本身份标签，不能从本实验推出官方 RA/ASafeR。
- 近邻分支属于可解释身份证据校准，不是已经训练完成的深度 embedding 模型。
- 训练数据没有第二个有标签 session，因此真正的跨会话结论需要额外标注数据或官方反馈，不能用伪答案替代。
