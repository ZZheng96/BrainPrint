# E202 双模型跨会话与拒识实验报告

## 协议

- 模型 A：E65 correlation-logSPD（Train-only SVD→96 维）+ 冻结 E162 embedding（256 维），GPU 训练身份 embedding，损失为 identity CE + supervised contrastive。
- 模型 B：冻结模型 A 后独立训练的未知拒识 gate（top1 相似度、top1-top2 margin、E65/E162 辅助匹配特征）。
- 注册：只用注册身份的 `frac<=0.6` fit rows 建立 prototype；被评估身份不进入该折 SVD、prototype 或 gate 训练。
- Test 只做一次冻结审计：全量 20 人 fit rows 训练，冻结后输出全部标签，最后才读取 `session` 统计 S1/S2。

## 11 项指标实测

以下除第 9、10 项外均为 Train 内多折实测（留 2/5/10 各 2 折，共 6 折）；第 11 项为 20 折严格 LOSO；第 9、10 项为冻结后 Test 单次审计。

| # | 指标 | 实测值 | 目标 | 判定 |
|---:|---|---:|---:|---|
| 1 | Train 拟合准确率 | **1.0000** | 1.00 | 达标（仅容量检查） |
| 2 | 注册身份闭集准确率 C（代理） | **0.9942–0.9959** | ≥0.95 | 达标（Train holdout 代理） |
| 3 | 身份 macro-F1 | **0.9940–0.9958** | ≥0.92 | 达标 |
| 4 | 连续时间块准确率（5 块均值） | **0.9942–0.9959**，最差块 0.9869–0.9910 | ≥0.90 | 达标 |
| 5 | pseudo-unknown AUROC | **0.9782–0.9875** | ≥0.98 | K=2/5 达标，K=10 未达 |
| 6 | U@95（已知接受 95% 时未知拒识） | **0.8942–0.9513** | ≥0.95 | K=10 未达 |
| 7 | RA（特征空间 PGD 代理） | **0.8410–0.8805** | ≥0.95 | **未达** |
| 8 | ASafeR（特征空间 PGD 代理） | **0.9845–0.9867** | ≥0.98 | 达标（代理口径） |
| 9 | Test S1 拒识率 | **0.7778**（3033 行） | ≥0.70 | 达标 |
| 10 | Test S2 接受率 | **0.2176**（19218 行） | ≥0.70 | **严重未达** |
| 11 | 严格 LOSO 拒识率（20 折） | **均值 0.9847、最低 0.9159、标准差 0.0216** | >0.90 | 达标 |

### 分折明细

| 留出 K | 折数 | C | 身份 macro-F1 | 块准确率均值 | 最差块 | AUROC | U@95 | RA 代理 | ASafeR 代理 | τB 下已知接受 | τB 下未知拒识 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.9949 | 0.9947 | 0.9950 | 0.9869 | 0.9854 | 0.9513 | 0.8410 | 0.9867 | 0.8940 | 0.9886 |
| 5 | 2 | 0.9959 | 0.9958 | 0.9959 | 0.9910 | 0.9875 | 0.9286 | 0.8683 | 0.9857 | 0.8992 | 0.9779 |
| 10 | 2 | 0.9942 | 0.9940 | 0.9942 | 0.9886 | 0.9782 | 0.8942 | 0.8805 | 0.9845 | 0.8954 | 0.9590 |

严格 LOSO 20 折：全部注册 19 人、留 1 人为未注册；平均注册识别率 `0.9954`，平均已知接受 `0.8942`；最弱身份为 Subject 10（拒识 `0.9159`）。

### 冻结 Test 审计

- 输出行数 22251，整体判为未知比例 `0.7818`。
- S1（session1，3033 行）拒识率 `0.7778`。
- S2（session2，19218 行）接受率 `0.2176`。
- 冻结阈值 `τB=0.9362`，Train 已知接受 `0.8935`。
- 提交文件：[submission_e202.csv](file:///d:/WORK/BCIC2026/BrainPrint/research/E202_dual_model_cross_session/submission_e202.csv)
- Test 官方 AUC/F1/C/RA/ASafeR/U/Score：**N/A**（无真实标签）。

## 结论

- **达标**：训练拟合、C 代理、身份 macro-F1、连续块准确率、严格 LOSO（均值 0.9847）。
- **部分达标**：pseudo-unknown AUROC（K=10 掉到 0.9782）、U@95（K=10 掉到 0.8942）、ASafeR 代理。
- **未达标**：RA 代理（0.84–0.88）、**Test S2 接受率（0.2176）**。
- 核心矛盾已经定位：Train 内身份区分度约 0.995、LOSO 拒识约 0.985，但换成真实跨会话 Test 后，已注册者被大面积拒识。即**跨会话分布偏移**没有被身份 embedding 消除，gate 在 Test 上把大量合法注册样本推到了阈值以下。
- Test S1 拒识 0.7778 说明 gate 并非"全拒"，而是 Separating 能力在跨会话下丢失；单纯调低 τB 会把 S2 抬高但同步压低 S1，属于已知的跷跷板。
- 因此后续必须针对"S2 接受率"做跨会话对齐（例如用 Train 内的 session 风格扰动、协方差域对齐、以及只用 Train 标定的分数归一化），而不是继续优化 LOSO。禁止用 Test S1/S2 反选阈值或融合权重。

## 产物

- 指标明细：[metrics.json](file:///d:/WORK/BCIC2026/BrainPrint/research/E202_dual_model_cross_session/metrics.json)
- 严格 LOSO：[loso20.json](file:///d:/WORK/BCIC2026/BrainPrint/research/E202_dual_model_cross_session/loso20.json)
- Test 审计：[test_audit.json](file:///d:/WORK/BCIC2026/BrainPrint/research/E202_dual_model_cross_session/test_audit.json)
- 冻结包：[frozen_bundle_e202.npz](file:///d:/WORK/BCIC2026/BrainPrint/research/E202_dual_model_cross_session/bundle/frozen_bundle_e202.npz)
- 权重：`checkpoints/model_a_frozen20.pt`、`model_a_loso1_*.pt`、`model_a_k*.pt`
