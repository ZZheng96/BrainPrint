# E140 监督度量学习嵌入（ArcFace / batch-hard triplet）

目标：把 Test 诊断 AUC（session1 负类 vs session2 正类）从 PCA-96 基线的 0.7566 抬高，使三项均值 (LOSO 拒识 + Test S1 拒识 + Test S2 接受)/3 尽量逼近 0.85。

**结论：失败。** 9 组监督度量学习配置的 Test AUC 只有 0.49~0.57（最好 G 组 0.5698），
全部低于同协议下的未学习 PCA 对照（0.7599）与基线（0.7566）；三项均值最高 0.6799，
低于基线 0.7784，距离 0.85 更远。监督度量学习在这份单会话 Train 上只会放大"同会话身份可分"这一
已知与 Test 可分性负相关的捷径。

## 协议（全流程 Train-only）

- 拟合块：frac ≤ 0.6（训练、标准化、PCA 基底均只用这块）；参考块：frac ∈ (0.6, 0.8]，训练时完全未见过。
- 预处理：按 task 用拟合块均值/方差标准化 → PCA 到 96/128/160 维 → 按主成分标准差缩放。
- 嵌入网络：MLP（input→512→256→emb，BatchNorm+ReLU+Dropout，L2 归一化），20 类 ArcFace 和/或 batch-hard triplet。
- 增强（在原始 855 维空间做，再投影）：5 个 171 维频带随机缩放/丢弃、高斯噪声、特征 dropout、整体缩放。
- 打分：嵌入空间 (task, identity) k-means 原型（k∈{1,3,8}），分数 = 负最近欧氏距离，再按 task 用参考块统计量做 z 分数。
- 阈值：参考块 (frac∈(0.6,0.8]) 分数分位数；LOSO 每折用已注册身份同区间行做已知端、留出身份同区间行做未知端，阈值取已知端分位数。
- 主配置为预注册（PCA 128 / 128 维嵌入 / ArcFace / mild 增强），其余变体仅作透明性记录；**未用 Test 或 LOSO 指标做任何选择**。

## 各变体 Test 诊断 AUC

| 变体 | PCA | 嵌入 | 损失 | 增强 | 原型源 | k | Test AUC(z) | Test AUC(raw) |
|---|---:|---:|---|---|---|---:|---:|---:|
| A_D96_e128_arc_mild | 96 | 128 | arcface | mild | fit | 1 | 0.4933 | 0.4942 |
| A_D96_e128_arc_mild | 96 | 128 | arcface | mild | all | 1 | 0.4892 | 0.4891 |
| A_D96_e128_arc_mild | 96 | 128 | arcface | mild | fit | 3 | 0.4990 | 0.4995 |
| A_D96_e128_arc_mild | 96 | 128 | arcface | mild | all | 3 | 0.4935 | 0.4923 |
| A_D96_e128_arc_mild | 96 | 128 | arcface | mild | fit | 8 | 0.4950 | 0.4948 |
| A_D96_e128_arc_mild | 96 | 128 | arcface | mild | all | 8 | 0.4987 | 0.4964 |
| B_D128_e128_arc_mild | 128 | 128 | arcface | mild | fit | 1 | 0.5327 | 0.5318 |
| B_D128_e128_arc_mild | 128 | 128 | arcface | mild | all | 1 | 0.5365 | 0.5351 |
| B_D128_e128_arc_mild | 128 | 128 | arcface | mild | fit | 3 | 0.5361 | 0.5352 |
| B_D128_e128_arc_mild | 128 | 128 | arcface | mild | all | 3 | 0.5405 | 0.5379 |
| B_D128_e128_arc_mild | 128 | 128 | arcface | mild | fit | 8 | 0.5371 | 0.5356 |
| B_D128_e128_arc_mild | 128 | 128 | arcface | mild | all | 8 | 0.5483 | 0.5472 |
| C_D160_e128_arc_mild | 160 | 128 | arcface | mild | fit | 1 | 0.5051 | 0.5047 |
| C_D160_e128_arc_mild | 160 | 128 | arcface | mild | all | 1 | 0.5028 | 0.5023 |
| C_D160_e128_arc_mild | 160 | 128 | arcface | mild | fit | 3 | 0.5099 | 0.5095 |
| C_D160_e128_arc_mild | 160 | 128 | arcface | mild | all | 3 | 0.5050 | 0.5037 |
| C_D160_e128_arc_mild | 160 | 128 | arcface | mild | fit | 8 | 0.5061 | 0.5060 |
| C_D160_e128_arc_mild | 160 | 128 | arcface | mild | all | 8 | 0.5163 | 0.5149 |
| D_D128_e128_triplet_mild | 128 | 128 | triplet | mild | fit | 1 | 0.4988 | 0.4991 |
| D_D128_e128_triplet_mild | 128 | 128 | triplet | mild | all | 1 | 0.4984 | 0.4983 |
| D_D128_e128_triplet_mild | 128 | 128 | triplet | mild | fit | 3 | 0.4963 | 0.4964 |
| D_D128_e128_triplet_mild | 128 | 128 | triplet | mild | all | 3 | 0.5000 | 0.5003 |
| D_D128_e128_triplet_mild | 128 | 128 | triplet | mild | fit | 8 | 0.4983 | 0.4986 |
| D_D128_e128_triplet_mild | 128 | 128 | triplet | mild | all | 8 | 0.5020 | 0.5048 |
| E_D128_e128_arc_triplet_mild | 128 | 128 | arc_triplet | mild | fit | 1 | 0.5242 | 0.5236 |
| E_D128_e128_arc_triplet_mild | 128 | 128 | arc_triplet | mild | all | 1 | 0.5285 | 0.5272 |
| E_D128_e128_arc_triplet_mild | 128 | 128 | arc_triplet | mild | fit | 3 | 0.5256 | 0.5249 |
| E_D128_e128_arc_triplet_mild | 128 | 128 | arc_triplet | mild | all | 3 | 0.5351 | 0.5337 |
| E_D128_e128_arc_triplet_mild | 128 | 128 | arc_triplet | mild | fit | 8 | 0.5249 | 0.5241 |
| E_D128_e128_arc_triplet_mild | 128 | 128 | arc_triplet | mild | all | 8 | 0.5392 | 0.5370 |
| F_D128_e64_arc_mild | 128 | 64 | arcface | mild | fit | 1 | 0.5335 | 0.5345 |
| F_D128_e64_arc_mild | 128 | 64 | arcface | mild | all | 1 | 0.5338 | 0.5343 |
| F_D128_e64_arc_mild | 128 | 64 | arcface | mild | fit | 3 | 0.5357 | 0.5361 |
| F_D128_e64_arc_mild | 128 | 64 | arcface | mild | all | 3 | 0.5405 | 0.5414 |
| F_D128_e64_arc_mild | 128 | 64 | arcface | mild | fit | 8 | 0.5321 | 0.5325 |
| F_D128_e64_arc_mild | 128 | 64 | arcface | mild | all | 8 | 0.5468 | 0.5478 |
| G_D128_e128_arc_strong | 128 | 128 | arcface | strong | fit | 1 | 0.5523 | 0.5512 |
| G_D128_e128_arc_strong | 128 | 128 | arcface | strong | all | 1 | 0.5556 | 0.5539 |
| G_D128_e128_arc_strong | 128 | 128 | arcface | strong | fit | 3 | 0.5588 | 0.5565 |
| G_D128_e128_arc_strong | 128 | 128 | arcface | strong | all | 3 | 0.5632 | 0.5613 |
| G_D128_e128_arc_strong | 128 | 128 | arcface | strong | fit | 8 | 0.5563 | 0.5544 |
| G_D128_e128_arc_strong | 128 | 128 | arcface | strong | all | 8 | 0.5698 | 0.5695 |
| H_D128_e128_arc_mild_ep20 | 128 | 128 | arcface | mild | fit | 1 | 0.5130 | 0.5116 |
| H_D128_e128_arc_mild_ep20 | 128 | 128 | arcface | mild | all | 1 | 0.5213 | 0.5175 |
| H_D128_e128_arc_mild_ep20 | 128 | 128 | arcface | mild | fit | 3 | 0.5189 | 0.5164 |
| H_D128_e128_arc_mild_ep20 | 128 | 128 | arcface | mild | all | 3 | 0.5272 | 0.5230 |
| H_D128_e128_arc_mild_ep20 | 128 | 128 | arcface | mild | fit | 8 | 0.5256 | 0.5239 |
| H_D128_e128_arc_mild_ep20 | 128 | 128 | arcface | mild | all | 8 | 0.5292 | 0.5249 |
| I_D128_e64_arc_strong_wd1e-2 | 128 | 64 | arcface | strong | fit | 1 | 0.5241 | 0.5236 |
| I_D128_e64_arc_strong_wd1e-2 | 128 | 64 | arcface | strong | all | 1 | 0.5204 | 0.5193 |
| I_D128_e64_arc_strong_wd1e-2 | 128 | 64 | arcface | strong | fit | 3 | 0.5275 | 0.5265 |
| I_D128_e64_arc_strong_wd1e-2 | 128 | 64 | arcface | strong | all | 3 | 0.5319 | 0.5299 |
| I_D128_e64_arc_strong_wd1e-2 | 128 | 64 | arcface | strong | fit | 8 | 0.5244 | 0.5236 |
| I_D128_e64_arc_strong_wd1e-2 | 128 | 64 | arcface | strong | all | 8 | 0.5350 | 0.5332 |

## 对照：未学习的 PCA 输入空间（同一打分/阈值协议）

白化 = 各主成分除以拟合块标准差（网络输入用的就是白化空间）；非白化即基线做法。

**PCA128_whitened_control**（PCA 128，白化=True）

| k | 原型源 | Test AUC(z) | Test AUC(raw) |
|---:|---|---:|---:|
| 1 | fit | 0.6911 | 0.6957 |
| 1 | all | 0.6961 | 0.6955 |
| 3 | fit | 0.7037 | 0.7034 |
| 3 | all | 0.6966 | 0.7034 |

最好工作点：k=3（fit），接受率 0.60，LOSO 0.8538 / S1 0.9654 / S2 0.1975 / 三项均值 0.6722，Test AUC 0.7037。

**PCA128_raw_control**（PCA 128，白化=False）

| k | 原型源 | Test AUC(z) | Test AUC(raw) |
|---:|---|---:|---:|
| 1 | fit | 0.7389 | 0.7493 |
| 1 | all | 0.7509 | 0.7491 |
| 3 | fit | 0.7599 | 0.7606 |
| 3 | all | 0.7524 | 0.7595 |

最好工作点：k=3（fit），接受率 0.80，LOSO 0.9037 / S1 0.9439 / S2 0.3351 / 三项均值 0.7276，Test AUC 0.7599。


## 主配置三项指标随接受率变化

主配置 B_D128_e128_arc_mild，原型 k=8（fit），Test AUC 0.5371。

| 接受率 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 |
|---|---:|---:|---:|---:|
| 0.60 | 1.0000 | 0.9984 | 0.0117 | 0.6700 |
| 0.65 | 1.0000 | 0.9977 | 0.0145 | 0.6707 |
| 0.70 | 1.0000 | 0.9947 | 0.0193 | 0.6713 |
| 0.75 | 1.0000 | 0.9878 | 0.0275 | 0.6718 |
| 0.80 | 1.0000 | 0.9730 | 0.0383 | 0.6704 |
| 0.85 | 1.0000 | 0.9446 | 0.0580 | 0.6675 |
| 0.90 | 1.0000 | 0.8991 | 0.0929 | 0.6640 |
| 0.95 | 1.0000 | 0.8058 | 0.1777 | 0.6612 |

## 最好变体 G 三项指标随接受率变化

G_D128_e128_arc_strong，原型 k=8（all），Test AUC 0.5698。全部 9 组配置 × 3 个 k × 2 个原型源的
完整工作点见 `results.json`。

| 接受率 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 |
|---|---:|---:|---:|---:|
| 0.60 | 1.0000 | 1.0000 | 0.0057 | 0.6686 |
| 0.65 | 1.0000 | 1.0000 | 0.0076 | 0.6692 |
| 0.70 | 1.0000 | 1.0000 | 0.0103 | 0.6701 |
| 0.75 | 1.0000 | 0.9993 | 0.0140 | 0.6711 |
| 0.80 | 1.0000 | 0.9977 | 0.0193 | 0.6723 |
| 0.85 | 1.0000 | 0.9967 | 0.0270 | 0.6746 |
| 0.90 | 1.0000 | 0.9848 | 0.0443 | 0.6764 |
| 0.95 | 1.0000 | 0.9472 | 0.0925 | 0.6799 |

即使把接受率推到 0.95，Test S2 接受也只有 0.0925；此时 S1 拒识 0.9472 —— 两项互斥，三项均值被
S2 拖到 0.68 附近。

## 各变体最好工作点（仅描述，不用于选择）

| 变体 | k | 原型源 | 接受率 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 | Test AUC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A_D96_e128_arc_mild | 8 | fit | 0.75 | 1.0000 | 0.9881 | 0.0352 | 0.6744 | 0.4950 |
| B_D128_e128_arc_mild | 8 | fit | 0.75 | 1.0000 | 0.9878 | 0.0275 | 0.6718 | 0.5371 |
| C_D160_e128_arc_mild | 8 | all | 0.90 | 1.0000 | 0.9769 | 0.0510 | 0.6760 | 0.5163 |
| D_D128_e128_triplet_mild | 8 | fit | 0.65 | 1.0000 | 0.9941 | 0.0191 | 0.6711 | 0.4983 |
| E_D128_e128_arc_triplet_mild | 8 | all | 0.90 | 1.0000 | 0.9723 | 0.0571 | 0.6765 | 0.5392 |
| F_D128_e64_arc_mild | 8 | all | 0.95 | 1.0000 | 0.9261 | 0.1040 | 0.6767 | 0.5468 |
| G_D128_e128_arc_strong | 8 | all | 0.95 | 1.0000 | 0.9472 | 0.0925 | 0.6799 | 0.5698 |
| H_D128_e128_arc_mild_ep20 | 3 | fit | 0.75 | 1.0000 | 0.9848 | 0.0436 | 0.6761 | 0.5189 |
| I_D128_e64_arc_strong_wd1e-2 | 8 | fit | 0.85 | 1.0000 | 0.9601 | 0.0740 | 0.6781 | 0.5244 |

## 为什么失败（机理）

1. **监督信号只有同会话标签。** Train 只有 session1，20 类 ArcFace/triplet 的最优解就是
   "把同被试同会话的 trial 拉到一起"。这正是项目已经证实与 Test 可分性负相关的方向
   （LOSO 0.98 的视图 Test AUC 只有 0.48）：本实验的 LOSO 拒识全部 1.0000，而 Test 端崩掉。
2. **Test 端的可分性来自"非身份"的绝对量线索。** 未学习的 PCA 空间里，未知被试 trial 到
   所有原型的距离整体更远（域/能量偏移），因此 S1 拒识与 S2 接受有一条可用工作曲线（AUC 0.75）。
   嵌入网络输出 L2 归一化、距离被压到 [0,2] 且趋于饱和，这类绝对量线索被抹平，S1 与 S2 的
   分数分布几乎重合（AUC≈0.5），任何阈值都无法同时兼顾。
3. **不是过拟合或容量问题。** 只训练 20 epoch 的弱拟合变体 H 的 AUC 仍只有 0.51~0.53；
   增强加到最强（G，频带缩放 ±30%、丢带 12%、噪声 0.25）最好也只到 0.5698；
   降嵌入维（F/I，64 维）、加强 weight decay（I，1e-2）都没有改变量级。
   而未学习的 PCA 对照在同一打分/阈值协议下稳定在 0.74~0.76（见上）。可见问题出在目标函数方向，
   不是训练不足。
4. **尺度校准不是主因。** raw 分数与 per-task z 分数的 AUC 差距 <0.01，原型取 fit 块还是全部
   Train 也只差 0.01 量级；说明失败发生在可分性本身，而不是阈值落点。

## 结论

- 主配置最好三项均值 **0.6718**（LOSO 1.0000 / S1 0.9878 / S2 0.0275），Test AUC **0.5371**，对应接受率 0.75。
- 接受率 0.80 工作点（LOSO 1.0000 / S1 0.9730 / S2 0.0383 / 均值 0.6704）。
- 9 组超参中三项均值最高的是 G_D128_e128_arc_strong（0.6799，Test AUC 0.5698），仍远低于基线。
- 未学习对照 PCA128_whitened_control：最好 Test AUC 0.7037，三项均值 0.6722。
- 未学习对照 PCA128_raw_control：最好 Test AUC 0.7599，三项均值 0.7276。
- 对照基线 PCA-96 最近原型：Test AUC 0.7566，三项均值 0.7784（LOSO 0.9867 / S1 0.8665 / S2 0.4820）。
- 未学习对照 PCA128_raw_control 的 Test AUC 0.7599，与基线同一水平；进入嵌入网络后 Test AUC 反而掉到 0.49~0.57。

**对"这条路能否到 0.87 AUC"的判断：不能。** 9 组配置覆盖了 PCA 维（96/128/160）、嵌入维（64/128）、
损失（ArcFace / batch-hard triplet / 两者相加）、增强强度（mild/strong）、训练量（20/80 epoch），
Test AUC 全部落在 0.49~0.57，而未学习对照是 0.74~0.76：监督目标本身把与 Test 相关的绝对量线索抹掉了，
弱拟合（H，20 epoch，0.519）与强拟合（B，80 epoch，0.537）都救不回来。要突破必须引入跨会话监督或
跨会话无标签对齐（本项目清洁约束下不可用），度量学习最多只能作为辅助视图，上限仍在 0.76 量级，
无法达到 0.87。

## 产物

- `config.json`：协议与 9 组配置、两个对照、预注册主配置。
- `run_e140.py`：训练 + 打分 + LOSO/Test 评估 + 报告生成，可直接重跑。
- `results.json`：496 行完整工作点（每配置 × k × 原型源 × 接受率），含两个未学习对照。
- `model_*.pt`：9 个训练好的权重，内含 PCA 基底、主成分尺度、per-task 标准化统计，可离线复现分数。
- `run.log`：完整训练/评估日志。

项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。
