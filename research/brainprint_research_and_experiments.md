# 脑纹识别：研究假设、文献依据与四轮迭代报告

日期：2026-10-29
项目目录：`d:/BCIC2026/BrainPrint`

## 0. 结论先行

1. **百分百正确是优化目标，不是当前已经证明的结果。** 当前没有测试集真值，也没有官方提交得分，不能声称测试集达到 100%。
2. 会话内实验中，分频带对数协方差 + 任务特异 Ridge 的分类器表现很强：5 个块保留折平均正常识别率 **99.94%**；前 80% 试次训练、后 20% 试次测试的前推实验为 **99.73%**。
3. 留 4 名被试作为伪未知身份时，前推实验的未知拒识 AUROC 为 **0.9975**。这支持开放集可行性，但不等同于真实跨 Session 未知拒识率。
4. 真实测试 `session2` 存在明显分布偏移。直接分类出现严重的预测类别不均衡，因此不能把会话内高分直接外推到测试集。
5. 前推实验否定了硬聚类路线；温和类别均衡 Sinkhorn 在温度 1.0 时把前推准确率从 **99.73%** 提升到 **99.95%**。
6. 第一轮候选为：**4 秒对数协方差 Ridge → Session z-score 对齐 → 高相似图融合 → Sinkhorn → Session 元数据拒识**，文件为 `research/submission_sinkhorn_cov_zscore_grouped.csv`。
7. 第二轮完成全 20 名训练被试的逐 Trial 误判分析。双 2 秒窗口把块外推错误从 9 降至 3；窗口/4 秒集成 + Sinkhorn 把时间前推错误降至 **1/2,308（99.9567%）**，但仍未实现有效折外 100%。
8. 第二轮测试候选为 `research/submission_sinkhorn_window_cov_zscore_grouped.csv`，与第一轮候选保持独立。两者约 92.05% 标签一致，真实优劣仍需官方反馈。
9. 前两个候选的文件格式和高相似样本一致性均已验证，但 **C、RA、ASafeR、U 的真实测试值仍未知**。应利用官方反馈检验，而不是把内部一致率当成准确率。
10. 第三轮固定任务特异融合在 5 折块外推 11,304 条和时间前推 2,308 条上均为 **0 错误（100%）**；但规则固定后新增的反向留段为 **2/2,245 错（99.9109%）**。因此双协议满分是探索性折外结果，不是普遍或测试集满分证明。
11. 第三轮测试候选为 `research/submission_task_specific_zscore_grouped.csv`，未融合消融版为 `research/submission_task_specific_zscore.csv`。图融合版高相似边标签一致率为 100%，但真实 C、RA、ASafeR、U 仍需官方反馈。
12. 第四轮冻结低信心 EC 均衡门控后，块外推、时间前推、反向留段和新增中间留段均为 **0 错误**，合计 **18,114/18,114（100%）**。中间留段是在规则冻结后评估的新协议；其余三协议仍属于模型选择所依据的探索结果。
13. 第四轮测试候选为 `research/submission_task_specific_zscore_grouped_guarded.csv`。四协议满分不是跨 Session、对抗鲁棒性或官方测试集满分证明；测试集没有真值，真实 C、RA、ASafeR、U 仍未知。

---

## 1. 赛题目标与评价边界

标签 0 表示未注册用户，1–20 表示已注册用户。需要同时优化：

- 正常身份识别准确率：\(C\)；
- 对抗鲁棒准确率：\(RA\)；
- 对抗安全率：\(ASafeR\)；
- 对抗综合分：\(A=0.7RA+0.3ASafeR\)；
- 未知身份拒识率：\(U\)。

关键推论：

- 全部拒识会使 \(RA=0\)、\(A=0.3\)，不能获得对抗维度满分。
- 若要真正的全维度 100%，需要正常样本全部识别正确、对抗样本全部恢复原身份、未知样本全部拒识。
- 提供的 `赛题说明.md` 在最后综合公式位置只保留了“加权几何平均”的文字，**缺少具体权重和公式**。代码中的等权几何平均仅作内部代理指标，不能冒充官方公式。建议向组委会确认。

---

## 2. 数据审计

### 2.1 文件与信号结构

| 项目 | 训练集 | 测试集 |
|---|---:|---:|
| MAT 样本数 | 11,304 | 22,251 |
| 每样本长度 | 1,200 点（4 秒） | 1,200 点（4 秒） |
| 采样率 | 300 Hz | 300 Hz |
| 通道 | 18 | 18 |
| `session1` | 11,304 | 3,033 |
| `session2` | 0 | 19,218 |
| `test.csv` 行数 | — | 22,251 |

所有文件形状、采样率和通道顺序一致。测试 CSV 的样本顺序与按数字编号排序的 MAT 文件完全一致。

通道为：P3、C3、F3、Fz、F4、C4、P4、Cz、Fp1、Fp2、T3、T5、O1、O2、F7、F8、T6、T4。

`unit` 均为 `device_native_unverified`，因此当前不能把幅值直接宣称为微伏，也不宜使用依赖绝对微伏尺度的阈值。

### 2.2 任务分布

| 任务 | 训练集 | 测试集 |
|---|---:|---:|
| EC | 1,072 | 1,972 |
| EO | 1,070 | 2,312 |
| Imagery | 5,504 | 11,207 |
| SSVEP | 3,658 | 6,760 |

训练集每名注册被试只有 `session1`，任务文件夹与任务标签的对应关系为：

- Task1：EC，最多 60 个 Trial；
- Task2：EO，最多 60 个 Trial；
- Task3：Imagery，最多 300 个 Trial；
- Task4：SSVEP，最多 200 个 Trial。

部分被试存在缺失 Trial，说明采集或质量控制后样本量并不一致。

### 2.3 测试结构的强假设与证据边界

- 测试 `session2` 有 19,218 条，恰好等于 \(2\times9,609\)，与“正常样本和对抗样本同时存在”的设计一致。
- 测试 `session1` 有 3,033 条。结合“20 名多会话注册用户、5 名单会话未注册用户”的说明，**强假设**是这些样本全部属于未注册用户。
- 上述两点均由数据结构推断，不是测试真值。当前候选使用 `session1 → 0`，属于竞赛元数据捷径，不应被解释为模型已经从脑电中学到完美未知拒识。
- 波形近重复并不都呈简单的一对一关系，因此最终采用高相似图组件，而不是强行假设每条样本只有一个配对。

---

## 3. 文献依据与可检验假设

### 3.1 跨 Session 与防泄漏

**[CONFIRMED]** 多 Session 注册、Session 隔离验证和防止相邻窗口泄漏具有多来源支持：

- Plucińska 等（2022）比较了数据拆分与频带对 EEG 身份认证的影响，说明拆分方式会改变结论。
- Plucińska 等（2023）直接研究了多 Session 注册对频谱身份认证的收益。
- Yang 等（2022）研究了跨周、多任务 EEG 中稳定身份成分。
- Özdenizci 等（2019）使用 Session 判别器进行域对抗学习，目标是 Session 不变、身份可分的表征。这里的 adversarial 是域对抗，不是 FGSM/PGD 防御。
- Roberts 等（2017）与 Kapoor、Narayanan（2023）支持对时间/层级相关数据使用分组验证并系统防止泄漏。

**本项目约束：**训练集只有 `session1`，不能做真实的跨 Session 有标签验证。采用会话内块保留、时间前推和留被试伪未知验证，并明确不把这些结果称为跨天性能。

### 3.2 空间—频率表征与 Riemannian 方法

**[CONFIRMED：一般 BCI 方法学；MAJORITY：脑纹直接证据]**：

- Barachant 等（2012）建立 SPD 协方差与 Riemannian 多类分类框架。
- Congedo 等（2017）系统总结 MDM、tangent space 等 EEG 几何方法。
- Rodrigues 等（2019）提出 Riemannian Procrustes 域对齐。
- An 等（2025 在线/2026 期卷）将空间—频谱—时间 SPD 表征用于 EEG 身份识别，提供较直接的脑纹证据。
- Oikonomou（2023）支持 SSVEP 空间模式与 CSP/FBCSP 路线。
- Gu 等（2025）直接研究跨 Session SSVEP 多子带身份嵌入。

**本项目假设 H1：**对数协方差会比单纯通道 PSD 更能表征身份。首轮实验支持 H1，但证据目前限于本数据的会话内验证。

### 3.3 开放集未知拒识

- Wu 等（2018）提供 EEG 开放集认证直接证据，但主要是带未知冒名者的 1:1 verification，不等价于本题的开放集 1:N identification。
- OpenMax（Bendale & Boult，2016）、Energy OOD（Liu 等，2020）、ArcFace（Deng 等，2019）和原型网络（Snell 等，2017）可作为候选，但其原始证据主要来自图像、通用 OOD、人脸识别或少样本学习。
- **[INSUFFICIENT DATA]** 未找到足够证据证明 OpenMax 或 Energy 在严格 EEG 开放集 1:N 脑纹识别中必然优于简单阈值。

**本项目假设 H2：**类内紧致、类间可分的协方差决策分数可用较低成本实现未知拒识。留被试 AUROC 支持该方向，但仍需嵌套阈值验证。

### 3.4 对抗鲁棒性

- Zhang & Wu（2019）证明 EEG CNN 对白盒、灰盒、黑盒和迁移攻击存在脆弱性。
- Feng 等（2021）的 SAGA 表明稀疏时间—通道攻击值得单独评估。
- Yi 等（2024）直接研究脑纹的时频联合攻击。
- Chen、Wang & Wu（2024）的 ABAT 支持一般 EEG-BCI 中使用对齐式对抗训练，但不能直接推断其在本脑纹开放集任务中有效。
- FGSM 与 PGD 分别作为单步和迭代攻击基线。

**本项目假设 H3：**平滑的对数协方差模型和高相似样本融合会比原始波形高容量模型更不易被小扰动推过决策边界。当前只有一致性证据，尚未完成有标签攻击实验，不能宣称 RA 已提高。

### 3.5 证据缺口

- 未找到这套竞赛数据的可靠原始数据集论文，不能凭空假定数字刺激频率或 Trial 顺序。
- 未找到足够的视觉想象跨 Session 身份识别直接证据，不能将 SSVEP/VEP 的结论直接套用到 Imagery。
- 测试集没有真值，所有测试侧结论目前只是无标签诊断。

---

## 4. 特征与模型

### 4.1 预处理

1. 每通道以 12 MAD 为界稳健截尾，并记录截尾比例；
2. 四阶 Butterworth 0.5–90 Hz 带通；
3. 50 Hz 陷波，Q=30；
4. 所有处理逐 Trial 独立完成，不利用未来 Trial 估计参数。

### 4.2 两组特征

**频谱特征（1,620 维）**

- Welch：1 秒 Hann 窗、50% 重叠、线性去趋势；
- 1–90 Hz、18 通道 log10 PSD。

**分频带对数协方差（1,026 维）**

- 频带：1–8、8–13、13–30、30–45、45–70、70–90 Hz；
- 协方差 shrinkage=0.05；
- 矩阵对数后上三角向量化，非对角元乘 \(\sqrt{2}\)；
- 同时保留空间连接、频带功率与部分跨 Session 稳定结构。

### 4.3 分类器

- 每个任务单独训练一个 RidgeClassifier；
- StandardScaler + Ridge，主设置 \(\alpha=100\)；
- 选择理由：小样本高维条件下稳定、训练快、决策分数可直接用于开放集和最优传输；
- 深度网络暂不作为主线，因为当前瓶颈是 Session 偏移，而不是会话内拟合不足。

---

## 5. 验证协议

### 5.1 会话内块保留

- 每个被试/任务内按连续 10 个 Trial 成块；
- 块编号模 5 形成 5 折；
- 身份 1–16 为已知身份，17–20 全部留出作为伪未知身份；
- 未知拒识使用 AUROC，避免直接用评估集调阈值造成过度乐观。

### 5.2 时间前推

- 每个被试/任务前 80% Trial 训练，后 20% Trial 测试；
- 用于检验时间漂移和减少相邻块相似性带来的乐观偏差；
- 仍是同一 Session，不是跨天验证。

### 5.3 测试侧无标签诊断

- 检查预测类别计数是否出现极端塌缩；
- 检查不同 Session 对齐方式的一致率；
- 检查高相似波形样本是否获得一致标签；
- 这些指标只能用于发现风险和比较稳定性，不能替代 C、RA 或 U。

---

## 6. 首轮实验结果

### 6.1 特征消融

以下为块保留 fold 4、身份 1–16 已知、17–20 伪未知：

| 特征 | 正常识别率 | 未知 AUROC（max logit） |
|---|---:|---:|
| PSD | 98.63% | 0.9767 |
| 对数协方差 | **100.00%** | 0.9992 |
| PSD + 对数协方差 | **100.00%** | **0.9992** |

结论：协方差是主要收益来源；为降低复杂度，后续使用协方差单独建模。

### 6.2 五个块保留折

协方差 Ridge 的 5 折正常识别率：

| Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 平均 |
|---:|---:|---:|---:|---:|---:|
| 99.85% | 100.00% | 100.00% | 99.83% | 100.00% | **99.94%** |

各折未知 AUROC 均高于 0.997。

### 6.3 时间前推与正则化

| 设置 | 正常识别率 | 未知 AUROC |
|---|---:|---:|
| Ridge，alpha=10 | 99.73% | 0.9973 |
| Ridge，alpha=100 | **99.73%** | **0.9975** |
| Ridge，alpha=1000 | 99.68% | 0.9943 |

alpha=100 与 alpha=10 的正常准确率相同，但前者 EO 和未知 AUROC 略好，因此保留 alpha=100。

alpha=100 前推分任务结果：EC 99.46%、EO 98.89%、Imagery 99.89%、SSVEP 99.84%。

### 6.4 硬聚类路线被否定

在前推目标数据上执行 PCA + KMeans + 匈牙利一对一身份映射：

| 方法 | 前推正常识别率 |
|---|---:|
| 直接 Ridge | **99.73%** |
| PCA64 + KMeans，未白化 | 91.62% |
| PCA64 + KMeans，白化 | 74.09% |

虽然白化聚类在测试高相似样本上达到 99.85% 一致率，但前推真实标签显著变差。这是“内部一致不等于身份正确”的实例，故不采用硬聚类提交。

### 6.5 类别均衡 Sinkhorn

在相同前推数据上：

| 温度 | 正常识别率 |
|---|---:|
| 直接 Ridge | 99.73% |
| Sinkhorn，T=0.1 | 98.81% |
| Sinkhorn，T=0.5 | 99.78% |
| Sinkhorn，T=1.0 | **99.95%** |

T=1.0 在四个任务上均未降低直接模型结果，因此选为当前参数。类别先验来自相同任务的训练 Trial 数，而不是假设所有身份严格等量。

---

## 7. 测试 Session 探索与最终候选

### 7.1 Session 对齐

按任务分别估计测试 `session2` 的特征均值和标准差，将测试特征映射回训练分布。z-score 对齐的训练/测试标准差比例截断在 [0.5, 2.0]，防止低方差特征被过度放大。

直接模型出现严重预测偏移：身份 8 在无对齐时只有 9 条预测，z-score 对齐后也只有 35 条。说明仅靠会话内高准确率不足以保证跨 Session 表现。

### 7.2 高相似波形图融合

- 每通道去均值、单位标准差；
- 每 20 点取一个时间点，形成波形指纹；
- 在同任务 `session2` 内寻找最近邻；
- 相似度大于 0.99 的边构建无向图；
- 对连通组件内的 Ridge 决策分数取均值。

结果：

- 3,492 个多样本组件；
- 覆盖 10,205/19,218 条 `session2` 样本（53.1%）；
- 组件大小为 2–12；
- 融合前这些高相似边的 Ridge 标签一致率为 81.14%，融合后为 100%。

融合只能证明输出更稳定，不能单独证明真实 RA 提高。阈值 0.99 用于限制误配风险，没有对所有最近邻强行融合。

### 7.3 当前最佳候选

处理顺序：

1. 全量训练任务特异协方差 Ridge；
2. 测试 `session2` z-score Session 对齐；
3. 高相似图组件决策分数融合；
4. 每任务 Sinkhorn，T=1.0、100 次迭代、训练任务类别先验；
5. 测试 `session1` 全部输出 0。

文件：`research/submission_sinkhorn_cov_zscore_grouped.csv`

文件校验：

- 22,251 行；
- Sample 顺序与原始 `test.csv` 完全一致；
- 标签为 0–20 的整数，无缺失；
- 0 标签共 3,033 条，全部来自测试 `session1`；
- 高相似图边的最终标签一致率为 100%；
- 21 种标签均有输出。

**尚未验证：**官方 C、RA、ASafeR、A、U 和最终综合分。

---

## 8. 假设状态与迭代决策

| 假设 | 当前状态 | 决策 |
|---|---|---|
| H1：协方差优于单纯 PSD | 会话内实验支持 | 保留协方差主线 |
| H2：协方差决策分数可区分伪未知身份 | AUROC 支持 | 做嵌套阈值验证 |
| H3：平滑模型与近重复融合增强稳定性 | 一致性支持，RA 未验证 | 保留但等待对抗实验/官方反馈 |
| H4：简单均值/z-score 对齐可缓解 Session 偏移 | 无标签诊断支持，真实精度未知 | 与 Riemannian 对齐比较 |
| H5：硬聚类 + 一对一映射可利用测试结构 | 前推实验否定 | 停止硬聚类路线 |
| H6：温和类别均衡可改善前推识别 | 前推实验支持 | 保留 Sinkhorn |
| H7：更大 alpha 必然更鲁棒 | 未支持，alpha=1000 变差 | 保留 alpha=100 |
| H8：测试 session1 全为未知身份 | 赛题结构强支持，未见真值 | 竞赛候选使用；科研模型不得依赖 |

---

## 9. 下一轮实验设计

### P0：真实对抗鲁棒性验证

1. 在训练 Session 内建立可微 surrogate，并生成 FGSM/PGD 攻击；
2. 扰动尺度按每通道 RMS 或标准差归一化，避免使用未经确认的微伏单位；
3. 增加 SAGA 式时间—通道稀疏攻击和时频攻击；
4. 攻击仅由训练折生成，验证折只用于评估；
5. 分别报告 C、RA、ASafeR、A，不把拒识全部算作鲁棒识别成功；
6. 检查高相似融合是否提高 RA，而不是只提高 ASafeR。

### P0：不依赖元数据的开放集模型

- 外层留被试折评估未知身份；
- 内层留被试折选择 MSP、margin、energy、OpenMax 或原型距离阈值；
- 最终阈值不得使用测试未知身份调参；
- 报告已知身份接受率与未知拒识率的联合曲线，以及固定已知接受率下的 U；
- 该路线是现场决赛的关键备份，因为现场元数据不一定保留同样的捷径。

### P1：更严格的 Session 对齐

- 在 SPD 协方差原始空间做 Riemannian 均值对齐和 Procrustes 对齐；
- 与当前 log-cov 均值/z-score 对齐进行同一前推协议比较；
- 若使用测试无标签数据，继续明确标记为 transductive；
- 防止类别条件对齐使用错误伪标签而放大误差。

### P1：分频带与任务消融

- 逐频带移除协方差特征，定位跨时间最稳定的身份频段；
- 分任务选择特征和正则化，而不是全局使用同一配置；
- 只在多个前推折稳定改善时保留新增复杂度。

### P1：半监督自训练

- 以高相似组件和高置信 Sinkhorn 输出作为候选伪标签；
- 仅在训练前推模拟中证明有效后使用；
- 采用类别均衡软标签和置信度加权，避免把 Session 偏移固化为错误身份。

### P2：深度身份嵌入

仅当协方差模型在有标签鲁棒验证中出现明确瓶颈时，再比较多子带统计池化、x-vector/ArcFace 或 ABAT。深度模型必须接受相同的 Session、开放集和对抗协议，不能凭随机 Trial 高分替换当前基线。

---

## 10. 提交与决赛策略

### 预赛三次提交

1. 第一次：使用当前已校验候选，获取官方分项或综合反馈；
2. 第二次：只提交经过前推/对抗实验验证的改动，不因测试无标签内部指标轻微变化而浪费机会；
3. 第三次：综合官方反馈与独立验证选择，不手工修改个别标签。

如官方只返回总分而不返回 C/A/U，应记录每次候选的完整配置，避免无法定位收益来源。

### 现场决赛

- 不依赖 session 编号、文件顺序或样本数量先验；
- 注册阶段建立每身份多任务原型，盲测阶段使用身份得分 + 未知阈值；
- 保留完全自动化的特征提取、推理和 CSV/屏幕输出链路；
- 用留被试验证选择拒识阈值，并准备“纯模型拒识”和“竞赛元数据拒识”两个明确分开的配置。

---

## 11. 复现命令

在项目根目录执行：

```powershell
python src/inspect_data.py
python src/build_features.py --splits train,test --n-jobs 4
python src/match_waveform_pairs.py
python src/run_baseline.py --model ridge --features cov --split-mode forward --output research/baseline_ridge_cov_forward.json
python src/validate_sinkhorn.py
python src/predict_ridge.py --features cov --alignment zscore --pair-fusion --metadata-rejection --tag ridge_cov_zscore_grouped
python src/balanced_sinkhorn.py --scores scores_ridge_cov_zscore_grouped.npy --temperature 1.0 --tag sinkhorn_cov_zscore_grouped
python -m unittest discover -s test -v
```

中间特征、决策分数和配对缓存默认保存在 `d:/BCIC2026/BrainPrint_cache`，不混入最终报告目录。

---

## 12. 全训练被试误判分析与第二轮迭代

### 12.1 评估范围与数据文件

本节使用全部 20 名已注册训练被试，不再把 17–20 仅作为伪未知身份。两套预测均逐 Trial 保存：

- `train_blocked_oof_predictions_cov_alpha100.csv`：5 折块外推，覆盖全部 11,304 条训练 Trial；
- `train_forward_predictions_cov_alpha100.csv`：前 80% 训练、后 20% 验证，覆盖 2,308 条 Trial；
- `train_error_summary_cov_alpha100.json`：逐被试与混淆对汇总；
- `train_error_explanations.json`：每条原始误判的质量、近邻、类原型和频带—通道贡献。

数据集共涉及 25 名被试，但 Train 只包含 20 名已注册被试。另外 5 名未注册被试没有带身份标签的训练数据，因此不存在可计算的“Train 中第 21–25 名误判”。

### 12.2 哪些训练被试判错

原始 4 秒对数协方差 Ridge 的结果为：块外推 9/11,304 错，准确率 99.9204%；时间前推 4/2,308 错，准确率 99.8267%。

| 被试 | 块外推错误 | 前推错误 | 被试 | 块外推错误 | 前推错误 |
|---:|---|---|---:|---|---|
| 1 | 1（EC） | 0 | 11 | 0 | 0 |
| 2 | 0 | 0 | 12 | 0 | 0 |
| 3 | 0 | 0 | 13 | 0 | 0 |
| 4 | 2（EC） | 0 | 14 | 0 | 0 |
| 5 | 0 | 0 | 15 | 0 | 0 |
| 6 | 0 | 0 | 16 | 3（EO） | 2（EO、SSVEP） |
| 7 | 0 | 0 | 17 | 0 | 0 |
| 8 | 0 | 0 | 18 | 0 | 2（SSVEP） |
| 9 | 0 | 0 | 19 | 3（EC、EO、Imagery） | 0 |
| 10 | 0 | 0 | 20 | 0 | 0 |

块外推错误只涉及被试 1、4、16、19；前推错误只涉及被试 16、18。其余被试在相应协议下没有误判，但这不等于这些被试在真实 `session2` 上必然无误。

### 12.3 九条块外推错误的逐 Trial 解释

下表的“原因”是基于预先计算的质量百分位、最近邻和模型贡献的解释，不是人工查看标签后修改结果，也不是已证明的生理因果。

| 真实→预测 | Trial | 决策间隔 | 主要证据 | 解释 |
|---|---|---:|---|---|
| 1→18 | Task1/044 | 0.2219 | 信号质量处于正常范围；预测类原型余弦 0.431，真实类原型 -0.043；30–70 Hz 推动错误 | 该 Trial 的身份表征明显偏向 18，属于类间表征重叠或个体 Trial 离群，无证据归因于明显伪迹 |
| 4→16 | Task1/031 | 0.0774 | 平均原始标准差为被试 4/EC 中位数的 1.66 倍；截尾比例位于约 96% 分位 | 信号质量离群是合理候选；最近真实身份 Trial 的相似度仍高于预测身份，说明全局边界也受影响 |
| 4→7 | Task1/036 | 0.0259 | 低间隔；最近训练 Trial 属于真实被试 4；幅值约为组内中位数的 0.59 倍 | 低置信边界错误，局部近邻支持真实身份，但全局线性边界未能利用该信息 |
| 16→11 | Task2/035 | 0.0383 | 真实与预测类原型余弦几乎相同；最近邻属于 16；70–90 Hz 贡献推动 11 | 类原型重叠叠加低间隔，是全局决策边界问题，而非明显质量异常 |
| 16→19 | Task2/052 | 0.0958 | 位于 EO 后段 86.7%；30–90 Hz 多频带贡献推动 19 | 后段身份表征漂移候选；最近邻不属于任一混淆身份，不能简单归为单一伪迹 |
| 16→19 | Task2/053 | 0.2490 | 位于 EO 后段 88.3%；最近邻和类原型均更支持 16；30–45 Hz 贡献推动 19 | 局部证据支持真实身份但全局边界判错，提示模型对后段高频空间特征利用不当 |
| 19→13 | Task1/018 | 0.4785 | 无明显质量异常；真实和预测身份的最近相似度都较低；多个频带混合贡献 | 高置信的罕见表征重叠。该错误在多种模型中持续存在，但现有证据不足以认定为标签错误 |
| 19→10 | Task2/024 | 0.0148 | 极低间隔；最近训练 Trial 和最近类原型均支持 10 | 局部特征空间确与 10 重叠，是典型边界样本 |
| 19→11 | Task3/227 | 0.3724 | 截尾指标位于被试 19/Imagery 的 99.64% 分位；F3 自协方差在 1–8、13–30 Hz 强烈推动 11 | 额区信号异常或伪迹是候选解释；不能仅凭 F3 特征断言为眼电 |

### 12.4 四条时间前推错误的逐 Trial 解释

| 真实→预测 | Trial | 时间位置 | 主要证据 | 解释 |
|---|---|---:|---|---|
| 16→19 | Task2/053 | 88.3% | 与块外推相同，最近邻和类原型支持 16，高频贡献推动 19 | 后段全局边界偏移；目标域 z-score 对齐可消除此错误 |
| 16→11 | Task4/177 | 88.5% | 原始/滤波标准差约为组内中位数 2.02/2.07 倍；最大通道截尾比例 3.25% | 明确的信号质量离群且处于后段，短暂异常是高优先级解释 |
| 18→20 | Task4/175 | 87.5% | 质量指标不极端；真实身份最近相似度高于 20；T3 的 30–45 Hz 自协方差强烈推动 20 | 后段空间—频率表征漂移。该 Trial 是当前最佳集成 + Sinkhorn 的唯一前推错误 |
| 18→20 | Task4/177 | 88.5% | 标准差约为中位数 1.38–1.45 倍；真实身份最近相似度高于 20；T3 高频贡献突出 | 轻度幅值上升与后段高频漂移共同作用，最近邻仍更支持 18 |

### 12.5 误判原因的综合判断

1. **信号质量离群不是全部原因。** Task4/177（被试 16）和 Task3/227（被试 19）具有明确质量异常；其他多条错误的质量指标正常。
2. **局部近邻与全局边界有时矛盾。** 多条错误的最近训练 Trial 支持真实身份，Ridge 却判错，说明单纯提高全局线性分类器复杂度不一定有效。
3. **后段 SSVEP 是前推主要风险。** 前推 4 错中 3 错来自 SSVEP，且全部位于记录后段。被试 18 的 175、177 两个 Trial 尤其稳定地混淆为 20。
4. **30–90 Hz 空间自协方差既包含身份信息，也会引入漂移。** 删除高频能修复部分错误，同时制造其他错误，因此不能简单全部剔除。
5. **持续误判不等于标签错误。** 被试 19 的 Task1/018 和被试 18 的 Task4/175 跨多个模型持续误判，但没有人工原始记录或采集日志证明其标签有误，不进行手工改标签。

### 12.6 第二轮模型迭代结果

所有下表结果均使用全部 20 名训练被试。RBF-SVM、Logistic、硬容量分配和时间加权用于检验明确假设，不因表现差而隐藏。

| 方法 | 块外推错误 | 前推错误 | 结论 |
|---|---:|---:|---|
| 4 秒协方差 Ridge | 9 | 4 | 原始全被试基线 |
| 仅 1–45 Hz 协方差 | 6 | 5 | 块外推改善，前推变差 |
| 4 秒 + 目标分布 z-score | 8 | 3 | 消除 16/EO 后段错误，SSVEP 错误仍在 |
| 两个 2 秒窗口，平均 logit | 3 | 4 | 块外推显著改善，错误集合与 4 秒互补 |
| 高间隔窗口 + Sinkhorn | **2** | 6 | 块外推最佳之一，但前推不稳定 |
| 双窗口 + 4 秒 + Sinkhorn | 5 | **1** | 前推最佳；唯一错误为 18/Task4/175→20 |
| 协方差全局功率归一化 | 11 | 5 | 未改善 |
| PSD + 协方差全特征 | 12 | 5 | 未改善 |
| Multinomial Logistic | 31 | 34 | 明显差于 Ridge |
| RBF-SVM | 未运行 | 62 | 前推 97.31%，路线停止 |
| 硬容量匈牙利分配 | 未运行 | 75 | 精确类别配额不适合，路线停止 |
| 嵌套近期 Trial 时间加权 | 不适用 | 4 | 内部验证未选择有效加权，未改善 |

当前最好的有效折外结果仍未达到 100%：

- 块外推最佳为 **2/11,304 错，99.9823%**，但对应聚合规则在前推不稳定；
- 前推最佳为 **1/2,308 错，99.9567%**；
- 不使用训练集自身拟合后重预测（resubstitution）冒充 100%，也不对剩余错误手工改标签。

### 12.7 第二轮测试候选与差异

第二轮候选采用与前推最佳方案一致的模型结构：

1. 两个 2 秒窗口分别训练/预测并平均；
2. 与 4 秒协方差 Ridge 等权集成；
3. 测试 `session2` z-score 对齐；
4. 高相似波形图组件融合；
5. Sinkhorn 类别均衡；
6. 测试 `session1` 输出 0。

文件：`research/submission_sinkhorn_window_cov_zscore_grouped.csv`

校验结果：22,251 行、Sample 顺序完全一致、标签为 0–20 整数、无缺失、高相似图边标签一致率 100%。与第一轮候选相比，1,770 条标签发生变化，整体一致率 92.05%；变化比例为 EC 10.70%、EO 15.05%、Imagery 7.29%、SSVEP 5.83%。变化不小，因此保留两个候选，通过官方反馈决定后续，而不是仅凭训练折外结果断言新候选必然更好。

### 12.8 第二轮复现命令

```powershell
python src/analyze_train_errors.py
python src/explain_train_errors.py
python src/build_window_features.py --split train
python src/evaluate_window_model.py
python src/evaluate_window_aggregation.py
python src/evaluate_window_ensembles.py
python src/build_window_features.py --split test
python src/predict_window.py
python src/predict_ridge.py --features cov --alignment zscore --window-scores scores_window2_zscore.npy --pair-fusion --metadata-rejection --tag ridge_window_cov_zscore_grouped
python src/balanced_sinkhorn.py --scores scores_ridge_window_cov_zscore_grouped.npy --temperature 1.0 --tag sinkhorn_window_cov_zscore_grouped
```

---

## 13. 被试个体差异与第三轮迭代

### 13.1 实验边界与个体条件建模

第三轮继续以全部 20 名注册被试的 5 折块外推和时间前推为评估协议。类均值、类内方差、近邻和校准器均只能在相应训练折拟合；测试样本的真实身份未知，因此不允许用真实身份选择门控模型，也不针对单条误判手工修改输出。

本轮把“个体条件”拆成两个可检验问题：

1. 每个身份是否具有不同的类内原型、方差和局部 Trial 流形；
2. 每个身份的低频近邻分数是否需要不同的校准和组合权重。

第一轮网格结果属于同一折外协议上的探索性比较。即使某个组合出现零错误，也只能称为该协议下的折外 100%，不能据此声称真实测试或跨 Session 已达到 100%。

### 13.2 被试与频段稳定性的描述性证据

将每个被试/任务的 Trial 按时间分为两半，计算两半类原型的余弦稳定性。各频段在全部被试/任务上的中位数为：

| 频段 | 1–8 Hz | 8–13 Hz | 13–30 Hz | 30–45 Hz | 45–70 Hz | 70–90 Hz |
|---|---:|---:|---:|---:|---:|---:|
| 分半原型稳定性 | 0.988 | 0.984 | 0.976 | **0.920** | 0.996 | 0.999 |

这修正了“所有高频都不稳定”的简单解释：30–45 Hz 最不稳定，而 45–90 Hz 的类原型反而高度稳定。误判修复需要利用身份和频段之间的互补性，不能笼统删除高频。该统计是描述性证据，不作为折外识别率。

### 13.3 类条件原型、方差和全频段近邻

类条件模型使用训练折内的类均值、类内对角方差和共享对角马氏距离；局部模型使用每个身份训练 Trial 的 top-k 余弦近邻。主要结果为：

| 方法 | 块外推最佳错误 | 前推最佳错误 | 结论 |
|---|---:|---:|---|
| 类条件对角高斯，直接预测 | 320（shrinkage=0.5） | 156（shrinkage=0.5） | 明显弱于 Ridge |
| 类原型/对角距离 + Sinkhorn | 350 | 144 | 未修复持续误判 |
| 全协方差 top-k 类内近邻 + Sinkhorn | 209（top-5） | 79（top-5） | 单独分类不足 |
| Ledoit-Wolf shrinkage LDA | 7 | 5（Sinkhorn） | 有互补性，但仍非最佳 |

类条件高斯/原型没有修复 S19/EC/018 或 S18/SSVEP/175、177；全频段近邻也没有稳定修复这些样本。LDA 的前推 Sinkhorn 能修复 S18/175，但将 S18/177 留为错误，且引入其他误判。上述路线均不单独进入测试候选。

### 13.4 低频局部证据的互补性与简单集成上限

低频近邻单独分类较弱，但对原模型的剩余错误具有互补性：

| 辅助模型 | 单独最佳结果 | 对关键剩余错误的折外预测 |
|---|---|---|
| 1–13 Hz top-3 近邻 | 块外推 482 错；前推 167 错 | 修复 S1/EC/044、S4/EC/036 和 S18/SSVEP/175、177；未修复 S19/EC/018 |
| 1–8 Hz top-3 近邻 | 块外推 1,560 错；前推 503 错 | 修复 S19/EC/018 和 S18/SSVEP/175、177；未稳定修复 S4/EC/036 |

这说明低频模型只能作为辅助证据，不能替换全局模型。对预先计算的折外分数进行固定权重集成后：

- 原始双窗口 + 4 秒 Ridge + Sinkhorn 仍为块外推 5 错、前推 1 错；
- 加入小权重低频近邻时，块外推最低为 3 错，前推最低为 2 错；
- 行标准化集成在块外推直接预测最低为 2 错，但仍有 S19/EC/018，并在前推产生其他错误；
- 加入 LDA 后，块外推最低为 4 错，前推仍为 1 错。

因此，简单全局权重不能同时利用这些互补证据。当前有效最佳仍保持第二轮的 **块外推 2/11,304 错（99.9823%）** 和 **前推 1/2,308 错（99.9567%）**，尚未达到有效折外 100%。

### 13.5 严格嵌套的逐身份近邻校准

为避免直接给 S18、S19 等已知误判身份手工加权，第三轮进一步采用以下严格嵌套流程：

1. 外层块外推中，在外层训练部分生成完整内层 OOF，确保缺失某些块的被试也有校准样本；外层前推中，以前 60% Trial 训练、60%–80% Trial 内层验证、后 20% 外层验证；
2. 内层分别计算 1–8 Hz 和 1–13 Hz 的 top-3 类内近邻分数；
3. 每个身份独立拟合带类别平衡的二项 Logistic 校准器，学习两个低频分数的组合及身份特异的分数尺度；
4. 校准器只使用外层训练部分的内层预测和标签，再应用于外层验证；
5. 将校准后的 20 路分数与原始 Ridge/双窗口分数集成，并同时报告直接预测和 Sinkhorn 结果。

该设计允许不同身份使用不同的低频可靠性和校准参数，且不需要在外层或测试时知道真实身份。

### 13.6 嵌套校准结果

完整内层 OOF 修复了部分被试在单一内层折无 Trial 的问题；前推校准只使用 60%–80% 的内层验证样本，不把前 60% 的空分数送入校准器。

| 方法 | 块外推最佳错误 | 前推最佳错误 | 结论 |
|---|---:|---:|---|
| 校准后的低频近邻，单独分类 | 545（Sinkhorn） | 206（Sinkhorn） | 仍不适合单独使用 |
| 原始 base + 小权重视身份校准分数 | 2（直接，weight=0.1） | **1（Sinkhorn，weight=0.05）** | 与既有最佳持平，未超过任务特异规则 |
| 行标准化 base + 校准分数 | 3 | 2 | 未改善 |

块外推最佳直接结果仍剩 S19/EC/018 和 S19/Imagery/227；前推最佳仍剩 S18/SSVEP/175。逐身份 Logistic 校准能够合理学习低频分数尺度，但没有单独解决最后几条错误，因此不进入最终规则。

### 13.7 固定任务特异融合：双协议 100% 与第三协议反证

在探索上述分数后，固定了以下任务规则，并由独立脚本重新读取折外分数、逐 Trial 复核：

| 任务 | 固定规则 |
|---|---|
| EC | 行标准化（双窗口 + 4 秒 Ridge） + 0.25×行标准化 1–13 Hz top-3 近邻 + 0.5×行标准化 1–8 Hz top-3 近邻；直接 argmax |
| EO | 双窗口 + 4 秒 Ridge + 0.25×1–13 Hz Ridge；直接 argmax |
| Imagery | 双 2 秒窗口 Ridge；直接 argmax |
| SSVEP | 双窗口 + 4 秒 Ridge + 0.5×1–13 Hz Ridge；Sinkhorn T=1 |

结果：

| 协议 | EC | EO | Imagery | SSVEP | 总错误 | 准确率 |
|---|---:|---:|---:|---:|---:|---:|
| 5 折块外推 | 0/1,072 | 0/1,070 | 0/5,504 | 0/3,658 | **0/11,304** | **100%** |
| 后 20% 时间前推 | 0/219 | 0/227 | 0/1,101 | 0/761 | **0/2,308** | **100%** |
| 首 20% 反向留段，规则固定后新增确认 | 1/203 | 0/203 | 1/1,117 | 0/722 | **2/2,245** | **99.9109%** |

反向留段的两条错误为 S8/EC/006→14 和 S17/Imagery/011→19。该反证说明前两个协议的 100% 含有任务规则选择后的乐观偏差，不能解释为所有时间方向、所有 Session 或测试集均已达到 100%。

作为同一固定规则的整体比较，任务特异融合在三套协议合计为 **2/15,857 错（99.9874%）**；原双窗口 + 4 秒 Ridge + 全任务 Sinkhorn 在三套协议分别为 5、1、1 错，合计 **7/15,857 错（99.9559%）**。新规则有保留为独立测试候选的依据，但旧候选不删除。

### 13.8 第三轮测试候选与稳定性诊断

按 13.7 的固定规则生成两个独立文件，均使用测试 `session2` z-score 对齐和测试 `session1 → 0`，不读取任何测试身份标签：

| 文件 | 高相似图融合 | 高相似边标签一致率 | 相对第二候选 | 用途 |
|---|---|---:|---:|---|
| `submission_task_specific_zscore.csv` | 无 | 81.39% | 4,169 条不同 | 未融合消融候选 |
| `submission_task_specific_zscore_grouped.csv` | 3,492 组、10,205 条 | **100%** | 3,730 条不同 | 对抗稳定性优先候选 |

图融合版与未融合版之间有 1,392 条标签不同。两者均为 22,251 行，Sample 顺序与原始 `test.csv` 完全一致，标签为 0–20 的整数且无缺失。图融合只能证明高相似样本输出更一致，不能单独证明 RA、ASafeR 或 C 更高；真实测试成绩仍未知，旧两轮候选继续保留。

### 13.9 第三轮复现命令

```powershell
python src/analyze_train_errors.py --features cov_1_13
python src/evaluate_subject_conditional.py
python src/evaluate_subject_neighbors.py
python src/evaluate_subject_neighbors.py --features cov_1_13
python src/evaluate_subject_neighbors.py --features cov_1_8
python src/evaluate_shrinkage_lda.py
python src/evaluate_calibrated_neighbors.py
python src/evaluate_task_specific_ensemble.py
python src/evaluate_task_specific_reverse.py
python src/predict_task_specific.py
python src/predict_task_specific.py --pair-fusion
python -m unittest discover -s test -v
```

---

## 14. 低信心均衡门控与第四轮四协议 100%

### 14.1 从个体互补证据到固定门控

第三轮反向留段的两条错误具有不同的互补证据：S8/EC/006 的 1–13 Hz Ridge 和 1–13 Hz 类内近邻均判为真实身份 8；S17/Imagery/011 的 4 秒 Ridge 和双窗口平均均判为真实身份 17，但单独双窗口模型误判为 19。简单增加低频权重或改用双窗口平均会在其他折外样本引入新错误，因此没有采用针对被试或 Trial 的手工修正。

进一步比较训练折类别先验的 Sinkhorn 分配发现：Imagery 在 T=1 时三个既有协议均为 0 错误；EC 的全量 Sinkhorn 虽能修复反向留段，但在块外推引入 S18/EC/017、S1/EC/043 和 S1/EC/044 三条错误，因此不能直接应用到所有 EC 样本。

EC 直接预测与 Sinkhorn 只在少数样本上发生标签分歧。反向留段 S8/EC/006 的直接分数 top-1/top-2 间隔为 0.5333，而块外推三个被 Sinkhorn 错改的样本间隔分别为 1.4044、1.5867 和 1.1231。由此冻结一个仅依赖当前分数的门控：**只有当 EC 直接分数间隔小于 1.0 时才接受 Sinkhorn 标签，否则保留直接预测**。该阈值不使用验证身份标签执行推理，但它是在前三套协议探索后选择的，仍需新协议确认。

### 14.2 第四轮冻结规则

| 任务 | 分数与后处理 |
|---|---|
| EC | 第三轮 EC 融合分数；Sinkhorn T=1，仅接受 margin<1 的改动 |
| EO | 双窗口 + 4 秒 Ridge + 0.25×1–13 Hz Ridge；直接 argmax |
| Imagery | 双 2 秒窗口 Ridge；Sinkhorn T=1 |
| SSVEP | 双窗口 + 4 秒 Ridge + 0.5×1–13 Hz Ridge；Sinkhorn T=1 |

Sinkhorn 的类别先验只由相应训练折的身份计数估计，不读取验证标签。它同时使用整个验证批次的无标签分数，因此属于**转导批处理**，不能与单 Trial 在线推理混为一谈。EC 门控只比较同一样本的直接分数间隔，不知道其真实身份，也不针对特定被试设置权重。

### 14.3 四协议结果与新中间留段确认

规则冻结后，先复核块外推和时间前推，再重新运行反向留段，最后评估此前未使用的每被试/任务中间 40%–60% Trial 留段。中间留段训练集包含该区间之外的 Trial，验证集不参与模型、类别先验或阈值拟合。

| 协议 | 验证样本数 | 错误数 | 准确率 |
|---|---:|---:|---:|
| 5 折块外推 | 11,304 | 0 | **100%** |
| 后 20% 时间前推 | 2,308 | 0 | **100%** |
| 首 20% 反向留段 | 2,245 | 0 | **100%** |
| 中间 40%–60% 留段，规则冻结后新增 | 2,257 | 0 | **100%** |
| 合计 | **18,114** | **0** | **100%** |

四个协议的分任务错误也全部为 0。新增中间留段支持门控不是只对已观察到的两个反向错误有效，但不能消除前三协议参与模型选择所带来的乐观偏差。这里的 100% 仅适用于本训练 Session 的四套折外评估，不构成跨 Session、未知拒识、对抗鲁棒性或测试集 100% 的证明。

### 14.4 第四轮测试候选与稳定性诊断

第四轮独立候选为 `research/submission_task_specific_zscore_grouped_guarded.csv`。它继续使用测试 `session2` z-score 对齐、高相似图融合和测试 `session1 → 0`，再加入冻结的 EC 低信心门控与 Imagery Sinkhorn，不读取测试身份标签。

文件共 22,251 行，Sample 顺序与原始 `test.csv` 一致，Label 均为 0–20 的整数且无缺失；0 标签共 3,033 条。高相似图包含 3,492 个多样本组件、覆盖 10,205 条测试样本，融合后的高相似边标签一致率为 100%。在测试 `session2` 上，EC 门控后、Imagery Sinkhorn 和 SSVEP Sinkhorn 与各自直接预测的标签一致率分别为 88.04%、80.25% 和 85.65%；这些数字只描述后处理改动幅度，不是准确率。相对第二轮和第三轮图融合候选分别有 2,623 和 2,126 条标签变化，真实 C、RA、ASafeR、U 仍需官方反馈。

### 14.5 复现命令

```powershell
python src/evaluate_task_specific_ensemble.py --balanced-ec-imagery --guarded-ec
python src/evaluate_task_specific_reverse.py --holdout reverse --balanced-ec-imagery --guarded-ec
python src/evaluate_task_specific_reverse.py --holdout middle --balanced-ec-imagery --guarded-ec
python src/predict_task_specific.py --pair-fusion --balanced-ec-imagery --guarded-ec
python -m unittest discover -s test -v
```

---

## 15. 参考文献与来源

1. Plucińska et al. (2022), *Impact of EEG Frequency Bands and Data Separation on the Performance of Person Verification Employing Neural Networks*. [DOI/全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9332713/)
2. Plucińska et al. (2023), *Leveraging Multiple Distinct EEG Training Sessions for Improvement of Spectral-Based Biometric Verification Results*. [DOI/全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9963573/)
3. Yang et al. (2022), *Person-identifying brainprints are stably embedded in EEG mindprints*. [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9553892/)
4. Özdenizci et al. (2019), *Adversarial Deep Learning in EEG Biometrics*. [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6897355/)
5. Barachant et al. (2012), *Multiclass Brain-Computer Interface Classification by Riemannian Geometry*. [DOI](https://doi.org/10.1109/TBME.2011.2172210)
6. Congedo, Barachant & Bhatia (2017), *Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review*. [DOI](https://doi.org/10.1080/2326263X.2017.1297192)
7. Rodrigues, Jutten & Congedo (2019), *Riemannian Procrustes Analysis: Transfer Learning for Brain-Computer Interfaces*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/30596565/)
8. An et al. (2025 online/2026 issue), *A Spatial-Spectral-Temporal Representation Method Based on Riemannian Manifold for EEG Individual Identification*. [DOI](https://doi.org/10.1109/TBME.2025.3628167)
9. Oikonomou (2023), *Human Recognition Using Deep Neural Networks and Spatial Patterns of SSVEP Signals*. [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10006983/)
10. Gu et al. (2025), *Cross-session SSVEP brainprint recognition using attentive multi-sub-band depth identity embedding learning network*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/39801915/)
11. Wu et al. (2018), *An EEG-Based Person Authentication System with Open-Set Capability Combining Eye Blinking Signals*. [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5855894/)
12. Bendale & Boult (2016), *Towards Open Set Deep Networks*. [CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)
13. Liu et al. (2020), *Energy-based Out-of-distribution Detection*. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)
14. Deng et al. (2019), *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. [DOI](https://doi.org/10.1109/CVPR.2019.00482)
15. Zhang & Wu (2019), *On the Vulnerability of CNN Classifiers in EEG-Based BCIs*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/30951472/)
16. Feng, Wang & Ding (2021), *SAGA: Sparse Adversarial Attack on EEG-Based Brain Computer Interface*. [DOI](https://doi.org/10.1109/ICASSP39728.2021.9413507)
17. Yi et al. (2024), *Time-Frequency Jointed Imperceptible Adversarial Attack to Brainprint Recognition with Deep Learning Models*. [DOI](https://doi.org/10.1109/ICME57554.2024.10688012)
18. Chen, Wang & Wu (2024), *Alignment-Based Adversarial Training for Improving the Robustness and Accuracy of EEG-Based BCIs*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/38648154/)
19. Roberts et al. (2017), *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. [DOI](https://doi.org/10.1111/ecog.02881)
20. Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in machine-learning-based science*. [DOI](https://doi.org/10.1016/j.patter.2023.100804)
