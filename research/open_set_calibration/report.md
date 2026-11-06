# 开集跨分布校准：无监督目标域对齐与分数归一化

日期：2026-11-06

数据：Train Session1，20 名已注册被试，11,304 个 4 秒 Trial；测试集 22,251 条且无标签

硬件：NVIDIA GeForce RTX 3080 Laptop GPU，PyTorch 2.7.1+cu118

## 1. 结论先行

- **无监督目标域特征 z-score 在内部开集验证中有效。** `base_target_zscore` 的外层折宏平均 AUROC 为 **0.9960**，逐 Trial 汇总 macro-F1 为 **0.9653**、unknown-F1 为 **0.9746**，C=0.9424，U=0.9951；相对无对齐，AUROC、unknown-F1 和 detection-macro-F1 均在 **5/5 折**改善，C 在 **4/5 折**改善。
- **full 与 base 各有优势。** `full_target_zscore` 的 C=0.9461、macro-F1=0.9659，略高于 base；base 的 U、AUROC 和 detection-macro-F1 更高。因此后续保留两个简单分支，不增加复杂融合。
- **逐 Trial 类别分数 row-zscore 失败。** 它虽然把 C 提高到 0.9952/0.9982，却把 U 降至 0.6706/0.2812，unknown-F1 降至 0.8014/0.4389。高 C 是以大量未知误接受为代价，不能视为改进；不生成该阶段候选，也不再沿此方向扩展。
- **测试候选仍不可提交。** `base_target_zscore` 候选格式正确，但拒识 20,974/22,251 条（**94.2609%**），比旧无对齐候选的 81.77% 更严重。说明内部判别改善并未解决源域阈值到真实测试分布的迁移；正式提交未修改。
- 所有 AUC/F1 都是 **Train Session1 内部伪未知验证结果**，不是跨 Session 或真实测试成绩。真实 5 名未注册者的测试标签不可用，因此不能声称未知拒识问题已经在测试集解决，也不能据此承诺晋级分数。

## 2. 数据与实验问题

赛题包含 25 名被试：20 名多会话被试为已注册用户，另外 5 名单会话被试为未注册用户；任务包括静息态 EC/EO、SSVEP 和视觉想象。标签 0 表示未注册，1–20 表示已注册身份。测试还包含保留原身份的对抗扰动样本，因此过度拒识会同时损害正常识别与对抗鲁棒识别。

本实验只使用 Train Session1 的 20 名已注册被试。通过留被试构造“伪未知”，检验两个单因素：

1. **目标域特征对齐**：能否只用无标签目标 Trial 的特征统计，减轻分布偏移并改善开集校准。
2. **类别分数 row-zscore**：在特征对齐基础上，能否通过逐 Trial 分数标准化进一步稳定拒识阈值。

不使用 `Session1 → 0`、测试身份标签、测试 Session 划分或其他元数据拒识规则。

## 3. 方法

### 3.1 基线与无监督目标域 z-score

- `full`：4 秒对数协方差特征的 Ridge 分类器。
- `base`：4 秒与双 2 秒窗口 Ridge 分数等权平均。
- `none`：直接使用目标特征，复现上一阶段的 full/base 分支。
- `target_zscore`：按 task 分别对整个无标签目标评估批次估计每个特征的均值和标准差，再映射到源模型训练特征的均值和标准差：

\[
x_{aligned}=(x-\mu_{target})\cdot \mathrm{clip}(\sigma_{source}/\sigma_{target},0.5,2)+\mu_{source}.
\]

该操作只依赖目标特征，不接收身份、Session 或已知/未知标签。目标统计包含当前评估批次的注册者与伪未知 Trial，与真实测试时对整批无标签 Trial 推理的设置一致。它是批次级无监督对齐，不是逐 Trial 独立推理。

### 3.2 逐 Trial 类别分数 row-zscore

第二阶段只在 `target_zscore` 上追加单因素：对每条 Trial 的所有注册类别分数做均值/标准差标准化，再计算置信度。源域 blocked-OOF 分数采用相同变换后估计阈值，避免只变换目标分数造成额外口径不一致。

### 3.3 无泄漏嵌套评估

- 外层：5 折留被试，每折 4 人全部 Trial 作为未知，其余 16 人为注册者；注册者每 task 后 20% Trial 仅用于外层评估。
- 内层：在 16 名注册者中做 2 折伪未知划分；模型用源前 60% Trial 训练，内层注册者的 60%–80% Trial 与伪未知被试前 80% Trial 用于选择置信度分数和阈值。
- 阈值：内层已知置信度的 5% 分位，目标已知接受率约 95%。外层标签不参与模型、分数、对齐统计或阈值选择。
- 汇总：C/U/F1 按全部外层 Trial 汇总；AUROC/AUPR/EER 为 5 个外层折宏平均。每阶段累计评估 9,232 条已知 Trial 和 11,304 条伪未知 Trial。
- 不确定性：200 次按被试聚类 bootstrap。报告 C/U/unknown-F1 的 95% 区间；AUROC 只报告外层折范围，不把 pooled bootstrap AUROC 当作折宏 AUROC 的严格置信区间。

指标中，C 为注册者拒识后的正确身份识别率，U 为未知拒识率；AUROC 以“已知”为正类；unknown-F1 以未知拒识为正类；detection-macro-F1 为已知/未知二分类宏平均。

## 4. 第一阶段：目标域特征对齐结果

| 配置 | 拒识前 C | C | 已知接受率 | U | AUROC（折宏） | macro-F1 | unknown-F1 | detection-macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_none | 0.9982 | 0.9361 | 0.9361 | 0.9866 | 0.9921 | 0.9567 | 0.9678 | 0.9633 |
| full_target_zscore | 0.9977 | **0.9461** | 0.9461 | 0.9920 | 0.9958 | **0.9659** | 0.9745 | 0.9709 |
| base_none | **0.9988** | 0.9321 | 0.9321 | 0.9901 | 0.9935 | 0.9556 | 0.9680 | 0.9634 |
| base_target_zscore | 0.9985 | 0.9424 | 0.9425 | **0.9951** | **0.9960** | 0.9653 | **0.9746** | **0.9710** |

相对各自无对齐分支：

| 配置 | ΔC | ΔU | ΔAUROC | Δmacro-F1 | Δunknown-F1 | Δdetection-macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| full：target − none | +0.0100 | +0.0055 | +0.0036 | +0.0092 | +0.0066 | +0.0076 |
| base：target − none | +0.0103 | +0.0050 | +0.0025 | +0.0097 | +0.0066 | +0.0076 |

AUROC 的 5 折范围：full_none 0.9897–0.9942，full_target_zscore 0.9920–0.9978；base_none 0.9914–0.9953，base_target_zscore 0.9949–0.9982。

### 被试 bootstrap 95% 区间

| 配置 | C | U | unknown-F1 |
|---|---|---|---|
| full_none | 0.8985–0.9709 | 0.9773–0.9950 | 0.9543–0.9826 |
| full_target_zscore | 0.9167–0.9765 | 0.9871–0.9963 | 0.9629–0.9870 |
| base_none | 0.8893–0.9711 | 0.9826–0.9973 | 0.9524–0.9841 |
| base_target_zscore | 0.9095–0.9735 | 0.9919–0.9977 | 0.9622–0.9871 |

**解释：** 两个模型上的改善方向一致，支持目标特征分布偏移确实是内部验证中的一个误差来源；但增益幅度有限，且被试区间仍重叠，不应把结果解释为已解决真实跨 Session 偏移。

## 5. 第二阶段：row-zscore 单因素结果

| 配置 | C | 已知接受率 | U | AUROC（折宏） | macro-F1 | unknown-F1 | detection-macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_target_zscore_row_zscore | 0.9952 | 0.9965 | 0.6706 | 0.9936 | 0.8419 | 0.8014 | 0.8160 |
| base_target_zscore_row_zscore | **0.9982** | **0.9996** | 0.2812 | 0.9955 | 0.7046 | 0.4389 | 0.5666 |

| 配置 | C 的 95% 区间 | U 的 95% 区间 | unknown-F1 的 95% 区间 | AUROC 5 折范围 |
|---|---|---|---|---|
| full_target_zscore_row_zscore | 0.9899–0.9995 | 0.6269–0.7145 | 0.7696–0.8324 | 0.9923–0.9944 |
| base_target_zscore_row_zscore | 0.9959–1.0000 | 0.1298–0.4237 | 0.2296–0.5951 | 0.9947–0.9964 |

与只做 target_zscore 相比，full 的 U 下降 0.3215、unknown-F1 下降 0.1730；base 的 U 下降 0.7139、unknown-F1 下降 0.5357。AUROC 仍较高而固定阈值下 U 崩溃，说明**分数排序能力不等于阈值可迁移性**。row-zscore 改变了已知/未知分数的校准关系，使大量未知 Trial 越过源域阈值，因此否决该方案。

## 6. 测试候选诊断与提交决定

候选配置只依据内层结果确定：`base_target_zscore` 的 5 折内层 detection-macro-F1 均值为 0.9470，高于 full_target_zscore 的 0.9335；最终置信度为 margin（3/5 外折对应的内层选择）。源域 blocked-OOF 已知 margin 的 5% 分位阈值仍为 0.7826477893，目标特征统计只使用各 task 的全部无标签测试 Trial，不使用 Session 拆分。

| 诊断项 | 结果 |
|---|---:|
| 候选行数 | 22,251 |
| 拒识行数 | 20,974 |
| 拒识比例 | **94.2609%** |
| 旧无对齐候选拒识比例 | 81.77% |
| margin 分位数 5% / 25% / 50% / 75% / 95% | 0.0164 / 0.0927 / 0.2319 / 0.4536 / 0.8067 |

候选文件已通过格式检查：行数与 Sample 顺序和 `test.csv` 一致，标签均为 0–20 整数。但格式合格不代表性能合格。绝大多数测试置信度低于源域阈值，且拒识比例进一步升高，故该文件仅保留为失败诊断，**不提交、不覆盖正式答案**。测试无标签，以上比例不是测试 U，也不能计算测试 C、AUC 或 F1。

## 7. 局限与下一步

1. 内部未知者来自同一 Session1 的留出被试，不包含真实测试中的新被试、跨 Session 变化和对抗扰动；内部高 AUC/F1 不能外推为测试成绩。
2. target_zscore 依赖整批无标签目标特征和 task 路由，不是单样本在线算法；若目标批次组成或 task 信息改变，统计量和预测也会改变。
3. 当前失败核心已从“闭集身份不可分”转为“真实跨分布置信度阈值失效”。继续微调同一源域 5% 分位阈值缺少依据，不能用无标签测试拒识比例反向调参。
4. 下一步应优先获取或构造独立、有标签且更接近目标分布的校准/验证数据；若仍只有当前训练集，应研究不依赖固定源域分数尺度的逐 Trial 开集方法，并用嵌套外层验证和独立候选诊断共同筛选，而不是继续堆叠分数归一化。
5. 目前不建议替换正式提交，也不能据内部结果承诺达到特定排行榜或晋级分数。

## 8. 复现与产物

在仓库根目录运行（需要 CUDA GPU）：

```powershell
py -m src.open_set_calibration.run_calibration --output research/open_set_calibration/calibration_results.json --generate-candidate base_target_zscore
py -m src.open_set_calibration.run_calibration --alignments target_zscore --score-normalization row_zscore --output research/open_set_calibration/row_score_results.json
py -m pytest -q
```

- 第一阶段完整结果：`calibration_results.json`
- 第二阶段完整结果：`row_score_results.json`
- 失败诊断候选：`submission_base_target_zscore.csv`（不可提交）
- 实现：`src/open_set_calibration/core.py`、`src/open_set_calibration/run_calibration.py`
- 回归测试：`test/test_open_set_calibration.py`
