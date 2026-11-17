# final2：LOSO 优先的注册、匹配与拒识重构计划

## Context

当前 `final2`（E253）相对 `final` 只改变了身份排序头：使用类内白化余弦排序，但拒识仍使用原始 PCA 空间欧氏距离。因此 Test 已注册身份排序有所提升，而严格 identity-LOSO 的 #5/#6/#11 与 E200 完全相同，分别约为 0.7920、0.1758、0.2947，不能满足本项目要求。

本轮目标明确调整为：**先让严格 nested identity-LOSO 达标，再审计官方 C/U 与初赛 S1/S2；若不能同时达标，不用 Test 审计数字掩盖 LOSO 失败。** 决策原则是把“注册脑纹与身份模板的匹配度”作为核心，把“是否足够匹配、是否存在歧义”作为独立开放集拒识条件，而不是仅寻找最近邻后再用单一欧氏阈值拒识。

决赛说明确认最终决赛不以对抗样本为重点，核心是已有用户跨时段/跨设备识别、新用户注册后识别、未注册检测；模型仍必须遵守 Train-only、官方接口和无 Test 适配约束。

## 推荐实现路线

### 1. 新建独立实验目录，不改旧实验与当前 final2

创建新的 `research/E254_loso_registration/`，所有脚本、配置、折级结果、模型权重、报告和哈希均放入该目录。复用已经验证的 `BrainPrint_cache/e253_subwindow/train_feat.npy` 与现有特征实现，不重复提取或训练已有特征模型。

关键复用文件：

- `research/E253_session_robust/gates11.py`：数据读取、任务解析和指标口径；
- `research/E247_train_temporal_loso/run_e247.py`：LDA/WCCN-cos、时间块和严格身份留出思路；
- `final2/team_hymn/model.py`：部署特征提取、任务解析、接口约束；
- `research/E200_final_freeze/build_bundle.py`：全量 Train 冻结与提交包构建方式。

### 2. 建立严格 nested identity-LOSO 评估器

外层执行 20 个身份留出。每个外层折中，留出身份的任何样本不得参与：任务标准化、PCA、WCCN/LDA、原型、阈值或候选选择；留出身份仅作为 pseudo-unknown 评估。

外层训练集内部再执行身份留出，用于选择候选和校准阈值。查询块与原型样本必须不相交，并加入断言检查：

- held identity 不在任一拟合行；
- 查询样本不在其身份原型拟合行；
- Test 特征/标签不在选择脚本中读取；
- 判决不使用 session 字段。

同时保留 early/middle/late 时间块分析，但时间块只作为 Train-only 稳定性诊断，不能替代外层 identity-LOSO。

### 3. 以匹配度而不是单一最近距离作为核心

固定有限候选，避免无边界调参：

- 特征：E253 已验证的 6×200 子窗 855 维特征；
- PCA 维度：32、48、64、96；
- 匹配空间：原始欧氏、余弦、WCCN-cos、LDA-cos；
- 原型：稳健均值 k=1、有限 k=2 时间块/medoid；
- 分数：第一名匹配度 `s1`、第二名 `s2`、歧义间隔 `s1-s2`。

主路径使用 Train 内拟合的 WCCN/LDA 变换后余弦匹配：

```text
q = W(feature - mean)
prototype_score[c] = max_j cosine(q, W(prototype[c,j] - mean))
identity = argmax_c prototype_score[c]
```

注册原型候选须单独比较：k=1 稳健均值优先；k=2 仅在 nested LOSO 中证明拒识和最差身份没有恶化时保留。最终部署的 1–20 号身份原型使用全量 Train，但评估查询不得进入原型。

### 4. 以绝对匹配度 + 歧义度构成开放集拒识

先计算：

```text
s1 = 最高身份匹配度
s2 = 次高身份匹配度
margin = s1 - s2
```

候选拒识规则固定为：

```text
accept = (s1 >= tau_score) and (margin >= tau_margin)
```

仅当 nested LOSO 证明能够减少误接受且已知接受率损失可接受时，才加入 E253 的欧氏半径作为第三个门：

```text
accept = accept and (d_min <= tau_dist)
```

所有阈值只能由内层 identity-LOSO 的交叉拟合分数确定。不能根据 Test S1/S2 调整。最终部署仍通过官方 `(N, 24)` 分数矩阵表达：身份列为匹配分数，第 0 列只编码已经由上述规则确定的 reject 布尔结果，保持 `argmax` 接口不变。

### 5. 以门禁顺序选择候选

候选选择采用词典序，而非只看 pooled AUC：

1. 严格 nested LOSO unknown rejection 达到目标；
2. strict LOSO AUROC、U@95 达到项目门禁；
3. 最差身份拒识率和折间稳定性；
4. 已知接受率、unknown-positive F1、20 类 macro-F1；
5. Train temporal late；
6. 只有模型和阈值冻结后，才做 Test 官方/初赛审计。

必须同时输出 macro、micro、每个 held identity、最差身份、AUC、F1、known acceptance 和 unknown rejection。若某候选只提高 AUC、但实际 reject 率或最差身份不达标，则判定失败。

### 6. 按官方口径补齐 Train-only 代理

对最终候选使用严格交叉拟合预测计算：

- 已知 Clean Accuracy `C`：拒识和错误身份均算错；
- 未知拒识 `U`；
- 20 类 identity macro-F1；
- unknown-positive rejection F1；
- 若决赛确认无对抗样本，则 #7/#8 标为 N/A，不把旧方案攻击数字移植到新评分规则。

根据官方公式记录可计算部分：

```text
Score = 100 × C^0.40 × (0.7×RA + 0.3×ASafeR)^0.35 × U^0.25
```

若 RA/ASafeR 不适用，明确 N/A，不伪造完整官方分数。

### 7. 全量 Train 冻结并生成真正不同的 final2

仅当候选满足 LOSO 门禁后：

- 用全部 Train 拟合任务统计量、PCA、WCCN/LDA；
- 用全部 Train 建立 1–20 号注册原型；
- 保存 `tau_score`、`tau_margin`、可选 `tau_dist`、原型数量和协议版本；
- 改造 `final2/team_hymn/model.py`，拆分 `_match_scores()`、`_open_set_gate()`、`_predict_labels()`；
- 保留 `enroll()`，对 21–23 使用注册样本建立稳健原型，不更新预训练参数；
- 生成新的 README、team.json、权重 manifest 和压缩包，明确说明其与 E200/E253 的本质差异。

若候选未达 LOSO 门禁，不更新正式 `final2`，只保留 E254 报告和已保存的折级模型，避免交付一个“声称达标但实际未达标”的方案。

## 验证

### 离线验证

- 复现 E200/E253 严格 LOSO 数值，确认新评估器口径一致；
- 运行 E254 nested identity-LOSO，保存每折模型、阈值、预测、AUC/F1 和拒识混淆矩阵；
- 检查查询/原型、held identity、Test 数据隔离断言；
- CPU/CUDA、N=1/7/128、`meta=None`、输入不修改、无 NaN、输出 `(N,24)`；
- 注册 21–23 后验证原有 1–20 参数不变，新原型可用；
- 冻结模型后一次性生成 Test 审计：C/U、已注册接受率、未注册拒识率、S1/S2 及官方限制说明。

### 成功条件

严格 nested identity-LOSO 的 #5/#6/#11 全部达到项目既定门禁，且没有明显单身份崩溃；同时 Train-only C/F1 和新用户注册流程可复现。满足后才替换 `final2`。

### 失败条件

在预先固定的特征、维度、WCCN/LDA、k=1/k=2 原型和 `s1/margin/radius` 小网格内仍不能达标，则停止该方向并报告“当前数据无法证明 LOSO 达标”，不再用 Test S1/S2 反向选参数，也不声称 final2 已完成。
