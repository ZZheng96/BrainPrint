# E247 Train-only 时间块严格 LOSO 稳定性实验计划

## Context

E243–E246 已覆盖特征融合、欧氏距离、LDA 白化余弦、归一化、批次中心化和阈值扫描。LDA-cos 在同一 session 的 Train LOSO 很强，但不能解释 Test session2 的接受率；Test 没有真值，且项目规则禁止使用 Test 统计量选模型或阈值。因此下一步只验证一个未覆盖的假设：Train session1 内按 trial 顺序划分的时间漂移代理，能否筛掉只在早期窗口有效、后期稳定性变差的评分方案。

目标仍是严格 LOSO 未注册拒识约 0.90，同时尽可能提高后期已知用户接受；不修改正式提交包，不使用 Test 进行选择。

## 推荐实施

### 1. 独立实验目录

新增 `research/E247_train_temporal_loso/`，不修改 E246、`final/team_hymn/model.py` 或原始缓存。入口为 `run_e247.py`，产物包括逐折结果、汇总结果、配置/切分清单、每折模型参数、模型文件和 `REPORT.md`。

### 2. 时间块与数据边界

- 仅使用 Train session1。
- 对每个 `subject × task` 按 `trial` 顺序划分：Early 前 60%、Middle 60%–80%、Late 后 20%。按完整 trial 切分，避免同一 trial 跨块。
- 由于 Train 只有一个 session，报告称为“trial 顺序时间块稳定性代理”，不宣称已证明跨 session 泛化。
- 所有表示、标准化、PCA、原型、白化和阈值均只使用 Train 拟合段；Middle/Late 只评估。
- 禁止 batch centering、Test 统计、session 参与判决、伪标签或基于评估窗口调参。

### 3. 三个预先固定候选

1. 当前冻结欧氏 + 原提交 tau：历史参照，不作为严格 LOSO 结论。
2. 欧氏 + inner-LOSO 阈值：严格主候选。
3. LDA-cos（shrink=0.01）+ inner-LOSO 阈值：严格主候选。

不再搜索 shrinkage、中心化、评分函数或阈值网格，避免把同一 Train 数据用于反复选型。

### 4. 嵌套 subject-LOSO 协议

- 外层 20 折：每次完整留出一个 subject 作为未知身份，其余 19 人为注册者。
- 严格候选的标准化、PCA、原型、协方差/白化只用 19 名注册者的 Early 拟合段。
- 在外层注册者 Early 内再按时间切分拟合段/校准段：前 2/3 Early 用于重拟合，后 1/3 Early 只用于 inner 阈值校准。
- inner 对注册者逐身份留出；留出身份不得参与标准化、PCA、原型、白化或阈值拟合。用伪未知校准分数固定 FAR 目标，保守报告 FAR=1%、5%、10% 的诊断阈值；主阈值采用预先固定的 FAR≤10% 规则。
- 外层用完整 Early 重拟合后，冻结阈值评估 Middle 和 Late。不得用 Middle/Late 调整阈值。
- 欧氏和 LDA-cos 使用相同的数据切分与阈值规则；关闭 batch128。

### 5. 指标与判定

Middle、Late 分别按外层 subject 宏平均报告：

- AUC；
- unknown-positive 拒识 F1（明确类别比例）；
- known acceptance；
- unknown rejection；
- TPR@FAR 1%、5%、10%，同时记录外层实际 FAR；
- Middle→Late 的 AUC、known acceptance、unknown rejection 跌落；
- 逐折均值、标准差、最差折和配对差异。

只有严格候选在 Late 仍接近目标拒识、已知接受不明显恶化且时间跌落不加剧，才进入后续考虑；只提升 pooled AUC 或靠拒绝全部样本达标均不采用。Test 指标在报告中写 `N/A`，如做 Test 只作事后描述并明确不参与选择。

### 6. GPU、保存与验证

- 用 CUDA 执行投影、PCA/SVD、协方差分解、距离和余弦评分；元数据汇总在 CPU。
- 保存每个外层折的标准化/PCA/原型/白化/阈值模型，避免重复计算。
- 运行前断言特征与 metadata 行数、trial 顺序和任务映射一致；断言外层留出身份不进入任何拟合集合；断言评估集合不参与阈值计算。
- 运行入口：
  - `python research/E247_train_temporal_loso/run_e247.py --validate-only`
  - CUDA 单折 smoke
  - CUDA 全部外层折并支持 `--resume`
- 完成后写 `REPORT.md`，包含 AUC 和 F1，并在确认结果后再决定是否提交；不替换正式提交模型。

## 关键复用文件

- `research/E246_submission_ldacos/run_e246.py`：`Space`、`build`、`whiten`、评分实现参考；需重新实现时间边界，不能直接复用其使用 late 估阈值的主流程。
- `final/team_hymn/model.py`：提交特征和 artifact 加载参考，不修改。
- `BrainPrint_cache/train_metadata.csv`：身份、task、trial 和顺序切分依据。
- `BrainPrint_cache/e201_final_selftest/train_raw_scores_inputs.npy`：Train 855 维特征缓存。
- `赛题说明.md`：确认 session 结构、标签和官方指标语义。

## 验证命令

先执行 `--validate-only` 检查数据对齐、时间块互斥、外层/内层拟合集合无泄漏；再执行 CUDA 单折 smoke，最后执行全 20 折并生成报告。实验结果仅在 Train 侧解释，不能把单 session 时间块代理等同于真实跨 session 验证。
