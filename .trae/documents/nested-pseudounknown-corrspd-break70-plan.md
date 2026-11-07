# 无 Session 方案继续改进计划

## Context

当前五个无 Session 候选在 Session1 内部 blocked-OOF 已接近饱和，但 Test 拒识率为 79%–95%，说明主要瓶颈是分布迁移和未知身份校准，而不是训练集分类能力。此前 task-specific 阈值和单次稳健 Riemannian 参考没有带来可靠提升；不能把约 69.6 的经验值当作已验证成绩，也不能承诺一定突破 70。

本轮目标是建立更严格、可证伪的改进实验，寻找是否存在支持超过 70 的证据；若没有证据则停止，不通过调阈值或 Test 拒识率反推分数。

## 推荐方案

在新目录 `research/session_blind_candidate/nested_pseudounknown_corrspd_v1/` 实现一个唯一候选与一个原始特征对照：

1. **完整嵌套 blocked + 留身份伪未知校准**
   - 外层使用既有 5 个身份组与 blocked 划分。
   - 内层遍历留出身份组和 blocked 组合，伪未知身份完全不参与拟合、标准化和阈值校准。
   - 只使用 `subject/task`；禁止读取 `session/trial/path/fs` 和 Test 标签。
   - 固定使用 `TaskSpecificModel(alpha=100)`、`fused` 分数和全局 `max_logit`。
   - 阈值由内层独立预测的已知/伪未知样本共同选择，目标为 `C^0.40 * U^0.25`；不再搜索多个阈值体系。

2. **唯一特征改进：逐频带相关矩阵 SPD 变换**
   - 每样本每频带恢复 SPD 协方差 `C`。
   - 使用 `D^-1/2 C D^-1/2` 去除通道方差尺度，再做固定 `lambda=1e-3` 的单位阵收缩。
   - full 和两个窗口使用相同变换；CUDA float64 进行 SPD 运算，输入 Ridge 前转 float32。
   - 不搜索 lambda，不使用 Test 拒识率选择配置。
   - 原始 log-cov 作为同协议对照，防止把内部变化误判为改进。

3. **内部鲁棒性验证**
   - 外层报告 C、U、AUC、macro-F1、unknown-F1、已知接受率、无拒识准确率和最差外层单元。
   - 若实现现有特征空间 PGD，则预注册固定 PGD 配置，报告 RA、ASafeR 及“特征空间 PGD 内部官方公式代理分”；若攻击未能严格匹配则记为 null，不用 C 或 1 补齐。
   - Test 无标签时 C/U/AUC/F1/RA/ASafeR/官方分全部为 null，只报告拒识率、task 分布、标签计数和结构校验。

4. **部署与 CSV**
   - 外层验收后用同一规则在完整 Train 上做身份组×blocked 校准，冻结部署阈值。
   - 全量拟合后才读取 Test 特征，生成唯一 `candidate_corrspd.csv`。
   - 使用模板 Sample 对齐；检查 22,251 行、Sample 唯一且顺序一致、Label 为 0–20。
   - 不覆盖两份正式提交和原五候选。

## 关键文件

- 新增：`research/session_blind_candidate/nested_pseudounknown_corrspd_v1/run.py`
- 新增：`config.json`、`manifest.json`、`splits.npz`、`metrics.json`、`report.md`、部署记录及候选 CSV
- 复用：[session_blind_candidate/run.py](file:///d:/WORK/BCIC2026/BrainPrint/src/session_blind_candidate/run.py) 的白名单加载
- 复用：[core.py](file:///d:/WORK/BCIC2026/BrainPrint/src/open_set_improvement/core.py) 的 Ridge、置信度和开集指标
- 复用：[core.py](file:///d:/WORK/BCIC2026/BrainPrint/src/ea_riemannian_openset/core.py) 的 SPD 恢复/编码
- 复用：[core.py](file:///d:/WORK/BCIC2026/BrainPrint/src/open_set_calibration/core.py) 的提交模板对齐

## 验证与停止条件

- 所有校准样本必须由未见该样本的模型预测；伪未知身份不得进入对应拟合。
- 运行同一配置两次应得到一致结果；正式 CSV SHA-256 保持不变。
- 候选只有在相对原始对照的平均 C 下降不超过 1 个百分点、U/AUC/F1 不下降，且攻击代理分平均差值为正并通过 subject 聚类配对不确定性检查时才标记为推荐。
- 不满足验收则停止本轮，不追加变换、融合或阈值搜索。
- 即使内部代理分超过 70，也只能报告“内部代理分超过 70”，不能声称官方 Test 分超过 70；官方分必须等待真实回执验证。

## 执行命令

```powershell
py -3.11 research/session_blind_candidate/nested_pseudounknown_corrspd_v1/run.py
```
