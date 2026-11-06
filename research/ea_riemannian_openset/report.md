# SPD 对齐与轻量开集识别实验

日期：2026-11-07

## 结论

- **E0 合规检查通过。** 修改全部测试 Session 字段后，冻结 AS-norm 模型的预测与置信度逐值不变；现有及新增回归测试全部通过。正式提交未修改。
- **E1 没有发现稳定优于 target-zscore 的新对齐方法。** Riemannian recentering 在正向、反向时间协议中均优于无对齐，并改善最差折，但两种协议的总体指标与 target-zscore 互有胜负、差异很小。
- **EA 不保留。** 它在两个时间协议的 Fold 4 均明显退化，未知拒识和检测 F1 不稳定。
- 所有 AUROC、F1、C/U 均来自 **Train Session1 内部伪未知验证**，不是真实跨 Session 或测试成绩。本实验未读取 Test，未生成 candidate，不能据此声称已经解决测试阈值迁移或真实未知拒识。

## 1. 实验计划与约束

执行顺序为 E0 合规审计、E1 无监督 SPD 对齐、E2 轻量分类器消融。只有前一阶段在多个外层协议上显示稳定收益，才进入下一阶段。

所有模型只使用 EEG 特征和 task。源域参考只由当前训练折估计；目标域参考由完整无标签评估批次按 task 估计，不使用身份、Session 或已知/未知标签。阈值始终由内层伪未知数据选择，外层只评估。

## 2. 数据与协议

Train 包含 20 名被试、11,304 个 Trial，只有 Session1；任务为 EC、EO、Imagery、SSVEP。由于缺少有标签的真实跨 Session 数据，本实验只能提供时间和留被试代理证据。

- 外层：5 折留被试，每折 4 人全部作为未知，其余 16 人作为注册者。
- 内层：2 折伪未知，只用于选择置信度类型和阈值；阈值为内层已知置信度的 5% 分位。
- 正向时间协议：源前 60% 训练，注册者 60%–80% 内层校准，后 20% 外层测试。
- 反向时间协议：源后 60% 训练，注册者 20%–40% 内层校准，前 20% 外层测试。
- 模型：固定 4 秒 full log-cov + Ridge，避免同时改变表征和分类器。
- 汇总：AUROC 为外折宏平均；C/U/F1 为逐 Trial 汇总，同时报告最差折 detection-macro-F1。

## 3. E0：合规与回归

审计发现，旧实验的 Session 读取位于事后诊断路径。新增测试通过临时缓存实际调用冻结的 AS-norm 预测函数，在第二次调用前改写全部 Session 值，断言预测、置信度和冻结配置完全不变。

E1 的 SPD 对齐接口不接收 subject 或 Session。源参考仅使用 `fit_mask`，目标参考仅使用 EEG 协方差和 task；没有测试候选生成路径。

## 4. E1 方法

从缓存的加权 log-cov 上三角还原 18×18 对称矩阵，经矩阵指数恢复 shrinkage SPD，避免重新提取原始 EEG。每个 task、每个频带分别估计参考：

- `none`：原始 log-cov。
- `target_zscore`：现有逐维目标均值/方差对齐。
- `ea`：源域和目标域分别使用算术协方差参考均值进行白化。
- `riemannian_recenter`：源域和目标域分别使用仿射不变 Riemannian 几何均值进行重居中。

两种 SPD 方法均计算 `R^-1/2 C R^-1/2`，再重新提取 log-cov；源训练特征也经过相同类型的本域变换。特征分解和参考估计使用 CUDA float64。

## 5. E1 正向时间结果

| 配置 | C | U | AUROC | macro-F1 | unknown-F1 | detection-macro-F1 | 最差折 detection-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_none | 0.9361 | 0.9866 | 0.9921 | 0.9567 | 0.9678 | 0.9633 | 0.9590 |
| full_target_zscore | 0.9461 | **0.9920** | **0.9958** | 0.9659 | 0.9745 | 0.9709 | 0.9621 |
| full_ea | 0.9433 | 0.9634 | 0.9847 | 0.9493 | 0.9592 | 0.9544 | 0.8928 |
| full_riemannian_recenter | **0.9497** | 0.9907 | 0.9945 | **0.9671** | **0.9752** | **0.9719** | **0.9655** |

Riemannian 相对无对齐的 AUROC 增加 0.0024、macro-F1 增加 0.0104；相对 target-zscore，C/F1 略高，但 AUROC 下降 0.0012、U 下降 0.0013。

## 6. E1 反向时间结果

| 配置 | C | U | AUROC | macro-F1 | unknown-F1 | detection-macro-F1 | 最差折 detection-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_none | 0.9318 | 0.9904 | 0.9933 | 0.9585 | 0.9688 | 0.9638 | 0.9504 |
| full_target_zscore | **0.9512** | 0.9926 | 0.9957 | **0.9702** | **0.9774** | **0.9739** | 0.9473 |
| full_ea | 0.9587 | 0.9523 | 0.9838 | 0.9512 | 0.9597 | 0.9549 | 0.8896 |
| full_riemannian_recenter | 0.9449 | **0.9949** | **0.9958** | 0.9680 | 0.9760 | 0.9722 | **0.9591** |

Riemannian 在反向协议仍优于无对齐，并显著提高最差折；但 target-zscore 的总体 C、macro-F1、unknown-F1 和 detection-macro-F1 更高。两种协议简单平均后，target-zscore 的 AUROC 为 0.9957、macro-F1 为 0.9680，Riemannian 分别为 0.9952、0.9675，不支持宣称新对齐整体占优。

拒识前 C 在各方法间仍接近饱和：正向为 0.9927–0.9982，反向为 0.9902–0.9991。主要差异来自接受/拒识决策，而不是闭集身份分类大幅提升。

## 7. 当前判断

1. Riemannian 可进入固定表征的轻量分类器消融，理由是跨两种时间方向稳定优于无对齐且最差折较稳，而不是因为已经击败最强基线。
2. EA 在两个协议的 Fold 4 均出现明显开集退化，停止该路线。
3. 现有内层校准已经包含“时间错位已知样本 + 留出伪未知”，并非单纯随机 Trial 阈值；下一阶段的单因素应是分类器，而不是继续手工调整分位数。
4. 即使内部指标提高，也不能外推为真实跨 Session 改进；测试无标签时不使用拒识率或身份均衡反向调参。

## 8. 复现与产物

```powershell
py -m src.ea_riemannian_openset.run --time-protocol forward
py -m src.ea_riemannian_openset.run --time-protocol reverse
py -m pytest -q
```

- 完整结果：`e1_results.json`、`e1_reverse_results.json`
- 日志：`e1_run.log`、`e1_reverse_run.log`
- 实现：`src/ea_riemannian_openset/`
- 回归：`test/test_ea_riemannian_openset.py`、`test/test_session_invariance.py`
