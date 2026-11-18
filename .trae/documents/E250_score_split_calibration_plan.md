# E250 评分与拒识分离实验计划

## Context

E249 的 C2 漂移白化让 Train 内 P1/P2 排序和严格 LOSO 略有提升，但真实 Test S1/S2 未改善，对抗攻击下错误身份接受率明显升高。问题已从“特征表达不足”转为“身份评分与拒识评分共用同一个 max-cos 分数”。本实验在不改变正式提交包的前提下，验证 Train-only 的分离式拒识评分能否改善综合 11 项指标。

## 推荐方案

新建独立实验目录 `research/E250_score_split_calibration`，实现 E250-S1：

- 身份判决仍由 top-1 cosine 原型分数决定，不让拒识校准改变身份排序。
- 拒识分数使用 top-1 分数、top-1/top-2 margin、注册原型 cohort 统计和 hubness 统计。
- 在严格嵌套 LOSO 内，仅使用内层拟合集生成伪未知/已知样本，完成 median/MAD 标准化、带 L2 正则的 logistic 校准和 FAR10 阈值选择。
- 先实现并比较三个固定候选：E249 C0 max-cos 基线、top1+margin、top1+margin+cohort/hubness 校准；不使用 Test 结果选参。
- 与 E191、E249 C2 的同一工作点结果对照，单独标明 Test 为看过 Test 后的一次性描述性审计。

## 关键文件与复用

- 新增：`research/E250_score_split_calibration/run_e250.py`、`config.json`、结果 CSV/JSON、报告。
- 复用 [E249 run_e249.py](../../research/E249_drift_nap/run_e249.py) 的 `outer_plan`、`fit_all`、`transforms`、`all_task_scores`、`score_table`，保留 C0/C2 的特征空间和原型拟合。
- 复用 [E247 run_e247.py](../../research/E247_train_temporal_loso/run_e247.py) 的 `Space`、`fit_model`、`weighted_far_threshold`、严格 LOSO 组织方式。
- 复用 [E195 run_e195.py](../../research/E195_eleven_gates_loso90/run_e195.py) 的 11 项门禁口径和 PGD 结果汇总逻辑；必要时只对最终候选补测 #7/#8。

## 折内执行流程

1. 外层留出一个完整身份；外层训练只使用其余身份。
2. 外层训练集内部再做身份留出，内层拟合变换、原型、cohort/hubness 特征、MAD、logistic 和 FAR10 阈值。
3. 用内层留出身份产生伪未知样本，完成候选选择；不读取 Test。
4. 以选定候选在外层已知身份上重新拟合全部 Train-only 参数，仅评估外层 Late/块外数据。
5. 保存每折模型、阈值、特征统计、结果和可复现配置；检查模型 reload 前后分数一致。
6. 全部外层完成后，使用预先固定的最终候选做一次 Test 审计；Session 只用于 S1/S2 分组，不参与判决或选参。

## 验证与运行

```powershell
& .\.venv\Scripts\python.exe research\E250_score_split_calibration\run_e250.py --validate-only
& .\.venv\Scripts\python.exe research\E250_score_split_calibration\run_e250.py --outer-subject 1
& .\.venv\Scripts\python.exe research\E250_score_split_calibration\run_e250.py --all-folds --resume
```

验证内容：C0 与 E249/E247 结果逐数复现；所有变换和校准参数均来自折内 Train；无 Test 统计量、Session 特征或伪标签泄漏；报告 11 项指标、最差身份、身份级 bootstrap CI；对最终候选补测同一 PGD 协议下的 #7/#8，并区分 SNR30 主口径与 SNR20 参考口径。

## 风险控制

margin 可能降低跨会话接受；cohort/hubness 可能继续放大原型 hub。若校准不能同时改善未知拒识和已知接受，应保留 E191 为正式部署候选，不把 E250 的 Train 代理收益外推为 Test S2 改善。