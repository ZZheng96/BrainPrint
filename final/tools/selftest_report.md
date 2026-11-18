# E201 决赛提交包自测

方案：E200 冻结方案 `subwin6_pca96_k1 @ a=0.90`，tau = 19.263826。参数从 `research/E200_final_freeze/bundle/frozen_bundle.pt` 原样复制，未重新拟合。提交包为 `final/team_hymn.zip`（371,060 B，sha256 `a0155efa154e5f4098154368428c2ca5fb89a2cd669cfe37da69569a76a7a1dd`），zip 内仅 model.py / weights.pt / team.json / requirements.txt / README.md，顶层即 `team_hymn/`。

2026-11-18 合规复核仅修正 README 中两处“已注册接受率”的指标名称，`model.py` 与 `weights.pt` 哈希均未改变。已用 `package.py` 重跑隔离 ZIP 的 CPU/CUDA、60 条注册、N=1/128、24 列、有限值和输入不变检查，全部通过；未重跑 Test 或 Train 模拟。

**当前 AUC / F1**：决赛测试集不公开，所以 N/A。初赛对应值见 E200：S1 0.7135，S2 0.7069，身份头 OOF macro-F1 0.9882。

本地补充检查：在 Test 前 128 条上，提供 `meta['task']` 与 `meta=None` 的 argmax 一致率为 0.7656；因此官方若提供任务元数据，应按 README 的 `task` 字段传入。该检查无测试真值，不代表准确率。

## 接口检查（`selftest.py`、`package.py`）

全部通过：
- `weights_only=True` 加载
- CPU 和 cuda:0 下 N = 1/7/128 都输出 (N, 24)，无 NaN，每行有有限值
- 不修改传入的 x
- meta=None 可运行
- 注册前不会输出 21–23
- enroll 只接受标签 21–23，不会覆盖 1–20 的预训练原型
- 从 zip 解压到隔离目录后可以 import，60 条注册样本 enroll 成功
- 官方 infer.py 模拟（`BrainPrint_cache/e201_final_selftest/harness.py`）：cpu / cuda:0 / cuda 三种 device，meta 为 None/dict/空列表/长度不符/中文/陌生键/字符串行，x 为 float64/torch CPU/torch CUDA/N=1/含 NaN/全零，全部输出 (N,24) float32、无 NaN、每行有有限值；确定性重复一致；包内文件零写入；CPU N=128 单批 0.14 s
- `model.py` 以 Python 3.10 语法解析通过（ast feature_version=(3,10)）

## 运行环境与官方一致性

本地 `.venv` 已重建为官方版本组合，全部测试在该环境下重跑：

| 项 | 官方要求 | 本地 |
|---|---|---|
| Python | 3.10.22 | 3.10.22 |
| PyTorch | 2.5.1+cu121 | 2.5.1+cu121 |
| NumPy | 1.26.4 | 1.26.4 |
| SciPy | 1.13.1 | 1.13.1 |
| CUDA runtime | 12.1 | 12.1（RTX 3080 Laptop，驱动 621.14） |

`model.py` 只 import `os`、`numpy`、`torch`，不含 scipy/pandas 等额外依赖；`requirements.txt` 仅作说明，官方不执行安装。旧环境保留在 `.venv-py311`（已 gitignore），确认无用后可删除。

## 与 E200 一致性

Test 前 1024 条的预测标签与 `submission_frozen.csv` 一致率为 1.0。换环境前后 20 次模拟的逐次数值完全相同（acc_known 0.7983 / acc_new 0.8027 / rej_unknown 0.6593），说明 torch 2.5.1+cu121 与 2.7.1+cu118 在本方案上数值一致。

## Train 决赛流程模拟（20 次随机抽取，seed 20261116）

每次抽取：3 名新用户各用 20 条早期试次（frac ≤ 0.6）注册，3 名未注册者删除原型，其余 14 名为已注册用户。测试集为后 20% 试次。

| 注册方式 | 已注册识别 | 新用户识别 | 未注册拒识 |
|---|---|---|---|
| 按任务分别建原型（每任务约 5 条） | 0.801 | 0.694 | 0.673 |
| **全部 20 条合并建原型（采用）** | 0.798 | **0.803** | 0.659 |

结果是同会话的，偏乐观。明细见 `selftest_result.json`。
