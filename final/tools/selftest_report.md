# E201 决赛提交包自测

方案：E200 冻结方案 `subwin6_pca96_k1 @ a=0.90`，tau = 19.263826。参数从 `research/E200_final_freeze/bundle/frozen_bundle.pt` 原样复制，未重新拟合。提交包为 `final/brainprint.zip`（370,903 B，sha256 `8f7bb32c2cdbe49322dc8e95e0a7a9f470e371d53d6c2d5db559669165c4af98`）。

**当前 AUC / F1**：决赛测试集不公开，所以 N/A。初赛对应值见 E200：S1 0.7135，S2 0.7069，身份头 OOF macro-F1 0.9882。

## 接口检查（`selftest.py`、`package.py`）

全部通过：
- `weights_only=True` 加载
- CPU 和 cuda:0 下 N = 1/7/128 都输出 (N, 24)，无 NaN，每行有有限值
- 不修改传入的 x
- meta=None 可运行
- 注册前不会输出 21–23
- 从 zip 解压到隔离目录后可以 import，60 条注册样本 enroll 成功

## 与 E200 一致性

Test 前 1024 条的预测标签与 `submission_frozen.csv` 一致率为 1.0。

## Train 决赛流程模拟（20 次随机抽取，seed 20261116）

每次抽取：3 名新用户各用 20 条早期试次（frac ≤ 0.6）注册，3 名未注册者删除原型，其余 14 名为已注册用户。测试集为后 20% 试次。

| 注册方式 | 已注册识别 | 新用户识别 | 未注册拒识 |
|---|---|---|---|
| 按任务分别建原型（每任务约 5 条） | 0.801 | 0.694 | 0.673 |
| **全部 20 条合并建原型（采用）** | 0.798 | **0.803** | 0.659 |

结果是同会话的，偏乐观。明细见 `selftest_result.json`。
