# E262 波形输入模块：Train 行映射与一致性

独立运行 `.\.venv\Scripts\python.exe -B research/E262_wave_pair/views.py` 成功。`BrainPrint_cache/e262_wave_pair/` 下的 `clean.npy`、`train_aug.npy`、`stress.npy` 均为 `(11304, 855)`、float32，全部元素有限。

**行映射。** 输出第 i 行对应 `BrainPrint_cache/train_metadata.csv` 第 i 条数据记录（不计表头，未经排序）。`path` 相对 `Brainprint_Competition/`；全部路径唯一，与 `Train` 下的 MAT 文件集合完全相等。逐行核对了 subject、task_folder、trial 与路径，以及 MAT 内 fs、session_id、task_label 与 CSV。首行为 `Train/Subject_01/Task1/trial_001.mat`，末行为 `Train/Subject_20/Task4/trial_200.mat`。原始 `data` 为 1200 点 × 18 通道，转置为通道优先后提取。

**来源与增强。** 与 `research/E253_session_robust/dump_features.py` 共用 `final/team_hymn/model.py` 的 `_subwindow_feature` 和 `weights.pt` 的 feature 配置。每行在 GPU 上分别提取 6 个 200 点子窗的 855 维特征并取均值；未使用 PCA、原型或跨行统计。增强作用于整条原波形，再切窗。

| 视图 | CUDA 种子 | 每条每通道均匀增益 | 减去通道均值的比例 | 高斯噪声标准差 / 原通道标准差 |
|---|---:|---:|---:|---:|
| clean | — | 1 | 0 | 0 |
| train_aug | 26201 | [0.95, 1.05] | 0.08 | 0.01 |
| stress | 26202 | [0.90, 1.10] | 0.16 | 0.025 |

**一致性。** 对 `BrainPrint_cache/e253_subwindow/train_feat.npy` 沿六子窗轴取 NumPy 均值，与从原波形独立计算的 clean 逐行逐元素比较。最大绝对误差 `9.53674316406e-07`，平均绝对误差 `3.15466500722e-08`；全部通过 `atol=rtol=1e-5`，数值一致但非逐位相同。

AUC、F1：不适用，本次仅构造输入，没有训练、预测或分类评估。未读取 Test，未操作 git 或 final2。
