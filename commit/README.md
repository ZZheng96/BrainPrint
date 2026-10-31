# BrainPrint 提交包

本目录只保留最终答案、可复现代码、模型说明和训练集报告。原始 MAT 数据与缓存特征不随仓库提交；它们应按项目根目录中的数据结构准备。

## 目录

- `answer/submission_task_specific_zscore_grouped_guarded.csv`：最终测试集提交答案。
- `code/`：特征提取、训练、任务特异融合、Sinkhorn 后处理及预测代码。
- `model/`：模型说明。模型不依赖持久化权重，运行时使用 `StandardScaler + RidgeClassifier(alpha=100)` 重新拟合。
- `report/`：只记录训练集验证结果和指标。
- `review/`：研究过程、测试侧无标签诊断及其他不属于最终提交包的材料。
- `requirements.txt`：Python 依赖版本。

## 数据和缓存

代码默认读取项目根目录下的 `BrainPrint_cache/`。缓存由 `code/build_features.py` 和 `code/build_window_features.py` 生成；若按当前项目目录运行，需将 `code` 目录作为脚本目录，并把缓存路径改为项目根目录下的 `BrainPrint_cache`，或建立同名目录映射。原始数据目录为 `Brainprint_Competition/Train` 与 `Brainprint_Competition/Test`。

最小输入包括：训练/测试特征与元数据、双窗口特征、Ridge/窗口/邻居分数，以及启用波形图融合时的 `waveform_pairs.csv`。这些缓存文件体积较大且可由原始 MAT 数据重建，因此不纳入提交。

## 复现步骤

```powershell
py -3 -m pip install -r commit/requirements.txt
py -3 commit/code/build_features.py --splits train,test --output-dir BrainPrint_cache
py -3 commit/code/build_window_features.py --split train
py -3 commit/code/build_window_features.py --split test
```

随后按代码所需的训练/测试分数缓存运行：

```powershell
py -3 commit/code/predict_task_specific.py --pair-fusion --balanced-ec-imagery --guarded-ec
```

该命令生成任务特异融合候选；最终答案文件已经放在 `answer/`。训练集指标可由已有缓存运行：

```powershell
py -3 commit/code/evaluate_task_specific_ensemble.py --balanced-ec-imagery --guarded-ec
py -3 commit/code/evaluate_task_specific_reverse.py --holdout reverse --balanced-ec-imagery --guarded-ec
py -3 commit/code/evaluate_task_specific_reverse.py --holdout middle --balanced-ec-imagery --guarded-ec
```

## 结果边界

训练集验证结果为块外推 11,304/11,304、时间前推 2,308/2,308、反向留段 2,245/2,245、中间留段 2,257/2,257 正确。上述结果均为训练集内部验证，不是测试集准确率，也不能替代官方 C、RA、ASafeR、U 评分。测试集没有标签；`session1 -> 0` 是基于数据结构的元数据规则，不能解释为已通过脑电模型证明未知身份识别率为 100%。
