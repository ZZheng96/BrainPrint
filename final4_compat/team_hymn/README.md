## 1. 方法简述
两个冻结分支独立把 4 秒 EEG 划为 6 个 200 点窗口，各提取五频带通道相关矩阵的收缩 logSPD 特征，窗口平均后按任务标准化。门控分支在 PCA-96 空间按最近原型欧氏距离门控（距离严格小于冻结阈值才接受，平局拒识）；排序分支用 PCA-256、类内白化及每身份四原型的最大余弦相似度排序。是否拒识只由欧氏距离门控决定。

## 2. 输入、输出及元数据
`model.predict_batch(artifact, x, meta)`：`x` 为 float32 `(N,18,1200)`，1≤N≤128，固定通道顺序，不修改输入。返回 NumPy float32 `(N,24)`。第 1–20 列是原用户，第 21–23 列是新用户；未注册列为 `-inf`。第 0 列编码门控（最高身份分 ±1；无有限身份分时为 0），不是概率。逐行 argmax 后将列 0 映射为标签 **-1**。

`meta` 通常为逐样本字典列表。仅当主办方显式提供任务/范式时才传入，如 `{'task':'EC'}`；支持键 task/Task/TASK、paradigm/Paradigm、task_name，以及 EO、Imagery、SSVEP 或 Task1–Task4 等完整别名。缺失、长度不符或未识别时按未知任务匹配：每个身份在四种任务中取最佳分。不可从文件名、路径、目录、样本排列顺序或隐藏信息推断任务或身份，也不要在调用方预先从这些信息生成 task。只有注册时的 `labels` 指定新身份。

## 3. 训练数据来源与预处理
基础模型来自主办方初赛 Train（20 名注册用户、11,304 条样本）。基础数据由主办方完成 50 Hz 陷波、1–40 Hz 带通、坏道插值、4 秒分段、异常片段剔除及逐通道去均值。本包接收预处理后的 300 Hz 波形，不重复执行整套采集预处理；实际在线特征提取见 `src/euclidean_gate.py` 与 `src/cosine_rank.py`。不含训练或测试数据。

## 4. 新用户注册
`artifact = model.enroll(artifact, x, meta, labels)`：一次传入 60 条注册波形，`labels` 是 `(60,)` 的 int64，每人 20 条，取值 21、22、23。两分支都按显式标签聚合，不按 meta 分组；在四套任务投影中分别建模板。门控使用均值；排序在同标签组内依输入次序分四块取均值，不足四条时复制总均值。分块不用于确定身份。仅更新内存中新用户模板和有效掩码；再次注册同一 ID 覆盖其模板，不是增量累加。重新加载需重新注册。

## 5. 部署与本地自测
解压 `team_hymn.zip` 后直接得到 `team_hymn/`。主办方入口为该目录根部 `model.py`；`inference.py` 导出相同的加载与预测函数，`enroll.py` 导出相同的注册函数，不是命令行程序。`load_model(model_dir, device='cpu')` 从所传目录读取 `weights.pt`；可传队伍根目录或其 `model/` 子目录，两处权重字节相同。支持 `cpu`、`cuda:0`，CUDA 不可用时回落 CPU。

在解压后的 **team_hymn/** 目录启动 Python，运行以下合成数据接口检查（无需外部文件；不是准确率测试）：

```python
import numpy as np
import model
rng = np.random.default_rng(42)
a = model.load_model('.', device='cpu')
x = rng.standard_normal((60, 18, 1200), dtype=np.float32)
a = model.enroll(a, x, [{} for _ in x], np.repeat([21, 22, 23], 20).astype('int64'))
q = rng.standard_normal((1, 18, 1200), dtype=np.float32)
s = model.predict_batch(a, q, [{}])
assert s.shape == (1, 24) and not np.isnan(s).any() and np.isfinite(s).any(axis=1).all()
i = s.argmax(axis=1)
print(np.where(i == 0, -1, i))
```

## 6. 已知限制与失败情况
不重采样、不补道、不推断通道布局。其他设备须由评测方先对齐到约定输入；设备或会话分布变化、任务缺失和少样本注册可导致错认或误拒。固定门控不做测试集适配，余弦分和门控列不能当作校准概率。错误维度、缺失或不兼容权重不受支持。

## 7. 依赖与离线运行
目标为 CentOS 7.8.2003、Python 3.10.22、torch 2.5.1+cu121、NumPy 1.26.4、CUDA runtime 12.1、cuDNN 9.1.0.70。平台提供 SciPy 1.13.1，但提交入口不导入它。`requirements.txt` 仅声明依赖，不执行 pip install。无外部模型、网络服务、自定义 CUDA 扩展或训练步骤；预测不读磁盘。
