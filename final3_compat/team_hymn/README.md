## 方法简述

每条脑电分为六个 200 点子窗。每窗在 1–8、8–13、13–20、20–30、30–40 Hz 五个频带计算通道相关矩阵，经收缩、矩阵对数和上三角标准化得到 855 维特征。六窗取平均后按任务标准化，投影至 256 维 PCA 空间。每个身份取四个原型中最大的白化余弦分数。最大分数加上第一、第二名分差的一半，低于阈值则拒识。旧用户阈值约为 0.407559；新用户阈值比它低 0.03。

## 输入与输出

`load_model(model_dir, device)` 仅读取指定目录的 `weights.pt`，可传队伍根目录或其 `model/` 子目录。支持 CPU 和 CUDA。根 `model.py` 提供三个接口；也可从 `inference.py` 导入加载和预测函数、从 `enroll.py` 导入注册函数。

`predict_batch(artifact, x, meta)` 接收 float32 `(N,18,1200)`，N 为 1–128，不改变输入或通道顺序。返回 float32 `(N,24)`：第 0 列胜出表示未知（标签 -1），第 1–20 列为原用户，第 21–23 列为新用户。尚未注册的身份分数为负无穷。

显式任务字段（如 `task`、`paradigm`、`task_name`）支持 EC、EO、Imagery、SSVEP 或 Task1–4。无 meta 或无法识别任务时，逐身份取四个任务空间中的最大分数。不使用文件名、目录或身份字段。

## 训练数据来源和预处理

使用主办方提供的 20 名原用户 Train 脑电。输入为主办方预处理后的 300 Hz、4 秒片段；模型保留全部 18 通道。任务标准化、PCA、白化矩阵与原用户原型均在训练数据上拟合；拒识阈值由训练集交叉拟合分数确定。

## 新用户注册方法

`enroll(artifact, x, meta, labels)` 使用 21–23 各 20 条带标签样本。对每名用户的全部注册样本，在每个任务空间分别投影，按提供顺序分为四组取均值作为原型，匹配时才执行白化。返回新的状态，不修改传入状态、训练参数、旧身份原型或阈值。

## 本地运行与自测

从解压所得 `team_hymn/` 的父目录运行以下 Python 代码。使用固定种子的合成数据，只检查接口，不衡量识别率。

```python
import sys
import numpy as np
sys.path.insert(0, 'team_hymn')
from model import load_model, predict_batch, enroll
rng = np.random.default_rng(123)
r = rng.standard_normal((60, 18, 1200), dtype=np.float32)
y = np.repeat([21, 22, 23], 20).astype(np.int64)
a = load_model('team_hymn', 'cpu')
b = enroll(a, r, None, y)
for n in (1, 128):
    x = rng.standard_normal((n, 18, 1200), dtype=np.float32)
    original = x.copy()
    old = predict_batch(a, x, None)
    new = predict_batch(b, x, None)
    assert new.shape == (n, 24) and not np.isnan(new).any()
    assert np.isfinite(new).any(1).all() and np.array_equal(x, original)
    assert np.array_equal(old[:, 1:21], new[:, 1:21])
    assert np.isneginf(old[:, 21:]).all() and np.isfinite(new[:, 21:]).all()
    col = new.argmax(1)
    labels = np.where(col == 0, -1, col)
```

## 已知限制和失败情况

跨设备或跨时段可能造成绝对匹配分下降，从而误拒已注册用户。新用户注册样本过少或不具代表性会使原型不稳定；缺任务字段时跨任务取最大值可能增加误接受。

## 依赖说明

使用 Python 3.10.22、torch 2.5.1+cu121 和 numpy 1.26.4。不需要安装其他包、自定义 CUDA 扩展或访问网络。
