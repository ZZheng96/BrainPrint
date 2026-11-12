# E20：EEG 图像化 + DINOv2 冻结特征 脑纹识别

日期：2026-11-12
状态：E20A / E20B 已完成，路线判定为**弱于协方差基线**

## 1. 结论先行

**图像化 + 冻结 DINOv2 在本数据上可行，但明显弱于既往协方差 + Ridge 基线。**

| 指标 | 本实验最佳（topomap + attention_pool） | 既往协方差基线（E02 base） | 差距 |
|---|---:|---:|---:|
| C | 0.822 | 0.932 | −0.110 |
| 身份 macro-F1 | 0.790 | 0.956 | −0.166 |
| U | 0.191 | 0.990 | −0.799 |
| unknown AUROC | 0.809 | 0.9935 | −0.185 |
| unknown F1 | 0.310 | 0.968 | −0.658 |

按 C 排序的最佳配置是 topomap + cls_patch_mean（C=0.841、AUROC=0.789）；按 AUROC 排序最佳是 topomap + attention_pool（AUROC=0.809、C=0.822）。

**核心问题不在分类头，而在表征**：冻结 DINOv2 的 CLS/patch 特征对 EEG 图像的身份判别力不足，且开集拒识能力（U、unknown F1）严重落后。

## 2. 实验协议

- 数据：Train Session1，20 名被试，11,304 条 4 秒 Trial
- 外层：5 折留被试，每折 4 人作未知、16 人作注册
- 已知测试：注册者每 (subject, task) 后 20% Trial，共 2,308 条
- 未知测试：留出被试全部 Trial
- 分类器：StandardScaler + LogisticRegression(C=1.0)
- 拒识：top1−top2 margin，阈值取已知置信度 5% 分位
- 骨干：`vit_base_patch14_dinov2.lvd142m`，img_size=224，768 维，**完全冻结**
- 环境：D 盘 `.venv`，torch 2.7.1+cu118，RTX 3080 Laptop 16G

## 3. E20A：五种图像表示 + CLS 线性探针

| 表示 | C | 身份 macro-F1 | U | unknown AUROC | unknown F1 |
|---|---:|---:|---:|---:|---:|
| **topomap** | **0.832** | **0.801** | **0.172** | **0.774** | **0.283** |
| correlation | 0.825 | 0.795 | 0.146 | 0.772 | 0.245 |
| spectrogram | 0.771 | 0.739 | 0.124 | 0.740 | 0.212 |
| mtf | 0.448 | 0.426 | 0.055 | 0.544 | 0.100 |
| gaf | 0.404 | 0.387 | 0.060 | 0.544 | 0.110 |

**发现：**

1. **topomap 与 correlation 最好**，两者都直接编码通道间空间结构，与身份信息最相关。
2. **spectrogram 次之**，时频图信息量大但 18 通道 × 5 频带压进 224×224 有损严重。
3. **GAF / MTF 基本失效**（AUROC≈0.54，接近随机）。原因：18 通道波形降采样到 224 点后做 GAF，图像极其稀疏且通道间差异被平均掉；MTF 的 8-bin 量化进一步丢失信息。
4. 所有表示的 **U 都极低（0.05–0.17）**，说明 margin 置信度在图像特征上无法区分已知/未知。

## 4. E20B：patch token 池化对比

对 topomap 与 correlation 两种最优表示，比较 4 种特征聚合方式。

### topomap

| 池化 | C | 身份 F1 | U | unknown AUROC | unknown F1 |
|---|---:|---:|---:|---:|---:|
| cls | 0.832 | 0.801 | 0.172 | 0.774 | 0.283 |
| patch_mean | 0.835 | 0.804 | 0.163 | 0.778 | 0.270 |
| cls+patch_mean | **0.841** | **0.810** | **0.181** | 0.789 | **0.296** |
| attention_pool | 0.822 | 0.790 | **0.191** | **0.809** | **0.310** |

### correlation

| 池化 | C | 身份 F1 | U | unknown AUROC | unknown F1 |
|---|---:|---:|---:|---:|---:|
| cls | 0.825 | 0.795 | 0.146 | 0.772 | 0.245 |
| patch_mean | 0.814 | 0.784 | 0.137 | 0.761 | 0.232 |
| cls+patch_mean | **0.837** | **0.807** | **0.156** | **0.784** | **0.260** |
| attention_pool | 0.790 | 0.762 | 0.153 | 0.759 | 0.255 |

**发现：**

1. **patch token 确实携带额外信息**：topomap 上 attention_pool 把 AUROC 从 0.774 提到 0.809（+0.035），unknown F1 从 0.283 提到 0.310。
2. **但 C 与 AUROC 存在权衡**：attention_pool 提升开集排序能力，却降低闭集准确率（0.832→0.822）。
3. **cls+patch_mean 是 C 最优**（topomap 0.841、correlation 0.837），说明 CLS 与 patch 均值互补。
4. **correlation 上 attention_pool 反而变差**，说明该表示的可学习池化容易过拟合。

## 5. 为什么弱于协方差基线

| 原因 | 说明 |
|---|---|
| **图像化有损** | 协方差基线用 6 频带 × 171 维 = 1,026 维精细空间结构；图像化压到 224×224×3 后，通道间细微差异被插值和归一化抹平 |
| **预训练域不匹配** | DINOv2 训练于自然图像，其 CLS 编码的是物体语义，对 EEG 的细粒度个体差异不敏感 |
| **冻结骨干容量受限** | 线性探针只能利用预训练特征中已有的方向；EEG 身份信息可能不在 DINOv2 的主成分方向上 |
| **开集能力缺失** | 图像特征上 margin 置信度无法区分已知/未知，U 仅 0.19，而协方差基线达 0.99 |
| **GAF/MTF 表示本身不适合** | 波形级图像丢失频域结构，而身份信息主要藏在频带功率与空间协方差中 |

## 6. 与调研预期的差异

调研中 ICML 2026 论文显示 DINOv3 线性探针在 EEG 下游任务上优于 CBraMod。本实验未能复现该优势，可能原因：

1. 该论文的 EEG→图像适配协议（5.1.2 节）未公开细节，可能包含针对 EEG 的专门设计（如通道重排、多切片堆叠），而非本实验的直接图像化。
2. 该论文的下游任务是癫痫检测、睡眠分期等**粗粒度任务**，与脑纹识别这种**细粒度身份区分**难度不同。
3. 该论文用 DINOv3（更大规模预训练），本实验用 DINOv2 ViT-B。

## 7. 后续可选方向（未执行）

按投入产出排序：

1. **图像特征 + 协方差特征融合**：把 DINOv2 特征与 1,026 维协方差拼接，看是否互补。但这已不是"纯图像路线"。
2. **LoRA 微调 DINOv2**：让骨干适应 EEG 图像域，可能显著提升，但训练成本高且有过拟合风险。
3. **改进图像表示**：把协方差矩阵直接作为图像输入（而非相关矩阵），或设计多切片堆叠保留更多频带细节。
4. **换用 EEG 专用基础模型**：如 CBraMod、NeuroShield，但这就偏离了"视觉模型"的原始思路。

## 8. 产物

| 文件 | 内容 |
|---|---|
| `plan.md` | 方案与调研依据 |
| `run_e20a.py` | 五种图像表示 + CLS 线性探针 |
| `run_e20b.py` | patch token 池化对比 |
| `e20a_linear_probe.json` | E20A 完整指标 |
| `e20b_pooling.json` | E20B 完整指标 |
| `e20a_run.log` / `e20b_run.log` / `e20b_corr.log` | 运行日志 |
| `src/eeg_image_dinov2/image_encoders.py` | 五种图像编码器 |
| `src/eeg_image_dinov2/backbone.py` | 冻结 DINOv2 封装 |
| `src/eeg_image_dinov2/heads.py` | 轻量头 + ArcFace（E20B 未使用，留作后续） |

## 9. 复现

```powershell
# 环境（D 盘 venv，节省 C 盘空间）
D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe

# E20A：五种表示 + CLS 线性探针
$env:HF_ENDPOINT="https://hf-mirror.com"
.venv\Scripts\python.exe research\E20_eeg_image_dinov2\run_e20a.py

# E20B：patch token 池化
.venv\Scripts\python.exe research\E20_eeg_image_dinov2\run_e20b.py
```
