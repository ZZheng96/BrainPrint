# E269 深度度量编码器：严格 Train-only LOSO 与新用户注册模拟

> **官方 Test AUC = N/A；官方 Test F1 = N/A。** 本实验未读取或加载任何 Test 数据、test_metadata 或隐藏标签。
> **当前新模型 LOSO clean AUROC = 0.8317，闭集 macro-F1 = 0.8340；stress AUROC = 0.7208，闭集 macro-F1 = 0.6736。**
> **注册模拟 clean AUROC = 0.7718，注册身份闭集 macro-F1 = 0.6934；stress AUROC = 0.6981，闭集 macro-F1 = 0.5660。**
> **晋级结论：不晋级，不建议替换。四场景等权 proxy 配对差 = -0.1966；本轮未修改 final/final2。**

## 1. 完成度与固定配方

- 实际完成 **20/20 个 LOSO 折 + 20/20 个注册模拟折**；无早停、无按晚段选折。
- 每折仅用老身份 `f<=0.6` 训练；老用户原型使用老身份全部早 60%；新用户每人每任务最早 5 条、共 20 条注册，且均满足 `f<=0.6`。
- 阈值仅取 14/19 个老身份中段 `(0.6,0.8]` clean base 的第 10 百分位；所有相关身份 `f>0.8` 只评估。查询不进入自身模板。
- 每身份×任务按 trial 顺序四等分，各自均值后 L2 归一化；同任务余弦，身份取四原型最大，`base=s1+0.5*(s1-s2)`。
- 输入一次性在 GPU 上对每个样本每通道独立去均值、除以 std（clamp=1e-6）；网络没有额外输入 RMS/方差归一化。
- 网络为 depthwise temporal k65/s2 + pointwise 到 64 宽，两个 depthwise/pointwise block，GroupNorm+ELU，mean/std/max 池化，Linear→128 并 L2 normalize；CosFace scale=24、margin=0.15。
- CE + 固定 0.1 supervised contrastive（temp=0.1）+ 固定 0.1 clean/aug 余弦一致性；AdamW lr=0.002、wd=1e-4、batch=128、20 epochs、cosine、AMP。
- 20 类时总参数量 **39,826**（每折实际头大小随老身份数变化，均 <200k）。LOSO seed=269000+h；注册 seed=279000+h。
- Train-only 增强包括通道增益±10%、通道 dropout=0.05、SNR15–25dB 白噪声、循环时移±60、0.9–1 随机 crop 后线性重采样、谱倾斜±0.35、部分公共平均参考0.1；全部在标准化后的 GPU 波形上执行。
- 技术修复：首次运行在 registration_h07 发现循环移位后的 old 列表非升序，初版 `searchsorted` 标签映射因此越界。修复仅为 `classes=sorted(old)`；此前已完成的 h00–h06 的 old 列表本就升序，数值不受影响。模型、epoch、增强、阈值和评估配方均未改变。

## 2. 四场景主代理（注册模拟，20 折）

均值为 20 折宏平均；最差折对错排以外指标取最小。`old_device` 使用合成 stress，**不是实际跨设备成绩**。

|场景|E254 均值 / 最差折|新模型均值 / 最差折|配对差均值|
|---|---:|---:|---:|
|old_time|0.8533 / 0.8222|0.7846 / 0.7394|-0.0688|
|old_device|0.8206 / 0.7930|0.5921 / 0.5256|-0.2285|
|new_time|0.1242 / 0.0000|0.1167 / 0.0085|-0.0075|
|URR|0.9917 / 0.9504|0.5100 / 0.3739|-0.4817|
|proxy_score|0.6975 / 0.6546|0.5008 / 0.4611|-0.1966|

等权 `proxy_score=(old_time+old_device+new_time+URR)/4`。晋级判定预先固定为四个场景均值均不下降且 proxy 均值实际提高；本次 `not_harms=False`，`actual_improvement=False`。

## 3. LOSO：新模型与 E254 研究基线配对对照

F1 为晚段已知老身份的闭集 macro-F1；old_e2e 要求身份正确且接受。U95/R90 是晚段已知接受率 95%/90% 时的未知拒识诊断，不是部署阈值。accepted_misrank 的最差折取最大。

|方法|视图|AUROC 均值 / 最差|F1 均值 / 最差|old_e2e 均值 / 最差|known_accept|unknown_reject|accepted_misrank|U95|R90|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E254_research|clean|0.9873 / 0.9350|0.9932 / 0.9916|0.8569 / 0.8416|0.8569|0.9928|0.0000|0.9540|0.9818|
|E254_research|stress|0.9825 / 0.9237|0.9897 / 0.9883|0.8234 / 0.8104|0.8234|0.9940|0.0000|0.9290|0.9750|
|new|clean|0.8317 / 0.7283|0.8340 / 0.8094|0.7861 / 0.7683|0.8714|0.5233|0.0852|0.2572|0.4391|
|new|stress|0.7208 / 0.6301|0.6736 / 0.6384|0.5818 / 0.5568|0.7272|0.5735|0.1454|0.1425|0.2541|

**clean 配对差（新模型−E254）**：AUROC -0.1556，F1 -0.1592，old_e2e -0.0707，unknown_reject -0.4696，accepted_misrank +0.0852。

**stress 配对差（新模型−E254）**：AUROC -0.2617，F1 -0.3161，old_e2e -0.2416，unknown_reject -0.4204，accepted_misrank +0.1454。

## 4. 注册模拟详细结果

|方法|视图|AUROC|闭集 F1|old_e2e|new_e2e|new closed acc|known_accept|unknown_reject|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|E254_research|clean|0.9519|0.9416|0.8533|0.1242|0.7277|0.7250|0.9917|
|E254_research|stress|0.9445|0.9343|0.8206|0.1178|0.7045|0.6969|0.9913|
|new|clean|0.7718|0.6934|0.7846|0.1167|0.1941|0.8095|0.5100|
|new|stress|0.6981|0.5660|0.5921|0.0723|0.1456|0.6797|0.5875|

clean 配对差：AUROC -0.1802，F1 -0.2482，old_e2e -0.0688，new_e2e -0.0075，URR -0.4817。
stress 配对差：AUROC -0.2464，F1 -0.3683，old_e2e -0.2285，new_e2e -0.0455，URR -0.4038。

## 5. 权重、运行与复现

- 新模型及折级原型、valid、tau、classes/people、tasks、固定配置和逐 epoch 训练日志均保存在 `research/E269_deep_metric/weights/loso_h01.pt`–`loso_h20.pt` 与 `registration_h00.pt`–`registration_h19.pt`。
- 同一权重文件还保存 E254 的 fold-fit projection、原型、valid、tau 与逐折结果，便于严格配对复核。
- 40 折 fold 总耗时 **10.9 min**；其中新模型训练 **10.6 min**；最终汇总进程墙钟 **3.1 min**。
- 复跑：`.venv\Scripts\python.exe research/E269_deep_metric/run.py --validate`，然后 `.venv\Scripts\python.exe research/E269_deep_metric/run.py`。完整且配置一致的折自动断点复用。
- 逐折训练及复用记录见 `research/E269_deep_metric/run.log`；全部折级均值/min/max/配对差见 `RESULTS.json`。

## 6. 严格边界与局限

- 本实验只使用 Train 元数据与 Train 缓存；官方 Test AUC/F1 始终为 N/A，不能从本报告推断隐藏测试成绩。
- E254 是按 E259 投影规则在每个 fold 仅用 old fit 重新拟合的研究基线，clean/stress 分别使用 E262 的 clean.npy/stress.npy；**不是现包 final2 的精确双白化实现**。
- late stress 在标准化波形上以固定 seed 按全局晚段行顺序即时生成：通道增益±20%、dropout=0.1、SNR12–20dB、谱倾斜±0.7、部分公共平均参考0.25、时移±80。波形不保存，所有折一致。
- **合成 stress 不等同真实跨设备**；时间块来自同一可见 session，也不等同真实跨时段。40 折共享试次，不是 40 个独立数据集。
- 新用户 backbone 只用 14 个 old 的早 60% 训练；new 注册只建模板，不更新 backbone。校准不看 new 或 unknown，晚段不参与调架构、epoch、增强、阈值或选折。
- 晋级仅表示值得后续独立复核，本轮没有修改 final/final2，也没有生成部署模型。
