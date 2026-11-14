# E66：冻结 NeuroShield-FM 四窗嵌入与归一化身份原型

日期：2026-11-13。目标是检验外部大规模EEG身份预训练能否改善开放集与跨会话表征；不根据Test结果回调阈值或选择变体。

## 结论与当前 AUC / F1

十个inner run按五外层折聚合：**direct C=0.6207，threshold C=0.6048，direct身份F1=0.6080，含拒识身份F1=0.5986，pseudo-unknown AUROC=0.6698，unknown AUPR=0.4799，U@95=0.1234，工作点U=0.0711**。

拟合块身份准确率均值=0.7053；该值只作流程诊断。E66未运行攻击套件，**Train RA/ASafeR=N/A（未评估）**，不声称Train全部门禁通过。

## 冻结表征与决策

- 官方NeuroShield-FM（MIT）权重严格加载并全部冻结；预训练数据源为TUH-EEG、DVS-rsEEG、PREDICT，官方仓库未列BrainPrint。输出每个1秒窗128维L2归一化嵌入。
- 通道顺序固定为P3/C3/F3/Fz/F4/C4/P4/Cz/Fp1/Fp2/T3/T5/O1/O2/F7/F8/T6/T4，3D位置由官方MNE standard_1005接口得到。
- 单条4秒trial内逐通道median/1.4826×MAD稳健归一化（近零时RMS回退），随后CAR；GPU FFT保留1–45Hz并从300Hz/1200点重采样到500Hz/2000点。无跨trial或批次统计。
- BrainPrint幅值单位未确认，因此不采用官方绝对µV质量阈值或±400µV裁剪；这是与官方输入协议的明确偏差。
- 四个不重叠500点窗分别推理，嵌入算术平均后再次L2归一化。身份模型仅为fit≤60%块的类均值归一化原型，无WCCN、无encoder微调。
- 置信度为最大原型余弦相似度；低于clean-known q02输出0，否则输出最近身份。q05仅用于U@95诊断。阈值不使用unknown或Test。

## Train inner 五外层折均值

| direct C | threshold C | direct身份F1 | 含拒识身份F1 | 连续块 | AUROC | unknown AUPR | U@95 | 工作点U | 已知接受 | unknown F1 | joint F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6207 | 0.6048 | 0.6080 | 0.5986 | 0.6207 | 0.6698 | 0.4799 | 0.1234 | 0.0711 | 0.9778 | 0.1099 | 0.4520 |

逐外层折：

| outer | direct C | threshold C | direct F1 | 含拒识F1 | AUROC | AUPR | U@95 | U | fit准确率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6284 | 0.6125 | 0.6145 | 0.6062 | 0.7092 | 0.5653 | 0.2227 | 0.1914 | 0.7360 |
| 2 | 0.6554 | 0.6375 | 0.6404 | 0.6306 | 0.6830 | 0.4732 | 0.1406 | 0.0156 | 0.7087 |
| 3 | 0.6222 | 0.6062 | 0.6154 | 0.6050 | 0.6493 | 0.4322 | 0.0742 | 0.0273 | 0.6982 |
| 4 | 0.6018 | 0.5818 | 0.5871 | 0.5733 | 0.6520 | 0.4422 | 0.0547 | 0.0156 | 0.6739 |
| 5 | 0.5957 | 0.5859 | 0.5827 | 0.5778 | 0.6555 | 0.4867 | 0.1250 | 0.1055 | 0.7095 |

## 连续块置信度漂移

漂移为当前块统计减去同run的fit≤60%统计；不是跨Session漂移。

| 块 | mean | q02 | q05 | median | mean漂移 | q02漂移 | q05漂移 | median漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit≤60% | 0.9660 | 0.8879 | 0.9202 | 0.9732 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 60–70% | 0.9614 | 0.8586 | 0.9100 | 0.9708 | -0.0046 | -0.0293 | -0.0101 | -0.0023 |
| 70–80% | 0.9643 | 0.8889 | 0.9139 | 0.9715 | -0.0017 | 0.0010 | -0.0063 | -0.0017 |

## 预注册 Test 门禁

| 门禁 | 实测 | 最低值 | 结果 |
|---|---:|---:|---|
| clean_direct_C | 0.6207 | 0.9500 | 未通过 |
| clean_identity_macro_F1_direct | 0.6080 | 0.9200 | 未通过 |
| clean_AUROC | 0.6698 | 0.9800 | 未通过 |
| clean_U_at_95 | 0.1234 | 0.9500 | 未通过 |

四项clean门禁全部通过：**False**。

## Test 执行状态与工程审计

**Test=N/A，未读取Test，未生成e66_predictions.csv**。原因：clean_direct_C、clean_identity_macro_F1_direct、clean_AUROC、clean_U_at_95未达到预注册门禁；没有尝试任何Test变体。

**Test AUC/F1/C/RA/ASafeR/U/Score=N/A（无真实标签）**。S1拒识/S2接受若存在仅为预测工程审计，不是官方真值指标。

## 证据边界与复现

- 所有AUC/F1/C/U均来自Train Session1 pseudo-unknown与连续文件块代理，不是跨Session或官方新被试真值证据。
- inner阈值和known指标使用同一60–80%已知连续块；unknown来自互斥被试。final20才把60–70%校准与70–80% sanity分开。
- E66不依赖E58/E65的CNN、logSPD或Test预测；E54仅用于复用被试角色与分层抽样。没有Session、文件顺序、批次身份分布或配对信息进入推理。

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E66_neuroshield_frozen\run_e66.py" --stage smoke
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E66_neuroshield_frozen\run_e66.py" --stage inner
# 仅当四项clean门禁全部通过：
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" "D:\WORK\BCIC2026\BrainPrint\research\E66_neuroshield_frozen\run_e66.py" --stage test
```

配置：`D:\WORK\BCIC2026\BrainPrint\research\E66_neuroshield_frozen\config.json`；结果：`D:\WORK\BCIC2026\BrainPrint\research\E66_neuroshield_frozen\results.json`；缓存：`D:\WORK\BCIC2026\BrainPrint\BrainPrint_cache\e66_neuroshield_fm`。断点绑定配置和代码SHA256。
