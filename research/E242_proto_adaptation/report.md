# E242 Train-only 高置信原型自适应

完整20折，设备 cuda，耗时 19.6s。Test任何文件/标签均未读取。

## 严格协议与范围
- 阅读赛题说明：20注册+5未知，Test含对抗样本；会话不是身份/未知标签。
- 只读复用E241 af0d5c05ca9e的fit-only标准化、PCA96/k12原型及共同E65特征缓存，不使用增强视图。断言fit/query行ID与E241完全一致。
- 外层fit<=0.6；当前窗口前半(0.6,0.7]只适应，后半(0.7,0.8]只评估。baseline也只评估同一后半，不能直接对比E241完整窗口指标。
- 内层两组pseudo-unknown及时间块沿用E241，所有pseudo-unknown从表征/原型拟合排除。每个块前半适应后半查询；只有边界无后半的块3排除。base表征沿用leave-block-out，可使用其他较晚fit块；适应严格先于查询，不宣称全流程前向链训练。
- train-simulated adaptation实际执行在Train inner OOF时间块中。此最小版本不另训练自适应编码器，不把同一机制伪称独立第三模型。
- 高置信门槛=known baseline inner OOF history分数90%分位数；每个inner窗口用另身份组且另时间块计算门槛；outer/final用汇总inner分数冻结。拒识阈值由各候选adapted inner unknown后半分数90%分位数冻结。
- EMA alpha=0.1，按trial稳定顺序更新；center_mean将当前中心作一个伪观测与高置信样本均值融合。两者仅更新最近身份-task簇中心，不改表征、不重新聚类，分配与筛选均用冻结初始中心。EMA不滚动重筛。
- 所有适应函数只接收特征、task、冻结原型与门槛，不接收query身份；Train身份仅用于规定LOSO/时间分块和离线指标。
- inner目标0.90不保证outer拒识0.90，两组身份库大小差异存在校准偏移；未按outer结果回调参数。

|候选|LOSO AUC|unknown-positive F1|未知拒识micro|已知接受micro|
|---|---:|---:|---:|---:|
|baseline|0.897450|0.315439|0.810360|0.828104|
|query_ema|0.896855|0.313779|0.805092|0.828058|
|query_center_mean|0.896229|0.313366|0.805092|0.827734|

AUC与unknown-positive F1均为逐折均值；正类=外层held身份，AUC使用负置信分数。

## 最终Train内层模拟（校准集，不是独立外层表现）
|候选|AUC|unknown-positive F1|
|---|---:|---:|
|baseline|0.899112|0.848503|
|query_ema|0.898616|0.848126|
|query_center_mean|0.898499|0.847498|

## 结论与阻塞
本次为最小可验证Train-only版本，不宣称满足LOSO拒识约0.90约束或提升S2。Test严格时间顺序及无身份标签的流分组尚未确认，因此宁可不读取Test，也不使用Test身份字段构造前后半窗口。Test AUC/F1=N/A，S1/S2=N/A；S2未来也仅可描述性。未计算独立train-simulated元训练候选。

## 产物与复现
`models`保存每内外折冻结/自适应原型、行ID、分数、表征状态和最终Train模型；`cache`保存投影特征与OOF校准分数；`frozen_thresholds.json`保存全部门槛。源E241文件保持只读。
运行：`.venv-py311\Scripts\python.exe research\E242_proto_adaptation\run_e242.py`。
