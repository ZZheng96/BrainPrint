# Enrollment template session-robust v1 scaled 实验报告

严格外层五折 subject holdout；每个 outer fold 仅使用 fit mask 的 full/window 特征拟合 GPU StandardScaler，模板和测试使用同一 scaler。窗口模板按 trial 两窗口平均后进行 robust medoid 聚合；Train/Test 推理不读取 session。

|决策|AUC|identity macro-F1|detection macro-F1|unknown-F1|U|known acceptance|C|
|---|---:|---:|---:|---:|---:|---:|---:|
|prototype_only|0.734395|0.522782|0.666725|0.740041|0.822470|0.514617|0.468454|
|prototype_margin|0.751237|0.471204|0.646420|0.777203|0.967284|0.362823|0.355117|
|prototype_window|0.721265|0.550953|0.663254|0.729501|0.795345|0.532012|0.493816|

选择方法：prototype_only。输出候选 22251 行，拒识 1136 行。
阈值来自全量 Train 的 fresh blocked OOF 已知置信度 5% 分位；Test 无真值，因此 Test AUC/F1 不可计算。
与批准 Ridge 基线 AUC=0.965632、F1=0.936207 比较：prototype_only AUC=0.734395、identity macro-F1=0.522782；prototype_margin AUC=0.751237、identity macro-F1=0.471204；prototype_window AUC=0.721265、identity macro-F1=0.550953，均低于 Ridge 基线。
Session 仅在候选写出后做拒识计数审计，未参与模板、阈值、决策或后处理。
