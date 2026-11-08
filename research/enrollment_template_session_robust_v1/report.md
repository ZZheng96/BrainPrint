# Enrollment template session-robust v1 实验报告

严格外层五折 subject holdout；Train/Test 推理不读取 session。

|决策|AUC|identity macro-F1|detection macro-F1|unknown-F1|U|known acceptance|C|
|---|---:|---:|---:|---:|---:|---:|---:|
|prototype_only|0.676291|0.439350|0.623150|0.701272|0.782879|0.469729|0.381086|
|prototype_margin|0.689446|0.345020|0.586035|0.745551|0.938700|0.291880|0.265931|
|prototype_window|0.670312|0.396202|0.595823|0.713092|0.846569|0.371331|0.316435|

选择方法：prototype_only。输出候选 22251 行，拒识 11273 行。
阈值来自全量 Train 的 fresh blocked OOF 已知置信度 5% 分位；Test 无真值，因此 Test AUC/F1 不可计算。
Session 仅在候选写出后做拒识计数审计，未参与模板、阈值、决策或后处理。
