# Ridge session robust v1 实验报告

严格五折 subject holdout；outer enrolled 的后20% trial 为 known，held subjects 全部为 unknown；inner 伪未知只用于阈值选择。拒识统一使用 top-1 margin，Test Session 仅候选写出后审计。

|方法|AUC|identity macro-F1|detection macro-F1|unknown-F1|U|known acceptance|C|
|---|---:|---:|---:|---:|---:|---:|---:|
|full|0.992699|0.930372|0.944097|0.952543|0.995577|0.883960|0.883960|
|base|0.993997|0.925737|0.942171|0.951435|0.996516|0.878477|0.878477|
|robust|0.993997|0.925737|0.942171|0.951435|0.996516|0.878477|0.878477|

历史 Ridge 基线：AUC=0.965632，identity macro-F1=0.936207。
稳健归一化低于 Ridge 基线，淘汰原因：外层平均 AUC 或 identity macro-F1 未达到 0.965632/0.936207，因此候选不采用 robust；候选使用 base Ridge 对照。
最终候选方法：base；Test 行数=22251，拒识=17619。Test 无真值，Test AUC/F1 不可计算。
Session 仅在候选写出后做按会话拒识计数审计，未参与训练、阈值、模型选择或后处理。
