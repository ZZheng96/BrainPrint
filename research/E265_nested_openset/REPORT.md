# E265：嵌套未注册训练的六子窗 DeepSets 门控

**联合判定：不通过；保留 final2。本轮不读取 Test，不修改 E264 或 final2。**

表中为 20 个身份外折等权均值 / 最差折；accepted_misrank 的最差为最大值，其余为最小值。两臂 AUROC/U@95/rej@90 均使用原 base gate，F1 使用原 argmax；深度分数仅决定冻结阈值下的接受/拒识。

|视图与规则|AUROC|U95|R90|F1|old_e2e|known_accept|unknown_reject|accepted_misrank|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clean_package|0.98858 / 0.95382|0.95465 / 0.63710|0.98814 / 0.87903|0.99454 / 0.99303|0.86662 / 0.85492|0.86662 / 0.85492|0.99452 / 0.94355|0.00000 / 0.00000|
|clean_nested|0.98858 / 0.95382|0.95465 / 0.63710|0.98814 / 0.87903|0.99454 / 0.99303|0.91264 / 0.86786|0.91285 / 0.86786|0.97840 / 0.87097|0.00021 / 0.00046|
|stress_package|0.98497 / 0.94391|0.93158 / 0.58065|0.98061 / 0.84677|0.99234 / 0.99026|0.83587 / 0.82700|0.83587 / 0.82700|0.99426 / 0.94355|0.00000 / 0.00000|
|stress_nested|0.98497 / 0.94391|0.93158 / 0.58065|0.98061 / 0.84677|0.99234 / 0.99026|0.88976 / 0.82738|0.88979 / 0.82738|0.98161 / 0.88710|0.00002 / 0.00045|

候选减同折现包（百分点；accepted_misrank 越低越好）：

|视图|AUROC|U95|R90|F1|old_e2e|known_accept|unknown_reject|accepted_misrank|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clean|+0.0000|+0.0000|+0.0000|+0.0000|+4.6024|+4.6229|-1.6126|+0.0205|
|stress|+0.0000|+0.0000|+0.0000|+0.0000|+5.3889|+5.3912|-1.2648|+0.0023|

选中 q 分布：0.05: 16 折，0.10: 4 折。

## 11 项指标

|#|指标|数值与证据性质|
|---:|---|---|
|1|Train 样本内 acc|0.9996461429582448（现包历史值；E265未重算）|
|2|OOF acc|0.9988499646142959（现包历史值；E265未重算）|
|3|OOF macro-F1|0.9988253655342916（现包历史值；E265未重算）|
|4|forward acc|0.9982509838216004（现包历史值；E265未重算）|
|5|strict LOSO AUROC|0.988579331209（E265 Train 晚段20折均值）|
|6|U@95|0.954647276468（E265 Train 晚段20折均值）|
|7|RA|N/A|
|8|ASafeR|N/A|
|9|Test S2|0.22416484545738372（现包历史值；E265未审计）|
|10|Test S1|0.5918232772832179（现包历史值；E265未审计）|
|11|strict LOSO rej@90|0.988142981172（E265 Train 晚段20折均值）|

## 冻结方法与防泄漏

- 每折留出一个身份；其余19人的前60%拟合任务标准化、PCA-256、类内白化及深度门控。中段(60%,80%]只做阈值选择，所有身份末20%只评估一次；held 身份不参与任何拟合、原型、网络或阈值。
- 19人按位置模5构造5个 inner enrollment folds，每次约15人注册，其余作为 inner unknown。每个 inner fold 内再按身份×任务的时间奇偶分半，查询使用另一半 clean 原型；clean 与 train_aug 都训练。label 为原 top1 是否等于真实身份，inner unknown 和已知错排均为0。
- 每个查询仅评估原 base 第一名：6×4子窗证据经共享4→32→32 DeepSets、mean/min/max池化和E264统计；8维 cohort 为 s1、s2、margin、gate、mean、std、s1−mean、(s1−mean)/std，经8→16→16 MLP后融合。残差 logit=12×(base_gate−0.33)+residual，末层零初始化，BCE pos_weight 按全体训练标签平衡。
- 每个 inner fold 用早段完整注册原型和中段 clean 查询校准。base tau 为 known base 分 P10；候选 q 网格固定为[.05,.10,.15,.20,.25,.30]，先要求 inner known accept 与 unknown reject 均值均不低于 base，再依次最大化 known accept、unknown reject、较小 q；无合格项时最大化两项差值的较小者，再取较小 q。outer 候选 tau 为19人中段 clean candidate 分的选中 q 分位。stress 从未训练或校准。
- U@95/rej@90 是晚段 base 连续分经验 ROC 上已知接受率达到95%/90%时的未知拒识，不是冻结阈值拒识率。old_e2e 要求已知身份排对且接受；accepted_misrank 是已知错排且接受的比例。逐折权重在 weights/held01.pt 至 held20.pt，q、inner 校准诊断及逐折结果在 results.json。

## 验收条件

- 两臂 AUROC/U@95/rej@90 与 package 逐折完全一致：True
- 两臂 F1/argmax 与 package 完全一致：True
- clean/stress old_e2e、known_accept、unknown_reject 均值与最差均不降：False
- clean/stress accepted_misrank 均值不增：False
- 联合 passed：False

复跑：`..\.venv\Scripts\python.exe -B research/E265_nested_openset/run.py`。
