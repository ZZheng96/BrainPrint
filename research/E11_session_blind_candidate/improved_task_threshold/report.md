# Task-specific 阈值与单次稳健目标参考实验

## 设计与边界

使用 CUDA；五个外层 blocked 折，内层固定留出一个完整 blocked 折校准（其余三折训练），外层模型用四折重拟合。全局/task 阈值均固定为校准置信度 5% 分位；最终 Test 阈值由完整训练 blocked-OOF 得分确定。未读取 session/trial/path/fs 或 Test 标签。

高置信参考：每 task 取初始置信度前 50%，按白化切空间距离剔除最远 10%，几何均值与原参考作 0.5 测地线收缩，仅一次。该规则未根据验证结果选择。

训练 OOF 只有已注册身份，C 为拒识计错的身份准确率，F1 为标签 1–20 宏平均（拒识作为相应真实类别 FN）；U、开集 AUC 不可计算。Test C/U/AUC/F1 均不可计算；Test 拒识率不等于 U。未模拟对抗，不能估计官方综合分。

内层单折校准是严格独立的 blocked holdout，并非完整四折内层交叉验证；减少计算但阈值方差可能较大。目标批次对齐是无标签传导推理，不是单样本归纳推理。

## 同口径嵌套 OOF 与 Test 拒识比较

|方法|阈值|OOF C|OOF F1|OOF 拒识率|Test 拒识率|
|---|---|---:|---:|---:|---:|
|full_none|global|0.966561|0.982500|0.033351|0.792279|
|full_none|task|0.961607|0.979536|0.038128|0.791964|
|base_none|global|0.965410|0.981793|0.034590|0.792998|
|base_none|task|0.960014|0.978753|0.039986|0.796414|
|full_target_zscore|global|0.967534|0.983004|0.032378|0.948766|
|full_target_zscore|task|0.961164|0.979315|0.038570|0.942250|
|base_target_zscore|global|0.965145|0.981604|0.034855|0.953575|
|base_target_zscore|task|0.960103|0.978812|0.039897|0.949620|
|full_riemannian_recenter|global|0.964968|0.981664|0.034943|0.822435|
|full_riemannian_recenter|task|0.961341|0.979467|0.038570|0.824547|
|full_riemannian_robust_once|global|0.964349|0.981343|0.035563|0.832996|
|full_riemannian_robust_once|task|0.961430|0.979675|0.038482|0.839288|

## 与原五候选比较

|方法|原 Test 拒识率|新 task Test 拒识率|标签变化数|task-global OOF ΔC|ΔF1|
|---|---:|---:|---:|---:|---:|
|full_none|0.792279|0.791964|723|-0.004954|-0.002963|
|base_none|0.792998|0.796414|478|-0.005396|-0.003039|
|full_target_zscore|0.948766|0.942250|569|-0.006369|-0.003689|
|base_target_zscore|0.953575|0.949620|360|-0.005042|-0.002793|
|full_riemannian_recenter|0.822435|0.824547|669|-0.003627|-0.002197|

稳健参考相对原黎曼方法（均 task 阈值）：ΔC=+0.000088，ΔF1=+0.000208。

以上只能判断已注册身份的训练 OOF 表现，不能证明 Test 或未知拒识改进，不按 Test 拒识率选优；不替换正式提交。按 task 的详细指标及阈值见 metrics.json、manifest.json。所有 12 份候选均为 22251 行，Sample 与模板对齐、Label 为 0–20 整数。正式提交 SHA256 保持不变。

## 复现

`py -3.11 research/E11_session_blind_candidate/improved_task_threshold/run.py`

GPU: NVIDIA GeForce RTX 3080 Laptop GPU；耗时 359.5 秒。
