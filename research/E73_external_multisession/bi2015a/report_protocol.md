# BI2015a 协议报告

## 协议

本目录实现 BI2015a 的动态跨 Session 评估协议：43 名受试者、3 个 Session、32 通道、300 samples、每 session 的实际 events.tsv trial 数，固定 6 个 run。训练严格只使用 enrolled 受试者的 source/target Session 的 run 0、1、2；run 3 仅作 template，run 4 用于 calibration（5 名 unknown 与已知校准角色按折配置隔离），run 5 用于 evaluation（14 名 outer unknown）。固定三折规模为 enrolled=24、calibration_unknown=5、outer_unknown=14，固定方向为 1→2、2→1、1→3、3→1、2→3、3→2。trial 顺序不作为特征，run 仅用于数据隔离；Test/competition 模式禁止。

## 数据阻塞

目标派生文件应位于 `cache/derived/subject_XX/session_Y.npz`，并由严格只读加载器校验 `raw`、`run`、`trial_index` 的键、shape、dtype、有限性及 run 分区。当前数据准备状态为 blocked：subject-1 的 events.tsv/json 可通过 NEMAR API 下载并解析，但三 session 的 BDF 均未完整下载；ses-0 留有 `.part`，因此未重复下载或读取不完整文件，未产生 NPZ。测试不会伪造数据；缺失文件会明确抛出 `FileNotFoundError`。

## 测试结果

已运行动态协议测试与既有协议测试；在无完整 BDF/NPZ 时，缺失数据测试确认加载器拒绝伪造通过。events.tsv 解析与 BDF 元数据读取因主体文件未完成而阻塞。

- AUC=N/A（未训练模型）
- F1=N/A（未训练模型）
