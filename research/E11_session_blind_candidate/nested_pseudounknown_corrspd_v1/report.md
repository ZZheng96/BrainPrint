# nested pseudounknown corrSPD v1

固定 lambda=1e-3；corrSPD 对每个频带恢复 SPD、归一化为相关矩阵、shrinkage 后重新 log-vectorize。严格 blocked 外层评估，并在每折留出一个身份作为伪未知校准；未读取 session/trial/path/fs/Test 标签。

|方法|外层已知 C|外层 F1|伪未知 U|AUC|已知接受率|最差折 C|最差折 F1|Test 拒识率|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|logcov|0.998222|0.949012|0.949757|null|0.000000|0.997085|0.948326|0.374770|
|corrspd|0.994105|0.946536|0.949757|null|0.000000|0.989310|0.944429|0.429284|

AUC、RA、ASafeR、代理分均为 null：本实验严格使用留身份伪未知校准，不构造攻击或真实未知标签，不能补值。

复现：`py -3.11 research/E11_session_blind_candidate/nested_pseudounknown_corrspd_v1/run.py`
