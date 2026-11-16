E202 双模型实验

运行：
.venv\Scripts\python.exe research\E202_dual_model_cross_session\run_e202.py --device cuda --epochs 5 --max-folds 2 --k 2 5 10

协议：只读取 Train metadata、E65 train feature、E162 train embedding；不读取 Test。留出身份排除模型 A fit/prototype；模型 B 使用注册身份内层 unknown 训练，最终留出身份只用于评估。

产物：results.json、report.md、checkpoints/model_a_*.pt、bundle/gate_*.npz。
