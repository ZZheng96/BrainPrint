# E213 E170 自适应跨会话 gate

## 协议

仅复用 E170 `frozen_scores.npz`，不重新提取原始 EEG。gate 输入为 E65、logbp、E162 及 E170 分数的 6 个两两绝对差、跨流标准差和极差。轻量 NumPy logistic gate 使用 Train OOF known 与 Train LOSO unknown 训练；正则强度只按 Train LOSO AUC 选择。Test 只在冻结模型后审计。

## 结果

- Train OOF AUC：**0.9512**
- Train OOF F1（gate known/unknown，阈值 0.5）：**0.9389**
- LOSO AUC（训练选择依据）：**0.9527**
- Test 官方 AUC/F1：**N/A**（Test 无标签；仅报告 session 诊断 AUC 0.7449）

### 冻结参考工作点 acceptance=0.90

- LOSO 拒识：**0.8718**
- Test S1 拒识：**0.8566**
- Test S2 接受：**0.5076**
- W 审计值（不用于选参/选点）：**0.6859**

## 产物

- `run_e213.py`：可复现实验脚本
- `results.json`：完整曲线与指标
- `model.npz`：冻结 gate 参数
- `adaptive_scores.npz`：冻结 Train/Test gate 分数
- `report.md`：本报告

运行命令：`d:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe research\E213_e170_adaptive_cross_session_gate\run_e213.py`
