# full_none_global vs raw-PGD 统一评估

## 口径

- 数据仅使用 Train Session1；五折按 subject 划分，阈值仅由 inner holdout 选择；未使用 Test 标签、Test Session 或 Session 规则。
- `full_none_global`：完整协方差特征的 full Ridge 分支，none 对齐、全局 operating point。
- `adversarial_ridge`：复用 `src/raw_eeg_robust/run.py` 已完成的 raw-PGD 五折结果：clean + 当前拟合折内 raw-PGD 增强并重拟合 Ridge。两者的攻击缓存、折划分和官方指标公式一致；raw-PGD 数值来源为已有真实结果缓存，未伪造或覆盖。

## 五折加权汇总

| 模型 | 攻击 | clean C | clean F1 | RA | ASafeR | A | U | unknown AUROC(AUC) | unknown F1 | 官方 Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_none_global | untargeted_snr20 | 0.9361 | 0.9068 | 0.0000 | 0.3325 | 0.0998 | 0.9866 | 0.9921 | 0.9678 | 43.32 |
| full_none_global | untargeted_snr30 | 0.9361 | 0.9068 | 0.0087 | 0.7135 | 0.2201 | 0.9866 | 0.9921 | 0.9678 | 57.15 |
| full_none_global | targeted_snr20 | 0.9361 | 0.9068 | 0.0000 | 0.0040 | 0.0012 | 0.9866 | 0.9921 | 0.9678 | 9.23 |
| adversarial_ridge | untargeted_snr20 | 0.9072 | 0.8894 | 0.4541 | 0.9694 | 0.6087 | 0.9421 | 0.9718 | 0.9337 | 79.61 |
| adversarial_ridge | untargeted_snr30 | 0.9072 | 0.8894 | 0.9266 | 0.9999 | 0.9486 | 0.9421 | 0.9718 | 0.9337 | 93.01 |
| adversarial_ridge | targeted_snr20 | 0.9072 | 0.8894 | 0.3882 | 0.8576 | 0.5290 | 0.9421 | 0.9718 | 0.9337 | 75.81 |

`A = 0.7×RA + 0.3×ASafeR`; 官方 Score = `100×C^0.40×A^0.35×U^0.25`. clean F1 为含拒识的宏 F1；unknown F1 为未知检测 F1。

## 运行

`py -3.11 research/unified_full_none_vs_raw_pgd/run_unified.py`

结果文件：`metrics.json`、`metrics.csv`、本报告。
