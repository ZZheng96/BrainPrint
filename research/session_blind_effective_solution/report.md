# Session-blind effective solution

## 推荐
推荐 `test_candidate_fused.csv`：direct identity 使用训练域 blocked-OOF q05 阈值，raw-PGD adversarial-ridge 仅在 direct 拒识或两者一致时接管；另生成 `test_candidate_direct_oof_q01.csv` 作为较少拒识的训练域阈值候选。两者均不读取 Session，且不使用 Test 全量统计。

## 身份评估口径与五折 Train 数据
已注册身份评估为 `known_eval` 中的正确身份接受率：C=预测身份等于真实身份的比例（已知样本不含伪未知）；U=伪未知样本被拒识比例。伪未知为外层留出身份，不等同 Test 未注册真值。修正后五折阈值只由外层训练折内 4 折 inner-OOF 分数拟合；此前约 C=0.5、U=1 的结果使用 fit 数据自身阈值，已不作为当前口径。

| method | C | acceptance | U | unknown AUC | unknown F1 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| direct_identity | 0.9980 | 0.9999 | 0.0616 | 0.9925 | 0.1158 | 0.6645 |

## 按任务五折均值
| task | C | acceptance | U | unknown AUC | unknown F1 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| EC | 0.9988 | 1.0000 | 0.0636 | 0.9891 | 0.1186 | 0.6814 |
| EO | 0.9956 | 1.0000 | 0.0665 | 0.9811 | 0.1237 | 0.6811 |
| Imagery | 0.9995 | 0.9998 | 0.0603 | 0.9955 | 0.1132 | 0.6808 |
| SSVEP | 0.9964 | 1.0000 | 0.0618 | 0.9933 | 0.1161 | 0.6766 |

## 抗干扰 Train 结果（同一 raw-PGD 攻击缓存）
| attack | method | C | RA | ASafeR | A | U | AUC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| untargeted_snr20 | direct | 0.9228 | 0.0000 | 0.4405 | 0.1321 | 0.9567 | 0.9803 | 0.9472 |
| untargeted_snr20 | adversarial_ridge | 0.9072 | 0.4541 | 0.9694 | 0.6087 | 0.9421 | 0.9718 | 0.9337 |
| untargeted_snr30 | direct | 0.9228 | 0.0100 | 0.7879 | 0.2434 | 0.9567 | 0.9803 | 0.9472 |
| untargeted_snr30 | adversarial_ridge | 0.9072 | 0.9266 | 0.9999 | 0.9486 | 0.9421 | 0.9718 | 0.9337 |
| targeted_snr20 | direct | 0.9228 | 0.0000 | 0.0071 | 0.0021 | 0.9567 | 0.9803 | 0.9472 |
| targeted_snr20 | adversarial_ridge | 0.9072 | 0.3882 | 0.8576 | 0.5290 | 0.9421 | 0.9718 | 0.9337 |

## Test 候选分布审计
Test 共 22251 条；raw-PGD CSV 与 Test template 的 Sample 顺序已逐行核对一致。融合候选拒识 11939 条（0.5366），标签计数见 `metrics.json`。direct OOF-q01 候选阈值为 0.09804343，拒识率为 0.2280。推理未读 Session；Session1/2 与 task 仅用于生成后审计。

| task | session | n | fused reject rate | direct reject rate | robust reject rate |
|---|---|---:|---:|---:|---:|
| EC | session1 | 292 | 0.3151 | 0.4863 | 0.3596 |
| EC | session2 | 1680 | 0.2655 | 0.6089 | 0.3012 |
| EO | session1 | 291 | 0.5292 | 0.5292 | 0.5876 |
| EO | session2 | 2021 | 0.6437 | 0.6556 | 0.8238 |
| Imagery | session1 | 1459 | 0.5442 | 0.5442 | 0.5915 |
| Imagery | session2 | 9748 | 0.5842 | 0.5846 | 0.8402 |
| SSVEP | session1 | 991 | 0.4995 | 0.5005 | 0.5923 |
| SSVEP | session2 | 5769 | 0.5134 | 0.5394 | 0.6658 |

## 推荐
推荐 `test_candidate_fused.csv`：身份分支使用训练域 blocked-OOF q05 阈值，robust 分支仅在 direct 拒识或两者一致时接管；raw-PGD 输入已确认与 Test sample 顺序一致。若优先降低拒识，使用 `test_candidate_direct_oof_q01.csv`，其阈值和标签计数已写入 `metrics.json`。Train 五折 direct 的均值 C=0.9980、U=0.0616、unknown AUC=0.9925、unknown F1=0.1158、macro-F1=0.6645；Test 无真实标签，因此 Test AUC/F1 不计算。
