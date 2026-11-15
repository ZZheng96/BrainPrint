# E175 老实验（E26–E35）LOSO 补算与 W 重建

性质：只读补算。未修改任何已有实验目录，未提交 git。
Test 官方 AUC / F1：**N/A**（无真实标签）。

```
W = 0.25 × LOSO 拒识 + 0.25 × Test S1 拒识 + 0.5 × Test S2 接受
```

## 0. 结论先行

- 共补算 **159** 个候选，其中 **159** 个为 `clean` 且重建一致率 ≥ 99%。
- 最高可信 **W = 0.6394**（E27_unknown_rejection / `reject30`，LOSO 0.7501 / S1 0.8622 / S2 0.4726）。
- 与当前已知最高可信值 0.7171（E170 固定工作点）比较：**没有超过**（差 -0.0777）。
- 重建失败（一致率 < 99%）的候选：**0** 个。
- 因缺缓存 / 缺代码而跳过的实验：**0** 个。

## 1. 方法与有效性判定

1. 逐实验按原 `run_*.py` 的代码路径重算 Test 分数 → 应用该候选 CSV 对应的阈值 → 得到预测标签，与已有 CSV 逐行比对（一致率）。全部候选的最低一致率为 1.0000。
2. LOSO 用项目标准协议：每次留出 1 个身份，原型 / 标准化 / 白化 / PCA / 身份头**只用**其余 19 人在 `frac <= 0.6` 的 trial；被评分为 known 块（注册者，`0.6 < frac <= 0.8`）与 unknown 块（留出身份，`0.6 < frac <= 0.8`）。
3. LOSO 阈值与该候选**同一规则**：折内 known 块分数的同一分位（`1 - rate`）；E35 `per_task` 用逐任务分位。
4. `W` 的 S1 / S2 直接取自候选 CSV（session1 = 3,033 行，session2 = 19,218 行）。

有效性标记：`clean`（Train-only 定阈值 + LOSO 无原型污染）、`test_statistic`（阈值或分数用 Test 统计）、`loso_contaminated`、`pipeline_mismatch`、`skipped`。

**阈值规则核查**：这 10 个实验的候选阈值全部来自 Train 侧（内层分位 / 10 个内层模型分位），没有任何候选使用 Test 分位数，因此无 `test_statistic`。
需要单独说明的是 E31 的 `task_model_*`、E32、E34、E35 的 `global` 策略：原脚本全量模型的类中心把 `inner` 块（0.6–0.8）也算了进去，于是「Train 内层分位」的分位参考被压低（阈值更严）。这属于 Train 侧标定瑕疵，不影响 Test 标签本身；本实验的 LOSO 已按干净协议重建。

## 2. 逐实验逐候选

### E26_final_candidates

阈值来自全量模型 ~inner&~forward 原型 + Train 内层分位；无 Test 统计。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `reject1` | 0.2074 | 0.9653 | 1.0000 | 0.0915 | 0.5574 | clean |
| `reject2` | 0.3093 | 0.9300 | 1.0000 | 0.1415 | 0.5777 | clean |
| `reject5` | 0.4669 | 0.8450 | 1.0000 | 0.2433 | 0.6001 | clean |
| `reject20` | 0.7570 | 0.5994 | 1.0000 | 0.5920 | 0.6370 | clean |
| `reject50` | 0.9588 | 0.2793 | 1.0000 | 0.9164 | 0.6085 | clean |
| `reject70` | 0.9898 | 0.1326 | 1.0000 | 0.9794 | 0.5586 | clean |
| `reject85` | 0.9987 | 0.0389 | 1.0000 | 0.9946 | 0.5178 | clean |
| `reject95` | 1.0000 | 0.0037 | 1.0000 | 1.0000 | 0.5018 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E27_unknown_rejection

阈值来自全量模型 ~inner&~forward 原型 + Train 内层分位；无 Test 统计。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `reject1` | 0.2074 | 0.9653 | 1.0000 | 0.0915 | 0.5574 | clean |
| `reject2` | 0.3093 | 0.9300 | 1.0000 | 0.1415 | 0.5777 | clean |
| `reject5` | 0.4669 | 0.8450 | 1.0000 | 0.2433 | 0.6001 | clean |
| `reject10` | 0.6037 | 0.7503 | 1.0000 | 0.3800 | 0.6211 | clean |
| `reject20` | 0.7570 | 0.5994 | 1.0000 | 0.5920 | 0.6370 | clean |
| `reject30` | 0.8622 | 0.4726 | 1.0000 | 0.7501 | 0.6394 | clean |
| `reject40` | 0.9261 | 0.3616 | 1.0000 | 0.8592 | 0.6272 | clean |
| `reject50` | 0.9588 | 0.2793 | 1.0000 | 0.9164 | 0.6085 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E28_dino_openset

cov5 / cov5+topo5 两个特征集；阈值均为 Train 内层分位。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `cov5_reject1` | 0.4352 | 0.9520 | 1.0000 | 0.0108 | 0.5875 | clean |
| `cov5_reject2` | 0.4784 | 0.9295 | 1.0000 | 0.0219 | 0.5898 | clean |
| `cov5_reject5` | 0.5308 | 0.8820 | 1.0000 | 0.0681 | 0.5907 | clean |
| `cov5_reject10` | 0.5740 | 0.7983 | 1.0000 | 0.1290 | 0.5749 | clean |
| `cov5_reject20` | 0.6337 | 0.6578 | 1.0000 | 0.2677 | 0.5543 | clean |
| `cov5_reject30` | 0.6815 | 0.5419 | 1.0000 | 0.4108 | 0.5440 | clean |
| `cov5_topo5_reject1` | 0.4379 | 0.9467 | 1.0000 | 0.0108 | 0.5855 | clean |
| `cov5_topo5_reject2` | 0.4784 | 0.9237 | 1.0000 | 0.0218 | 0.5869 | clean |
| `cov5_topo5_reject5` | 0.5331 | 0.8638 | 1.0000 | 0.0764 | 0.5843 | clean |
| `cov5_topo5_reject10` | 0.5628 | 0.7956 | 1.0000 | 0.1490 | 0.5757 | clean |
| `cov5_topo5_reject20` | 0.6189 | 0.6776 | 1.0000 | 0.2669 | 0.5603 | clean |
| `cov5_topo5_reject30` | 0.6874 | 0.5383 | 1.0000 | 0.4204 | 0.5461 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E29_dino_calibration

阈值经 10 次内层留出被试模型标定（Train-only）。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `q01` | 0.4296 | 0.9548 | 1.0000 | 0.0108 | 0.5875 | clean |
| `q02` | 0.4705 | 0.9328 | 1.0000 | 0.0219 | 0.5895 | clean |
| `q05` | 0.5285 | 0.8851 | 1.0000 | 0.0681 | 0.5917 | clean |
| `q10` | 0.5664 | 0.8127 | 1.0000 | 0.1290 | 0.5802 | clean |
| `q20` | 0.6182 | 0.7060 | 1.0000 | 0.2677 | 0.5745 | clean |
| `q30` | 0.6502 | 0.6198 | 1.0000 | 0.4108 | 0.5751 | clean |
| `q40` | 0.6904 | 0.5250 | 1.0000 | 0.5406 | 0.5702 | clean |
| `q50` | 0.7293 | 0.4334 | 1.0000 | 0.6479 | 0.5610 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E30_dino_multiview

分位融合的参考分布只用 Train；LOSO 逐折用 19 人 known 块作参考。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `cov5_dist_q01` | 0.4352 | 0.9521 | 1.0000 | 0.0108 | 0.5876 | clean |
| `cov5_dist_q02` | 0.4784 | 0.9295 | 1.0000 | 0.0224 | 0.5899 | clean |
| `cov5_dist_q05` | 0.5295 | 0.8831 | 1.0000 | 0.0685 | 0.5910 | clean |
| `cov5_dist_q10` | 0.5740 | 0.7983 | 1.0000 | 0.1290 | 0.5749 | clean |
| `cov5_dist_q20` | 0.6337 | 0.6579 | 1.0000 | 0.2677 | 0.5543 | clean |
| `cov5_dist_q30` | 0.6815 | 0.5419 | 1.0000 | 0.4108 | 0.5440 | clean |
| `cov5_wccn_q01` | 0.4804 | 0.9423 | 1.0000 | 0.0135 | 0.5946 | clean |
| `cov5_wccn_q02` | 0.5031 | 0.9207 | 1.0000 | 0.0271 | 0.5929 | clean |
| `cov5_wccn_q05` | 0.5404 | 0.8725 | 1.0000 | 0.0789 | 0.5911 | clean |
| `cov5_wccn_q10` | 0.5852 | 0.7886 | 1.0000 | 0.1413 | 0.5760 | clean |
| `cov5_wccn_q20` | 0.6634 | 0.6237 | 1.0000 | 0.3524 | 0.5658 | clean |
| `cov5_wccn_q30` | 0.7303 | 0.4875 | 1.0000 | 0.5161 | 0.5554 | clean |
| `cov5_wccn+pca_q01` | 0.4860 | 0.9383 | 1.0000 | 0.0140 | 0.5942 | clean |
| `cov5_wccn+pca_q02` | 0.5038 | 0.9205 | 1.0000 | 0.0312 | 0.5940 | clean |
| `cov5_wccn+pca_q05` | 0.5427 | 0.8711 | 1.0000 | 0.0813 | 0.5915 | clean |
| `cov5_wccn+pca_q10` | 0.5823 | 0.7998 | 1.0000 | 0.1422 | 0.5810 | clean |
| `cov5_wccn+pca_q20` | 0.6591 | 0.6391 | 1.0000 | 0.3320 | 0.5673 | clean |
| `cov5_wccn+pca_q30` | 0.7155 | 0.5159 | 1.0000 | 0.4839 | 0.5578 | clean |
| `cov5_dist+wccn_q01` | 0.4715 | 0.9428 | 1.0000 | 0.0124 | 0.5923 | clean |
| `cov5_dist+wccn_q02` | 0.4929 | 0.9239 | 1.0000 | 0.0245 | 0.5913 | clean |
| `cov5_dist+wccn_q05` | 0.5387 | 0.8737 | 1.0000 | 0.0757 | 0.5904 | clean |
| `cov5_dist+wccn_q10` | 0.5783 | 0.7939 | 1.0000 | 0.1367 | 0.5757 | clean |
| `cov5_dist+wccn_q20` | 0.6525 | 0.6259 | 1.0000 | 0.3165 | 0.5552 | clean |
| `cov5_dist+wccn_q30` | 0.7095 | 0.5027 | 1.0000 | 0.4728 | 0.5469 | clean |
| `cov5_dist+pca_q01` | 0.4738 | 0.9433 | 1.0000 | 0.0124 | 0.5932 | clean |
| `cov5_dist+pca_q02` | 0.4972 | 0.9213 | 1.0000 | 0.0253 | 0.5913 | clean |
| `cov5_dist+pca_q05` | 0.5387 | 0.8754 | 1.0000 | 0.0758 | 0.5913 | clean |
| `cov5_dist+pca_q10` | 0.5783 | 0.7996 | 1.0000 | 0.1375 | 0.5787 | clean |
| `cov5_dist+pca_q20` | 0.6439 | 0.6435 | 1.0000 | 0.3038 | 0.5587 | clean |
| `cov5_dist+pca_q30` | 0.7010 | 0.5310 | 1.0000 | 0.4495 | 0.5531 | clean |
| `cov5_dist+wccn+pca_q01` | 0.4771 | 0.9421 | 1.0000 | 0.0119 | 0.5933 | clean |
| `cov5_dist+wccn+pca_q02` | 0.5021 | 0.9179 | 1.0000 | 0.0262 | 0.5911 | clean |
| `cov5_dist+wccn+pca_q05` | 0.5394 | 0.8745 | 1.0000 | 0.0759 | 0.5911 | clean |
| `cov5_dist+wccn+pca_q10` | 0.5803 | 0.7966 | 1.0000 | 0.1393 | 0.5782 | clean |
| `cov5_dist+wccn+pca_q20` | 0.6515 | 0.6346 | 1.0000 | 0.3219 | 0.5606 | clean |
| `cov5_dist+wccn+pca_q30` | 0.7138 | 0.5132 | 1.0000 | 0.4739 | 0.5535 | clean |
| `cov5+topo5_dist_q01` | 0.1405 | 0.9637 | 1.0000 | 0.0324 | 0.5251 | clean |
| `cov5+topo5_dist_q02` | 0.2367 | 0.9362 | 1.0000 | 0.0562 | 0.5413 | clean |
| `cov5+topo5_dist_q05` | 0.3459 | 0.8881 | 1.0000 | 0.1003 | 0.5556 | clean |
| `cov5+topo5_dist_q10` | 0.4883 | 0.8121 | 1.0000 | 0.1768 | 0.5723 | clean |
| `cov5+topo5_dist_q20` | 0.6093 | 0.6741 | 1.0000 | 0.3066 | 0.5660 | clean |
| `cov5+topo5_dist_q30` | 0.6805 | 0.5386 | 1.0000 | 0.4302 | 0.5470 | clean |
| `cov5_dist+pca+topo5_dist_q01` | 0.1850 | 0.9645 | 1.0000 | 0.0317 | 0.5364 | clean |
| `cov5_dist+pca+topo5_dist_q02` | 0.2872 | 0.9382 | 1.0000 | 0.0517 | 0.5538 | clean |
| `cov5_dist+pca+topo5_dist_q05` | 0.4418 | 0.8810 | 1.0000 | 0.0979 | 0.5755 | clean |
| `cov5_dist+pca+topo5_dist_q10` | 0.5552 | 0.8066 | 1.0000 | 0.1701 | 0.5847 | clean |
| `cov5_dist+pca+topo5_dist_q20` | 0.6380 | 0.6673 | 1.0000 | 0.3177 | 0.5726 | clean |
| `cov5_dist+pca+topo5_dist_q30` | 0.6878 | 0.5341 | 1.0000 | 0.4489 | 0.5512 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E31_task_conditional

任务条件化中心；阈值 Train-only，但 task_model_* 的分位原型含 inner 块。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `global_global_q01` | 0.4339 | 0.9534 | 1.0000 | 0.0108 | 0.5879 | clean |
| `global_global_q02` | 0.4787 | 0.9290 | 1.0000 | 0.0219 | 0.5897 | clean |
| `global_global_q05` | 0.5345 | 0.8765 | 1.0000 | 0.0681 | 0.5889 | clean |
| `global_global_q10` | 0.5773 | 0.7885 | 1.0000 | 0.1290 | 0.5708 | clean |
| `global_global_q20` | 0.6409 | 0.6477 | 1.0000 | 0.2677 | 0.5510 | clean |
| `global_taskcenter_q01` | 0.4382 | 0.9505 | 1.0000 | 0.0104 | 0.5874 | clean |
| `global_taskcenter_q02` | 0.4801 | 0.9275 | 1.0000 | 0.0210 | 0.5890 | clean |
| `global_taskcenter_q05` | 0.5351 | 0.8763 | 1.0000 | 0.0691 | 0.5892 | clean |
| `global_taskcenter_q10` | 0.5869 | 0.7704 | 1.0000 | 0.1373 | 0.5662 | clean |
| `global_taskcenter_q20` | 0.6588 | 0.6062 | 1.0000 | 0.2871 | 0.5396 | clean |
| `task_model_taskcenter_q01` | 0.4464 | 0.9469 | 1.0000 | 0.0110 | 0.5878 | clean |
| `task_model_taskcenter_q02` | 0.4817 | 0.9276 | 1.0000 | 0.0216 | 0.5896 | clean |
| `task_model_taskcenter_q05` | 0.5384 | 0.8724 | 1.0000 | 0.0706 | 0.5885 | clean |
| `task_model_taskcenter_q10` | 0.5869 | 0.7731 | 1.0000 | 0.1301 | 0.5658 | clean |
| `task_model_taskcenter_q20` | 0.6578 | 0.6034 | 1.0000 | 0.2955 | 0.5400 | clean |
| `task_model_wccn_q01` | 0.5087 | 0.9166 | 1.0000 | 0.0145 | 0.5891 | clean |
| `task_model_wccn_q02` | 0.5279 | 0.8901 | 1.0000 | 0.0327 | 0.5852 | clean |
| `task_model_wccn_q05` | 0.5582 | 0.8405 | 1.0000 | 0.0769 | 0.5790 | clean |
| `task_model_wccn_q10` | 0.6195 | 0.7065 | 1.0000 | 0.1477 | 0.5450 | clean |
| `task_model_wccn_q20` | 0.7537 | 0.4713 | 1.0000 | 0.3630 | 0.5148 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E32_distance_ratio

dist/ratio/gap 的分位融合；gap 方向与原脚本一致（越大越像已知）。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `dist_q01` | 0.5087 | 0.9166 | 1.0000 | 0.0145 | 0.5891 | clean |
| `dist_q02` | 0.5279 | 0.8901 | 1.0000 | 0.0327 | 0.5852 | clean |
| `dist_q05` | 0.5582 | 0.8405 | 1.0000 | 0.0774 | 0.5791 | clean |
| `dist_q10` | 0.6195 | 0.7065 | 1.0000 | 0.1477 | 0.5450 | clean |
| `dist_q20` | 0.7537 | 0.4713 | 1.0000 | 0.3648 | 0.5153 | clean |
| `ratio_q01` | 0.6044 | 0.7222 | 1.0000 | 0.0503 | 0.5248 | clean |
| `ratio_q02` | 0.8071 | 0.4788 | 1.0000 | 0.1203 | 0.4713 | clean |
| `ratio_q05` | 0.9172 | 0.2827 | 1.0000 | 0.3428 | 0.4563 | clean |
| `ratio_q10` | 0.9782 | 0.1513 | 1.0000 | 0.6103 | 0.4728 | clean |
| `ratio_q20` | 0.9937 | 0.0697 | 1.0000 | 0.8687 | 0.5005 | clean |
| `gap_q01` | 0.0000 | 0.9999 | 1.0000 | 0.0000 | 0.5000 | clean |
| `gap_q02` | 0.0003 | 0.9996 | 1.0000 | 0.0000 | 0.4999 | clean |
| `gap_q05` | 0.0040 | 0.9977 | 1.0000 | 0.0000 | 0.4998 | clean |
| `gap_q10` | 0.0073 | 0.9957 | 1.0000 | 0.0000 | 0.4997 | clean |
| `gap_q20` | 0.0142 | 0.9932 | 1.0000 | 0.0000 | 0.5002 | clean |
| `dist+ratio_q01` | 0.5058 | 0.8936 | 1.0000 | 0.0244 | 0.5793 | clean |
| `dist+ratio_q02` | 0.5516 | 0.8442 | 1.0000 | 0.0587 | 0.5747 | clean |
| `dist+ratio_q05` | 0.6736 | 0.6445 | 1.0000 | 0.1837 | 0.5366 | clean |
| `dist+ratio_q10` | 0.8391 | 0.3628 | 1.0000 | 0.3902 | 0.4887 | clean |
| `dist+ratio_q20` | 0.9512 | 0.1502 | 1.0000 | 0.6991 | 0.4877 | clean |
| `dist+gap_q01` | 0.0076 | 0.9962 | 1.0000 | 0.0000 | 0.5000 | clean |
| `dist+gap_q02` | 0.0106 | 0.9953 | 1.0000 | 0.0000 | 0.5003 | clean |
| `dist+gap_q05` | 0.0214 | 0.9912 | 1.0000 | 0.0000 | 0.5009 | clean |
| `dist+gap_q10` | 0.0280 | 0.9879 | 1.0000 | 0.0013 | 0.5013 | clean |
| `dist+gap_q20` | 0.0382 | 0.9805 | 1.0000 | 0.0062 | 0.5014 | clean |
| `ratio+gap_q01` | 0.0363 | 0.9883 | 1.0000 | 0.0000 | 0.5032 | clean |
| `ratio+gap_q02` | 0.0422 | 0.9853 | 1.0000 | 0.0000 | 0.5032 | clean |
| `ratio+gap_q05` | 0.0508 | 0.9806 | 1.0000 | 0.0019 | 0.5035 | clean |
| `ratio+gap_q10` | 0.0630 | 0.9742 | 1.0000 | 0.0110 | 0.5056 | clean |
| `ratio+gap_q20` | 0.0861 | 0.9587 | 1.0000 | 0.0499 | 0.5134 | clean |
| `dist+ratio+gap_q01` | 0.0399 | 0.9877 | 1.0000 | 0.0000 | 0.5038 | clean |
| `dist+ratio+gap_q02` | 0.0465 | 0.9845 | 1.0000 | 0.0010 | 0.5041 | clean |
| `dist+ratio+gap_q05` | 0.0719 | 0.9773 | 1.0000 | 0.0100 | 0.5091 | clean |
| `dist+ratio+gap_q10` | 0.5256 | 0.8788 | 1.0000 | 0.0836 | 0.5917 | clean |
| `dist+ratio+gap_q20` | 0.6769 | 0.5989 | 1.0000 | 0.2794 | 0.5385 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E33_arcface_head

ArcFace/CE 头需重训，若一致性 <99% 记为 pipeline_mismatch。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `arcface_m0.0_d0.5_q01` | 0.2694 | 0.9256 | 1.0000 | 0.0314 | 0.5380 | clean |
| `arcface_m0.0_d0.5_q02` | 0.3699 | 0.8688 | 1.0000 | 0.0706 | 0.5446 | clean |
| `arcface_m0.0_d0.5_q05` | 0.4504 | 0.7908 | 1.0000 | 0.1690 | 0.5502 | clean |
| `arcface_m0.0_d0.5_q10` | 0.5562 | 0.6543 | 1.0000 | 0.2983 | 0.5408 | clean |
| `arcface_m0.0_d0.5_q20` | 0.6963 | 0.4667 | 1.0000 | 0.5320 | 0.5404 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E34_covariance_hr

covariance_hr 特征 + 任务内 WCCN。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `covariance_hr_q01` | 0.4557 | 0.9487 | 1.0000 | 0.0440 | 0.5993 | clean |
| `covariance_hr_q02` | 0.4906 | 0.9269 | 1.0000 | 0.0657 | 0.6025 | clean |
| `covariance_hr_q05` | 0.5493 | 0.8634 | 1.0000 | 0.0992 | 0.5938 | clean |
| `covariance_hr_q10` | 0.6175 | 0.7680 | 1.0000 | 0.1626 | 0.5790 | clean |
| `covariance_hr_q20` | 0.6960 | 0.5940 | 1.0000 | 0.2642 | 0.5371 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

### E35_per_task_analysis

global 与 per_task 两种阈值策略。

| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |
|---|---:|---:|---:|---:|---:|---|
| `global_q01` | 0.5087 | 0.9166 | 1.0000 | 0.0145 | 0.5891 | clean |
| `global_q02` | 0.5279 | 0.8901 | 1.0000 | 0.0327 | 0.5852 | clean |
| `global_q05` | 0.5582 | 0.8405 | 1.0000 | 0.0769 | 0.5790 | clean |
| `global_q10` | 0.6195 | 0.7065 | 1.0000 | 0.1477 | 0.5450 | clean |
| `global_q20` | 0.7537 | 0.4713 | 1.0000 | 0.3630 | 0.5148 | clean |
| `per_task_q01` | 0.4510 | 0.9565 | 1.0000 | 0.0141 | 0.5945 | clean |
| `per_task_q02` | 0.4866 | 0.9353 | 1.0000 | 0.0281 | 0.5964 | clean |
| `per_task_q05` | 0.5216 | 0.8964 | 1.0000 | 0.0692 | 0.5959 | clean |
| `per_task_q10` | 0.5658 | 0.8289 | 1.0000 | 0.1476 | 0.5928 | clean |
| `per_task_q20` | 0.6169 | 0.7242 | 1.0000 | 0.3867 | 0.6130 | clean |

阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。

## 3. 总排序（仅 `clean` 且重建一致）

| # | 实验 | 候选 | S1 拒识 | S2 接受 | LOSO | W |
|---:|---|---|---:|---:|---:|---:|
| 1 | E27_unknown_rejection | `reject30` | 0.8622 | 0.4726 | 0.7501 | 0.6394 |
| 2 | E26_final_candidates | `reject20` | 0.7570 | 0.5994 | 0.5920 | 0.6370 |
| 3 | E27_unknown_rejection | `reject20` | 0.7570 | 0.5994 | 0.5920 | 0.6370 |
| 4 | E27_unknown_rejection | `reject40` | 0.9261 | 0.3616 | 0.8592 | 0.6272 |
| 5 | E27_unknown_rejection | `reject10` | 0.6037 | 0.7503 | 0.3800 | 0.6211 |
| 6 | E35_per_task_analysis | `per_task_q20` | 0.6169 | 0.7242 | 0.3867 | 0.6130 |
| 7 | E26_final_candidates | `reject50` | 0.9588 | 0.2793 | 0.9164 | 0.6085 |
| 8 | E27_unknown_rejection | `reject50` | 0.9588 | 0.2793 | 0.9164 | 0.6085 |
| 9 | E34_covariance_hr | `covariance_hr_q02` | 0.4906 | 0.9269 | 0.0657 | 0.6025 |
| 10 | E26_final_candidates | `reject5` | 0.4669 | 0.8450 | 0.2433 | 0.6001 |
| 11 | E27_unknown_rejection | `reject5` | 0.4669 | 0.8450 | 0.2433 | 0.6001 |
| 12 | E34_covariance_hr | `covariance_hr_q01` | 0.4557 | 0.9487 | 0.0440 | 0.5993 |
| 13 | E35_per_task_analysis | `per_task_q02` | 0.4866 | 0.9353 | 0.0281 | 0.5964 |
| 14 | E35_per_task_analysis | `per_task_q05` | 0.5216 | 0.8964 | 0.0692 | 0.5959 |
| 15 | E30_dino_multiview | `cov5_wccn_q01` | 0.4804 | 0.9423 | 0.0135 | 0.5946 |
| 16 | E35_per_task_analysis | `per_task_q01` | 0.4510 | 0.9565 | 0.0141 | 0.5945 |
| 17 | E30_dino_multiview | `cov5_wccn+pca_q01` | 0.4860 | 0.9383 | 0.0140 | 0.5942 |
| 18 | E30_dino_multiview | `cov5_wccn+pca_q02` | 0.5038 | 0.9205 | 0.0312 | 0.5940 |
| 19 | E34_covariance_hr | `covariance_hr_q05` | 0.5493 | 0.8634 | 0.0992 | 0.5938 |
| 20 | E30_dino_multiview | `cov5_dist+wccn+pca_q01` | 0.4771 | 0.9421 | 0.0119 | 0.5933 |
| 21 | E30_dino_multiview | `cov5_dist+pca_q01` | 0.4738 | 0.9433 | 0.0124 | 0.5932 |
| 22 | E30_dino_multiview | `cov5_wccn_q02` | 0.5031 | 0.9207 | 0.0271 | 0.5929 |
| 23 | E35_per_task_analysis | `per_task_q10` | 0.5658 | 0.8289 | 0.1476 | 0.5928 |
| 24 | E30_dino_multiview | `cov5_dist+wccn_q01` | 0.4715 | 0.9428 | 0.0124 | 0.5923 |
| 25 | E29_dino_calibration | `q05` | 0.5285 | 0.8851 | 0.0681 | 0.5917 |
| 26 | E32_distance_ratio | `dist+ratio+gap_q10` | 0.5256 | 0.8788 | 0.0836 | 0.5917 |
| 27 | E30_dino_multiview | `cov5_wccn+pca_q05` | 0.5427 | 0.8711 | 0.0813 | 0.5915 |
| 28 | E30_dino_multiview | `cov5_dist+pca_q05` | 0.5387 | 0.8754 | 0.0758 | 0.5913 |
| 29 | E30_dino_multiview | `cov5_dist+pca_q02` | 0.4972 | 0.9213 | 0.0253 | 0.5913 |
| 30 | E30_dino_multiview | `cov5_dist+wccn_q02` | 0.4929 | 0.9239 | 0.0245 | 0.5913 |
| 31 | E30_dino_multiview | `cov5_dist+wccn+pca_q05` | 0.5394 | 0.8745 | 0.0759 | 0.5911 |
| 32 | E30_dino_multiview | `cov5_wccn_q05` | 0.5404 | 0.8725 | 0.0789 | 0.5911 |
| 33 | E30_dino_multiview | `cov5_dist+wccn+pca_q02` | 0.5021 | 0.9179 | 0.0262 | 0.5911 |
| 34 | E30_dino_multiview | `cov5_dist_q05` | 0.5295 | 0.8831 | 0.0685 | 0.5910 |
| 35 | E28_dino_openset | `cov5_reject5` | 0.5308 | 0.8820 | 0.0681 | 0.5907 |
| 36 | E30_dino_multiview | `cov5_dist+wccn_q05` | 0.5387 | 0.8737 | 0.0757 | 0.5904 |
| 37 | E30_dino_multiview | `cov5_dist_q02` | 0.4784 | 0.9295 | 0.0224 | 0.5899 |
| 38 | E28_dino_openset | `cov5_reject2` | 0.4784 | 0.9295 | 0.0219 | 0.5898 |
| 39 | E31_task_conditional | `global_global_q02` | 0.4787 | 0.9290 | 0.0219 | 0.5897 |
| 40 | E31_task_conditional | `task_model_taskcenter_q02` | 0.4817 | 0.9276 | 0.0216 | 0.5896 |
| 41 | E29_dino_calibration | `q02` | 0.4705 | 0.9328 | 0.0219 | 0.5895 |
| 42 | E31_task_conditional | `global_taskcenter_q05` | 0.5351 | 0.8763 | 0.0691 | 0.5892 |
| 43 | E31_task_conditional | `task_model_wccn_q01` | 0.5087 | 0.9166 | 0.0145 | 0.5891 |
| 44 | E32_distance_ratio | `dist_q01` | 0.5087 | 0.9166 | 0.0145 | 0.5891 |
| 45 | E35_per_task_analysis | `global_q01` | 0.5087 | 0.9166 | 0.0145 | 0.5891 |
| 46 | E31_task_conditional | `global_taskcenter_q02` | 0.4801 | 0.9275 | 0.0210 | 0.5890 |
| 47 | E31_task_conditional | `global_global_q05` | 0.5345 | 0.8765 | 0.0681 | 0.5889 |
| 48 | E31_task_conditional | `task_model_taskcenter_q05` | 0.5384 | 0.8724 | 0.0706 | 0.5885 |
| 49 | E31_task_conditional | `global_global_q01` | 0.4339 | 0.9534 | 0.0108 | 0.5879 |
| 50 | E31_task_conditional | `task_model_taskcenter_q01` | 0.4464 | 0.9469 | 0.0110 | 0.5878 |
| 51 | E30_dino_multiview | `cov5_dist_q01` | 0.4352 | 0.9521 | 0.0108 | 0.5876 |
| 52 | E28_dino_openset | `cov5_reject1` | 0.4352 | 0.9520 | 0.0108 | 0.5875 |
| 53 | E29_dino_calibration | `q01` | 0.4296 | 0.9548 | 0.0108 | 0.5875 |
| 54 | E31_task_conditional | `global_taskcenter_q01` | 0.4382 | 0.9505 | 0.0104 | 0.5874 |
| 55 | E28_dino_openset | `cov5_topo5_reject2` | 0.4784 | 0.9237 | 0.0218 | 0.5869 |
| 56 | E28_dino_openset | `cov5_topo5_reject1` | 0.4379 | 0.9467 | 0.0108 | 0.5855 |
| 57 | E31_task_conditional | `task_model_wccn_q02` | 0.5279 | 0.8901 | 0.0327 | 0.5852 |
| 58 | E32_distance_ratio | `dist_q02` | 0.5279 | 0.8901 | 0.0327 | 0.5852 |
| 59 | E35_per_task_analysis | `global_q02` | 0.5279 | 0.8901 | 0.0327 | 0.5852 |
| 60 | E30_dino_multiview | `cov5_dist+pca+topo5_dist_q10` | 0.5552 | 0.8066 | 0.1701 | 0.5847 |
| 61 | E28_dino_openset | `cov5_topo5_reject5` | 0.5331 | 0.8638 | 0.0764 | 0.5843 |
| 62 | E30_dino_multiview | `cov5_wccn+pca_q10` | 0.5823 | 0.7998 | 0.1422 | 0.5810 |
| 63 | E29_dino_calibration | `q10` | 0.5664 | 0.8127 | 0.1290 | 0.5802 |
| 64 | E32_distance_ratio | `dist+ratio_q01` | 0.5058 | 0.8936 | 0.0244 | 0.5793 |
| 65 | E32_distance_ratio | `dist_q05` | 0.5582 | 0.8405 | 0.0774 | 0.5791 |
| 66 | E34_covariance_hr | `covariance_hr_q10` | 0.6175 | 0.7680 | 0.1626 | 0.5790 |
| 67 | E31_task_conditional | `task_model_wccn_q05` | 0.5582 | 0.8405 | 0.0769 | 0.5790 |
| 68 | E35_per_task_analysis | `global_q05` | 0.5582 | 0.8405 | 0.0769 | 0.5790 |
| 69 | E30_dino_multiview | `cov5_dist+pca_q10` | 0.5783 | 0.7996 | 0.1375 | 0.5787 |
| 70 | E30_dino_multiview | `cov5_dist+wccn+pca_q10` | 0.5803 | 0.7966 | 0.1393 | 0.5782 |
| 71 | E26_final_candidates | `reject2` | 0.3093 | 0.9300 | 0.1415 | 0.5777 |
| 72 | E27_unknown_rejection | `reject2` | 0.3093 | 0.9300 | 0.1415 | 0.5777 |
| 73 | E30_dino_multiview | `cov5_wccn_q10` | 0.5852 | 0.7886 | 0.1413 | 0.5760 |
| 74 | E28_dino_openset | `cov5_topo5_reject10` | 0.5628 | 0.7956 | 0.1490 | 0.5757 |
| 75 | E30_dino_multiview | `cov5_dist+wccn_q10` | 0.5783 | 0.7939 | 0.1367 | 0.5757 |
| 76 | E30_dino_multiview | `cov5_dist+pca+topo5_dist_q05` | 0.4418 | 0.8810 | 0.0979 | 0.5755 |
| 77 | E29_dino_calibration | `q30` | 0.6502 | 0.6198 | 0.4108 | 0.5751 |
| 78 | E28_dino_openset | `cov5_reject10` | 0.5740 | 0.7983 | 0.1290 | 0.5749 |
| 79 | E30_dino_multiview | `cov5_dist_q10` | 0.5740 | 0.7983 | 0.1290 | 0.5749 |
| 80 | E32_distance_ratio | `dist+ratio_q02` | 0.5516 | 0.8442 | 0.0587 | 0.5747 |
| 81 | E29_dino_calibration | `q20` | 0.6182 | 0.7060 | 0.2677 | 0.5745 |
| 82 | E30_dino_multiview | `cov5_dist+pca+topo5_dist_q20` | 0.6380 | 0.6673 | 0.3177 | 0.5726 |
| 83 | E30_dino_multiview | `cov5+topo5_dist_q10` | 0.4883 | 0.8121 | 0.1768 | 0.5723 |
| 84 | E31_task_conditional | `global_global_q10` | 0.5773 | 0.7885 | 0.1290 | 0.5708 |
| 85 | E29_dino_calibration | `q40` | 0.6904 | 0.5250 | 0.5406 | 0.5702 |
| 86 | E30_dino_multiview | `cov5_wccn+pca_q20` | 0.6591 | 0.6391 | 0.3320 | 0.5673 |
| 87 | E31_task_conditional | `global_taskcenter_q10` | 0.5869 | 0.7704 | 0.1373 | 0.5662 |
| 88 | E30_dino_multiview | `cov5+topo5_dist_q20` | 0.6093 | 0.6741 | 0.3066 | 0.5660 |
| 89 | E31_task_conditional | `task_model_taskcenter_q10` | 0.5869 | 0.7731 | 0.1301 | 0.5658 |
| 90 | E30_dino_multiview | `cov5_wccn_q20` | 0.6634 | 0.6237 | 0.3524 | 0.5658 |
| 91 | E29_dino_calibration | `q50` | 0.7293 | 0.4334 | 0.6479 | 0.5610 |
| 92 | E30_dino_multiview | `cov5_dist+wccn+pca_q20` | 0.6515 | 0.6346 | 0.3219 | 0.5606 |
| 93 | E28_dino_openset | `cov5_topo5_reject20` | 0.6189 | 0.6776 | 0.2669 | 0.5603 |
| 94 | E30_dino_multiview | `cov5_dist+pca_q20` | 0.6439 | 0.6435 | 0.3038 | 0.5587 |
| 95 | E26_final_candidates | `reject70` | 0.9898 | 0.1326 | 0.9794 | 0.5586 |
| 96 | E30_dino_multiview | `cov5_wccn+pca_q30` | 0.7155 | 0.5159 | 0.4839 | 0.5578 |
| 97 | E26_final_candidates | `reject1` | 0.2074 | 0.9653 | 0.0915 | 0.5574 |
| 98 | E27_unknown_rejection | `reject1` | 0.2074 | 0.9653 | 0.0915 | 0.5574 |
| 99 | E30_dino_multiview | `cov5+topo5_dist_q05` | 0.3459 | 0.8881 | 0.1003 | 0.5556 |
| 100 | E30_dino_multiview | `cov5_wccn_q30` | 0.7303 | 0.4875 | 0.5161 | 0.5554 |
| 101 | E30_dino_multiview | `cov5_dist+wccn_q20` | 0.6525 | 0.6259 | 0.3165 | 0.5552 |
| 102 | E30_dino_multiview | `cov5_dist_q20` | 0.6337 | 0.6579 | 0.2677 | 0.5543 |
| 103 | E28_dino_openset | `cov5_reject20` | 0.6337 | 0.6578 | 0.2677 | 0.5543 |
| 104 | E30_dino_multiview | `cov5_dist+pca+topo5_dist_q02` | 0.2872 | 0.9382 | 0.0517 | 0.5538 |
| 105 | E30_dino_multiview | `cov5_dist+wccn+pca_q30` | 0.7138 | 0.5132 | 0.4739 | 0.5535 |
| 106 | E30_dino_multiview | `cov5_dist+pca_q30` | 0.7010 | 0.5310 | 0.4495 | 0.5531 |
| 107 | E30_dino_multiview | `cov5_dist+pca+topo5_dist_q30` | 0.6878 | 0.5341 | 0.4489 | 0.5512 |
| 108 | E31_task_conditional | `global_global_q20` | 0.6409 | 0.6477 | 0.2677 | 0.5510 |
| 109 | E33_arcface_head | `arcface_m0.0_d0.5_q05` | 0.4504 | 0.7908 | 0.1690 | 0.5502 |
| 110 | E30_dino_multiview | `cov5+topo5_dist_q30` | 0.6805 | 0.5386 | 0.4302 | 0.5470 |
| 111 | E30_dino_multiview | `cov5_dist+wccn_q30` | 0.7095 | 0.5027 | 0.4728 | 0.5469 |
| 112 | E28_dino_openset | `cov5_topo5_reject30` | 0.6874 | 0.5383 | 0.4204 | 0.5461 |
| 113 | E31_task_conditional | `task_model_wccn_q10` | 0.6195 | 0.7065 | 0.1477 | 0.5450 |
| 114 | E32_distance_ratio | `dist_q10` | 0.6195 | 0.7065 | 0.1477 | 0.5450 |
| 115 | E35_per_task_analysis | `global_q10` | 0.6195 | 0.7065 | 0.1477 | 0.5450 |
| 116 | E33_arcface_head | `arcface_m0.0_d0.5_q02` | 0.3699 | 0.8688 | 0.0706 | 0.5446 |
| 117 | E30_dino_multiview | `cov5_dist_q30` | 0.6815 | 0.5419 | 0.4108 | 0.5440 |
| 118 | E28_dino_openset | `cov5_reject30` | 0.6815 | 0.5419 | 0.4108 | 0.5440 |
| 119 | E30_dino_multiview | `cov5+topo5_dist_q02` | 0.2367 | 0.9362 | 0.0562 | 0.5413 |
| 120 | E33_arcface_head | `arcface_m0.0_d0.5_q10` | 0.5562 | 0.6543 | 0.2983 | 0.5408 |
| 121 | E33_arcface_head | `arcface_m0.0_d0.5_q20` | 0.6963 | 0.4667 | 0.5320 | 0.5404 |
| 122 | E31_task_conditional | `task_model_taskcenter_q20` | 0.6578 | 0.6034 | 0.2955 | 0.5400 |
| 123 | E31_task_conditional | `global_taskcenter_q20` | 0.6588 | 0.6062 | 0.2871 | 0.5396 |
| 124 | E32_distance_ratio | `dist+ratio+gap_q20` | 0.6769 | 0.5989 | 0.2794 | 0.5385 |
| 125 | E33_arcface_head | `arcface_m0.0_d0.5_q01` | 0.2694 | 0.9256 | 0.0314 | 0.5380 |
| 126 | E34_covariance_hr | `covariance_hr_q20` | 0.6960 | 0.5940 | 0.2642 | 0.5371 |
| 127 | E32_distance_ratio | `dist+ratio_q05` | 0.6736 | 0.6445 | 0.1837 | 0.5366 |
| 128 | E30_dino_multiview | `cov5_dist+pca+topo5_dist_q01` | 0.1850 | 0.9645 | 0.0317 | 0.5364 |
| 129 | E30_dino_multiview | `cov5+topo5_dist_q01` | 0.1405 | 0.9637 | 0.0324 | 0.5251 |
| 130 | E32_distance_ratio | `ratio_q01` | 0.6044 | 0.7222 | 0.0503 | 0.5248 |
| 131 | E26_final_candidates | `reject85` | 0.9987 | 0.0389 | 0.9946 | 0.5178 |
| 132 | E32_distance_ratio | `dist_q20` | 0.7537 | 0.4713 | 0.3648 | 0.5153 |
| 133 | E31_task_conditional | `task_model_wccn_q20` | 0.7537 | 0.4713 | 0.3630 | 0.5148 |
| 134 | E35_per_task_analysis | `global_q20` | 0.7537 | 0.4713 | 0.3630 | 0.5148 |
| 135 | E32_distance_ratio | `ratio+gap_q20` | 0.0861 | 0.9587 | 0.0499 | 0.5134 |
| 136 | E32_distance_ratio | `dist+ratio+gap_q05` | 0.0719 | 0.9773 | 0.0100 | 0.5091 |
| 137 | E32_distance_ratio | `ratio+gap_q10` | 0.0630 | 0.9742 | 0.0110 | 0.5056 |
| 138 | E32_distance_ratio | `dist+ratio+gap_q02` | 0.0465 | 0.9845 | 0.0010 | 0.5041 |
| 139 | E32_distance_ratio | `dist+ratio+gap_q01` | 0.0399 | 0.9877 | 0.0000 | 0.5038 |
| 140 | E32_distance_ratio | `ratio+gap_q05` | 0.0508 | 0.9806 | 0.0019 | 0.5035 |
| 141 | E32_distance_ratio | `ratio+gap_q02` | 0.0422 | 0.9853 | 0.0000 | 0.5032 |
| 142 | E32_distance_ratio | `ratio+gap_q01` | 0.0363 | 0.9883 | 0.0000 | 0.5032 |
| 143 | E26_final_candidates | `reject95` | 1.0000 | 0.0037 | 1.0000 | 0.5018 |
| 144 | E32_distance_ratio | `dist+gap_q20` | 0.0382 | 0.9805 | 0.0062 | 0.5014 |
| 145 | E32_distance_ratio | `dist+gap_q10` | 0.0280 | 0.9879 | 0.0013 | 0.5013 |
| 146 | E32_distance_ratio | `dist+gap_q05` | 0.0214 | 0.9912 | 0.0000 | 0.5009 |
| 147 | E32_distance_ratio | `ratio_q20` | 0.9937 | 0.0697 | 0.8687 | 0.5005 |
| 148 | E32_distance_ratio | `dist+gap_q02` | 0.0106 | 0.9953 | 0.0000 | 0.5003 |
| 149 | E32_distance_ratio | `gap_q20` | 0.0142 | 0.9932 | 0.0000 | 0.5002 |
| 150 | E32_distance_ratio | `dist+gap_q01` | 0.0076 | 0.9962 | 0.0000 | 0.5000 |
| 151 | E32_distance_ratio | `gap_q01` | 0.0000 | 0.9999 | 0.0000 | 0.5000 |
| 152 | E32_distance_ratio | `gap_q02` | 0.0003 | 0.9996 | 0.0000 | 0.4999 |
| 153 | E32_distance_ratio | `gap_q05` | 0.0040 | 0.9977 | 0.0000 | 0.4998 |
| 154 | E32_distance_ratio | `gap_q10` | 0.0073 | 0.9957 | 0.0000 | 0.4997 |
| 155 | E32_distance_ratio | `dist+ratio_q10` | 0.8391 | 0.3628 | 0.3902 | 0.4887 |
| 156 | E32_distance_ratio | `dist+ratio_q20` | 0.9512 | 0.1502 | 0.6991 | 0.4877 |
| 157 | E32_distance_ratio | `ratio_q10` | 0.9782 | 0.1513 | 0.6103 | 0.4728 |
| 158 | E32_distance_ratio | `ratio_q02` | 0.8071 | 0.4788 | 0.1203 | 0.4713 |
| 159 | E32_distance_ratio | `ratio_q05` | 0.9172 | 0.2827 | 0.3428 | 0.4563 |

## 4. 是否超过 0.7171

- 最高 clean W = **0.6394**（E27_unknown_rejection / `reject30`）。
- 超过 0.7171 的 clean 候选数量：**0**。
- 结论：这批老实验（E26–E35）**没有一个** clean 候选能超过 0.7171；最高值还差 0.0777。

原因：

- 这批实验的最优工作点都在 S2 接受率 0.6–0.9 之间，而 LOSO 拒识与 S1 拒识高度耦合（Test AUROC 只有 0.65–0.74）。要拿到接近 0.7171 的 W，需要在 S2 接受 0.78 附近同时有 LOSO ≈ 0.69、S1 ≈ 0.61；本批次任何候选都达不到这个组合。
- E26 / E27 的 W 最高（0.6394 = E27 reject30；0.6370 = E26/E27 reject20），靠的是协方差原始特征 + SNR30 增广，代价是 S2 接受率被压到 0.47（reject30），与 0.7171 方案（S2 0.7813）不在同一工作区。
- DINO 系列（E28–E35）的 LOSO 只有 0.01–0.53：同会话留出身份与注册者的距离分布重叠严重，即使 Test 侧 S1 拒识不错，LOSO 端也拉不起来。
- E32 的 `gap` 单用（W ≈ 0.50）与所有 `*gap` 组合是方向错误导致的退化，不构成有效方案。

## 5. 无法补算的实验

- 无。10 个实验的特征缓存（`BrainPrint_cache/train_features.npy`、`BrainPrint_cache/e25_robust/train_cov_snr30.npy`、`BrainPrint_cache/eeg_image_dinov2/*_dinov2.npy`）与代码均存在，全部补算完成。

## 6. 复核要点

- 所有候选的 CSV S1 / S2 与实验 JSON 记录逐项对比（`reported diff` 均为 0），说明 CSV 与报告的对应关系正确。
- E26 与 E27 的 `reject1/2/5/20/50` 是同一管线（CSV 内容相同），不是独立方案。
- E32 的 `gap` 方向与原脚本一致（越大越像已知），因此 `gap` 单用退化为「全部接受」，W ≈ 0.50 属恒等式产物，不具判别意义。
- E30 `cov5+topo5_dist` 及 E32 的 `*gap` 组合的低 W 是融合方向错误，不是可信提升。

运行耗时：215.2s；候选总数 159；日志 `run.log`。

