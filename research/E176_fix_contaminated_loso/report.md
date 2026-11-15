# E176 修正被污染的 LOSO：九个实验的重测

**目标口径**：`W = 0.25×LOSO 拒识 + 0.25×Test S1 拒识 + 0.5×Test S2 接受`。

**Bug**：留一身份时，19 个注册身份的评估块（0.6<frac≤0.8）也被拿去建了原型，被测 trial 出现在自己的原型里 ⇒ known 距离趋近 0 ⇒ 阈值被抬高 ⇒ LOSO 拒识虚高。

```python
# 错误
p2, k2 = build_prototypes(a, subjects, tasks, pm, enrolled, k)
# 正确
p2, k2 = build_prototypes(a, subjects, tasks, pm & fit_rows, enrolled, k)
```

Test S1/S2 不经过原型集合，因此**未受影响**；本次按各实验原始配置重建 Test 管线，并在原始接受率上复现 S1/S2（容差 0.005）后，才用同一管线的修正 LOSO 计算 W。

## 1. 修正前后对照大表

| 实验 | 配置 | a | S1 拒识 | S2 接受 | LOSO（污染） | LOSO（修正） | W（污染） | W（修正） | 虚高 | 标记 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.80 | 0.8592 | 0.4469 | 1.0000 | 0.7294 | 0.6883 | 0.6206 | +0.0676 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.85 | 0.7939 | 0.5417 | 1.0000 | 0.6112 | 0.7193 | 0.6222 | +0.0972 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.88 | 0.7352 | 0.6177 | 0.9996 | 0.5378 | 0.7425 | 0.6271 | +0.1155 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.90 | 0.7023 | 0.6689 | 0.9971 | 0.4664 | 0.7593 | 0.6266 | +0.1327 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.91 | 0.6861 | 0.6950 | 0.9954 | 0.4145 | 0.7679 | 0.6226 | +0.1452 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.92 | 0.6647 | 0.7242 | 0.9895 | 0.3709 | 0.7756 | 0.6210 | +0.1547 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.93 | 0.6400 | 0.7525 | 0.9792 | 0.3179 | 0.7810 | 0.6157 | +0.1653 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.94 | 0.6245 | 0.7774 | 0.9649 | 0.2639 | 0.7861 | 0.6108 | +0.1753 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.95 | 0.5987 | 0.8022 | 0.9475 | 0.2135 | 0.7877 | 0.6042 | +0.1835 | corrected |
| E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.96 | 0.5836 | 0.8216 | 0.9044 | 0.1790 | 0.7828 | 0.6014 | +0.1814 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.80 | 0.9354 | 0.2839 | 1.0000 | 0.9291 | 0.6258 | 0.6081 | +0.0177 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.85 | 0.8754 | 0.3907 | 0.9995 | 0.8523 | 0.6641 | 0.6273 | +0.0368 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.88 | 0.8256 | 0.4798 | 0.9991 | 0.7789 | 0.6961 | 0.6410 | +0.0551 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.90 | 0.7804 | 0.5468 | 0.9987 | 0.7152 | 0.7182 | 0.6473 | +0.0709 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.91 | 0.7587 | 0.5813 | 0.9983 | 0.6723 | 0.7299 | 0.6484 | +0.0815 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.92 | 0.7310 | 0.6184 | 0.9979 | 0.6313 | 0.7414 | 0.6498 | +0.0916 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.93 | 0.6947 | 0.6584 | 0.9958 | 0.5731 | 0.7518 | 0.6462 | +0.1057 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.94 | 0.6686 | 0.6960 | 0.9946 | 0.5114 | 0.7638 | 0.6430 | +0.1208 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.95 | 0.6360 | 0.7429 | 0.9876 | 0.4428 | 0.7774 | 0.6412 | +0.1362 | corrected |
| E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.96 | 0.5997 | 0.7888 | 0.9724 | 0.3784 | 0.7875 | 0.6390 | +0.1485 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.80 | 0.9459 | 0.3431 | 1.0000 | 0.9640 | 0.6580 | 0.6490 | +0.0090 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.85 | 0.8968 | 0.4504 | 0.9990 | 0.9150 | 0.6992 | 0.6782 | +0.0210 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.88 | 0.8454 | 0.5326 | 0.9976 | 0.8523 | 0.7270 | 0.6907 | +0.0363 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.90 | 0.7979 | 0.5960 | 0.9963 | 0.7936 | 0.7465 | 0.6958 | +0.0507 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.91 | 0.7732 | 0.6228 | 0.9940 | 0.7632 | 0.7532 | 0.6955 | +0.0577 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.92 | 0.7352 | 0.6576 | 0.9891 | 0.7270 | 0.7599 | 0.6944 | +0.0655 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.93 | 0.7056 | 0.6876 | 0.9795 | 0.6859 | 0.7651 | 0.6917 | +0.0734 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.94 | 0.6640 | 0.7189 | 0.9688 | 0.6520 | 0.7676 | 0.6884 | +0.0792 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.95 | 0.6129 | 0.7543 | 0.9395 | 0.6002 | 0.7652 | 0.6804 | +0.0848 | corrected |
| E139_feature_fusion | A_E65（PCA96, k=12） | 0.96 | 0.5572 | 0.7909 | 0.9172 | 0.5340 | 0.7641 | 0.6683 | +0.0958 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.80 | 0.9420 | 0.3541 | 1.0000 | 0.9509 | 0.6626 | 0.6503 | +0.0123 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.85 | 0.8833 | 0.4618 | 1.0000 | 0.9085 | 0.7017 | 0.6789 | +0.0229 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.88 | 0.8315 | 0.5404 | 0.9990 | 0.8554 | 0.7278 | 0.6919 | +0.0359 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.90 | 0.7910 | 0.5935 | 0.9981 | 0.8021 | 0.7440 | 0.6950 | +0.0490 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.91 | 0.7623 | 0.6243 | 0.9972 | 0.7692 | 0.7520 | 0.6950 | +0.0570 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.92 | 0.7405 | 0.6503 | 0.9962 | 0.7340 | 0.7593 | 0.6938 | +0.0656 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.93 | 0.7151 | 0.6810 | 0.9934 | 0.7084 | 0.7677 | 0.6964 | +0.0713 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.94 | 0.6848 | 0.7099 | 0.9872 | 0.6756 | 0.7729 | 0.6951 | +0.0779 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.95 | 0.6370 | 0.7462 | 0.9810 | 0.6181 | 0.7776 | 0.6869 | +0.0907 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.96 | 0.5763 | 0.7866 | 0.9687 | 0.5435 | 0.7795 | 0.6732 | +0.1063 | corrected |
| E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.97 | 0.5124 | 0.8290 | 0.9481 | 0.4735 | 0.7796 | 0.6610 | +0.1186 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.80 | 0.9275 | 0.3731 | 0.9941 | 0.9279 | 0.6669 | 0.6504 | +0.0165 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.85 | 0.8665 | 0.4820 | 0.9867 | 0.8466 | 0.7043 | 0.6693 | +0.0350 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.88 | 0.8051 | 0.5585 | 0.9649 | 0.7803 | 0.7217 | 0.6756 | +0.0461 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.90 | 0.7656 | 0.6044 | 0.9451 | 0.7311 | 0.7299 | 0.6764 | +0.0535 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.91 | 0.7468 | 0.6312 | 0.9347 | 0.6993 | 0.7360 | 0.6771 | +0.0588 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.92 | 0.7158 | 0.6586 | 0.9176 | 0.6506 | 0.7376 | 0.6709 | +0.0667 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.93 | 0.6861 | 0.6884 | 0.8924 | 0.6064 | 0.7388 | 0.6673 | +0.0715 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.94 | 0.6571 | 0.7226 | 0.8711 | 0.5730 | 0.7433 | 0.6688 | +0.0745 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.95 | 0.6159 | 0.7597 | 0.8244 | 0.5242 | 0.7399 | 0.6649 | +0.0751 | corrected |
| E138_scale_free_score | form=d1 (k=3, PCA96) | 0.96 | 0.5658 | 0.8025 | 0.7666 | 0.4740 | 0.7343 | 0.6612 | +0.0731 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.80 | 0.8849 | 0.3404 | 0.9995 | 0.9720 | 0.6413 | 0.6344 | +0.0069 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.85 | 0.7794 | 0.4615 | 0.9974 | 0.9347 | 0.6750 | 0.6593 | +0.0157 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.88 | 0.6637 | 0.5509 | 0.9953 | 0.8831 | 0.6902 | 0.6622 | +0.0281 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.90 | 0.5740 | 0.6133 | 0.9902 | 0.8334 | 0.6977 | 0.6585 | +0.0392 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.91 | 0.5199 | 0.6398 | 0.9864 | 0.8076 | 0.6965 | 0.6518 | +0.0447 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.92 | 0.4675 | 0.6705 | 0.9797 | 0.7819 | 0.6970 | 0.6476 | +0.0494 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.93 | 0.4131 | 0.7048 | 0.9712 | 0.7454 | 0.6985 | 0.6420 | +0.0564 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.94 | 0.3597 | 0.7356 | 0.9513 | 0.7000 | 0.6956 | 0.6327 | +0.0628 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.95 | 0.3112 | 0.7655 | 0.9195 | 0.6584 | 0.6905 | 0.6252 | +0.0653 | corrected |
| E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.96 | 0.2526 | 0.8048 | 0.8645 | 0.5928 | 0.6817 | 0.6137 | +0.0679 | corrected |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.80 | 0.3851 | 0.8901 | 0.6237 | 0.6237 | 0.6973 | 0.6973 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.85 | 0.3647 | 0.8971 | 0.5421 | 0.5421 | 0.6752 | 0.6752 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.88 | 0.3501 | 0.9032 | 0.4769 | 0.4769 | 0.6584 | 0.6584 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.90 | 0.3379 | 0.9085 | 0.4255 | 0.4255 | 0.6451 | 0.6451 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.91 | 0.3333 | 0.9106 | 0.3969 | 0.3969 | 0.6379 | 0.6379 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.92 | 0.3221 | 0.9127 | 0.3661 | 0.3661 | 0.6284 | 0.6284 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.93 | 0.3149 | 0.9146 | 0.3322 | 0.3322 | 0.6191 | 0.6191 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.94 | 0.3004 | 0.9185 | 0.2941 | 0.2941 | 0.6078 | 0.6078 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.95 | 0.2938 | 0.9210 | 0.2585 | 0.2585 | 0.5986 | 0.5986 | +0.0000 | pipeline_mismatch |
| E140_metric_learning | PCA128_raw_control (k=3, fit 原型) | 0.96 | 0.2756 | 0.9263 | 0.2123 | 0.2123 | 0.5851 | 0.5851 | +0.0000 | pipeline_mismatch |
| E136_clean_three_way | P96_k3 | 0.80 | 0.9773 | 0.2374 | 0.9046 | 0.9046 | 0.5892 | 0.5892 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.85 | 0.9390 | 0.3495 | 0.8301 | 0.8301 | 0.6170 | 0.6170 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.88 | 0.8978 | 0.4383 | 0.7641 | 0.7641 | 0.6347 | 0.6347 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.90 | 0.8612 | 0.5008 | 0.6954 | 0.6954 | 0.6396 | 0.6396 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.91 | 0.8342 | 0.5369 | 0.6576 | 0.6576 | 0.6414 | 0.6414 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.92 | 0.8121 | 0.5679 | 0.6246 | 0.6246 | 0.6431 | 0.6431 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.93 | 0.7794 | 0.6124 | 0.5888 | 0.5888 | 0.6483 | 0.6483 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.94 | 0.7441 | 0.6515 | 0.5440 | 0.5440 | 0.6478 | 0.6478 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.95 | 0.7043 | 0.6930 | 0.4819 | 0.4819 | 0.6431 | 0.6431 | +0.0000 | corrected |
| E136_clean_three_way | P96_k3 | 0.96 | 0.6568 | 0.7413 | 0.4239 | 0.4239 | 0.6408 | 0.6408 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.80 | 0.8734 | 0.4690 | 0.7248 | 0.7248 | 0.6340 | 0.6340 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.85 | 0.7933 | 0.5872 | 0.6261 | 0.6261 | 0.6484 | 0.6484 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.88 | 0.7194 | 0.6657 | 0.5512 | 0.5512 | 0.6505 | 0.6505 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.90 | 0.6630 | 0.7138 | 0.4886 | 0.4886 | 0.6448 | 0.6448 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.91 | 0.6317 | 0.7402 | 0.4518 | 0.4518 | 0.6410 | 0.6410 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.92 | 0.5958 | 0.7664 | 0.4326 | 0.4326 | 0.6403 | 0.6403 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.93 | 0.5687 | 0.7914 | 0.4020 | 0.4020 | 0.6384 | 0.6384 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.94 | 0.5437 | 0.8175 | 0.3629 | 0.3629 | 0.6354 | 0.6354 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.95 | 0.5104 | 0.8459 | 0.3121 | 0.3121 | 0.6286 | 0.6286 | +0.0000 | corrected |
| E133_mean_objective | G_global（k=1 均值原型） | 0.96 | 0.4662 | 0.8752 | 0.2598 | 0.2598 | 0.6191 | 0.6191 | +0.0000 | corrected |

> `pipeline_mismatch` 的行保留原值仅供对照，**不参与排名**，其 W 不可信。

## 2. 按修正后 W 降序排名（每个实验取最好的接受率）

| 排名 | 实验 | 配置 | a | S1 | S2 | LOSO（修正） | W（修正） | W（污染，同点） | 虚高 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | E141_weighted_image_route | score:B_E65logspd (k=32, 单视图, per-task z) | 0.93 | 0.7151 | 0.6810 | 0.7084 | **0.6964** | 0.7677 | +0.0713 |
| 2 | E139_feature_fusion | A_E65（PCA96, k=12） | 0.90 | 0.7979 | 0.5960 | 0.7936 | **0.6958** | 0.7465 | +0.0507 |
| 3 | E138_scale_free_score | form=d1 (k=3, PCA96) | 0.91 | 0.7468 | 0.6312 | 0.6993 | **0.6771** | 0.7360 | +0.0588 |
| 4 | E137_multi_view | greedy=E21_covdz_k3+E65_logspd_k3 | 0.88 | 0.6637 | 0.5509 | 0.8831 | **0.6622** | 0.6902 | +0.0281 |
| 5 | E133_mean_objective | G_global（k=1 均值原型） | 0.88 | 0.7194 | 0.6657 | 0.5512 | **0.6505** | 0.6505 | +0.0000 |
| 6 | E139_feature_fusion | C_plus_cov（PCA96, k=32） | 0.92 | 0.7310 | 0.6184 | 0.6313 | **0.6498** | 0.7414 | +0.0916 |
| 7 | E136_clean_three_way | P96_k3 | 0.93 | 0.7794 | 0.6124 | 0.5888 | **0.6483** | 0.6483 | +0.0000 |
| 8 | E142_cdf_fusion | rawz:B_plus_cleanA (k=32, 6 views, Train-only z) | 0.88 | 0.7352 | 0.6177 | 0.5378 | **0.6271** | 0.7425 | +0.1155 |

## 3. 修正后是否还有 W > 0.7171 的实验？

**没有。** 修正后没有任何一个工作点的 W 超过 0.7171。当前最高是 E141_weighted_image_route 的 score:B_E65logspd (k=32, 单视图, per-task z) @a=0.93，W = 0.6964。

## 4. 各实验的自检与标记

| 实验 | Test S1/S2 复现 | LOSO 污染值复现 | 结论 | 说明 |
|---|---|---|---|---|
| E142_cdf_fusion | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | CDF 参考分布取 Train 5 折 out-of-fold 分数（ref_raw），不是 Test 侧统计，未标 test_statistic；最佳工作点用的是 rawz，不涉及 CDF。 |
| E139_feature_fusion | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | - |
| E139_feature_fusion | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | - |
| E141_weighted_image_route | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | 该实验里 DINOv2 图像视图的 W 都低于纯 EEG 的 E65 log-SPD 单视图，最佳点其实是纯 EEG 分数。；该实验未使用 CDF 映射，无 Test 侧统计。 |
| E138_scale_free_score | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | - |
| E137_multi_view | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | greedy 视图选择本身是用被污染的 LOSO AUC 做的，此处沿用原始选择以保持配置可比。；融合诊断 AUC 只有 0.6408，S2 接受率在低阈值区间才会高。 |
| E140_metric_learning | ΔS1=0.4810, ΔS2=0.3627 | ΔLOSO=0.2883 | **pipeline_mismatch** | 距离管线可精确复现：重建的 Test raw AUC = 0.7570182995（E140 记录 test_auc_raw = 0.7605860288）。；但记录里的 z 空间校准（per-task st）与 LOSO 无法复现：S1 记录 0.8190 / 重建 0.3379，LOSO 记录 0.7138 / 重建 0.4255，同一份 E140 自带 helper 函数在相同 embedding 上也给不出记录值，说明存档行与当前源码不一致 ⇒ 按规则判 pipeline_mismatch，记录 W 0.6561 不可验证。；E140 源码的 LOSO 原型本身是 `enrolled & fit_mask`，不存在本实验要修的那个 bug。 |
| E136_clean_three_way | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | E136 源码里 LOSO 原型已经是 `enrolled & frac<=0.6`，没有被污染；本次重算确认 before==after。；S1/S2 用日志距离阈值，ref 为 Train 5 折 OOF。 |
| E133_mean_objective | ΔS1=0.0000, ΔS2=0.0000 | ΔLOSO=0.0000 | **corrected** | E133 只记录 (LOSO + S2)/2，没有记录 S1；此处用重建管线的 S1 （已在同一阈值上复现记录的 S2 与 LOSO）。；E133 源码的 LOSO 原型已是 `enrolled & frac<=0.6`，未被污染。；除 G_global 外的其它配置用的是逐身份分位阈值，与全局接受率网格不可比。 |
| E119_score_battery | - | - | **skipped** | 纯 Test 诊断实验：没有 LOSO 折、没有接受率网格、也没有 Train 侧阈值规则；其记录的 best_S2_accept_at_S1_rej_0.80 是扫 Test 阈值得到的 oracle 点。重算需要把 22251 条 Test raw EEG 重新喂进 E95 EEGNet + E65 特征提取（BrainPrint_cache 里没有对应缓存），成本过高，判 skipped。 |

## 5. 结论

- 九个被点名实验里，**真正存在该 bug 的是 5 个**：E142_cdf_fusion、E139_feature_fusion、E141_weighted_image_route、E138_scale_free_score、E137_multi_view（共 6 个工作点配置）。
- **误报 2 个**：E136_clean_three_way、E133_mean_objective 的源码本来就用了 `enrolled & frac<=0.6` 建原型，重算后污染值与修正值完全相同（E144 的自动清单按目录名打标，没有逐行核对源码）。
- **无法复现**：E140_metric_learning（见第 4 节，Test S1/S2 或 LOSO 与原记录不一致，按规则停止该配置）。
- 修正后最高 W = **0.6964**（E141_weighted_image_route / score:B_E65logspd (k=32, 单视图, per-task z) @a=0.93）；修正前同名表格里的最高记录是 0.7877。
- 虚高幅度（同一配置、同一接受率下 W_before − W_after）在 +0.0069 ~ +0.1835 之间；被污染时 LOSO 被判成 0.95~1.00，修正后回落到 0.48~0.88。
- 复核 E145 的独立结论：E139 的 A_E65 k=12 @a=0.90 修正后 LOSO=0.7936、S1=0.7979、S2=0.5960、W=0.6958，与 E145/E143 的 B_only 完全一致。
- 修正后 LOSO 与 Test S1 首次处在同一量级，Test S1/S2 与 LOSO 的负相关（E134 的发现）依然存在：接受率越高 S2 越好、LOSO 越差，W 峰值仍被压在 0.70 附近。

项目 Test 官方 AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。

