# Session-blind shift calibration：严格嵌套五折结果

## 协议
独立重跑模型并使用 CUDA。每个外层折的 direct/base 分数和 conditional 阈值只由外层训练折的内层预测拟合；外层评估标签不参与任何选择。conditional 为 base max_logit 按训练折质量信号三桶分层的 5% 分位阈值。

## 五折宏平均
| 方法 | C | U | unknown AUC | unknown F1 | macro-F1 | known acceptance | 官方公式代理分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 0.9980 | 0.2032 | 0.9890 | 0.3193 | 0.7115 | 0.9999 | 64.09 |
| base | 0.9988 | 0.2216 | 0.9892 | 0.3354 | 0.7223 | 1.0000 | 64.82 |
| conditional | 0.9987 | 0.2220 | 0.9892 | 0.3359 | 0.7227 | 0.9999 | 64.80 |

## Test 说明
Test AUC/F1 不可计算：测试集没有公开标签。本轮未生成 Test CSV，也未改变正式提交文件。

## 门槛判断
本实验只覆盖 clean/unknown 维度，官方代理分缺少 RA/ASafeR，不能替代官方总分；conditional 未达到提交所需的完整验收证据，因此本轮不通过，不生成候选 CSV。
