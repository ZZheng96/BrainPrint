# E238 多视图身份融合 + 一致性门控

> **一句话结论**：图像分支参与身份判定**无效**：唯一数值上达标的 `vote_weighted` 与 `eeg_only` 逐行相同（0.5 ≥ 3×1/6 的代数退化），其余候选 LOSO 闭集 macro-F1 都低于 eeg_only 的 0.9779，身份方案保持 `eeg_only`。 一致性门控的 S2@S1.95 = 0.5119，未超过 0.8776，**门控不冻结**（保留 E237 的 `gate_score_fused`，同读法 S2 = 0.8776）。

口径：EEG 分支是 E233 `mean_align` 的冻结产物（E227 deploy 表征 + 对称批次均值对齐，pred 逐行复现 E233）；图像分支沿用 E149 像素视图 + E237 的打分流程（fit 行逐 task 标准化 → PCA96 → 每 (task, identity) k=12 原型 → 负最近距离 → 同一套批次均值对齐）。四个分支现在都输出身份（`id_b`、`d1_b`、`d2_b`），身份不再只来自 EEG 分支。

**官方口径提醒**：Test 无身份标签，**C / RA / ASafeR 与官方身份 macro-F1 仍为 N/A**。本报告所有 macro-F1 都是**训练域闭集值**（20 折 LOSO 折内，held identity 已排除；或 fit 行 inner 5-block OOF），不是 Test 值。

## 1. 复现检查

|项目|本轮实测|参考|判定|
|---|---:|---:|:---:|
|EEG 分支 Test pred 与 E233 逐行一致|True|True|通过|
|EEG 分支 Test 分数最大绝对差|2.67e-05|0|通过|
|`gate_score_fused` 复现 E237 S2 / S2@S1.95|0.2906 / 0.8776|0.2906 / 0.8776|通过|
|融合分数与 E237 缓存最大绝对差|1.43e-06|0|校验|

## 2. 四个分支的身份能力（无标签，训练域闭集）

`LOSO 闭集`：20 折，每折只用已注册（enrolled）身份的 fit 行原型，在评估窗（trial 比例 (0.6,0.8]）的**已注册行**上算 macro-F1 / accuracy；held identity 的行不参与。`OOF 闭集`：fit 行 inner 5-block OOF，20 个身份全部参与。`Test AUC`：S2-vs-S1 的对齐后距离分离度（-d1），只是行为指标。

|分支|Test AUC(对齐后)|LOSO macro-F1|±std|LOSO acc|OOF macro-F1|d1 均值|(d2-d1) 均值|
|---|---:|---:|---:|---:|---:|---:|---:|
|EEG|0.8795|0.9779|0.002|0.9784|0.9898|28.743|3.209|
|X_mw_cov4|0.9353|0.4175|0.012|0.4235|0.5142|39.615|4.302|
|X_covariance_hr|0.9021|0.6561|0.009|0.6596|0.7309|33.501|4.323|
|IMG_px_all|0.9470|0.8308|0.008|0.8328|0.9016|-0.584|0.219|

对照：E65/E228 记录的 LOSO 身份 macro-F1 ≈0.985，本轮 EEG 分支 0.9779 与之同量级；差别来自本轮把评估限制在 trial 比例 (0.6,0.8] 的已注册行、且每折只用 19 个 enroled 身份计算 macro-F1。三个图像分支的闭集身份能力依次为 X_mw_cov4 0.42 / X_covariance_hr 0.66 / IMG_px_all 0.83，与 EEG 差 15–55 个百分点，这是后面融合必然稀释 EEG 身份的原因。


## 3. 身份融合候选（本轮重点）

`LOSO macro-F1` 是判据主指标（训练域闭集，mean±std over 20 折）；`Δ` 相对 `eeg_only`；`纯度(全 S2)` 是 k=20 固定划分下、**全部 19218 条 S2 行**上用该候选预测身份算的 micro 纯度（与门控阈值无关，四个候选可直接比较）；`@gate` 是 S1≥0.95 门控接受行上的纯度。

|候选|LOSO macro-F1|±std|Δ vs eeg_only|LOSO acc|OOF macro-F1|纯度(全 S2)|@gate 纯度|接受集合覆盖|与 eeg_only 相同比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|eeg_only|0.9779|0.002|+0.0000|0.9784|0.9898|0.6350|0.6488|20|1.0000|
|vote_equal|0.8992|0.005|-0.0787|0.9002|0.9466|0.5564|0.5525|20|0.7041|
|vote_weighted|0.9779|0.002|+0.0000|0.9784|0.9898|0.6350|0.6488|20|1.0000|
|rank_sum_identity|0.8630|0.007|-0.1148|0.8615|0.9077|0.5328|0.5321|20|0.6032|
|eeg_plus_best_img|0.9016|0.006|-0.0763|0.9018|0.9466|0.5596|0.5586|20|0.6524|

判据（config 预先写死）：LOSO macro-F1 ≥ eeg_only **且** 全 S2 纯度 ≥ eeg_only − 0.005。

- `vote_equal`：ΔF1 -0.0787（未达标）、Δ纯度 -0.0786（未达标）
- `vote_weighted`：ΔF1 +0.0000（达标）、Δ纯度 +0.0000（达标）、**与 eeg_only 预测逐行相同 → 代数退化，不计入采纳**
- `rank_sum_identity`：ΔF1 -0.1148（未达标）、Δ纯度 -0.1022（未达标）
- `eeg_plus_best_img`：ΔF1 -0.0763（未达标）、Δ纯度 -0.0753（未达标）

**为什么 `vote_weighted` 与 `eeg_only` 完全一样**：EEG 一票 0.5，三个图像分支各 1/6（合计 0.5），所以 EEG 的票数永远不小于三个图像分支之和；平票时按 config 用 EEG 的 d1 打破，于是图像票从不改变结果。这不是数值巧合，而是这组固定权重的代数性质。


## 4. 分支一致性（无标签）

|对照|4/4 一致|≥3/4 一致|≥2/4 一致|平均最大票数|
|---|---:|---:|---:|---:|
|Test 全部|0.1236|0.4061|0.8383|2.368|
|Test S2|0.1299|0.4126|0.8477|2.390|
|Test S1|0.0837|0.3653|0.7781|2.227|
|fit 行 OOF|0.4421|0.7984|0.9713|3.212|

两两一致率（Test 全部行）：EEG|X_mw_cov4 0.2319、EEG|X_covariance_hr 0.2941、EEG|IMG_px_all 0.4648、X_mw_cov4|X_covariance_hr 0.2679、X_mw_cov4|IMG_px_all 0.3897、X_covariance_hr|IMG_px_all 0.4301。

一致性作为独立的已知/未知判别器（无标签可读）：`-vote entropy` 的 S2-vs-S1 AUC = 0.5852（0.5 = 无信息）。它只是弱信号，这决定了下面一致性门控的天花板。

## 5. 门控候选

`S1/S2(LOSO)` 在 **LOSO 未知拒识 = 0.90** 的 tau 上读出；`S1 接受@gate` 是 S1≥0.95 门槛上 S1 的**接受率**（应≈0.05，拒识 ≈0.95）；`S2@S1.95` 是把 tau 设为该候选自己 Test S1 分数的 95% 分位后的 S2（**官方门控读法**）；`sepAUC` 与阈值无关；`整体拒识(LOSO tau)` 是该 tau 下全部 Test 行的拒识率；`20 折 S2 代理` 是 Train 侧已知行在该 tau 下的接受率。纯度列用 EEG 分支身份（与 E237 同口径）。

|门控|S1(LOSO)|S2(LOSO)|sepAUC|S1 接受@gate|S2@S1.95|整体拒识(LOSO tau)|覆盖身份|@gate 纯度|20 折 S2 代理|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|gate_score_fused|1.0000|0.2906|0.9611|0.0501|0.8776|0.7490|17|0.6488|0.7246±0.012|
|gate_agreement|0.7583|0.7617|0.8561|0.0501|0.5119|0.3092|20|0.7621|0.7029±0.004|

阈值扫描（`gate_agreement`，200 个 S1 分位点，**只输出曲线，不选点**）：
- S1 分位 0.50→0.999 时，S2 接受从 0.9202 变到 0.3983；曲线数组见 results.json 的 `gate_sweep`。
- 曲线上的点都不是本轮工作点；官方门控读法是 S1≥0.95（两个门控的 S1 接受率都是 0.0501，即 S1 拒识 0.9499）。
- **一致性门控没有超过 0.8776**：它的 sepAUC 从对照的 0.9611 降到 0.8561，S2@S1.95 从 0.8776 降到 0.5119。原因可见第 4 节：一致性只有 0.5852 的独立 AUC，按 0.5/0.5 与 融合分数相加只是把强排序稀释；这也说明"分支身份一致"在本数据上不足以替代分数证据。

## 6. Test 侧身份分布与覆盖（无标签）

|候选|覆盖身份(全体)|最大单一身份占比(全 S2)|接受集合 n|接受集合覆盖|
|---|---:|---:|---:|---:|
|eeg_only|20|0.1077|16865|20|
|vote_equal|20|0.1014|16865|20|
|vote_weighted|20|0.1077|16865|20|
|rank_sum_identity|20|0.1183|16865|20|
|eeg_plus_best_img|20|0.1219|16865|20|

## 7. 判定与冻结

- 身份方案：**`eeg_only`（保持基线）**。`vote_weighted` 虽然数值上"≥ eeg_only"，但它与 eeg_only 逐行相同（见第 3 节），按"图像是否真的参与身份"这一目的不构成证据，已排除在采纳之外。
- 门控：**`gate_score_fused`（E237 对照）**。官方门控读法（S1≥0.95）下 S2 = 0.8776，与 E237 相同，本轮一致性门控没有提供额外增益（其 S2 = 0.5119，低于 0.8776）。
- 一致性门控在 LOSO=0.90 工作点上的 S2 = 0.7617（对照 0.2906），但同一工作点上 S1 拒识只有 0.7583（对照 1.0000）——它是靠多放进 S1 换来的 S2，不可用。
- 门槛读法下一致性门控把 S2 从对照的 0.8776 改为 0.5119，sepAUC 从 0.9611 改为 0.8561。这是**转导式门槛读法**（tau 由 Test S1 分位决定），不等价于"用 Train 侧阈值就能多接受"。

## 8. 提交文件

`submission/test.csv`：22251 行、原始 metadata 顺序，`label` 0–20（0 = 在 S1≥0.95 门控下拒识，1–20 = 冻结身份方案 `eeg_only` 的预测身份）；`metrics.json` 记录该工作点：S1 拒识 0.9499、S2 接受 0.8776、整体拒识 0.2352、覆盖身份 20。

## 9. 限制

- Test has no identity labels: all F1 numbers are train-domain (LOSO fold-in / inner 5-block OOF). The closed-set F1 and the Test cluster purity measure different things and can disagree.
- Only 7 image types have both Train and Test pixel caches; GAF / MTF / spectrogram / DINOv3 exist only for Train and are unusable here.
- Test has been inspected repeatedly since E65; every S2 number carries a repeated-inspection bias. All weights, ranks, entropy definitions and decision rules were written in this file before the run.
- E232 showed session and registration status are confounded (S1 = 5 unregistered subjects, S2 = 20 enrolled users); alignment uses session only as a batch key and never to decide an identity.
- The mean alignment and the S1-quantile gate are transductive: they use unlabelled statistics of the batch being scored.
- The identity votes are nearest-prototype decisions in a 96-d space; the image branches can only re-rank the same 20 enrolled identities, so fusion cannot create identity information that no branch has.
- 运行：`.venv\Scripts\python.exe research/E238_multiview_identity_fusion/run_e238.py`（CUDA），耗时 18s。逐分支身份/距离缓存在 `cache/`，图像投影与原型存于 `models/e238_identity_fusion.pt`。
