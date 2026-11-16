# E200：`subwin6_pca96_k1 @ a=0.90` 的可移植冻结包与 predict.py

日期：2026-11-16。本实验**只打包**，不重训、不改动任何已有实验目录、不提交 git。
目标是把 E191 使用但从未落盘的**门控拟合态**（task 均值/标准差、PCA 均值与基、原型、阈值）
固化成一个自包含 bundle，并给出只依赖 bundle + 原始 `.mat` 的 `predict.py`。

## 0. 结论先看

| 项目 | 结果 |
|---|---|
| bundle 体积 | `bundle/frozen_bundle.pt` **576 KiB**（589,841 B） |
| bundle sha256 | `7f6f4f6a6e5741551f3f13331e19e8d78335f5fcfb49cbe390494b39d5f051d0` |
| 打包态与 E189 冻结态的一致性 | task 均值/标准差、PCA 均值、PCA 基 **逐位相同**（max\|Δ\| = 0）；原型 3.8e-06；OOF 分数 7.6e-06 |
| 阈值 | `tau_dist = 19.263826`（= Train 5 块 OOF 最近距离的 90% 分位）；等价归一化 `tau_norm = -1.305229`（与 E191 `audit_v3.json` 记录 **完全一致**） |
| Test 拒识总数 | **7,797 / 22,251（35.04%）** |
| Test S1 拒识 / S2 接受 | **0.713485 / 0.706889**（E191：0.713485 / 0.706889，**差 0.000000**） |
| 复现校验（256 条冒烟） | 特征 max\|Δ\| = **0.0**（阈值 1e-3），标签 **256/256 逐位一致（100%）** |
| 复现校验（全量 22,251 条） | 与「缓存特征 + 同原型/同阈值」路径 **100.0000%** 一致；与 E189 保存的冻结 Test 分数 **100.0000%** 一致 |
| 提交合规 | 22,251 行、`Sample` 顺序与原 metadata 逐行一致、`Label ∈ {0..20}`、无 NaN |
| Test 官方 AUC / F1 | **N/A**（Test 无真实身份标签；本实验不产生官方指标） |

Train 侧参考读数（沿用 E191 记录，未重算，用于说明本门控的强弱）：
LOSO AUROC **0.8835**、U@95% **0.3292**、LOSO 已知接受率 0.9070；
身份头（本门控不使用）5 块 OOF 闭集 macro-F1 0.9882。**Test 官方 AUC/F1 = N/A。**

## 1. 系统口径（逐条从代码复核，不凭记忆）

- 流：`subwin6_pca96_k1`（E171 `train_subwin_mean_n6.npy` / E168 子窗口均值，N = 6）
- 子窗口：18×1200 沿时间切成 **6 个不重叠窗口（每个 200 采样点）**，**每个子窗口独立跑完整 E65 流程**
  （5 段理想 FFT 频带隔离 → 去均值 → 逐通道标准化（ddof=1）→ 18×18 相关矩阵 → 0.05 收缩 →
  eigh/logm → 上三角 171 维（非对角 ×√2）→ E36 `detrend_zband`），再把 6 个 855 维向量取算术平均。
  `detrend_zband` 是非线性的，所以子窗口均值 ≠ 全窗特征（E171 已用 N=1 逐位复现 E65 验证）。
- 标准化：按 task 用**该 task 的全部 Train 行**的 mean/std（E143/E189 规则）；
- PCA：96 维，均值与基只用 **fit_rows（frac ≤ 0.6）** 拟合；
- 原型：每个 (task, 20 身份) 一个原型（k = 1 = fit_rows 上的 k-means 单中心），共 4×20×1×96；
- 分数：同 task 内到 20 个原型的**最近欧氏距离**取负；
- 判决：`最近距离 > tau → Label 0（未注册）`，否则取最近原型的身份（1–20）。
  与 E189 的等价写法：`(oof_mu - dist) / oof_sd >= tau_norm`。

上述口径与 `research/E189_balanced_gate_gates/config.json` 的 `streams.subwin6_pca96_k1`
（pca 96 / k 1 / dim 855 / kind subwin6）以及 E199 `config.json` 的三阶段定义一致：
阶段 1 = 全部 20 人的标准化 + PCA（折不变），阶段 2 = 每身份 1 个原型（fit_rows），
阶段 3 = 部署逐行打分 + 单阈值拒识。本实验把阶段 1/2 的产物真正落盘。

## 2. 冻结包内容清单

`bundle/frozen_bundle.pt`（`torch.save` 的一个 dict，加载用 `weights_only=False`）：

| 段 | 键 | 形状 / 内容 |
|---|---|---|
| `stage1` | `feature` | 子窗口切分与 E65 全部常数：N=6、窗口长 200、fs=300、5 段频带 [1,8,8,13,13,20,20,30,30,40]、ddof=1、收缩 0.05、特征值 floor 1e-12、非对角 √2、zband eps 1e-6、dim 855 |
| | `task_stats_mu` / `task_stats_sd` | (4, 855)（task 顺序 EC/EO/Imagery/SSVEP） |
| | `pca_mean` | (855,) |
| | `pca_basis` | (855, 96) |
| `stage2` | `prototypes` | (4, 20, 1, 96)，按 (task, identity−1) 索引 |
| | `classes` | 1..20 |
| `stage3` | `tau_dist` | 19.26382579803467 |
| | `tau_norm` / `oof_mu` / `oof_sd` | −1.3052289546016027 / −15.592114543121754 / 2.8130782787237183 |
| | `train_known_acceptance` | 0.8999（Train OOF 接受率，a = 0.90） |
| | `decision` / `threshold_source` / `sensitivity_window_tau_dist` | 判决式、阈值来源、eval 窗分位对照 19.187788 |
| | `identity_backbone` | 7 个权重文件的相对路径 + sha256（见 §2.2） |
| `meta` | `git_commit` / `built_at` / `feature_dim` / `classes` / `tasks` / `source_checksums` / `state_checks_vs_e189` / `test_reproduction` | 构建时间 2026-11-16 09:52:16，git `c8e3fba5eb3e` |

同步导出：

| 文件 | 体积 | sha256 | 用途 |
|---|---:|---|---|
| `bundle/frozen_bundle.pt` | 589,841 B | `7f6f4f6a6e5741551f3f13331e19e8d78335f5fcfb49cbe390494b39d5f051d0` | 唯一运行依赖 |
| `bundle/prototypes.npz` | 363,352 B | `224a6f97f8007f9537f7dd061ca064de652d28ae7866e946fb7cc1a8c646de14` | 原型 (4,20,1,96) + PCA 基/均值 + task 统计，便于人工检查 |
| `bundle/manifest.json` | 5,744 B | `078766dbce8ef62323f0f6b2b7a74907c809ed122bc20916de76bbdbe5541047` | 人类可读清单 + 全部 sha256 |

体积说明：PCA 基 328,320 B + 原型 30,720 B（+ 统计量），合计 389,820 B，
直接内联在 bundle 里即可（0.56 MiB），**不需要**拆成外部 npz；`prototypes.npz` 是附加检查文件。

### 2.1 打包态与 E189 冻结态的核对（build 时自动断言）

| 项 | max\|Δ\| |
|---|---:|
| task 均值 / 标准差 | 0.0 / 0.0 |
| PCA 均值 / PCA 基 | 0.0 / 0.0 |
| 原型 | 3.81e-06 |
| 5 块 OOF 分数 | 7.63e-06 |
| `oof_mu` / `oof_sd` | −3.2e-08 / +1.4e-09 |

即：本 bundle 就是 E191 实际使用的门控拟合态（差异只剩 float32 累加顺序）。

### 2.2 身份骨干权重（本门控**不使用**，仅登记）

`subwin6` 门控只用距离原型，不看 CNN 身份头。以下权重属于 E191 的闭集读出口径（#1–#4），
放在 bundle 里只为可追溯：

| 文件 | sha256 |
|---|---|
| `research/E191_fix_gate1_gate7/model_branch_a_v3.pt` | `d7204c9a32c56d926c8786e1c7759f5bda4e6e2fed0f363099cc5a1fcfeb6387` |
| `research/E191_fix_gate1_gate7/cache/stage_a_v3.pt` | `83d5ac7d233754b68c0d224bb82a12750ed09a51749d74fe6e3cb13c87a5f91d` |
| `.../cache/fold_model_v3_0.pt` | `f69e5733ce7ee7be4789298d9afa758c4222e7fe55c22430d52a15c22515ae25` |
| `.../cache/fold_model_v3_1.pt` | `7acc91b2882afab78c84b0a513d4f8fc3af2b0348a4ad74adcff5447a079fe3e` |
| `.../cache/fold_model_v3_2.pt` | `241a5b9585eecb2ee4cee42f2ee15d5527301238b61d2bc26ef2a8c34288d8f3` |
| `.../cache/fold_model_v3_3.pt` | `c80b12bb3bd4172d96974ace509a2aef59d0fc8d7c7c0a86dbd06dd0faf57e2c` |
| `.../cache/fold_model_v3_4.pt` | `5d73bb07e4bee52ae5666e6065e43c359178be24ba99d912e602334f89dab9d9` |

源数据指纹：`train_subwin_mean_n6.npy` `c93a7aa8…9b610`、`test_subwin_mean_n6.npy` `5a45e4cd…9daf`、
`E189/cache/stream_scores.npz` `ec439626…38514`、`train_metadata.csv` `f3be516e…b545`、
`test_metadata.csv` `6af2d22a…310e`（完整值见 `bundle/manifest.json`）。

## 3. 阈值 tau：数值与 Train 分位来源

- 规则：`tau = quantile(OOF 最近距离, a)`，本实验 **a = 0.90**（项目既定「Train 已知接受 90%」约定，
  `E189/config.json → threshold.rule`，**不是**在 Test 上调出来的）。
- OOF 距离：5 个连续 frac 块，每块的 20 个原型只用 `fit_rows & ~块` 重建，然后给该块打分；
  11304 条 Train 全部覆盖，无 NaN。**只用 Train**。
- 数值：`tau_dist = 19.263826`；等价归一化 `tau_norm = (oof_mu − tau_dist)/oof_sd = −1.305229`，
  与 `E191/audit_v3.json` 里 `subwin6_pca96_k1` 在 a=0.90 的 `tau = −1.3052290320396425` 一致到 1e-9。
- Train 侧接受率（OOF）：0.8999（分位定义天然 ≈ 0.90）；E191 记录的 LOSO 已知接受率 0.9070 是
  20 折「留一身份」口径，两者不是同一个量，不要混用。
- **对照（未部署）**：若按本次任务描述里的字面口径「eval 窗 0.6<frac≤0.8 的 known 分数 90% 分位」，
  得 `tau_dist = 19.187788`（同样只用 Train 行）。该变体的 Test 读数是 S1 0.7201 / S2 0.6992，
  见 §5.3 的解释。

## 4. predict.py：用法与冒烟结果

```
python predict.py --meta <test_metadata.csv> --root <数据根目录> --out <submission.csv> \
                  [--device cuda] [--limit N] [--batch-size 512] [--io-workers 8] [--work <dir>]
```

- 依赖：`bundle/frozen_bundle.pt` + 原始 `.mat`。**不 import 任何 `research/E*` 模块**，
  不使用任何 Test 统计量、不做任何推理期拟合（无白化、无归一化统计、无自适应阈值）。
- 读 `.mat`：`loadmat(..., variable_names=["data"])["data"]` → (1200,18) → 转置 18×1200，
  校验形状与有限性。
- 断点续算：特征写入 `<work>/features.npy`（memmap），每 1024 条 flush 并更新
  `<work>/progress.json`（绑定 metadata 的 sha256 与总行数）；重跑自动从 `rows_done` 续算。
  `--work` 默认 `<out>_work`。
- 输出：`Sample,Label` 两列，`Sample` 统一格式化为 `test_XXX`（三位零填充），行数 = metadata 行数。
- 日志：总行数、拒识数、各 Label 计数、距离分布，以及**落盘之后**才读取的按 session 分组拒识率。

冒烟（`--limit 256`，RTX 3080 Laptop，0 s）：

```
rows            : 256
tau_dist        : 19.263826 (a=0.90)
rejected (0)    : 100 (0.3906)
label counts    : 0:100, 1:16, 2:4, 3:10, 4:3, 6:13, 7:10, 10:7, 11:3, 12:1, 13:18, 14:6,
                  15:15, 16:13, 17:10, 18:25, 19:1, 20:1
distance        : mean 18.560, median 17.865, min 11.676, max 30.764
session audit   : session1 N=28 rejected=23 (0.8214); session2 N=228 rejected=77 (0.3377)
```

全量（22,251 条，10 s）：拒识 7,797（0.3504）；`session1` 3,033 条拒识 2,164（0.7135）、
`session2` 19,218 条接受 13,585（0.7069）。

## 5. 复现校验（三项）

### 5.1 256 条逐位比对

对「`predict.py` 从 `.mat` 重算」与「E171 冻结特征缓存 + 同一 bundle 原型/阈值」两条路径：

| 项 | 结果 |
|---|---|
| 特征 max\|Δ\|（855 维 × 256 条） | **0.000e+00**（要求 ≤ 1e-3 ✅） |
| 标签逐位一致 | **256 / 256 = 100%** ✅ |
| 差异条数 | 0 |

### 5.2 全量 22,251 条

| 对照 | 一致率 |
|---|---|
| vs「缓存特征 + 同原型/同阈值」 | **1.000000**（0 条不一致） |
| vs E189 `subwin6_pca96_k1__test` 冻结分数（E191 审计所依据的那批分数） | **1.000000** |
| 距离 max\|Δ\| vs E189 | 1.9e-05 |

E191 目录里**没有**保存预测 CSV（只有 `audit_v3.json` / `attack_v3.json` 读数），
因此上面的比对对象是「E191 审计所依据的冻结分数」而不是某个 csv。

### 5.3 与 E191 的 S1 / S2 比对

| 指标 | 本实验（bundle 部署 tau） | E191 记录 | 差 |
|---|---:|---:|---:|
| Test S1 拒识率 | 0.713485 | 0.713485 | **+0.000000** |
| Test S2 接受率 | 0.706889 | 0.706889 | **+0.000000** |
| 拒识总数 | 7,797 | 0.6496 × 22,251 ≈ 7,797（接受率口径反推） | 0 |
| Train 侧 tau | 19.263826（dist）/ −1.305229（norm） | −1.305229 | 0 |

**唯一超过 0.002 的差异来自另一个可选阈值口径（明确记录，不是部署口径）**：
若改用「eval 窗 0.6<frac≤0.8 的 known 分数 90% 分位」（`tau_dist = 19.187788`），
S1 = 0.7201（+0.0066）、S2 = 0.6992（−0.0077），S2 跌破 0.70。原因是 eval 窗同 session 的
已知分数整体比 5 块 OOF 略高（OOF 的早期 frac 块更难），分位点因此在距离轴上左移 0.076，
单阈值下 S1/S2 必然反向移动。**部署仍采用 E189/E191 的 OOF 分位口径**：它是本系统被审计过的
那一个 τ，且与 E191 记录逐位相同；eval 窗口径只作为敏感性对照写进 `stage3`。

## 6. 提交合规性检查

| 检查 | 结果 |
|---|---|
| 行数 | 22,251 = metadata 行数 ✅ |
| 列 | `Sample,Label` ✅（与 `commit/answer/赞美诗_第3次.csv` 同格式） |
| `Sample` 顺序 | 与原 metadata 逐行一致（`test_001…test_XXX`，三位零填充）✅ |
| `Label` 取值 | min 0、max 20，全部 ∈ {0..20} ✅ |
| NaN / 空值 | 无 ✅ |

产物：`submission_frozen.csv`（309,309 B，sha256 `ec8fe77abd725a7a682f4b919574e2729d8d57e2313ec66f45fb88c32cb821a6`）。

## 7. 部署注意事项

1. **session 字段完全不参与判决**：`predict.py` 只在 CSV 落盘后单独读一次 `session` 列做描述性统计；
   特征、分数、阈值、标签代码路径里没有任何 session 判断。Session 也未用于选阈值或选权重。
2. **无 Test 统计量**：推理严格逐条（单 trial 内统计：逐通道均值/标准差、频带滤波、相关矩阵、
   `detrend_zband` 都在单条 trial 内部完成），不做批次归一化、不做 Test 端 PCA/白化/阈值重估；
   `--batch-size` 只影响吞吐（E171 已验证单条与批量逐位一致，本实验再次复现 max\|Δ\| = 0）。
3. **单阈值**：全系统只有一个 τ（一个实数），作用在「最近欧氏距离」上；没有按 task、按 session、
   按身份分支的阈值，也没有多阈值拼接。
4. **权重来源**：门控侧（均值/标准差/PCA/原型/τ）全部只用 Train 拟合，`frac>0.6` 的评估块从未进入
   PCA 或原型；阈值来自 Train 5 块 OOF 的 90% 分位。**本门控没有任何 Test 选权/选阈值**；
   bundle 里登记的 CNN 权重只对应 E191 的闭集读出口径，`subwin6` 判决不使用它们。
5. **单点口径**：本 bundle 对应一个固定工作点（a = 0.90），不提供 a 网格；换 λ/a 需要重新打包并重跑本报告。
6. **残差风险（如实说明）**：本门控是距离型，Train 的 OOF 拒识（LOSO 0.5554 @ a=0.90）与 Test S1 的
   0.7135 并不一致，说明跨 session 时已知分数分布上移；因此 τ 的绝对水平只在「同批次部署 + 已注册 20 人」
   语义下成立。Test 的 S1/S2 是预测工程审计读数，不是官方真值指标。

## 8. 合规边界

- 冻结包与阈值只用 Train（`BrainPrint_cache/train_metadata.csv` 的 11304 条，session1）。
- Test（22,251 条）只用于最终打分与复现校验，未参与任何拟合、选择或阈值。
- 未修改任何已有实验目录（E189/E191/E171 的缓存均以只读方式打开），未提交 git。
- Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = **N/A**（无真实身份标签）。

## 9. 文件与本实验产物

| 文件 | 内容 |
|---|---|
| `build_bundle.py` | 从冻结特征缓存重建并校验门控拟合态，产出 bundle + manifest + prototypes.npz |
| `predict.py` | 自包含推理（bundle + `.mat`），支持 `--limit` 与断点续算 |
| `verify_repro.py` | 三条路径的逐位比对、提交合规检查，写 `cache/repro_check.json` |
| `bundle/frozen_bundle.pt` | 冻结包（576 KiB） |
| `bundle/manifest.json` | 清单 + 全部 sha256 + 构建信息 |
| `bundle/prototypes.npz` | 原型 / PCA / task 统计（检查用） |
| `submission_frozen.csv` | 全量提交文件（22,251 行） |
| `cache/reference_test.npz` | 缓存路径的参考距离/身份/标签、两种 τ 的标签 |
| `cache/smoke_256.csv`、`cache/smoke_256_work/` | 冒烟输出与特征（可续算） |
| `submission_frozen_work/` | 全量特征 memmap + 进度（可续算、可复核） |
| `cache/repro_check.json` | 复现校验结果（机器可读） |
| `run.log` | 构建 / 冒烟 / 全量 / 校验的完整日志 |

```powershell
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" `
    "D:\WORK\BCIC2026\BrainPrint\research\E200_final_freeze\build_bundle.py"
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" `
    "D:\WORK\BCIC2026\BrainPrint\research\E200_final_freeze\predict.py" `
    --meta "D:\WORK\BCIC2026\BrainPrint\BrainPrint_cache\test_metadata.csv" `
    --root "D:\WORK\BCIC2026\BrainPrint\Brainprint_Competition" `
    --out "D:\WORK\BCIC2026\BrainPrint\research\E200_final_freeze\submission_frozen.csv" --batch-size 512
& "D:\WORK\BCIC2026\BrainPrint\.venv\Scripts\python.exe" `
    "D:\WORK\BCIC2026\BrainPrint\research\E200_final_freeze\verify_repro.py"
```
