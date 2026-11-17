"""E230: emit official-format Test predictions from the frozen E228 model.

This script only performs inference with research/E228_constrained_maxS2/models/deploy_frozen.pt.
It does not retrain anything and never changes tau, the feature definitions, the block weights,
the PCA basis or the prototypes. All outputs are written inside this directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE / "config.json").read_text(encoding="utf8"))
LOG_PATH = HERE / "run.log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
EPS = 1.0e-12


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf8") as f:
        f.write(str(msg) + "\n")


def tensor(x, dtype=torch.float32, device=DEVICE):
    return torch.as_tensor(x, dtype=dtype, device=device)


def quantile_tau(x, q: float) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), q))


def auc_ties(pos, neg) -> float:
    """AUC with average ranks for ties; high score = positive class."""
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    s = np.r_[pos, neg]
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def fold_auc(unknown, known) -> float:
    """E228 fold AUC: P(unknown score < known score), ties count 0.5."""
    u = np.asarray(unknown, dtype=float)[:, None]
    k = np.asarray(known, dtype=float)[None, :]
    return float(((u < k).sum() + 0.5 * (u == k).sum()) / (len(k.ravel()) * len(u.ravel())))


def load_rep(split: str, block_order, block_dims) -> np.ndarray:
    parts = []
    for block in block_order:
        path = ROOT / CFG["feature_paths"][split][block]
        if not path.exists():
            raise FileNotFoundError(f"missing feature cache: {path}")
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != int(block_dims[block]):
            raise ValueError(f"{path}: shape {arr.shape}, expected width {block_dims[block]}")
        parts.append(arr)
    out = np.concatenate(parts, axis=1)
    expected = int(CFG["representation"]["input_dim"])
    if out.shape[1] != expected:
        raise ValueError(f"{split}: concatenated width {out.shape[1]}, expected {expected}")
    return out


def transform(state, raw: np.ndarray, tasks: np.ndarray) -> torch.Tensor:
    """Frozen E228 transform: task-wise z-score -> sqrt(w) -> centre -> PCA."""
    x = tensor(raw)
    z = torch.empty_like(x)
    for i, task in enumerate(TASKS):
        rows = tasks == task
        mu = tensor(state["mu"][i])
        sd = tensor(state["sd"][i])
        z[rows] = (x[rows] - mu) / sd
    if state.get("w") is not None:
        z = z * torch.sqrt(tensor(state["w"]))
    pm = tensor(state["pm"], dtype=torch.float64)
    w = tensor(state["W"], dtype=torch.float64)
    return ((z.double() - pm) @ w).float()


def predict(state, z: torch.Tensor, tasks: np.ndarray):
    """Plain scoring: score = -min distance to same-task prototypes; identity = that prototype."""
    proto_task = np.array([t for t, _ in state["keys"]])
    proto_id = np.array([int(s) for _, s in state["keys"]], dtype=int)
    protos = tensor(state["p"])
    score = np.full(len(tasks), np.nan, dtype=np.float64)
    pred = np.full(len(tasks), -1, dtype=int)
    for task in TASKS:
        rows = np.flatnonzero(tasks == task)
        cols = np.flatnonzero(proto_task == task)
        if not len(rows) or not len(cols):
            continue
        dist = torch.cdist(z[rows], protos[cols]).double()
        dmin, amin = dist.min(dim=1)
        score[rows] = -dmin.cpu().numpy()
        pred[rows] = proto_id[cols][amin.cpu().numpy()]
    if not np.isfinite(score).all() or (pred < 1).any() or (pred > 20).any():
        raise RuntimeError("scoring produced non-finite scores or invalid identities")
    return score, pred


def fold_metrics(A: float):
    """Audit metrics recomputed from the frozen E227 auc2_lofo per-fold score caches."""
    fold_dir = ROOT / "research/E227_joint_frontier/cache"
    known_scores, unknown_scores = [], []
    aucs, per_fold_rej = [], []
    n_known = n_unknown = n_correct = n_accepted_known = n_rejected_unknown = 0
    for fold in range(1, 21):
        path = fold_dir / f"fold_v1_auc2_lofo_{fold:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing fold cache: {path}")
        d = np.load(path, allow_pickle=True)
        score = d["score"].astype(np.float64)
        pred = d["pred"].astype(int)
        subject = d["subject"].astype(int)
        known = d["known"].astype(bool)
        unknown = d["unknown"].astype(bool)
        oof_fit = d["oof_fit"].astype(bool)
        tau = quantile_tau(d["oof_s"][oof_fit], 1.0 - A)
        accepted = score >= tau

        n_known += int(known.sum())
        n_unknown += int(unknown.sum())
        n_correct += int(((pred == subject) & known).sum())
        n_accepted_known += int((accepted & known).sum())
        n_rejected_unknown += int((~accepted[unknown]).sum())
        per_fold_rej.append(float((~accepted[unknown]).mean()) if unknown.any() else np.nan)

        known_scores.append(score[known])
        unknown_scores.append(score[unknown])
        aucs.append(fold_auc(score[unknown], score[known]))

    known_all = np.concatenate(known_scores)
    unknown_all = np.concatenate(unknown_scores)
    tau95 = quantile_tau(known_all, 0.05)
    u95_micro = float((unknown_all < tau95).mean())
    u95_folds = []
    u95_num = u95_den = 0
    for k, u in zip(known_scores, unknown_scores):
        t95 = quantile_tau(k, 0.05)
        u95_folds.append(float((u < t95).mean()))
        u95_num += int((u < t95).sum())
        u95_den += len(u)
    u95_perfold = float(u95_num / u95_den)
    return dict(
        A=float(A),
        n_known=int(n_known),
        n_unknown=int(n_unknown),
        block_accuracy=float(n_correct / n_known),
        known_acceptance=float(n_accepted_known / n_known),
        loso_rejection=float(n_rejected_unknown / max(n_unknown, 1)),
        u95=u95_micro,
        u95_perfold=u95_perfold,
        u95_macro=float(np.mean(u95_folds)),
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        per_fold_rejection_mean=float(np.nanmean(per_fold_rej)),
        per_fold_rejection_min=float(np.nanmin(per_fold_rej)),
        per_fold_rejection_max=float(np.nanmax(per_fold_rej)),
    )


def fit_accuracy(model) -> float:
    """In-sample 20-class nearest-prototype accuracy on the model's own fit rows."""
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    max_trial = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    frac = meta.trial.to_numpy(float) / np.maximum(max_trial, 1.0)
    fit = frac <= 0.6
    x = load_rep("train", model["block_order"], CFG["representation"]["block_dims"])
    if len(x) != len(meta):
        raise ValueError(f"train features {len(x)} vs metadata {len(meta)}")
    z = transform(model["state"], x, meta.task.to_numpy())
    _, pred = predict(model["state"], z, meta.task.to_numpy())
    return float((pred[fit] == meta.subject.to_numpy(int)[fit]).mean()), int(fit.sum())


def main() -> None:
    LOG_PATH.write_text("", encoding="utf8")
    t0 = time.time()
    log(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
        f"torch={torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: E230 requires the GPU venv")

    model_path = ROOT / CFG["frozen_model"]
    if not model_path.exists():
        raise FileNotFoundError(f"frozen model missing: {model_path}")
    model = torch.load(model_path, weights_only=False)
    state = model["state"]
    tau = float(model["tau"])
    A = float(model["A"])
    log(f"model={model_path}")
    log(f"candidate={model['candidate']} rep={model['rep']} variant={model['variant']} "
        f"source={model['source']}")
    log(f"tau={tau:.9f} A={A:.9f} pca={model['pca']} k={model['k']} dim={model['dim']}")
    if abs(tau - float(CFG["threshold"]["tau"])) > 1e-9:
        raise RuntimeError("tau in config.json does not match the frozen model")
    if abs(A - float(CFG["threshold"]["A"])) > 1e-12:
        raise RuntimeError("A in config.json does not match the frozen model")
    if list(model["block_order"]) != list(CFG["representation"]["block_order"]):
        raise RuntimeError("block order mismatch between model and config")
    if not np.allclose(np.asarray(model["block_weights"], dtype=np.float64),
                       np.asarray(CFG["representation"]["block_weights_sqrt_applied_before_pca"],
                                  dtype=np.float64), atol=1e-12):
        raise RuntimeError("block weights mismatch between model and config")

    meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    if len(meta) != int(CFG["data"]["test_trials"]):
        raise ValueError(f"test metadata rows {len(meta)} != {CFG['data']['test_trials']}")
    if not meta["sample"].is_unique:
        raise ValueError("test metadata sample column is not unique")
    tasks = meta.task.to_numpy()
    sessions = meta.session.astype(str).str.casefold().to_numpy()
    is_s1 = np.isin(sessions, ["session1", "s1", "1"])
    log(f"rows={len(meta)} S1={int(is_s1.sum())} S2={int((~is_s1).sum())} "
        f"tasks={dict(pd.Series(tasks).value_counts())}")

    x_test = load_rep("test", model["block_order"], CFG["representation"]["block_dims"])
    if len(x_test) != len(meta):
        raise ValueError(f"test features {len(x_test)} vs metadata {len(meta)}")
    log(f"test feature matrix={x_test.shape}")

    z_test = transform(state, x_test, tasks)
    score, pred_identity = predict(state, z_test, tasks)
    rejected = score < tau
    label = np.where(rejected, 0, pred_identity).astype(int)
    if label.min() < 0 or label.max() > 20:
        raise ValueError("label outside 0-20")

    # ---------------- consistency check 1: frozen per-sample scores
    ref_path = ROOT / CFG["reference_scores"]["path"]
    if not ref_path.exists():
        raise FileNotFoundError(f"frozen score cache missing: {ref_path}")
    ref = np.load(ref_path, allow_pickle=True)
    ref_score = ref["score"].astype(np.float64)
    ref_pred = ref["pred"].astype(int)
    if len(ref_score) != len(meta):
        raise ValueError(f"reference score rows {len(ref_score)} != {len(meta)}")
    if not np.array_equal(ref["is_s1"].astype(bool), is_s1):
        raise ValueError("reference session flags do not align with test_metadata.csv")
    if not np.array_equal(np.asarray(ref["task"], dtype=str), tasks.astype(str)):
        raise ValueError("reference task labels do not align with test_metadata.csv")
    max_abs_score_diff = float(np.max(np.abs(score - ref_score)))
    pred_identical = bool(np.array_equal(pred_identity, ref_pred))
    log(f"reload check vs {ref_path.name}: max|dscore|={max_abs_score_diff:.3e} "
        f"pred_identical={pred_identical}")

    # Negative control: E228/cache only contains the unweighted plain candidate.
    plain_path = ROOT / CFG["e228_plain_cache_for_negative_control"]
    plain_diff = float("nan")
    if plain_path.exists():
        plain = np.load(plain_path, allow_pickle=True)
        plain_diff = float(np.max(np.abs(score - plain["score"].astype(np.float64))))
        log(f"negative control: E228 unweighted plain cache max|dscore|={plain_diff:.6f} "
            f"(expected non-zero; different candidate)")

    # ---------------- consistency check 2: S1/S2 anchors
    S1_rejection = float(rejected[is_s1].mean())
    S2_acceptance = float((~rejected[~is_s1]).mean())
    S2_rejection = float(rejected[~is_s1].mean())
    sep_auc = auc_ties(score[~is_s1], score[is_s1])
    anchors = CFG["expected_anchors"]
    dS1 = S1_rejection - float(anchors["test_S1_rejection"])
    dS2 = S2_acceptance - float(anchors["test_S2_acceptance"])
    dAUC = sep_auc - float(anchors["test_separation_auc"])
    log(f"Test S1 rejection={S1_rejection:.9f} (N={int(is_s1.sum())}) "
        f"S2 acceptance={S2_acceptance:.9f} (N={int((~is_s1).sum())})")
    log(f"anchor deltas: dS1={dS1:+.3e} dS2={dS2:+.3e} dSepAUC={dAUC:+.3e}")
    if abs(dS1) > 1e-4 or abs(dS2) > 1e-4:
        raise RuntimeError("S1/S2 do not reproduce the E228 anchors within 1e-4")

    per_task = {}
    for task in TASKS:
        m1 = is_s1 & (tasks == task)
        m2 = (~is_s1) & (tasks == task)
        per_task[task] = dict(
            n_S1=int(m1.sum()),
            S1_rejection=float(rejected[m1].mean()) if m1.any() else None,
            n_S2=int(m2.sum()),
            S2_acceptance=float((~rejected[m2]).mean()) if m2.any() else None,
            rejected_total=int(rejected[tasks == task].sum()),
        )

    # ---------------- consistency check 3: fold-side audit metrics
    fm = fold_metrics(A)
    fit_acc, n_fit_rows = fit_accuracy(model)
    log(f"fold audit: known={fm['n_known']} unknown={fm['n_unknown']} "
        f"block_acc={fm['block_accuracy']:.6f} known_accept={fm['known_acceptance']:.6f} "
        f"loso_rej={fm['loso_rejection']:.6f} auc={fm['auc_mean']:.6f} "
        f"U@95(perfold)={fm['u95_perfold']:.6f} (global={fm['u95']:.6f})")
    log(f"in-sample fit accuracy={fit_acc:.6f} on {n_fit_rows} fit rows")

    # ---------------- label distribution
    counts = np.bincount(label, minlength=21)
    n_rejected = int(counts[0])
    identity_counts = {i: int(counts[i]) for i in range(1, 21)}
    unpredicted = [i for i in range(1, 21) if counts[i] == 0]
    log(f"labels: rejected={n_rejected}/{len(label)}={n_rejected / len(label):.6f}; "
        f"accepted={len(label) - n_rejected}")
    log("identity counts: " + ", ".join(f"{i}:{identity_counts[i]}" for i in range(1, 21)))
    log(f"identities never predicted: {unpredicted if unpredicted else 'none'}")

    # ---------------- write submission files
    sub = pd.DataFrame({
        "Sample": meta["sample"].astype(str).to_numpy(),
        "Label": label.astype(int),
    })
    if len(sub) != 22251:
        raise AssertionError("test.csv row count is not 22251")
    sub.to_csv(HERE / "test.csv", index=False)

    detailed = pd.DataFrame({
        "sample": meta["sample"].astype(str).to_numpy(),
        "path": meta["path"].astype(str).to_numpy(),
        "task": tasks.astype(str),
        "session": meta["session"].astype(str).to_numpy(),
        "score": score,
        "pred_identity": pred_identity.astype(int),
        "rejected": rejected.astype(int),
        "label": label.astype(int),
    })
    detailed.to_csv(HERE / "test_predictions_detailed.csv", index=False,
                    float_format="%.10f")

    # Independent read-back audit of the written submission files.
    sub_back = pd.read_csv(HERE / "test.csv")
    detail_back = pd.read_csv(HERE / "test_predictions_detailed.csv")
    sample_list = meta["sample"].astype(str).tolist()
    if len(sub_back) != 22251 or sub_back["Sample"].astype(str).tolist() != sample_list:
        raise AssertionError("test.csv read-back row count or order does not match test_metadata.csv")
    if len(detail_back) != 22251 or detail_back["sample"].astype(str).tolist() != sample_list:
        raise AssertionError("detailed read-back row count or order does not match test_metadata.csv")
    if not set(sub_back["Label"].astype(int)).issubset(set(range(21))):
        raise AssertionError("test.csv contains a label outside 0-20")
    if not np.array_equal(sub_back["Label"].to_numpy(int), label):
        raise AssertionError("test.csv labels differ from in-memory labels")
    log("read-back audit: 22251 rows, Sample order identical to test_metadata.csv, labels 0-20 OK")
    log(f"wrote test.csv ({len(sub)} rows) and test_predictions_detailed.csv")

    # ---------------- markdown deliverables (all numbers above are from this run)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    runtime = time.time() - t0
    metrics_md = f"""# E230 提交指标汇总（E228 冻结模型）

- 生成时间：{now}
- 冻结模型：`{CFG['frozen_model']}`（候选 `{model['candidate']}`；来源：{model['source']}）
- 阈值：`tau = {tau:.9f}`（`A* = {A:.9f}`）；`score < tau` → 0，否则最近原型身份 1–20
- 设备：{torch.cuda.get_device_name(0)}；torch {torch.__version__}
- 提交文件：`test.csv`（{len(sub)} 行，原始顺序，表头 `Sample,Label`）

> **口径声明**：官方 Test 没有身份标签，也没有公开的对抗样本标签。因此官方 Test 的 C、身份 macro-F1、RA、ASafeR、AUROC/F1 无法在本机计算，表中一律写 **N/A**；没有用训练侧代理值冒充官方指标。S1/S2 是模型冻结后按 Test 会话分层的结构审计，不是官方 C 或 U。

## 1. 十一项指标

| # | 指标 | 数值 | 口径 / 依据 |
|---:|---|---:|---|
| 1 | 拟合准确率 | {fit_acc:.4f} | 部署模型在自身 fit 行（trial fraction ≤ 0.6，{n_fit_rows} 行）上的 20 类最近原型闭集准确率；in-sample，仅用于容量检查，不是泛化证据 |
| 2 | C（闭集准确率） | N/A | 官方 Test C 需要 Test 真实身份标签，官方未提供；未用训练侧代理值替代 |
| 3 | 身份 macro-F1 | N/A | 官方 Test 身份 macro-F1 需要 Test 真实身份标签，官方未提供；未用训练侧代理值替代 |
| 4 | 真正留块准确率 | {fm['block_accuracy']:.4f} | 严格 20 折 subject-LOSO，评估窗口 (0.6, 0.8] 为真正留出的时间块；闭集最近原型 micro 准确率，共 {fm['n_known']} 行已知评估样本 |
| 5 | AUROC | {fm['auc_mean']:.4f} | 项目 #5 口径的伪未知 AUROC：20 折 LOSO 每折 unknown-vs-known AUC 的折均值；无阈值、无 Test 信息。**官方 Test AUROC = N/A（无标签）** |
| 6 | U@95 | {fm['u95_perfold']:.4f} | 项目 #6 口径：每折把已知接受固定为 95%（逐折 tau95），再 pooled micro 统计未知拒识（{fm['n_unknown']} 个留出身份未知评估样本）；折均值 {fm['u95_macro']:.4f}，全局单一 tau95 为 {fm['u95']:.4f}；不是 Test U |
| 7 | RA | N/A | 需要官方对抗样本及其原始身份标签，官方未提供；无法区分 Test S2 中的正常与对抗样本 |
| 8 | ASafeR | N/A | 同 RA：需要对抗样本标签；未用自建攻击代理值冒充官方 ASafeR |
| 9 | S1 拒识 | {S1_rejection:.4f} | Test session1 共 {int(is_s1.sum())} 行，`score < tau` 比例；Session1 全部为未注册者是强推断 |
| 10 | S2 接受 | {S2_acceptance:.4f} | Test session2 共 {int((~is_s1).sum())} 行，`score ≥ tau` 比例；Session2 含正常+对抗样本，该值不是 C 或 RA |
| 11 | LOSO 拒识 | {fm['loso_rejection']:.4f} | 严格 20 折 subject-LOSO：留出身份不参与标准化/PCA/原型/阈值；逐折 tau 后 pooled micro 未知拒识率 |

补充：官方 Test AUC / F1 = **N/A**，原因是 Test 没有公开真实身份标签；本提交不产生官方 C/RA/ASafeR/U 的估计。

## 2. 一致性校验

| 校验 | 结果 |
|---|---|
| 重载 `deploy_frozen.pt` 复算 Test 分数，与冻结逐样本缓存 `{CFG['reference_scores']['path']}` 逐行比对 | 最大绝对差 `{max_abs_score_diff:.3e}`；最近原型身份逐行一致 `{pred_identical}` |
| S1 拒识 vs E228 记录值 0.963732 | `{S1_rejection:.6f}`，Δ = `{dS1:+.3e}`（容差 1e-4，通过） |
| S2 接受 vs E228 记录值 0.304298 | `{S2_acceptance:.6f}`，Δ = `{dS2:+.3e}`（容差 1e-4，通过） |
| Test S2-vs-S1 分离 AUC vs E228 记录值 0.791840 | `{sep_auc:.6f}`，Δ = `{dAUC:+.3e}` |
| 行数与顺序 | `test.csv` {len(sub)} 行；`sample` 与 `test_metadata.csv` 逐行完全一致 |
| 负对照：E228/cache 的 unweighted plain 缓存 | 最大绝对差 `{plain_diff:.6f}`（预期非零；该缓存对应无块权重的 plain 候选，不是本冻结模型） |

说明：E228 `cache/` 只保存了 `e226_all_blocks_v2/plain` 等候选的逐样本 Test 分数；冻结候选 `e226_auc2W` 的逐样本分数复用自 **E227** 的冻结缓存 `test_v1_auc2_lofo.npz`（E228 `build_frozen()` 即以此为参考），因此一致性比对使用该文件。

## 3. 标签分布

- 拒识（标签 0）：**{n_rejected} / {len(label)} = {n_rejected / len(label):.4f}**
- 接受（标签 1–20）：**{len(label) - n_rejected} / {len(label)} = {(len(label) - n_rejected) / len(label):.4f}**
- 完全未被预测到的身份：**{', '.join(map(str, unpredicted)) if unpredicted else '无'}**

| 身份 | 预测样本数 |
|---:|---:|
""" + "".join(f"| {i} | {identity_counts[i]} |\n" for i in range(1, 21)) + f"""
## 4. 会话/任务分层（审计）

| task | S1 行数 | S1 拒识 | S2 行数 | S2 接受 |
|---|---:|---:|---:|---:|
""" + "".join(
        f"| {t} | {per_task[t]['n_S1']} | {per_task[t]['S1_rejection']:.4f} | "
        f"{per_task[t]['n_S2']} | {per_task[t]['S2_acceptance']:.4f} |\n"
        for t in TASKS
    ) + f"""

运行耗时 {runtime:.1f}s。完整流程见 `report.md`，逐样本审计见 `test_predictions_detailed.csv`。
"""
    (HERE / "metrics_summary.md").write_text(metrics_md, encoding="utf8")

    weights_table = "\n".join(
        f"| `{b}` | {int(CFG['representation']['block_dims'][b])} | {float(w):.4f} |"
        for b, w in zip(model["block_order"], model["block_weights"])
    )
    report_md = f"""# E230 提交产物：E228 冻结模型的 Test 逐样本预测

## 0. 摘要

- 本提交只加载 `{CFG['frozen_model']}` 做前向推理；**没有重新训练、没有改阈值、没有改特征/块权重/PCA/原型**。
- 冻结候选：`{model['candidate']}`（表征 `{model['rep']}`，打分 `{model['variant']}`，来源 `{model['source']}`）。
- 阈值：`tau = {tau:.9f}`（`A* = {A:.9f}`）；`score < tau` → 标签 0，否则输出最近原型身份 1–20。
- 输出：`test.csv`（{len(sub)} 行，与 `test_metadata.csv` 原始顺序逐行一致，表头 `Sample,Label`）与 `test_predictions_detailed.csv`（含 score、pred_identity、rejected、label）。
- 一致性：重载模型复算分数与冻结缓存最大绝对差 **{max_abs_score_diff:.3e}**，身份预测逐行一致 **{pred_identical}**；S1 拒识 {S1_rejection:.6f}、S2 接受 {S2_acceptance:.6f}，与 E228 记录值一致（Δ 均 < 1e-4）。
- 主要风险：Test 无身份标签；S2 接受仅 {S2_acceptance:.4f}（拒识率 {S2_rejection:.4f}），远高于官方 Session2 拒识 ≤2% 的验收要求（等价于接受率应 ≥98%）；Test 自 E65 起被反复查看，S1/S2 带评估偏倚。

## 1. 冻结模型与不变项

| 项 | 值 |
|---|---|
| 模型文件 | `{CFG['frozen_model']}` |
| 候选 | `{model['candidate']}` |
| 表征 | `{model['rep']}`（10 个特征块顺序拼接，{model['dim']} 维） |
| 打分规则 | `score = -min Euclidean distance to same-task prototypes`；identity = 最近原型所属身份 |
| 门控 | 全局单一阈值 `tau = {tau:.9f}`；不做按身份/按 task 阈值 |
| PCA | {model['pca']} 维（fit 行中心化协方差特征分解） |
| 原型 | 每 (task, identity) 独立 k-means，k = {model['k']}；20 身份 × 4 task = 960 个原型 |
| 标定 | `A* = {A:.9f}`；阈值来自 Train fit 行 inner-OOF known 分数的 (1−A*) 分位 |
| 输入 | 18 通道、300 Hz、1200 点/trial；Test {len(meta)} 条 |

## 2. 特征块与部署权重

按以下顺序拼接特征，然后在 PCA 之前对每个维度乘 `sqrt(w_b)`（`w_b` 为部署模型内置块权重，取自 E227 auc2_lofo 的 20 折留一折均值，归一化到均值 1）：

| 特征块 | 维度 | 部署权重 |
|---|---:|---:|
{weights_table}

## 3. 推理流程

1. 按冻结特征定义读取 Test 的 10 个特征块，按上表顺序拼接成 `x`（22251 × 1569）。
2. 按 `task`（EC / EO / Imagery / SSVEP）使用 Train fit 行的均值/标准差做逐维 z-score（标准差为总体标准差 + 1e-6）。
3. 逐维乘 `sqrt(w_b)`（块权重在模型 `state['w']` 中）。
4. 减去 fit 行均值 `pm`，乘 PCA 投影矩阵 `W`（1569 × 96），得到 96 维表示 `z`。
5. 在同一 task 的 20 组 × 12 个原型上计算欧氏距离，取最小距离 `d_min` 及其原型身份。
6. `score = -d_min`；若 `score < tau` 则输出 0（拒识/未注册），否则输出最近原型身份（1–20）。
7. 按 `test_metadata.csv` 原始顺序写入 `test.csv` 与 `test_predictions_detailed.csv`。

## 4. 预测分布

- 拒识（0）：**{n_rejected} / {len(label)} = {n_rejected / len(label):.4f}**
- 接受（1–20）：**{len(label) - n_rejected} / {len(label)} = {(len(label) - n_rejected) / len(label):.4f}**
- Session1（{int(is_s1.sum())} 行）拒识 **{S1_rejection:.4f}**；Session2（{int((~is_s1).sum())} 行）接受 **{S2_acceptance:.4f}**（拒识 {S2_rejection:.4f}）。
- 未被预测到的身份：**{', '.join(map(str, unpredicted)) if unpredicted else '无'}**。

| task | S1 行数 | S1 拒识 | S2 行数 | S2 接受 | 全部行拒识数 |
|---|---:|---:|---:|---:|---:|
""" + "".join(
        f"| {t} | {per_task[t]['n_S1']} | {per_task[t]['S1_rejection']:.4f} | "
        f"{per_task[t]['n_S2']} | {per_task[t]['S2_acceptance']:.4f} | "
        f"{per_task[t]['rejected_total']} |\n"
        for t in TASKS
    ) + f"""

逐身份预测数见 `metrics_summary.md` 第 3 节。

## 5. 一致性校验

| 校验 | 结果 |
|---|---|
| 重载冻结模型复算 Test 分数 vs `{CFG['reference_scores']['path']}` | 最大绝对差 **{max_abs_score_diff:.3e}**；身份预测逐行一致 **{pred_identical}** |
| S1 拒识 vs E228 的 0.963732 | **{S1_rejection:.6f}**，Δ = {dS1:+.3e}（通过 1e-4） |
| S2 接受 vs E228 的 0.304298 | **{S2_acceptance:.6f}**，Δ = {dS2:+.3e}（通过 1e-4） |
| 分离 AUC vs E228 的 0.791840 | **{sep_auc:.6f}**，Δ = {dAUC:+.3e} |
| 行数/顺序 | {len(sub)} 行；`sample` 与 `test_metadata.csv` 逐行一致 |
| LOSO 拒识复算 | **{fm['loso_rejection']:.4f}**（E228 记录 0.900043；{fm['n_unknown']} 个未知评估样本） |
| 拟合准确率复算 | **{fit_acc:.4f}**（{n_fit_rows} 个 fit 行） |

E228 `cache/` 只包含 `e226_all_blocks_v2/plain` 等候选的逐样本 Test 分数，不含冻结 auc2W 的逐样本分数；后者由 E228 `build_frozen()` 从 E227 的 `test_v1_auc2_lofo.npz` 读取，因此本报告以该文件作为逐行比对基准。以 E228 的 unweighted plain 缓存作负对照时最大绝对差为 {plain_diff:.6f}，与预期一致（候选不同，不应相等）。

## 6. 指标可用性

详细表见 `metrics_summary.md`。要点：

- 可计算并已给出：拟合准确率 {fit_acc:.4f}；真正留块准确率 {fm['block_accuracy']:.4f}；伪未知 LOSO AUROC {fm['auc_mean']:.4f}；U@95 {fm['u95_perfold']:.4f}；S1 拒识 {S1_rejection:.4f}；S2 接受 {S2_acceptance:.4f}；LOSO 拒识 {fm['loso_rejection']:.4f}。
- **N/A**：官方 Test C、身份 macro-F1（需 Test 真实身份标签）；RA、ASafeR（需官方对抗样本及原始身份标签）；官方 Test AUROC/F1（无标签）。官方未提供这些标签，本提交不以代理值冒充。

## 7. 限制与风险

1. **Test 无身份标签**：无法计算官方 C、身份 macro-F1、RA、ASafeR、U 与最终 Score；S1/S2 仅是会话分层的结构审计。
2. **Test 已被反复查看**：自 E65 起 Test 的 S1/S2 行为被多次读取，候选/工作点的比较含评估偏倚；本提交不是无偏泛化估计。
3. **S2 接受率仅 {S2_acceptance:.4f}**：对应 Session2 拒识率 {S2_rejection:.4f}。官方研发验收目标是 Session2 拒识 ≤2%（即接受 ≥98%），当前差距约 {100 * (0.98 - S2_acceptance):.1f} 个百分点。以当前门控直接提交，注册者正常样本会被大量拒识为 0，C/RA 会被严重压低。
4. **LOSO 与 Test 未知集难度不一致**：LOSO 的未知是同一采集批次内留出的已注册身份；Test S1 是 5 名未注册者且存在跨 Session/跨域漂移。LOSO 拒识 {fm['loso_rejection']:.4f} 与 Test S1 拒识 {S1_rejection:.4f} 不能互相换算，LOSO 0.90 的约束不代表 Test 未知拒识达标。
5. **S2 混合正常与对抗样本**：官方未给攻击标签，无法把 S2 接受率拆成 C 与 RA/ASafeR；拒识已知样本虽可能提高 ASafeR，但会直接降低 C/RA。
6. **单一全局阈值**：tau 在 Train fit 行上标定，未使用 Test 统计量；但 Train 只有 session1，跨会话漂移不可观测，阈值对 S2 偏严。
7. **标签 0 的语义**：模型对已知但分数低于 tau 的样本也输出 0；在官方评分中这会同时影响 C、RA 与 ASafeR，不能解释为"只拒未注册者"。

## 8. 交付文件与复现

| 文件 | 内容 |
|---|---|
| `test.csv` | 22251 行，表头 `Sample,Label`，原始顺序，Label ∈ 0–20 整数 |
| `test_predictions_detailed.csv` | sample / path / task / session / score / pred_identity / rejected / label |
| `metrics_summary.md` | 十一项指标、N/A 原因、一致性校验、标签分布 |
| `report.md` | 本文件 |
| `run_e230.py` | 推理、复算校验、产物生成 |
| `config.json` | 冻结模型、阈值、特征路径、块顺序/权重、锚点与容差 |
| `run.log` | 本次运行日志 |

复现：

```text
.venv\\Scripts\\python.exe research/E230_submission_artifacts/run_e230.py
```

本次运行耗时 {runtime:.1f}s，设备 {torch.cuda.get_device_name(0)}。未修改任何历史实验，未写入 `research/E229_raw_robust_encoder/`，未提交 git。
"""
    (HERE / "report.md").write_text(report_md, encoding="utf8")
    log(f"wrote metrics_summary.md and report.md ({runtime:.1f}s total)")


if __name__ == "__main__":
    main()
