import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(r"d:\WORK\BCIC2026\BrainPrint")
V3 = ROOT / "research/E11_session_blind_candidate/calibrated_pseudo_ground_truth_v3"
OUT = V3 / "five_method_eval"
SCENARIOS = V3 / "scenarios"
CANDIDATES = ROOT / "research/E11_session_blind_candidate/five_candidates"
P1 = ROOT / "commit/answer/赞美诗_第1次.csv"
P2 = ROOT / "commit/answer/赞美诗_第2次.csv"
ASAFER = 0.95
U = 1.0
TARGETS = {"official_submission_1": 69.17, "official_submission_2": 67.94}
METHOD_FILES = {
    "full_none": "candidate_full_none.csv",
    "base_none": "candidate_base_none.csv",
    "full_target_zscore": "candidate_full_target_zscore.csv",
    "base_target_zscore": "candidate_base_target_zscore.csv",
    "full_riemannian_recenter": "candidate_full_riemannian_recenter.csv",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(c, ra, asafer=ASAFER, u=U):
    return 100.0 * c**0.40 * (0.7 * ra + 0.3 * asafer) ** 0.35 * u**0.25


def calc_c(pred, pseudo):
    mask = pseudo != 0
    return float((pred[mask] == pseudo[mask]).sum() / max(mask.sum(), 1))


formal = pd.read_csv(P1)
p2 = pd.read_csv(P2)
if len(formal) != 22251 or not formal.Sample.equals(p2.Sample):
    raise ValueError("正式提交行数或 Sample 顺序不符合 v3")
input_sha = {str(p): sha256(p) for p in (P1, P2)}

candidate_frames = {}
candidate_sha = {}
for method, filename in METHOD_FILES.items():
    path = CANDIDATES / filename
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != len(formal) or not formal.Sample.equals(frame.Sample):
        raise ValueError(f"{method} 的行数或 Sample 顺序不符合正式提交")
    labels = frame.Label.to_numpy(dtype=int)
    if not np.isin(labels, np.arange(21)).all():
        raise ValueError(f"{method} 含非法 Label")
    candidate_frames[method] = frame
    candidate_sha[method] = sha256(path)

rows = []
scenario_meta = []
for i in range(1, 17):
    scenario = f"scenario_{i:02d}"
    pseudo_path = SCENARIOS / scenario / "pseudo_answer.csv"
    groups_path = SCENARIOS / scenario / "pseudo_groups.csv"
    pseudo = pd.read_csv(pseudo_path)
    groups = pd.read_csv(groups_path)
    if len(pseudo) != len(formal) or not formal.Sample.equals(pseudo.Sample):
        raise ValueError(f"{scenario} 的 pseudo_answer 顺序不符合正式提交")
    if not formal.Sample.equals(groups.Sample):
        raise ValueError(f"{scenario} 的 pseudo_groups 顺序不符合正式提交")
    pseudo_labels = pseudo.Label.to_numpy(dtype=int)
    category_counts = groups.category.value_counts().to_dict()
    seed = int(groups.seed.iloc[0])
    scenario_meta.append({"scenario": scenario, "seed": seed, "category_counts": json.dumps(category_counts, ensure_ascii=False)})
    for method, frame in candidate_frames.items():
        pred = frame.Label.to_numpy(dtype=int)
        c = calc_c(pred, pseudo_labels)
        f1 = float(f1_score(pseudo_labels, pred, labels=list(range(21)), average="macro", zero_division=0))
        # v3 has no candidate attack-version labels; retain the same explicit conditional RA range.
        rows.append({
            "scenario": scenario,
            "seed": seed,
            "method": method,
            "n": len(pred),
            "C": c,
            "RA": np.nan,
            "ASafeR": ASAFER,
            "U": U,
            "score": np.nan,
            "score_range_low": score(c, 0.0),
            "score_range_high": score(c, 1.0),
            "macro_F1_against_pseudo": f1,
            "AUC": np.nan,
            "RA_status": "not_observed; explicit conditional RA range [0,1]",
            "category_counts": json.dumps(category_counts, ensure_ascii=False),
        })

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "candidate_scenario_metrics.csv", index=False)

summary = []
for method, group in metrics.groupby("method", sort=False):
    low = group.score_range_low
    high = group.score_range_high
    summary.append({
        "method": method,
        "scenario_count": len(group),
        "conditional_mean_low_RA0": float(low.mean()),
        "conditional_std_low_RA0": float(low.std(ddof=1)),
        "conditional_min_low_RA0": float(low.min()),
        "conditional_max_low_RA0": float(low.max()),
        "conditional_mean_high_RA1": float(high.mean()),
        "conditional_std_high_RA1": float(high.std(ddof=1)),
        "conditional_min_high_RA1": float(high.min()),
        "conditional_max_high_RA1": float(high.max()),
        "AUC": np.nan,
        "F1_mean": float(group.macro_F1_against_pseudo.mean()),
        "F1_std": float(group.macro_F1_against_pseudo.std(ddof=1)),
        "F1_min": float(group.macro_F1_against_pseudo.min()),
        "F1_max": float(group.macro_F1_against_pseudo.max()),
        "RA_status": "not_observed; explicit conditional RA range [0,1]",
    })
pd.DataFrame(summary).to_csv(OUT / "candidate_mean_std_summary.csv", index=False)

# Confirm the two v3 official anchors remain exact and their source SHA is unchanged.
v3_metrics = pd.read_csv(V3 / "scenario_metrics.csv")
anchor_errors = {}
for method, target in TARGETS.items():
    values = v3_metrics.loc[v3_metrics.method == method, "score"]
    anchor_errors[method] = float(np.max(np.abs(values - target)))
formal_after_sha = {str(p): sha256(p) for p in (P1, P2)}
if formal_after_sha != input_sha:
    raise RuntimeError("正式提交 SHA-256 发生变化")

assumptions = {
    "purpose": "five session-blind candidates evaluated on the exact v3 synthetic scenarios; not real official scores",
    "formula": "Score=100*C^0.40*(0.7*RA+0.3*ASafeR)^0.35*U^0.25",
    "scenario_count": 16,
    "fixed_proxies": {"ASafeR": ASAFER, "U": U},
    "candidate_rule": "Reuse v3: C is agreement with pseudo labels on nonzero pseudo rows; candidate files have no attack-version labels, so RA is not imputed and Score is reported conditionally for RA in [0,1].",
    "candidate_sha256": candidate_sha,
    "formal_submission_sha256_before": input_sha,
    "formal_submission_sha256_after": formal_after_sha,
    "formal_submissions_unchanged": formal_after_sha == input_sha,
    "anchor_targets": TARGETS,
    "anchor_max_abs_error": anchor_errors,
    "candidate_count": len(METHOD_FILES),
    "rows": len(metrics),
    "scenario_meta": scenario_meta,
}
(OUT / "assumptions.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")

report_lines = [
    "# calibrated_pseudo_ground_truth_v3：五个无 Session 依赖候选评估",
    "",
    "## 结论与边界",
    "",
    "本次严格复用 v3 已生成的 16 份合成场景、伪答案、类别分配、公式和假设，评估五个无 Session 依赖候选。结果是**合成校准条件分**，不是真实官方分，也不是 Test 真实标签下的官方 AUC/F1。",
    "",
    "v3 公式为 `Score=100*C^0.40*(0.7*RA+0.3*ASafeR)^0.35*U^0.25`，固定条件代理为 `ASafeR=0.95`、`U=1.0`。候选 CSV 没有攻击样本的原始身份标签或攻击版本标签，因此不能合法推断 RA；依照 v3 同样规则，RA 保持未观测，并对 `RA∈[0,1]` 输出 Score 条件范围，没有隐藏或填补 RA。",
    "",
    "## 两次正式提交锚点",
    "",
    f"- 第一次正式提交：目标 69.17；复用 v3 的 16 个场景最大绝对误差 `{anchor_errors['official_submission_1']:.15g}`。",
    f"- 第二次正式提交：目标 67.94；复用 v3 的 16 个场景最大绝对误差 `{anchor_errors['official_submission_2']:.15g}`。",
    "- 两次提交仍作为 v3 的校准锚点，不把候选结果冒充官方反馈。",
    "",
    "## 五个候选及结果摘要",
    "",
    "| 候选 | 场景数 | Score 条件均值 RA=0 | Score 条件均值 RA=1 | Score 范围（跨场景） | F1 均值±标准差 | AUC |",
    "|---|---:|---:|---:|---:|---:|---|",
]
summary_df = pd.DataFrame(summary)
for _, r in summary_df.iterrows():
    report_lines.append(
        f"| {r['method']} | {int(r['scenario_count'])} | {r['conditional_mean_low_RA0']:.4f} | {r['conditional_mean_high_RA1']:.4f} | "
        f"{r['conditional_min_low_RA0']:.4f}–{r['conditional_max_high_RA1']:.4f} | {r['F1_mean']:.4f} ± {r['F1_std']:.4f} | null（无概率分数） |"
    )
report_lines += [
    "",
    "`candidate_scenario_metrics.csv` 提供每个候选×16 场景的 `C、RA、ASafeR、U、Score` 字段；由于 RA 未观测，`score` 为 null，同时提供 `score_range_low`/`score_range_high`。`candidate_mean_std_summary.csv` 提供条件均值、标准差和范围。",
    "",
    "## AUC/F1 说明",
    "",
    "AUC：null。五个候选文件只包含硬标签，没有可用于 ROC 的连续概率分数，且 Test 没有公开真实标签。F1：报告的是候选标签相对每个合成伪答案的 macro-F1，已逐场景记录并汇总均值/标准差/范围；它不是官方 Test F1。",
    "",
    "## 完整性核验",
    "",
    "- 五个候选均核验为 22,251 行，Sample 顺序与正式提交一致，Label 为 0–20 整数；每个候选对应 16 行结果，共 80 行。",
    "- 两次正式提交的 SHA-256 在运行前后保持不变；候选 SHA-256 和核验摘要写入 `assumptions.json`。",
    "- 详细场景类别计数、seed、RA 条件范围和 F1 位于 `candidate_scenario_metrics.csv`；本目录脚本为 `run.py`。",
]
(OUT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print(json.dumps({"rows": len(metrics), "out": str(OUT), "formal_submissions_unchanged": formal_after_sha == input_sha, "anchor_max_abs_error": anchor_errors}, ensure_ascii=False))
