import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(r"d:\WORK\BCIC2026\BrainPrint")
OUT = ROOT / "research/session_blind_candidate/calibrated_pseudo_ground_truth_v4"
SCENARIOS = OUT / "scenarios"
P1 = ROOT / "commit/answer/赞美诗_第1次.csv"
P2 = ROOT / "commit/answer/赞美诗_第2次.csv"
META = ROOT / "BrainPrint_cache/test_metadata.csv"
CANDIDATES = {
    "full_none": ROOT / "research/session_blind_candidate/five_candidates/candidate_full_none.csv",
    "base_none": ROOT / "research/session_blind_candidate/five_candidates/candidate_base_none.csv",
    "full_target_zscore": ROOT / "research/session_blind_candidate/five_candidates/candidate_full_target_zscore.csv",
    "base_target_zscore": ROOT / "research/session_blind_candidate/five_candidates/candidate_base_target_zscore.csv",
    "full_riemannian_recenter": ROOT / "research/session_blind_candidate/five_candidates/candidate_full_riemannian_recenter.csv",
}
TARGETS = {"official_submission_1": 69.17, "official_submission_2": 67.94}
ASAFER_VALUES = (0.80, 0.95, 1.00)
U_VALUES = (0.80, 1.00)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(c, ra, asafer, u):
    return 100.0 * c**0.40 * (0.7 * ra + 0.3 * asafer) ** 0.35 * u**0.25


def implied_ra(target, c, asafer, u):
    a = (target / 100.0 / (c**0.40 * u**0.25)) ** (1.0 / 0.35)
    return (a - 0.3 * asafer) / 0.7


def mode(values):
    counts = Counter(int(x) for x in values)
    best = max(counts.values())
    return min(k for k, v in counts.items() if v == best)


formal = pd.read_csv(P1)
p2 = pd.read_csv(P2)
meta = pd.read_csv(META)
if len(formal) != 22251 or not formal.Sample.equals(p2.Sample):
    raise ValueError("正式提交行数或 Sample 顺序不符合要求")
if not {"sample", "task"}.issubset(meta.columns):
    raise ValueError("测试元数据缺少 sample/task")

frames = {"official_submission_1": formal, "official_submission_2": p2}
for name, path in CANDIDATES.items():
    frame = pd.read_csv(path)
    if len(frame) != len(formal) or not formal.Sample.equals(frame.Sample):
        raise ValueError(f"{name} 行数或 Sample 顺序不一致")
    if not np.isin(frame.Label.to_numpy(), np.arange(21)).all():
        raise ValueError(f"{name} Label 非 0-20")
    frames[name] = frame

base = pd.DataFrame({"Sample": formal.Sample, "p1": formal.Label, "p2": p2.Label})
for name in CANDIDATES:
    base[name] = frames[name].Label.to_numpy()
base = base.merge(meta[["sample", "task"]], left_on="Sample", right_on="sample", how="left").drop(columns="sample")
base["common_official"] = base.p1.eq(base.p2)
base["discordant"] = ~base.common_official

# Evidence-first labels. No random flips and no use of session metadata.
label_cols = ["full_none", "base_none", "full_target_zscore", "base_target_zscore", "full_riemannian_recenter"]
pred_matrix = base[label_cols].to_numpy(dtype=int)
main = np.empty(len(base), dtype=int)
conservative = np.empty(len(base), dtype=int)
liberal = np.empty(len(base), dtype=int)
evidence = []
confidence = []
ambiguity = []
reason = []
for i, row in base.iterrows():
    candidates = pred_matrix[i]
    if row.common_official:
        main[i] = conservative[i] = liberal[i] = int(row.p1)
        evidence.append("official_common_label")
        confidence.append("high_observed")
        ambiguity.append("")
        reason.append("two official submissions agree")
        continue
    counts = Counter(int(x) for x in candidates)
    best_count = max(counts.values())
    best_labels = sorted(k for k, v in counts.items() if v == best_count)
    nonzero = [int(x) for x in candidates if int(x) != 0]
    nonzero_mode = mode(nonzero) if nonzero else 0
    main[i] = best_labels[0]
    conservative[i] = 0 if counts.get(0, 0) >= 2 else main[i]
    liberal[i] = nonzero_mode if nonzero else 0
    if best_count >= 3:
        evidence.append("candidate_majority")
        confidence.append("medium")
    elif len(set(candidates)) <= 2:
        evidence.append("candidate_partial_consensus")
        confidence.append("low_medium")
    else:
        evidence.append("candidate_ambiguous")
        confidence.append("low")
    ambiguity.append("{" + ",".join(map(str, sorted(set(candidates)))) + "}")
    reason.append("discordant official labels; candidate vote without random flip")

base["label_main"] = main
base["label_conservative"] = conservative
base["label_liberal"] = liberal
base["evidence_source"] = evidence
base["confidence"] = confidence
base["ambiguity_set"] = ambiguity
base["rule_reason"] = reason

scenario_specs = {
    "main_evidence_consensus": ("label_main", "minimal_extra_assumption"),
    "conservative_reject_ambiguous": ("label_conservative", "ambiguous_rows_to_unknown_0"),
    "liberal_known_consensus": ("label_liberal", "nonzero_candidate_consensus"),
}
input_paths = [P1, P2, META, *CANDIDATES.values()]
input_sha = {str(p): sha256(p) for p in input_paths}
rows = []
anchor_rows = []
summary_rows = []
for scenario, (label_col, scenario_rule) in scenario_specs.items():
    sdir = SCENARIOS / scenario
    sdir.mkdir(parents=True, exist_ok=True)
    labels = base[label_col].to_numpy(dtype=int)
    pseudo = pd.DataFrame({"Sample": base.Sample, "Label": labels})
    pseudo.to_csv(sdir / "pseudo_answer.csv", index=False)
    category = np.where(labels == 0, "unknown_hypothesis", "known_identity_hypothesis")
    category = np.where(base.common_official, "observed_label_category_unobserved", category)
    groups = base[["Sample", "task", "p1", "p2", *label_cols, "evidence_source", "confidence", "ambiguity_set", "rule_reason"]].copy()
    groups.insert(1, "Label", labels)
    groups["category_hypothesis"] = category
    groups["clean_known_or_adversarial_known"] = np.where(labels == 0, "not_applicable_unknown", "unobserved_clean_or_adversarial")
    groups["hard_constraint"] = np.where(base.common_official, "official_common_fixed", "candidate_evidence_rule")
    groups["scenario"] = scenario
    groups["scenario_rule"] = scenario_rule
    groups.to_csv(sdir / "pseudo_groups.csv", index=False)
    for method, frame in frames.items():
        pred = frame.Label.to_numpy(dtype=int)
        nonzero = labels != 0
        c = float((pred[nonzero] == labels[nonzero]).mean())
        f1 = float(f1_score(labels, pred, labels=list(range(21)), average="macro", zero_division=0))
        for asafer in ASAFER_VALUES:
            for u in U_VALUES:
                low = score(c, 0.0, asafer, u)
                high = score(c, 1.0, asafer, u)
                rows.append({"scenario": scenario, "method": method, "C_proxy": c, "macro_F1_against_pseudo": f1,
                             "ASafeR_assumption": asafer, "U_assumption": u, "score_RA0": low, "score_RA1": high,
                             "AUC": np.nan, "RA_status": "unobserved_conditional_range"})
        if method in TARGETS:
            for asafer in ASAFER_VALUES:
                for u in U_VALUES:
                    anchor_rows.append({"scenario": scenario, "method": method, "ASafeR": asafer, "U": u,
                                        "C_proxy": c, "implied_RA": implied_ra(TARGETS[method], c, asafer, u)})

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "scenario_metrics.csv", index=False)
pd.DataFrame(anchor_rows).to_csv(OUT / "anchor_implied_ra.csv", index=False)
summary = metrics.groupby("method", sort=False).agg(
    scenario_count=("scenario", "nunique"), C_proxy_mean=("C_proxy", "mean"),
    F1_mean=("macro_F1_against_pseudo", "mean"), F1_std=("macro_F1_against_pseudo", "std"),
    score_RA0_mean=("score_RA0", "mean"), score_RA1_mean=("score_RA1", "mean"),
    score_min=("score_RA0", "min"), score_max=("score_RA1", "max"), AUC=("AUC", "first")
).reset_index()
summary.to_csv(OUT / "candidate_mean_std_summary.csv", index=False)

# Provide one representative answer per scenario at the top level.
for scenario, (label_col, _) in scenario_specs.items():
    pd.DataFrame({"Sample": base.Sample, "Label": base[label_col]}).to_csv(OUT / f"pseudo_answer_{scenario}.csv", index=False)

# The main evidence-consensus scenario is the default deliverable; alternatives remain separate.
pd.DataFrame({"Sample": base.Sample, "Label": base.label_main}).to_csv(OUT / "pseudo_answer.csv", index=False)
base[["Sample", "label_main", "label_conservative", "label_liberal", "evidence_source", "confidence", "ambiguity_set", "rule_reason"]].to_csv(OUT / "pseudo_groups.csv", index=False)

formal_after_sha = {str(p): sha256(p) for p in (P1, P2)}
if formal_after_sha != {str(p): input_sha[str(p)] for p in (P1, P2)}:
    raise RuntimeError("正式提交 SHA-256 发生变化")
assumptions = {
    "purpose": "evidence-first synthetic calibration; not real Test labels or official score prediction",
    "official_formula": "Score=100*C^0.40*(0.7*RA+0.3*ASafeR)^0.35*U^0.25",
    "scenario_count": len(scenario_specs), "scenario_rules": {k: v[1] for k, v in scenario_specs.items()},
    "hard_constraints": ["common official labels retained", "no random label flips", "adversarial samples retain identity labels 1-20", "session metadata not used for label construction"],
    "unobserved": ["clean/adversarial/unknown category", "candidate RA", "candidate ASafeR", "candidate U"],
    "sensitivity": {"ASafeR": ASAFER_VALUES, "U": U_VALUES, "RA": "[0,1]"},
    "input_sha256": input_sha, "formal_sha256_after": formal_after_sha,
    "sample_count": len(base), "official_targets": TARGETS,
}
(OUT / "assumptions.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")
report = f"""# calibrated pseudo ground truth v4 — evidence-first scenarios

## 结论与边界

本目录生成 3 个证据优先情景，不是真实 Test 标签，也不声称恢复真实官方分。v4 不再对官方冲突行随机翻转标签，也不把对抗样本当成独立标签；依据赛题规则，对抗样本若为已知身份，其真实标签仍应为 1–20。由于类别来源和攻击身份不可观测，`category_hypothesis` 只表示假设，不是真值。

## 情景规则

- `main_evidence_consensus`：保留官方共识；冲突行使用五个候选的多数标签。
- `conservative_reject_ambiguous`：候选中至少两个拒识时设为 0，其余使用候选多数。
- `liberal_known_consensus`：冲突行优先采用非零候选的多数；没有非零证据才设为 0。

三种规则均不使用 session；逐行证据、置信等级、候选集合和原因在每个情景的 `pseudo_groups.csv` 中。

## 评估

`scenario_metrics.csv` 使用官方公式，并对未观测的 RA 显式报告 RA=0 与 RA=1 条件分；同时对 `ASafeR={ASAFER_VALUES}`、`U={U_VALUES}` 做敏感性分析。官方提交的 69.17/67.94 仅用于计算锚点所需的隐含 RA，记录在 `anchor_implied_ra.csv`，不把隐含 RA 当作真实标签证据。

AUC：不可计算（Test 无真实标签，候选 CSV 无连续概率输出）。F1：是相对伪答案的审计 F1，不是官方 Test F1。`candidate_mean_std_summary.csv` 给出当前条件分和 F1 汇总，不能替代官方成绩。

## 完整性

输入 Sample 均已对齐；正式提交 SHA-256 运行前后保持不变；样本数为 {len(base)}。标签 0/1–20 和对抗样本语义依据 `赛题说明.md`。
"""
(OUT / "report.md").write_text(report, encoding="utf-8")
print(json.dumps({"out": str(OUT), "scenarios": list(scenario_specs), "rows": len(metrics), "formal_unchanged": True}, ensure_ascii=False))
