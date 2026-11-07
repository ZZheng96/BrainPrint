import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(r"d:\WORK\BCIC2026\BrainPrint")
V4 = ROOT / "research/session_blind_candidate/calibrated_pseudo_ground_truth_v4"
OUT = V4 / "five_method_eval"
PSEUDO = V4 / "pseudo_answer.csv"
P1 = ROOT / "commit/answer/赞美诗_第1次.csv"
P2 = ROOT / "commit/answer/赞美诗_第2次.csv"
CANDIDATES = {
    "full_none": ROOT / "research/session_blind_candidate/five_candidates/candidate_full_none.csv",
    "base_none": ROOT / "research/session_blind_candidate/five_candidates/candidate_base_none.csv",
    "full_target_zscore": ROOT / "research/session_blind_candidate/five_candidates/candidate_full_target_zscore.csv",
    "base_target_zscore": ROOT / "research/session_blind_candidate/five_candidates/candidate_base_target_zscore.csv",
    "full_riemannian_recenter": ROOT / "research/session_blind_candidate/five_candidates/candidate_full_riemannian_recenter.csv",
}
ASAFER_VALUES = (0.80, 0.95, 1.00)
U_VALUES = (0.80, 1.00)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(c, ra, asafer, u):
    return 100.0 * c**0.40 * (0.7 * ra + 0.3 * asafer) ** 0.35 * u**0.25


formal = pd.read_csv(P1)
p2 = pd.read_csv(P2)
pseudo = pd.read_csv(PSEUDO)
if len(formal) != 22251 or not formal.Sample.equals(p2.Sample) or not formal.Sample.equals(pseudo.Sample):
    raise ValueError("正式提交或 v4 主伪答案行数/顺序不一致")
labels = pseudo.Label.to_numpy(dtype=int)
if not np.isin(labels, np.arange(21)).all():
    raise ValueError("v4 主伪答案含非法 Label")

rows = []
for method, path in CANDIDATES.items():
    frame = pd.read_csv(path)
    if len(frame) != len(formal) or not formal.Sample.equals(frame.Sample):
        raise ValueError(f"{method} 行数或 Sample 顺序不一致")
    pred = frame.Label.to_numpy(dtype=int)
    if not np.isin(pred, np.arange(21)).all():
        raise ValueError(f"{method} 含非法 Label")
    nonzero = labels != 0
    c = float((pred[nonzero] == labels[nonzero]).mean())
    overall = float((pred == labels).mean())
    f1 = float(f1_score(labels, pred, labels=list(range(21)), average="macro", zero_division=0))
    for asafer in ASAFER_VALUES:
        for u in U_VALUES:
            rows.append({
                "method": method,
                "n": len(pred),
                "pseudo_nonzero_n": int(nonzero.sum()),
                "C_proxy_nonzero_accuracy": c,
                "overall_accuracy_against_pseudo": overall,
                "macro_F1_against_pseudo": f1,
                "ASafeR_assumption": asafer,
                "U_assumption": u,
                "score_RA0": score(c, 0.0, asafer, u),
                "score_RA1": score(c, 1.0, asafer, u),
                "AUC": np.nan,
                "RA_status": "unobserved; conditional range only",
            })

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "candidate_metrics.csv", index=False)
summary = metrics.groupby("method", sort=False).agg(
    pseudo_nonzero_n=("pseudo_nonzero_n", "first"),
    C_proxy=("C_proxy_nonzero_accuracy", "first"),
    overall_accuracy=("overall_accuracy_against_pseudo", "first"),
    macro_F1=("macro_F1_against_pseudo", "first"),
    score_RA0_min=("score_RA0", "min"),
    score_RA0_max=("score_RA0", "max"),
    score_RA1_min=("score_RA1", "min"),
    score_RA1_max=("score_RA1", "max"),
).reset_index()
summary.to_csv(OUT / "candidate_mean_std_summary.csv", index=False)

input_paths = [P1, P2, PSEUDO, *CANDIDATES.values()]
assumptions = {
    "purpose": "five session-blind candidates evaluated against v4 main evidence-consensus pseudo answer",
    "pseudo_answer": str(PSEUDO),
    "formula": "Score=100*C^0.40*(0.7*RA+0.3*ASafeR)^0.35*U^0.25",
    "ASafeR_assumptions": ASAFER_VALUES,
    "U_assumptions": U_VALUES,
    "RA_assumption": "unknown; report explicit RA=0 and RA=1 conditional bounds",
    "AUC": "not computable: no Test truth and candidate CSV contains hard labels only",
    "F1": "macro-F1 against v4 pseudo answer, audit metric only",
    "input_sha256": {str(p): sha256(p) for p in input_paths},
    "formal_submission_sha256_after": {str(p): sha256(p) for p in (P1, P2)},
}
(OUT / "assumptions.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# v4 主伪答案下的五个无 Session 方案评估",
    "",
    "## 结论与边界",
    "",
    "本次使用 v4 的 `pseudo_answer.csv`（main_evidence_consensus）评估五个无 Session 方案。结果是相对伪答案的条件估计，不是真实 Test 官方分。伪答案依据官方共同标签和候选证据构造，不再随机翻转标签；但 Test 的真实标签、clean/adversarial/unknown 分组和候选 RA/ASafeR/U 仍不可观测。",
    "",
    "## 结果",
    "",
    "| 方法 | C 代理 | 相对伪答案总体准确率 | 宏 F1 | RA=0 条件分范围 | RA=1 条件分范围 | AUC |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for _, r in summary.iterrows():
    lines.append(f"| {r['method']} | {r['C_proxy']:.4f} | {r['overall_accuracy']:.4f} | {r['macro_F1']:.4f} | {r['score_RA0_min']:.2f}–{r['score_RA0_max']:.2f} | {r['score_RA1_min']:.2f}–{r['score_RA1_max']:.2f} | null |")
lines += [
    "",
    "`score_RA0`/`score_RA1` 只表示在固定 ASafeR/U 敏感性设置下的上下界；候选没有真实攻击版本标签，因此不能选定真实 RA。AUC 不可计算；F1 是相对 v4 伪答案的审计指标，不是官方 Test F1。",
    "",
    "## 文件",
    "",
    "- `candidate_metrics.csv`：每个方法在 ASafeR/U 敏感性组合下的条件结果。",
    "- `candidate_mean_std_summary.csv`：方法汇总。",
    "- `assumptions.json`：输入哈希和假设。",
]
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"rows": len(metrics), "methods": list(CANDIDATES), "pseudo": str(PSEUDO)}, ensure_ascii=False))
