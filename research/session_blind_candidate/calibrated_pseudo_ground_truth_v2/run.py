import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(r"d:\WORK\BCIC2026\BrainPrint")
OUT = ROOT / "research/session_blind_candidate/calibrated_pseudo_ground_truth_v2"
P1 = ROOT / "commit/answer/赞美诗_第1次.csv"
P2 = ROOT / "commit/answer/赞美诗_第2次.csv"
V1 = ROOT / "research/session_blind_candidate/pseudo_ground_truth_v1/pseudo_answer.csv"
LOG = ROOT / "research/session_blind_candidate/nested_pseudounknown_corrspd_v1/candidate_logcov.csv"
CORR = ROOT / "research/session_blind_candidate/nested_pseudounknown_corrspd_v1/candidate_corrspd.csv"
META = ROOT / "BrainPrint_cache/test_metadata.csv"

SCORES = {"official_submission_1": 69.17, "official_submission_2": 67.94}
ASAFER_PROXY = 0.95
U_FIXED = 1.0


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_formula(c, ra, asafer, u):
    a = 0.7 * ra + 0.3 * asafer
    return 100.0 * c**0.40 * a**0.35 * u**0.25


def calibrated_ra(score, c):
    a = (score / 100.0 / (c**0.40 * U_FIXED**0.25)) ** (1.0 / 0.35)
    return (a - 0.3 * ASAFER_PROXY) / 0.7

p1 = pd.read_csv(P1)
p2 = pd.read_csv(P2)
v1 = pd.read_csv(V1)
log = pd.read_csv(LOG).rename(columns={"Label": "Label_logcov"})
corr = pd.read_csv(CORR).rename(columns={"Label": "Label_corrspd"})
meta = pd.read_csv(META)
if not (p1.Sample.equals(p2.Sample) and p1.Sample.equals(v1.Sample) and p1.Sample.equals(log.Sample) and p1.Sample.equals(corr.Sample)):
    raise ValueError("输入CSV Sample顺序不一致")
if len(p1) != 22251:
    raise ValueError("正式CSV行数不为22251")

x = p1.rename(columns={"Label": "Label_p1"}).merge(p2.rename(columns={"Label": "Label_p2"}), on="Sample")
x = x.merge(v1.rename(columns={"Label": "Label_v1"}), on="Sample")
x = x.merge(log, on="Sample").merge(corr, on="Sample")
x = x.merge(meta[["sample", "session", "task"]], left_on="Sample", right_on="sample", how="left").drop(columns="sample")
x["same_official"] = x.Label_p1.eq(x.Label_p2)
x["same_candidate"] = x.Label_logcov.eq(x.Label_corrspd)
x["both_official_zero"] = x.Label_p1.eq(0) & x.Label_p2.eq(0)
x["pseudo_source"] = np.where(x.same_official, "official_common", "v1_candidate_tiebreak")
x["known_proxy"] = x.Label_v1.ne(0)
x["confidence"] = np.where(x.same_official, "high", "low")
x["Label"] = x.Label_v1.astype(int)
x[["Sample", "Label"]].to_csv(OUT / "pseudo_answer.csv", index=False)

# Groups only expose observable/assumed status; they do not claim hidden official categories.
g = x[["Sample", "session", "task", "Label_p1", "Label_p2", "Label_logcov", "Label_corrspd", "Label", "same_official", "same_candidate", "both_official_zero", "pseudo_source", "confidence"]].copy()
g["observed_group"] = np.select(
    [g.both_official_zero, g.same_official & ~g.both_official_zero, ~g.same_official],
    ["common_zero_observation", "common_nonzero_observation", "official_discordance"],
    default="unclassified")
g["hidden_official_category"] = "unknown_not_identifiable"
g["attack_status"] = "unknown_not_identifiable"
g.to_csv(OUT / "pseudo_groups.csv", index=False)

# Calibration uses pseudo nonzero agreement for C; U is explicitly fixed to 1 because the
# common-zero fraction is an observed composition proxy, not a test-set U estimate.
rows = []
for name, pred in [("official_submission_1", x.Label_p1), ("official_submission_2", x.Label_p2), ("logcov", x.Label_logcov), ("corrspd", x.Label_corrspd)]:
    c = float(((pred == x.Label) & (x.Label != 0)).sum() / max((x.Label != 0).sum(), 1))
    f1 = float(f1_score(x.Label, pred, labels=list(range(21)), average="macro", zero_division=0))
    if name.startswith("official"):
        ra = calibrated_ra(SCORES[name], c)
        exact = official_formula(c, ra, ASAFER_PROXY, U_FIXED)
        score_low, score_high = exact, exact
        ra_status = "calibrated_implied_proxy"
    else:
        ra = None
        exact = None
        score_low = official_formula(c, 0.0, ASAFER_PROXY, U_FIXED)
        score_high = official_formula(c, 1.0, ASAFER_PROXY, U_FIXED)
        ra_status = "not_observed; range uses RA in [0,1]"
    rows.append({"method": name, "n": len(x), "C_pseudo_nonzero_agreement": c, "RA": ra,
                 "ASafeR": ASAFER_PROXY, "U": U_FIXED, "A": None if ra is None else 0.7 * ra + 0.3 * ASAFER_PROXY,
                 "score": exact, "score_range_low": score_low, "score_range_high": score_high,
                 "macro_F1_against_pseudo": f1, "AUC": None, "RA_status": ra_status})
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "candidate_eval.csv", index=False)

assumptions = {
    "purpose": "可审计校准伪答案/伪分项，不是恢复真实标签",
    "official_formula": "Score=100*C^0.40*A^0.35*U^0.25; A=0.7*RA+0.3*ASafeR",
    "non_identifiability": ["仅凭提交CSV和总分无法唯一识别真实RA、ASafeR、U", "测试集未提供Clean/Adversarial/Unknown分组"],
    "pseudo_answer": {"source": "v1 pseudo_answer.csv", "common_official_labels": "两次正式标签相同则观测保留；冲突行沿用v1候选支持裁决", "status": "代理标签"},
    "C_assumption": "C_pseudo_nonzero_agreement=候选/正式预测与伪答案在伪答案非零行上的一致率；不是官方Clean Accuracy",
    "U_assumption": {"value": U_FIXED, "reason": "固定为1.0的单独校准假设；共同0比例3033/22251=0.1363084805仅作为可观测组成记录，未冒充官方U"},
    "ASafeR_assumption": {"value": ASAFER_PROXY, "reason": "固定基准代理，不来自攻击样本；不能解释为真实ASafeR"},
    "RA_assumption": "正式提交按官方总分锚定反解的隐含代理；logcov/corrSPD缺少攻击版本，RA不编造，报告RA∈[0,1]条件范围",
    "candidate_range": "对候选使用同一C/U/ASafeR假设，RA未知范围[0,1]；因此仅为条件范围",
    "auc_f1": "AUC因只有硬标签且无概率分数设为null；macro-F1为相对伪答案的审计指标",
    "input_sha256": {str(p): sha256(p) for p in [P1, P2, V1, LOG, CORR]},
    "common_zero_count": int(x.both_official_zero.sum()),
    "common_zero_fraction": float(x.both_official_zero.mean()),
    "sample_count": len(x)
}
(OUT / "assumptions.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")

calibration = {"formula": assumptions["official_formula"], "official_targets": SCORES,
               "fixed": {"U": U_FIXED, "ASafeR_proxy": ASAFER_PROXY},
               "rows": rows,
               "official_errors": {r["method"]: abs(r["score"] - SCORES[r["method"]]) for r in rows if r["method"].startswith("official")}}
(OUT / "calibration_metrics.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

report = f"""# calibrated pseudo ground truth v2

## 结论

- 本目录只构造可审计的校准伪答案/伪分项，**不声称恢复真实标签或真实 RA/ASafeR/U**。
- 样本数：**{len(x)}**；两次正式标签共同为0：**{int(x.both_official_zero.sum())} / {len(x)} = {x.both_official_zero.mean():.12f}**。该比例仅作观测组成记录。
- 固定校准假设：`U=1.0`，`ASafeR=0.95`（均为明确代理/基准，不是真实测量）。C 使用伪答案非零行一致率。
- 两次正式总分通过同一官方公式严格校准；精确误差见 `calibration_metrics.json`，目标误差 `<1e-6`。

## 不可识别性与分组

赛题测试集同时含正常、对抗和未注册样本，但本地没有真实分组、攻击版本或真实标签。仅凭两个CSV及两个总分，未知量多于约束，真实 `RA/ASafeR/U` 不可唯一反解。`pseudo_groups.csv` 因此把隐藏官方类别与攻击状态明确记为 `unknown_not_identifiable`；session1 全部共同0只是可观察结构，不能直接等同官方未知子集的真值评估。

## 结果（统一公式的条件校准）

|方法|C（伪答案条件）|RA|ASafeR|U|Score/范围|AUC|macro-F1（伪答案）|
|---|---:|---:|---:|---:|---:|---:|---:|
"""
for r in rows:
    score = f"{r['score']:.12f}" if r['score'] is not None else f"[{r['score_range_low']:.6f}, {r['score_range_high']:.6f}]"
    ra = "隐含代理" if r['RA'] is not None else "未知"
    report += f"|{r['method']}|{r['C_pseudo_nonzero_agreement']:.12f}|{ra}|{ASAFER_PROXY:.2f}|{U_FIXED:.1f}|{score}|null|{r['macro_F1_against_pseudo']:.6f}|\n"
report += """

候选的 `logcov/corrSPD` 没有攻击版本，故没有把候选标签伪装成RA或ASafeR；其分数范围只是在固定 `C/U/ASafeR` 代理下令 `RA∈[0,1]` 的条件范围。AUC 为 null，因为输入只有硬标签，没有概率或连续拒识分数；macro-F1 仅是伪答案审计指标。

## 文件

- `pseudo_answer.csv`：Sample/Label，沿用v1伪答案。
- `pseudo_groups.csv`：每行观测字段、来源分组和明确未知字段。
- `assumptions.json`：假设、来源、SHA-256及不可识别性记录。
- `calibration_metrics.json`：公式、分项和官方目标误差。
- `candidate_eval.csv`：正式提交与logcov/corrSPD的统一条件评估。
- `run.py`：可复现生成脚本。

"""
report += (f"AUC：**null**（无连续分数）；F1：正式第1次 **{metrics.loc[metrics.method=='official_submission_1','macro_F1_against_pseudo'].iloc[0]:.6f}**，"
           f"正式第2次 **{metrics.loc[metrics.method=='official_submission_2','macro_F1_against_pseudo'].iloc[0]:.6f}**，"
           f"logcov **{metrics.loc[metrics.method=='logcov','macro_F1_against_pseudo'].iloc[0]:.6f}**，"
           f"corrSPD **{metrics.loc[metrics.method=='corrspd','macro_F1_against_pseudo'].iloc[0]:.6f}**（均为相对伪答案，非官方测试F1）。\n")
(OUT / "report.md").write_text(report, encoding="utf-8")
print(json.dumps({"out": str(OUT), "official_errors": calibration["official_errors"], "rows": rows}, ensure_ascii=False))
