import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(r"d:\WORK\BCIC2026\BrainPrint")
OUT = ROOT / "research/session_blind_candidate/calibrated_pseudo_ground_truth_v3"
SCENARIOS = OUT / "scenarios"
P1 = ROOT / "commit/answer/赞美诗_第1次.csv"
P2 = ROOT / "commit/answer/赞美诗_第2次.csv"
V1 = ROOT / "research/session_blind_candidate/pseudo_ground_truth_v1/pseudo_answer.csv"
LOG = ROOT / "research/session_blind_candidate/nested_pseudounknown_corrspd_v1/candidate_logcov.csv"
CORR = ROOT / "research/session_blind_candidate/nested_pseudounknown_corrspd_v1/candidate_corrspd.csv"
META = ROOT / "BrainPrint_cache/test_metadata.csv"
TARGETS = {"official_submission_1": 69.17, "official_submission_2": 67.94}
ASAFER = 0.95
U = 1.0
N_SCENARIOS = 16


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(c, ra, asafer=ASAFER, u=U):
    return 100.0 * c**0.40 * (0.7 * ra + 0.3 * asafer) ** 0.35 * u**0.25


def implied_ra(target, c):
    a = (target / 100.0 / (c**0.40 * U**0.25)) ** (1.0 / 0.35)
    return (a - 0.3 * ASAFER) / 0.7


def calc_c(pred, pseudo):
    mask = pseudo != 0
    return float(((pred[mask] == pseudo[mask]).sum()) / max(mask.sum(), 1))


formal = pd.read_csv(P1)
p2 = pd.read_csv(P2)
v1 = pd.read_csv(V1)
log = pd.read_csv(LOG)
corr = pd.read_csv(CORR)
meta = pd.read_csv(META)
for frame in (p2, v1, log, corr):
    if not formal.Sample.equals(frame.Sample):
        raise ValueError("输入CSV Sample顺序不一致")
if len(formal) != 22251:
    raise ValueError("正式CSV行数不为22251")
base = formal.rename(columns={"Label": "p1"}).copy()
base["p2"] = p2.Label.to_numpy()
base["v1"] = v1.Label.to_numpy()
base["logcov"] = log.Label.to_numpy()
base["corrspd"] = corr.Label.to_numpy()
base = base.merge(meta[["sample", "session", "task"]], left_on="Sample", right_on="sample", how="left").drop(columns="sample")
base["common_official"] = base.p1.eq(base.p2)
base["common_zero"] = base.p1.eq(0) & base.p2.eq(0)
base["discordant"] = ~base.common_official

input_paths = [P1, P2, V1, LOG, CORR]
input_sha = {str(p): sha256(p) for p in input_paths}
rng_master = np.random.default_rng(20261107)
scenario_rows = []
all_candidate_rows = []

for i in range(1, N_SCENARIOS + 1):
    seed = int(rng_master.integers(1, 2**31 - 1))
    rng = np.random.default_rng(seed)
    sname = f"scenario_{i:02d}"
    sdir = SCENARIOS / sname
    sdir.mkdir(parents=True, exist_ok=True)
    pseudo = base.v1.to_numpy(dtype=int).copy()
    discord = np.flatnonzero(base.discordant.to_numpy())
    # Only discordant/unobservable rows may vary; common observations are hard-fixed.
    if len(discord):
        mode = i % 4
        if mode == 0:
            pseudo[discord] = base.p1.to_numpy(dtype=int)[discord]
        elif mode == 1:
            pseudo[discord] = base.p2.to_numpy(dtype=int)[discord]
        else:
            source = base.p1.to_numpy(dtype=int) if mode == 2 else base.p2.to_numpy(dtype=int)
            pseudo[discord] = source[discord]
            flip = rng.random(len(discord)) < (0.08 if mode == 2 else 0.14)
            pseudo[discord[flip]] = rng.integers(1, 21, size=int(flip.sum()))
    # Hard constraint: any common row retains its observed official label.
    common = base.common_official.to_numpy()
    pseudo[common] = base.p1.to_numpy(dtype=int)[common]
    out = base.copy()
    out["Label"] = pseudo
    out[["Sample", "Label"]].to_csv(sdir / "pseudo_answer.csv", index=False)

    hidden = np.array(["unknown"] * len(out), dtype=object)
    hidden[common] = "clean"
    hidden[~common] = rng.choice(["clean", "adversarial", "unknown"], size=(~common).sum(), p=[0.25, 0.45, 0.30])
    attack_pair = np.full(len(out), "", dtype=object)
    attack_pair[hidden == "adversarial"] = [f"{sname}_pair_{j:06d}" for j in np.flatnonzero(hidden == "adversarial")]
    source = np.where(common, "observed_common_official", np.where(hidden == "adversarial", "synthetic_attack_perturbation", "synthetic_unobservable_rule"))
    confidence = np.where(common, "observed_high", np.where(hidden == "unknown", "synthetic_low", "synthetic_medium"))
    groups = pd.DataFrame({
        "Sample": out.Sample, "Label": pseudo, "category": hidden,
        "attack_pair_id": attack_pair, "source": source, "confidence": confidence,
        "session": out.session, "task": out.task,
        "common_official_observation": common,
        "hard_constraint": np.where(common, "fixed_to_official_common_label", "not_observed_randomized"),
        "seed": seed,
    })
    groups.to_csv(sdir / "pseudo_groups.csv", index=False)

    for method in ["official_submission_1", "official_submission_2", "logcov", "corrspd"]:
        pred = out["p1" if method == "official_submission_1" else "p2" if method == "official_submission_2" else method].to_numpy(dtype=int)
        c = calc_c(pred, pseudo)
        f1 = float(f1_score(pseudo, pred, labels=list(range(21)), average="macro", zero_division=0))
        if method in TARGETS:
            ra = implied_ra(TARGETS[method], c)
            exact = score(c, ra)
            low = high = exact
            ra_status = "calibrated_implied_proxy"
        else:
            ra = None
            exact = None
            low, high = score(c, 0.0), score(c, 1.0)
            ra_status = "not_observed; explicit conditional RA range [0,1]"
        scenario_rows.append({"scenario": sname, "seed": seed, "method": method, "n": len(out),
                              "C": c, "RA": ra, "ASafeR": ASAFER, "U": U, "score": exact,
                              "score_range_low": low, "score_range_high": high,
                              "macro_F1_against_pseudo": f1, "AUC": None, "RA_status": ra_status,
                              "category_counts": json.dumps(pd.Series(hidden).value_counts().to_dict(), ensure_ascii=False)})

    for method in ["logcov", "corrspd"]:
        pred = out[method].to_numpy(dtype=int)
        c = calc_c(pred, pseudo)
        all_candidate_rows.append({"scenario": sname, "seed": seed, "method": method, "C": c,
                                   "score_low_RA0": score(c, 0.0), "score_high_RA1": score(c, 1.0),
                                   "F1": float(f1_score(pseudo, pred, labels=list(range(21)), average="macro", zero_division=0)),
                                   "AUC": None, "missing_attack_labels_rule": "candidate hard labels have no attack-version labels; report explicit RA∈[0,1] range"})

metrics = pd.DataFrame(scenario_rows)
metrics.to_csv(OUT / "scenario_metrics.csv", index=False)
cand = pd.DataFrame(all_candidate_rows)
cand.to_csv(OUT / "candidate_scenario_metrics.csv", index=False)

summary = []
for method, g in metrics.groupby("method"):
    vals = g.score.dropna() if g.score.notna().any() else (g.score_range_low + g.score_range_high) / 2
    summary.append({"method": method, "scenario_count": len(g), "conditional_mean": float(vals.mean()),
                    "conditional_std": float(vals.std(ddof=1)), "conditional_min": float(vals.min()),
                    "conditional_max": float(vals.max()), "AUC": None,
                    "F1_mean": float(g.macro_F1_against_pseudo.mean())})
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT / "candidate_mean_std_summary.csv", index=False)

formal_after_sha = {str(p): sha256(p) for p in [P1, P2]}
if formal_after_sha != {str(p): input_sha[str(p)] for p in [P1, P2]}:
    raise RuntimeError("正式提交SHA发生变化")
assumptions = {
    "purpose": "synthetic calibration scenarios；不是真实Test标签或官方预测",
    "formula": "Score=100*C^0.40*(0.7*RA+0.3*ASafeR)^0.35*U^0.25",
    "scenario_count": N_SCENARIOS, "master_seed": 20261107,
    "fixed_proxies": {"ASafeR": ASAFER, "U": U},
    "hard_constraints": ["common official labels are retained exactly", "category randomization only on official-discordant/unobservable rows"],
    "candidate_rule": "logcov/corrspd have no attack-version labels; do not impute hidden labels; calculate C against each scenario pseudo labels and expose score range for RA in [0,1]",
    "non_identifiability": "scenario averages are model averages over assumptions, not resolution of non-identifiability",
    "input_sha256": input_sha, "formal_sha256_after_run": formal_after_sha,
    "common_zero_count": int(base.common_zero.sum()), "sample_count": len(base),
    "anchor_targets": TARGETS,
}
(OUT / "assumptions.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")

anchor_errors = {}
for method, target in TARGETS.items():
    vals = metrics.loc[metrics.method == method, "score"]
    anchor_errors[method] = {"max_abs_error": float(np.max(np.abs(vals - target))), "all_below_1e-6": bool(np.all(np.abs(vals - target) < 1e-6))}
report = f"""# calibrated pseudo ground truth v3 — synthetic calibration scenarios

## 结论与边界

本目录生成 **{N_SCENARIOS} 份 synthetic calibration scenarios**，不是官方真实 Test 标签，也不声称是真实官方预测。每份场景包含 `pseudo_answer.csv` 与 `pseudo_groups.csv`；共同正式观测行保持硬约束，攻击类别只在两次正式提交不一致的不可观测行随机分配。

固定条件代理为 `ASafeR={ASAFER}`、`U={U}`，严格使用：`Score=100*C^0.40*(0.7*RA+0.3*ASafeR)^0.35*U^0.25`。每个场景的正式提交锚点都通过隐含 RA 微调精确复现目标。

## 锚点复现

- 第一次正式提交目标：69.17；最大绝对误差：{anchor_errors['official_submission_1']['max_abs_error']:.15g}；严格通过 `<1e-6`：{anchor_errors['official_submission_1']['all_below_1e-6']}。
- 第二次正式提交目标：67.94；最大绝对误差：{anchor_errors['official_submission_2']['max_abs_error']:.15g}；严格通过 `<1e-6`：{anchor_errors['official_submission_2']['all_below_1e-6']}。

## 候选场景结果

`scenario_metrics.csv` 给出每个场景的 C、RA、ASafeR、U、分数/范围、AUC 和相对伪答案 F1。`candidate_mean_std_summary.csv` 给出场景条件平均分、标准差和范围；其中 logcov/corrSPD 没有攻击样本标签，不能隐藏补值，使用明确的 `RA∈[0,1]` 条件范围。AUC 均为 null（没有概率分数），F1 是相对场景伪答案的审计指标，不是官方 F1。

场景平均只代表在这些合成条件下的模型平均，不解决真实标签、攻击分组、RA、ASafeR、U 的不可识别性。

## 完整性

运行时核验两次正式 CSV SHA-256 未改变；输入与运行后摘要见 `assumptions.json`。每个场景参数（seed、类别计数、硬约束规则）见 `scenario_metrics.csv` 和场景 `pseudo_groups.csv`。

## AUC/F1

AUC：null（无连续概率输出）；F1：各场景逐方法记录于 `scenario_metrics.csv`，汇总均值记录于 `candidate_mean_std_summary.csv`。
"""
(OUT / "report.md").write_text(report, encoding="utf-8")
print(json.dumps({"out": str(OUT), "scenario_count": N_SCENARIOS, "anchor_errors": anchor_errors, "formal_sha_unchanged": formal_after_sha == {str(p): input_sha[str(p)] for p in [P1, P2]}}, ensure_ascii=False))
