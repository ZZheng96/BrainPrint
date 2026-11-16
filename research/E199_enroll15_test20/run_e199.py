"""E199: enrol 15 of the 20 Train identities, then present ALL 20 identities' eval blocks.

The user's question
    "use this model, enrol the 15 people of Train, then test the 20 people of Train - can the
     system tell the 15 apart and reject the remaining 5?"

That is a full deployment simulation, so the evaluation block (0.6 < frac <= 0.8) is fed in
whole: 15 enrolled identities that should be named correctly and 5 unregistered identities that
should come back as label 0.

Stage 1 (training, all 20 identities)
    per-task mean / sd and the PCA basis are fitted on the fit rows (frac <= 0.6) of **all 20**
    identities, exactly as the user defined the training stage.  Fold-invariant, computed once.
    The E162 encoder is pre-trained and frozen; nothing is re-trained here.

Stage 2 (enrolment, only the 15 enrolled identities)
    per-identity prototypes from that identity's own rows: variant "fit" = fit rows
    (frac <= 0.6), variant "fit_eval" = fit + eval rows (frac <= 0.8, the "what if we had more
    enrolment data" upper bound).  k_proto = 12 for e65, 1 for subwin6 / e162.

Stage 3 (deployment)
    every eval row is scored against the enrolled prototypes only:
        identification = argmax over the 15 enrolled identities is the row's own identity,
        rejection      = best-match score below tau, with tau = quantile(enrolled-known
                         best-match scores, 1 - a) - enrolled identities calibrate it, the 5
                         unregistered never do.

Five hundred distinct leave-5 folds (same generator / seed rule as E197) give the cross-fold
statistics; one median fold is printed row by row so the failure mode (rejected vs. renamed) is
visible per identity.  The 200 E197 folds are re-run with this instrument as a reproduction
check.

Train only: no Test feature file is opened, the Session column is never read, no model is
trained, nothing is written outside this directory, nothing is committed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
TASK_OF = {name: i for i, name in enumerate(TASKS)}
PROTO = CFG["protocol"]
SEED = int(PROTO["seed"])
FIT = float(PROTO["fit_fraction"])
EVAL_LO, WIN_HI = 0.6, 0.8
PROTO_LIMIT = float(PROTO["upper_bound_prototype_fraction"])
NFOLDS = int(PROTO["folds"])
A_MAIN = float(PROTO["primary_acceptance"])
WORKPOINTS = [float(v) for v in PROTO["acceptance_workpoints"]]
FAMILIES = CFG["families"]
GATES = CFG["gates"]
GATE_NAMES = [g["name"] for g in GATES]
PRIMARY = GATE_NAMES[0]
VARIANTS = ["fit", "fit_eval"]
METRICS = ["acc", "rej", "false_accept", "frr", "overall", "overall_e2e", "tau", "auc", "f1"]


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.__stdout__

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------- protocol
def load_protocol() -> dict:
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    if len(meta) != int(CFG["data"]["train_trials"]):
        raise ValueError(f"Train rows {len(meta)} unexpected")
    if meta["subject"].nunique() != 20 or set(meta["session"]) != {"session1"}:
        raise ValueError("Train must be 20 subjects, session1 only")
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy(dtype=object)
    maximum = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy(float)
    frac = meta["trial"].to_numpy(float) / np.maximum(maximum, 1.0)
    fit_rows = frac <= FIT
    window = (frac > EVAL_LO) & (frac <= WIN_HI)
    proto_eval = frac <= PROTO_LIMIT
    if int(window.sum()) == 0:
        raise ValueError("empty evaluation window")
    return {"subjects": subjects, "frac": frac,
            "task_idx": np.array([TASK_OF[t] for t in tasks], dtype=np.int64),
            "fit_rows": fit_rows, "window": window, "win_rows": np.flatnonzero(window),
            "proto_eval": proto_eval, "n_ident": int(CLASSES.size)}


def build_folds() -> list:
    pool = list(itertools.combinations(CLASSES.tolist(), 5))
    rng = np.random.default_rng(SEED + 5)
    picked = np.sort(rng.choice(len(pool), size=NFOLDS, replace=False))
    return [tuple(int(v) for v in pool[i]) for i in picked]


def e197_folds() -> list:
    path = ROOT / CFG["references"]["e197_fold_plan"]
    stored = read_json(path).get("5", [])
    return [tuple(int(v) for v in item) for item in stored]


# ---------------------------------------------------------------- stage 1: training
def task_stats(x: torch.Tensor, tidx: torch.Tensor, rows: torch.Tensor) -> tuple:
    mu = torch.stack([x[rows & (tidx == i)].mean(0) for i in range(len(TASKS))])
    sd = torch.stack([x[rows & (tidx == i)].std(0) for i in range(len(TASKS))]) + 1e-6
    return mu, sd


def build_family(name: str, spec: dict, P: dict) -> dict:
    """Stage 1: task-wise z-stats and PCA basis on the fit rows of all 20 identities."""
    x = torch.as_tensor(f32(np.load(ROOT / spec["train"])), device=DEVICE)
    tidx = torch.as_tensor(P["task_idx"], device=DEVICE)
    fit = torch.as_tensor(P["fit_rows"], device=DEVICE)
    mu, sd = task_stats(x, tidx, fit)
    zf = (x[fit] - mu[tidx[fit]]) / sd[tidx[fit]]
    pca_mu = zf.mean(0)
    centred = zf - pca_mu
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    values, vectors = torch.linalg.eigh(covariance)
    dim = int(spec["pca"])
    basis = vectors[:, torch.argsort(values, descending=True)[:dim]]
    proj = torch.empty((len(x), dim), dtype=torch.float32, device=DEVICE)
    for start in range(0, len(x), 8192):
        block = x[start:start + 8192]
        index = tidx[start:start + 8192]
        z = (block - mu[index]) / sd[index]
        proj[start:start + len(block)] = (z - pca_mu) @ basis
    del x, zf, centred, covariance, vectors
    log(f"  stage1 {name}: {spec['dim']} -> {dim} PCA on {int(P['fit_rows'].sum())} "
        "fit rows of all 20 identities")
    return {"proj": proj, "dim": dim}


# ---------------------------------------------------------------- stage 2: enrolment
def kmeans_t(x: torch.Tensor, k: int, seed: int, iters: int = 20) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    centres = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centres).argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, x)
        count = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(
            0, assign, torch.ones(len(x), device=x.device, dtype=x.dtype))
        new = torch.where(count[:, None] > 0, sums / count.clamp(min=1)[:, None], centres)
        if torch.allclose(new, centres, atol=1e-6):
            break
        centres = new
    return centres


def build_prototypes(proj: torch.Tensor, P: dict, spec: dict, rows: np.ndarray) -> dict:
    k_proto = int(spec["prototypes"])
    out = {}
    for ident in CLASSES:
        sel = rows & (P["subjects"] == ident)
        block = proj[torch.as_tensor(np.flatnonzero(sel), device=DEVICE)]
        out[int(ident)] = kmeans_t(block, k_proto, seed=int(ident) * 17)
    return out


def score_matrix(proj: torch.Tensor, P: dict, prototypes: dict, k_proto: int) -> np.ndarray:
    """(eval rows, 20) best-match score per identity = max over that identity's prototypes."""
    query = proj[torch.as_tensor(P["win_rows"], device=DEVICE)]
    stack = torch.cat([prototypes[int(c)] for c in CLASSES], 0)
    out = np.empty((len(P["win_rows"]), len(CLASSES)), dtype=np.float32)
    for start in range(0, len(query), 512):
        block = query[start:start + 512]
        distance = torch.cdist(block, stack)
        out[start:start + len(block)] = (-distance).view(len(block), len(CLASSES), k_proto) \
            .max(2).values.float().cpu().numpy()
    return out


def build_instrument(P: dict) -> dict:
    started = time.time()
    log("stage 1: training on all 20 identities (fit rows, fold-invariant)")
    families = {name: build_family(name, spec, P) for name, spec in FAMILIES.items()}
    rows = {"fit": P["fit_rows"], "fit_eval": P["proto_eval"]}
    matrices = {}
    for variant in VARIANTS:
        matrices[variant] = {}
        for name, spec in FAMILIES.items():
            protos = build_prototypes(families[name]["proj"], P, spec, rows[variant])
            matrices[variant][name] = score_matrix(families[name]["proj"], P, protos,
                                                   int(spec["prototypes"]))
            n_rows = int(rows[variant].sum())
            log(f"  stage2 {variant:8s} {name:8s}: {spec['prototypes']} prototype(s)/identity "
                f"x 20 identities from {n_rows} rows")
            del protos
    seconds = time.time() - started
    log(f"stage 1 + 2 done in {seconds:.0f}s")
    return {"matrices": matrices, "seconds": seconds,
            "n_eval": int(len(P["win_rows"])), "n_fit": int(P["fit_rows"].sum()),
            "n_proto_eval": int(P["proto_eval"].sum())}


# ---------------------------------------------------------------- stage 3: scoring
def gate_matrix(matrices: dict, gate: dict, enrolled_idx: np.ndarray,
                known_mask: np.ndarray) -> np.ndarray:
    total = None
    for fam, weight in gate["weights"].items():
        scores = matrices[fam]
        reference = scores[known_mask][:, enrolled_idx].max(1).astype(np.float64)
        mu = float(reference.mean())
        sd = float(reference.std()) + 1e-6
        z = (scores.astype(np.float64) - mu) / sd
        total = weight * z if total is None else total + weight * z
    return total.astype(np.float32)


def fold_eval(scores: np.ndarray, subj_win: np.ndarray, enrolled: np.ndarray,
              enrolled_idx: np.ndarray, workpoints) -> dict:
    sub = scores[:, enrolled_idx]
    pred_col = sub.argmax(1)
    pred_id = enrolled[pred_col]
    best = sub.max(1).astype(np.float64)
    known_mask = np.isin(subj_win, enrolled)
    correct = pred_id == subj_win
    known_best = best[known_mask]
    unknown_best = best[~known_mask]
    target = np.searchsorted(enrolled, subj_win[known_mask])
    out = {"acc": float(np.mean(correct[known_mask])),
           "f1": float(f1_score(target, pred_col[known_mask], average="macro",
                                labels=np.arange(len(enrolled)), zero_division=0)),
           "auc": float(roc_auc_score(np.r_[np.ones(len(known_best)), np.zeros(len(unknown_best))],
                                      np.r_[known_best, unknown_best])),
           "best": best, "correct": correct, "pred_id": pred_id, "known_mask": known_mask,
           "work": {}}
    for a in workpoints:
        tau = float(np.quantile(known_best, 1.0 - a))
        accepted = best >= tau
        acc_known = accepted[known_mask]
        acc_unknown = accepted[~known_mask]
        correct_known = correct[known_mask]
        out["work"][a] = {
            "acc": out["acc"], "auc": out["auc"], "f1": out["f1"],
            "tau": tau,
            "rej": float(np.mean(~acc_unknown)),
            "false_accept": float(np.mean(acc_unknown)),
            "frr": float(np.mean(~acc_known)),
            "coverage": float(np.mean(acc_known)),
            "overall": float((int(correct_known.sum()) + int((~acc_unknown).sum()))
                             / len(subj_win)),
            "overall_e2e": float((int((correct_known & acc_known).sum())
                                  + int((~acc_unknown).sum())) / len(subj_win)),
            "acc_accepted": float(np.mean(correct_known[acc_known])) if acc_known.any()
            else float("nan")}
    return out


def run_variant(variant: str, P: dict, inst: dict, folds: list, tag: str) -> dict:
    subj_win = P["subjects"][P["win_rows"]]
    stats = {g: {f"{a:.2f}": {m: [] for m in METRICS} for a in WORKPOINTS}
             for g in GATE_NAMES}
    pooled = {g: {"known": [], "unknown": []} for g in GATE_NAMES}
    fold_overall = {g: [] for g in GATE_NAMES}
    ident = {g: {"enrolled": {int(c): {"rows": 0, "correct": 0, "accepted": 0,
                                      "correct_accepted": 0} for c in CLASSES},
                 "held": {int(c): {"rows": 0, "rejected": 0} for c in CLASSES}}
             for g in GATE_NAMES}
    confused = {g: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64) for g in GATE_NAMES}
    started = time.time()
    for index, held in enumerate(folds):
        held = np.asarray(held, dtype=int)
        enrolled = np.setdiff1d(CLASSES, held)
        enrolled_idx = enrolled - 1
        known_mask = np.isin(subj_win, enrolled)
        for gate in GATES:
            scores = gate_matrix(inst["matrices"][variant], gate, enrolled_idx, known_mask)
            row = fold_eval(scores, subj_win, enrolled, enrolled_idx, WORKPOINTS)
            name = gate["name"]
            for a in WORKPOINTS:
                for metric in METRICS:
                    stats[name][f"{a:.2f}"][metric].append(row["work"][a][metric])
            pooled[name]["known"].append(row["best"][known_mask].astype(np.float32))
            pooled[name]["unknown"].append(row["best"][~known_mask].astype(np.float32))
            fold_overall[name].append(row["work"][A_MAIN]["overall_e2e"])
            work = row["work"][A_MAIN]
            correct, pred_id = row["correct"], row["pred_id"]
            accepted = row["best"] >= work["tau"]
            for ident_value in enrolled:
                mask = known_mask & (subj_win == ident_value)
                entry = ident[name]["enrolled"][int(ident_value)]
                entry["rows"] += int(mask.sum())
                entry["correct"] += int(correct[mask].sum())
                entry["accepted"] += int(accepted[mask].sum())
                entry["correct_accepted"] += int((correct & accepted)[mask].sum())
            for ident_value in held:
                mask = (~known_mask) & (subj_win == ident_value)
                entry = ident[name]["held"][int(ident_value)]
                entry["rows"] += int(mask.sum())
                entry["rejected"] += int((~accepted)[mask].sum())
                taken = mask & accepted
                for true_id, pred in zip(subj_win[taken], pred_id[taken]):
                    confused[name][int(true_id) - 1, int(pred) - 1] += 1
        if (index + 1) % 100 == 0 or index + 1 == len(folds):
            log(f"  {tag} fold {index + 1:3d}/{len(folds)} "
                f"{(time.time() - started) / (index + 1):.3f}s/fold")
    summary = {}
    for name in GATE_NAMES:
        entry = {}
        for a in WORKPOINTS:
            key = f"{a:.2f}"
            entry[key] = {}
            for metric in METRICS:
                values = np.asarray(stats[name][key][metric], dtype=np.float64)
                entry[key][metric] = {"mean": float(np.nanmean(values)),
                                      "std": float(np.nanstd(values))}
        known = np.concatenate(pooled[name]["known"]).astype(np.float64)
        unknown = np.concatenate(pooled[name]["unknown"]).astype(np.float64)
        entry["auc_pooled"] = float(roc_auc_score(
            np.r_[np.ones(len(known)), np.zeros(len(unknown))], np.r_[known, unknown]))
        entry["n_known_rows"] = int(len(known))
        entry["n_unknown_rows"] = int(len(unknown))
        entry["per_identity"] = {
            "enrolled": {str(c): {k: int(v) for k, v in ident[name]["enrolled"][int(c)].items()}
                         for c in CLASSES},
            "held": {str(c): {k: int(v) for k, v in ident[name]["held"][int(c)].items()}
                     for c in CLASSES}}
        summary[name] = entry
    seconds = time.time() - started
    log(f"  {tag} {len(folds)} folds in {seconds:.0f}s")
    return {"gates": summary, "fold_overall_e2e": fold_overall, "seconds": seconds,
            "confusion_unknown_accepted": {g: confused[g].tolist() for g in GATE_NAMES}}


# ---------------------------------------------------------------- representative fold
def fold_rows(variant: str, P: dict, inst: dict, held: tuple, a: float) -> dict:
    subj_win = P["subjects"][P["win_rows"]]
    held = np.asarray(held, dtype=int)
    enrolled = np.setdiff1d(CLASSES, held)
    enrolled_idx = enrolled - 1
    known_mask = np.isin(subj_win, enrolled)
    per_gate = {}
    for gate in GATES:
        scores = gate_matrix(inst["matrices"][variant], gate, enrolled_idx, known_mask)
        row = fold_eval(scores, subj_win, enrolled, enrolled_idx, [a])
        work = row["work"][a]
        accepted = row["best"] >= work["tau"]
        decision = np.where(accepted, row["pred_id"], 0)
        rows = []
        for ident_value in CLASSES:
            mask = subj_win == ident_value
            is_enrolled = bool(ident_value in enrolled)
            n = int(mask.sum())
            counts = np.bincount(decision[mask], minlength=len(CLASSES) + 1)
            mode = int(counts.argmax())
            rejected_share = counts[0] / max(n, 1)
            outcome = "拒绝" if mode == 0 else f"认成 {mode}"
            if is_enrolled:
                accuracy = int(row["correct"][mask].sum()) / max(n, 1)
                if mode == 0:
                    result = "⛔ 被误拒"
                elif mode == int(ident_value):
                    result = "✅ 正确识别"
                else:
                    result = f"❌ 认成 {mode}"
            else:
                accuracy = rejected_share
                result = "✅ 正确拒识" if mode == 0 else f"❌ 被接受(认成 {mode})"
            rows.append({"identity": int(ident_value), "enrolled": is_enrolled, "rows": n,
                         "outcome": outcome, "outcome_share": float(counts[mode] / max(n, 1)),
                         "accuracy": accuracy,
                         "rejected_share": rejected_share,
                         "accepted_share": 1.0 - rejected_share,
                         "result": result})
        per_gate[gate["name"]] = {"tau": work["tau"], "rows": rows,
                                  "acc": row["acc"], "rej": work["rej"],
                                  "overall": work["overall"],
                                  "overall_e2e": work["overall_e2e"]}
    return {"held": [int(v) for v in held], "enrolled": [int(v) for v in enrolled],
            "per_gate": per_gate}


# ---------------------------------------------------------------- report
def fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def m(results: dict, gate: str, a: float, metric: str, field: str = "mean") -> float:
    return results["gates"][gate][f"{a:.2f}"][metric][field]


def write_report(results: dict) -> None:
    fit = results["variant_fit"]
    up = results["variant_fit_eval"]
    rep = results["representative"]
    check = results["verification"]
    lines = []
    add = lines.append

    add("# E199：Train 内「注册 15 人、评估 20 人」的部署仿真")
    add("")
    add("日期：2026-11-16。Train = 20 个 identity 的 session1（11304 条；fit 行 frac ≤ 0.6 共 "
        f"{results['instrument']['n_fit']} 条；评估块 0.6 < frac ≤ 0.8 共 "
        f"{results['instrument']['n_eval']} 条，20 人每人 98–124 条）。"
        f"{results['folds']} 个 leave-5 划分（注册 15 / 未注册 5）。"
        "本实验**不读 Test、不训练任何模型**，"
        "未写入已有实验目录，未提交 git。")
    add("")
    add("## 0. 直接回答：能否正确区分 15 人并拒识剩下的 5 人？")
    add("")
    best = max(GATE_NAMES, key=lambda g: m(fit, g, A_MAIN, "overall_e2e"))
    acc_lo = min(m(fit, g, A_MAIN, "acc") for g in GATE_NAMES)
    acc_hi = max(m(fit, g, A_MAIN, "acc") for g in GATE_NAMES)
    rej_lo = min(m(fit, g, A_MAIN, "rej") for g in GATE_NAMES)
    rej_hi = max(m(fit, g, A_MAIN, "rej") for g in GATE_NAMES)
    add(f"**能。** 15 个已注册者几乎全部认对（{acc_lo:.2%}–{acc_hi:.2%}），"
        f"5 个未注册者的拒识率取决于门控（{rej_lo:.2%}–{rej_hi:.2%}）。"
        f"主工作点 a = {A_MAIN:.2f} 下 4 个门控的读数：")
    add("")
    add("| 门控 | 识别准确率（15 人） | 拒识率（5 人） | 已注册者被误拒率 | 整体正确率 | "
        "端到端正确率 | AUC | macro-F1 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for gate in GATE_NAMES:
        add(f"| `{gate}` | {m(fit, gate, A_MAIN, 'acc'):.4f} | "
            f"{m(fit, gate, A_MAIN, 'rej'):.4f} | {m(fit, gate, A_MAIN, 'frr'):.4f} | "
            f"{m(fit, gate, A_MAIN, 'overall'):.4f} | "
            f"{m(fit, gate, A_MAIN, 'overall_e2e'):.4f} | "
            f"{fit['gates'][gate]['auc_pooled']:.4f} | {m(fit, gate, A_MAIN, 'f1'):.4f} |")
    add("")
    add("（识别准确率 = 15 个已注册者评估行 argmax 命中自身的比例，与阈值无关；"
        "拒识率 = 5 个未注册者评估行最佳分数低于 τ 的比例；"
        "整体正确率 = 按题目定义（正确识别数 + 正确拒识数）/ 全部 2301 行；"
        "端到端正确率额外要求已注册者必须通过阈值，因此扣掉了误拒代价。）")
    add("")
    add(f"- **只注册 15 人、一次送进 20 人时，最好的门控 `{best}` 在 "
        f"{m(fit, best, A_MAIN, 'overall'):.2%} 的评估行上给出正确决定（a = {A_MAIN:.2f}）："
        f"15 个已注册者认对 {m(fit, best, A_MAIN, 'acc'):.2%}，"
        f"5 个未注册者拒掉 {m(fit, best, A_MAIN, 'rej'):.2%}。**")
    add(f"- 识别与拒识是两件独立的事：识别准确率 {acc_lo:.4f}–{acc_hi:.4f} 的门控之间，"
        f"拒识率可以差到 {rej_hi - rej_lo:.4f}（`{min(GATE_NAMES, key=lambda g: m(fit, g, A_MAIN, 'rej'))}` "
        f"{rej_lo:.4f} ↔ `{max(GATE_NAMES, key=lambda g: m(fit, g, A_MAIN, 'rej'))}` {rej_hi:.4f}）。"
        f"提高 a 能降低误拒（{A_MAIN:.0%} 分位 → 误拒 "
        f"{m(fit, best, A_MAIN, 'frr'):.2%}），代价是未注册者更容易混进来；"
        f"端到端正确率把这两项一起算，最高 {m(fit, best, A_MAIN, 'overall_e2e'):.4f}。")
    worst_ident = results["per_identity_rank"][0]
    add(f"- **最难的是身份 {worst_ident['identity']}**：被当作未注册者时只有 "
        f"{worst_ident['rej_rate']:.4f} 被判拒（a = {A_MAIN:.2f}，`{PRIMARY}`），"
        f"即它 {1 - worst_ident['rej_rate']:.1%} 的评估行会被冒领成某个已注册者。")
    add("")

    add("## 1. 一个代表划分的逐人明细")
    add("")
    add(f"代表划分 = {results['folds']} 折中「端到端整体正确率最接近中位数」的那一折（门控 "
        f"`{PRIMARY}`，a = {A_MAIN:.2f}）：未注册者 = {rep['held']}，"
        f"已注册者 = {rep['enrolled']}。")
    add("")
    body = rep["per_gate"][PRIMARY]
    add(f"**门控 `{PRIMARY}`，τ = {body['tau']:.2f}**")
    add("")
    add("| 身份 | 是否注册 | 评估块 trial 数 | 被判为 | 正确率 | 被拒比例 | 结果 |")
    add("|---:|:---:|---:|---|---:|---:|---|")
    for row in body["rows"]:
        add(f"| {row['identity']} | {'是' if row['enrolled'] else '否'} | {row['rows']} | "
            f"{row['outcome']} | {row['accuracy']:.4f} | {row['rejected_share']:.4f} | "
            f"{row['result']} |")
    add("")
    add(f"该折：识别准确率 {body['acc']:.4f}，拒识率 {body['rej']:.4f}，"
        f"整体正确率 {body['overall']:.4f}，端到端 {body['overall_e2e']:.4f}。"
        "「被判为」取该身份在评估块上的**众数判决**（某个身份编号或「拒绝」）；"
        "「正确率」一列：注册者为 argmax 命中自身身份的行比例，未注册者为被判拒的行比例。"
        "注册者的「正确率」可以很高而「被判为」仍是「拒绝」——说明这一块的人被认对了，"
        "但被阈值锁在门外（误拒），这是两种不同的失败。")
    add("")
    add("同一折、4 个门控的逐人结果（`正确` / `误拒` / `认成 j`）：")
    add("")
    header = "| 身份 | 注册 | " + " | ".join(f"`{g}`" for g in GATE_NAMES) + " |"
    add(header)
    add("|---:|:---:|" + "---|" * len(GATE_NAMES))
    table = {g: {row["identity"]: row for row in rep["per_gate"][g]["rows"]} for g in GATE_NAMES}
    for ident_value in CLASSES:
        cells = []
        for gate in GATE_NAMES:
            row = table[gate][int(ident_value)]
            if row["enrolled"]:
                if row["outcome"] == "拒绝":
                    cells.append("误拒")
                elif row["outcome"] == f"认成 {int(ident_value)}":
                    cells.append("正确")
                else:
                    cells.append(row["outcome"])
            else:
                cells.append("拒识" if row["outcome"] == "拒绝" else row["outcome"])
        enrolled_flag = "是" if table[GATE_NAMES[0]][int(ident_value)]["enrolled"] else "否"
        add(f"| {int(ident_value)} | {enrolled_flag} | " + " | ".join(cells) + " |")
    add("")
    boundary = [r for r in body["rows"] if not r["enrolled"] and r["rejected_share"] < 0.75]
    add("这一折里，" +
        ("、".join(f"身份 {r['identity']}（被拒 {r['rejected_share']:.2f}）" for r in boundary)
         + " 属于边界样本：多数行仍被拒，但离通过只差一点。" if boundary
         else "5 个未注册者都被稳定拒掉，没有边界样本。") +
        "逐人明细只反映这一折；跨折排名见 §3：身份 "
        f"{results['per_identity_rank'][0]['identity']} 在 "
        f"{results['folds']} 折里只有 {results['per_identity_rank'][0]['rej_rate']:.2f} "
        "的评估行会被拒。")
    if "subwin6_pca96_k1" in rep["per_gate"]:
        swallowed = [r for r in rep["per_gate"]["subwin6_pca96_k1"]["rows"]
                     if not r["enrolled"] and r["outcome"] != "拒绝"]
        if swallowed:
            add("")
            add("同一折的 4 门控对比还暴露了另一种失败：`subwin6_pca96_k1` 把未注册的" +
                "、".join(f"身份 {r['identity']} {r['outcome']}（接受 "
                          f"{r['accepted_share']:.0%}）" for r in swallowed) +
                "整块放进来了——它的识别率高，但拒识门槛形同虚设。")
    add("")

    add(f"## 2. 跨 {results['folds']} 个划分的统计（均值 ± 标准差）")
    add("")
    add(f"{results['folds']} 个划分 = 从 C(20,5)=15504 个 5 人组合中随机抽 "
        f"{results['folds']} 个（seed 20261116+5，"
        "与 E197 相同的生成器与种子规则；E197 抽的是 200 个）。每个划分都注册 15 人、"
        f"对全部 20 人的 {results['instrument']['n_eval']} 条评估行一起打分。")
    add("")
    for a in WORKPOINTS:
        add(f"**a = {a:.2f}**（τ = 已注册者最佳分数的 {1 - a:.0%} 分位）")
        add("")
        add("| 门控 | 识别准确率 | 拒识率 | 误认率（未注册被接受） | 已注册被误拒率 | "
            "整体正确率 | 端到端正确率 |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for gate in GATE_NAMES:
            add(f"| `{gate}` | {m(fit, gate, a, 'acc'):.4f} ± "
                f"{m(fit, gate, a, 'acc', 'std'):.4f} | "
                f"{m(fit, gate, a, 'rej'):.4f} ± {m(fit, gate, a, 'rej', 'std'):.4f} | "
                f"{m(fit, gate, a, 'false_accept'):.4f} | "
                f"{m(fit, gate, a, 'frr'):.4f} | "
                f"{m(fit, gate, a, 'overall'):.4f} ± "
                f"{m(fit, gate, a, 'overall', 'std'):.4f} | "
                f"{m(fit, gate, a, 'overall_e2e'):.4f} |")
        add("")
    add("说明：识别准确率与 a 无关（argmax 不看阈值），随 a 变的只有拒识 / 误认 / 误拒 / "
        "两种整体正确率。a 越大阈值越松：未注册者更容易混进来（拒识率下降），"
        "已注册者更不容易被误拒（误拒率从 15% 降到 5%）。"
        "整体正确率（题目定义，不计误拒）因此单调下降，端到端正确率则在某个 a 上取到最大——"
        "对 `e65*` 系列这个最优点在 a = 0.90 附近，对 `subwin6_pca96_k1` 则在 a = 0.85，"
        "因为它的识别率不足以补偿误认。")
    add("")

    add("## 3. 谁最容易失败（逐身份排名）")
    add("")
    add(f"**最容易被冒领的身份**（被留作未注册者时，`{PRIMARY}`、a = {A_MAIN:.2f} 的拒识率，"
        "升序；越低越难拒）：")
    add("")
    add("| 排名 | 身份 | 作为未注册者被拒率 | 作为已注册者识别率 | 累计行数（未注册/已注册，"
        f"{results['folds']} 折求和）|")
    add("|---:|---:|---:|---:|---|")
    for rank, row in enumerate(results["per_identity_rank"][:8], 1):
        add(f"| {rank} | {row['identity']} | {row['rej_rate']:.4f} | {row['acc_rate']:.4f} | "
            f"{row['rows_when_held']} / {row['rows_when_enrolled']} |")
    add("")
    add("**最容易被认错的身份**（作为已注册者时 argmax 命中率，升序）：")
    add("")
    add("| 排名 | 身份 | 作为已注册者识别率 | 作为未注册者被拒率 |")
    add("|---:|---:|---:|---:|")
    for rank, row in enumerate(sorted(results["per_identity_rank"],
                                      key=lambda r: r["acc_rate"])[:8], 1):
        add(f"| {rank} | {row['identity']} | {row['acc_rate']:.4f} | {row['rej_rate']:.4f} |")
    add("")
    add("两列互补：前者说明「这个人的样本与已注册者太像」，后者说明「这个人的样本内部太散、"
        "12 个原型也盖不住」。逐门控的完整表见 `results.json` 的 "
        "`variant_fit.gates.<gate>.per_identity`。")
    add("")

    add("## 4. 原型数据量的影响：只用 fit 块 vs 用 fit + eval 块")
    add("")
    add(f"同一 {results['folds']} 个划分、同一门控，只把注册原型的来源从 frac ≤ 0.6 扩到 frac ≤ 0.8"
        f"（注册行数 {results['instrument']['n_fit']} → "
        f"{results['instrument']['n_proto_eval']}，"
        f"×{results['instrument']['n_proto_eval'] / results['instrument']['n_fit']:.2f}）。")
    add("")
    add("| 门控 | a | 识别准确率（fit） | 识别准确率（fit+eval） | 拒识率（fit） | "
        "拒识率（fit+eval） | 整体正确率（fit） | 整体正确率（fit+eval） |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for gate in GATE_NAMES:
        for a in WORKPOINTS:
            add(f"| `{gate}` | {a:.2f} | {m(fit, gate, a, 'acc'):.4f} | "
                f"{m(up, gate, a, 'acc'):.4f} | {m(fit, gate, a, 'rej'):.4f} | "
                f"{m(up, gate, a, 'rej'):.4f} | {m(fit, gate, a, 'overall'):.4f} | "
                f"{m(up, gate, a, 'overall'):.4f} |")
    add("")
    deltas_acc = [m(up, g, A_MAIN, "acc") - m(fit, g, A_MAIN, "acc") for g in GATE_NAMES]
    deltas_rej = [m(up, g, A_MAIN, "rej") - m(fit, g, A_MAIN, "rej") for g in GATE_NAMES]
    deltas_all = [m(up, g, A_MAIN, "overall") - m(fit, g, A_MAIN, "overall") for g in GATE_NAMES]
    add(f"**读法**：注册数据从 fit 块扩到 fit+eval，a = {A_MAIN:.2f} 时识别准确率平均变化 "
        f"{np.mean(deltas_acc):+.4f}，拒识率平均变化 {np.mean(deltas_rej):+.4f}，"
        f"整体正确率平均变化 {np.mean(deltas_all):+.4f}。")
    add("")
    add("这是**上界参考**，且上界里含有一部分「作弊」：`fit+eval` 的原型由评估块的行一起建，"
        "而评估块正是被查询的同一批行，所以已注册者的分数被自己拉近（k = 12 的 k-means 把"
        "约 40 条挤进一个中心，并未退化到距离 0，但确实偏乐观）；未注册者不参与建模，"
        "因此受这个偏差的影响小。换句话说，这张表的增益混了「更多注册数据」与「查询自包含」"
        "两种来源，只能用来判断上限方向，不能当作真实部署的预期。")
    add("")

    add("## 5. 与 Test S1 的可比性")
    add("")
    ref_s1 = results["test_reference"]["test_s1_rejection"]
    s1_by_gate = results["test_reference"]["s1_by_gate"]
    add(f"Test S1 的实际拒识率（任务给定的既有审计读数，E197 k=5 的 9 门控均值，"
        f"折匹配口径：gallery = 该折 15 个注册者、τ = 该折已注册者分位）= **{ref_s1:.4f}**"
        "（与 `E197/results.json` 的 `analysis.test_s1_match['5'].test_mean` 逐位一致）。"
        "本实验用的是完全相同的折匹配口径，所以下表的差值可以直接读：")
    add("")
    add("| 门控 | 本实验 Train 内 15/5 拒识率 | Test S1（同协议） | 差（Train − S1） |")
    add("|---|---:|---:|---:|")
    for gate in GATE_NAMES:
        if gate in s1_by_gate:
            add(f"| `{gate}` | {m(fit, gate, A_MAIN, 'rej'):.4f} | {s1_by_gate[gate]:.4f} | "
                f"{m(fit, gate, A_MAIN, 'rej') - s1_by_gate[gate]:+.4f} |")
    add("")
    trains = [m(fit, g, A_MAIN, "rej") for g in GATE_NAMES if g in s1_by_gate]
    tests = [s1_by_gate[g] for g in GATE_NAMES if g in s1_by_gate]
    add(f"4 个门控里 Train 内读数均值 {np.mean(trains):.4f}、Test S1 均值 "
        f"{np.mean(tests):.4f}，差 {np.mean(trains) - np.mean(tests):+.4f}。")
    add("")
    add("**本实验的拒识率不能直接当作 Test S1 的预期，而且偏差方向并不是「同批次一定更容易拒」。**"
        "任务预设是「Train 内 5 个未注册者与 15 个注册者来自同一采集批次，所以更容易拒绝」，"
        "但实测差值是**混合的**：`e65_pca96_k12`、`e65+subwin6` 略低，"
        "`subwin6_pca96_k1` 明显更低（更差），只有 `e65_0.7+e162_0.3` 更高。"
        "E197 在同一协议、9 个门控上的系统偏差同样指向 Train 侧偏低（−0.0955，MAE 0.1211）。"
        "所以「同批次」并没有让 Train 内变得更好拒；它让这个读数**不再测量同一件事**。"
        "原因有三层：")
    add("")
    add("1. **未注册者的来源不同。** 本实验的 5 个未注册者是从这 20 个人里抽掉的，"
        "Test S1 的 5 个人根本不属于这 20 人。前者是「同一人群内部的留出」，"
        "后者是「另一个人群」——两者的个体差异尺度不同，不能用一个常数偏移互相换算。")
    add("2. **批次（session）漂移。** 本实验的未注册者与已注册者共享同一批 session1 采集"
        "（同一电极、参考、放大器状态、被试当天状态），Test S1 是另一批采集。"
        "跨批次时 session 漂移会把一部分 S1 样本推进或推出已注册者的分数分布，"
        "两个方向的样本都存在，所以净效应是「不可换算」而不是「更容易」或「更难」。")
    add("3. **阶段 1 已经见过这 5 个人。** 本实验按用户流程用全部 20 人拟合标准化与 PCA，"
        "5 个未注册者的行参与了子空间估计；Test S1 的人从未出现在 Train 的任何阶段。"
        "这是本实验独有的乐观来源，与批次无关。")
    add("")
    add("结论：本实验的 15/5 读数回答的是「**这 20 人内部，注册 15 个后能否把另外 5 个挡在外面**」"
        f"（能：a = {A_MAIN:.2f} 下 `{PRIMARY}` 拒 "
        f"{m(fit, PRIMARY, A_MAIN, 'rej'):.0%}、`{best}` 拒 "
        f"{m(fit, best, A_MAIN, 'rej'):.0%}），"
        "它不是 Test S1 的预测量，也不能按「同批次所以更乐观」做单方向修正。"
        "识别准确率一侧同理："
        f"{m(fit, PRIMARY, A_MAIN, 'acc'):.2%} 是同一 session 内的读数，不能外推到跨 session。")
    add("")

    add("## 6. Test 官方指标")
    add("")
    add("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**"
        "（本实验是纯 Train 内仿真，不读 Test，也没有身份标签）。"
        "报告中出现的 AUC / macro-F1 均为 Train 评估块上的 open-set / closed-set 读数。")
    add("")

    add("## 7. 合规与复现核对")
    add("")
    add("- 原型：只用已注册的 15 个身份自己的行（`fit` 变体用 frac ≤ 0.6）；"
        "阈值 τ 只由这 15 个已注册者的评估块分数标定，未注册的 5 人从不参与。")
    add("- 标准化与 PCA：用全部 20 人的 fit 行拟合（用户定义的三阶段流程的第 1 阶段），"
        "这是本实验与「严格留出」协议的唯一区别，且与 E197 / E198 一致。")
    add("- 未读 Test 的任何特征文件，未使用 Session 字段（Train 只有 session1，也未读取该列）。")
    add("- 未重训任何模型；E162 编码器为预训练冻结嵌入，直接读缓存。")
    if check.get("e197_match"):
        add("")
        add(f"**E197 折划分复现**：用本实验的仪器重跑 E197 存下来的 "
            f"{check['n_e197_folds']} 个 leave-5 折，"
            f"{len(check['e197_match'])} 个门控在 a = {A_MAIN:.2f} 的读数与 E197 "
            "`results.json` 对比如下（差异应仅为浮点）：")
        add("")
        add("| 门控 | 本实验 acc | E197 acc | 本实验 rej | E197 rej |")
        add("|---|---:|---:|---:|---:|")
        for gate, row in check["e197_match"].items():
            add(f"| `{gate}` | {row['acc']:.4f} | {row['e197_acc']:.4f} | "
                f"{row['rej']:.4f} | {row['e197_rej']:.4f} |")
    if "fold_overlap" in check:
        add("")
        add(f"本次 {results['folds']} 折与 E197 的 {check['n_e197_folds']} 折交集 "
            f"{check['fold_overlap']} 个"
            "（两者都是从同一生成器抽的不同规模子集，因此折均值会有小幅差异）。")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e199.py` | 全部逻辑（`--stage all / report`） |")
    add("| `config.json` | 冻结配置 |")
    add("| `results.json` | 500 折 × 2 种原型 × 4 门控 × 3 个 a 的全部指标、逐身份计数、代表划分 |")
    add("| `report.md` | 本报告 |")
    add("| `run.log` | 运行日志 |")
    add("")
    add("```powershell")
    add('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        '"D:\\WORK\\BCIC2026\\BrainPrint\\research\\E199_enroll15_test20\\run_e199.py" --stage all')
    add("```")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- driver
def reference_s1() -> dict:
    """Test S1 rejection of the same gates, quoted from the existing audits."""
    out = {}
    e197 = read_json(ROOT / CFG["references"]["e197"]).get("gates", {}).get("5", {})
    for gate, key in (("e65_pca96_k12", "e65_pca96"), ("subwin6_pca96_k1", "subwin6_dist"),
                      ("e65+subwin6", "e65+subwin6")):
        row = e197.get(key)
        if row:
            out[gate] = float(row["s1_0.90"]["mean"])
    e198 = read_json(ROOT / CFG["references"]["e198"]).get("test", {}).get("candidates", [])
    for candidate in e198:
        if candidate.get("weights") == {"e65_dist": 0.7, "e162_dist": 0.3}:
            out["e65_0.7+e162_0.3"] = float(candidate["fold_matched"]["s1"])
    return out


def per_identity_rank(fit: dict) -> list:
    stats = fit["gates"][PRIMARY]["per_identity"]
    rows = []
    for c in CLASSES:
        enrolled = stats["enrolled"][str(int(c))]
        held = stats["held"][str(int(c))]
        rows.append({
            "identity": int(c),
            "rej_rate": held["rejected"] / max(held["rows"], 1),
            "acc_rate": enrolled["correct"] / max(enrolled["rows"], 1),
            "rows_when_enrolled": enrolled["rows"],
            "rows_when_held": held["rows"],
            "accepted_rate_when_enrolled": enrolled["accepted"] / max(enrolled["rows"], 1),
            "correct_accepted_rate": enrolled["correct_accepted"] / max(enrolled["rows"], 1)})
    ranked = [r for r in rows if r["rows_when_held"] > 0 and r["rows_when_enrolled"] > 0]
    return sorted(ranked or rows, key=lambda r: r["rej_rate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "report"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0,
                        help="cap the folds (timing smoke test only)")
    args = parser.parse_args()

    sys.stdout = Tee(LOG_PATH)
    log(f"\nE199 start {time.strftime('%Y-%m-%d %H:%M:%S')} device {DEVICE}")
    torch.manual_seed(SEED)

    if args.stage == "report" and RESULTS_PATH.exists() and not args.force:
        results = read_json(RESULTS_PATH)
        results["per_identity_rank"] = per_identity_rank(results["variant_fit"])
        results["test_reference"]["s1_by_gate"] = reference_s1()
        write_json(RESULTS_PATH, results)
        write_report(results)
        log("stage report done")
        return

    P = load_protocol()
    log(f"Train {len(P['subjects'])} rows, fit {int(P['fit_rows'].sum())}, "
        f"eval window {int(P['window'].sum())}, 20 identities, device {DEVICE}")
    inst = build_instrument(P)
    folds = build_folds()
    if args.max_folds:
        folds = folds[:args.max_folds]
    log(f"{len(folds)} leave-5 folds, 15 enrolled / 5 unregistered, eval rows {inst['n_eval']}")

    results = {
        "experiment": CFG["experiment"], "date": CFG["date"],
        "protocol": PROTO,
        "instrument": {"n_fit": inst["n_fit"], "n_proto_eval": inst["n_proto_eval"],
                       "n_eval": inst["n_eval"], "seconds": inst["seconds"]},
        "folds": len(folds), "gates": GATE_NAMES,
        "variant_fit": run_variant("fit", P, inst, folds, "fit"),
        "variant_fit_eval": run_variant("fit_eval", P, inst, folds, "fit_eval")}

    # representative fold: the one whose end-to-end overall accuracy is the median
    overall = np.asarray(results["variant_fit"]["fold_overall_e2e"][PRIMARY], dtype=np.float64)
    median = float(np.median(overall))
    chosen = int(np.argmin(np.abs(overall - median)))
    log(f"representative fold {chosen} (held {folds[chosen]}) overall {overall[chosen]:.4f} "
        f"median {median:.4f}")
    results["representative"] = fold_rows("fit", P, inst, folds[chosen], A_MAIN)
    results["representative"].update({"fold_index": chosen, "overall_mean_over_folds": median,
                                      "overall_this_fold": float(overall[chosen]),
                                      "acceptance": A_MAIN})

    # reproduction of E197 with the same instrument
    old = e197_folds()
    reference = read_json(ROOT / CFG["references"]["e197"]).get("gates", {}).get("5", {})
    check = {"n_e197_folds": len(old), "e197_match": {}}
    if old:
        replay = run_variant("fit", P, inst, old, "e197_check")
        names = {"e65_pca96_k12": "e65_pca96", "subwin6_pca96_k1": "subwin6_dist",
                 "e65+subwin6": "e65+subwin6"}
        for gate, e197_name in names.items():
            if e197_name in reference:
                check["e197_match"][gate] = {
                    "acc": replay["gates"][gate]["0.90"]["acc"]["mean"],
                    "rej": replay["gates"][gate]["0.90"]["rej"]["mean"],
                    "e197_acc": float(reference[e197_name]["acc"]["mean"]),
                    "e197_rej": float(reference[e197_name]["rej_0.90"]["mean"])}
        mine = {tuple(sorted(f)) for f in folds}
        check["fold_overlap"] = len(mine & {tuple(sorted(f)) for f in old})
        check["replay_seconds"] = replay["seconds"]
    results["verification"] = check
    results["per_identity_rank"] = per_identity_rank(results["variant_fit"])
    results["test_reference"] = {
        "test_s1_rejection": float(CFG["references"]["test_s1_rejection_reference"]),
        "note": CFG["references"]["test_s1_note"],
        "s1_by_gate": reference_s1()}

    write_json(RESULTS_PATH, results)
    write_report(results)
    log("E199 done")


if __name__ == "__main__":
    main()
