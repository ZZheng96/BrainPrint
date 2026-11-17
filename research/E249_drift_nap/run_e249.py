"""E249: drift-robust LDA-cos scoring (NAP / WCCN+drift) under train-only nested temporal subject LOSO.

Drift scatter D is estimated only from the fold's fit identities and fit time blocks:
  temporal drift  = per (subject, task) prototype difference between the two halves of the fit block;
  cross-task drift = per subject prototype difference between tasks a,b, minus the mean of that
                     difference over subjects (removes the shared task offset).
No Test data is read. final/ is only imported (read-only) through run_e247.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RES = OUT / "results"
sys.path.insert(0, str(ROOT / "research" / "E247_train_temporal_loso"))
import run_e247 as R  # noqa: E402

DEV, DIM = R.DEV, R.DIM

# ---------------------------------------------------------------- protocol (fixed before execution)
PROTOCOL = {
    "version": "E249-v1-drift-nap-wccn",
    "outer_fit": "outer-known subjects, frac <= 0.6 (Early)",
    "outer_D_temporal_split": [0.3, 0.6],      # frac<=0.3 vs (0.3,0.6], outer-known only
    "inner_fit": "inner-remaining 18 subjects, frac <= 0.4",
    "inner_D_temporal_split": [0.2, 0.4],      # frac<=0.2 vs (0.2,0.4], inner-remaining only
    "inner_calibration": "held-out subject 0.4 < frac <= 0.6 (unknown); remaining subjects 0.4 < frac <= 0.6 (known, selection only)",
    "evaluation_block": "Late frac > 0.8",
    "D_definition": "D = 0.5*Dt/tr(Dt) + 0.5*Dx/tr(Dx); Dt = mean outer product of temporal prototype diffs over (subject,task); "
                    "Dx = mean outer product of subject-centred cross-task prototype diffs over included unordered task pairs",
    "D_pair_exclusion": "P1 uses all 6 task pairs; P2 ordered pair (A,B) uses D re-estimated without unordered pair {A,B}",
    "C0": "E247 strict LDA-cos, shrink 0.01",
    "C1": "NAP: P = I - U_k U_k^T (top-k eigvecs of D); project samples & prototypes; within-class cov re-estimated on projected residuals; LDA-cos shrink 0.01",
    "C1_k": [4, 8, 16],
    "C2": "WCCN+drift: whitening cov = Sw + alpha * D * tr(Sw)/tr(D); LDA-cos shrink 0.01",
    "C2_alpha": [0.5, 1.0, 2.0],
    "shrink": 0.01,
    "far": 0.10,
    "threshold": "E247 weighted (equal per held-out subject) tie-safe FAR10 tail, separately per candidate config and per endpoint (P1, each ordered P2 pair)",
    "P1": "Early prototypes, Late samples, same-task prototype",
    "P2": "Late samples of task B scored against Early prototypes of task A != B (sample standardised by its own task B); 12 ordered pairs, averaged",
    "inner_selection": "per family choose k/alpha maximising inner known acceptance on P2 (mean over 12 ordered pairs of mean over 19 inner folds of "
                       "fraction of remaining-subject (0.4,0.6] scores >= that pair's inner FAR10 threshold); ties -> first in list; same hparam used for P1 and P2",
    "bootstrap": "identity-level paired, 10000 resamples of 20 outer folds, seed 249, percentile 95% CI",
    "primary_criterion": "P2 known_acceptance(Cx) - C0 mean > 0 with 95% CI excluding 0, and mean unknown_rejection >= 0.88 on both P1 and P2; "
                         "among passing pick highest P2 known_acceptance; else 'none'",
}
K_LIST = PROTOCOL["C1_k"]
ALPHAS = PROTOCOL["C2_alpha"]
SHRINK = PROTOCOL["shrink"]
FAR = PROTOCOL["far"]
CFGS = ["C0"] + [f"C1_k{k}" for k in K_LIST] + [f"C2_a{a}" for a in ALPHAS]
PAIRS = list(combinations(range(4), 2))
DKEYS = ["all"] + [f"p{a}{b}" for a, b in PAIRS]
EPS = [("P1", None, None, "all")] + [(f"P2_{A}{B}", A, B, f"p{min(A, B)}{max(A, B)}") for A, B in permutations(range(4), 2)]
P2_EPS = [e[0] for e in EPS if e[0] != "P1"]
N_BOOT, SEED = 10000, 249


def included_pairs(dkey):
    if dkey == "all":
        return list(PAIRS)
    ex = (int(dkey[1]), int(dkey[2]))
    return [p for p in PAIRS if p != ex]


# ---------------------------------------------------------------- split plan (shared by run and validation)
def outer_plan(subj, frac, outer):
    known = np.arange(1, 21)[np.arange(1, 21) != outer]
    t1, t2 = PROTOCOL["outer_D_temporal_split"]
    fit = np.isin(subj, known) & (frac <= t2)
    plan = {"known": known, "fit": fit, "d1": fit & (frac <= t1), "d2": fit & (frac > t1) & (frac <= t2),
            "late": frac > .8, "inner": []}
    i1, i2 = PROTOCOL["inner_D_temporal_split"]
    for held in known:
        remain = known[known != held]
        ifit = np.isin(subj, remain) & (frac <= i2)
        plan["inner"].append({"held": int(held), "remain": remain, "fit": ifit,
                              "d1": ifit & (frac <= i1), "d2": ifit & (frac > i1) & (frac <= i2),
                              "cal_unknown": (subj == held) & (frac > .4) & (frac <= .6),
                              "cal_known": np.isin(subj, remain) & (frac > .4) & (frac <= .6)})
    return plan


def validate_plan(subj, frac, outer, plan):
    middle_late = frac > .6
    assert not plan["fit"][subj == outer].any()
    for m in ("d1", "d2"):
        assert not plan[m][subj == outer].any(), "outer D uses outer identity"
        assert not (plan[m] & middle_late).any(), "outer D uses Middle/Late"
        assert not (plan[m] & ~plan["fit"]).any()
    assert not (plan["d1"] & plan["d2"]).any()
    assert not (plan["late"] & plan["fit"]).any()
    for inn in plan["inner"]:
        held = inn["held"]
        for m in ("fit", "d1", "d2"):
            assert not inn[m][(subj == held) | (subj == outer)].any(), "inner fit/D uses held or outer identity"
            assert not (inn[m] & (frac > .4)).any(), "inner fit/D beyond frac 0.4"
        assert not (inn["d1"] & inn["d2"]).any()
        assert not (inn["cal_unknown"] & inn["fit"]).any() and not (inn["cal_known"] & inn["fit"]).any()
        assert not inn["cal_unknown"][subj != held].any()
        assert not inn["cal_known"][(subj == held) | (subj == outer)].any()
        assert not ((inn["cal_unknown"] | inn["cal_known"]) & middle_late).any()
    for name, A, B, dkey in EPS:
        if A is not None:
            assert (min(A, B), max(A, B)) not in included_pairs(dkey), f"{name} D contains evaluated task pair"
            assert len(included_pairs(dkey)) == 5


# ---------------------------------------------------------------- fitting
def project_all(space, feats, tasks):
    P = torch.empty(len(tasks), DIM, device=DEV)
    for t in range(4):
        ix = np.flatnonzero(tasks == t)
        P[torch.as_tensor(ix, device=DEV)] = space.proj(feats[torch.as_tensor(ix, device=DEV)], tasks[ix])
    return P


def group_means(P, rows, subj, tasks, known):
    ix = np.flatnonzero(rows)
    gid = torch.as_tensor((subj[ix] - 1) * 4 + tasks[ix], device=DEV)
    sums = torch.zeros(80, DIM, dtype=torch.float64, device=DEV).index_add_(0, gid, P[torch.as_tensor(ix, device=DEV)].double())
    cnt = torch.bincount(gid, minlength=80).double()
    want = torch.as_tensor(np.array([(s - 1) * 4 + t for s in known for t in range(4)]), device=DEV)
    assert bool((cnt[want] > 0).all()), "empty temporal half"
    return sums[want] / cnt[want, None]


def fit_all(feats, tasks, subj, known, fit, d1, d2):
    assert not ((d1 | d2) & ~fit).any()
    fitted = R.fit_model(feats, tasks, subj, fit, known)
    P = project_all(fitted["space"], feats, tasks)
    protos = fitted["protos"].double()
    ix = np.flatnonzero(fit)
    r = P[torch.as_tensor(ix, device=DEV)].double() - protos[torch.as_tensor(tasks[ix], device=DEV), torch.as_tensor(subj[ix] - 1, device=DEV)]
    Sw = r.T @ r / (len(r) - 1)
    dt = group_means(P, d2, subj, tasks, known) - group_means(P, d1, subj, tasks, known)
    Dt = dt.T @ dt / len(dt)
    kidx = torch.as_tensor(known - 1, device=DEV)
    pair_s = {}
    for a, b in PAIRS:
        d = protos[a, kidx] - protos[b, kidx]
        d = d - d.mean(0)
        pair_s[(a, b)] = d.T @ d / len(d)
    Ds = {}
    for key in DKEYS:
        inc = included_pairs(key)
        Dx = sum(pair_s[p] for p in inc) / len(inc)
        Ds[key] = 0.5 * Dt / torch.trace(Dt) + 0.5 * Dx / torch.trace(Dx)
    return fitted, Sw, Ds, P


def whiten_cov(S):
    S = (S + S.T) / 2
    w, v = torch.linalg.eigh(S)
    return R.whiten((w.float(), v.float(), float(w.mean())), SHRINK)


def transforms(fitted, Sw, Ds):
    W0 = R.whiten(fitted["eig"], SHRINK)
    I = torch.eye(DIM, dtype=torch.float64, device=DEV)
    T = {}
    for key, D in Ds.items():
        T[("C0", key)] = W0
        _, dv = torch.linalg.eigh((D + D.T) / 2)
        for k in K_LIST:
            U = dv[:, -k:]
            Pk = I - U @ U.T
            T[(f"C1_k{k}", key)] = Pk.float() @ whiten_cov(Pk @ Sw @ Pk)
        for a in ALPHAS:
            T[(f"C2_a{a}", key)] = whiten_cov(Sw + a * D * torch.trace(Sw) / torch.trace(D))
    return T


def all_task_scores(P, ix, T, fitted):
    x = F.normalize(P[torch.as_tensor(ix, device=DEV)] @ T, dim=1)
    pr = F.normalize(fitted["protos"] @ T, dim=2)
    mat = torch.einsum("nd,tsd->nts", x, pr)
    mat[:, :, ~fitted["valid"]] = -float("inf")
    mx, who = mat.max(2)
    return mx.cpu().numpy().astype(np.float64), who.cpu().numpy().astype(np.int64) + 1


def score_table(feats, tasks, P, rows, fitted, T):
    """Return {(cfg, ep): (scores, pred, row_index)} for all configs/endpoints on `rows`."""
    ix = np.flatnonzero(rows)
    tk = tasks[ix]
    out = {}
    cache = {}
    for cfg in CFGS:
        for name, A, B, dkey in EPS:
            ck = (cfg, "all" if cfg == "C0" else dkey)
            if ck not in cache:
                cache[ck] = all_task_scores(P, ix, T[(cfg, ck[1])], fitted)
            mx, who = cache[ck]
            if A is None:
                ar = np.arange(len(ix))
                out[(cfg, name)] = (mx[ar, tk], who[ar, tk], ix)
            else:
                sel = tk == B
                out[(cfg, name)] = (mx[sel, A], who[sel, A], ix[sel])
    sv, who, rix = R.score(feats, tasks, rows, fitted, "strict_ldacos_shrink001")
    assert np.array_equal(rix, ix)
    assert np.allclose(sv, out[("C0", "P1")][0], rtol=1e-5, atol=1e-5), "general path disagrees with E247 score"
    out[("C0", "P1")] = (sv, who, rix)  # exact E247 path for C0/P1
    return out


# ---------------------------------------------------------------- fold
def ep_metrics(sv, who, subj, n, rix, outer, thr):
    rows = np.zeros(n, dtype=bool)
    rows[rix] = True
    m = R.metrics(sv, who, subj, rows, outer, thr)
    kn = subj[rix] != outer
    m["closed_set_acc"] = float((who[kn] == subj[rix][kn]).mean())
    return m


MCOLS = ["AUC", "F1_unknown_rejection", "known_acceptance", "unknown_rejection", "closed_set_acc",
         "known_identification_accuracy", "actual_FAR"]


def run_fold(feats, tasks, subj, frac, outer):
    plan = outer_plan(subj, frac, outer)
    validate_plan(subj, frac, outer, plan)
    unk = {(c, e[0]): [] for c in CFGS for e in EPS}
    kn = {(c, e[0]): [] for c in CFGS for e in EPS}
    for inn in plan["inner"]:
        fitted, Sw, Ds, P = fit_all(feats, tasks, subj, inn["remain"], inn["fit"], inn["d1"], inn["d2"])
        T = transforms(fitted, Sw, Ds)
        su = score_table(feats, tasks, P, inn["cal_unknown"], fitted, T)
        sk = score_table(feats, tasks, P, inn["cal_known"], fitted, T)
        for key in unk:
            unk[key].append(su[key][0])
            kn[key].append(sk[key][0])
    thr = {f"{c}|{e}": R.weighted_far_threshold(unk[(c, e)], FAR) for c, e in unk}
    inner_ka = {f"{c}|{e}": float(np.mean([(v >= thr[f"{c}|{e}"]).mean() for v in kn[(c, e)]])) for c, e in unk}
    sel_score = {c: float(np.mean([inner_ka[f"{c}|{e}"] for e in P2_EPS])) for c in CFGS}
    selected = {}
    for fam in ("C1", "C2"):
        opts = [c for c in CFGS if c.startswith(fam + "_")]
        selected[fam] = opts[int(np.argmax([sel_score[c] for c in opts]))]

    fitted, Sw, Ds, P = fit_all(feats, tasks, subj, plan["known"], plan["fit"], plan["d1"], plan["d2"])
    T = transforms(fitted, Sw, Ds)
    assert not (plan["late"] & plan["fit"]).any()
    so = score_table(feats, tasks, P, plan["late"], fitted, T)
    n = len(subj)
    rows, pair_rows = [], []
    for cfg in CFGS:
        p2 = []
        for name, A, B, _ in EPS:
            sv, who, rix = so[(cfg, name)]
            m = ep_metrics(sv, who, subj, n, rix, outer, thr[f"{cfg}|{name}"])
            base = {"outer_subject": outer, "candidate": cfg, "threshold_value": thr[f"{cfg}|{name}"],
                    "inner_known_acceptance": inner_ka[f"{cfg}|{name}"]}
            if A is None:
                rows.append({**base, "endpoint": "P1", **m})
            else:
                pair_rows.append({**base, "endpoint": "P2", "proto_task": A, "eval_task": B, **m})
                p2.append(m)
        rows.append({"outer_subject": outer, "candidate": cfg, "endpoint": "P2", "threshold_value": float("nan"),
                     "inner_known_acceptance": float(np.mean([inner_ka[f"{cfg}|{e}"] for e in P2_EPS])),
                     **{k: float(np.mean([m[k] for m in p2])) for k in MCOLS},
                     **{k: int(np.sum([m[k] for m in p2])) for k in ("accepted_count", "known_count", "unknown_count")}})
    for r in rows:
        r["inner_selection_score"] = sel_score[r["candidate"]]
        r["hparam"] = r["candidate"].split("_", 1)[1] if "_" in r["candidate"] else ""
    for fam, cfg in selected.items():
        rows += [{**r, "candidate": fam, "selected_config": cfg} for r in rows if r["candidate"] == cfg]

    fold_dir = RES / f"outer_{outer:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    mp = fold_dir / "model.pt"
    torch.save({"protocol": PROTOCOL, "outer_subject": outer, "known": plan["known"].tolist(),
                "fitted": R.cpu_model(fitted), "Sw": Sw.cpu(), "D": {k: v.cpu() for k, v in Ds.items()},
                "selected": selected, "inner_selection_score": sel_score, "thresholds": thr}, mp)
    saved = torch.load(mp, map_location="cpu", weights_only=False)
    f2 = R.device_model(saved["fitted"])
    T2 = transforms(f2, saved["Sw"].to(DEV), {k: v.to(DEV) for k, v in saved["D"].items()})
    so2 = score_table(feats, tasks, project_all(f2["space"], feats, tasks), plan["late"], f2, T2)
    for key in so:
        assert np.allclose(so[key][0], so2[key][0], rtol=1e-5, atol=1e-6), f"reload mismatch {key}"
        assert np.array_equal(so[key][2], so2[key][2])
    split = {"protocol_version": PROTOCOL["version"], "outer_subject": outer, "known": plan["known"].tolist(),
             "outer_fit_rows": int(plan["fit"].sum()), "outer_D_rows": [int(plan["d1"].sum()), int(plan["d2"].sum())],
             "late_rows": int(plan["late"].sum()), "selected": selected, "reload_check": "passed",
             "metadata_sha256": hashlib.sha256(R.META_PATH.read_bytes()).hexdigest(),
             "feature_sha256": hashlib.sha256(R.FEATURE_PATH.read_bytes()).hexdigest()}
    (fold_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (fold_dir / "results.json").write_text(json.dumps({"rows": rows, "pairs": pair_rows}, indent=2), encoding="utf-8")
    return rows, pair_rows


# ---------------------------------------------------------------- summary
def write_outputs(rows, pair_rows):
    RES.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "per_fold.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(RES / "per_fold_p2_pairs.csv", index=False)
    summ = df.groupby(["candidate", "endpoint"], as_index=False).agg({m: ["mean", "std", "min"] for m in MCOLS})
    summ.columns = ["_".join(x).strip("_") for x in summ.columns]
    summ.to_csv(RES / "summary.csv", index=False)

    rng = np.random.default_rng(SEED)
    folds = sorted(df.outer_subject.unique())
    idx = rng.integers(0, len(folds), (N_BOOT, len(folds)))
    ci = []
    for fam in ("C1", "C2"):
        for ep in ("P1", "P2"):
            a = df[(df.candidate == fam) & (df.endpoint == ep)].set_index("outer_subject").loc[folds]
            b = df[(df.candidate == "C0") & (df.endpoint == ep)].set_index("outer_subject").loc[folds]
            for m in ("known_acceptance", "unknown_rejection", "AUC"):
                d = (a[m] - b[m]).to_numpy()
                boot = d[idx].mean(1)
                ci.append({"contrast": f"{fam}-C0", "endpoint": ep, "metric": m, "mean_diff": float(d.mean()),
                           "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                           "n_folds": len(folds)})
    cidf = pd.DataFrame(ci)
    cidf.to_csv(RES / "paired_ci.csv", index=False)

    def mean_of(c, ep, m):
        return float(df[(df.candidate == c) & (df.endpoint == ep)][m].mean())

    crit = {}
    for fam in ("C1", "C2"):
        r = cidf[(cidf.contrast == f"{fam}-C0") & (cidf.endpoint == "P2") & (cidf.metric == "known_acceptance")].iloc[0]
        crit[fam] = {"p2_known_acc_diff": r.mean_diff, "ci": [r.ci_low, r.ci_high],
                     "p1_unknown_rejection": mean_of(fam, "P1", "unknown_rejection"),
                     "p2_unknown_rejection": mean_of(fam, "P2", "unknown_rejection"),
                     "p2_known_acceptance": mean_of(fam, "P2", "known_acceptance")}
        c = crit[fam]
        c["pass"] = bool(c["p2_known_acc_diff"] > 0 and c["ci"][0] > 0 and c["p1_unknown_rejection"] >= .88 and c["p2_unknown_rejection"] >= .88)
    passing = [f for f in crit if crit[f]["pass"]]
    choice = max(passing, key=lambda f: crit[f]["p2_known_acceptance"]) if passing else "none"
    worst = {}
    for c in ("C0", "C1", "C2"):
        for ep in ("P1", "P2"):
            w = df[(df.candidate == c) & (df.endpoint == ep)].nsmallest(3, "unknown_rejection")
            worst[f"{c}|{ep}"] = [{"subject": int(s), "unknown_rejection": float(v)} for s, v in zip(w.outer_subject, w.unknown_rejection)]
    hp = {fam: df[(df.candidate == fam) & (df.endpoint == "P1")].set_index("outer_subject")["selected_config"].to_dict() for fam in ("C1", "C2")}
    repro = {}
    e247 = ROOT / "research" / "E247_train_temporal_loso" / "results" / "per_fold.csv"
    if e247.exists():
        ref = pd.read_csv(e247)
        ref = ref[(ref.candidate == "strict_ldacos_shrink001") & (ref.block == "late") & (ref.threshold == "far_10")].set_index("outer_subject")
        mine = df[(df.candidate == "C0") & (df.endpoint == "P1")].set_index("outer_subject")
        common = [f for f in folds if f in ref.index]
        for m in ("AUC", "known_acceptance", "unknown_rejection", "F1_unknown_rejection"):
            repro[m] = {"max_abs_diff_per_fold": float(np.max(np.abs(mine.loc[common, m] - ref.loc[common, m]))),
                        "E249_mean": float(mine.loc[common, m].mean()), "E247_mean": float(ref.loc[common, m].mean())}
        repro["threshold_max_abs_diff"] = float(np.max(np.abs(mine.loc[common, "threshold_value"] - ref.loc[common, "threshold_value"])))
        repro["folds"] = [int(f) for f in common]
    sel = {"protocol": PROTOCOL, "criterion_results": crit, "selection": choice, "selected_hparams_per_fold": hp,
           "worst3_unknown_rejection": worst, "C0_vs_E247_late_far10": repro, "n_folds": len(folds)}
    (RES / "selection.json").write_text(json.dumps(sel, indent=2, default=float), encoding="utf-8")
    show = summ[summ.candidate.isin(["C0", "C1", "C2"])][["candidate", "endpoint", "AUC_mean", "F1_unknown_rejection_mean",
                                                          "known_acceptance_mean", "unknown_rejection_mean", "closed_set_acc_mean"]]
    print(show.to_string(index=False))
    print(cidf.to_string(index=False))
    print(json.dumps({"selection": choice, "criterion": crit, "repro": repro, "worst": worst}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--outer-subject", type=int, nargs="+")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    meta = pd.read_csv(R.META_PATH)
    feats = torch.as_tensor(np.array(np.load(R.FEATURE_PATH, mmap_mode="r"), dtype=np.float32, copy=True), device=DEV)
    R.validate(meta, feats)
    _, frac = R.temporal_blocks(meta)
    subj = meta.subject.to_numpy()
    tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))
    if args.validate_only:
        for outer in range(1, 21):
            validate_plan(subj, frac, outer, outer_plan(subj, frac, outer))
        print(json.dumps({"validate": "passed", "folds": 20, "eps": len(EPS), "dkeys": DKEYS, "device": str(DEV)}))
        return
    folds = list(range(1, 21)) if args.all_folds else (args.outer_subject or [])
    if not folds:
        raise SystemExit("choose --validate-only, --outer-subject N [N..], or --all-folds")
    rows, pairs = [], []
    t_all = time.time()
    for outer in folds:
        fd = RES / f"outer_{outer:02d}"
        if args.resume and (fd / "results.json").exists():
            sp = json.loads((fd / "split.json").read_text(encoding="utf-8"))
            if sp.get("protocol_version") == PROTOCOL["version"]:
                res = json.loads((fd / "results.json").read_text(encoding="utf-8"))
                rows += res["rows"]
                pairs += res["pairs"]
                continue
        t0 = time.time()
        r, p = run_fold(feats, tasks, subj, frac, outer)
        rows += r
        pairs += p
        print(f"outer={outer} seconds={time.time() - t0:.1f}", flush=True)
    print(f"total_seconds={time.time() - t_all:.1f}")
    write_outputs(rows, pairs)


if __name__ == "__main__":
    main()
