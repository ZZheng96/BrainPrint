"""E248: metric backfill for E245/E246/E247 (Train only, read-only for existing artifacts).

A. E245: recompute the 20-fold LDA-cosine scores and report the *true* TPR at unknown-FAR
   (FAR = fraction of unknown-identity eval rows wrongly accepted), plus the legacy
   "TPR_FAR1/5" quantities (which are actually known false-rejection operating points,
   i.e. the unknown rejection rate at 1%/5% known rejection) for fidelity checking.
B. E246: replay the exact 20 reps / 2 regimes / 2 centers simulation (same RNG call order),
   adding true TPR@FAR rows for the registered / known / new groups.
C. E247: read cached fold scores, add closed-set accuracy, subject-level bootstrap CIs and
   the paired strict_ldacos - strict_euclid CI.

Score convention: higher = more "known". No Test data is read. No other directory is written.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
E245_DIR = ROOT / "research" / "E245_lda_cos_deploy"
E246_DIR = ROOT / "research" / "E246_submission_ldacos"
E247_DIR = ROOT / "research" / "E247_train_temporal_loso"
FARS = (1, 5, 10)
BOOT = 10000
SEED = 20261117
TOL_FAR = 1e-9
TOL_AUC = 1e-9

log = logging.getLogger("e248")


def setup_log() -> Path:
    path = Path(tempfile.gettempdir()) / f"e248_metric_backfill_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    log.info("log file: %s", path)
    log.info("device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    return path


def part_a() -> dict:
    """E245: true TPR@FAR + legacy operating points."""
    sys.path.insert(0, str(E245_DIR))
    import run_e245 as e245  # noqa: E402

    assert Path(e245.ROOT) == ROOT, (e245.ROOT, ROOT)
    cfg = e245.e243.CFG
    meta = pd.read_csv(ROOT / cfg["data"]["train_metadata"])
    subj = meta.subject.to_numpy(int)
    tasks = meta.task.astype(str).to_numpy()
    fr = e245.e243.frac(meta)
    raw = np.load(ROOT / cfg["data"]["eeg"]).astype(np.float32)
    dim = int(cfg["pca_dim"])
    fit_all = fr <= float(cfg["fit_fraction"])
    eval_all = (fr > float(cfg["fit_fraction"])) & (fr <= float(cfg["score_fraction"][1]))
    rows_out = []
    t0 = time.time()
    for held in range(1, 21):
        ids = np.array([s for s in range(1, 21) if s != held])
        outer_sc, idx = e245.run_config_scores(raw, subj, tasks, fit_all, eval_all, ids, dim)
        is_unk = subj[idx] == held
        assert is_unk.sum() > 0 and (~is_unk).sum() > 0
        for c, v in outer_sc.items():
            k_s, u_s = v[~is_unk], v[is_unk]
            base = {"shrink": c[0], "center": c[1], "reduce": c[2], "held": held,
                    "AUC": e245.e243.auc(-u_s, -k_s),
                    "tpr_known_frr1": float((u_s < np.quantile(k_s, 0.01)).mean()),
                    "tpr_known_frr5": float((u_s < np.quantile(k_s, 0.05)).mean())}
            for far in FARS:
                thr = float(np.quantile(u_s, 1 - far / 100, method="higher"))
                far_real = float((u_s >= thr).mean())
                # quantile ties at the threshold are accepted, so the realised FAR can exceed the
                # nominal level by at most one unknown sample (n_u is small: ~62 rows per fold)
                assert far_real <= far / 100 + 1.0 / len(u_s) + TOL_FAR, (held, c, far, far_real)
                rows_out.append({**base, "far_level": far, "tpr": float((k_s >= thr).mean()),
                                 "far_real": far_real, "f1_unknown": e245.f1(k_s, u_s, thr)})
        log.info("E245 held=%d %.0fs", held, time.time() - t0)
    df = pd.DataFrame(rows_out)
    assert len(df) == 20 * 16 * len(FARS), len(df)
    df.to_csv(OUT / "e245_per_fold_far.csv", index=False)

    keys = ["shrink", "center", "reduce", "far_level"]
    g = df.groupby(keys, sort=False)
    summ = g.agg(mean=("tpr", "mean"), std=("tpr", "std"), min=("tpr", "min")).reset_index()
    extra = g.agg(AUC=("AUC", "mean"), far_real_mean=("far_real", "mean"),
                  f1_unknown_mean=("f1_unknown", "mean"),
                  tpr_known_frr1=("tpr_known_frr1", "mean"),
                  tpr_known_frr5=("tpr_known_frr5", "mean")).reset_index()
    summ = summ.merge(extra, on=keys).sort_values(keys).reset_index(drop=True)
    summ.to_csv(OUT / "e245_summary_far.csv", index=False)

    # fidelity: legacy quantities must reproduce E245 summary.csv TPR_FAR1/TPR_FAR5
    old = pd.read_csv(E245_DIR / "summary.csv")
    old["center"] = old["center"].astype(bool)
    ref = df.groupby(["shrink", "center", "reduce"]).agg(
        TPR_FAR1=("tpr_known_frr1", "mean"), TPR_FAR5=("tpr_known_frr5", "mean")).reset_index()
    m = old.merge(ref, on=["shrink", "center", "reduce"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), m["_merge"].value_counts()
    d1 = float(np.max(np.abs(m["TPR_FAR1_x"] - m["TPR_FAR1_y"])))
    d5 = float(np.max(np.abs(m["TPR_FAR5_x"] - m["TPR_FAR5_y"])))
    log.info("E245 fidelity vs summary.csv: max|dTPR_FAR1|=%.3e max|dTPR_FAR5|=%.3e", d1, d5)
    assert max(d1, d5) <= 1e-9, (d1, d5)

    sel = {}
    for center in (True, False):
        r = summ[(summ.shrink == 0.01) & (summ.center == center) & (summ.reduce == "max")]
        assert len(r) == len(FARS)
        key = f"0.01/{center}/max"
        entry = {f"far{int(x.far_level)}": {"tpr": float(x.mean),
                                            "far_real": float(x.far_real_mean),
                                            "f1_unknown": float(x.f1_unknown_mean)} for x in r.itertuples()}
        entry["legacy_frr1"] = float(r.tpr_known_frr1.iloc[0])
        entry["legacy_frr5"] = float(r.tpr_known_frr5.iloc[0])
        entry["delta_new_vs_legacy"] = {"far1_minus_frr1": entry["far1"]["tpr"] - entry["legacy_frr1"],
                                        "far5_minus_frr5": entry["far5"]["tpr"] - entry["legacy_frr5"]}
        sel[key] = entry
    return {"fidelity_max_abs_diff": {"TPR_FAR1": d1, "TPR_FAR5": d5},
            "selected": sel, "n_rows": len(df)}


def part_b() -> dict:
    """E246: replay simulation exactly, add TPR@FAR rows."""
    sys.path.insert(0, str(E246_DIR))
    import run_e246 as R  # noqa: E402

    M = R.M
    assert Path(R.ROOT) == ROOT
    t0 = time.time()
    tm = pd.read_csv(R.ROOT / "BrainPrint_cache/train_metadata.csv")
    tm["frac"] = tm.trial / tm.groupby(["subject", "task"])["trial"].transform("max")
    feats = torch.as_tensor(np.load(R.ROOT / "BrainPrint_cache/e201_final_selftest/train_raw_scores_inputs.npy"), device=R.DEV)
    tasks = M._task_indices([{"task": t} for t in tm.task], len(tm))
    subj, frac = tm.subject.to_numpy(), tm.frac.to_numpy()
    art = M.load_model(str(R.ROOT / "final/team_hymn"), str(R.DEV))
    tau = art["tau"]
    scorers = ["euclid"] + [f"ldacos{l}" for l in R.SHRINK]
    rng = np.random.default_rng(20261116)
    far_rows, fixed_rows, fid_rows = [], [], []
    for rep in range(20):
        perm = rng.permutation(np.arange(1, 21))
        new_s, unk_s, known = perm[:3], perm[3:6], np.sort(perm[6:])
        mapping = {int(s): 21 + k for k, s in enumerate(new_s)}
        pick = np.concatenate([rng.choice(np.flatnonzero((subj == s) & (frac <= 0.6)), 20, replace=False) for s in new_s])
        lab = np.array([mapping[int(s)] for s in subj[pick]])
        late = frac > 0.8
        r_known, r_new, r_unk = late & np.isin(subj, known), late & np.isin(subj, new_s), np.isin(subj, unk_s)
        truth_known, truth_new = subj[r_known], np.array([mapping[int(s)] for s in subj[r_new]])
        for regime in ["frozen", "honest"]:
            stat = np.isin(subj, known)
            space = R.Space(feats, tasks, art=art) if regime == "frozen" else R.Space(feats, tasks, stat_rows=stat, pca_rows=stat & (frac <= 0.6))
            protos, valid, fm, eig = R.build(feats, tasks, subj, frac, space, known, pick, lab)
            W = {"euclid": None, **{f"ldacos{l}": R.whiten(eig, l) for l in R.SHRINK}}
            inner = {(s_, c): [] for s_ in scorers for c in R.CENTER}
            for j in known:
                kj = known[known != j]
                if regime == "honest":
                    sj = np.isin(subj, kj)
                    space_j = R.Space(feats, tasks, stat_rows=sj, pca_rows=sj & (frac <= 0.6))
                else:
                    space_j = space
                pj, vj, fmj, eigj = R.build(feats, tasks, subj, frac, space_j, kj, pick, lab)
                Wj = {"euclid": None, **{f"ldacos{l}": R.whiten(eigj, l) for l in R.SHRINK}}
                for c in R.CENTER:
                    rows_j = late & np.isin(subj, known)
                    res = R.score(feats, tasks, rows_j, space_j, pj, vj, fmj, Wj, c, rng)
                    isj = subj[np.flatnonzero(rows_j)] == j
                    for s_ in scorers:
                        inner[(s_, c)].append(float(np.quantile(res[s_][0][isj], R.Q)))
            ev = r_known | r_new | r_unk
            ev_idx = np.flatnonzero(ev)
            gk, gn, gu = r_known[ev_idx], r_new[ev_idx], r_unk[ev_idx]
            reg_mask = gk | gn
            assert reg_mask.sum() > 0 and gu.sum() > 0
            for c in R.CENTER:
                res = R.score(feats, tasks, ev, space, protos, valid, fm, W, c, rng)
                for s_ in scorers:
                    mx, am = res[s_]
                    ths = {"loso": float(np.median(inner[(s_, c)]))}
                    if s_ == "euclid":
                        ths["tau"] = -tau
                    base_auc = R.auc(mx[~gu], mx[gu])
                    closed_acc = float(np.concatenate([am[gk] == truth_known, am[gn] == truth_new]).mean())
                    for th_name, th in ths.items():
                        pred = np.where(mx >= th, am, 0)
                        tp = int((pred[gu] == 0).sum()); fp = int((pred[~gu] == 0).sum()); fn = int(gu.sum()) - tp
                        fixed_rows.append({"rep": rep, "regime": regime, "scorer": s_, "center": c, "thr": th_name,
                                           "AUC": base_auc, "closed_acc": closed_acc,
                                           "tpr_reg": float((mx[reg_mask] >= th).mean()),
                                           "tpr_known": float((mx[gk] >= th).mean()),
                                           "tpr_new": float((mx[gn] >= th).mean()),
                                           "far_real": float((mx[gu] >= th).mean()),
                                           "f1_unknown": 2 * tp / max(2 * tp + fp + fn, 1)})
                        fid_rows.append({"rep": rep, "regime": regime, "scorer": s_, "center": c,
                                         "thr": th_name, "AUC": base_auc, "closed_acc": closed_acc})
                    for far in FARS:
                        thr = float(np.quantile(mx[gu], 1 - far / 100, method="higher"))
                        pred = np.where(mx >= thr, am, 0)
                        tp = int((pred[gu] == 0).sum()); fp = int((pred[~gu] == 0).sum()); fn = int(gu.sum()) - tp
                        far_real = float((mx[gu] >= thr).mean())
                        assert far_real <= far / 100 + 1.0 / gu.sum() + TOL_FAR, (rep, regime, s_, c, far, far_real)
                        far_rows.append({"rep": rep, "regime": regime, "scorer": s_, "center": c,
                                         "far_level": far, "AUC": base_auc,
                                         "tpr_reg": float((mx[reg_mask] >= thr).mean()),
                                         "tpr_known": float((mx[gk] >= thr).mean()),
                                         "tpr_new": float((mx[gn] >= thr).mean()),
                                         "far_real": far_real,
                                         "f1_unknown": 2 * tp / max(2 * tp + fp + fn, 1),
                                         "closed_acc": closed_acc})
        log.info("E246 rep=%d %.0fs", rep, time.time() - t0)

    far_df = pd.DataFrame(far_rows)
    assert len(far_df) == 20 * 2 * 3 * 2 * len(FARS), len(far_df)
    far_df.to_csv(OUT / "e246_per_rep_far.csv", index=False)

    keys = ["regime", "scorer", "center", "far_level"]
    g = far_df.groupby(keys, sort=False)
    summ = g.agg(mean=("tpr_reg", "mean"), std=("tpr_reg", "std"), min=("tpr_reg", "min")).reset_index()
    extra = g.agg(AUC=("AUC", "mean"),
                  tpr_known_mean=("tpr_known", "mean"), tpr_new_mean=("tpr_new", "mean"),
                  far_real_mean=("far_real", "mean"), f1_unknown_mean=("f1_unknown", "mean"),
                  closed_acc_mean=("closed_acc", "mean")).reset_index()
    summ = summ.merge(extra, on=keys).sort_values(keys).reset_index(drop=True)
    summ.to_csv(OUT / "e246_summary_far.csv", index=False)

    # fidelity vs E246 per_rep.csv: every (rep, regime, scorer, center, thr) AUC and closed_acc
    fid = pd.DataFrame(fid_rows)
    ref = pd.read_csv(E246_DIR / "per_rep.csv")
    m = fid.merge(ref[["rep", "regime", "scorer", "center", "thr", "AUC", "closed_acc"]],
                  on=["rep", "regime", "scorer", "center", "thr"], how="outer", indicator=True, suffixes=("", "_ref"))
    assert (m["_merge"] == "both").all() and len(m) == len(ref), (m["_merge"].value_counts(), len(m), len(ref))
    d_auc = float(np.max(np.abs(m["AUC"] - m["AUC_ref"])))
    d_cls = float(np.max(np.abs(m["closed_acc"] - m["closed_acc_ref"])))
    log.info("E246 fidelity vs per_rep.csv: max|dAUC|=%.3e max|dclosed_acc|=%.3e (rows=%d)", d_auc, d_cls, len(m))
    if d_auc > TOL_AUC:
        raise RuntimeError(f"E246 AUC fidelity failed: max abs diff {d_auc:.6e} > {TOL_AUC:.0e}")
    assert d_cls <= 1e-9, d_cls

    fixed = pd.DataFrame(fixed_rows)
    assert len(fixed) == 20 * 2 * 2 * 4, len(fixed)
    fx = fixed.groupby(["regime", "scorer", "center", "thr"]).mean(numeric_only=True).reset_index()
    fx.to_csv(OUT / "e246_summary_fixed_thr.csv", index=False)

    sel = {}
    for scorer in ("ldacos0.01", "euclid"):
        r = summ[(summ.regime == "honest") & (summ.scorer == scorer) & (summ.center == "none")]
        sel[f"honest/{scorer}/none"] = {f"far{int(x.far_level)}": {"tpr_reg": float(x.mean),
                                                                   "far_real": float(x.far_real_mean)} for x in r.itertuples()}
    return {"fidelity_max_abs_diff": {"AUC": d_auc, "closed_acc": d_cls},
            "selected": sel, "n_rows": len(far_df)}


def _boot_ci(values: np.ndarray, rng) -> dict:
    idx = rng.integers(0, len(values), size=(BOOT, len(values)))
    means = values[idx].mean(1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n_folds": int(len(values)), "n_boot": BOOT}


def part_c() -> dict:
    """E247: closed-set accuracy, bootstrap CIs, paired CI (cached results only)."""
    meta = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    subj_all = meta.subject.to_numpy(int)
    per_fold = pd.read_csv(E247_DIR / "results" / "per_fold.csv")
    rows, fid = [], []
    for outer in range(1, 21):
        fold = E247_DIR / "results" / f"outer_{outer:02d}"
        split = json.loads((fold / "split.json").read_text(encoding="utf-8"))
        assert int(split["outer_subject"]) == outer
        d = np.load(fold / "scores_and_rows.npz")
        sub = per_fold[per_fold.outer_subject == outer]
        for cand in ("frozen_euclid_tau", "strict_euclid", "strict_ldacos_shrink001"):
            for block in ("middle", "late"):
                key = f"{cand}_{block}"
                sv, idx, pid = d[f"{key}_score"], d[f"{key}_row_index"], d[f"{key}_pred_id"]
                true = subj_all[idx]
                known = true != outer
                assert known.any() and (~known).any()
                closed_acc = float((pid[known] == true[known]).mean())
                for r in sub[(sub.candidate == cand) & (sub.block == block)].itertuples():
                    assert r.known_count + r.unknown_count == len(idx)
                    assert r.known_count == int(known.sum())
                    pred = np.where(sv >= r.threshold_value, pid, 0)
                    acc = float((pred[known] == true[known]).mean())
                    fid.append(abs(acc - r.known_identification_accuracy))
                    rows.append({"outer_subject": outer, "candidate": cand, "block": block,
                                 "threshold": r.threshold, "threshold_value": float(r.threshold_value),
                                 "AUC": float(r.AUC), "actual_FAR": float(r.actual_FAR),
                                 "known_acceptance": float(r.known_acceptance),
                                 "unknown_rejection": float(r.unknown_rejection),
                                 "known_identification_accuracy": float(r.known_identification_accuracy),
                                 "closed_set_acc": closed_acc})
    extra = pd.DataFrame(rows)
    assert len(extra) == 20 * 14, len(extra)
    extra.to_csv(OUT / "e247_extra.csv", index=False)
    d_acc = float(np.max(fid))
    log.info("E247 fidelity vs per_fold known_identification_accuracy: max abs diff=%.3e", d_acc)
    assert d_acc <= 1e-9, d_acc

    base = extra[(extra.threshold == "far_10") & extra.candidate.isin(["strict_euclid", "strict_ldacos_shrink001"])]
    metrics = ["AUC", "known_acceptance", "unknown_rejection", "closed_set_acc"]
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for (cand, block), g in base.groupby(["candidate", "block"], sort=False):
        assert len(g) == 20, (cand, block, len(g))
        for met in metrics:
            boot_rows.append({"candidate": cand, "block": block, "metric": met,
                              "resample_unit": "outer_subject", **_boot_ci(g[met].to_numpy(float), rng)})
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "e247_bootstrap.csv", index=False)

    paired_rows = []
    for block in ("middle", "late"):
        piv = base[base.block == block].pivot_table(index="outer_subject", columns="candidate", values=metrics)
        assert len(piv) == 20
        for met in metrics:
            delta = (piv[(met, "strict_ldacos_shrink001")] - piv[(met, "strict_euclid")]).to_numpy(float)
            st = _boot_ci(delta, rng)
            paired_rows.append({"block": block, "metric": met, "delta": "strict_ldacos_shrink001-strict_euclid",
                                "delta_mean": st["mean"], "ci_low": st["ci_low"], "ci_high": st["ci_high"],
                                "n_pairs": st["n_folds"], "n_boot": BOOT})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "e247_paired_ci.csv", index=False)

    closed = extra[extra.threshold == "far_10"].groupby(["candidate", "block"]).closed_set_acc.mean()
    ci = boot[boot.metric == "closed_set_acc"].set_index(["candidate", "block"])[["ci_low", "ci_high"]]
    sel = {f"{c}/{b}": {"closed_set_acc": float(closed[(c, b)]),
                        "ci_low": float(ci.loc[(c, b), "ci_low"]),
                        "ci_high": float(ci.loc[(c, b), "ci_high"])} for c, b in closed.index}
    return {"fidelity_max_abs_diff_known_identification_accuracy": d_acc,
            "closed_set_acc": sel,
            "paired_ldacos_minus_euclid": paired.to_dict("records"),
            "n_rows": len(extra)}


def main():
    setup_log()
    t0 = time.time()
    a = part_a()
    b = part_b()
    c = part_c()
    summary = {
        "script": str(Path(__file__).resolve()),
        "runtime_s": time.time() - t0,
        "protocol": {
            "far_definition": "fraction of unknown-identity eval rows accepted (score >= threshold); "
                              "FAR threshold = np.quantile(unknown_scores, 1-far/100, method='higher')",
            "legacy_E245_naming": "E245 TPR_FAR1/TPR_FAR5 = unknown rejection rate at 1%/5% known false rejection "
                                  "(known false-rejection operating point), NOT TPR@FAR",
            "score_convention": "higher = more known",
            "test_data_read": False,
        },
        "E245": a, "E246": b, "E247": c,
    }
    (OUT / "e248_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", OUT / "e248_summary.json")
    log.info("total %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
