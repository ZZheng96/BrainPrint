"""E244: compare open-set scoring functions and cohort normalisation under strict LOSO.

Same split as E243 (fit fraction <= 0.6, eval (0.6, 0.8], last outer-train subject = validation
for MLP early stopping). Fix vs E243: known_eval only contains *enrolled* subjects (the validation
subject is neither enrolled nor held, so it is excluded from both known and unknown).
Test is never read. Score convention: higher = more "known".
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E243_de_deep_fusion"))
import run_e243 as e243  # noqa: E402

DEVICE = e243.DEVICE
TASKS = e243.TASKS
SEED = 20261117
FEATSETS = {"e65": [], "e65_e146": ["wpli", "pac", "spectral_entropy"]}
E146 = {k: f"research/E146_phase_features/train_{k}.npy" for k in ["wpli", "pac", "spectral_entropy"]}


def auc(known, unknown):
    # positive = unknown, lower score = more unknown
    return e243.auc(-unknown, -known)


def tpr_at_far(known, unknown, far):
    th = float(np.quantile(known, far, method="lower"))
    return float((unknown < th).mean())


def f1_at_acc(known, unknown, acc=0.90):
    th = float(np.quantile(known, 1 - acc, method="lower"))
    tp = int((unknown < th).sum()); fp = int((known < th).sum()); fn = len(unknown) - tp
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def prototypes(x, subj, tasks, mask, ids):
    """Per (task, subject) centroid on GPU, plus pooled within-class covariance."""
    xt = torch.as_tensor(x, device=DEVICE)
    protos = {}
    resid = []
    for t in TASKS:
        rows = []
        for s in ids:
            m = torch.as_tensor(np.flatnonzero(mask & (tasks == t) & (subj == s)), device=DEVICE)
            mu = xt[m].mean(0)
            rows.append(mu); resid.append(xt[m] - mu)
        protos[t] = torch.stack(rows)
    r = torch.cat(resid).double()
    cov = r.T @ r / max(len(r) - 1, 1)
    lam = 0.1
    cov = (1 - lam) * cov + lam * torch.trace(cov) / cov.shape[0] * torch.eye(cov.shape[0], device=DEVICE, dtype=cov.dtype)
    w, v = torch.linalg.eigh(cov)
    whiten = (v / w.sqrt()).float()  # x @ whiten -> within-class whitened
    return protos, whiten


def proto_scores(x, tasks, rows, protos, whiten):
    """Return dict of per-row similarity matrices [n, n_enrolled] for each geometry."""
    xt = torch.as_tensor(x[rows], device=DEVICE)
    tr = tasks[rows]
    n, k = len(rows), next(iter(protos.values())).shape[0]
    out = {g: torch.empty(n, k, device=DEVICE) for g in ["cos", "euc", "maha", "lda_cos"]}
    for t in TASKS:
        idx = torch.as_tensor(np.flatnonzero(tr == t), device=DEVICE)
        if not len(idx):
            continue
        a, p = xt[idx], protos[t]
        out["cos"][idx] = torch.nn.functional.normalize(a, dim=1) @ torch.nn.functional.normalize(p, dim=1).T
        out["euc"][idx] = -torch.cdist(a, p)
        aw, pw = a @ whiten, p @ whiten
        out["maha"][idx] = -torch.cdist(aw, pw)
        out["lda_cos"][idx] = torch.nn.functional.normalize(aw, dim=1) @ torch.nn.functional.normalize(pw, dim=1).T
    return out


def reduce_scores(sim, znorm_stats=None):
    top2 = sim.topk(2, dim=1).values
    res = {"max": top2[:, 0], "margin": top2[:, 0] - top2[:, 1]}
    # t-norm: top1 vs the distribution of the remaining enrolled prototypes (cohort = enrolled ids)
    rest = sim.sort(dim=1, descending=True).values[:, 1:]
    res["tnorm"] = (top2[:, 0] - rest.mean(1)) / (rest.std(1) + 1e-6)
    if znorm_stats is not None:
        mu, sd = znorm_stats
        res["znorm"] = ((sim - mu) / sd).max(1).values
    return {k: v.cpu().numpy().astype(np.float64) for k, v in res.items()}


def znorm_stats(sim_train, subj_train, ids):
    """Per-prototype impostor stats from training rows of *other* enrolled subjects."""
    st = torch.as_tensor(subj_train, device=DEVICE)
    mus, sds = [], []
    for j, s in enumerate(ids):
        imp = sim_train[st != int(s), j]
        mus.append(imp.mean()); sds.append(imp.std() + 1e-6)
    return torch.stack(mus)[None], torch.stack(sds)[None]


def mlp_scores(model, x, tasks):
    model.eval()
    xx = torch.as_tensor(x, device=DEVICE); tt = torch.as_tensor(e243.onehot(tasks), device=DEVICE)
    with torch.no_grad():
        lg = model(xx, tt)
    top2 = lg.topk(2, 1).values
    return {"mlp_msp": torch.softmax(lg, 1).max(1).values.cpu().numpy().astype(np.float64),
            "mlp_energy": torch.logsumexp(lg, 1).cpu().numpy().astype(np.float64),
            "mlp_logit_margin": (top2[:, 0] - top2[:, 1]).cpu().numpy().astype(np.float64)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=20)
    args = ap.parse_args(); t0 = time.time()
    (OUT / "models").mkdir(exist_ok=True); (OUT / "scores").mkdir(exist_ok=True)
    cfg = e243.CFG
    meta = pd.read_csv(ROOT / cfg["data"]["train_metadata"])
    subj, tasks, fr = meta.subject.to_numpy(int), meta.task.astype(str).to_numpy(), e243.frac(meta)
    raw = {"e65": np.load(ROOT / cfg["data"]["eeg"]).astype(np.float32)}
    for k, p in E146.items():
        raw[k] = np.load(ROOT / p).astype(np.float32)
    fit_all = fr <= float(cfg["fit_fraction"])
    eval_all = (fr > float(cfg["fit_fraction"])) & (fr <= float(cfg["score_fraction"][1]))
    dim = int(cfg["pca_dim"])
    scores, folds = {}, []
    for held in range(1, args.folds + 1):
        outer = fit_all & (subj != held)
        ids = np.sort(np.unique(subj[outer]))
        val_s = int(ids[-1]); ids = ids[:-1]
        train = outer & (subj != val_s); val = outer & (subj == val_s)
        known = eval_all & np.isin(subj, ids); unknown = eval_all & (subj == held)
        pseudo = eval_all & (subj == val_s)  # unenrolled validation identity -> pseudo-unknown for threshold
        sel = np.flatnonzero(known | unknown | pseudo); is_unk = unknown[sel]; is_ps = pseudo[sel]; is_kn = known[sel]
        for fs, extra in FEATSETS.items():
            blocks, states = [], {}
            for name in ["e65"] + extra:
                z, st = e243.standardize_and_pca(raw[name], tasks, train, dim)
                blocks.append(z); states[name] = st
            x = np.concatenate(blocks, 1).astype(np.float32)
            fold_scores = {}
            if fs == "e65":  # MLP baselines only on E65 (identical to E243 eeg_only training)
                model, _ = e243.train_model(x, tasks, subj, train, val, SEED + held)
                fold_scores.update(mlp_scores(model, x[sel], tasks[sel]))
                torch.save({"model": model.state_dict(), "states": states, "held": held, "val": val_s},
                           OUT / "models" / f"outer_{held:02d}_mlp_e65.pt")
            protos, whiten = prototypes(x, subj, tasks, train, ids)
            tr_rows = np.flatnonzero(train)
            sim_tr = proto_scores(x, tasks, tr_rows, protos, whiten)
            sim_ev = proto_scores(x, tasks, sel, protos, whiten)
            for g in sim_ev:
                zs = znorm_stats(sim_tr[g], subj[tr_rows], ids)
                for r, v in reduce_scores(sim_ev[g], zs).items():
                    fold_scores[f"{g}_{r}"] = v
            torch.save({"protos": {t: p.cpu() for t, p in protos.items()}, "whiten": whiten.cpu(), "states": states,
                        "enrolled": ids, "held": held}, OUT / "models" / f"outer_{held:02d}_proto_{fs}.pt")
            for name, v in fold_scores.items():
                key = f"{fs}:{name}" if not name.startswith("mlp") else name
                k_s, u_s, p_s = v[is_kn], v[is_unk], v[is_ps]
                scores[f"{key}|{held:02d}|known"] = k_s; scores[f"{key}|{held:02d}|unknown"] = u_s
                scores[f"{key}|{held:02d}|pseudo"] = p_s
                th = float(np.quantile(p_s, 0.90))
                tp = int((u_s < th).sum()); fp = int((k_s < th).sum())
                folds.append({"scorer": key, "held": held, "AUC": auc(k_s, u_s),
                              "dep_known_acc": float((k_s >= th).mean()), "dep_unknown_rej": float((u_s < th).mean()),
                              "dep_F1": float(2 * tp / max(2 * tp + fp + len(u_s) - tp, 1)),
                              "TPR_FAR1": tpr_at_far(k_s, u_s, .01), "TPR_FAR5": tpr_at_far(k_s, u_s, .05),
                              "TPR_FAR10": tpr_at_far(k_s, u_s, .10), "F1_Acc90": f1_at_acc(k_s, u_s)})
        print(f"held={held} done {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT / "scores" / "outer_scores.npz", **scores)
    df = pd.DataFrame(folds); df.to_csv(OUT / "per_fold.csv", index=False)
    base = df[df.scorer == "mlp_msp"].set_index("held")
    summ = []
    for sc, g in df.groupby("scorer"):
        g = g.set_index("held").loc[base.index]
        row = {"scorer": sc, **{m: float(g[m].mean()) for m in ["AUC", "TPR_FAR1", "TPR_FAR5", "TPR_FAR10", "F1_Acc90", "dep_known_acc", "dep_unknown_rej", "dep_F1"]}}
        for m in ["AUC", "F1_Acc90", "TPR_FAR5", "dep_F1"]:
            d = g[m].to_numpy() - base[m].to_numpy()
            row[f"d_{m}"] = float(d.mean()); row[f"se_{m}"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            row[f"win_{m}"] = int((d > 0).sum())
        summ.append(row)
    s = pd.DataFrame(summ).sort_values("F1_Acc90", ascending=False)
    s.to_csv(OUT / "summary.csv", index=False)
    (OUT / "results.json").write_text(json.dumps({"device": str(DEVICE), "folds": args.folds, "runtime_s": time.time() - t0,
                                                   "summary": summ}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(s[["scorer", "AUC", "TPR_FAR1", "TPR_FAR5", "F1_Acc90", "dep_known_acc", "dep_unknown_rej", "dep_F1", "d_AUC", "se_AUC", "win_AUC"]].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
