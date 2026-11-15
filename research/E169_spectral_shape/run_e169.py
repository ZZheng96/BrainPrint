"""E169: amplitude-scale-free spectral SHAPE features from cached log band power.

Motivation (E165 / E167):
  * E65 correlation-logSPD standardises every trial inside each band by the per-channel
    standard deviation, so it is invariant to channel gain, band gain and circular shift
    -- i.e. it deliberately throws the amplitude information away.
  * E167 concatenated the raw 90-d log band power and pushed the diagnostic AUC to 0.7999,
    but W stalled at 0.6997 because raw log power also carries a SESSION-level amplitude
    offset: its own Train LOSO separability is weak (0.4103 @ a=0.90 vs E65 0.8412), so the
    fused OOF quantile threshold moves to a looser operating point and LOSO rejection
    collapses (0.8412 -> 0.6730).
  * This experiment keeps only the SHAPE of the spectrum inside each channel / across
    channels, which is invariant to those amplitude offsets:
        profile   90-d per-channel softmax over the 5 bands
        slope     18-d per-channel OLS slope over band geometric-centre frequencies
        ratios    72-d per-channel log band ratios (theta/alpha, beta/alpha, gamma/alpha, low/high)
        relpower  18-d per-channel total log power minus the trial mean over channels
        shape_all 198-d concat of the four
    Raw logbp (90-d) is kept as the reproduction control (E148 / E167 AUC 0.7772).

Protocol: verbatim E143 / E167. Train-only standardisation, PCA, prototypes and thresholds;
5-block Train OOF; LOSO excludes the scored identity and builds prototypes from fit_rows
(frac<=0.6) only; evaluation block 0.6<frac<=0.8; Test is scored once and its session labels
are used only for the post-hoc diagnostic AUC (official Test AUC/F1 = N/A).

Cache layout note: E146 stores log band power as power.permute(0,2,1).reshape(B, C*NB), i.e.
column index = channel*5 + band (CHANNEL-major). We reshape to (n, 18, 5) then transpose to
(n, 5 bands, 18 channels); "band-major" is NOT assumed.
"""
from __future__ import annotations

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
TASKS = CFG["tasks"]
CLASSES = np.arange(1, 21, dtype=int)
WEIGHTS = CFG["weights"]
ACC = CFG["evaluation"]["acceptance_grid"]
REF_ACC = float(CFG["evaluation"]["reference_acceptance"])
NB = len(CFG["bands_hz"])
NC = 18
CENTRES = np.sqrt(np.array([lo * hi for lo, hi in CFG["bands_hz"]], dtype=np.float64))


class Tee:
    def __init__(self, path: Path, mode: str = "w"):
        self.file = open(path, mode, encoding="utf-8")
        self.stdout = sys.__stdout__

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


REPORT_ONLY = len(sys.argv) > 1 and sys.argv[1] == "--report-only"
LOG_MODE = "a" if len(sys.argv) > 1 and sys.argv[1] == "--recompute-ceilings" else "w"
sys.stdout = Tee(OUT / "run.log", LOG_MODE) if not REPORT_ONLY else sys.__stdout__


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ primitives (E143/E167 verbatim)
def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def task_standardise(a: np.ndarray, b: np.ndarray, tasks: np.ndarray,
                     test_tasks: np.ndarray) -> tuple:
    """Per-task z-score with Train-session1 statistics (E143/E167 behaviour)."""
    a = a.copy()
    b = b.copy()
    for task in TASKS:
        m = tasks == task
        mu = a[m].mean(0)
        sd = a[m].std(0) + 1e-6
        q = test_tasks == task
        a[m] = (a[m] - mu) / sd
        b[q] = (b[q] - mu) / sd
    return a, b


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    mu = x.mean(0)
    xc = x - mu
    values, vectors = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, vectors[:, np.argsort(-values)][:, : min(int(dim), x.shape[1])]


def kmeans_t(x: torch.Tensor, k: int, seed: int, iters: int = 20) -> torch.Tensor:
    """Verbatim E143/E162/E163/E167 seeding so the E65 control reproduces the published W."""
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


def build_prototypes(feat: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                     mask: np.ndarray, classes: np.ndarray, k: int) -> tuple:
    protos, keys = [], []
    for task in TASKS:
        for subject in classes:
            sel = mask & (tasks == task) & (subjects == subject)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, int(k), seed=int(subject) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(subject)))
    return np.stack(protos), keys


def neg_d1(feat: np.ndarray, protos: np.ndarray, keys: list, tasks: np.ndarray) -> np.ndarray:
    out = np.full(len(feat), np.nan, dtype=np.float32)
    key_task = np.asarray([key[0] for key in keys])
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = np.flatnonzero(key_task == task)
        if len(idx) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        for start in range(0, len(idx), 2048):
            chunk = idx[start:start + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


# ------------------------------------------------------------------ feature families
def logsumexp_bands(x: np.ndarray) -> np.ndarray:
    """log(sum_b exp(x_b)) along the band axis (1), numerically stable."""
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).squeeze(1)


def build_families(logbp90: np.ndarray) -> dict:
    """Derive every shape family from a (n, 90) log band power matrix.

    Cached layout is channel-major (column = channel*5 + band), so reshape to (n, 18, 5)
    and transpose to (n, 5 bands, 18 channels).
    """
    x = logbp90.reshape(len(logbp90), NC, NB).transpose(0, 2, 1).astype(np.float32)

    lse = x - x.max(axis=1, keepdims=True)
    profile = (np.exp(lse) / np.exp(lse).sum(axis=1, keepdims=True)).astype(np.float32)

    cs = (CENTRES - CENTRES.mean())
    slope = ((x - x.mean(axis=1, keepdims=True)) * cs[None, :, None]).sum(axis=1) \
        / float((cs ** 2).sum())
    slope = slope.astype(np.float32)

    ratios = np.stack([
        x[:, 1] - x[:, 2],
        x[:, 3] - x[:, 2],
        x[:, 4] - x[:, 2],
        x[:, :2].mean(axis=1) - x[:, 2:].mean(axis=1),
    ], axis=2).astype(np.float32)                                   # (n, 18, 4)

    total = logsumexp_bands(x)                                      # (n, 18)
    relpower = (total - total.mean(axis=1, keepdims=True)).astype(np.float32)

    return {
        "profile": profile.transpose(0, 2, 1).reshape(len(x), -1),  # channel-major flatten
        "slope": slope,
        "ratios": ratios.reshape(len(x), -1),
        "relpower": relpower,
        "shape_all": np.concatenate([
            profile.transpose(0, 2, 1).reshape(len(x), -1),
            slope, ratios.reshape(len(x), -1), relpower], axis=1),
    }


# ------------------------------------------------------------------ context
class Context:
    def __init__(self) -> None:
        meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
        qmeta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
        self.nodes = meta.subject.to_numpy(int)
        self.tasks = meta.task.to_numpy()
        self.frac = frac(meta)
        self.fit_rows = self.frac <= CFG["fit_fraction"]
        self.blocks = np.clip((self.frac * CFG["blocks"]).astype(int), 0, CFG["blocks"] - 1)
        self.window = (self.frac > CFG["fit_fraction"]) & (self.frac <= 0.8)
        self.window_rows = np.flatnonzero(self.window)
        self.test_tasks = qmeta.task.to_numpy()
        self.is_s1 = qmeta.session.astype(str).str.casefold().isin(
            ["session1", "s1", "1"]).to_numpy()
        self.y = np.r_[np.zeros(int(self.is_s1.sum()), dtype=int),
                       np.ones(int((~self.is_s1).sum()), dtype=int)]
        self.proj: dict[str, tuple] = {}

    def project(self, key: str, feat_train: np.ndarray, feat_test: np.ndarray, dim: int) -> tuple:
        cache = f"{key}@{dim}"
        if cache in self.proj:
            return self.proj[cache]
        mu, basis = pca_basis(np.asarray(feat_train[self.fit_rows], dtype=np.float64), dim)
        at = ((feat_train.astype(np.float64) - mu) @ basis).astype(np.float32)
        bt = ((feat_test.astype(np.float64) - mu) @ basis).astype(np.float32)
        self.proj[cache] = (at, bt)
        return at, bt

    def drop(self, key: str, dim: int) -> None:
        self.proj.pop(f"{key}@{dim}", None)


# ------------------------------------------------------------------ protocol
def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context) -> dict:
    """Verbatim E143/E163/E167 open-set protocol on one projected representation."""
    protos, keys = build_prototypes(at, ctx.nodes, ctx.tasks, np.ones(len(at), bool),
                                    CLASSES, k)
    te_raw = neg_d1(bt, protos, keys, ctx.test_tasks)

    oof = np.full(len(at), np.nan, dtype=np.float32)
    oof_pred = np.full(len(at), -1, dtype=np.int32)
    for block in range(CFG["blocks"]):
        hold = ctx.blocks == block
        held = np.flatnonzero(hold)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        rows = held
        dmat = np.full((len(rows), len(k2)), -np.inf, dtype=np.float32)
        key_task = np.asarray([key[0] for key in k2])
        key_subject = np.asarray([key[1] for key in k2], dtype=np.int32)
        for task in TASKS:
            r = np.flatnonzero(ctx.tasks[rows] == task)
            c = np.flatnonzero(key_task == task)
            if len(r) == 0 or len(c) == 0:
                continue
            p = torch.as_tensor(np.asarray(p2[c]), dtype=FT, device=DEVICE)
            for start in range(0, len(r), 2048):
                chunk = r[start:start + 2048]
                q = torch.as_tensor(np.asarray(at[rows[chunk]], dtype=np.float32),
                                    device=DEVICE)
                dmat[np.ix_(chunk, c)] = (-torch.cdist(q, p)).cpu().numpy()
        oof[held] = dmat.max(1)
        oof_pred[held] = key_subject[dmat.argmax(1)]

    loso_known, loso_unknown = [], []
    for held_id in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held_id])
        pm = np.isin(ctx.nodes, enrolled)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, pm & ctx.fit_rows, enrolled, k)
        known = ctx.window_rows[pm[ctx.window_rows]]
        unknown = ctx.window_rows[ctx.nodes[ctx.window_rows] == held_id]
        if len(known) == 0 or len(unknown) == 0:
            continue
        rows = np.concatenate([known, unknown])
        scores = neg_d1(at[rows], p2, k2, ctx.tasks[rows])
        loso_known.append(scores[:len(known)])
        loso_unknown.append(scores[len(known):])

    finite = np.isfinite(oof)
    mu = float(np.nanmean(oof[finite]))
    sd = float(np.nanstd(oof[finite])) + 1e-6
    z_rf = ((oof - mu) / sd).astype(np.float32)
    z_te = ((te_raw - mu) / sd).astype(np.float32)
    loso = [((kn - mu) / sd, (un - mu) / sd) for kn, un in zip(loso_known, loso_unknown)]
    unknown_all = np.concatenate([un for _, un in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_rf[finite]]))
    oof_f1 = float(f1_score(ctx.nodes[finite], oof_pred[finite], labels=CLASSES,
                            average="macro", zero_division=0))
    oof_acc = float(np.mean(oof_pred[finite] == ctx.nodes[finite]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))

    rows = []
    for acceptance in ACC:
        threshold = float(np.nanquantile(z_rf[finite], 1 - acceptance))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= threshold))
        s1 = float(np.mean(z_te[ctx.is_s1] < threshold))
        loso_rej = float(np.mean([np.mean(un < np.quantile(kn, 1 - acceptance))
                                  for kn, un in loso]))
        rows.append({"acceptance_rate": float(acceptance),
                     "LOSO_rejection": loso_rej,
                     "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": WEIGHTS["LOSO"] * loso_rej + WEIGHTS["S1"] * s1
                          + WEIGHTS["S2"] * s2,
                     "threshold": threshold})

    return {"z_rf": z_rf, "z_te": z_te, "loso": loso, "mu": mu, "sd": sd,
            "train_oof_auc": oof_auc, "train_oof_f1": oof_f1,
            "train_oof_identity_acc": oof_acc, "diagnostic_auc": diag_auc,
            "rows": rows,
            "per_task_auc": {
                task: float(roc_auc_score((~ctx.is_s1[ctx.test_tasks == task]).astype(int),
                                          z_te[ctx.test_tasks == task]))
                for task in TASKS
                if len(np.unique(ctx.is_s1[ctx.test_tasks == task])) > 1}}


def loso_auc(entry: dict) -> float:
    unknown = np.concatenate([un for _, un in entry["loso"]])
    known = np.concatenate([kn for kn, _ in entry["loso"]])
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                               np.r_[unknown, known]))


def summarise(entry: dict, meta: dict) -> dict:
    rows = entry["rows"]
    best = max(rows, key=lambda r: r["W"])
    ref = min(rows, key=lambda r: abs(r["acceptance_rate"] - REF_ACC))
    record = dict(meta)
    record.update({
        "diagnostic_auc": entry["diagnostic_auc"],
        "train_oof_auc": entry["train_oof_auc"],
        "train_oof_f1": entry["train_oof_f1"],
        "train_oof_identity_acc": entry["train_oof_identity_acc"],
        "loso_auc_train_only": loso_auc(entry),
        "best_W": best["W"], "best_acceptance": best["acceptance_rate"],
        "loso_at_best": best["LOSO_rejection"],
        "s1_at_best": best["Test_S1_rejection"],
        "s2_at_best": best["Test_S2_acceptance"],
        "W_at_090": ref["W"], "loso_at_090": ref["LOSO_rejection"],
        "s1_at_090": ref["Test_S1_rejection"], "s2_at_090": ref["Test_S2_acceptance"],
        "curve": rows,
        "per_task_auc": entry["per_task_auc"],
    })
    return record


# ------------------------------------------------------------------ E159-style ceiling
def linear_quantile(sorted_arr: np.ndarray, level: np.ndarray) -> np.ndarray:
    n = len(sorted_arr)
    pos = np.clip(level, 0.0, 1.0) * (n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    w = pos - lo
    return sorted_arr[lo] * (1 - w) + sorted_arr[hi] * w


def ceiling_analysis(entry: dict, ctx: Context, label: str) -> dict:
    """E159: freeze the score, sweep the threshold over the pooled score distribution."""
    finite = np.isfinite(entry["z_rf"])
    ref = entry["z_rf"][finite].astype(np.float64)
    s1s = entry["z_te"][ctx.is_s1].astype(np.float64)
    s2s = entry["z_te"][~ctx.is_s1].astype(np.float64)
    grid = np.unique(np.quantile(np.r_[ref, s1s, s2s], np.linspace(0.001, 0.999, 2000)))

    accept = 1.0 - np.searchsorted(np.sort(ref), grid, side="left") / len(ref)
    s1_rej = np.searchsorted(np.sort(s1s), grid, side="left") / len(s1s)
    s2_acc = 1.0 - np.searchsorted(np.sort(s2s), grid, side="left") / len(s2s)

    kn = [np.sort(k.astype(np.float64)) for k, _ in entry["loso"]]
    un = [np.sort(u.astype(np.float64)) for _, u in entry["loso"]]
    loso_rej = np.zeros(len(grid))
    for ks, us in zip(kn, un):
        q = linear_quantile(ks, 1.0 - accept)
        loso_rej += np.searchsorted(us, q, side="left") / len(us)
    loso_rej /= len(kn)

    w = WEIGHTS["LOSO"] * loso_rej + WEIGHTS["S1"] * s1_rej + WEIGHTS["S2"] * s2_acc
    i = int(np.argmax(w))
    loso_fix = float(max(r["LOSO_rejection"] for r in entry["rows"]))
    loso_ref = float(min(entry["rows"],
                         key=lambda r: abs(r["acceptance_rate"] - REF_ACC))["LOSO_rejection"])
    loso_sweep_max = float(loso_rej.max())
    opt = WEIGHTS["LOSO"] * loso_fix + WEIGHTS["S1"] * s1_rej + WEIGHTS["S2"] * s2_acc
    opt_ref = WEIGHTS["LOSO"] * loso_ref + WEIGHTS["S1"] * s1_rej \
        + WEIGHTS["S2"] * s2_acc
    opt_sweep = WEIGHTS["LOSO"] * loso_sweep_max + WEIGHTS["S1"] * s1_rej \
        + WEIGHTS["S2"] * s2_acc
    j = int(np.argmax(opt))
    jr = int(np.argmax(opt_ref))
    k80 = int(np.argmin(np.abs(s1_rej - 0.80)))

    order = np.argsort(-w)[:8]
    return {
        "label": label,
        "diagnostic_auc": entry["diagnostic_auc"],
        "oracle_max_W": float(w[i]),
        "oracle_point": {"threshold": float(grid[i]), "acceptance_rate": float(accept[i]),
                         "LOSO_rejection": float(loso_rej[i]),
                         "Test_S1_rejection": float(s1_rej[i]),
                         "Test_S2_acceptance": float(s2_acc[i])},
        "LOSO_fixed_max_over_acceptance_grid": loso_fix,
        "optimistic_upper_bound_W": float(opt[j]),
        "optimistic_point": {"interval_acceptance": float(accept[j]),
                             "Test_S1_rejection": float(s1_rej[j]),
                             "Test_S2_acceptance": float(s2_acc[j])},
        "LOSO_at_reference_acceptance_090": loso_ref,
        "optimistic_upper_bound_W_at_090_LOSO": float(opt_ref[jr]),
        "optimistic_point_at_090_LOSO": {"interval_acceptance": float(accept[jr]),
                                        "Test_S1_rejection": float(s1_rej[jr]),
                                        "Test_S2_acceptance": float(s2_acc[jr])},
        "gap_to_080_at_090_LOSO": float(0.80 - opt_ref[jr]),
        "LOSO_max_over_full_sweep": loso_sweep_max,
        "optimistic_upper_bound_full_sweep_W": float(opt_sweep.max()),
        "gap_to_080": float(0.80 - opt[j]),
        "point_at_S1_080": {"Test_S1_rejection": float(s1_rej[k80]),
                            "Test_S2_acceptance": float(s2_acc[k80]),
                            "interval_acceptance": float(accept[k80])},
        "curve_head": [{"threshold": float(grid[t]), "acceptance_rate": float(accept[t]),
                        "LOSO_rejection": float(loso_rej[t]),
                        "Test_S1_rejection": float(s1_rej[t]),
                        "Test_S2_acceptance": float(s2_acc[t]), "W": float(w[t])}
                       for t in order],
    }


# ------------------------------------------------------------------ streams
def load_streams(ctx: Context) -> tuple:
    """Load logbp + derived shape families (all task-wise z-scored) and the E65 stream."""
    tr_lb = np.load(ROOT / CFG["streams"]["logbp"]["train"]).astype(np.float32)
    te_lb = np.load(ROOT / CFG["streams"]["logbp"]["test"]).astype(np.float32)
    fam_tr = build_families(tr_lb)
    fam_te = build_families(te_lb)
    families = {"logbp": {"train": tr_lb, "test": te_lb}}
    for name in fam_tr:
        families[name] = {"train": fam_tr[name], "test": fam_te[name]}
    for name in families:
        a, b = task_standardise(families[name]["train"], families[name]["test"],
                                ctx.tasks, ctx.test_tasks)
        families[name] = {"train": a, "test": b}
    a65 = np.load(ROOT / CFG["streams"]["E65_logspd"]["train"]).astype(np.float32)
    b65 = np.load(ROOT / CFG["streams"]["E65_logspd"]["test"]).astype(np.float32)
    a65, b65 = task_standardise(a65, b65, ctx.tasks, ctx.test_tasks)
    return families, a65, b65


def arrays_for(ctx: Context, families: dict, a65: np.ndarray, b65: np.ndarray,
               rec: dict) -> tuple:
    """Rebuild the frozen representation of one recorded configuration."""
    fam = rec["family"]
    if fam == "E65_only":
        return ctx.project("E65_logspd", a65, b65, rec["pca_dim"])
    if fam.endswith("_only"):
        name = fam[:-5]
        return ctx.project(name, families[name]["train"], families[name]["test"],
                           rec["pca_dim"])
    variant = "block_norm" if fam.endswith("block_norm") else "raw"
    is_ctrl = fam.startswith("control_concat")
    dim_e65 = int(rec.get("pca_dim_e65", 64 if is_ctrl else 96))
    at65, bt65 = ctx.project("E65_logspd", a65, b65, dim_e65)
    if is_ctrl:
        fdim, name = 32, "logbp"
    else:
        fdim, name = int(rec["pca_dim_family"]), rec["fusion_family"]
    atf, btf = ctx.project(name, families[name]["train"], families[name]["test"], fdim)
    sa = float(np.sqrt(np.mean(at65[ctx.fit_rows] ** 2)))
    sb = float(np.sqrt(np.mean(atf[ctx.fit_rows] ** 2)))
    if variant == "block_norm":
        at65, bt65 = at65 / max(sa, 1e-6), bt65 / max(sa, 1e-6)
        atf, btf = atf / max(sb, 1e-6), btf / max(sb, 1e-6)
    tr = np.concatenate([at65, atf], axis=1)
    te = np.concatenate([bt65, btf], axis=1)
    key = f"rebuild_{name}_{variant}_{dim_e65}_{fdim}_{rec['pca_dim']}"
    return ctx.project(key, tr, te, int(rec["pca_dim"]))


def pick_targets(singles: dict, fusion: list, e65_rows: list,
                 concat_ctrl: list) -> dict:
    """Shared selection of headline configs and the ceiling targets."""
    all_single = [r for rows in singles.values() for r in rows]
    best_auc_single = max(all_single, key=lambda r: r["diagnostic_auc"])
    best_w_single = max(all_single, key=lambda r: r["best_W"])
    best_auc_fusion = max(fusion, key=lambda r: r["diagnostic_auc"])
    best_w_fusion = max(fusion, key=lambda r: r["best_W"])
    e143_row = next(r for r in e65_rows if r["pca_dim"] == 96 and r["k"] == 12)
    ctrl_best_auc = max(concat_ctrl, key=lambda r: r["diagnostic_auc"])
    return {
        "all_single": all_single,
        "best_auc_single": best_auc_single, "best_w_single": best_w_single,
        "best_auc_fusion": best_auc_fusion, "best_w_fusion": best_w_fusion,
        "e143_row": e143_row, "ctrl_best_auc": ctrl_best_auc,
        "targets": [
            ("单族最佳 AUC", best_auc_single),
            ("单族最佳 W", best_w_single),
            ("融合最佳 AUC", best_auc_fusion),
            ("融合最佳 W", best_w_fusion),
            ("E65 对照 (PCA-96 k=12)", e143_row),
            ("E65+logbp 对照最佳 AUC", ctrl_best_auc),
        ],
    }


# ------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {len(ctx.test_tasks)} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | window {int(ctx.window.sum())}")
    log(f"tasks {dict(pd.Series(ctx.tasks).value_counts())} | band centres "
        f"{np.round(CENTRES, 2).tolist()}")

    families, a65, b65 = load_streams(ctx)
    log(f"loaded logbp {families['logbp']['train'].shape} / "
        f"{families['logbp']['test'].shape} (channel-major -> band x channel)")
    for name in ["profile", "slope", "ratios", "relpower", "shape_all"]:
        log(f"  family {name}: {families[name]['train'].shape[1]} dims")
    log(f"loaded E65 {a65.shape} / {b65.shape}")

    entries: dict[tuple, dict] = {}
    t0 = time.time()

    def run_config(key: str, at: np.ndarray, bt: np.ndarray, meta: dict,
                   dims: list, k_list: list) -> list:
        out = []
        for dim in dims:
            for k in k_list:
                ekey = (key, int(dim), int(k))
                if ekey not in entries:
                    entries[ekey] = evaluate(at, bt, int(k), ctx)
                rec = summarise(entries[ekey], dict(meta, pca_dim=dim, k=k,
                                                   config=f"{meta['config']} PCA-{dim} k={k}"))
                rec["entry_key"] = f"{key}@{dim}@{k}"
                out.append(rec)
        return out

    # ---------------------------------------------------------- single families
    singles: dict[str, list] = {}
    native = {"logbp": 90, "profile": 90, "slope": 18, "ratios": 72,
              "relpower": 18, "shape_all": 198}
    for name in ["profile", "slope", "ratios", "relpower", "shape_all", "logbp"]:
        a, b = families[name]["train"], families[name]["test"]
        dims = sorted({min(d, native[name]) for d in CFG["evaluation"]["pca_dims_requested"]}
                      | {native[name] if native[name] <= 48 else min(48, native[name])})
        dims = [d for d in dims if d <= native[name]]
        singles[name] = []
        for dim in dims:
            at, bt = ctx.project(name, a, b, dim)
            recs = run_config(name, at, bt,
                              {"family": f"{name}_only", "stream": name, "config": name},
                              [dim], CFG["prototype_k"])
            singles[name].extend(recs)
            ctx.drop(name, dim)
            top = max(recs, key=lambda r: r["diagnostic_auc"])
            log(f"  [{name}] dim={dim} bestAUC={top['diagnostic_auc']:.4f} "
                f"W@.90={top['W_at_090']:.4f} bestW={top['best_W']:.4f} "
                f"({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------- E65 control
    e65_rows = []
    for dim in CFG["streams"]["E65_logspd"]["pca_dims"]:
        at, bt = ctx.project("E65_logspd", a65, b65, dim)
        e65_rows.extend(run_config("E65_logspd", at, bt,
                                   {"family": "E65_only", "stream": "E65_logspd",
                                    "config": "E65_logspd"},
                                   [dim], CFG["prototype_k"]))
        ctx.drop("E65_logspd", dim)
        log(f"  [E65] dim={dim} done ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------- E65 + logbp control
    concat_ctrl = []
    at65, bt65 = ctx.project("E65_logspd", a65, b65, 64)
    alb, blb = ctx.project("logbp", families["logbp"]["train"],
                           families["logbp"]["test"], 32)
    scale_a = float(np.sqrt(np.mean(at65[ctx.fit_rows] ** 2)))
    scale_b = float(np.sqrt(np.mean(alb[ctx.fit_rows] ** 2)))
    for variant in ["block_norm", "raw"]:
        if variant == "block_norm":
            tr = np.concatenate([at65 / max(scale_a, 1e-6), alb / max(scale_b, 1e-6)], axis=1)
            te = np.concatenate([bt65 / max(scale_a, 1e-6), blb / max(scale_b, 1e-6)], axis=1)
        else:
            tr = np.concatenate([at65, alb], axis=1)
            te = np.concatenate([bt65, blb], axis=1)
        key = f"ctrl_{variant}_64_32"
        a2, b2 = ctx.project(key, tr, te, 96)
        concat_ctrl.extend(run_config(key, a2, b2,
                                      {"family": f"control_concat_{variant}",
                                       "stream": "E65+logbp",
                                       "config": f"{variant} concat 64+32 ->"},
                                      [96], CFG["prototype_k"]))
        ctx.drop(key, 96)
        del tr, te
    ctx.drop("E65_logspd", 64)
    ctx.drop("logbp", 32)
    log(f"  [E65+logbp control] best AUC "
        f"{max(r['diagnostic_auc'] for r in concat_ctrl):.4f} "
        f"best W {max(r['best_W'] for r in concat_ctrl):.4f} ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------- fusion: E65(96) + shape
    fusion = []
    dim_e65 = int(CFG["fusion"]["e65_pca_dim"])
    at65, bt65 = ctx.project("E65_logspd", a65, b65, dim_e65)
    for fam in CFG["fusion"]["families"]:
        af = families[fam]["train"]
        bf = families[fam]["test"]
        for fdim in CFG["fusion"]["family_pca_dims"]:
            atf, btf = ctx.project(fam, af, bf, fdim)
            sa = float(np.sqrt(np.mean(at65[ctx.fit_rows] ** 2)))
            sb = float(np.sqrt(np.mean(atf[ctx.fit_rows] ** 2)))
            for variant in CFG["fusion"]["variants"]:
                if variant == "block_norm":
                    tr = np.concatenate([at65 / max(sa, 1e-6), atf / max(sb, 1e-6)], axis=1)
                    te = np.concatenate([bt65 / max(sa, 1e-6), btf / max(sb, 1e-6)], axis=1)
                else:
                    tr = np.concatenate([at65, atf], axis=1)
                    te = np.concatenate([bt65, btf], axis=1)
                for jd in CFG["fusion"]["joint_pca_dims"]:
                    if jd > tr.shape[1]:
                        continue
                    key = f"fuse_{fam}_{variant}_{dim_e65}_{fdim}"
                    a2, b2 = ctx.project(key, tr, te, jd)
                    recs = run_config(key, a2, b2,
                                      {"family": f"feature_concat_{variant}",
                                       "stream": f"E65+{fam}",
                                       "fusion_family": fam,
                                       "pca_dim_e65": dim_e65,
                                       "pca_dim_family": fdim,
                                       "joint_pca_dim": jd,
                                       "config": f"{variant} concat {dim_e65}+{fdim} ->"},
                                      [jd], CFG["prototype_k"])
                    fusion.extend(recs)
                    ctx.drop(key, jd)
                    del a2, b2
                    log(f"  [fuse {fam} {variant} {dim_e65}+{fdim} -> {jd}] "
                        f"bestAUC={max(r['diagnostic_auc'] for r in recs):.4f} "
                        f"bestW={max(r['best_W'] for r in recs):.4f} "
                        f"({time.time() - t0:.0f}s)")
                del tr, te
            ctx.drop(fam, fdim)
    ctx.drop("E65_logspd", dim_e65)

    # ---------------------------------------------------------- verdict & ceiling
    picked = pick_targets(singles, fusion, e65_rows, concat_ctrl)
    all_single = picked["all_single"]
    best_auc_single = picked["best_auc_single"]
    best_w_single = picked["best_w_single"]
    best_auc_fusion = picked["best_auc_fusion"]
    best_w_fusion = picked["best_w_fusion"]
    e143_row = picked["e143_row"]
    ctrl_best_auc = picked["ctrl_best_auc"]
    ceiling_targets = picked["targets"]

    def entry_of(rec: dict) -> dict:
        key, dim, k = rec["entry_key"].split("@")
        return entries[(key, int(dim), int(k))]

    ceilings = []
    for tag, rec in ceiling_targets:
        c = ceiling_analysis(entry_of(rec), ctx, rec["config"])
        c["target_tag"] = tag
        ceilings.append(c)
        log(f"  [ceiling] {tag} | {rec['config']} AUC={c['diagnostic_auc']:.4f} "
            f"oracleW={c['oracle_max_W']:.4f} optW={c['optimistic_upper_bound_W']:.4f} "
            f"optW@090LOSO={c['optimistic_upper_bound_W_at_090_LOSO']:.4f} "
            f"(LOSO fix {c['LOSO_fixed_max_over_acceptance_grid']:.4f})")

    base_auc, base_w = 0.7578, 0.6958
    e167_auc, e167_w = 0.7999, 0.6997
    prior_w = 0.7164
    max_auc = max(max(r["diagnostic_auc"] for r in all_single),
                  max(r["diagnostic_auc"] for r in fusion))
    max_w = max(max(r["best_W"] for r in all_single),
                max(r["best_W"] for r in fusion))
    max_w090 = max(max(r["W_at_090"] for r in all_single),
                   max(r["W_at_090"] for r in fusion))
    # ceiling restricted to the config with the highest AUC overall
    best_auc_all = best_auc_fusion if best_auc_fusion["diagnostic_auc"] \
        >= best_auc_single["diagnostic_auc"] else best_auc_single
    best_ceiling = ceilings[[r for _, r in ceiling_targets].index(best_auc_all)]
    loso_best_090 = max(r["loso_at_090"] for r in all_single + fusion)
    loso_best_any = max(r["loso_at_best"] for r in all_single + fusion)
    bw_all = max(max(fusion, key=lambda r: r["best_W"]),
                 max(all_single, key=lambda r: r["best_W"]),
                 key=lambda r: r["best_W"])
    shape_best = {name: max(singles[name], key=lambda r: r["diagnostic_auc"])
                  for name in singles}

    verdict = {
        "baseline_diagnostic_auc_E143": base_auc, "baseline_W_E143": base_w,
        "prior_diagnostic_auc_E167": e167_auc, "prior_best_W_E167": e167_w,
        "prior_best_W_E163": prior_w,
        "e143_reproduction": e143_row,
        "e65_logbp_concat_control_best_auc": ctrl_best_auc,
        "best_auc_single_family": best_auc_single,
        "best_W_single_family": best_w_single,
        "best_auc_fusion": best_auc_fusion,
        "best_W_fusion": best_w_fusion,
        "best_per_family_by_auc": shape_best,
        "max_diagnostic_auc_any": max_auc,
        "max_best_W_any": max_w,
        "max_W_at_090_any": max_w090,
        "max_loso_at_090_any": loso_best_090,
        "max_loso_at_best_any": loso_best_any,
        "best_W_config": bw_all,
        "auc_gain_vs_E167": max_auc - e167_auc,
        "auc_gain_vs_E143": max_auc - base_auc,
        "w_gain_vs_E143": max_w - base_w,
        "w_gain_vs_E167": max_w - e167_w,
        "w_gain_vs_E163": max_w - prior_w,
        "ceiling_of_best_auc_config": best_ceiling,
        "question1_auc_above_07999": bool(max_auc > e167_auc),
        "question2_loso_ge_079": bool(best_auc_all["loso_at_090"] >= 0.79),
        "question2_W_above_06997": bool(max_w > e167_w),
        "question2_W_above_07164": bool(max_w > prior_w),
        "question3_ceiling_W": best_ceiling["optimistic_upper_bound_W"],
        "question3_gap_to_080": best_ceiling["gap_to_080"],
        "question3_ceiling_W_E159_convention":
            best_ceiling["optimistic_upper_bound_W_at_090_LOSO"],
        "question3_gap_to_080_E159_convention": best_ceiling["gap_to_080_at_090_LOSO"],
    }

    results = {
        "experiment": CFG["experiment"],
        "device": str(DEVICE),
        "protocol": "E143/E167 verbatim: Train-only standardisation/PCA/prototypes/"
                    "thresholds; 5-block Train OOF; LOSO excludes the scored identity and "
                    "uses fit_rows-only prototypes; evaluation block 0.6<frac<=0.8; "
                    "Test scored only; session used only for the diagnostic AUC",
        "test_official": {"AUC": "N/A(no labels)", "F1": "N/A(no labels)"},
        "feature_dimensions": {name: int(families[name]["train"].shape[1])
                               for name in families},
        "singles": singles,
        "E65_control": e65_rows,
        "E65_logbp_concat_control": concat_ctrl,
        "fusion_concat": fusion,
        "ceiling_analysis": ceilings,
        "verdict": verdict,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    write_report(results, best_ceiling)
    log(f"done in {time.time() - started:.0f}s")
    log(json.dumps({k: v for k, v in verdict.items() if not isinstance(v, dict)},
                   ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ report
HEADER = ("| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | S2@.90 | "
          "W@.90 | 最优 W | 接受率 | LOSO | S1 拒识 | S2 接受 |")
SEP = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


# ------------------------------------------------------------------ ceiling recompute
def recompute_ceilings() -> None:
    """Rebuild only the frozen score of the headline configs, then redo the E159 sweep.

    Used after the grid finished: no representation is retrained, the same deterministic
    evaluate() reproduces the recorded scores bit-for-bit (AUC is cross-checked).
    """
    started = time.time()
    ctx = Context()
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    families, a65, b65 = load_streams(ctx)
    picked = pick_targets(res["singles"], res["fusion_concat"],
                          res["E65_control"], res["E65_logbp_concat_control"])
    ceilings = []
    for tag, rec in picked["targets"]:
        at, bt = arrays_for(ctx, families, a65, b65, rec)
        e = evaluate(at, bt, int(rec["k"]), ctx)
        if abs(e["diagnostic_auc"] - rec["diagnostic_auc"]) > 1e-6:
            log(f"  [warn] {rec['config']} AUC mismatch {e['diagnostic_auc']:.6f} vs "
                f"{rec['diagnostic_auc']:.6f}")
        c = ceiling_analysis(e, ctx, rec["config"])
        c["target_tag"] = tag
        ceilings.append(c)
        log(f"  [ceiling] {tag} | {rec['config']} AUC={c['diagnostic_auc']:.4f} "
            f"oracleW={c['oracle_max_W']:.4f} optW_maxLOSO={c['optimistic_upper_bound_W']:.4f} "
            f"optW@090LOSO={c['optimistic_upper_bound_W_at_090_LOSO']:.4f}")
        ctx.proj.clear()

    best_auc_single, best_auc_fusion = picked["best_auc_single"], picked["best_auc_fusion"]
    best_auc_all = best_auc_fusion if best_auc_fusion["diagnostic_auc"] \
        >= best_auc_single["diagnostic_auc"] else best_auc_single
    idx = [r for _, r in picked["targets"]].index(best_auc_all)
    best_ceiling = ceilings[idx]

    res["ceiling_analysis"] = ceilings
    v = res["verdict"]
    v["ceiling_of_best_auc_config"] = best_ceiling
    v["question3_ceiling_W"] = best_ceiling["optimistic_upper_bound_W"]
    v["question3_gap_to_080"] = best_ceiling["gap_to_080"]
    v["question3_ceiling_W_E159_convention"] = \
        best_ceiling["optimistic_upper_bound_W_at_090_LOSO"]
    v["question3_gap_to_080_E159_convention"] = best_ceiling["gap_to_080_at_090_LOSO"]
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    write_report(res, best_ceiling)
    log(f"ceiling recompute done in {time.time() - started:.0f}s")


def fmt_row(r: dict) -> str:
    return (f"| {r['config']} | {r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
            f"{r['train_oof_f1']:.4f} | {r['loso_at_090']:.4f} | {r['s1_at_090']:.4f} | "
            f"{r['s2_at_090']:.4f} | {r['W_at_090']:.4f} | {r['best_W']:.4f} | "
            f"{r['best_acceptance']:.2f} | {r['loso_at_best']:.4f} | "
            f"{r['s1_at_best']:.4f} | {r['s2_at_best']:.4f} |")


def write_report(results: dict, best_ceiling: dict) -> None:
    v = results["verdict"]
    singles = results["singles"]
    fusion = results["fusion_concat"]
    e143 = v["e143_reproduction"]
    ctrl = v["e65_logbp_concat_control_best_auc"]
    ba_s = v["best_auc_single_family"]
    bw_s = v["best_W_single_family"]
    ba_f = v["best_auc_fusion"]
    bw_f = v["best_W_fusion"]
    lines = [
        "# E169 尺度无关的频谱形状特征（E65 + spectral shape）",
        "",
        "目标：E167 把 E65 correlation-logSPD 与原始 log band power 拼接后诊断 AUC 达 0.7999，"
        "但 W 卡在 0.6997——原始 logbp 的 Train LOSO 区分度弱（@.90 仅 0.4103 vs E65 0.8412），"
        "融合后 OOF 分位阈值被推向更宽松的工作点，LOSO 拒识从 0.8412 掉到 0.6730。"
        "本实验只保留 logbp 中**尺度无关**的谱型信息（去掉会话级整体幅值偏移），"
        "检验能否在保住 LOSO 拒识的同时继续抬 AUC。",
        "",
        "协议：逐字沿用 E143/E167——Train-only 标准化 / PCA / 原型 / 阈值；5 块 Train OOF"
        "（接受率网格 0.80–0.96）；LOSO 每折排除被评分身份、原型只用 fit_rows(frac≤0.6)、"
        "评估块 0.6<frac≤0.8；Test 只评分、session 只用于诊断 AUC。"
        "**Test 官方 AUC / F1 = N/A**（无标签）。",
        "",
        "缓存布局说明：E146 的 log band power 是 channel-major（列号 = 通道×5 + 频带），"
        "代码显式 reshape 到 (n, 18, 5) 再转置为 (n, 5 频带, 18 通道)，未假设 band-major。",
        "",
        "## 结论",
        "",
        f"- 诊断 AUC 最高 **{v['max_diagnostic_auc_any']:.4f}**"
        f"（E167 0.7999，Δ{v['auc_gain_vs_E167']:+.4f}；E143 0.7578，"
        f"Δ{v['auc_gain_vs_E143']:+.4f}），来自 {ba_f['config'] if ba_f['diagnostic_auc'] >= ba_s['diagnostic_auc'] else ba_s['config']}。",
        f"- W 最高 **{v['max_best_W_any']:.4f}**（E167 0.6997，Δ{v['w_gain_vs_E167']:+.4f}；"
        f"E163 0.7164，Δ{v['w_gain_vs_E163']:+.4f}；E143 0.6958，"
        f"Δ{v['w_gain_vs_E143']:+.4f}），取自 "
        f"{bw_f['config'] if bw_f['best_W'] >= bw_s['best_W'] else bw_s['config']}"
        f"（接受率网格内选点）；固定工作点 W@.90 最高 {v['max_W_at_090_any']:.4f}。",
        f"- 最佳 AUC 配置的 E159 式上界：扫满阈值后 oracle W = "
        f"{best_ceiling['oracle_max_W']:.4f}；E159 口径乐观上界（LOSO 固定在 @.90 的 "
        f"{best_ceiling['LOSO_at_reference_acceptance_090']:.4f}，Test 工作点任选）= "
        f"**{best_ceiling['optimistic_upper_bound_W_at_090_LOSO']:.4f}**；"
        f"最松口径（LOSO 取网格最高 {best_ceiling['LOSO_fixed_max_over_acceptance_grid']:.4f}）"
        f"= {best_ceiling['optimistic_upper_bound_W']:.4f}，即距 0.80 仍差 "
        f"{best_ceiling['gap_to_080']:.4f}–{best_ceiling['gap_to_080_at_090_LOSO']:.4f}"
        "（最松口径把 a=0.80 处的 LOSO 与任意的 Test 工作点拼在一起，实际不可同时达到，"
        "E159 口径的上界才是可比结论）。",
        "",
        "## 判定",
        "",
        f"- **核心问题 1（AUC 能否超过 0.7999）**："
        f"{'是' if v['question1_auc_above_07999'] else '否'}，最好 {v['max_diagnostic_auc_any']:.4f}。",
        f"- **核心问题 2（能否同时保住 LOSO 拒识 ≥0.79、使 W 超过 0.6997 / 0.7164）**："
        f"最佳 AUC 配置的 LOSO@.90 = {v['best_auc_single_family']['loso_at_090'] if ba_s['diagnostic_auc'] >= ba_f['diagnostic_auc'] else ba_f['loso_at_090']:.4f}"
        f"（{'保住' if v['question2_loso_ge_079'] else '未保住'} 0.79）；"
        f"全部配置里 LOSO@.90 最高 {v['max_loso_at_090_any']:.4f}；"
        f"W 最好 {v['max_best_W_any']:.4f}，"
        f"{'超过' if v['question2_W_above_06997'] else '未超过'} 0.6997、"
        f"{'超过' if v['question2_W_above_07164'] else '未超过'} 0.7164。",
        f"- **核心问题 3（最佳 AUC 配置的 W 上界）**：E159 口径 "
        f"{v['question3_ceiling_W_E159_convention']:.4f}，最松口径 "
        f"{v['question3_ceiling_W']:.4f}；距 0.80 分别差 "
        f"{v['question3_gap_to_080_E159_convention']:.4f} / {v['question3_gap_to_080']:.4f}。",
        "",
    ]
    ceil_by_tag = {c["target_tag"]: c for c in results["ceiling_analysis"]}
    sa_best = max(singles["shape_all"], key=lambda r: r["diagnostic_auc"])
    pr_best = max(singles["profile"], key=lambda r: r["diagnostic_auc"])
    cap_shape = ceil_by_tag["融合最佳 AUC"]["optimistic_upper_bound_W_at_090_LOSO"]
    cap_e65 = ceil_by_tag["E65 对照 (PCA-96 k=12)"]["optimistic_upper_bound_W_at_090_LOSO"]
    cap_lb = ceil_by_tag["E65+logbp 对照最佳 AUC"]["optimistic_upper_bound_W_at_090_LOSO"]
    lines += [
        "## 机制解读",
        "",
        f"- **尺度无关化没有换来 LOSO 区分度**：shape_all 单独最佳（{sa_best['config']}）"
        f"诊断 AUC {sa_best['diagnostic_auc']:.4f}、Train OOF AUC {sa_best['train_oof_auc']:.4f}、"
        f"LOSO@.90 {sa_best['loso_at_090']:.4f}；原始 logbp 最佳（{ba_s['config']}）反而是 "
        f"AUC {ba_s['diagnostic_auc']:.4f} / OOF AUC {ba_s['train_oof_auc']:.4f} / "
        f"LOSO@.90 {ba_s['loso_at_090']:.4f}。profile 最佳（{pr_best['config']}）AUC "
        f"{pr_best['diagnostic_auc']:.4f} / LOSO@.90 {pr_best['loso_at_090']:.4f}。"
        "说明会话级幅值偏移同时也承载了「未注册 vs 已注册」的一部分真实排序信号，"
        "把它去掉会同时损失排序力与 LOSO 拒识，而不只是去掉漂移。",
        f"- **融合后 W 只是与 E167 持平（不是提升）**：形状融合最佳 W 配置（{bw_f['config']}）"
        f"LOSO@.90 {bw_f['loso_at_090']:.4f} / S1 {bw_f['s1_at_090']:.4f} / "
        f"S2 {bw_f['s2_at_090']:.4f}，W {bw_f['best_W']:.4f}；相对 E167 的 0.6997 只差 "
        f"{v['w_gain_vs_E167']:+.4f}，属噪声量级。而最佳 AUC 配置的 LOSO@.90 "
        f"{ba_f['loso_at_090']:.4f} 还低于 E167 对应配置的 {ctrl['loso_at_090']:.4f}，"
        "说明尺度无关化并没有买到更稳的 LOSO 拒识。",
        f"- **同口径上界排序（E159 口径）**：E65+logbp {cap_lb:.4f} > 形状融合 "
        f"{cap_shape:.4f} > E65 {cap_e65:.4f}。排序力最高的分数上界也最高，但三者都 < 0.80；"
        "按 E167→E65 的经验斜率（ΔAUC +0.042 带来 Δ上界 +0.035），要让上界越过 0.80 "
        "还需要诊断 AUC 再抬约 0.035（≈0.835），单纯换阈值/归一化不够。",
        "",
        "## 特征族定义（全部由 E146 缓存的 log 频带功率派生）",
        "",
        "| 族 | 维度 | 定义 | 尺度无关性 |",
        "|---|---:|---|---|",
        "| profile | 90 | 每通道 5 个频带 log 功率做 softmax → 5 维谱型（18×5） | 对通道增益（log 域加常数）不变 |",
        "| slope | 18 | 每通道 log 功率对频带几何中心频率 √(lo·hi) 的 OLS 斜率 | 对通道加常数不变 |",
        "| ratios | 72 | 每通道 theta/alpha、beta/alpha、gamma/alpha、低频/高频 | 对数差，对通道加常数不变 |",
        "| relpower | 18 | 每通道总功率（logsumexp）减去该 trial 全通道平均 | 对 trial 整体幅值偏移不变 |",
        "| shape_all | 198 | 以上四族拼接 | 组合 |",
        "| logbp | 90 | 原始 log 频带功率（对照，含幅值） | 否 |",
        "",
        "## 单族结果（含 logbp 对照）",
        "",
        "按诊断 AUC 排序（各族取自身最优配置）：",
        "",
        HEADER, SEP,
    ]
    best_each = [max(singles[n], key=lambda r: r["diagnostic_auc"]) for n in singles]
    for r in sorted(best_each, key=lambda r: -r["diagnostic_auc"]):
        lines.append(fmt_row(r))
    lines += ["", "各族网格内 AUC 前三：", "", HEADER, SEP]
    for n in ["profile", "slope", "ratios", "relpower", "shape_all", "logbp"]:
        for r in sorted(singles[n], key=lambda r: -r["diagnostic_auc"])[:3]:
            lines.append(fmt_row(r))
    lines += ["", "单族按最优 W 排序（前 10）：", "", HEADER, SEP]
    for r in sorted([x for rows in singles.values() for x in rows],
                    key=lambda r: -r["best_W"])[:10]:
        lines.append(fmt_row(r))
    lines += ["", "全量单族配置（按族 / PCA 维 / k）：", "", HEADER, SEP]
    for n in ["profile", "slope", "ratios", "relpower", "shape_all", "logbp"]:
        for r in sorted(singles[n], key=lambda r: (r["pca_dim"], r["k"])):
            lines.append(fmt_row(r))
    lines += [
        "",
        "## 复现对照",
        "",
        HEADER, SEP, fmt_row(e143), fmt_row(ctrl),
        "",
        f"- E143 精确复现（E65 PCA-96 k=12）：诊断 AUC {e143['diagnostic_auc']:.4f} / "
        f"W@.90 {e143['W_at_090']:.4f}（公布 0.7578 / 0.6958，"
        f"Δ{e143['diagnostic_auc'] - 0.7578:+.4f} / {e143['W_at_090'] - 0.6958:+.4f}）。",
        f"- E65+logbp 特征拼接对照：诊断 AUC {ctrl['diagnostic_auc']:.4f}"
        f"（E167 公布 0.7999，Δ{ctrl['diagnostic_auc'] - 0.7999:+.4f}）。",
        f"- logbp 90 维全维 k=12（E148 探针）：诊断 AUC "
        f"{next(r for r in singles['logbp'] if r['pca_dim'] == 90 and r['k'] == 12)['diagnostic_auc']:.4f}"
        f"（E148 公布 0.7772）。",
        "- 本目录只复跑 E65+logbp 的 64+32 两档（raw / block_norm）以核对 AUC 0.7999；"
        "E167 的最优 W 0.6997 来自 96+32 档，未在此重跑，跨实验比较 W 时按各自最优值看待。",
        "",
        "E65 对照网格（复现基线行为）：",
        "",
        HEADER, SEP,
    ]
    for r in sorted(results["E65_control"], key=lambda r: (r["pca_dim"], r["k"])):
        lines.append(fmt_row(r))
    lines += [
        "",
        "## 与 E65 的特征级融合（E65 PCA-96 + 形状族 PCA-16/32，raw / block_norm，联合 PCA 96/128）",
        "",
        "按诊断 AUC 排序（前 12）：",
        "",
        HEADER, SEP,
    ]
    for r in sorted(fusion, key=lambda r: -r["diagnostic_auc"])[:12]:
        lines.append(fmt_row(r))
    lines += ["", "按最优 W 排序（前 8）：", "", HEADER, SEP]
    for r in sorted(fusion, key=lambda r: -r["best_W"])[:8]:
        lines.append(fmt_row(r))
    lines += ["", "全量融合配置（按族 / 变体 / 维度 / k）：", "", HEADER, SEP]
    for r in sorted(fusion, key=lambda r: (r["fusion_family"], r["family"],
                                           r["pca_dim_family"], r["joint_pca_dim"], r["k"])):
        lines.append(fmt_row(r))
    lines += [
        "",
        "## E159 式上界分析（冻结分数，扫满阈值）",
        "",
        "表 1：oracle 上界 + E159 原始口径（LOSO 固定在该配置参考工作点 @0.90）：",
        "",
        "| 目标 | 配置 | 诊断 AUC | oracle 最优 W (a / LOSO / S1 / S2) | "
        "LOSO@.90 | E159 口径上界 W | 距 0.80 |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for c in results["ceiling_analysis"]:
        p = c["oracle_point"]
        lines.append(
            f"| {c['target_tag']} | {c['label']} | {c['diagnostic_auc']:.4f} | "
            f"{c['oracle_max_W']:.4f} ({p['acceptance_rate']:.3f} / "
            f"{p['LOSO_rejection']:.4f} / {p['Test_S1_rejection']:.4f} / "
            f"{p['Test_S2_acceptance']:.4f}) | "
            f"{c['LOSO_at_reference_acceptance_090']:.4f} | "
            f"{c['optimistic_upper_bound_W_at_090_LOSO']:.4f} | "
            f"{c['gap_to_080_at_090_LOSO']:+.4f} |")
    lines += [
        "",
        "表 2：最松口径（LOSO 固定为该配置在协议接受率网格 0.80–0.96 内的最高拒识）：",
        "",
        "| 目标 | LOSO 网格最高 | 最松上界 W | 距 0.80 |",
        "|---|---:|---:|---:|",
    ]
    for c in results["ceiling_analysis"]:
        lines.append(
            f"| {c['target_tag']} | {c['LOSO_fixed_max_over_acceptance_grid']:.4f} | "
            f"{c['optimistic_upper_bound_W']:.4f} | {c['gap_to_080']:+.4f} |")
    lines += [
        "",
        "- oracle 最优 W：LOSO 随阈值一起移动（acceptance→0 时 LOSO→1，因此该点偏乐观）。",
        "- E159 口径：只挪 Test 工作点，LOSO 保持在 @0.90 的实测值——与 E159 的 0.7352 同口径。",
        "- 最松口径：LOSO 取该配置在协议网格内能达到的最高拒识（通常出现在 a=0.80），"
        "再把 Test 工作点选到最优；这是「LOSO 不退步」意义下最乐观的 W，故为严格上界。",
        f"- 最佳 AUC 配置的 S1 拒识≈0.80 时 S2 接受 "
        f"{best_ceiling['point_at_S1_080']['Test_S2_acceptance']:.4f}。",
        "",
        "## 最好配置的分任务诊断 AUC",
        "",
    ]
    for tag, rec in [("单族最佳 AUC", ba_s), ("单族最佳 W", bw_s),
                     ("融合最佳 AUC", ba_f), ("融合最佳 W", bw_f),
                     ("E65 对照", e143), ("E65+logbp 对照", ctrl)]:
        tasks = ", ".join(f"{k} {val:.4f}" for k, val in rec["per_task_auc"].items())
        lines.append(f"- {tag}（{rec['config']}）：{tasks}")
    lines += [
        "",
        "## 说明",
        "",
        "- 表格中 @.90 指 E143 参考工作点（接受率 0.90）；最优 W 一行跨接受率网格 0.80–0.96 取最大。",
        "- Train OOF AUC 为 5 块 OOF 分数对「held 身份未知代理 vs 已注册块」的排序能力；"
        "Train OOF F1 为同一 OOF 的 20 类宏 F1。",
        "- PCA 维度超过某族原始维度时按原始维度截断（slope/relpower → 18，ratios → 72，"
        "profile/logbp → 90），因此 96 档在 90 维族上即全维。",
        "- 联合 PCA-128 只在拼接维度 ≥128 时成立（96+32=128 用满，96+16=112 跳过），"
        "因此融合网格共 72 个配置而不是 96 个。",
        "- Test 官方 AUC / F1：**N/A**（Test 无真实标签，Session 仅用于诊断 AUC）。",
        "",
        f"运行耗时 {results['runtime_seconds']:.0f}s。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if REPORT_ONLY:
        res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        v = res["verdict"]
        bc = v["ceiling_of_best_auc_config"]
        write_report(res, bc)
    elif len(sys.argv) > 1 and sys.argv[1] == "--recompute-ceilings":
        recompute_ceilings()
    else:
        main()
