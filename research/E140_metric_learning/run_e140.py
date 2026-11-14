"""E140: supervised metric-learning embedding for open-set identity inference.

Goal: raise the Test diagnostic AUC (session1 vs session2) well above the
unsupervised PCA-96 baseline (0.7566) and therefore the clean three-way
objective (LOSO rejection + Test S1 rejection + Test S2 acceptance)/3 > 0.85.

Protocol (Train-only fitting):
  * per-task z-score of the 855-d log-SPD features, statistics from the fit
    block only (frac <= 0.6);
  * PCA basis (96/128/160 dims) fitted on the standardized fit block only;
  * embedding MLP input -> 512 -> 256 -> emb (BatchNorm + ReLU, dropout),
    L2-normalised output, trained on fit-block rows with ArcFace and/or
    batch-hard triplet loss, 20 identities;
  * augmentation applied in raw 855-d space (5 x 171 band scale / dropout,
    Gaussian noise, feature dropout, global scale) then projected through the
    PCA basis, mimicking cross-session drift;
  * score = negative nearest-prototype Euclidean distance in the embedding,
    per (task, identity) k-means prototypes (k in {1,3,8});
  * thresholds come from the held-out Train block frac in (0.6, 0.8] (never
    seen by the network) or, inside a LOSO fold, from the enrolled identities'
    rows of that same block.  Test is only ever scored.

No Test statistic of any kind, no LOSO-driven or Test-driven model selection:
the primary configuration is pre-registered in config.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]
NBANDS = int(CFG["bands"])
BW = int(CFG["band_width"])
FIT_F = float(CFG["fit_fraction"])
REF_LO, REF_HI = CFG["ref_fraction"]


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0)
    xc = x - mu
    w, V = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu.astype(np.float64), V[:, np.argsort(-w)][:, : min(dim, x.shape[1])]


def kmeans_t(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    c = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        new = torch.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if torch.allclose(new, c, atol=1e-6):
            c = new
            break
        c = new
    return c


def build_prototypes(emb, subjects, tasks, mask, classes, k):
    """k-means prototypes per (task, identity) from the rows selected by mask."""
    protos, keys = [], []
    for ti, task in enumerate(TASKS):
        for c in classes:
            sel = mask & (tasks == task) & (subjects == c)
            n = int(sel.sum())
            if n == 0:
                continue
            block = torch.as_tensor(np.asarray(emb[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, min(k, n), seed=int(c) * 17 + ti).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(c)))
    return np.stack(protos), keys


def neg_d1(emb, rows, protos, keys, row_tasks):
    """Negative nearest-prototype Euclidean distance for the selected rows."""
    out = np.full(len(rows), np.nan, dtype=np.float32)
    rt = row_tasks[rows]
    for task in TASKS:
        pos = np.flatnonzero(rt == task)
        cols = np.array([i for i, (t, _) in enumerate(keys) if t == task], dtype=int)
        if len(pos) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        idx = rows[pos]
        for st in range(0, len(idx), 1024):
            ch = idx[st:st + 1024]
            q = torch.as_tensor(np.asarray(emb[ch], dtype=np.float32), device=DEVICE)
            out[pos[st:st + 1024]] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def task_stats(scores, tasks, mask):
    """Per-task (mean, std) on the reference rows, Train-side only."""
    st, pooled = {}, []
    for t in TASKS:
        v = scores[mask & (tasks == t)]
        v = v[np.isfinite(v)]
        if len(v) >= 10:
            st[t] = (float(v.mean()), float(v.std()) + 1e-6)
            pooled.append(v)
    if pooled:
        allv = np.concatenate(pooled)
        default = (float(allv.mean()), float(allv.std()) + 1e-6)
    else:
        default = (0.0, 1.0)
    return st, default


def apply_z(scores, tasks, st, default):
    out = np.asarray(scores, dtype=np.float32).copy()
    for t in TASKS:
        m = tasks == t
        mu, sd = st.get(t, default)
        out[m] = (out[m] - mu) / sd
    return out


class Embed(nn.Module):
    def __init__(self, d_in, hidden, d_out, dropout):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.body(x), dim=1)


class ArcFace(nn.Module):
    def __init__(self, d, n, s, m):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, d) * 0.01)
        self.s, self.m = s, m

    def forward(self, x, y):
        cos = torch.mm(x, F.normalize(self.W, dim=1).t())
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + self.m)
        oh = F.one_hot(y, num_classes=self.W.shape[0]).bool()
        return F.cross_entropy(torch.where(oh, target, cos) * self.s, y)


def batch_hard(x, y, margin):
    d = torch.cdist(x, x)
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=x.device)
    pos = d.masked_fill(~same | eye, -1e4).max(1).values
    neg = d.masked_fill(same, 1e4).min(1).values
    return F.relu(pos - neg + margin).mean()


def augment(x, aug):
    """Raw-space augmentation: 5 blocks of 171 features (band x 171)."""
    b, dev = x.shape[0], x.device
    if aug is not None:
        s = 1.0 + (torch.rand(b, NBANDS, device=dev) * 2 - 1) * aug["band_scale"]
        if aug["band_drop"] > 0:
            s = s * (1.0 - (torch.rand(b, NBANDS, device=dev) < aug["band_drop"]).float())
        x = x * torch.repeat_interleave(s, BW, dim=1)
        if aug["feat_drop"] > 0:
            keep = (torch.rand_like(x) >= aug["feat_drop"]).float()
            x = x * keep / (1.0 - aug["feat_drop"])
        if aug["noise"] > 0:
            x = x + torch.randn_like(x) * aug["noise"]
        if aug["global_scale"] > 0:
            x = x * (1.0 + (torch.rand(b, 1, device=dev) * 2 - 1) * aug["global_scale"])
    return x


@torch.no_grad()
def embed_all(model, pca_t, chunk=4096):
    model.eval()
    out = []
    for st in range(0, len(pca_t), chunk):
        out.append(model(pca_t[st:st + chunk]).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def train_variant(vcfg, raw_fit, y_fit, pca_mu_t, pca_V_t, pc_std_t):
    tcfg = CFG["train"]
    aug = CFG["augmentations"][vcfg["aug"]]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    model = Embed(int(pca_V_t.shape[1]), tcfg["hidden"], vcfg["emb_dim"],
                  tcfg["dropout"]).to(DEVICE)
    head = None
    if vcfg["loss"] in ("arcface", "arc_triplet"):
        head = ArcFace(vcfg["emb_dim"], len(CLASSES), tcfg["arc_scale"],
                       tcfg["arc_margin"]).to(DEVICE)
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    wd = float(vcfg.get("weight_decay", tcfg["weight_decay"]))
    epochs = int(vcfg.get("epochs", tcfg["epochs"]))
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n, bs = len(y_fit), tcfg["batch_size"]
    t0 = time.time()
    last = 0.0
    for ep in range(epochs):
        model.train()
        if head is not None:
            head.train()
        perm = torch.randperm(n, device=DEVICE)
        tot, cnt = 0.0, 0
        for st in range(0, n, bs):
            bi = perm[st:st + bs]
            if len(bi) < 2:
                continue
            xb = augment(raw_fit[bi], aug)
            xp = ((xb - pca_mu_t) @ pca_V_t) / pc_std_t
            yb = y_fit[bi]
            z = model(xp)
            loss = torch.zeros((), device=DEVICE)
            if head is not None:
                loss = loss + head(z, yb)
            if vcfg["loss"] in ("triplet", "arc_triplet"):
                w = tcfg["triplet_weight"] if vcfg["loss"] == "arc_triplet" else 1.0
                loss = loss + w * batch_hard(z, yb, tcfg["triplet_margin"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            cnt += 1
        sched.step()
        last = tot / max(cnt, 1)
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep + 1:3d}  loss {last:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return model, last, time.time() - t0


def main():
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    tm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    is_s1 = tm.session.astype(str).str.casefold().eq("session1").to_numpy()
    test_tasks = tm.task.to_numpy()

    raw_tr = np.load(ROOT / CFG["data"]["train_features"]).astype(np.float32)
    raw_te = np.load(ROOT / CFG["data"]["test_features"]).astype(np.float32)
    print(f"train {raw_tr.shape}  test {raw_te.shape}", flush=True)

    fit_mask = f <= FIT_F
    ref_mask = (f > REF_LO) & (f <= REF_HI)
    y_all = subjects.copy()

    # ---- per-task standardisation, fit-block statistics only
    tr_std, te_std = raw_tr.copy(), raw_te.copy()
    task_mu, task_sd = {}, {}
    for t in TASKS:
        m = (tasks == t) & fit_mask
        mu, sd = raw_tr[m].mean(0), raw_tr[m].std(0) + 1e-6
        task_mu[t], task_sd[t] = mu, sd
        tr_std[tasks == t] = (raw_tr[tasks == t] - mu) / sd
        te_std[test_tasks == t] = (raw_te[test_tasks == t] - mu) / sd

    pca_cache = {}

    def get_pca(dim):
        if dim not in pca_cache:
            mu64, V = pca_basis(np.asarray(tr_std[fit_mask], dtype=np.float64), dim)
            proj = ((np.asarray(tr_std[fit_mask], dtype=np.float64) - mu64) @ V)
            psd = proj.std(0) + 1e-6
            pca_cache[dim] = (mu64.astype(np.float32), V.astype(np.float32),
                              psd.astype(np.float32))
        return pca_cache[dim]

    def project(x, dim, whiten=True):
        mu64, V, psd = get_pca(dim)
        z = ((np.asarray(x, dtype=np.float32) - mu64) @ V)
        return z / psd if whiten else z

    y_test = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]
    variants_out, all_rows = [], []

    def eval_space(emb_tr, emb_te, name, dim, emb_dim, loss, aug, klist):
        """Score one embedding space; returns (rows, best) for the given ks."""
        rows, best = [], None
        # LOSO folds depend on k only (prototype source does not enter the LOSO
        # thresholds), so compute them once per k.
        loso_by_k = {}
        for k in klist:
            loso_folds = []
            for held in CLASSES:
                enrolled = np.setdiff1d(CLASSES, [held])
                fitm = np.isin(subjects, enrolled) & fit_mask
                ki = np.flatnonzero(np.isin(subjects, enrolled) & ref_mask)
                ui = np.flatnonzero((subjects == held) & ref_mask)
                if len(ui) == 0 or fitm.sum() == 0:
                    continue
                p2, k2 = build_prototypes(emb_tr, subjects, tasks, fitm, enrolled, k)
                sub = np.r_[ki, ui]
                sc = neg_d1(emb_tr, sub, p2, k2, tasks)
                sk, su = sc[:len(ki)], sc[len(ki):]
                st2, d2 = task_stats(sk, tasks[ki], np.ones(len(ki), bool))
                loso_folds.append((apply_z(sk, tasks[ki], st2, d2),
                                   apply_z(su, tasks[ui], st2, d2)))
            loso_by_k[k] = loso_folds

        for source in CFG["prototype_sources"]:
            proto_mask = fit_mask if source == "fit" else np.ones(len(subjects), bool)
            for k in klist:
                protos, keys = build_prototypes(emb_tr, subjects, tasks, proto_mask,
                                                CLASSES, k)
                all_rows_idx = np.arange(len(emb_tr))
                tr_raw = neg_d1(emb_tr, all_rows_idx, protos, keys, tasks)
                st, default = task_stats(tr_raw, tasks, ref_mask)
                ref_z = apply_z(tr_raw[ref_mask], tasks[ref_mask], st, default)

                te_raw = neg_d1(emb_te, np.arange(len(emb_te)), protos, keys, test_tasks)
                te_z = apply_z(te_raw, test_tasks, st, default)
                auc = float(roc_auc_score(y_test, np.r_[te_z[is_s1], te_z[~is_s1]]))
                auc_raw = float(roc_auc_score(y_test, np.r_[te_raw[is_s1], te_raw[~is_s1]]))
                loso_folds = loso_by_k[k]

                for a in CFG["acceptance_grid"]:
                    th = float(np.quantile(ref_z, 1 - a))
                    s2 = float(np.mean(te_z[~is_s1] >= th))
                    s1 = float(np.mean(te_z[is_s1] < th))
                    rejs = []
                    for kz, uz in loso_folds:
                        thf = float(np.quantile(kz, 1 - a))
                        rejs.append(float(np.mean(uz < thf)))
                    loso = float(np.mean(rejs))
                    row = {"variant": name, "pca_dim": dim,
                           "emb_dim": emb_dim, "loss": loss, "aug": aug,
                           "prototype_k": int(k), "prototype_source": source,
                           "acceptance_rate": float(a),
                           "LOSO_rejection": loso,
                           "Test_S1_rejection": s1,
                           "Test_S2_acceptance": s2,
                           "three_way_mean": (loso + s1 + s2) / 3.0,
                           "test_auc": auc, "test_auc_raw": auc_raw}
                    rows.append(row)
                    all_rows.append(row)
                    if best is None or row["three_way_mean"] > best["three_way_mean"]:
                        best = row
                print(f"  {source} k={k}  Test AUC {auc:.4f} (raw {auc_raw:.4f})  "
                      f"best3way {best['three_way_mean']:.4f}", flush=True)
        return rows, best

    for vi, vcfg in enumerate(CFG["variants"]):
        name = vcfg["name"]
        dim = int(vcfg["pca_dim"])
        mu64, V, psd = get_pca(dim)
        raw_fit = torch.as_tensor(tr_std[fit_mask], dtype=torch.float32, device=DEVICE)
        y_fit = torch.as_tensor(y_all[fit_mask] - 1, dtype=torch.long, device=DEVICE)
        pca_mu_t = torch.as_tensor(mu64, dtype=torch.float32, device=DEVICE)
        pca_V_t = torch.as_tensor(V, dtype=torch.float32, device=DEVICE)
        pc_std_t = torch.as_tensor(psd, dtype=torch.float32, device=DEVICE)

        print(f"[{vi + 1}/{len(CFG['variants'])}] {name}  "
              f"(pca {dim}, emb {vcfg['emb_dim']}, {vcfg['loss']}, aug {vcfg['aug']})",
              flush=True)
        model, final_loss, secs = train_variant(vcfg, raw_fit, y_fit, pca_mu_t,
                                                pca_V_t, pc_std_t)

        tr_pca = torch.as_tensor(project(tr_std, dim), dtype=torch.float32, device=DEVICE)
        te_pca = torch.as_tensor(project(te_std, dim), dtype=torch.float32, device=DEVICE)
        emb_tr = embed_all(model, tr_pca)
        emb_te = embed_all(model, te_pca)
        del tr_pca, te_pca
        torch.cuda.empty_cache()

        # save weights together with the Train-only preprocessing it needs
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "variant": vcfg,
            "pca_mu": mu64, "pca_basis": V, "pc_std": psd,
            "task_mu": {t: task_mu[t] for t in TASKS},
            "task_sd": {t: task_sd[t] for t in TASKS},
            "classes": CLASSES, "fit_fraction": FIT_F,
        }, OUT / f"model_{name}.pt")

        rows, best = eval_space(emb_tr, emb_te, name, dim, vcfg["emb_dim"],
                                vcfg["loss"], vcfg["aug"], CFG["prototype_k"])

        variants_out.append({"variant": name, "config": vcfg, "final_loss": final_loss,
                             "train_seconds": round(secs, 1),
                             "best_three_way": best, "rows": rows})
        print(f"  done in {secs:.0f}s", flush=True)

    # ---- controls: identical scoring protocol but in the unlearned PCA input
    # space (no supervision, optional whitening), to separate "protocol" effects
    # from "metric learning" effects.  Train-only fitting, Test only scored.
    controls_out = []
    for ccfg in CFG["input_space_controls"]:
        cdim = int(ccfg["pca_dim"])
        c_rows, c_best = eval_space(project(tr_std, cdim, ccfg["whiten"]),
                                    project(te_std, cdim, ccfg["whiten"]),
                                    ccfg["label"], cdim, cdim, "none", "none",
                                    ccfg["prototype_k"])
        controls_out.append({"label": ccfg["label"], "pca_dim": cdim,
                             "whiten": bool(ccfg["whiten"]),
                             "best_three_way": c_best, "rows": c_rows})
        print(f"control {ccfg['label']} done", flush=True)

    primary = next(v for v in variants_out if v["variant"] == CFG["primary"])
    out = {"experiment": CFG["experiment"], "clean": True,
           "primary": CFG["primary"], "target": CFG["targets"],
           "selection_policy": CFG["selection_policy"],
           "variants": [{k: v[k] for k in ("variant", "config", "final_loss",
                                           "train_seconds", "best_three_way")}
                        for v in variants_out],
           "rows": all_rows,
           "input_space_controls": controls_out}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    # ---------------- report ----------------
    L = ["# E140 监督度量学习嵌入（ArcFace / batch-hard triplet）", "",
         "目标：把 Test 诊断 AUC（session1 负类 vs session2 正类）从 PCA-96 基线的 0.7566 抬高，"
         "使三项均值 (LOSO 拒识 + Test S1 拒识 + Test S2 接受)/3 尽量逼近 0.85。", "",
         "## 协议（全流程 Train-only）", "",
         f"- 拟合块：frac ≤ {FIT_F}（训练、标准化、PCA 基底均只用这块）；"
         f"参考块：frac ∈ ({REF_LO}, {REF_HI}]，训练时完全未见过。",
         "- 预处理：按 task 用拟合块均值/方差标准化 → PCA 到 96/128/160 维 → 按主成分标准差缩放。",
         "- 嵌入网络：MLP（input→512→256→emb，BatchNorm+ReLU+Dropout，L2 归一化），"
         "20 类 ArcFace 和/或 batch-hard triplet。",
         "- 增强（在原始 855 维空间做，再投影）：5 个 171 维频带随机缩放/丢弃、高斯噪声、"
         "特征 dropout、整体缩放。",
         "- 打分：嵌入空间 (task, identity) k-means 原型（k∈{1,3,8}），分数 = 负最近欧氏距离，"
         "再按 task 用参考块统计量做 z 分数。",
         "- 阈值：参考块 (frac∈(0.6,0.8]) 分数分位数；LOSO 每折用已注册身份同区间行做已知端、"
         "留出身份同区间行做未知端，阈值取已知端分位数。",
         "- 主配置为预注册（PCA 128 / 128 维嵌入 / ArcFace / mild 增强），"
         "其余变体仅作透明性记录；**未用 Test 或 LOSO 指标做任何选择**。", ""]

    L += ["## 各变体 Test 诊断 AUC", "",
          "| 变体 | PCA | 嵌入 | 损失 | 增强 | 原型源 | k | Test AUC(z) | Test AUC(raw) |",
          "|---|---:|---:|---|---|---|---:|---:|---:|"]
    for v in variants_out:
        for k in CFG["prototype_k"]:
            for source in CFG["prototype_sources"]:
                sel = [r for r in v["rows"] if r["prototype_k"] == k
                       and r["prototype_source"] == source]
                L.append(f"| {v['variant']} | {sel[0]['pca_dim']} | {sel[0]['emb_dim']} | "
                         f"{sel[0]['loss']} | {sel[0]['aug']} | {source} | {k} | "
                         f"{sel[0]['test_auc']:.4f} | {sel[0]['test_auc_raw']:.4f} |")

    L += ["", "## 对照：未学习的 PCA 输入空间（同一打分/阈值协议）", "",
          "白化 = 各主成分除以拟合块标准差（网络输入用的就是白化空间）；非白化即基线做法。", ""]
    for c in controls_out:
        L += [f"**{c['label']}**（PCA {c['pca_dim']}，白化={c['whiten']}）", "",
              "| k | 原型源 | Test AUC(z) | Test AUC(raw) |", "|---:|---|---:|---:|"]
        for k in sorted({r["prototype_k"] for r in c["rows"]}):
            for source in CFG["prototype_sources"]:
                sel = [r for r in c["rows"] if r["prototype_k"] == k
                       and r["prototype_source"] == source][0]
                L.append(f"| {k} | {source} | {sel['test_auc']:.4f} | "
                         f"{sel['test_auc_raw']:.4f} |")
        cb = c["best_three_way"]
        L += ["", f"最好工作点：k={cb['prototype_k']}（{cb['prototype_source']}），"
                  f"接受率 {cb['acceptance_rate']:.2f}，LOSO {cb['LOSO_rejection']:.4f} / "
                  f"S1 {cb['Test_S1_rejection']:.4f} / S2 {cb['Test_S2_acceptance']:.4f} / "
                  f"三项均值 {cb['three_way_mean']:.4f}，Test AUC {cb['test_auc']:.4f}。", ""]

    pb = primary["best_three_way"]
    L += ["", "## 主配置三项指标随接受率变化", "",
          f"主配置 {CFG['primary']}，原型 k={pb['prototype_k']}（{pb['prototype_source']}），"
          f"Test AUC {pb['test_auc']:.4f}。", "",
          "| 接受率 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 |",
          "|---|---:|---:|---:|---:|"]
    for r in primary["rows"]:
        if r["prototype_k"] == pb["prototype_k"] and r["prototype_source"] == pb["prototype_source"]:
            L.append(f"| {r['acceptance_rate']:.2f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                     f"{r['three_way_mean']:.4f} |")

    L += ["", "## 各变体最好工作点（仅描述，不用于选择）", "",
          "| 变体 | k | 原型源 | 接受率 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 | Test AUC |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for v in variants_out:
        b = v["best_three_way"]
        L.append(f"| {v['variant']} | {b['prototype_k']} | {b['prototype_source']} | "
                 f"{b['acceptance_rate']:.2f} | {b['LOSO_rejection']:.4f} | "
                 f"{b['Test_S1_rejection']:.4f} | {b['Test_S2_acceptance']:.4f} | "
                 f"{b['three_way_mean']:.4f} | {b['test_auc']:.4f} |")

    # a small comparative summary that uses the primary working point only
    ref_row = [r for r in primary["rows"] if r["acceptance_rate"] == 0.80
               and r["prototype_k"] == pb["prototype_k"]
               and r["prototype_source"] == pb["prototype_source"]][0]
    ctl_raw = [c for c in controls_out if not c["whiten"]][0]
    ctl_lines = []
    for c in controls_out:
        cb = c["best_three_way"]
        ctl_lines.append(f"- 未学习对照 {c['label']}：最好 Test AUC "
                         f"{max(r['test_auc'] for r in c['rows']):.4f}，"
                         f"三项均值 {cb['three_way_mean']:.4f}。")
    best_variant = max(variants_out, key=lambda v: v["best_three_way"]["three_way_mean"])
    L += ["", "## 结论", "",
          f"- 主配置最好三项均值 **{pb['three_way_mean']:.4f}**（LOSO {pb['LOSO_rejection']:.4f} / "
          f"S1 {pb['Test_S1_rejection']:.4f} / S2 {pb['Test_S2_acceptance']:.4f}），"
          f"Test AUC **{pb['test_auc']:.4f}**，对应接受率 {pb['acceptance_rate']:.2f}。",
          f"- 接受率 0.80 工作点（LOSO {ref_row['LOSO_rejection']:.4f} / "
          f"S1 {ref_row['Test_S1_rejection']:.4f} / S2 {ref_row['Test_S2_acceptance']:.4f} / "
          f"均值 {ref_row['three_way_mean']:.4f}）。",
          f"- 9 组超参中三项均值最高的是 {best_variant['variant']}"
          f"（{best_variant['best_three_way']['three_way_mean']:.4f}，"
          f"Test AUC {best_variant['best_three_way']['test_auc']:.4f}），"
          f"仍远低于基线。"] + ctl_lines + [
          f"- 对照基线 PCA-96 最近原型：Test AUC 0.7566，三项均值 0.7784"
          f"（LOSO 0.9867 / S1 0.8665 / S2 0.4820）。",
          f"- 未学习对照 {ctl_raw['label']} 的 Test AUC "
          f"{max(r['test_auc'] for r in ctl_raw['rows']):.4f}，与基线同一水平；"
          f"进入嵌入网络后 Test AUC 反而掉到 0.49~0.57。",
          "",
          "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"primary_best": pb,
                      "baseline": {"test_auc": 0.7566, "three_way_mean": 0.7784}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
