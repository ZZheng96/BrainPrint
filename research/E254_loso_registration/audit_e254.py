"""E254g: 11-gate audit for the frozen E254 package.

Train-only gates (#1-#4) use cached Train features and the frozen transform.
LOSO gates (#5/#6/#11) are read from the nested identity-LOSO run, which refits
every transform inside each outer fold.
Test gates (#9/#10) are a one-shot audit after the model and threshold were
frozen; Test labels are never used for fitting or selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache"
PKG = ROOT / "final2/team_hymn"
OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_ID = 20


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def load_train():
    meta = pd.read_csv(CACHE / "train_metadata.csv")
    feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32).mean(1)).to(DEV)
    return meta, feat


def main() -> None:
    sys.path.insert(0, str(PKG))
    import model as M  # noqa: E402

    art = M.load_model(str(PKG), "cuda:0" if torch.cuda.is_available() else "cpu")
    meta, feat = load_train()
    tasks = torch.as_tensor(M._task_indices([{"task": t} for t in meta["task"]], len(meta)), device=DEV)
    subj = torch.as_tensor(meta["subject"].to_numpy(), device=DEV)
    order = torch.as_tensor(meta.groupby(["subject", "task"]).cumcount().to_numpy(), device=DEV)
    g = meta.groupby(["subject", "task"])
    frac = torch.as_tensor(((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy(), device=DEV)

    def build(rows, k):
        protos = torch.zeros(4, N_ID, k, art["pca_basis"].shape[-1], device=DEV)
        for s in range(1, N_ID + 1):
            for t in range(4):
                idx = torch.nonzero(rows & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                if len(idx) == 0:
                    continue
                for j in range(k):
                    chunk = idx[j * len(idx) // k:(j + 1) * len(idx) // k]
                    if len(chunk):
                        protos[t, s - 1, j] = M._project(art, feat[chunk], t).mean(0)
        return protos

    def match(protos, rows):
        idx = torch.nonzero(rows).flatten()
        out = torch.full((len(idx), N_ID), -float("inf"), device=DEV)
        for t in range(4):
            m = tasks[idx] == t
            if not bool(m.any()):
                continue
            q = unit(M._project(art, feat[idx[m]], t) @ art["whiten"])
            p = unit(protos[t].reshape(-1, protos.shape[-1]) @ art["whiten"]).reshape(
                N_ID, protos.shape[2], -1)
            out[m] = (q[:, None, None, :] * p[None]).sum(-1).max(2).values
        return out

    k = art["n_proto"]
    report = {}

    # 1. in-sample closed-set accuracy with the frozen full-Train prototypes
    full = build(torch.ones(len(feat), dtype=torch.bool, device=DEV), k)
    s = match(full, torch.ones(len(feat), dtype=torch.bool, device=DEV))
    pred = (s.argmax(1) + 1).cpu().numpy()
    report["1_train_in_sample_acc"] = float((pred == subj.cpu().numpy()).mean())

    # 2/3. 5 contiguous blocks per (identity, task); the scored block is excluded
    block = torch.as_tensor((g.cumcount().to_numpy() * 5) // g["trial"].transform("size").to_numpy(),
                            device=DEV).clamp(0, 4)
    oof_pred = np.zeros(len(feat), dtype=np.int64)
    for b in range(5):
        pb = build(block != b, k)
        rows = block == b
        sb = match(pb, rows)
        oof_pred[rows.cpu().numpy()] = (sb.argmax(1) + 1).cpu().numpy()
    truth = subj.cpu().numpy()
    report["2_oof_closed_acc"] = float((oof_pred == truth).mean())
    f1s = []
    for c in range(1, N_ID + 1):
        tp = int(((oof_pred == c) & (truth == c)).sum())
        fp = int(((oof_pred == c) & (truth != c)).sum())
        fn = int(((oof_pred != c) & (truth == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    report["3_oof_macro_f1"] = float(np.mean(f1s))

    # 4. forward block-out accuracy: prototypes from frac<=0.6, queries frac>0.8
    pf = build(frac <= 0.6, k)
    sf = match(pf, frac > 0.8)
    report["4_forward_block_out_acc"] = float(
        ((sf.argmax(1) + 1).cpu().numpy() == truth[(frac > 0.8).cpu().numpy()]).mean())

    nested = json.loads((OUT / "loso_nested.json").read_text(encoding="utf-8"))
    report["5_strict_loso_auroc"] = nested["outer_auroc_mean"]
    report["6_strict_loso_u_at_95"] = nested["outer_u95_mean"]
    report["11_strict_loso_unknown_rej_at_90"] = nested["outer_rej90_mean"]
    report["5_fold_min_auroc"] = nested["outer_auroc_min"]
    report["6_fold_min_u95"] = nested["outer_u95_min"]
    report["11_fold_min_rej90"] = nested["outer_rej90_min"]
    report["7_RA"] = "N/A (final round has no adversarial samples)"
    report["8_ASafeR"] = "N/A (final round has no adversarial samples)"

    # 9/10. one-shot Test audit with the frozen package
    te = pd.read_csv(CACHE / "test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    te_feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/test_feat.npy").astype(np.float32).mean(1)).to(DEV)
    te_tasks = torch.as_tensor(M._task_indices([{"task": t} for t in te["task"]], len(te)), device=DEV)
    keep = torch.ones(len(te), dtype=torch.bool, device=DEV)
    st = torch.full((len(te), N_ID), -float("inf"), device=DEV)
    for t in range(4):
        m = te_tasks == t
        if not bool(m.any()):
            continue
        q = unit(M._project(art, te_feat[m], t) @ art["whiten"])
        p = unit(full[t].reshape(-1, full.shape[-1]) @ art["whiten"]).reshape(
            N_ID, full.shape[2], -1)
        st[m] = (q[:, None, None, :] * p[None]).sum(-1).max(2).values
    top2 = st.topk(2, dim=1).values
    gate = top2[:, 0] + art["beta"] * (top2[:, 0] - top2[:, 1])
    accept = (gate >= art["tau"]).cpu().numpy()
    pred_te = (st.argmax(1) + 1).cpu().numpy()
    label = te["Label"].to_numpy()
    reg = label > 0
    unknown = label == 0
    report["9_test_registered_accept"] = float(accept[reg].mean())
    report["9b_test_registered_top1_given_accept"] = float(
        (pred_te[reg & accept] == label[reg & accept]).mean())
    report["9c_test_registered_e2e"] = float(((pred_te == label) & accept)[reg].mean())
    report["10_test_unregistered_rejection"] = float((~accept[unknown]).mean())
    report["tau"] = art["tau"]
    report["test_rows"] = int(len(te))

    (OUT / "gates11_e254.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
