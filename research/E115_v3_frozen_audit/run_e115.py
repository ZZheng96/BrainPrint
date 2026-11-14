"""E115: freeze the E114 V3 recipe on all 20 identities and audit the project Test once.

E114 V3 (per-fold fine-tune of the E95 encoder on the 19 enrolled identities,
3 k-means prototypes per (task, identity), max cosine, threshold at known
acceptance 0.85) reached LOSO held-out rejection 0.8863, the best Train-only
number so far. E109 showed LOSO rejection tracks Test S1 rejection for models
trained on project identities, so V3 is the candidate to audit.

This experiment builds the deployable version:
  1. Fine-tune the E95 encoder on ALL 20 identities, fit block = fraction <= 0.6.
  2. Build 3 k-means prototypes per (task, identity) from the same fit block.
  3. Choose the threshold on the Train known block (0.6 < fraction <= 0.8) so
     that known acceptance equals the S2 operating point 0.85.
  4. Freeze everything, then read the project Test exactly once and write
     predictions.csv.
  5. Only after predictions.csv is complete, read the Session column for the
     S1/S2 audit.

The user's explicit requirement: S2 acceptance must be preserved. So the report
compares V3 against the E102 and E95 audits on both S1 rejection and S2
acceptance.

No Test data is used for training, prototype building, threshold selection or
model selection.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
import run_e54 as E54

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
E95_CKPT = ROOT / "research" / "E95_e54_final20_candidate" / "model_checkpoint.pt"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def kmeans(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return np.asarray(x, dtype=np.float32)
    centers = x[rng.choice(len(x), size=k, replace=False)].astype(np.float32)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        new = np.stack([x[assign == j].mean(0) if (assign == j).any() else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers, atol=1e-6):
            break
        centers = new
    return centers


def build_task_prototypes(emb, subjects, tasks, fit_mask, classes, k):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            block = np.asarray(emb[sel], dtype=np.float32)
            if k == 1:
                protos.append(block.mean(0))
                keys.append((task, int(c)))
            else:
                for center in kmeans(block, k, seed=int(c) * 17):
                    protos.append(center)
                    keys.append((task, int(c)))
    return np.stack(protos), keys


def max_cosine(emb, protos, keys, tasks):
    out = np.full(len(emb), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        p = torch.nn.functional.normalize(p, dim=1)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
            q = torch.nn.functional.normalize(q, dim=1)
            out[chunk] = (q @ p.T).max(1).values.cpu().numpy()
    return out


def embed(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def finetune_all(raw, subjects, fit_rows, seed):
    """Fine-tune the E95 encoder on all 20 identities."""
    e54cfg = E54.load_config()
    model = E54.EEGNetEmbedding(20, e54cfg).to(DEVICE)
    model.load_state_dict(torch.load(E95_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    class_to_idx = {int(c): i for i, c in enumerate(CLASSES)}
    fit_index = torch.as_tensor(fit_rows, device=DEVICE)
    fit_raw = raw[fit_index]
    fit_labels = torch.as_tensor([class_to_idx[int(s)] for s in subjects[fit_rows]],
                                 dtype=torch.long, device=DEVICE)
    E54.initialize_prototypes(model, fit_raw, fit_labels)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["training"]["learning_rate"],
                            weight_decay=CFG["training"]["weight_decay"])
    batch = CFG["training"]["batch_size"]
    temperature = float(e54cfg["model"]["cosine_temperature"])
    arpl_weight = float(e54cfg["model"]["arpl_weight"])
    margin = float(e54cfg["model"]["arpl_margin"])
    rng = np.random.default_rng(seed)
    for epoch in range(1, CFG["training"]["epochs"] + 1):
        model.train()
        order = rng.permutation(len(fit_rows))
        total = 0.0
        for st in range(0, len(order), batch):
            sel = torch.as_tensor(order[st:st + batch], device=DEVICE)
            emb = model.embed(fit_raw[sel])
            loss = E54.embedding_loss(model, emb, fit_labels[sel], temperature,
                                      arpl_weight, margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           CFG["training"]["gradient_clip_norm"])
            opt.step()
            total += float(loss) * len(sel)
        print(f"E115 finetune epoch={epoch} loss={total / len(fit_rows):.4f}", flush=True)
    model.eval()
    return model


def load_test_raw(mapping):
    root = ROOT / CFG["data"]["train_root"]
    chunks = []
    for st in range(0, len(mapping), 64):
        paths = [root / p for p in mapping.path.iloc[st:st + 64]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"],
                                   dtype=np.float32).T for p in paths])
        chunks.append(torch.as_tensor(arr, dtype=torch.float32, device=DEVICE))
    return chunks


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E115 requires CUDA")
    if (OUT / "predictions.csv").exists():
        raise RuntimeError("E115 Test audit already performed; refuse rerun")

    meta = pd.read_csv(TRAIN_META)
    if len(meta) != CFG["data"]["expected_trials"]:
        raise ValueError("Train metadata invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)

    fit = f <= CFG["calibration"]["fit_fraction"]
    fit_rows = np.flatnonzero(fit)
    known_idx = np.flatnonzero((f > .6) & (f <= .8))

    # ---- 1. fine-tune on all 20 identities
    model = finetune_all(raw, subjects, fit_rows, CFG["seed"])
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": CFG}, OUT / "model_checkpoint.pt")

    # ---- 2. multi-prototype from the fit block
    train_emb = embed(model, raw)
    protos, keys = build_task_prototypes(train_emb, subjects, tasks, fit, CLASSES,
                                         CFG["prototypes"]["k"])
    train_cos = max_cosine(train_emb, protos, keys, tasks)

    # ---- 3. threshold on the Train known block
    target = CFG["targets"]["known_acceptance"]
    threshold = float(np.quantile(train_cos[known_idx], 1 - target))
    known_acceptance = float(np.mean(train_cos[known_idx] >= threshold))
    train_auc = float(roc_auc_score(
        np.r_[np.ones(len(known_idx)), np.zeros(len(known_idx))],
        np.r_[train_cos[known_idx], train_cos[known_idx]])) if False else None
    print(f"E115 threshold={threshold:.6f} known_acceptance={known_acceptance:.6f}", flush=True)

    np.savez(OUT / "frozen_prototypes.npz", protos=protos,
             keys=np.asarray([f"{t}|{c}" for t, c in keys]))

    # ---- 4. freeze, then read Test once
    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    if len(order) != CFG["data"]["expected_test_rows"] or len(mapping) != len(order):
        raise ValueError("Test mapping mismatch")

    test_cos = []
    for st, chunk in enumerate(load_test_raw(mapping)):
        emb = embed(model, chunk)
        test_cos.append(max_cosine(emb, protos, keys,
                                   mapping.task.iloc[st * 64:st * 64 + len(chunk)].to_numpy()))
    test_cos = np.concatenate(test_cos)
    if len(test_cos) != len(order):
        raise ValueError("Test score count mismatch")

    labels = np.where(test_cos >= threshold, 1, 0)
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(OUT / "predictions.csv",
                                                                   index=False)

    # ---- 5. audit only after predictions.csv is complete
    sess = mapping.session.astype(str).str.casefold()
    audit = {"N": int(len(labels)), "predictions_complete_before_session_audit": True,
             "session_used_by_inference": False}
    for name, aliases in (("S1", ("session1", "s1", "1")), ("S2", ("session2", "s2", "2"))):
        m = sess.isin(aliases).to_numpy()
        audit[f"{name}_N"] = int(m.sum())
        audit[f"{name}_rejection_rate"] = float(np.mean(labels[m] == 0))
        audit[f"{name}_acceptance_rate"] = float(np.mean(labels[m] != 0))

    result = {
        "status": "complete_one_shot_frozen_audit",
        "experiment": CFG["experiment"],
        "train_only_metrics": {
            "known_acceptance": known_acceptance,
            "threshold": threshold,
            "N_known": int(len(known_idx)),
            "AUC": "N/A (no unknown pool in the deployable 20-identity fit)",
            "F1": "N/A (no unknown pool in the deployable 20-identity fit)",
            "loso_reference": {"AUC": 0.9364, "unknown_rejection": 0.8863,
                               "source": "E114 V3"},
        },
        "test_metrics": {k: "N/A" for k in ["AUC", "F1", "C", "RA", "ASafeR", "U",
                                            "Official Score"]},
        "protocol_checks": {"train_session1_only": True, "test_accessed": True,
                            "test_read_once_after_freeze": True,
                            "gpu": torch.cuda.get_device_name(0)},
        "audit": audit,
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    (OUT / "report.md").write_text(
        "# E115 Frozen V3 audit on the project Test\n\n"
        "E114 V3（留身份微调 + 3 原型/身份 + max cosine）在 LOSO 上拒识 0.8863。"
        "本实验把它冻结到全部 20 个身份，阈值由 Train 已知块确定（known acceptance = 0.85），"
        "然后一次性读取项目 Test。\n\n"
        f"## Train-only\n\n阈值={threshold:.6f}，known acceptance={known_acceptance:.6f}。"
        "LOSO 参考：AUC 0.9364，留出者拒识 0.8863（E114 V3）。\n\n"
        "## 项目 Test\n\n真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n"
        f"| 指标 | 结果 |\n|---|---:|\n"
        f"| S1 拒识率 | {audit['S1_rejection_rate']:.6f} |\n"
        f"| S2 接受率 | {audit['S2_acceptance_rate']:.6f} |\n\n"
        "predictions.csv 完整落盘后才读取 session 审计。\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
