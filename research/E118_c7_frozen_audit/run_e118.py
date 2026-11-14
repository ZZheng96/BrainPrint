"""E118: freeze C7 (V3k3 + SPD ratio rank fusion) on all 20 identities and audit Test once.

E117 found C7 to be the first candidate meeting both LOSO targets in 19/20 folds:
  known acceptance >= 0.85 AND unknown rejection >= 0.80
with LOSO AUC 0.9571.

C7 = 0.5 * rank(fine-tuned cosine with 3 prototypes) + 0.5 * rank(SPD distance ratio)

Deployable pipeline:
  1. Fine-tune the E95 encoder on all 20 identities (fit block = fraction <= 0.6).
  2. Build 3 k-means prototypes per (task, identity) from the same fit block.
  3. Build task-conditioned correlation-logSPD prototypes from the same fit block.
  4. Rank-normalise both scores on the Train known block (0.6 < fraction <= 0.8)
     and average them.
  5. Choose the threshold on the Train known block so known acceptance = 0.85.
  6. Freeze, read Test once, write predictions.csv.
  7. Only after predictions.csv is complete, read Session for the S1/S2 audit.

The rank normalisation reference is the Train known block only, so no Test
information enters the score construction.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e54 as E54
import run_e65
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
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


def build_task_prototypes(emb, subjects, tasks, fit_mask, classes, k=1):
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


def distance_ratio(feat, protos, keys, tasks):
    ratio = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
    return ratio


def embed(model, raw, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def finetune_all(raw, subjects, fit_rows, seed):
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
        print(f"E118 finetune epoch={epoch} loss={total / len(fit_rows):.4f}", flush=True)
    model.eval()
    return model


def rank_normalize(values, reference):
    """Map values to [0,1] using the reference distribution's empirical CDF."""
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E118 requires CUDA")
    if (OUT / "predictions.csv").exists():
        raise RuntimeError("E118 Test audit already performed; refuse rerun")

    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
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
    ft_protos, ft_keys = build_task_prototypes(train_emb, subjects, tasks, fit, CLASSES,
                                               CFG["prototypes"]["k"])
    train_cos = max_cosine(train_emb, ft_protos, ft_keys, tasks)

    # ---- 3. SPD ratio prototypes from the same fit block
    sp_protos, sp_keys = build_task_prototypes(feat, subjects, tasks, fit, CLASSES, k=1)
    train_ratio = distance_ratio(feat, sp_protos, sp_keys, tasks)

    # ---- 4. rank fusion, reference = Train known block only
    ref_cos = train_cos[known_idx]
    ref_ratio = train_ratio[known_idx]
    fused_train = 0.5 * (rank_normalize(train_cos, ref_cos) + rank_normalize(train_ratio, ref_ratio))

    # ---- 5. threshold on the Train known block
    target = CFG["targets"]["known_acceptance"]
    threshold = float(np.quantile(fused_train[known_idx], 1 - target))
    known_acceptance = float(np.mean(fused_train[known_idx] >= threshold))
    print(f"E118 threshold={threshold:.6f} known_acceptance={known_acceptance:.6f}", flush=True)

    np.savez(OUT / "frozen_prototypes.npz",
             ft_protos=ft_protos,
             ft_keys=np.asarray([f"{t}|{c}" for t, c in ft_keys]),
             sp_protos=sp_protos,
             sp_keys=np.asarray([f"{t}|{c}" for t, c in sp_keys]),
             ref_cos=ref_cos, ref_ratio=ref_ratio)

    # ---- 6. freeze, then read Test once
    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    if len(order) != CFG["data"]["expected_test_rows"] or len(mapping) != len(order):
        raise ValueError("Test mapping mismatch")

    root = ROOT / CFG["data"]["train_root"]
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    fused_test = []
    for st in range(0, len(mapping), 64):
        stop = min(st + 64, len(mapping))
        paths = [root / p for p in mapping.path.iloc[st:stop]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"],
                                   dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        tt = mapping.task.iloc[st:stop].to_numpy()

        emb = embed(model, x)
        cos = max_cosine(emb, ft_protos, ft_keys, tt)
        with torch.no_grad():
            tf = run_e65.gpu_correlation_logspd(x, e36, e65cfg).cpu().numpy()
        ratio = distance_ratio(tf, sp_protos, sp_keys, tt)
        fused_test.append(0.5 * (rank_normalize(cos, ref_cos) + rank_normalize(ratio, ref_ratio)))
    fused_test = np.concatenate(fused_test)
    if len(fused_test) != len(order):
        raise ValueError("Test score count mismatch")

    labels = np.where(fused_test >= threshold, 1, 0)
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(OUT / "predictions.csv",
                                                                   index=False)

    # ---- 7. audit only after predictions.csv is complete
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
            "loso_reference": {"AUC": 0.9571, "known_acceptance": 0.9373,
                               "unknown_rejection": 0.8090, "folds_feasible": "19/20",
                               "source": "E117 C7"},
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
        "# E118 Frozen C7 audit on the project Test\n\n"
        "C7 = 0.5 × rank(微调 cosine, k=3 原型) + 0.5 × rank(E102 SPD distance ratio)。"
        "E117 在 LOSO 上 19/20 折双达标（ka≥0.85 且 ur≥0.80），AUC 0.9571。"
        "本实验冻结到全部 20 个身份，阈值由 Train 已知块确定，然后一次性读取 Test。\n\n"
        f"## Train-only\n\n阈值={threshold:.6f}，known acceptance={known_acceptance:.6f}。"
        "LOSO 参考：AUC 0.9571，ka 0.9373 / ur 0.8090（19/20 折）。\n\n"
        "## 项目 Test\n\n真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n"
        f"| 指标 | 结果 |\n|---|---:|\n"
        f"| S1 拒识率 | {audit['S1_rejection_rate']:.6f} |\n"
        f"| S2 接受率 | {audit['S2_acceptance_rate']:.6f} |\n\n"
        "predictions.csv 完整落盘后才读取 session 审计。\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
