"""E104: session-invariant pretraining on external multi-session data, then 18-channel transfer.

Motivation: E102/E103 showed the project Test suffers a cross-session domain shift
that no threshold can fix. BCIC IV-2a has two sessions per subject (T and E), so it
can teach an encoder to keep identity while discarding session.

Pipeline
  1. Pretrain E88Spatial18Embedding on external 22->18 IDW-projected data with
     cross-session supervised contrastive loss (positives = same subject, different
     session). This is the only place external data is used.
  2. Freeze the encoder. Build task-conditioned prototypes from project
     Train/session1 only, and score with the distance ratio from E102.
  3. Choose the threshold on Train known blocks plus external unknown subjects
     (the 9 BCIC subjects are never enrolled in the project, so they are genuine
     unregistered people for the project task).
  4. Freeze, then read the project Test once; Session is used only for the audit.

No project Test data is used for training, calibration or model selection.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
E73 = ROOT / "research" / "E73_external_multisession"
E88 = ROOT / "research" / "E88_spatial_channel_projection"
sys.path.insert(0, str(E88))
sys.path.insert(0, str(E73))
from e88_model import E88Spatial18Embedding
from external_data import load_external_data, subject_folds

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CFG["data"]["channels_18"] = 18
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / "BrainPrint_cache/train_metadata.csv"
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def projection_matrix():
    p = CFG["channel_projection"]
    src = p["source_channels_22"]; tgt = p["target_channels_18"]
    sx = np.asarray([p["source_xy"][n] for n in src], dtype=np.float32)
    tx = np.asarray([p["target_xy"][n] for n in tgt], dtype=np.float32)
    d = np.sqrt(((tx[:, None, :] - sx[None, :, :]) ** 2).sum(-1))
    k = min(int(p["k"]), len(src))
    w = np.zeros_like(d)
    for j in range(len(tgt)):
        nn = np.argsort(d[j])[:k]
        if d[j, nn[0]] < 1e-8:
            w[j, nn[0]] = 1.0
        else:
            w[j, nn] = 1.0 / np.maximum(d[j, nn], 1e-8) ** float(p["power"])
            w[j] /= w[j].sum()
    return torch.as_tensor(w, dtype=torch.float32)


PROJECTION = projection_matrix()


def project18(raw):
    return torch.einsum("ij,bjt->bit", PROJECTION.to(raw.device, dtype=raw.dtype), raw)


def supcon(z, subject, session, temperature):
    z = torch.nn.functional.normalize(z.float(), dim=1)
    logits = z @ z.T / float(temperature)
    pos = (subject[:, None] == subject[None, :]) & (session[:, None] != session[None, :])
    if not bool(pos.any()):
        raise ValueError("cross-session positive missing")
    logits = logits.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -torch.inf)
    lp = logits - torch.logsumexp(logits, 1, keepdim=True)
    return -(torch.where(pos, lp, torch.zeros_like(lp)).sum(1) / pos.sum(1).clamp_min(1)).mean()


def pretrain():
    """Cross-session SupCon on external data; returns the frozen encoder."""
    ckpt = OUT / "encoder.pt"
    if ckpt.is_file():
        model = E88Spatial18Embedding(CFG).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["state_dict"])
        model.eval()
        return model, {"reused": True}
    base = json.loads((E73 / "config.json").read_text(encoding="utf-8"))
    base["data"]["feature_key"] = "data"
    base["data"]["channels"] = 22
    data = load_external_data(base, DEVICE)
    model = E88Spatial18Embedding(CFG).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["training"]["learning_rate"],
                            weight_decay=CFG["training"]["weight_decay"])
    rng = np.random.default_rng(CFG["seed"])
    subjects = data.subject; sessions = data.session
    groups = []
    for s in np.unique(subjects):
        for sess in (0, 1):
            idx = np.flatnonzero((subjects == s) & (sessions == sess))
            if len(idx):
                groups.append(idx)
    history = []
    for epoch in range(1, CFG["training"]["epochs"] + 1):
        perms = [rng.permutation(g) for g in groups]
        k = CFG["training"]["trials_per_group"]
        steps = min(len(p) for p in perms) // k
        total = 0.0
        model.train()
        for step in range(steps):
            ix = np.concatenate([p[step * k:(step + 1) * k] for p in perms])
            sel = torch.as_tensor(ix, device=DEVICE)
            sub = torch.as_tensor(subjects[ix], device=DEVICE)
            ses = torch.as_tensor(sessions[ix], device=DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = supcon(model.embed(project18(data.raw[sel])), sub, ses, CFG["training"]["temperature"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["training"]["gradient_clip_norm"])
            opt.step()
            total += float(loss) * len(ix)
        row = {"epoch": epoch, "loss": total / (steps * len(groups) * k)}
        history.append(row)
        print(f"E104 pretrain epoch={epoch} loss={row['loss']:.5f}", flush=True)
    model.eval()
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "history": history, "config": CFG}, ckpt)
    return model, {"reused": False, "history": history}


def embed_raw(model, raw, batch=256):
    out = []
    with torch.no_grad():
        for st in range(0, len(raw), batch):
            out.append(model.embed(raw[st:st + batch]).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_task_prototypes(emb, subjects, tasks, fit_mask, classes):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(emb[sel], dtype=np.float32).mean(0))
            keys.append((task, int(c)))
    return np.stack(protos), keys


def distance_ratio(emb, protos, keys, tasks):
    ratio = np.full(len(emb), np.nan, dtype=np.float32)
    pred = np.zeros(len(emb), dtype=np.int64)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        cls = np.asarray([keys[i][1] for i in cols])
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(emb[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
            pred[chunk] = cls[order[:, 0]]
    return ratio, pred


def external_embeddings(model):
    base = json.loads((E73 / "config.json").read_text(encoding="utf-8"))
    base["data"]["feature_key"] = "data"
    base["data"]["channels"] = 22
    data = load_external_data(base, DEVICE)
    return embed_raw(model, project18(data.raw)), data


def choose_threshold(known_ratio, unknown_ratio):
    best = None
    for q in CFG["threshold"]["grid_quantiles"]:
        th = float(np.quantile(known_ratio, q))
        ka = float(np.mean(known_ratio >= th))
        ur = float(np.mean(unknown_ratio < th))
        if ka + 1e-12 < CFG["targets"]["known_acceptance"]:
            continue
        if ur + 1e-12 < CFG["targets"]["unknown_rejection"]:
            continue
        cand = (q, th, ka, ur)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        grid = np.unique(np.quantile(np.r_[known_ratio, unknown_ratio], np.linspace(0, 1, 801)))
        rows = [(float(t), float(np.mean(known_ratio >= t)), float(np.mean(unknown_ratio < t))) for t in grid]
        gap = [(1 - ka) + (1 - ur) for _, ka, ur in rows]
        i = int(np.argmin(gap))
        th, ka, ur = rows[i]
        return {"threshold": th, "known_acceptance": ka, "unknown_rejection": ur,
                "quantile": None, "target_met": False,
                "selection_source": "Train known + external unknown fallback: joint-closest point"}
    q, th, ka, ur = best
    return {"threshold": th, "known_acceptance": ka, "unknown_rejection": ur,
            "quantile": float(q), "target_met": True,
            "selection_source": "Train known + external BCIC IV-2a unknown; no project Test"}


def load_test(model, protos, keys):
    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"], usecols=["sample", "path", "session", "task"])
    if len(order) != CFG["data"]["expected_test_rows"] or len(mapping) != len(order):
        raise ValueError("Test mapping mismatch")
    root = ROOT / CFG["data"]["train_root"]
    ratios = np.full(len(mapping), np.nan, dtype=np.float32)
    preds = np.zeros(len(mapping), dtype=np.int64)
    for st in range(0, len(mapping), 64):
        stop = min(st + 64, len(mapping))
        paths = [root / p for p in mapping.path.iloc[st:stop]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"], dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        emb = embed_raw(model, x)
        r, p = distance_ratio(emb, protos, keys, mapping.task.iloc[st:stop].to_numpy())
        ratios[st:stop] = r
        preds[st:stop] = p
    return order, mapping, ratios, preds


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E104 requires CUDA")
    if (OUT / "predictions.csv").exists():
        raise RuntimeError("E104 Test audit already performed; refuse rerun")
    model, train_info = pretrain()

    meta = pd.read_csv(TRAIN_META)
    if len(meta) != 11304:
        raise ValueError("Train metadata invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit = f <= CFG["calibration"]["fit_fraction"]
    cal_known = np.flatnonzero((f > .6) & (f <= .8))

    # Project Train raw is needed for embeddings; reuse the E54 loader for identical preprocessing.
    sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
    import run_e54 as E54
    e54cfg = E54.load_config()
    raw = E54.load_all_raw(meta, e54cfg, DEVICE)
    train_emb = embed_raw(model, raw)

    protos, keys = build_task_prototypes(train_emb, subjects, tasks, fit, CLASSES)
    ratio_all, pred_all = distance_ratio(train_emb, protos, keys, tasks)

    ext_emb, ext_data = external_embeddings(model)
    ext_ratio, _ = distance_ratio(ext_emb, protos, keys, np.full(len(ext_emb), "Imagery", dtype=object))
    ext_ratio = ext_ratio[np.isfinite(ext_ratio)]
    if len(ext_ratio) == 0:
        raise ValueError("no finite external ratio")

    rule = choose_threshold(ratio_all[cal_known], ext_ratio)
    auc = float(roc_auc_score(np.r_[np.ones(len(cal_known)), np.zeros(len(ext_ratio))],
                              np.r_[ratio_all[cal_known], ext_ratio]))
    cal_pred = np.where(ratio_all[cal_known] >= rule["threshold"], pred_all[cal_known], 0)
    f1 = float(f1_score(subjects[cal_known], cal_pred, labels=CLASSES, average="macro", zero_division=0))

    torch.save({"prototypes": protos, "keys": keys, "rule": rule, "classes": CLASSES,
                "config": CFG}, OUT / "model_checkpoint.pt")
    result = {"status": "frozen_before_test", "experiment": CFG["experiment"],
              "pretraining": train_info,
              "train_only_metrics": {"AUC": auc, "F1": f1,
                                     "N_known": int(len(cal_known)),
                                     "N_external_unknown": int(len(ext_ratio)),
                                     "rule": rule},
              "test_metrics": {k: "N/A" for k in ["AUC", "F1", "C", "RA", "ASafeR", "U", "Official Score"]},
              "protocol_checks": {"train_session1_only_for_prototypes": True,
                                  "external_used_for_pretraining_and_unknown": True,
                                  "test_accessed": False, "gpu": torch.cuda.get_device_name(0)}}

    order, mapping, ratios, preds = load_test(model, protos, keys)
    labels = np.where(ratios >= rule["threshold"], preds, 0)
    if len(labels) != len(order):
        raise ValueError("prediction count mismatch")
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(OUT / "predictions.csv", index=False)

    sess = mapping.session.astype(str).str.casefold()
    audit = {"N": int(len(labels)), "predictions_complete_before_session_audit": True,
             "session_used_by_inference": False}
    for name, aliases in (("S1", ("session1", "s1", "1")), ("S2", ("session2", "s2", "2"))):
        m = sess.isin(aliases).to_numpy()
        audit[f"{name}_N"] = int(m.sum())
        audit[f"{name}_rejection_rate"] = float(np.mean(labels[m] == 0))
        audit[f"{name}_acceptance_rate"] = float(np.mean(labels[m] != 0))
    audit["S1_target"] = CFG["targets"]["unknown_rejection"]
    audit["S2_target"] = CFG["targets"]["known_acceptance"]
    audit["S1_pass"] = audit["S1_rejection_rate"] >= CFG["targets"]["unknown_rejection"]
    audit["S2_pass"] = audit["S2_acceptance_rate"] >= CFG["targets"]["known_acceptance"]
    result["audit"] = audit
    result["protocol_checks"].update({"test_accessed": True, "test_read_once_after_freeze": True})
    result["status"] = "complete_one_shot_frozen_audit"
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "report.md").write_text(
        "# E104 session-invariant pretraining + 18-channel transfer\n\n"
        "外部 BCIC IV-2a 双 Session 做跨 Session SupCon 预训练（22→18 IDW 投影），"
        "冻结编码器后在项目 Train/session1 上建任务条件化原型，用 distance ratio 打分；"
        "阈值由 Train 已知块 + 外部未知确定，Test 冻结后一次性读取。\n\n"
        f"## Train + 外部\n\nAUC={auc:.6f}，F1={f1:.6f}；阈值={rule['threshold']:.6f}，"
        f"known acceptance={rule['known_acceptance']:.6f}，external unknown rejection={rule['unknown_rejection']:.6f}，"
        f"目标达成={rule['target_met']}。\n\n"
        "## 项目 Test\n\n真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n"
        f"| 项目 | 结果 | 目标 | 是否达标 |\n|---|---:|---:|---|\n"
        f"| S1 拒识率 | {audit['S1_rejection_rate']:.6f} | ≥{audit['S1_target']:.2f} | {audit['S1_pass']} |\n"
        f"| S2 接受率 | {audit['S2_acceptance_rate']:.6f} | ≥{audit['S2_target']:.2f} | {audit['S2_pass']} |\n\n"
        "predictions.csv 完整落盘后才读取 session 审计。\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
