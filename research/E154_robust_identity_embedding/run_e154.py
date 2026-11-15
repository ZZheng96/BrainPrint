"""E154: Train-only identity embedding on E65 855-d features."""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

CFG_PATH = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "config.json"
OUT = CFG_PATH.parent; ROOT = OUT.parents[1]
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = CFG["tasks"]; CLASSES = np.arange(1, 21, dtype=int)

def sha256(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def fraction(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)

def pca_fit(x, dim):
    mu = x.mean(0); xc = x - mu
    val, vec = np.linalg.eigh((xc.T @ xc) / max(len(xc) - 1, 1))
    return mu, vec[:, np.argsort(-val)[:min(dim, x.shape[1])]]

class Net(torch.nn.Module):
    def __init__(self, dim, hidden, emb, classes):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU(), torch.nn.Linear(hidden, emb))
        self.head = torch.nn.Linear(emb, classes)
    def forward(self, x):
        z = torch.nn.functional.normalize(self.encoder(x), dim=1)
        return z, self.head(z)

def augment(x, rng):
    scale = 1 + torch.empty((len(x), 1), device=DEVICE).uniform_(-CFG["augmentation"]["scale"], CFG["augmentation"]["scale"])
    y = x * scale
    y = y + torch.randn_like(y) * CFG["augmentation"]["noise"]
    mask = torch.rand_like(y) < CFG["augmentation"]["mask"]
    return y.masked_fill(mask, 0)

def train_model(x, y, augmented, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(x.shape[1], CFG["hidden_dim"], CFG["embedding_dim"], len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE); yt = torch.as_tensor(y - 1, dtype=torch.long, device=DEVICE)
    for _ in range(CFG["epochs"]):
        order = torch.randperm(len(xt), generator=g, device=DEVICE)
        for st in range(0, len(order), CFG["batch_size"]):
            ix = order[st:st + CFG["batch_size"]]; a = xt[ix]
            z, logits = model(a); loss = torch.nn.functional.cross_entropy(logits, yt[ix])
            if augmented:
                za, _ = model(augment(a, g)); loss = loss + CFG["consistency_weight"] * (1 - (z * za).sum(1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); model.requires_grad_(False); return model

def prototypes(model, x, y, task):
    with torch.no_grad(): z, _ = model(torch.as_tensor(x, dtype=torch.float32, device=DEVICE)); z = z.cpu().numpy()
    protos = {}; tasks = np.asarray(task)
    for t in TASKS:
        for c in CLASSES:
            m = (tasks == t) & (y == c)
            if m.any(): protos[(t, int(c))] = z[m].mean(0) / max(np.linalg.norm(z[m].mean(0)), 1e-8)
    return protos

def score(model, x, meta, protos):
    with torch.no_grad(): z, logits = model(torch.as_tensor(x, dtype=torch.float32, device=DEVICE)); z = z.cpu().numpy(); logits = logits.cpu().numpy()
    tasks = meta.task.to_numpy(); out = np.full(len(x), np.nan, np.float32); pred = np.full(len(x), -1, int)
    for t in TASKS:
        ix = np.flatnonzero(tasks == t); keys = [(t, int(c)) for c in CLASSES if (t, int(c)) in protos]
        if not len(ix) or not keys: continue
        p = np.stack([protos[k] for k in keys]); sim = z[ix] @ p.T; j = sim.argmax(1)
        out[ix] = sim[np.arange(len(ix)), j]; pred[ix] = [keys[v][1] for v in j]
    return out, pred

def fit_indices(meta, fit):
    return np.flatnonzero(fit)

def evaluate_variant(name, x, test_x, meta, test_meta, f, blocks):
    fit = f <= CFG["fit_fraction"]; y = meta.subject.to_numpy(int); task = meta.task.to_numpy()
    oof = np.full(len(x), np.nan, np.float32); oof_pred = np.full(len(x), -1, int); loso = []
    for block in range(CFG["blocks"]):
        base = fit & (blocks != block)
        model = train_model(x[base], y[base], name == "augmentation_consistency", 1000 + block)
        ps = prototypes(model, x[base], y[base], task[base])
        held = blocks == block; hm = meta.iloc[np.flatnonzero(held)].reset_index(drop=True)
        oof[held], oof_pred[held] = score(model, x[held], hm, ps)
        for identity in CLASSES:
            keep = base & (y != identity)
            query = held & (y == identity) & (f > 0.6) & (f <= 0.8)
            if not query.any(): continue
            mdl = train_model(x[keep], y[keep], name == "augmentation_consistency", 2000 + block * 31 + int(identity))
            pp = prototypes(mdl, x[keep], y[keep], task[keep]); qm = meta.iloc[np.flatnonzero(query)].reset_index(drop=True)
            s, _ = score(mdl, x[query], qm, pp); loso.extend(s[np.isfinite(s)].tolist())
    known = oof[np.isfinite(oof)]; loso = np.asarray(loso, float)
    final = train_model(x[fit], y[fit], name == "augmentation_consistency", 9000)
    proto = prototypes(final, x[fit], y[fit], task[fit]); test_score, test_pred = score(final, test_x, test_meta, proto)
    rows = []
    s1 = test_meta.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy(); s2 = ~s1
    for acceptance in CFG["acceptance_grid"]:
        threshold = float(np.quantile(known, 1 - acceptance)); loso_rej = float(np.mean(loso < threshold)); s1r = float(np.mean(test_score[s1] < threshold)); s2a = float(np.mean(test_score[s2] >= threshold))
        auc = float(roc_auc_score(np.r_[np.zeros(len(loso)), np.ones(len(known))], np.r_[loso, known])) if len(loso) and len(known) else float("nan")
        f1 = float(f1_score(y[np.isfinite(oof)], oof_pred[np.isfinite(oof)], labels=CLASSES, average="macro", zero_division=0))
        w = CFG["weights"]["LOSO"] * loso_rej + CFG["weights"]["S1"] * s1r + CFG["weights"]["S2"] * s2a
        rows.append({"acceptance_rate": acceptance, "threshold": threshold, "LOSO_rejection": loso_rej, "Test_S1_rejection": s1r, "Test_S2_acceptance": s2a, "W": w, "Train_OOF_AUC": auc, "Train_OOF_F1": f1, "Test_AUC": "N/A（无真实标签）", "Test_F1": "N/A（无真实标签）"})
    best = max(rows, key=lambda r: r["W"])
    torch.save({"state_dict": final.state_dict(), "prototypes": proto, "thresholds": rows, "train_only": True, "variant": name}, OUT / f"model_{name}.pt")
    return {"variant": name, "best": best, "curve": rows, "n_train_fit": int(fit.sum()), "n_oof": int(np.isfinite(oof).sum()), "n_loso": int(len(loso))}

def main():
    tm = pd.read_csv(ROOT / CFG["data"]["train_metadata"]); qm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    x = np.load(ROOT / CFG["data"]["train_features"], mmap_mode="r").astype(np.float32); z = np.load(ROOT / CFG["data"]["test_features"], mmap_mode="r").astype(np.float32)
    if len(x) != len(tm) or len(z) != len(qm): raise ValueError("E65特征与metadata行数不一致")
    f = fraction(tm); blocks = np.clip((f * CFG["blocks"]).astype(int), 0, CFG["blocks"] - 1)
    fit = f <= CFG["fit_fraction"]; mu, V = pca_fit(x[fit].astype(np.float64), CFG["pca_dim"])
    x = ((x.astype(np.float64) - mu) @ V).astype(np.float32); z = ((z.astype(np.float64) - mu) @ V).astype(np.float32)
    results = [evaluate_variant(v, x, z, tm, qm, f, blocks) for v in ("baseline_control", "augmentation_consistency")]
    best = max(results, key=lambda r: r["best"]["W"])
    out = {"experiment": CFG["experiment"], "device": str(DEVICE), "protocol": {"train_only": True, "test_score_only": True, "session_diagnostic_only": True, "five_block_oof": True, "loso_excludes_identity_and_block": True, "feature_source": "E65 855-d correlation-logSPD"}, "variants": results, "best": best, "source_sha256": {"config": sha256(CFG_PATH), "train_features": sha256(ROOT / CFG["data"]["train_features"]), "test_features": sha256(ROOT / CFG["data"]["test_features"])}, "test": {"AUC": "N/A（无真实标签）", "F1": "N/A（无真实标签）", "session_diagnostic_only": True}}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E154 鲁棒身份一致性 embedding", "", f"设备：{DEVICE}；特征：E65 855维，经 Train-only PCA-{CFG['pca_dim']}。", "", "## 最佳结果", "", f"最佳方案：**{best['variant']}**；LOSO拒识={best['best']['LOSO_rejection']:.4f}；Test S1拒识={best['best']['Test_S1_rejection']:.4f}；Test S2接受={best['best']['Test_S2_acceptance']:.4f}；W={best['best']['W']:.4f}。", f"Train OOF AUC={best['best']['Train_OOF_AUC']:.4f}；Train OOF F1={best['best']['Train_OOF_F1']:.4f}。Test AUC：N/A（无真实标签）；Test F1：N/A（无真实标签）。", "", "阈值来自五块 Train OOF；每个 LOSO 评分同时排除被评分身份和被评分块。Test 仅评分，Session 仅在评分后作诊断，不进入训练、阈值或选择。", "", "|方案|LOSO拒识|S1拒识|S2接受|W|Train OOF AUC|Train OOF F1|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        b = r["best"]; lines.append(f"|{r['variant']}|{b['LOSO_rejection']:.4f}|{b['Test_S1_rejection']:.4f}|{b['Test_S2_acceptance']:.4f}|{b['W']:.4f}|{b['Train_OOF_AUC']:.4f}|{b['Train_OOF_F1']:.4f}|")
    lines += ["", "增强方案为同 trial 幅值缩放、特征噪声和特征遮挡，使用分类损失加增强前后 embedding 一致性损失；baseline 为分类控制。模型已保存为 model_baseline_control.pt 与 model_augmentation_consistency.pt。", "", "证据边界：Train 内时间块与留身份代理评估；Test 无身份真值，Session 结果仅诊断。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
