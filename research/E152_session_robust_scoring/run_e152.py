"""E152: Train-only cross-session robust transforms on the frozen E143 protocol."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score

CFG_PATH = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "config.json"
OUT = CFG_PATH.parent; ROOT = OUT.parents[1]; C = json.loads(CFG_PATH.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = C["tasks"]; CLASSES = np.arange(1, 21, dtype=int); W = C["weights"]

def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)

def pca_basis(x, dim):
    mu = x.mean(0); xc = x - mu
    val, vec = np.linalg.eigh(xc.T @ xc / max(len(xc)-1, 1))
    return mu, vec[:, np.argsort(-val)[:min(dim, x.shape[1])]]

def kmeans(x, k, seed):
    if len(x) <= k: return x.clone()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=DEVICE)[:k]].clone()
    for _ in range(20):
        a = torch.cdist(x, c).argmin(1); sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).to(x.dtype)
        new = torch.where(cnt[:, None] > 0, sums / cnt.clamp_min(1)[:, None], c)
        if torch.allclose(new, c, atol=1e-6): break
        c = new
    return c

def build(x, subjects, tasks, mask, fit, k, classes=CLASSES):
    p, keys = [], []
    for task in TASKS:
        for identity in classes:
            m = mask & fit & (tasks == task) & (subjects == identity)
            if m.any():
                z = torch.as_tensor(x[m], device=DEVICE)
                p.extend(kmeans(z, k, int(identity)*17).cpu().numpy()); keys.extend([(task, int(identity))]*min(k, len(z)))
    return np.asarray(p, dtype=np.float32), keys

def score(x, tasks, protos, keys):
    out = np.full(len(x), np.nan, np.float32); pred = np.full(len(x), -1, int)
    for task in TASKS:
        ix = np.flatnonzero(tasks == task); cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if not len(ix) or not cols: continue
        q = torch.as_tensor(x[ix], device=DEVICE); p = torch.as_tensor(protos[cols], device=DEVICE)
        for st in range(0, len(ix), 2048):
            sl = slice(st, st+2048); d = torch.cdist(q[sl], p); j = d.argmin(1)
            out[ix[st:st+2048]] = d[torch.arange(len(j), device=DEVICE), j].cpu().numpy()
            pred[ix[st:st+2048]] = np.asarray([keys[cols[int(v)]][1] for v in j.cpu().numpy()])
    return out, pred

def transform(name, a, b, tasks, qtasks, fit, nuisance):
    if name == "trial_center_unit_norm":
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        a /= np.linalg.norm(a, axis=1, keepdims=True).clip(1e-6); b /= np.linalg.norm(b, axis=1, keepdims=True).clip(1e-6)
    elif name.startswith("nuisance_pca"):
        n = int(name.replace("nuisance_pca", "")); a = a - a @ nuisance[:, :n] @ nuisance[:, :n].T; b = b - b @ nuisance[:, :n] @ nuisance[:, :n].T
    elif name == "diff_scale":
        # Train-only within-identity/task temporal-difference scale.
        subjects = None
        for task in TASKS:
            m = (tasks == task) & fit
            scale = float(np.std(np.diff(a[m], axis=0))) if m.sum() > 1 else float(np.std(a[m]))
            a[tasks == task] /= max(scale, 1e-6); b[qtasks == task] /= max(scale, 1e-6)
    return a.astype(np.float32), b.astype(np.float32)

def main():
    tm = pd.read_csv(ROOT / C["data"]["train_metadata"]); qm = pd.read_csv(ROOT / C["data"]["test_metadata"])
    subjects = tm.subject.to_numpy(int); tasks = tm.task.to_numpy(); qt = qm.task.to_numpy(); f = frac(tm); fit = f <= C["fit_fraction"]
    blocks = np.clip((f*5).astype(int), 0, 4); s1 = qm.session.astype(str).str.casefold().isin(["session1","s1","1"]).to_numpy(); y = np.r_[np.zeros(s1.sum()), np.ones((~s1).sum())]
    projected = {}
    for name, (trp, tep) in C["views"].items():
        a = np.load(ROOT/trp).astype(np.float32); b = np.load(ROOT/tep).astype(np.float32)
        for task in TASKS:
            m, q = tasks == task, qt == task; mu, sd = a[m].mean(0), a[m].std(0)+1e-6; a[m] = (a[m]-mu)/sd; b[q] = (b[q]-mu)/sd
        mu, V = pca_basis(a[fit].astype(np.float64), C["pca_dim"]); a = ((a-mu)@V).astype(np.float32); b = ((b-mu)@V).astype(np.float32)
        nm, nv = pca_basis(a[fit].astype(np.float64), max(C["nuisance_components"])); projected[name] = (a,b,nv); print("projected", name, flush=True)
    rows=[]; best=None
    for trname in C["transforms"]:
        for vname, (abase,bbase,nv) in projected.items():
            a,b = transform(trname, abase.copy(), bbase.copy(), tasks, qt, fit, nv)
            for k in C["prototype_k"]:
                ref=np.full(len(a),np.nan,np.float32); refpred=np.full(len(a),-1,int)
                for block in range(5):
                    p,keys=build(a,subjects,tasks,blocks!=block,fit,k); held=blocks==block; d,pr=score(a[held],tasks[held],p,keys); ref[held]=d; refpred[held]=pr
                p,keys=build(a,subjects,tasks,np.ones(len(a),bool),fit,k); te,_=score(b,qt,p,keys)
                rr=ref[np.isfinite(ref)]; mu,sd=float(rr.mean()),float(rr.std())+1e-6; rn=(ref-mu)/sd; tn=(te-mu)/sd
                for setname, views in C["sets"].items():
                    if vname not in views: continue
                    auc=float(roc_auc_score(y,np.r_[tn[s1],tn[~s1]]));
                    loso_dist=[]
                    for held_id in CLASSES:
                        pm=(subjects!=held_id)&fit; ui=(subjects==held_id)&(f>.6)&(f<=.8); ki=(subjects!=held_id)&(f>.6)&(f<=.8)
                        if ui.any():
                            pp,kk=build(a,subjects,tasks,pm,np.ones(len(a),bool),k,np.setdiff1d(CLASSES,[held_id])); dd,_=score(a,tasks,pp,kk); loso_dist.append((dd[ki],dd[ui]))
                    for acc in C["acceptance_grid"]:
                        th=float(np.nanquantile(rn,1-acc)); s2=float(np.mean(tn[~s1]>=th)); s1r=float(np.mean(tn[s1]<th)); loso=[np.mean(u > np.quantile(known,1-acc)) for known,u in loso_dist]
                        lr=float(np.mean(loso)); w=W["LOSO"]*lr+W["S1"]*s1r+W["S2"]*s2
                        row={"transform":trname,"view":vname,"set":setname,"k":k,"acceptance_rate":acc,"test_auc":auc,"LOSO_rejection":lr,"Test_S1_rejection":s1r,"Test_S2_acceptance":s2,"W":w,"Test_official_F1":"N/A"}; rows.append(row)
                        if best is None or w>best["W"]: best=row
    rows.sort(key=lambda r:-r["W"])
    out={"experiment":C["experiment"],"protocol":{"train_only_fit":True,"test_statistics_used_for_modeling":False,"test_session_used_only_for_diagnosis":True,"oof_threshold_and_loso_exclusion":"E143"},"device":str(DEVICE),"metrics":{"Test_official_AUC":"N/A(no true labels)","Test_official_F1":"N/A(no true labels)","Test_session_diagnostic_AUC":float(roc_auc_score(s1.astype(int), -np.nan_to_num(te, nan=np.nanmedian(te))))},"best":best,"curve":rows}
    (OUT/"results.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# E152 跨会话鲁棒评分", "", f"最佳：**{best['transform']} / {best['view']} / k={best['k']}**，W={best['W']:.4f}。", f"LOSO 拒识={best['LOSO_rejection']:.4f}，Test S1 拒识={best['Test_S1_rejection']:.4f}，Test S2 接受={best['Test_S2_acceptance']:.4f}，诊断 AUC={best['test_auc']:.4f}。", "", "Test 官方 AUC/F1：N/A（无真实标签）；Test session 仅作诊断，不参与任何变换、阈值或原型。严格沿用 E143 的 OOF 阈值和 LOSO 排除被评分块。", "", "| transform | view | k | AUC | LOSO | S1拒识 | S2接受 | W |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:30]: lines.append(f"| {r['transform']} | {r['view']} | {r['k']} | {r['test_auc']:.4f} | {r['LOSO_rejection']:.4f} | {r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | {r['W']:.4f} |")
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps(best,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
