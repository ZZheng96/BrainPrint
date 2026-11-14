"""E76: frozen E74 embeddings with a structural Mahalanobis open-set head."""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

E76_DIR = Path(__file__).resolve().parent
ROOT = E76_DIR.parents[1]
E74_DIR = ROOT / "research" / "E74_multiscale_native22"
sys.path.insert(0, str(E74_DIR)); sys.path.insert(0, str(ROOT / "research" / "E73_external_multisession"))
from external_data import load_external_data, subject_folds
from multiscale_model import MultiScaleNative22

CFG = json.loads((E76_DIR / "config.json").read_text(encoding="utf-8"))
BASE = json.loads((E74_DIR / "config.json").read_text(encoding="utf-8-sig"))

def sha256(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def embed(model, raw, indices, batch=256):
    out=[]
    with torch.no_grad():
        for i in range(0,len(indices),batch):
            idx=torch.as_tensor(indices[i:i+batch],dtype=torch.long,device=raw.device)
            out.append(model.embed(raw[idx]).float())
    return torch.cat(out) if out else torch.empty((0,64),device=raw.device)

def load_encoder(data, roles, fold):
    ckpt=E74_DIR/"outputs"/"models"/f"supcon_outer{fold}.pt"
    if not ckpt.is_file(): raise FileNotFoundError(f"Missing frozen E74 checkpoint: {ckpt}")
    saved=torch.load(ckpt,map_location=data.raw.device,weights_only=False)
    expected={"config":sha256(E74_DIR/"config.json"),"run_e74":sha256(E74_DIR/"run_e74.py"),"external_data":sha256(E74_DIR/"external_data.py"),"multiscale_model":sha256(E74_DIR/"multiscale_model.py")}
    if saved.get("source_hashes") != expected: raise ValueError("E74 checkpoint source hash mismatch")
    if saved.get("roles") != {k:v.tolist() for k,v in roles.items()}: raise ValueError("E74 checkpoint roles mismatch")
    model=MultiScaleNative22(BASE).to(data.raw.device); model.load_state_dict(saved["model_state"]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, {"path":str(ckpt),"sha256":sha256(ckpt)}

def fit_head(train_x, train_y, template_x, template_y, classes):
    means=torch.stack([template_x[torch.as_tensor(template_y==c,device=template_x.device)].mean(0) for c in classes])
    residual=train_x-torch.stack([means[list(classes).index(int(y))] for y in train_y])
    cov=(residual.T@residual)/max(1,len(residual)-1)
    diag=torch.diag(torch.diag(cov)); cov=(1-CFG["head"]["shrinkage"])*cov+CFG["head"]["shrinkage"]*diag
    inv=torch.linalg.pinv(cov+CFG["head"]["eigenvalue_floor"]*torch.eye(cov.shape[0],device=cov.device))
    return means,inv

def scores(x, means, inv):
    d=x[:,None,:]-means[None,:,:]
    return -torch.einsum("nkd,df,nkf->nk",d,inv,d).cpu().numpy()

def metrics(known_s, unknown_s, true, classes, threshold):
    kc=known_s.max(1); uc=unknown_s.max(1); kp=classes[known_s.argmax(1)]; accepted=kc>=threshold
    labels=np.r_[np.ones(len(kc)),np.zeros(len(uc))]; auc=roc_auc_score(labels,np.r_[kc,uc])
    pred=np.where(accepted,kp,0); det=np.r_[accepted,uc>=threshold]
    return {"N_known":len(kc),"N_unknown":len(uc),"threshold":float(threshold),"AUROC":float(auc),"identity_macro_F1":float(f1_score(true,pred,labels=classes,average="macro",zero_division=0)),"unknown_F1":float(f1_score(labels,det,pos_label=0,zero_division=0)),"C":float(np.mean(pred==true)),"U":float(np.mean(uc<threshold)),"known_acceptance":float(np.mean(accepted))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("E76 formal evaluation requires CUDA")
    device=torch.device("cuda"); data=load_external_data(BASE,device); folds=subject_folds(BASE)[:1] if args.smoke else subject_folds(BASE)
    runs=[]
    for fold,roles in enumerate(folds,1):
        model,enc=load_encoder(data,roles,fold); enrolled=roles["enrolled"]; directions={}
        for source,target in (("T","E"),("E","T")):
            train_i=data.indices(enrolled,session=source,runs=[0,1,2]); train_j=data.indices(enrolled,session=target,runs=[0,1,2]); templ_i=data.indices(enrolled,session=source,runs=[3]); known_i=data.indices(enrolled,session=target,runs=[4]); cal_u=data.indices(roles["calibration_unknown"],session=target,runs=[5]); outer_i=data.indices(enrolled,session=target,runs=[5]); outer_u=data.indices(roles["outer_unknown"],session=target,runs=[5])
            tx,ty=embed(model,data.raw,train_i),data.subject[train_i]; ex=embed(model,data.raw,train_j); ty2=data.subject[train_j]; template=embed(model,data.raw,templ_i); template_y=data.subject[templ_i]
            means,inv=fit_head(torch.cat([tx,ex]),np.r_[ty,ty2],template,template_y,enrolled)
            known=scores(embed(model,data.raw,known_i),means,inv); cal_unknown=scores(embed(model,data.raw,cal_u),means,inv); threshold=float(np.quantile(known.max(1),.05))
            inner=metrics(known,cal_unknown,data.subject[known_i],enrolled,threshold)
            if args.smoke: outer="not_accessed_in_smoke"
            else: outer=metrics(scores(embed(model,data.raw,outer_i),means,inv),scores(embed(model,data.raw,outer_u),means,inv),data.subject[outer_i],enrolled,threshold)
            directions[f"{source}_to_{target}"]={"inner":inner,"calibration_unknown_diagnostic_only":True,"outer":outer}
            head_path=E76_DIR/"outputs"/"models"/f"mahalanobis_outer{fold}_{source}_to_{target}.pt"
            head_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"means":means.detach().cpu(),"inverse_covariance":inv.detach().cpu(),"classes":np.asarray(enrolled),"encoder":enc},head_path)
        runs.append({"outer_fold":fold,"roles":{k:v.tolist() for k,v in roles.items()},"frozen_encoder":enc,"directions":directions})
    if args.smoke: result={"experiment":CFG["experiment"],"smoke":True,"runs":runs,"test_accessed":False}
    else:
        rows=[r["directions"][d]["outer"] for r in runs for d in r["directions"]]; summary={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k],(int,float))}
        result={"experiment":CFG["experiment"],"runs":runs,"primary_summary":summary,"test":{"AUC":"N/A","F1":"N/A","accessed":False},"competition_test_accessed":False}
    out=E76_DIR/"outputs"; (out/"models").mkdir(parents=True,exist_ok=True); (out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not args.smoke:
        p=result["primary_summary"]; (out/"report.md").write_text("# E76 结构性 unknown-aware/open-set 评估\n\n- 冻结 E74 encoder；头部为 enrolled run0–2 协方差 + run3 prototype Mahalanobis 距离。\n- run4 仅用于 known calibration；calibration unknown 仅诊断，不参与阈值/模型选择。run5 为 outer evaluation。\n\n| 指标 | 数值 |\n|---|---:|\n| Detection AUC | %.6f |\n| Identity macro-F1 | %.6f |\n| Unknown-F1 | %.6f |\n| Test AUC | N/A |\n| Test F1 | N/A |\n\nE74 encoder 未重复训练；未访问 Test。\n"%(p["AUROC"],p["identity_macro_F1"],p["unknown_F1"]),encoding="utf-8")

if __name__=="__main__": main()
