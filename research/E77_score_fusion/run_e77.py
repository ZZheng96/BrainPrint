"""E77: preregistered EEG-only cosine/Mahalanobis score fusion."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

E77_DIR=Path(__file__).resolve().parent; ROOT=E77_DIR.parents[1]
E74_DIR=ROOT/"research"/"E74_multiscale_native22"; E73_DIR=ROOT/"research"/"E73_external_multisession"
sys.path.insert(0,str(E74_DIR)); sys.path.insert(0,str(E73_DIR))
from external_data import load_external_data, subject_folds
from multiscale_model import MultiScaleNative22
CFG=json.loads((E77_DIR/"config.json").read_text(encoding="utf-8")); BASE=json.loads((E74_DIR/"config.json").read_text(encoding="utf-8-sig"))

def sha256(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def embed(model,raw,indices,batch=256):
    out=[]
    with torch.no_grad():
        for i in range(0,len(indices),batch): out.append(model.embed(raw[torch.as_tensor(indices[i:i+batch],dtype=torch.long,device=raw.device)]).float())
    return torch.cat(out) if out else torch.empty((0,64),device=raw.device)

def load_encoder(data,roles,fold):
    p=E74_DIR/"outputs"/"models"/f"supcon_outer{fold}.pt"
    saved=torch.load(p,map_location=data.raw.device,weights_only=False)
    expected={"config":sha256(E74_DIR/"config.json"),"run_e74":sha256(E74_DIR/"run_e74.py"),"external_data":sha256(E74_DIR/"external_data.py"),"multiscale_model":sha256(E74_DIR/"multiscale_model.py")}
    if saved.get("source_hashes")!=expected or saved.get("roles")!={k:v.tolist() for k,v in roles.items()}: raise ValueError("Frozen E74 checkpoint provenance mismatch")
    model=MultiScaleNative22(BASE).to(data.raw.device); model.load_state_dict(saved["model_state"]); model.eval()
    for p0 in model.parameters(): p0.requires_grad_(False)
    return model,{"path":str(p),"sha256":sha256(p)}

def fit_head(train_x,train_y,template_x,template_y,classes):
    means=torch.stack([template_x[torch.as_tensor(template_y==c,device=template_x.device)].mean(0) for c in classes])
    residual=train_x-torch.stack([means[list(classes).index(int(y))] for y in train_y])
    cov=(residual.T@residual)/max(1,len(residual)-1); sh=float(CFG["head"]["shrinkage"])
    cov=(1-sh)*cov+sh*torch.diag(torch.diag(cov)); eye=torch.eye(cov.shape[0],device=cov.device)
    return means,torch.linalg.pinv(cov+float(CFG["head"]["eigenvalue_floor"])*eye)

def mahal(x,means,inv):
    d=x[:,None,:]-means[None,:,:]; return -torch.einsum("nkd,df,nkf->nk",d,inv,d).cpu().numpy()

def cosine(x,means):
    return (torch.nn.functional.normalize(x,dim=1)@torch.nn.functional.normalize(means,dim=1).T).cpu().numpy()

def fit_normalizer(cos,mah,labels,classes):
    idx=np.asarray([list(classes).index(int(y)) for y in labels]); c=cos[np.arange(len(idx)),idx]; m=mah[np.arange(len(idx)),idx]
    return {"cos_mean":float(c.mean()),"cos_std":float(c.std()+CFG["fusion"]["epsilon"]),"mah_mean":float(m.mean()),"mah_std":float(m.std()+CFG["fusion"]["epsilon"])}

def fuse(cos,mah,norm):
    c=(cos-norm["cos_mean"])/norm["cos_std"]; m=(mah-norm["mah_mean"])/norm["mah_std"]
    w=CFG["fusion"]["weights"]; return float(w["cosine_prototype"])*c+float(w["mahalanobis"])*m

def metric(known,unknown,true,classes,threshold):
    ks=known.max(1); us=unknown.max(1); accept=ks>=threshold; pred=np.where(accept,classes[known.argmax(1)],0)
    labels=np.r_[np.ones(len(ks)),np.zeros(len(us))]; det=np.r_[accept,us>=threshold]
    return {"N_known":len(ks),"N_unknown":len(us),"threshold":float(threshold),"AUROC":float(roc_auc_score(labels,np.r_[ks,us])),"identity_macro_F1":float(f1_score(true,pred,labels=classes,average="macro",zero_division=0)),"unknown_F1":float(f1_score(labels,det,pos_label=0,zero_division=0)),"known_acceptance":float(accept.mean()),"unknown_false_acceptance":float((us>=threshold).mean()),"unknown_rejection":float((us<threshold).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("E77 formal evaluation requires CUDA")
    device=torch.device("cuda"); data=load_external_data(BASE,device); folds=subject_folds(BASE)[:1] if args.smoke else subject_folds(BASE); runs=[]
    for fold,roles in enumerate(folds,1):
        model,enc=load_encoder(data,roles,fold); directions={}; enrolled=roles["enrolled"]
        for source,target in (("T","E"),("E","T")):
            train_i=data.indices(enrolled,session=source,runs=[0,1,2]); train_j=data.indices(enrolled,session=target,runs=[0,1,2]); templ_i=data.indices(enrolled,session=source,runs=[3]); known_i=data.indices(enrolled,session=target,runs=[4]); cal_u=data.indices(roles["calibration_unknown"],session=target,runs=[5]); outer_i=data.indices(enrolled,session=target,runs=[5]); outer_u=data.indices(roles["outer_unknown"],session=target,runs=[5])
            tx,ex,template=embed(model,data.raw,train_i),embed(model,data.raw,train_j),embed(model,data.raw,templ_i); ty,ey,template_y=data.subject[train_i],data.subject[train_j],data.subject[templ_i]
            means,inv=fit_head(torch.cat([tx,ex]),np.r_[ty,ey],template,template_y,enrolled); train_cos=cosine(torch.cat([tx,ex]),means); train_mah=mahal(torch.cat([tx,ex]),means,inv); norm=fit_normalizer(train_cos,train_mah,np.r_[ty,ey],enrolled)
            known=fuse(cosine(embed(model,data.raw,known_i),means),mahal(embed(model,data.raw,known_i),means,inv),norm); cal=fuse(cosine(embed(model,data.raw,cal_u),means),mahal(embed(model,data.raw,cal_u),means,inv),norm)
            thresholds={f"q{int(q*100):02d}":float(np.quantile(known.max(1),q)) for q in CFG["threshold"]["quantiles"]}; inner={k:metric(known,cal,data.subject[known_i],enrolled,v) for k,v in thresholds.items()}
            if args.smoke: outer={"q05":"not_accessed_in_smoke","q02":"not_accessed_in_smoke"}
            else:
                ox=embed(model,data.raw,outer_i); ou=embed(model,data.raw,outer_u); outer_known=fuse(cosine(ox,means),mahal(ox,means,inv),norm); outer_unknown=fuse(cosine(ou,means),mahal(ou,means,inv),norm); outer={k:metric(outer_known,outer_unknown,data.subject[outer_i],enrolled,v) for k,v in thresholds.items()}
            hp=E77_DIR/"outputs"/"models"/f"fusion_outer{fold}_{source}_to_{target}.pt"; hp.parent.mkdir(parents=True,exist_ok=True); torch.save({"means":means.detach().cpu(),"inverse_covariance":inv.detach().cpu(),"classes":enrolled,"normalizer":norm,"weights":CFG["fusion"]["weights"],"encoder":enc},hp)
            directions[f"{source}_to_{target}"]={"inner":inner,"outer":outer,"checkpoint":str(hp),"calibration_unknown_diagnostic_only":True}
        runs.append({"outer_fold":fold,"roles":{k:v.tolist() for k,v in roles.items()},"frozen_encoder":enc,"directions":directions})
    result={"experiment":CFG["experiment"],"smoke":args.smoke,"runs":runs,"test_accessed":False}
    if not args.smoke:
        rows=[r["directions"][d]["outer"]["q05"] for r in runs for d in r["directions"]]; result["primary_summary"]={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k],(int,float))}; result["gate_summary"]={"S1_known_acceptance_ge_95":result["primary_summary"]["known_acceptance"]>=.95,"S2_unknown_false_acceptance_le_2":result["primary_summary"]["unknown_false_acceptance"]<=.02}
    out=E77_DIR/"outputs"; out.mkdir(exist_ok=True); (out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not args.smoke and result.get("primary_summary"):
        p=result["primary_summary"]; g=result["gate_summary"]; (out/"report.md").write_text(f"# E77 EEG-only score fusion\n\n冻结 E74 encoder，复用 E76 Mahalanobis head；未使用 E75，未访问 competition Test。融合权重预注册为 cosine 0.5 / Mahalanobis 0.5；阈值仅由 inner known run4 的 q05/q02 决定，calibration unknown 仅诊断。\n\n| 指标 | 三折两方向 q05 均值 |\n|---|---:|\n| AUROC | {p['AUROC']:.6f} |\n| Identity macro-F1 | {p['identity_macro_F1']:.6f} |\n| Unknown-F1 | {p['unknown_F1']:.6f} |\n| S1 known acceptance | {p['known_acceptance']:.6f} (门禁 {'通过' if g['S1_known_acceptance_ge_95'] else '失败'}) |\n| S2 unknown false acceptance | {p['unknown_false_acceptance']:.6f} (门禁 {'通过' if g['S2_unknown_false_acceptance_le_2'] else '失败'}) |\n\nq02、逐折结果和 calibration unknown 诊断见 results.json。若门禁失败，结论为未达成，不扫阈值、不伪造。\n",encoding="utf-8")
if __name__=="__main__": main()
