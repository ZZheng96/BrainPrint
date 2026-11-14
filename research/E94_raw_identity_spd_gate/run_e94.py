"""E94 raw identity confidence with Train-only raw/SPD rejection gate."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
from run_e36 import load_config as load_e36_config, load_train_matrix, train_metadata
import run_e54 as E54
import run_e65 as E65

CFG_PATH = OUT / "config.json"; RESULTS = OUT / "results.json"; REPORT = OUT / "report.md"
CHECKPOINT = OUT / "model_checkpoint.pt"; PRED = OUT / "predictions.csv"

class EEGNet1d(nn.Module):
    def __init__(self, channels, classes, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels,width,9,2,4), nn.BatchNorm1d(width), nn.ELU(),
            nn.Conv1d(width,width*2,9,2,4), nn.BatchNorm1d(width*2), nn.ELU(),
            nn.Conv1d(width*2,width*4,9,2,4), nn.BatchNorm1d(width*4), nn.ELU(),
            nn.Conv1d(width*4,width*4,9,2,4), nn.BatchNorm1d(width*4), nn.ELU(), nn.AdaptiveAvgPool1d(1))
        self.drop = nn.Dropout2d(.10); self.head = nn.Linear(width*4, classes)
    def forward(self, x):
        x = x - x.mean(-1, keepdim=True); x = x / x.flatten(1).std(1).view(-1,1,1).clamp_min(1e-6)
        return self.head(self.net(self.drop(x)).squeeze(-1))

def augment(x, cfg, gen):
    p = cfg["model"]["channel_dropout"]
    x = x * (torch.rand((len(x),x.shape[1],1), device=x.device, generator=gen)>p).to(x.dtype)
    shift = cfg["training"]["shift_samples"]
    offsets = torch.randint(-shift, shift+1, (len(x),), device=x.device, generator=gen)
    x = torch.stack([torch.roll(v,int(s),-1) for v,s in zip(x,offsets)])
    return x * (1 + cfg["training"]["amplitude_std"] * torch.randn((len(x),1,1), device=x.device, generator=gen))

def train_model(raw, idx, y, classes, cfg, device, epochs):
    torch.manual_seed(cfg["seed"]); model=EEGNet1d(18,len(classes),cfg["model"]["width"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    gen=torch.Generator(device=device); gen.manual_seed(cfg["seed"]); rng=np.random.default_rng(cfg["seed"])
    for ep in range(epochs):
        model.train(); order=rng.permutation(len(idx)); last=0
        for st in range(0,len(order),cfg["training"]["batch_size"]):
            sel=torch.as_tensor(order[st:st+cfg["training"]["batch_size"]],device=device)
            x=augment(raw[torch.as_tensor(idx,device=device)[sel]],cfg,gen); yy=y[sel]
            logits=model(x); loss=F.cross_entropy(logits,yy); opt.zero_grad(); loss.backward(); opt.step(); last=float(loss)
        print(f"epoch={ep+1}/{epochs} loss={last:.4f}",flush=True)
    return model

def raw_scores(model, raw, idx, device):
    out=[]; model.eval()
    with torch.no_grad():
        for st in range(0,len(idx),512):
            q=raw[torch.as_tensor(idx[st:st+512],device=device)]; out.append(torch.softmax(model(q),1).cpu().numpy())
    return np.concatenate(out)

def spd_scores(raw, idx, e36, e65cfg, device, prototypes):
    out=[]
    with torch.no_grad():
        for st in range(0,len(idx),128):
            feat=E65.gpu_correlation_logspd(raw[torch.as_tensor(idx[st:st+128],device=device)],e36,e65cfg)
            p=torch.as_tensor(prototypes,device=device); q=feat
            out.append((-torch.cdist(q,p)).cpu().numpy())
    return np.concatenate(out)

def bounds(v):
    lo,hi=np.quantile(v,[.01,.99]); return float(lo),float(max(hi-lo,1e-8))
def scale(v,b): return np.clip((v-b[0])/b[1],0,1)

def select_gate(rawp, spdp, y, unknown_rawp, unknown_spdp, classes):
    rc,ru=rawp.max(1),unknown_rawp.max(1); sc,su=spdp.max(1),unknown_spdp.max(1)
    rb,sb=bounds(np.r_[rc,ru]),bounds(np.r_[sc,su]); rn,un=scale(rc,rb),scale(ru,rb); sn,us=scale(sc,sb),scale(su,sb)
    best=None
    for mode in ("raw","spd","weighted","or"):
        for q in [.01,.02,.05,.10,.20]:
            if mode=="raw": fk,fu=rn,un
            elif mode=="spd": fk,fu=sn,us
            elif mode=="weighted": fk,fu=.7*rn+.3*sn,.7*un+.3*us
            else: fk,fu=rn,un
            t=float(np.quantile(fk,q)); pred=np.where(fk>=t,classes[rawp.argmax(1)],0)
            if mode=="or": pred=np.where((rn>=t)|(sn>=t),classes[rawp.argmax(1)],0); accepted=(fu>=t)|(us>=t)
            else: accepted=fu>=t
            det=np.r_[np.ones(len(pred),int),np.zeros(len(accepted),int)]; acc=np.r_[pred!=0,accepted]
            val=f1_score(det,acc,pos_label=0,zero_division=0)+.1*f1_score(y,pred,labels=classes,average="macro",zero_division=0)
            if best is None or val>best[0]: best=(val,{"mode":mode,"raw_bounds":rb,"spd_bounds":sb,"threshold":t,"q":q})
    return best[1]

def apply_gate(rp,sp,p,classes):
    rn=scale(rp.max(1),p["raw_bounds"]); sn=scale(sp.max(1),p["spd_bounds"]); t=p["threshold"]
    if p["mode"]=="raw": ok=rn>=t
    elif p["mode"]=="spd": ok=sn>=t
    elif p["mode"]=="weighted": ok=.7*rn+.3*sn>=t
    else: ok=(rn>=t)|(sn>=t)
    return np.where(ok,classes[rp.argmax(1)],0), np.maximum(rn,sn)

def run(smoke=False):
    cfg=json.loads(CFG_PATH.read_text(encoding="utf-8")); assert torch.cuda.is_available(), "E94 requires CUDA"
    device=torch.device("cuda"); e36=load_e36_config(); e54cfg=E54.load_config(); meta=train_metadata(e36)
    raw=E54.load_all_raw(meta,e54cfg,device); subjects=meta.subject.to_numpy(int); trial=meta.trial.to_numpy(float)
    maxtrial=meta.groupby(["subject","task"]).trial.transform("max").to_numpy(float); frac=trial/np.maximum(maxtrial,1)
    classes=np.arange(1,21); fit=np.flatnonzero(frac<=.60); cal=np.flatnonzero((frac>.60)&(frac<=.80))
    # Inner Train-only calibration: reserve half of calibration subjects, never Test/session fields.
    rng=np.random.default_rng(cfg["protocol"]["inner_seed"]); csubjects=rng.permutation(classes); cal_unknown=csubjects[:4]; cal_known=np.setdiff1d(classes,cal_unknown)
    cal_k=np.flatnonzero(np.isin(subjects,cal_known)&(frac>.60)&(frac<=.80)); cal_u=np.flatnonzero(np.isin(subjects,cal_unknown)&(frac>.0)&(frac<=.80))
    epochs=cfg["training"]["smoke_epochs"] if smoke else cfg["training"]["epochs"]
    lookup={s:i for i,s in enumerate(classes)}; y=torch.as_tensor([lookup[s] for s in subjects[fit]],device=device)
    model=train_model(raw,fit,y,classes,cfg,device,epochs)
    e65cfg=json.loads((ROOT/"research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    with torch.no_grad(): fit_feat=E65.gpu_correlation_logspd(raw[torch.as_tensor(fit,device=device)],e36,e65cfg).cpu().numpy()
    prot=np.stack([fit_feat[subjects[fit]==c].mean(0) for c in classes]).astype(np.float32)
    rpk,rpu=raw_scores(model,raw,cal_k,device),raw_scores(model,raw,cal_u,device)
    spk,spu=spd_scores(raw,cal_k,e36,e65cfg,device,prot),spd_scores(raw,cal_u,e36,e65cfg,device,prot)
    params=select_gate(rpk,spk,subjects[cal_k],rpu,spu,classes)
    torch.save({"state_dict":model.state_dict(),"classes":classes,"spd_prototypes":prot,"params":params,"config":cfg},CHECKPOINT)
    train_pred,_=apply_gate(rpk,spk,params,classes)
    train_metrics={"AUC":float(roc_auc_score(np.r_[np.ones(len(rpk)),np.zeros(len(rpu))],np.r_[rpk.max(1),rpu.max(1)])),"F1":float(f1_score(subjects[cal_k],train_pred,labels=classes,average="macro",zero_division=0)),"C":float(np.mean(train_pred==subjects[cal_k])),"U":float(np.mean(apply_gate(rpu,spu,params,classes)[0]==0))}
    if smoke:
        result={"status":"complete_smoke_train_only","experiment":"E94_raw_identity_spd_gate","train_only_metrics":train_metrics,"gate_params":params,"test_metrics":{k:"N/A" for k in ("AUC","F1","C","RA","ASafeR","U","Official Score")},"protocol_checks":{"train_session1_only":True,"train_only_calibration":True,"gpu":torch.cuda.get_device_name(0),"test_accessed":False}}
        RESULTS.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x),encoding="utf-8")
        REPORT.write_text("# E94 raw identity SPD gate\n\nSmoke 仅运行 Train/session1；未读取 Test。\n\nTrain-only AUC=%.6f，F1=%.6f，C=%.6f，U=%.6f。\n\nTest AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n"%(train_metrics["AUC"],train_metrics["F1"],train_metrics["C"],train_metrics["U"]),encoding="utf-8")
        return result
    # Freeze checkpoint/parameters before the sole Test read.
    test_csv=ROOT/cfg["data"]["test_csv"]; test_meta=ROOT/cfg["data"]["test_metadata"]
    order=pd.read_csv(test_csv,usecols=["Sample"]); mapping=pd.read_csv(test_meta,usecols=["sample","path"])
    if len(order)!=cfg["data"]["expected_test_rows"] or len(mapping)!=len(order): raise ValueError("Test mapping mismatch")
    test_root=ROOT/"Brainprint_Competition"; labels=[]
    for st in range(0,len(mapping),128):
        arr=np.stack([load_train_matrix(test_root/p) for p in mapping.path.iloc[st:st+128]])
        tx=torch.as_tensor(arr,dtype=torch.float32,device=device); rp=torch.softmax(model(tx),1).detach().cpu().numpy()
        sf=E65.gpu_correlation_logspd(tx,e36,e65cfg); sp=(-torch.cdist(sf,torch.as_tensor(prot,device=device))).detach().cpu().numpy()
        labels.extend(apply_gate(rp,sp,params,classes)[0].tolist())
    pd.DataFrame({"Sample":order.Sample,"Label":labels}).to_csv(PRED,index=False)
    written=pd.read_csv(PRED); audit_meta=pd.read_csv(test_meta,usecols=["sample","session"])
    rej=written.Label.to_numpy()==0; sessions=audit_meta.session.astype(str).str.casefold().map({"session1":"S1","s1":"S1","1":"S1","session2":"S2","s2":"S2","2":"S2"}).to_numpy()
    audit={"N":len(written),"overall_rejection_rate":float(rej.mean()),"session_used_by_inference":False,"audit_after_predictions_csv":True}
    for s in ("S1","S2"):
        m=sessions==s; audit[f"{s}_N"]=int(m.sum()); audit[f"{s}_rejection_rate"]=float(rej[m].mean()); audit[f"{s}_acceptance_rate"]=float((~rej[m]).mean())
    det=np.r_[np.ones(len(rpk)),np.zeros(len(rpu))]; pred,_=apply_gate(rpk,spk,params,classes); accepted=np.r_[pred!=0, apply_gate(rpu,spu,params,classes)[0]!=0]
    metrics={"AUC":float(roc_auc_score(det,np.r_[rpk.max(1),rpu.max(1)])),"F1":float(f1_score(subjects[cal_k],pred,labels=classes,average="macro",zero_division=0)),"C":float(np.mean(pred==subjects[cal_k])),"U":float(np.mean(~accepted[len(pred):]))}
    result={"status":"complete_smoke" if smoke else "complete_one_shot_frozen_audit","experiment":"E94_raw_identity_spd_gate","train_only_metrics":metrics,"gate_params":params,"audit":audit,"test_metrics":{k:"N/A" for k in ("AUC","F1","C","RA","ASafeR","U","Official Score")},"protocol_checks":{"train_session1_only":True,"train_only_calibration":True,"gpu":torch.cuda.get_device_name(0),"test_frozen_before_read":True,"predictions_complete_before_session_audit":True}}
    RESULTS.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x),encoding="utf-8")
    REPORT.write_text("# E94 raw identity SPD gate\n\nTrain/session1 内层伪未知校准 raw confidence 与 correlation-logSPD confidence；冻结模型、参数后完成唯一一次 Test 推理。Session 仅在完整 predictions.csv 落盘后审计。\n\n## Train-only 指标\n\n| AUC | F1 | C | U |\n|---:|---:|---:|---:|\n| %.6f | %.6f | %.6f | %.6f |\n\n## Test 审计\n\nTest 无真实标签：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n| 项目 | 结果 | 目标 | 是否达标 |\n|---|---:|---:|---|\n| S1 拒识率 | %.6f | >=0.95 | %s |\n| S2 接受率 | %.6f | >=0.98 | %s |\n\n拒识规则：%s；阈值和候选均未使用 Test session。\n"%(metrics["AUC"],metrics["F1"],metrics["C"],metrics["U"],audit["S1_rejection_rate"],"通过" if audit["S1_rejection_rate"]>=.95 else "未通过",audit["S2_acceptance_rate"],"通过" if audit["S2_acceptance_rate"]>=.98 else "未通过",params["mode"]),encoding="utf-8")
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--full",action="store_true"); a=ap.parse_args()
    if a.smoke==a.full: ap.error("choose --smoke or --full")
    print(json.dumps(run(a.smoke),ensure_ascii=False,indent=2))
