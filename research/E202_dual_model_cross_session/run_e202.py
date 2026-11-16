from __future__ import annotations
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from model_a import train_model, embed
from model_b import UnknownGate, rejection_features
from protocol import fit_mask, eval_mask, fit_projection, build_prototypes, sample_splits

BASE=ROOT.parents[1]
META=BASE/'BrainPrint_cache'/'train_metadata.csv'
E65=BASE/'BrainPrint_cache'/'e65_correlation_spd'/'train_correlation_logspd.npy'
E162=ROOT.parent/'E162_raw_eeg_contrastive_v2'/'train_emb.npy'

def load_data():
    meta=pd.read_csv(META)
    # Existing Train metadata has no published frac; trial order is the only Train-only time coordinate.
    meta['frac']=meta['trial'].to_numpy(dtype=float)/meta.groupby(['subject','task'])['trial'].transform('max').to_numpy()
    subjects=meta.subject.to_numpy(int)
    tasks=meta.task.to_numpy(str)
    x65=np.load(E65,mmap_mode='r').astype(np.float32)
    x162=np.load(E162,mmap_mode='r').astype(np.float32)
    if len(meta)!=len(x65) or len(meta)!=len(x162): raise RuntimeError('feature/metadata row mismatch')
    return meta, subjects, tasks, x65, x162

def norm_rows(x):
    x=x.astype(np.float32); return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)

def _auc(y, p):
    order=np.argsort(p)
    ranks=np.empty(len(p),dtype=float); ranks[order]=np.arange(1,len(p)+1)
    pos=y==1; neg=~pos
    return (ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())

def _f1(y, pred):
    tp=np.sum((y==1)&pred); fp=np.sum((y==0)&pred); fn=np.sum((y==1)&~pred)
    return float(2*tp/max(1,2*tp+fp+fn))

def one_fold(meta, subjects, x65, x162, unknown_eval, k, device, epochs, seed):
    ids=np.arange(1,21); enrolled=np.array([s for s in ids if s not in set(unknown_eval)],dtype=int)
    fit=fit_mask(meta,subjects,enrolled)
    scaler,pca,x65p=fit_projection(x65,fit,96)
    model,z_fit,_=train_model(x65p[fit],x162[fit],subjects[fit]-1,20,device=device,epochs=epochs,seed=seed)
    torch.save({'state_dict':model.state_dict(),'e65_dim':int(x65p.shape[1]),'e162_dim':int(x162.shape[1]),'seed':seed}, ROOT/'checkpoints'/('model_a_k%d_%s.pt' % (k, '_'.join(map(str,unknown_eval)))))
    z=embed(model,x65p,x162,device=device)
    proto, proto_ids=build_prototypes(z,subjects,fit,enrolled)
    feat,pred_idx,best=rejection_features(z,proto,x65,x162)
    candidate_rows=(meta['frac'].to_numpy()>0.6)&(meta['frac'].to_numpy()<=0.8)
    # The gate's unknown training identities are an inner split of the enrolled pool.
    # Their prototype is excluded while making gate features; final unknown_eval is never used.
    inner_unknown=enrolled[:max(1,len(enrolled)//5)]
    inner_registered=np.array([s for s in enrolled if s not in set(inner_unknown)],dtype=int)
    inner_fit=fit_mask(meta,subjects,inner_registered)
    inner_proto,_=build_prototypes(z,subjects,inner_fit,inner_registered)
    inner_feat,_,_=rejection_features(z,inner_proto,x65,x162)
    gate_train_known=candidate_rows & np.isin(subjects,inner_registered) & ~inner_fit
    gate_train_unknown=candidate_rows & np.isin(subjects,inner_unknown)
    gate_eval=candidate_rows & np.isin(subjects,unknown_eval)
    train_rows=gate_train_known|gate_train_unknown
    labels=np.isin(subjects[train_rows],inner_registered).astype(int)
    gate=UnknownGate().fit(inner_feat[train_rows],labels)
    np.savez(ROOT/'bundle'/('gate_k%d_%s.npz' % (k, '_'.join(map(str,unknown_eval)))), mean=gate.mean, std=gate.std, w=gate.w, b=np.asarray([gate.b]), tau=np.asarray([gate.tau]), enrolled=enrolled, unknown_eval=unknown_eval)
    p=gate.predict_known(feat)
    pred_ids=proto_ids[pred_idx]
    known_eval=candidate_rows & np.isin(subjects,enrolled)
    accept_known=(p>=gate.tau)&known_eval
    unknown_rej=float(np.mean(p[gate_eval]<gate.tau)) if gate_eval.any() else float('nan')
    known_acc=float(np.mean(p[known_eval]>=gate.tau)) if known_eval.any() else float('nan')
    id_acc=float(np.mean(pred_ids[known_eval]==subjects[known_eval])) if known_eval.any() else float('nan')
    e_rows=gate_eval|known_eval
    y=np.isin(subjects[e_rows],enrolled).astype(int)
    auc=float(_auc(y,p[e_rows])) if len(np.unique(y))==2 else float('nan')
    f1=float(_f1(y,p[e_rows]>=gate.tau)) if len(np.unique(y))==2 else float('nan')
    return {'k':k,'unknown_eval':unknown_eval.tolist(),'enrolled':enrolled.tolist(),'unknown_train':inner_unknown.tolist(),'unknown_cal':[],'known_accept':known_acc,'unknown_rejection':unknown_rej,'registration_id_accuracy':id_acc,'auc':auc,'f1':f1,'n_eval':int(e_rows.sum()),'tau_b':float(gate.tau),'device':device}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--max-folds',type=int,default=1); ap.add_argument('--k',type=int,nargs='+',default=[2,5,10]); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    if args.smoke: args.epochs=1; args.max_folds=1; args.k=[2]
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    out=ROOT/'results.json'; meta,subjects,tasks,x65,x162=load_data(); results=[]
    for k in args.k:
        for unknown in sample_splits(range(1,21),k)[:args.max_folds]:
            print('fold',k,unknown.tolist(),flush=True)
            results.append(one_fold(meta,subjects,x65,x162,unknown,k,args.device,args.epochs,20261116+k+len(results)))
    payload={'experiment':'E202_dual_model_cross_session','protocol':'Train-only; no Test access; prototype fit rows exclude evaluation identities','results':results}
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
