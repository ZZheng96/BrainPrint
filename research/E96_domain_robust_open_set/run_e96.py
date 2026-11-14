from __future__ import annotations
import argparse, json, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, f1_score

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'research'/'E54_outlier_exposure'))
import run_e54 as E54
CFG_PATH=OUT/'config.json'; CFG=json.loads(CFG_PATH.read_text(encoding='utf-8'))
E95=ROOT/'research'/'E95_e54_final20_candidate'; E73=ROOT/'research'/'E73_external_multisession'

SRC_XY=np.asarray([[0,.75],[-.35,.38],[-.18,.40],[0,.40],[.18,.40],[.35,.38],[-.70,0],[-.38,0],[-.18,0],[0,0],[.18,0],[.38,0],[.70,0],[-.35,-.38],[-.18,-.40],[0,-.42],[.18,-.40],[.35,-.38],[-.18,-.72],[0,-.75],[.18,-.72],[0,-.98]],np.float32)
TGT_XY=np.asarray([[-.38,-.72],[-.38,0],[-.38,.75],[0,.75],[.38,.75],[.38,0],[.38,-.72],[0,0],[-.38,1.12],[.38,1.12],[-.98,0],[-.98,-.65],[-.38,-1.08],[.38,-1.08],[-.72,.82],[.72,.82],[.98,-.65],[.98,0]],np.float32)
def projection(device):
 d=np.sqrt(((TGT_XY[:,None]-SRC_XY[None])**2).sum(-1)); w=np.zeros_like(d)
 for j in range(18):
  ix=np.argsort(d[j])[:4]
  w[j,ix]=1/np.maximum(d[j,ix],1e-8)**2; w[j]/=w[j].sum()
 return torch.as_tensor(w,device=device)

def project(x,p): return torch.einsum('ij,bjt->bit',p,x)
def infer(model,raw,classes,batch=256):
 out=[]
 with torch.no_grad():
  for i in range(0,len(raw),batch): out.append(model.scores(raw[i:i+batch]).detach().cpu().numpy())
 return np.concatenate(out)
def quality(raw,fs=300):
 x=raw.float(); x=x-x.mean(-1,keepdim=True); rms=x.square().mean((-1,-2)).sqrt().clamp_min(1e-8)
 spec=torch.fft.rfft(x,dim=-1).abs().square().mean(1); f=torch.fft.rfftfreq(x.shape[-1],1/fs,device=x.device)
 bands=[]
 for lo,hi in zip(CFG['quality']['band_edges_hz'][:-1],CFG['quality']['band_edges_hz'][1:]): bands.append(spec[:,(f>=lo)&(f<hi)].mean(1))
 b=torch.stack(bands,1); rel=b/(b.sum(1,keepdim=True)+1e-8)
 entropy=-(rel*(rel+1e-8).log()).sum(1)/np.log(len(bands))
 return torch.stack([rms.log(),entropy],1).detach().cpu().numpy()
def norm_quality(q, ref):
 med=np.median(ref,0); mad=np.median(np.abs(ref-med),0)*1.4826+1e-6
 z=np.abs((q-med)/mad); return np.exp(-np.minimum(z.mean(1),8)/2)
def load_external(device):
 raws=[]; subs=[]; sess=[]; runs=[]
 root=E73/CFG['source']['external_root'].split('research/E73_external_multisession/',1)[-1]
 for s in range(1,10):
  for si,se in enumerate(('T','E')):
   p=root/f'subject_{s:02d}'/f'A{s:02d}{se}_300Hz_1-40Hz_4s.npz'
   with np.load(p,allow_pickle=False) as z:
    x=np.asarray(z['data'],np.float32); raws.append(x); subs.append(np.full(len(x),s)); sess.append(np.full(len(x),si)); runs.append(z['run_index'].astype(int))
 return torch.as_tensor(np.concatenate(raws),device=device),np.concatenate(subs),np.concatenate(sess),np.concatenate(runs)
def choose_params(known_score, pseudo_score, external_score, known_labels, pred_labels, cfg):
 best=None
 for q in cfg['calibration']['threshold_quantiles']:
  th=float(np.quantile(known_score,q)); ka=known_score>=th; pu=pseudo_score>=th; eu=external_score>=th
  if ka.mean()<cfg['calibration']['known_acceptance_target']: continue
  value=(1-pu.mean())+(1-eu.mean())+0.1*ka.mean()
  cand=(value,th,float(ka.mean()),float(pu.mean()),float(eu.mean()))
  if best is None or cand>best: best=cand
 if best is None:
  th=float(np.quantile(known_score,.01)); best=(0,th,float((known_score>=th).mean()),float((pseudo_score>=th).mean()),float((external_score>=th).mean()))
 _,th,ka,pfar,efar=best
 return {'threshold':th,'known_acceptance':ka,'pseudo_unknown_FAR':pfar,'external_unknown_FAR':efar,'selection_source':'Train pseudo-unknown plus E73 external calibration; no project Test'}
def apply(scores,q,params,classes):
 c=.7*scores.max(1)+.3*q; ix=scores.argmax(1); accept=c>=params['threshold']; return np.where(accept,classes[ix],0),c,accept

def main(smoke=False,full=False):
 if not torch.cuda.is_available(): raise RuntimeError('E96 requires CUDA')
 device=torch.device('cuda'); e54=E54.load_config(); e95cfg=json.loads((E95/'config.json').read_text(encoding='utf-8')); meta=E54.train_metadata(e54); raw=E54.load_all_raw(meta,e54,device)
 classes=np.arange(1,21); model=E54.EEGNetEmbedding(20,e54).to(device)
 ck=torch.load(E95/'model_checkpoint.pt',map_location=device,weights_only=False); model.load_state_dict(ck['state_dict']); model.eval()
 p=projection(device); frac=meta.trial.to_numpy(float)/np.maximum(meta.groupby(['subject','task']).trial.transform('max').to_numpy(float),1)
 rng=np.random.default_rng(CFG['seed']); pseudo=np.asarray(CFG['calibration']['pseudo_unknown_subjects']); known=np.setdiff1d(classes,pseudo)
 cal_known=np.flatnonzero(np.isin(meta.subject,known)&(frac>.6)&(frac<=.8)); cal_pseudo=np.flatnonzero(np.isin(meta.subject,pseudo)&(frac>.6)&(frac<=.8))
 ext,esub,esess,erun=load_external(device)
 ext18=project(ext,p); ext_scores=infer(model,ext18,classes); ext_q=quality(ext18)
 train_scores=infer(model,raw,classes); train_q=quality(raw)
 refq=train_q[cal_known]; q_known=norm_quality(train_q[cal_known],refq); q_pseudo=norm_quality(train_q[cal_pseudo],refq); q_ext=norm_quality(ext_q,refq)
 gate_known=.7*train_scores[cal_known].max(1)+.3*q_known; gate_pseudo=.7*train_scores[cal_pseudo].max(1)+.3*q_pseudo; gate_ext=.7*ext_scores.max(1)+.3*q_ext
 params=choose_params(gate_known,gate_pseudo,gate_ext,meta.subject.to_numpy()[cal_known],classes,CFG)
 pred_k,_,acc_k=apply(train_scores[cal_known],q_known,params,classes); pred_p,_,acc_p=apply(train_scores[cal_pseudo],q_pseudo,params,classes); pred_e,_,acc_e=apply(ext_scores,q_ext,params,classes)
 labels=np.r_[np.ones(len(gate_known)),np.zeros(len(gate_ext))]; auc=float(roc_auc_score(labels,np.r_[gate_known,gate_ext])); f1=float(f1_score(np.r_[np.ones(len(gate_known)),np.zeros(len(gate_ext))],np.r_[acc_k,acc_e],zero_division=0))
 result={'status':'frozen_before_test' if full else 'complete_external_train_only','experiment':CFG['experiment'],'external':{'N':len(ext),'AUC':auc,'F1':f1,'known_acceptance':float(acc_k.mean()),'unknown_FAR':float(acc_e.mean()),'unknown_rejection':float((~acc_e).mean()),'subjects':9,'source':'E73-derived BCIC IV-2a; projected 22->18 IDW'},'train_calibration':{'N_known':len(cal_known),'N_pseudo_unknown':len(cal_pseudo),'AUC':float(roc_auc_score(np.r_[np.ones(len(gate_known)),np.zeros(len(gate_pseudo))],np.r_[gate_known,gate_pseudo])),'F1':float(f1_score(np.r_[np.ones(len(gate_known)),np.zeros(len(gate_pseudo))],np.r_[acc_k,acc_p],zero_division=0))},'gate':params,'test_metrics':{k:'N/A' for k in ['AUC','F1','C','RA','ASafeR','U','Official Score']},'protocol_checks':{'train_session1_only':True,'external_used_as_unknown_domain':True,'test_accessed':False,'gpu':torch.cuda.get_device_name(0),'e95_checkpoint_reused':True}}
 torch.save({'state_dict':model.state_dict(),'classes':classes,'projection':p.detach().cpu(),'gate':params,'config':CFG},OUT/'model_checkpoint.pt')
 rows=[]
 for i in range(len(ext)): rows.append({'source':'external_E73','sample_index':i,'subject':int(esub[i]),'session':int(esess[i]),'run':int(erun[i]),'score':float(gate_ext[i]),'accepted':int(acc_e[i]),'predicted_identity':int(pred_e[i])})
 if full:
  order=pd.read_csv(ROOT/e95cfg['data']['test_csv'],usecols=['Sample']); mapping=pd.read_csv(ROOT/e95cfg['data']['test_metadata'],usecols=['sample','path','session']); test_preds=[]
  for st in range(0,len(mapping),128):
   arr=np.stack([np.asarray(loadmat(ROOT/e95cfg['data']['train_root']/x,variable_names=['data'])['data'],np.float32).T for x in mapping.path.iloc[st:st+128]])
   x=torch.as_tensor(arr,device=device); s=infer(model,x,classes); qq=quality(x); score=.7*s.max(1)+.3*norm_quality(qq,refq); test_preds.extend((score>=params['threshold']).astype(int) * classes[s.argmax(1)] .tolist())
  # preserve 0 rejection; avoid any session-dependent inference
  pd.DataFrame({'Sample':order.Sample,'Label':test_preds}).to_csv(OUT/'predictions.csv',index=False); result['protocol_checks'].update({'test_accessed':True,'test_read_once_after_freeze':True}); result['status']='complete_one_shot_frozen_audit'; m=mapping.session.astype(str).str.casefold(); result['audit']={n:{'N':int((m.str.contains('session1' if n=='S1' else 'session2')).sum()),'rejection_rate':float((np.asarray(test_preds)[m.str.contains('session1' if n=='S1' else 'session2')]==0).mean()),'acceptance_rate':float((np.asarray(test_preds)[m.str.contains('session1' if n=='S1' else 'session2')]!=0).mean())} for n in ('S1','S2')}
 else:
  pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 report=f"# E96 domain robust open set\n\n- 冻结 E95 raw identity encoder；E73 22 通道源数据按 IDW(k=4,p=2)投影为 18 通道。外部 BCIC IV-2a 只作为独立域 unknown/校准负例，未读取项目 Test 进行调参。\n\n## Train / 外部\n\n| 数据 | AUC | F1 | known acceptance | unknown FAR | unknown rejection |\n|---|---:|---:|---:|---:|---:|\n| Train pseudo-unknown calibration | {result['train_calibration']['AUC']:.6f} | {result['train_calibration']['F1']:.6f} | {params['known_acceptance']:.6f} | {params['pseudo_unknown_FAR']:.6f} | {1-params['pseudo_unknown_FAR']:.6f} |\n| E73 external | {auc:.6f} | {f1:.6f} | {result['external']['known_acceptance']:.6f} | {result['external']['unknown_FAR']:.6f} | {result['external']['unknown_rejection']:.6f} |\n\n## 项目 Test\n\n真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。S1/S2 仅在模型和参数冻结后一次性读取并做工程审计：S1 拒识率={result.get('audit',{}).get('S1',{}).get('rejection_rate','N/A')}，S2 接受率={result.get('audit',{}).get('S2',{}).get('acceptance_rate','N/A')}。\n"
 (OUT/'report.md').write_text(report,encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--full',action='store_true'); a=ap.parse_args();
 if a.smoke==a.full: ap.error('choose --smoke or --full')
 main(a.smoke,a.full)
