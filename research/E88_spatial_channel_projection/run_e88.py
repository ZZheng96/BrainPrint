"""E88 external-only standard 10-20 spatial 22-to-18 projection and hard-unknown evaluation."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; E73=ROOT/'research'/'E73_external_multisession'
sys.path.insert(0,str(E73)); sys.path.insert(0,str(HERE))
from external_data import load_external_data, subject_folds
from e88_model import E88Spatial18Embedding
CFG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def projection_matrix():
 p=CFG['channel_projection']; src=p['source_channels_22']; tgt=p['target_channels_18']; sx=np.asarray([p['source_xy'][n] for n in src],dtype=np.float32); tx=np.asarray([p['target_xy'][n] for n in tgt],dtype=np.float32)
 d=np.sqrt(((tx[:,None,:]-sx[None,:,:])**2).sum(-1)); k=min(int(p['k']),len(src)); w=np.zeros_like(d)
 for j in range(len(tgt)):
  nn=np.argsort(d[j])[:k]
  if p['method']=='nearest': w[j,nn[0]]=1.0
  elif p['method']=='idw':
   if d[j,nn[0]]<1e-8: w[j,nn[0]]=1.0
   else: w[j,nn]=1.0/np.maximum(d[j,nn],1e-8)**float(p['power']); w[j]/=w[j].sum()
  else: raise ValueError('projection method must be nearest or idw')
 return torch.as_tensor(w)
PROJECTION=projection_matrix()
def project18(raw): return torch.einsum('ij,bjt->bit',PROJECTION.to(raw.device,dtype=raw.dtype),raw)
def embed(model,data,idx):
 out=[]; b=int(CFG['features']['prediction_batch_size']) if 'prediction_batch_size' in CFG['features'] else 256
 for i in range(0,len(idx),b):
  take=torch.as_tensor(idx[i:i+b],device=data.raw.device); out.append(model.embed(project18(data.raw[take])).float())
 return torch.cat(out) if out else torch.empty((0,64),device=data.raw.device)
def supcon(z,s,t):
 z=torch.nn.functional.normalize(z.float(),dim=1); logits=z@z.T/float(CFG['training']['temperature']); pos=(s[:,None]==s[None,:])&(t[:,None]!=t[None,:])
 if not torch.all(pos.sum(1)>0): raise ValueError('cross-session positive missing')
 logits=logits.masked_fill(torch.eye(len(z),dtype=torch.bool,device=z.device),-torch.inf); lp=logits-torch.logsumexp(logits,1,keepdim=True)
 return -(torch.where(pos,lp,torch.zeros_like(lp)).sum(1)/pos.sum(1)).mean()
def train(data,roles,fold,path,force,smoke):
 model=E88Spatial18Embedding(CFG).to(data.raw.device); expected={'config':sha(HERE/'config.json'),'run':sha(HERE/'run_e88.py'),'model':sha(HERE/'e88_model.py')}
 if path.is_file() and not force:
  saved=torch.load(path,map_location=data.raw.device,weights_only=False)
  if saved.get('source_hashes')!=expected: raise ValueError('E88 checkpoint hash mismatch; use --force')
  model.load_state_dict(saved['model_state']); model.eval(); return model,saved['training_info']
 seed=int(CFG['seed_base'])+fold*1000; torch.manual_seed(seed); np.random.seed(seed); groups=[]
 for s in roles['enrolled']:
  for sess in CFG['data']['sessions']: groups.append(data.indices(np.array([s]),session=sess,runs=CFG['training']['train_runs']))
 opt=torch.optim.AdamW(model.parameters(),lr=CFG['training']['learning_rate'],weight_decay=CFG['training']['weight_decay']); rng=np.random.default_rng(seed); hist=[]; epochs=int(CFG['training']['smoke_epochs'] if smoke else CFG['training']['max_epochs'])
 for ep in range(1,epochs+1):
  perms=[rng.permutation(g) for g in groups]; k=int(CFG['training']['trials_per_subject_session']); steps=min(map(len,perms))//k; total=0; model.train()
  for step in range(steps):
   ix=np.concatenate([g[step*k:(step+1)*k] for g in perms]); sel=torch.as_tensor(ix,device=data.raw.device); sub=torch.as_tensor(data.subject[ix],device=data.raw.device); ses=torch.as_tensor(data.session[ix],device=data.raw.device); opt.zero_grad(set_to_none=True)
   amp=bool(CFG['training']['amp']) and data.raw.device.type=='cuda'
   with torch.autocast(device_type=data.raw.device.type,dtype=torch.float16,enabled=amp): loss=supcon(model.embed(project18(data.raw[sel])),sub,ses)
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['training']['gradient_clip_norm']); opt.step(); total+=float(loss)*len(ix)
  row={'epoch':ep,'loss':total/(steps*len(groups)*k)}; hist.append(row); print(f'E88 fold={fold} epoch={ep} loss={row["loss"]:.5f}',flush=True)
 model.eval(); info={'seed':seed,'epochs':epochs,'history':hist,'input_channels':18,'source_channels':22}; path.parent.mkdir(parents=True,exist_ok=True); torch.save({'experiment':CFG['experiment'],'outer_fold':fold,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'roles':{k:v.tolist() for k,v in roles.items()},'source_hashes':expected,'training_info':info},path); return model,info
def eer(k,u):
 y=np.r_[np.ones(len(k)),np.zeros(len(u))]; f,t,_=roc_curve(y,np.r_[k,u]); return float(np.min(np.maximum(f,1-t)))
def evaluate(model,data,roles,source,target,fold,writer):
 p=CFG['protocol']; enr=roles['enrolled']; ti=data.indices(enr,session=source,runs=[p['template_run']]); ki=data.indices(enr,session=target,runs=[p['calibration_known_query_run']]); ci=data.indices(roles['calibration_unknown'],session=target,runs=[p['calibration_unknown_query_run']]); oi=data.indices(enr,session=target,runs=[p['evaluation_query_run']]); ui=data.indices(roles['outer_unknown'],session=target,runs=[p['outer_unknown_query_run']]); templ=embed(model,data,ti); means=torch.stack([templ[torch.as_tensor(data.subject[ti]==s,device=templ.device)].mean(0) for s in enr]); means=torch.nn.functional.normalize(means,dim=1)
 def score(idx): return (embed(model,data,idx)@means.T).detach().cpu().numpy()
 kc0,uc0,known,unk=map(score,(ki,ci,oi,ui)); threshold=float(np.quantile(kc0.max(1),.05)); kc=known.max(1); uc=unk.max(1); arg=known.argmax(1); kp=enr[arg]; true=data.subject[oi]; ka=kc>=threshold; ua=uc>=threshold; pred=np.where(ka,kp,0); labels=np.r_[np.ones(len(kc),int),np.zeros(len(uc),int)]; accepted=np.r_[ka,ua]
 if writer:
  for n,idx in enumerate(oi): writer.writerow({'outer_fold':fold,'direction':source+'_to_'+target,'sample_index':int(idx),'subject':int(data.subject[idx]),'session':target,'run':int(data.run[idx]),'role':'known','score':float(kc[n]),'accepted':int(ka[n]),'predicted_identity':int(pred[n])})
  for n,idx in enumerate(ui): writer.writerow({'outer_fold':fold,'direction':source+'_to_'+target,'sample_index':int(idx),'subject':int(data.subject[idx]),'session':target,'run':int(data.run[idx]),'role':'unknown','score':float(uc[n]),'accepted':int(ua[n]),'predicted_identity':int(enr[arg[n]])})
 return {'N_known':len(kc),'N_unknown':len(uc),'known_acceptance':float(ka.mean()),'unknown_FAR':float(ua.mean()),'unknown_rejection':float((~ua).mean()),'AUROC':float(roc_auc_score(labels,np.r_[kc,uc])),'identity_macro_F1':float(f1_score(true,pred,labels=enr,average='macro',zero_division=0)),'unknown_F1':float(f1_score(labels,accepted,pos_label=0,zero_division=0)),'EER':eer(kc,uc),'C':float((pred==true).mean()),'threshold':threshold,'hard_unknown_policy':'subject-held-out outer run-5'}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
 if not torch.cuda.is_available(): raise RuntimeError('E88 requires CUDA; no metrics fabricated')
 device=torch.device('cuda'); base=json.loads((E73/'config.json').read_text(encoding='utf-8')); base['data']['feature_key']='data'; base['data']['channels']=22; data=load_external_data(base,device); folds=subject_folds(base)[:1] if a.smoke else subject_folds(base); out=HERE/('cache/smoke' if a.smoke else 'outputs'); (out/'models').mkdir(parents=True,exist_ok=True); records=[]; predpath=out/'predictions.csv'
 with predpath.open('w',newline='',encoding='utf-8') as fh:
  writer=csv.DictWriter(fh,fieldnames=['outer_fold','direction','sample_index','subject','session','run','role','score','accepted','predicted_identity']); writer.writeheader()
  for fold,roles in enumerate(folds,1):
   model,info=train(data,roles,fold,out/'models'/f'spatial_{CFG["channel_projection"]["method"]}_outer{fold}.pt',a.force,a.smoke); dirs={}
   for s,t in [('T','E'),('E','T')]: dirs[f'{s}_to_{t}']=None if a.smoke else evaluate(model,data,roles,s,t,fold,writer)
   records.append({'outer_fold':fold,'roles':{k:v.tolist() for k,v in roles.items()},'training':info,'directions':dirs}); del model; torch.cuda.empty_cache()
 result={'schema_version':1,'experiment':CFG['experiment'],'completed_at':now(),'smoke':a.smoke,'channel_projection':CFG['channel_projection'],'data':{'N':int(len(data.subject)),'source_channels':22,'model_channels':18,'competition_test_accessed':False},'runs':records,'test':{'AUC':'N/A','F1':'N/A','accessed':False}}
 if a.smoke: result['scope']='smoke trains one fold and does not access outer evaluation'
 else:
  rows=[r['directions'][d] for r in records for d in r['directions']]; result['primary_summary']={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k],(int,float))}; result['pos_label']=0
 (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); lines=['# E88 标准电极二维空间投影 + 分层 hard-unknown','','- 外部数据：E73 派生 BCIC IV-2a；22 通道源表征显式投影为 18 通道，未使用项目 Test。','- 18 通道由标准 10-20/10-10 二维坐标计算 nearest 或 IDW 插值，保留 22 通道信号信息；不访问项目 Test。','- hard-unknown：按被试 outer 留出，固定 run=5；阈值仅由 enrolled known run=4 q05 固定。','- unknown-F1 使用 `pos_label=0`。','', '| 指标 | 结果 |','|---|---:|']
 if a.smoke: lines += ['| Outer AUC/F1/EER | N/A（smoke 未做 outer 评估） |','| 项目 Test AUC/F1 | N/A（未访问 Test） |']
 else:
  s=result['primary_summary']; lines += [f'| AUC | {s["AUROC"]:.6f} |',f'| known acceptance | {s["known_acceptance"]:.6f} |',f'| unknown FAR | {s["unknown_FAR"]:.6f} |',f'| identity macro-F1 | {s["identity_macro_F1"]:.6f} |',f'| unknown-F1 | {s["unknown_F1"]:.6f} |',f'| EER | {s["EER"]:.6f} |','| 项目 Test AUC/F1 | N/A（未访问 Test） |']
 (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(result.get('primary_summary',{'smoke':True}),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
