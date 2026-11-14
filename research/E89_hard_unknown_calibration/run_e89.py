"""E89 layered hard-unknown threshold calibration on E88 embeddings."""
from __future__ import annotations
import argparse,csv,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,roc_curve
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; E88=ROOT/'research'/'E88_spatial_channel_projection'; E73=ROOT/'research'/'E73_external_multisession'
sys.path[:0]=[str(E88),str(E73)]
import run_e88 as e88
from external_data import load_external_data,subject_folds

def eer(k,u):
 y=np.r_[np.ones(len(k)),np.zeros(len(u))]; f,t,_=roc_curve(y,np.r_[k,u]); return float(np.min(np.maximum(f,1-t)))
def metrics(k,u,pred,accepted,true,threshold):
 labels=np.r_[np.ones(len(k),int),np.zeros(len(u),int)]; acc=accepted
 return {'N_known':len(k),'N_unknown':len(u),'known_acceptance':float(accepted[:len(k)].mean()),'unknown_FAR':float(accepted[len(k):].mean()),'unknown_rejection':float((~accepted[len(k):]).mean()),'AUROC':float(roc_auc_score(labels,np.r_[k,u])),'identity_macro_F1':float(f1_score(true,pred,labels=np.unique(true),average='macro',zero_division=0)),'unknown_F1':float(f1_score(labels,acc,pos_label=0,zero_division=0)),'EER':eer(k,u),'threshold':threshold}
def thresholds(kc0,uc0,known_subjects,calib_subject):
 known=kc0.max(1); unknown=uc0.max(1); arg=kc0.argmax(1)
 out={}
 out['q05_known_only']=('global',float(np.quantile(known,.05)))
 out['tail_q01_known_only']=('global',float(np.quantile(known,.01)))
 out['known_unknown_far10']=('global',float(np.quantile(unknown,.90)))
 y=np.r_[np.ones(len(known)),np.zeros(len(unknown))]; s=np.r_[known,unknown]; fpr,tpr,ts=roc_curve(y,s); out['known_unknown_youden']=('global',float(ts[np.argmax(tpr-fpr)]))
 per={}
 for i,s in enumerate(known_subjects):
  vals=kc0[calib_subject==s].max(1)
  per[int(s)]=float(np.quantile(vals,.05)) if len(vals) else out['q05_known_only'][1]
 out['per_identity_q05']=('per_identity',per)
 return out

def run_eval(model,data,roles,source,target,fold,writer):
 p=e88.CFG['protocol']; enr=roles['enrolled']; ti=data.indices(enr,session=source,runs=[p['template_run']]); ki=data.indices(enr,session=target,runs=[p['calibration_known_query_run']]); ci=data.indices(roles['calibration_unknown'],session=target,runs=[p['calibration_unknown_query_run']]); oi=data.indices(enr,session=target,runs=[p['evaluation_query_run']]); ui=data.indices(roles['outer_unknown'],session=target,runs=[p['outer_unknown_query_run']])
 templ=e88.embed(model,data,ti); means=np.stack([templ[np.asarray(data.subject[ti]==s)].mean(0).detach().cpu().numpy() for s in enr]); means=means/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12)
 def score(idx): return (e88.embed(model,data,idx).detach().cpu().numpy()@means.T)
 kc0,uc0,known,unk=map(score,(ki,ci,oi,ui)); cal_sub=data.subject[ki]; true=data.subject[oi]; out={}
 for name,(kind,thr) in thresholds(kc0,uc0,enr,cal_sub).items():
  kc=known.max(1); uc=unk.max(1); arg=known.argmax(1)
  if kind=='global': ka=kc>=thr; ua=uc>=thr
  else:
   per=np.asarray([thr.get(int(s),np.nan) for s in enr]); ka=kc>=per[arg]; ua=uc>=np.nanmean(per)
  pred=np.where(ka,enr[arg],0); accepted=np.r_[ka,ua]; out[name]=metrics(kc,uc,pred,accepted,true,float(thr) if kind=='global' else None)
  if writer:
   for n,idx in enumerate(oi): writer.writerow({'outer_fold':fold,'direction':source+'_to_'+target,'strategy':name,'sample_index':int(idx),'role':'known','subject':int(data.subject[idx]),'score':float(kc[n]),'accepted':int(ka[n]),'predicted_identity':int(pred[n])})
   for n,idx in enumerate(ui): writer.writerow({'outer_fold':fold,'direction':source+'_to_'+target,'strategy':name,'sample_index':int(idx),'role':'unknown','subject':int(data.subject[idx]),'score':float(uc[n]),'accepted':int(ua[n]),'predicted_identity':int(enr[arg[n]])})
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
 if not __import__('torch').cuda.is_available(): raise RuntimeError('E89 requires CUDA')
 import torch
 base=json.loads((E73/'config.json').read_text(encoding='utf-8')); base['data']['feature_key']='data'; base['data']['channels']=22; data=load_external_data(base,torch.device('cuda')); folds=subject_folds(base)[:1] if a.smoke else subject_folds(base); out=HERE/('cache/smoke' if a.smoke else 'outputs'); (out/'models').mkdir(parents=True,exist_ok=True); records=[]; predpath=out/'predictions.csv'
 with predpath.open('w',newline='',encoding='utf-8') as fh:
  writer=csv.DictWriter(fh,fieldnames=['outer_fold','direction','strategy','sample_index','role','subject','score','accepted','predicted_identity']); writer.writeheader()
  for fold,roles in enumerate(folds,1):
   ck=E88/'outputs'/'models'/f'spatial_{e88.CFG["channel_projection"]["method"]}_outer{fold}.pt'
   model,info=e88.train(data,roles,fold,ck,a.force,a.smoke)
   dirs={}
   if not a.smoke:
    for s,t in [('T','E'),('E','T')]: dirs[f'{s}_to_{t}']=run_eval(model,data,roles,s,t,fold,writer)
   records.append({'outer_fold':fold,'roles':{k:v.tolist() for k,v in roles.items()},'directions':dirs,'training':info}); del model; torch.cuda.empty_cache()
 result={'schema_version':1,'experiment':'E89 layered hard-unknown calibration using E88 embeddings','completed_at':datetime.now(timezone.utc).isoformat(),'smoke':a.smoke,'data':{'source':'E73-derived BCIC IV-2a','competition_test_accessed':False},'runs':records,'test':{'AUC':'N/A','F1':'N/A','accessed':False}}
 if a.smoke: result['scope']='smoke only; outer query not evaluated'
 else:
  names=list(records[0]['directions']['T_to_E']); result['strategy_summary']={n:{k:float(np.mean([r['directions'][d][n][k] for r in records for d in r['directions']])) for k in ['AUROC','known_acceptance','unknown_FAR','identity_macro_F1','unknown_F1','EER']} for n in names}; result['pos_label']=0
 (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lines=['# E89 分层 hard-unknown calibration','','基于 E88 外部嵌入；仅使用 E73 派生数据，未访问项目 Test。每个 outer fold 先用 calibration known/unknown 决策阈值，再对 outer query 只评估一次。','', '| 策略 | AUC | known acceptance | unknown FAR | identity macro-F1 | unknown-F1(pos_label=0) | EER |','|---|---:|---:|---:|---:|---:|---:|']
 if a.smoke: lines+=['| smoke | N/A（未评估 outer query） | - | - | - | - | - |']
 else:
  for n,v in result['strategy_summary'].items(): lines.append(f"| {n} | {v['AUROC']:.6f} | {v['known_acceptance']:.6f} | {v['unknown_FAR']:.6f} | {v['identity_macro_F1']:.6f} | {v['unknown_F1']:.6f} | {v['EER']:.6f} |")
 lines += ['', '## Trade-off', '- `q05_known_only` 保持 E88 基线的 known coverage，但不使用 calibration unknown，通常 unknown FAR 较高。','- `known_unknown_far10` 以 calibration unknown 的 90% 分位控制约 10% FAR，偏安全但可能降低 known acceptance。','- `known_unknown_youden` 在 calibration known/unknown 上最大化 Youden 指数，折中 FAR 与 acceptance；不使用 outer unknown。','- `per_identity_q05` 为每个 enrolled identity 单独用 calibration known 估计 q05，适合身份间分数尺度不同；`tail_q01_known_only` 更保守。','', '项目 Test AUC/F1：N/A（明确未访问）。']
 (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(result.get('strategy_summary',{'smoke':True}),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
