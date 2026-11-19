"""Train-only paired decision test; no threshold search."""
import runpy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score
H=Path(__file__).resolve().parent
b=runpy.run_path(str(H.parent/'E278_dual_gate/controls.py')); g=b['m277']; C,E=b['C'],b['E']
m=pd.read_csv(C/'train_metadata.csv'); assert m.path.str.startswith('Train/').all()
t=g['G']['_task_indices'](m[['task']].to_dict('records'),len(m)); y=m.subject.to_numpy(); late=(m.trial/m.groupby(['subject','task']).trial.transform('max')>.8).to_numpy()
x={k:torch.from_numpy(np.load(C/f'e262_wave_pair/{k}.npy')).cuda() for k in ('clean','stress')}; out=[]
with torch.inference_mode():
 for f in json.loads((E/'results.json').read_text())['folds']:
  a=torch.load(E/f"fits/seed{f['seed']}_compat.pt",map_location='cuda',weights_only=True); truth=y.copy(); truth[np.isin(y,f['unknown_ids'])]=0
  for i,p in enumerate(f['new_ids']):truth[y==p]=21+i
  for name,k,miss in g['SCENES']:
   ix=np.flatnonzero(late if miss is None else late&(t!=miss)); tasks=t[ix] if miss is None else np.full(len(ix),-1); v=truth[ix]; known=v!=0
   s=g['score'](a,x[k][ix],tasks,miss,np.array(f['old_ids'])-1,True); w=s.argmax(1)+1; sv=np.sort(s,axis=1); gap=sv[:,-1]-sv[:,-2]; sd=np.nanstd(np.where(np.isfinite(s),s,np.nan),axis=1); rel=gap-sd
   for shift in (0,.15):
    raw=sv[:,-1]-shift+.5*gap-(a['tau']-.06*(w>20))
    for method,margin in [('single',raw),('relative',np.maximum(raw,rel))]:
     pred=np.where(margin>=0,w,0); old=known&(v<=20); new=v>20
     out.append(dict(seed=f['seed'],scene=name,shift=shift,method=method,old=(pred[old]==v[old]).mean(),new=(pred[new]==v[new]).mean(),unknown=(pred[~known]==0).mean(),false_reject=(pred[known]==0).mean(),auc=roc_auc_score(known,margin),f1=f1_score(v,pred,average='macro')))
d=pd.DataFrame(out); d.to_json(H/'results.json',orient='records',indent=2); rows=[]
for (seed,shift,method),z in d.groupby(['seed','shift','method']):
 c=z[z.scene=='clean_complete'].iloc[0]; st=z[z.scene=='stress_complete'].iloc[0]; rows.append(dict(seed=seed,shift=shift,method=method,old=c.old,stress=st.old,new=c['new'],unknown=c.unknown,equal=np.mean([c.old,st.old,c['new'],c.unknown]),fr=c.false_reject,auc=z.auc.mean(),f1=z.f1.mean()))
r=pd.DataFrame(rows); print(r.round(4).to_string(index=False)); print(r.groupby(['shift','method']).mean().round(4).to_string())
