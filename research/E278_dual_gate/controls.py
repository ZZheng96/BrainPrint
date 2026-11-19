"""Score unchanged E275/E277 baselines with fresh task state for every scene."""
import json
import runpy
from pathlib import Path
import numpy as np
import pandas as pd
import torch

H=Path(__file__).resolve().parent
m277=runpy.run_path(str(H.parent/'E277_single_whitening/run.py'))
E,C=m277['E'],m277['C']

@torch.inference_mode()
def main():
    ref=json.loads((E/'results.json').read_text(encoding='utf8')); m=pd.read_csv(C/'train_metadata.csv'); assert len(m)==11304 and m.path.str.startswith('Train/').all(); t=m277['G']['_task_indices'](m[['task']].to_dict('records'),len(m)); y=m.subject.to_numpy(); late=(m.trial/m.groupby(['subject','task']).trial.transform('max')>.8).to_numpy()
    x={k:torch.from_numpy(np.load(C/f'e262_wave_pair/{k}.npy')).cuda() for k in ('clean','stress')}; out=[]
    for f in ref['folds']:
        a=torch.load(E/f"fits/seed{f['seed']}_compat.pt",map_location='cuda',weights_only=True); truth=y.copy(); truth[np.isin(y,f['unknown_ids'])]=0
        for i,p in enumerate(f['new_ids']):truth[y==p]=21+i
        for mode in ('current','single'):
            s={}
            for name,k,miss in m277['SCENES']:
                ix=np.flatnonzero(late if miss is None else late&(t!=miss)); task=t[ix] if miss is None else np.full(len(ix),-1)
                sim=m277['score'](a,x[k][ix],task,miss,np.array(f['old_ids'])-1,mode=='single'); s[name]=m277['measure'](sim,truth[ix],a)
            out.append(dict(seed=f['seed'],method=mode,scenes=s))
    (H/'controls.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    dual=json.loads((H/'results.json').read_text())['folds']; out.extend(dict(seed=f['seed'],method='dual',scenes=f['isolated']) for f in dual); rows=[]
    for f in out:
        s=f['scenes']; c=s['clean_complete']; v=[c['old_accuracy'],s['stress_complete']['old_accuracy'],c['new_accuracy'],c['unknown_rejection']]; rows.append([f['method'],f['seed'],*v,np.mean(v),*[np.mean([d[k] for d in s.values()]) for k in ('AUC','F1')]])
    df=pd.DataFrame(rows,columns=['method','seed','old','stress','new','URR','equal','AUC','F1']); print(df.to_string(index=False)); print(df.groupby('method').mean());print(pd.DataFrame([f['isolated']['clean_complete'] for f in dual]).mean())
if __name__=='__main__':main()
