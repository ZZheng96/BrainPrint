"""Run from project root with .venv/Scripts/python -B research/E274_image_gate/run.py."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.nn.functional import normalize

H = Path(__file__).resolve().parent
R = H.parents[1]
sys.path.insert(0, str(R))
from src.eeg_features import load_eeg
from src.eeg_image_dinov2.image_encoders import encode

C = R/'BrainPrint_cache'
E = R/'research/E272_final4'
D = C/'eeg_image_dinov2'
meta = pd.read_csv(C/'train_metadata.csv')
assert len(meta)==11304 and meta.path.is_unique and meta.path.str.startswith('Train/').all()
spec = importlib.util.spec_from_file_location('final3', R/'final3/team_hymn/model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
model._features = lambda a,x: x
X = torch.from_numpy(np.load(C/'e262_wave_pair/clean.npy')).cuda()
emb = D/'train_covariance5_dinov2.npy'
Z = torch.from_numpy(np.load(emb)).cuda()
assert Z.shape==(len(meta),768) and torch.isfinite(Z).all() and Z.norm(dim=1).min()>0
Z = normalize(Z)
y = meta.subject.to_numpy()
t = model._task_indices(meta[['task']].to_dict('records'),len(meta))
f = (meta.trial/meta.groupby(['subject','task']).trial.transform('max')).to_numpy()
late = np.flatnonzero(f>.8)

def alignment():
    images = np.load(D/'train_covariance5.npy',mmap_mode='r')
    # One early raw trial per subject/task, independent of scores.
    rows = meta.groupby(['subject','task']).head(1).index.tolist()
    for i in rows:
        data,_ = load_eeg(R/'Brainprint_Competition'/meta.path[i])
        pixels = np.clip(encode(data,'covariance5')*255,0,255).astype(np.uint8)
        assert np.array_equal(images[i],pixels), i
    return rows

def dataset(obj,old,new,unknown,ix,outer=False):
    fit,reg = map(np.asarray,(obj['fit_rows'],obj['reg_rows']))
    assert not (set(old)&set(new) or set(new)&set(unknown) or set(old)&set(unknown))
    assert np.array_equal(fit,np.flatnonzero((f<=.6)&np.isin(y,old)))
    assert np.array_equal(reg,np.concatenate([np.flatnonzero((f<=.6)&(y==u)&(t==k))[:5] for u in new for k in range(4)]))
    a = obj['rank' if outer else 'artifact']; a['new_winner_offset']=.03
    s = np.concatenate([model.predict_batch(a,X[b],meta[['task']].iloc[b].to_dict('records')) for b in np.array_split(ix,20 if outer else 4)])
    p = s[:,1:].argmax(1)+1
    v = np.sort(s[:,1:],axis=1)
    g = v[:,-1]+a['beta']*(v[:,-1]-v[:,-2])-np.where(p>20,np.float32(a['tau']-.03),np.float32(a['tau']))
    assert np.array_equal(g>=0,s.argmax(1)!=0)
    labels = list(old)+[21,22,23]; rows = np.r_[fit,reg]; ids = np.r_[y[fit],np.repeat([21,22,23],20)]
    proto = torch.stack([normalize(torch.stack([Z[rows[(t[rows]==k)&(ids==u)]].mean(0) for u in labels])) for k in range(4)])
    cos = torch.empty(len(ix),len(labels),device='cuda')
    for k in range(4):
        mask = t[ix]==k; cos[mask]=Z[ix[mask]]@proto[k].T
    agree = np.asarray(labels)[cos.argmax(1).cpu().numpy()]==p
    truth=y[ix].copy(); truth[np.isin(y[ix],unknown)]=0
    for j,u in enumerate(new): truth[y[ix]==u]=21+j
    return (p,g,agree,truth),dict(templates=proto.cpu(),labels=labels)

def score(d,delta,image):
    p,g,agree,_=d
    return g + np.float32(delta)*((p>20)&(agree if image else True))

def calibrate(d):
    p,_,_,truth=d; unk=truth==0
    def counts(delta,img):
        accept=score(d,delta,img)>=0
        return int((accept&unk).sum()),int((accept&(truth==p)&(p>20)).sum())
    grid=np.arange(201)/1000
    base=counts(0,True)[0]; budget=base+int(unk.sum())//100
    img=max((x for x in grid if counts(x,True)[0]<=budget),key=lambda x:(counts(x,True)[1],-x))
    target=counts(img,True)[0]
    ctl=min(grid,key=lambda x:(abs(counts(x,False)[0]-target),-counts(x,False)[1],x))
    return img,ctl,dict(n_unknown=int(unk.sum()),baseline_FA=base,image_FA=target,control_FA=counts(ctl,False)[0],image_delta=img,control_delta=ctl)

def measure(d,margin):
    p,g,_,truth=d; pred=np.where(margin>=0,p,0); known=truth!=0; new=truth>20; old=known&~new; n=int(known.sum())
    extra=(margin>=0)&(g<0)
    met=dict(old=float((pred[old]==truth[old]).mean()),new=float((pred[new]==truth[new]).mean()),URR=float((pred[~known]==0).mean()),AUC=float((rankdata(margin)[known].sum()-n*(n+1)/2)/(n*(len(p)-n))),F1=float(np.mean([2*((truth==k)&(pred==k)).sum()/((truth==k).sum()+(pred==k).sum()) for k in np.union1d(truth,pred)])))
    return met,dict(recovered=int((extra&(p==truth)).sum()),wrong_known=int((extra&known&(p!=truth)).sum()),extra_unknown=int((extra&~known).sum()),all_unknown_FA=int(((pred!=0)&~known).sum()))

@torch.inference_mode()
def main():
    audited=alignment(); folds=[]; patches={}
    for r in json.loads((E/'results.json').read_text())['folds']:
        h=r['h']; pool=r['old_ids']; inner=[]; splits=[]
        for j in range(2):
            a=torch.load(R/f'research/E270_new_calibration/fits/h{h:02}_i{j}.pt',weights_only=True,map_location='cuda'); old,new=a['old'],a['new']; unk=sorted(set(pool)-set(old+new)); assert len(unk)==3 and set(old+new+unk)==set(pool)
            ix=np.flatnonzero((f>.6)&(f<=.8)&np.isin(y,new+unk)); inner.append(dataset(a,old,new,unk,ix)[0]); splits.append(dict(old=old,new=new,unknown=unk))
        img,ctl,cal=calibrate(tuple(map(np.concatenate,zip(*inner))))
        a=torch.load(E/f'h{h:02}.pt',map_location='cuda',weights_only=True)
        d,patch=dataset(a,pool,r['new_ids'],r['unknown_ids'],late,True); patches[h]=dict(**patch,**cal)
        row=dict(h=h,inner=splits,calibration=cal,old=pool,new_ids=r['new_ids'],unknown=r['unknown_ids']); row['counts']={}
        for name,delta,use in [('final3',0,False),('image',img,True),('control',ctl,False)]:
            row[name],row['counts'][name]=measure(d,score(d,delta,use))
        assert row['final3']==r['final3']; folds.append(row)
    torch.save(patches,H/'templates.pt')
    summary={k:pd.DataFrame([r[k] for r in folds]).mean().to_dict() for k in ['final3','image','control']}
    counts={k:pd.DataFrame([r['counts'][k] for r in folds]).sum().to_dict() for k in summary}
    result=dict(summary=summary,counts=counts,folds=folds,alignment_rows=audited,gpu=torch.cuda.get_device_name(),sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [emb,C/'train_metadata.csv']}); (H/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(summary,counts); print('inner totals',pd.DataFrame([r['calibration'] for r in folds]).sum().to_dict(),'exact matches',sum(r['calibration']['image_FA']==r['calibration']['control_FA'] for r in folds)); print('new wins',sum(r['image']['new']>r['control']['new'] for r in folds))

if __name__=='__main__': main()
