"""From repository root: .venv/Scripts/python -B research/E272_final4/run.py"""
import ast, hashlib, json, types
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import rankdata

H=Path(__file__).resolve().parent; R=H.parents[1]; D='cuda'
names=['final','final2','final3','final4']; mods=[]; models=[]; digests={}
for name in names[:3]:
    with ZipFile(R/name/'team_hymn.zip') as z:
        files={k:z.read(next(n for n in z.namelist() if n.split('/')[-1]==k)) for k in ('model.py','weights.pt')}
    folder=R/name/'team_hymn'; assert all(v==(folder/k).read_bytes() for k,v in files.items()); digests[name]={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    mod=types.ModuleType(name); exec(compile(files['model.py'],str(folder/'model.py'),'exec'),mod.__dict__); mods.append(mod); models.append(mod.load_model(str(folder),D))
assert models[1]['feature']==models[2]['feature'] and models[1]['feature']!=models[0]['feature']
for fn in ['_features','_subwindow_feature','_project','enroll']:
    import inspect
    assert inspect.getsource(getattr(mods[1],fn))==inspect.getsource(getattr(mods[2],fn))
m=pd.read_csv(R/'BrainPrint_cache/train_metadata.csv'); assert len(m)==11304 and m.path.is_unique and m.path.str.startswith('Train/').all()
y=m.subject.to_numpy(); t=mods[0]._task_indices(m[['task']].to_dict('records'),len(m)); groups=m.groupby(['subject','task']); assert groups.trial.apply(lambda s:s.is_unique and s.is_monotonic_increasing).all(); f=(m.trial/groups.trial.transform('max')).to_numpy(); early=f<=.6; ix=np.flatnonzero(f>.8); ids=list(range(1,21)); assert sorted(set(y))==ids and set(t)==set(range(4))
def raw(rows):return np.stack([loadmat(R/'Brainprint_Competition'/m.path.iloc[i],variable_names=['data'])['data'].T.astype('float32') for i in rows])
def function(file,name,env):
    tree=ast.parse(file.read_text(encoding='utf8')); tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]; assert len(tree.body)==1; exec(compile(tree,str(file),'exec'),env); return env[name]
def cached(mod):
    p=types.ModuleType(mod.__name__+'_cached'); exec(compile((R/mod.__name__/'team_hymn/model.py').read_bytes(),mod.__name__,'exec'),p.__dict__); p._features=lambda a,x:x; return p
cp=list(map(cached,mods)); labels=np.repeat([21,22,23],20)
@torch.inference_mode()
def main():
    features=[torch.zeros(len(m),855,device=D) for _ in range(2)]; used=np.flatnonzero(early|(f>.8))
    for batch in np.array_split(used,(len(used)+95)//96):
        wave=raw(batch)
        for j in (0,1): features[j][batch]=mods[j]._features(models[j],wave)
    env=dict(X=features[1],early=early,y=y,t=t,np=np,torch=torch,D=D,w=models[1],tau=models[1]['tau'],beta=models[1]['beta'],pkg=cp[1],W=H); file=R/'research/E270_new_calibration/run.py'; fit=function(file,'fit',env); split=function(file,'split',env); f1=function(file,'f1_score',dict(np=np)); auc=function(file,'roc_auc_score',dict(rankdata=rankdata)); folds=[]
    for h in range(20):
        old,new,unk=split(ids,h); i=np.flatnonzero(early&np.isin(y,old)); x=features[0][i]; tt=t[i]; mu=torch.stack([x[tt==k].mean(0) for k in range(4)]); sd=torch.stack([x[tt==k].std(0,unbiased=False)+1e-6 for k in range(4)]); z=(x-mu[tt])/sd[tt]; pm=z.mean(0); z-=pm; b=torch.linalg.eigh(z.T@z)[1][:,-96:].flip([1]).contiguous(); q=z@b
        a=dict(models[0],task_mu=mu,task_sd=sd,pca_mean=pm,pca_basis=b,prototypes=torch.zeros_like(models[0]['prototypes']),valid=torch.zeros_like(models[0]['valid']))
        for k in range(4):
            for p in old:a['prototypes'][k,p-1]=q[(tt==k)&(y[i]==p)].mean(0); a['valid'][k,p-1]=True
        tag=f'h{h:02}'; rank,reg=fit(old,new,tag); wave=raw(reg); assert len(np.unique(reg))==60 and np.array_equal(y[reg],np.repeat(new,20)); mods[0].enroll(a,wave,None,labels); mods[1].enroll(rank,wave,None,labels); assert torch.equal(a['valid'],rank['valid']) and a['valid'].sum()==68
        torch.save(dict(gate={k:v.cpu() if torch.is_tensor(v) else v for k,v in a.items()},rank={k:v.cpu() if torch.is_tensor(v) else v for k,v in rank.items()},fit_rows=i.tolist(),reg_rows=reg.tolist()),H/(tag+'.pt'))
        outs=[]
        for j in range(3):
            outs.append(np.concatenate([cp[j].predict_batch(a if not j else rank,features[int(j>0)][g],m[['task']].iloc[g].to_dict('records')) for g in np.array_split(ix,20)]))
        ds,s,s3=outs; assert np.array_equal(s[:,1:],s3[:,1:]); preds=[v.argmax(1) for v in outs]; preds.append(np.where(preds[0]!=0,s[:,1:].argmax(1)+1,0)); assert np.array_equal(preds[0]!=0,preds[3]!=0)
        sim=np.sort(s[:,1:],axis=1); gate=sim[:,-1]+rank['beta']*(sim[:,-1]-sim[:,-2]); scores=[ds[:,1:].max(1),gate-rank['tau'],gate-np.where(s[:,1:].argmax(1)>=20,np.float32(rank['tau']-.03),np.float32(rank['tau'])),ds[:,1:].max(1)]; assert all(np.array_equal(preds[j]!=0,scores[j]>=0) for j in (1,2)); truth=y[ix].copy(); truth[np.isin(truth,unk)]=0
        for j,p in enumerate(new):truth[y[ix]==p]=21+j
        masks=dict(old=(truth>0)&(truth<=20),new=truth>20,URR=truth==0); row=dict(h=h,old_ids=old,new_ids=new,unknown_ids=unk)
        for name,p,g in zip(names,preds,scores):row[name]=dict(**{k:float((p[v]==truth[v]).mean()) for k,v in masks.items()},AUC=auc(truth!=0,g),F1=f1(truth,p))
        for base in (0,2):
            u=preds[base]==truth; v=preds[3]==truth; row['vs_'+names[base]]={k:[int((~u&v&mask).sum()),int((u&~v&mask).sum())] for k,mask in masks.items()}
        folds.append(row)
    result=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),late=len(ix),package_sha256=digests,configs=[a['feature'] for a in models],tau=[a['tau'] for a in models],folds=folds); (H/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for n in names:print(n,pd.DataFrame([r[n] for r in folds]).mean().to_dict())
    for n in ('final','final3'):
        d=pd.DataFrame([r['final4'] for r in folds])-pd.DataFrame([r[n] for r in folds]); print(n,'mean/worst/worst_fold/wins',d.mean().to_dict(),d.min().to_dict(),d.idxmin().to_dict(),(d>0).sum().to_dict()); print({k:np.sum([r['vs_'+n][k] for r in folds],axis=0).tolist() for k in masks})
    fr=pd.DataFrame([r['final4'] for r in folds]); print('final4 min',fr.min().to_dict(),fr.idxmin().to_dict(),'late',len(ix))
if __name__=='__main__':main()
