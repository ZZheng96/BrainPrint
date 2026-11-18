"""Run from root: .venv/Scripts/python -B research/E273_final3_template_weighting/run.py"""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import rankdata

H=Path(__file__).resolve().parent; R=H.parents[1]; E=R/'research/E272_final4'; C=R/'BrainPrint_cache'; pkg=R/'final3/team_hymn/model.py'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def module():
    spec=importlib.util.spec_from_file_location('final3_eval',pkg); obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
raw=module(); model=module(); model._features=lambda a,x:x
m=pd.read_csv(C/'train_metadata.csv'); assert len(m)==11304 and m.path.str.startswith('Train/').all() and m.path.is_unique
y=m.subject.to_numpy(); t=model._task_indices(m[['task']].to_dict('records'),len(m)); f=(m.trial/m.groupby(['subject','task']).trial.transform('max')).to_numpy(); ix=np.flatnonzero(f>.8); batches=np.array_split(ix,20)
ref=json.loads((E/'results.json').read_text()); assert sha(pkg)==ref['package_sha256']['final3']['model.py']; X=torch.from_numpy(np.load(C/'e262_wave_pair/clean.npy')).cuda(); assert X.shape==(11304,855)
def extract(a,rows): return raw._features(a,np.stack([loadmat(R/'Brainprint_Competition'/m.path.iloc[i],variable_names=['data'])['data'].T.astype('float32') for i in rows]))
def evaluate(a,X,truth):
    s=np.concatenate([model.predict_batch(a,X[b],m[['task']].iloc[b].to_dict('records')) for b in batches]); p=s.argmax(1); top=s[:,1:].argmax(1)+1; v=np.sort(s[:,1:],axis=1); margin=v[:,-1]+a['beta']*(v[:,-1]-v[:,-2])-np.where(top>20,np.float32(a['tau']-.03),np.float32(a['tau'])); assert np.array_equal(margin>=0,p!=0)
    known=truth!=0; n=int(known.sum()); masks={'new':truth>20,'old':known&(truth<=20),'URR':~known}; counts={k:int(mask.sum()) for k,mask in masks.items()}; good={k:(p==truth)[mask] for k,mask in masks.items()}; good['top1']=(top==truth)[masks['new']]; metrics={k:float(arr.mean()) for k,arr in good.items()}; metrics.update(AUC=float((rankdata(margin)[known].sum()-n*(n+1)/2)/(n*(len(p)-n))),F1=float(np.mean([2*((truth==k)&(p==k)).sum()/((truth==k).sum()+(p==k).sum()) for k in np.union1d(truth,p)])))
    state=np.where(good['new'],0,np.where(good['top1'],1,2)); return metrics,good,counts,state

@torch.inference_mode()
def main():
    deps={}; folds=[]; patches={}; source='E262 clean.npy'; reg_sources=[]
    for r in ref['folds']:
        h=r['h']; fit=E/f'h{h:02}.pt'; deps[str(fit.relative_to(R))]=sha(fit); obj=torch.load(fit,map_location='cuda',weights_only=True); a=obj['rank']; reg=np.array(obj['reg_rows']); truth=y[ix].copy(); truth[np.isin(y[ix],r['unknown_ids'])]=0
        for j,p in enumerate(r['new_ids']):truth[y[ix]==p]=21+j
        assert np.array_equal(reg,np.concatenate([np.flatnonzero((f<=.6)&(y==p)&(t==k))[:5] for p in r['new_ids'] for k in range(4)])); assert np.array_equal(obj['fit_rows'],np.flatnonzero((f<=.6)&np.isin(y,r['old_ids'])))
        base,bg,den,bs=evaluate(a,X,truth)
        if any(base[k]!=v for k,v in r['final3'].items()) and source=='E262 clean.npy':
            source='E272 clean extraction batch schedule'; used=np.flatnonzero((f<=.6)|(f>.8));
            for b in np.array_split(used,(len(used)+95)//96): X[b]=extract(a,b)
            base,bg,den,bs=evaluate(a,X,truth)
        assert all(base[k]==v for k,v in r['final3'].items()),(h,base,r['final3']); regfeat=X[reg]; labels=np.repeat([21,22,23],20); orig=a['prototypes'][:,20:].clone()
        def matches(feat):
            tmp=dict(a,prototypes=a['prototypes'].clone(),valid=a['valid'].clone()); model.enroll(tmp,feat,None,labels); return torch.equal(tmp['prototypes'][:,20:],orig)
        if not matches(regfeat): regfeat=extract(a,reg); reg_sources.append('E272 original enrollment waveform batch'); assert matches(regfeat)
        else:reg_sources.append(source)
        weights=torch.empty(4,3,4,5,device='cuda'); fallback=0
        for k in range(4):
            for j in range(3):
                q=model._project(a,regfeat[20*j:20*j+20],k).reshape(4,5,-1); u=model._unit(q@a['whiten']); cos=u@u.transpose(1,2); cos.diagonal(dim1=1,dim2=2).zero_(); w=(1+cos.sum(2)/4)/2; total=w.sum(1,keepdim=True); fallback+=int((total==0).sum()); w=torch.where(total==0,0.2,w/total); weights[k,j]=w; a['prototypes'][k,20+j]=(w[:,:,None]*q).sum(1)
        cand,cg,_,cs=evaluate(a,X,truth); paired={k:[int((~bg[k]&cg[k]).sum()),int((bg[k]&~cg[k]).sum())] for k in bg}; trans=np.bincount(bs*3+cs,minlength=9).reshape(3,3); folds.append(dict(h=h,new_ids=r['new_ids'],den=den,baseline=base,candidate=cand,corrected_harmed=paired,new_transition=trans.tolist(),uniform_fallback=fallback)); patches[h]=dict(prototypes=a['prototypes'][:,20:].cpu(),weights=weights.cpu(),reg_rows=reg.tolist(),fit=str(fit.relative_to(R)),sha256=deps[str(fit.relative_to(R))]);
    for p in [pkg,E/'run.py',E/'results.json',C/'e262_wave_pair/clean.npy',C/'train_metadata.csv']: deps[str(p.relative_to(R))]=sha(p)
    torch.save(patches,H/'new_templates.pt'); result=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),query_source=source,reg_sources=reg_sources,dependencies_sha256=deps,state_order=['correct_accepted','correct_rejected','wrong_rank'],folds=folds); (H/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8');
    B=pd.DataFrame([r['baseline'] for r in folds]); A=pd.DataFrame([r['candidate'] for r in folds]); d=A-B; print(pd.DataFrame(dict(base=B.mean(),candidate=A.mean(),delta=d.mean(),win=(d>0).sum(),tie=(d==0).sum(),loss=(d<0).sum())).to_string(float_format=lambda v:f'{v:.10f}')); print('transition',np.sum([r['new_transition'] for r in folds],0)); print('paired',{k:np.sum([r['corrected_harmed'][k] for r in folds],0) for k in bg},'den',pd.DataFrame([r['den'] for r in folds]).sum().to_dict()); print(source,set(reg_sources),result['gpu'])
if __name__=='__main__':main()
