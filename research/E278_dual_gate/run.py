"""E278: frozen final4 decisions on E275 Train folds. Run with python -B."""
import ast
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import rankdata

H = Path(__file__).resolve().parent
R = H.parents[1]; E = H.parent/'E275_threshold_validation'; C = R/'BrainPrint_cache'
P = R/'final4_compat/team_hymn'; sys.path.insert(0,str(P/'src'))
import brainprint_pipeline as pipe

def reuse(file, name, env):
    tree = ast.parse(file.read_text(encoding='utf8'))
    tree.body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(tree.body)==1
    exec(compile(tree,str(file),'exec'),env)
    return env[name]

def measure(truth, g, s):
    winner = s[:,1:].argmax(1)+1; accepted = g.argmax(1)!=0
    pred = np.where(accepted,winner,0); known = truth!=0; n = int(known.sum())
    auc = (rankdata(g[:,1:].max(1))[known].sum()-n*(n+1)/2)/(n*(len(g)-n))
    d = dict(AUC=auc,F1=np.mean([2*((pred==k)&(truth==k)).sum()/((pred==k).sum()+(truth==k).sum()) for k in np.union1d(pred,truth)]),unknown_rejection=(~accepted[~known]).mean(),unknown_n=int((~known).sum()))
    for name,mask in [('old',known&(truth<=20)),('new',truth>20)]:
        ok = winner[mask]==truth[mask]; rej = ~accepted[mask]
        d.update({name+'_accuracy':(ok&~rej).mean(),name+'_closed':ok.mean(),name+'_FRR':rej.mean(),name+'_rank_correct_rejected':int((ok&rej).sum()),name+'_n':int(mask.sum())})
    return d

@torch.inference_mode()
def main():
    ref=json.loads((E/'results.json').read_text(encoding='utf8')); zipfile=P.parent/'team_hymn.zip'
    with ZipFile(zipfile) as z:
        for p in [P/'weights.pt',*(P/'src').glob('*.py')]:
            assert z.read(next(i for i in z.namelist() if i.endswith('/'+p.relative_to(P).as_posix())))==p.read_bytes()
    base=pipe.load_model(P,'cuda'); gate,rank=pipe.gate,pipe.rank
    assert all(a['valid'][:,:20].all() and not a['valid'][:,20:].any() for a in base.values())
    m=pd.read_csv(C/'train_metadata.csv'); assert len(m)==11304 and m.path.is_unique and m.path.str.startswith('Train/').all()
    y=m.subject.to_numpy(); metas=m[['task']].to_dict('records'); t=gate._task_indices(metas,len(m)); f=(m.trial/m.groupby(['subject','task']).trial.transform('max')).to_numpy(); early=f<=.6; late=f>.8
    paths={'metadata':C/'train_metadata.csv',**{v:C/f'e262_wave_pair/{v}.npy' for v in ('clean','stress')}}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==ref['hashes'][k] for k,p in paths.items())
    gx={k:torch.from_numpy(np.load(paths[k])).cuda() for k in ('clean','stress')}; rx={k:torch.empty_like(v) for k,v in gx.items()}
    # E262 cache was generated with the gate config. Recreate the same waveform stress for the other branch.
    assert base['gate']['feature']==torch.load(R/'final/team_hymn/weights.pt',weights_only=True)['feature']!=base['rank']['feature']
    perturb=reuse(H.parent/'E262_wave_pair/views.py','perturb',dict(torch=torch)); rng=torch.Generator(device='cuda').manual_seed(26202)
    def raw(p):return np.ascontiguousarray(loadmat(R/'Brainprint_Competition'/p,variable_names=['data'])['data'].T,dtype='float32')
    with ThreadPoolExecutor(8) as pool:
        for i in range(0,len(m),512):
            x=torch.from_numpy(np.stack(list(pool.map(raw,m.path.iloc[i:i+512])))).cuda()
            for k,v in [('clean',x),('stress',perturb(x,rng,.10,.16,.025))]:
                def feat(mod,cfg):return torch.stack([mod._subwindow_feature(w,cfg) for w in v.split(200,-1)],1).mean(1)
                if i==0: assert torch.allclose(feat(gate,base['gate']['feature']),gx[k][:len(x)],atol=1e-5,rtol=1e-5)
                rx[k][i:i+len(x)]=feat(rank,base['rank']['feature'])
    gate._features=rank._features=lambda a,x:x
    folds=[]; effective=t.copy()
    for fold in ref['folds']:
        seed=fold['seed']; old,new,unk=[fold[k] for k in ('old_ids','new_ids','unknown_ids')]; tag=f'seed{seed}'
        perm=np.random.default_rng(seed).permutation(np.arange(1,21)); assert [sorted(v) for v in np.split(perm,[14,17])]==[old,new,unk]
        rows=np.flatnonzero(early&np.isin(y,old)); tt=t[rows]; x=gx['clean'][rows]; mu=torch.stack([x[tt==k].mean(0) for k in range(4)]); sd=torch.stack([x[tt==k].std(0,unbiased=False)+1e-6 for k in range(4)])
        z=(x-mu[tt])/sd[tt]; pm=z.mean(0); z-=pm; b=torch.linalg.eigh(z.T@z)[1][:,-96:].flip([1]).contiguous(); q=z@b
        a=dict(base['gate'],task_mu=mu,task_sd=sd,pca_mean=pm,pca_basis=b,prototypes=torch.zeros_like(base['gate']['prototypes']),valid=torch.zeros_like(base['gate']['valid']))
        for k in range(4):
            for p in old:a['prototypes'][k,p-1]=q[(tt==k)&(y[rows]==p)].mean(0); a['valid'][k,p-1]=True
        br=base['rank']; fit=reuse(H.parent/'E270_new_calibration/run.py','fit',dict(np=np,torch=torch,X=rx['clean'],t=t,y=y,early=early,D='cuda',w=br,tau=br['tau'],beta=br['beta'],pkg=rank,W=H)); r,reg=fit(old,new,tag)
        saved=torch.load(E/f'fits/{tag}.pt',weights_only=True); assert np.array_equal(reg,saved['reg_rows']) and np.array_equal(rows,saved['fit_rows']) and not late[reg].any()
        gate.enroll(a,gx['clean'][reg],None,np.repeat([21,22,23],20)); assert torch.equal(a['valid'],r['valid']) and a['valid'].sum()==68
        torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in a.items()},H/f'{tag}_gate.pt')
        truth=y.copy(); truth[np.isin(y,unk)]=0
        for j,p in enumerate(new):truth[y==p]=21+j
        scenarios={}; fixed={}
        for name,k,miss in [('clean_complete','clean',None),('stress_complete','stress',None),('clean_missing_EC','clean',0),('stress_missing_Imagery','stress',2)]:
            ix=np.flatnonzero(late if miss is None else late&(t!=miss)); assert len(ix)==fold['scenarios'][name]['candidate']['n']
            if miss is not None:effective[ix]=-1
            arts=[dict(v,valid=v['valid'].clone()) for v in (a,r)]
            if miss is not None:
                for art in arts:art['valid'][miss]=False
            for replay in ([True,False] if miss is None and (effective[ix]<0).any() else [True]):
                gs=[]; ss=[]
                for g in np.array_split(ix,(len(ix)+127)//128):
                    meta=[None if replay and effective[j]<0 else metas[j] for j in g]
                    gs.append(gate.predict_batch(arts[0],gx[k][g],meta)); ss.append(rank.predict_batch(arts[1],rx[k][g],meta))
                d=measure(truth[ix],np.concatenate(gs),np.concatenate(ss)); (scenarios if replay else fixed)[name]=d
            fixed.setdefault(name,scenarios[name])
        folds.append(dict(seed=seed,old=old,new=new,unknown=unk,scenarios=scenarios,isolated=fixed)); print('finished',seed,flush=True)
    result=dict(zip_sha256=hashlib.sha256(zipfile.read_bytes()).hexdigest(),tau=base['gate']['tau'],folds=folds)
    (H/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for fold in folds:
        for mode in ('scenarios','isolated'):
            s=fold[mode]; c=s['clean_complete']; v=[c['old_accuracy'],s['stress_complete']['old_accuracy'],c['new_accuracy'],c['unknown_rejection']]; print(fold['seed'],mode,*[round(float(i),6) for i in [*v,np.mean(v),np.mean([d['AUC'] for d in s.values()]),np.mean([d['F1'] for d in s.values()]),c['old_FRR'],c['new_FRR'],c['old_closed'],c['new_closed']]])
if __name__=='__main__':main()
