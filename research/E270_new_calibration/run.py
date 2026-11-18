"""Train-only nested new-winner calibration. Run with the project venv and -B."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from scipy.io import loadmat


def f1_score(truth, pred, average='macro'):
    return float(np.mean([2*((truth==k)&(pred==k)).sum()/((truth==k).sum()+(pred==k).sum()) for k in np.union1d(truth,pred)]))


def roc_auc_score(known, score):
    n=int(known.sum()); return float((rankdata(score)[known].sum()-n*(n+1)/2)/(n*(len(score)-n)))

H = Path(__file__).resolve().parent
R = H.parents[1]
C = R / 'BrainPrint_cache'
PK = R / 'final2/team_hymn'
D = 'cuda'

def module():
    spec = importlib.util.spec_from_file_location('e270_package', PK/'model.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

raw_pkg, pkg = module(), module()
# Only replace waveform -> cached feature in the in-memory research instance.
pkg._features = lambda art, features: features
w = torch.load(PK/'weights.pt', weights_only=True)
tau, beta = w['tau_score'], w['beta']
m = pd.read_csv(C/'train_metadata.csv')
assert len(m)==11304 and all(m.path.str.startswith('Train/')) and m.path.is_unique
assert m.groupby(['subject','task']).trial.apply(lambda x: x.is_monotonic_increasing and x.is_unique).all()
y = m.subject.to_numpy(); t = pkg._task_indices(m[['task']].to_dict('records'), len(m)); assert min(t)==0 and max(t)==3
f = (m.trial/m.groupby(['subject','task']).trial.transform('max')).to_numpy()
early, middle, late = f<=.6, (f>.6)&(f<=.8), f>.8
X,S = [torch.from_numpy(np.load(C/f'e262_wave_pair/{v}.npy')).cuda() for v in ('clean','stress')]
assert X.shape==S.shape==(len(m),855) and torch.isfinite(X).all() and torch.isfinite(S).all()
W = H/'fits'; W.mkdir(exist_ok=True)
ids = list(range(1,21))

def split(pool, h):
    p = pool[h:]+pool[:h]
    return sorted(p[:-6]), p[-6:-3], p[-3:]

def fit(old,new,tag):
    i=np.flatnonzero(early & np.isin(y,old)); x=X[i]; task=t[i]; yy=y[i]
    mu=torch.stack([x[task==k].mean(0) for k in range(4)])
    sd=torch.stack([x[task==k].std(0).clamp_min(1e-6) for k in range(4)])
    z=(x-mu[task])/sd[task]; pm=z.mean(0); z=z-pm
    v,b=torch.linalg.eigh(z.T@z/(len(z)-1)); b=b[:,v.argsort(descending=True)[:256]].contiguous(); q=z@b
    r=torch.cat([q[yy==p]-q[yy==p].mean(0) for p in old]); cov=r.T@r/(len(r)-len(old))
    v,e=torch.linalg.eigh(.95*cov+.05*cov.diagonal().mean()*torch.eye(256,device=D)); white=(e*v.clamp_min(1e-6).rsqrt())@e.T
    q=q@white; a=dict(device=D,feature=w['feature'], task_mu=mu,task_sd=sd,pca_mean=pm,pca_basis=b,whiten=white,prototypes=q.new_zeros(4,23,4,256),valid=torch.zeros(4,23,device=D,dtype=torch.bool),n_proto=4,tau=tau,beta=beta)
    for p in old:
        for k in range(4):
            rows=q[(yy==p)&(task==k)]; assert len(rows)>=4
            a['prototypes'][k,p-1]=torch.stack([rows[j*len(rows)//4:(j+1)*len(rows)//4].mean(0) for j in range(4)])
            a['valid'][k,p-1]=True
    reg=np.concatenate([np.flatnonzero(early&(y==p)&(t==k))[:5] for p in new for k in range(4)])
    labels=np.repeat([21,22,23],20); assert np.array_equal(y[reg],np.repeat(new,20)) and len(set(reg)&set(i))==0
    pkg.enroll(a,X[reg],None,labels)
    torch.save(dict(artifact={k:v.cpu() if torch.is_tensor(v) else v for k,v in a.items()},old=old,new=new,fit_rows=i.tolist(),reg_rows=reg.tolist()),W/f'{tag}.pt')
    return a,reg

def predict(a,feat,ix):
    return np.concatenate([pkg.predict_batch(a,feat[g],[{'task':v} for v in m.task.iloc[g]]) for g in np.array_split(ix,(len(ix)+127)//128)])

def data(a,feat,ix,new,unk):
    s=predict(a,feat,ix); pred=s[:,1:].argmax(1)+1
    vals=np.sort(s[:,1:],axis=1); gate=vals[:,-1]+beta*(vals[:,-1]-vals[:,-2]); assert np.array_equal(np.where(gate>=tau,pred,0),s.argmax(1))
    truth=y[ix].copy(); truth[np.isin(y[ix],unk)]=0
    for j,p in enumerate(new): truth[y[ix]==p]=21+j
    return pred,gate,truth

def measure(d,offset):
    p,g,y0=d; threshold=np.where(p>=21,np.float32(tau-offset),np.float32(tau)); score=g-threshold; accepted=score>=0
    out=np.where(accepted,p,0); new=y0>=21; unknown=y0==0; known=~unknown; old=known&~new
    return dict(old=float((out[old]==y0[old]).mean()),new=float((out[new]==y0[new]).mean()),URR=float((out[unknown]==0).mean()),AUROC=roc_auc_score(known,score),closed_F1=f1_score(y0[known],p[known],average='macro'),open_F1=f1_score(y0,out,average='macro'))

def calibrate(pool,tag):
    sets=[]; details=[]
    for j,h in enumerate([0,len(pool)//2]):
        old,new,unk=split(pool,h); a,_=fit(old,new,f'{tag}_i{j}'); ix=np.flatnonzero(middle&np.isin(y,new+unk)); sets.append(data(a,X,ix,new,unk)); details.append(dict(old=old,new=new,unknown=unk,mid_rows=ix.tolist()))
    p,g,y0=(np.concatenate(v) for v in zip(*sets)); curve=[]
    for step in range(21):
        offset=step/100; accept=g>=np.where(p>=21,np.float32(tau-offset),np.float32(tau)); curve.append(dict(offset=offset,correct=int(((p==y0)&accept&(y0>=21)).sum()),FA=int((accept&(y0==0)).sum())))
    nu=int(sum(y0==0)); feasible=[v for v in curve if 100*(v['FA']-curve[0]['FA'])<=nu]; best=max(feasible,key=lambda v:(v['correct'],-v['offset']))
    return best['offset'], dict(splits=details,curve=curve,n_unknown=nu,n_new=int(sum(y0>=21)))

def raw(ix):
    return np.stack([loadmat(R/'Brainprint_Competition'/m.path.iloc[k],variable_names=['data'])['data'].T.astype('float32') for k in ix])

def audit(a,reg,old,new,unk):
    # Re-run actual waveform enrollment and predict on early diagnostic rows only.
    b={k:(v.clone() if torch.is_tensor(v) else v) for k,v in a.items()}; b['prototypes'][:,20:]=0; b['valid'][:,20:]=False
    raw_pkg.enroll(b,raw(reg),None,np.repeat([21,22,23],20)); err=(a['prototypes']-b['prototypes']).abs().max().item()
    ix=np.concatenate([np.flatnonzero(early&(y==p)&(t==k))[-2:] for p in old+new+unk for k in range(4)])
    sc=raw_pkg.predict_batch(b,raw(ix),m[['task']].iloc[ix].to_dict('records')); ref=predict(a,X,ix); finite=np.isfinite(ref); se=float(abs(sc[finite]-ref[finite]).max()); match=bool(np.array_equal(sc.argmax(1),ref.argmax(1)) and np.array_equal(np.isfinite(sc),finite))
    return dict(n=len(ix),template_max_abs=err,score_max_abs=se,decisions_match=match,ok=match and err<1e-4 and se<1e-4)

@torch.inference_mode()
def main():
    with ZipFile(PK.parent/'team_hymn.zip') as zz:
        same=all(zz.read(next(p for p in zz.namelist() if p.split('/')[-1]==k))==(PK/k).read_bytes() for k in ('model.py','weights.pt'))
    folds=[]; check=None
    for h in range(20):
        old,new,unk=split(ids,h); off,inner=calibrate(old,f'h{h:02}'); a,reg=fit(old,new,f'h{h:02}_outer')
        if h==0: check=audit(a,reg,old,new,unk)
        row=dict(h=h,old=old,new=new,unknown=unk,offset=off,inner=inner)
        for view,feat in [('clean',X),('stress',S)]:
            d=data(a,feat,np.flatnonzero(late),new,unk); row[view]=[measure(d,0),measure(d,off)]; assert row[view][0]['old']==row[view][1]['old']
        folds.append(row); print(h,off,flush=True)
    keys=['old_clean','old_stress','new_clean','URR','proxy','clean_AUROC','clean_closed_F1','clean_open_F1','stress_AUROC','stress_closed_F1','stress_open_F1']; arr=[]
    for r in folds:
        pair=[]
        for j in (0,1):
            c,s=r['clean'][j],r['stress'][j]; v=[c['old'],s['old'],c['new'],c['URR']]; pair.append(v+[np.mean(v)]+[view[k] for view in (c,s) for k in ('AUROC','closed_F1','open_F1')]);
        arr.append(pair)
    arr=np.asarray(arr); diff=arr[:,1]-arr[:,0]; delta=diff.mean(0); passed=bool(delta[0]>=0 and delta[1]>=0 and delta[2]>=.02 and delta[3]>=-.01 and delta[4]>0 and same and check['ok']); deploy=None
    if passed:
        off,details=calibrate(ids,'all'); deploy=dict(new_winner_offset=off); (H/'deployment.json').write_text(json.dumps(deploy)); (H/'full_calibration.json').write_text(json.dumps(details))
    result=dict(tau=tau,beta=beta,zip_identical=same,audit=check,passed=passed,deployment=deploy,folds=folds,gpu=torch.cuda.get_device_name(),late_count=int(late.sum()),package_hashes={k:hashlib.sha256((PK/k).read_bytes()).hexdigest() for k in ('model.py','weights.pt')})
    (H/'results.json').write_text(json.dumps(result,indent=1),encoding='utf8')
    print('zip',same,'audit',check,'passed',passed,'deploy',deploy); print('metric baseline candidate delta_pp worst_base(h) worst_candidate(h) worst_delta_pp(h)')
    for i,k in enumerate(keys):
        a,b=arr[:,0,i],arr[:,1,i]; d=diff[:,i]; print(k,*(f'{x:.6f}' for x in (a.mean(),b.mean(),d.mean()*100)),f'{a.min():.6f}({a.argmin()}) {b.min():.6f}({b.argmin()}) {100*d.min():.6f}({d.argmin()})')

def verify_wiring():
    # Apply the proposed single-line replacement only in memory on the real package.
    source=(PK/'model.py').read_text(encoding='utf8'); old='reject = (~finite) | (gate < artifact["tau"])'
    new='reject = (~finite) | (gate < torch.where(sim.argmax(1) >= N_BASE, torch.full_like(gate, artifact["tau"] - artifact["new_winner_offset"]), torch.full_like(gate, artifact["tau"])))'
    assert source.count(old)==1
    wired={}; exec(compile(source.replace(old,new),'<E270-memory-only>','exec'),wired)
    rec=torch.load(W/'h00_outer.pt',weights_only=True); a={k:v.cuda() if torch.is_tensor(v) else v for k,v in rec['artifact'].items()}; off=json.loads((H/'deployment.json').read_text())['new_winner_offset']; a['new_winner_offset']=off
    ix=np.concatenate([np.flatnonzero(early&(y==p)&(t==k))[-8:] for p in ids for k in range(4)]); changed=0; new_wins=0
    for g in np.array_split(ix,10):
        xx=raw(g); meta=m[['task']].iloc[g].to_dict('records'); s=raw_pkg.predict_batch(a,xx,meta); out=wired['predict_batch'](a,xx,meta); assert np.array_equal(s[:,1:],out[:,1:]); win=s[:,1:].argmax(1)+1; t2=np.sort(s[:,1:],1)[:,-2:]; gate=t2[:,1]+beta*(t2[:,1]-t2[:,0]); expected=np.where(gate>=np.where(win>=21,np.float32(tau-off),np.float32(tau)),win,0); assert np.array_equal(expected,out.argmax(1)); assert np.array_equal(s[win<=20],out[win<=20]); changed+=int(sum(out.argmax(1)!=s.argmax(1))); new_wins+=int(sum(win>=21))
    print('raw wiring check: rows, new winners, changed decisions:',len(ix),new_wins,changed)

if __name__=='__main__':
    import sys
    if '--verify-wiring' in sys.argv: verify_wiring()
    else: main()
