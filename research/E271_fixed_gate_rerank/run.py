"""Paired final gate / final2 rank experiment. Run with .venv Python -B."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

H = Path(__file__).resolve().parent
R = H.parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

e = load('outer', R/'research/E270_new_calibration/run.py')
g = load('gate_package', R/'final/team_hymn/model.py')
g._features = lambda a, feat: feat
X, t, y = e.X, e.t, e.y

def scores(mod, a, feat, ix):
    return np.concatenate([mod.predict_batch(a, feat[k], e.m[['task']].iloc[k].to_dict('records'))[:, 1:] for k in np.array_split(ix, (len(ix)+127)//128)])

@torch.inference_mode()
def main():
    folds = []; weights = []
    gate_cfg=torch.load(R/'final/team_hymn/weights.pt', weights_only=True)['feature']  # E262 views.py built the shared cache with final's cfg
    for h in range(20):
        path = e.W/f'h{h:02}_outer.pt'; rec = torch.load(path, weights_only=True); a = {k:v.cuda() if torch.is_tensor(v) else v for k,v in rec['artifact'].items()}
        old,new,unknown = e.split(e.ids,h); assert (rec['old'],rec['new']) == (old,new)
        fit = np.flatnonzero(e.early & np.isin(y,old)); reg = np.concatenate([np.flatnonzero(e.early & (y==p) & (t==j))[:5] for p in new for j in range(4)])
        assert np.array_equal(fit,rec['fit_rows']) and np.array_equal(reg,rec['reg_rows']) and len(np.unique(reg))==60
        labels=np.repeat([21,22,23],20); saved=a['prototypes'][:,20:].clone(); a['prototypes'][:,20:].zero_(); e.pkg.enroll(a,X[reg],None,labels); regerr=float((saved-a['prototypes'][:,20:]).abs().max()); assert regerr<1e-4
        a['prototypes'][:,20:]=saved
        x=X[fit]; tt=t[fit]; mu=torch.stack([x[tt==k].mean(0) for k in range(4)]); sd=torch.stack([x[tt==k].std(0,unbiased=False)+1e-6 for k in range(4)]); z=(x-mu[tt])/sd[tt]; pm=z.mean(0); z=z-pm
        basis=torch.linalg.eigh(z.T@z)[1][:,-96:].flip([1]).contiguous(); q=z@basis
        b=dict(task_mu=mu,task_sd=sd,pca_mean=pm,pca_basis=basis,feature=gate_cfg,device='cuda',tau=0.,valid=torch.zeros_like(a['valid']),prototypes=q.new_zeros(4,23,96))
        olderr=0.
        for k in range(4):
            for p in old:
                mask=(tt==k)&(y[fit]==p); b['prototypes'][k,p-1]=q[mask].mean(0); b['valid'][k,p-1]=True
                qr=e.pkg._project(a,x[mask],k)@a['whiten']; n=len(qr); ref=torch.stack([qr[j*n//4:(j+1)*n//4].mean(0) for j in range(4)]); olderr=max(olderr,float((ref-a['prototypes'][k,p-1]).abs().max()))
        assert olderr<1e-4
        g.enroll(b,X[reg],None,labels); assert torch.equal(a['valid'],b['valid']) and int(b['valid'].sum())==68
        mid=np.flatnonzero(e.middle & np.isin(y,old+new)); tau=float(np.quantile(-scores(g,b,X,mid).max(1),.9)); b['tau']=tau
        row=dict(h=h,old=old,new=new,unknown=unknown,tau=tau,n_cal=len(mid),registration_error=regerr,old_white_error=olderr)
        ix=np.flatnonzero(e.late); truth=y[ix].copy(); truth[np.isin(y[ix],unknown)]=0
        for j,p in enumerate(new): truth[y[ix]==p]=21+j
        for name,feat in [('clean',X),('stress',e.S)]:
            ds=scores(g,b,feat,ix); score=ds.max(1); accept=score > -np.float32(tau); base=np.where(accept,ds.argmax(1)+1,0); cand=np.where(accept,scores(e.pkg,a,feat,ix).argmax(1)+1,0); assert np.array_equal(base!=0,cand!=0)
            known=truth!=0; v=dict(URR=float((~accept[~known]).mean()),AUC=e.roc_auc_score(known,score),base_F1=e.f1_score(truth,base),cand_F1=e.f1_score(truth,cand))
            for group,mask in [('old',known&(truth<=20)),('new',truth>=21)]:
                ok=base[mask]==truth[mask]; now=cand[mask]==truth[mask]; v.update({group+'_base':float(ok.mean()),group+'_cand':float(now.mean()),group+'_fixed':int((~ok&now).sum()),group+'_broken':int((ok&~now).sum()),group+'_n':int(mask.sum())})
            row[name]=v
        weights.append(dict(gate={k:v.cpu() if torch.is_tensor(v) else v for k,v in b.items()},rank_path=str(path.relative_to(R)),rank_sha256=hashlib.sha256(path.read_bytes()).hexdigest())); folds.append(row)
    torch.save(weights,H/'weights.pt'); summary={}
    for view in ['clean','stress']:
        vals={key:np.array([r[view][key] for r in folds]) for key in folds[0][view]}; summary[view]={k:(int(v.sum()) if k.endswith(('_n','_fixed','_broken')) else float(v.mean())) for k,v in vals.items()}
        for key in ['old','new','F1']:
            v=vals['cand_F1'] if key=='F1' else vals[key+'_cand']; d=v-(vals['base_F1'] if key=='F1' else vals[key+'_base']); summary[view][key+'_worst']=[int(v.argmin()),float(v.min()),int(d.argmin()),float(d.min())]
    result=dict(gpu=torch.cuda.get_device_name(),late=int(e.late.sum()),calibration='90th percentile of middle clean nearest distance among old+new known; strict package argmax tie rejection',folds=folds,summary=summary)
    (H/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(result['gpu'],result['late'],json.dumps(summary),max(r['registration_error'] for r in folds),max(r['old_white_error'] for r in folds))

if __name__=='__main__': main()
