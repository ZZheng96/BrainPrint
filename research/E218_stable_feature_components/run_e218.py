from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, f1_score

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
CFG=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED=CFG['seed']; torch.manual_seed(SEED); np.random.seed(SEED)
CACHE=OUT/'cache'; MODELS=OUT/'models'; CACHE.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
LOG=open(OUT/'run.log','w',encoding='utf-8'); sys.stdout=LOG

def log(x): print(x,flush=True)
def frac(meta):
    mx=meta.groupby(['subject','task']).trial.transform('max').to_numpy(float); return meta.trial.to_numpy(float)/np.maximum(mx,1)
def load_raw(meta):
    a=[]
    for p in meta.path:
        x=np.asarray(loadmat(ROOT/CFG['data']['root']/str(p),variable_names=['data'])['data'],dtype=np.float32)
        a.append(x.T)
    return torch.as_tensor(np.stack(a),device=DEVICE)
def masks(n=1200,fs=300):
    f=torch.fft.rfftfreq(n,1/fs,device=DEVICE); out=[]
    for i,(lo,hi) in enumerate(CFG['feature']['bands']): out.append((f>=lo)&(f<=hi if i==4 else f<hi))
    return torch.stack(out)
MASK=masks(); TRI=torch.triu_indices(18,18,1,device=DEVICE)
def e65(x):
    x=x-x.mean(-1,keepdim=True); spec=torch.fft.rfft(x,dim=-1); vals=[]
    for m in MASK:
        y=torch.fft.irfft(spec*m[None,None,:],n=x.shape[-1],dim=-1); y=y-y.mean(-1,keepdim=True); sd=y.std(-1,keepdim=True).clamp_min(1e-6); z=y/sd
        c=z@z.transpose(-1,-2)/(x.shape[-1]-1); tr=torch.diagonal(c,dim1=-2,dim2=-1).sum(-1); eye=torch.eye(18,device=DEVICE)
        s=(1-CFG['feature']['shrinkage'])*c+CFG['feature']['shrinkage']*tr[:,None,None]/18*eye
        w,v=torch.linalg.eigh(s); l=(v*w.clamp_min(1e-12).log().unsqueeze(-2))@v.transpose(-1,-2); q=l[:,TRI[0],TRI[1]]; q=q*CFG['feature']['offdiag_multiplier']; vals.append(q)
    return torch.cat(vals,1)
def logbp(x):
    spec=torch.fft.rfft(x-x.mean(-1,keepdim=True),dim=-1); p=spec.real.square()+spec.imag.square(); vals=[]
    f=torch.fft.rfftfreq(x.shape[-1],1/300,device=DEVICE)
    for lo,hi in CFG['feature']['bands']: vals.append(torch.log(p[:,:, (f>=lo)&(f<hi)].mean(-1)+1e-8))
    return torch.cat(vals,1)
def features(x): return torch.cat([e65(x),logbp(x)],1)
def pgd(x,centroid,eps):
    z=x.detach(); rms=(z-z.mean(-1,keepdim=True)).square().mean((1,2)).sqrt().clamp_min(1e-6); bound=eps*rms[:,None,None]; a=z+(torch.rand_like(z)*2-1)*bound
    for _ in range(CFG['perturbation']['steps']):
        a.requires_grad_(True); f=features(a); loss=((f-centroid)**2).mean(); g=torch.autograd.grad(loss,a)[0]; a=(a.detach()+2*eps/CFG['perturbation']['steps']*rms[:,None,None]*g.sign()).clamp(z-bound,z+bound)
    return a.detach()
def random_noise(x,delta,eps):
    rms=(x-x.mean(-1,keepdim=True)).square().mean((1,2)).sqrt().clamp_min(1e-6); lim=eps*rms[:,None,None]; n=torch.rand_like(x)*2-1; n=n/n.square().mean((1,2),keepdim=True).sqrt()*delta.square().mean((1,2),keepdim=True).sqrt(); return (x+n.clamp(-lim,lim)).detach()
def standardize(a,fit):
    mu=a[fit].mean(0); sd=a[fit].std(0).clamp_min(1e-5); return (a-mu)/sd,mu,sd
def score(x,centers,labels):
    d=torch.cdist(x,centers); v,i=d.min(1); return -v.cpu().numpy(),labels[i.cpu().numpy()]
def auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else float('nan')
def choose_features(clean,pg,rand,subjects,fit):
    x=clean[fit]; y=subjects[fit]; mu=x.mean(0); overall=((x-mu)**2).mean(0)+1e-8; within=np.zeros(x.shape[1],np.float32); counts=0
    for s in np.unique(y):
        z=x[y==s]; within+=((z-z.mean(0))**2).mean(0); counts+=1
    sep=(overall/(within/counts+1e-6)); stab=np.exp(-np.abs(pg[fit]-x).mean(0)/(x.std(0)+1e-5)); sens=np.abs(rand[fit]-x).mean(0)/(x.std(0)+1e-5)
    rank=sep*stab/(1+sens); order=np.argsort(-rank); same=order[:CFG['selection']['same_dim']]; low=order[:CFG['selection']['low_dim']]; rng=np.random.default_rng(SEED); rnd=rng.choice(x.shape[1],len(same),False)
    return {'all':np.arange(x.shape[1]),'stable_same':same,'stable_low':low,'random_same':rnd}, {'separation_mean':float(sep.mean()),'stability_mean':float(stab.mean()),'sensitivity_mean':float(sens.mean())}
def run():
    meta=pd.read_csv(ROOT/CFG['data']['metadata']); fr=frac(meta); meta=meta.assign(frac=fr,block=np.clip((fr*5).astype(int),0,4)); meta=meta[meta.task.isin(CFG['tasks'])].reset_index(drop=True)
    rng=np.random.default_rng(SEED); keep=[]
    for (s,t),g in meta.groupby(['subject','task']):
        target=np.linspace(0.3,0.7,int(CFG['sampling']['rows_per_subject_task']))
        ix=[int(np.argmin(np.abs((g.trial.to_numpy()/max(g.trial.max(),1))-q))) for q in target]
        keep.extend(g.iloc[ix].index.tolist())
    meta=meta.loc[sorted(set(keep))].reset_index(drop=True); raw=load_raw(meta); subjects=meta.subject.to_numpy(int); log(f'device={DEVICE}, rows={len(meta)}, dim=945')
    folds=[]; component=[]
    for held in sorted(np.unique(subjects)):
        outer=subjects!=held; fit=outer&(meta.frac.to_numpy()<=.6); evalm=outer&(meta.frac.to_numpy()>.6)&(meta.frac.to_numpy()<=.8); unknown=(subjects==held)&(meta.frac.to_numpy()>.6)&(meta.frac.to_numpy()<=.8)
        with torch.no_grad(): clean=features(raw).detach(); base=clean[fit].mean(0)
        advs=[]; rands=[]
        for eps in CFG['perturbation']['eps']:
            adv=pgd(raw,base,eps); rnd=random_noise(raw,adv-raw,eps); advs.append(features(adv).detach()); rands.append(features(rnd).detach())
        pg=torch.stack(advs).mean(0); rd=torch.stack(rands).mean(0); sel,diag=choose_features(clean.cpu().numpy(),pg.cpu().numpy(),rd.cpu().numpy(),subjects,fit); component.append({'held_subject':int(held),'n_fit':int(fit.sum()),'n_unknown':int(unknown.sum()),'diagnostics':diag,'selected':{k:v.tolist() for k,v in sel.items()}})
        fold_metrics={}
        for name,ix in sel.items():
            z,mu,sd=standardize(clean,fit); z=z[:,ix]; centers=torch.stack([z[fit&(subjects==s)].mean(0) for s in sorted(np.unique(subjects[outer]))]); labels=np.array(sorted(np.unique(subjects[outer])))
            sk,pk=score(z[evalm],centers,labels); su,pu=score(z[unknown],centers,labels); ref=z[fit]; sr,pr=score(ref,centers,labels); tau=float(np.quantile(sr,1-CFG['protocol']['acceptance'])); pred_known=np.where(sk>=tau,pk,0); pred_unknown=np.where(su>=tau,pu,0); yk=subjects[evalm]; identity_correct=(pred_known==yk); fold_metrics[name]={'auc':auc(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[sk,su]),'f1':float(f1_score(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[sk>=tau,su>=tau])) if (len(sk)+len(su)) else 0.0,'known_acceptance':float((sk>=tau).mean()) if len(sk) else None,'unknown_rejection':float((su<tau).mean()) if len(su) else None,'identity_correct_but_rejected':float(((pk==yk)&(sk<tau)).mean()) if len(yk) else None,'threshold':tau}
        folds.append({'held_subject':int(held),'metrics':fold_metrics}); log(f'held={held} done')
    summary={}
    for n in ['all','stable_same','stable_low','random_same']:
        vals=[f['metrics'][n] for f in folds]; summary[n]={k:float(np.nanmean([v[k] for v in vals])) for k in vals[0] if isinstance(vals[0][k],(int,float))}
    result={'experiment':CFG['experiment'],'device':str(DEVICE),'diagnostic_subset':True,'n_rows':len(meta),'feature_dim':945,'eps':CFG['perturbation']['eps'],'summary':summary,'folds':folds,'component_ledger':component,'official_test':{'AUC':'N/A','F1':'N/A'},'limitations':['Train metadata contains registered identities only; LOSO held identity is used as unknown proxy.','PGD is untargeted feature-distance maximization on a fixed outer-train centroid; diagnostic subset capped at two rows per subject-task.']}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');
    goals=np.array(['feature_distance_pgd' for _ in CFG['perturbation']['eps']],dtype='U32')
    np.savez_compressed(CACHE/'feature_pairs.npz',clean=clean.cpu().numpy(),pgd=pg.cpu().numpy(),random=rd.cpu().numpy(),row_id=np.arange(len(meta)),subject=subjects,task=meta.task.to_numpy(),trial=meta.trial.to_numpy(),seed=np.full(len(meta),SEED),eps=np.asarray(CFG['perturbation']['eps']),goal=goals)
    np.savez_compressed(MODELS/'e218_last_fold_state.npz',selected=np.array([component[-1]['selected']['stable_same']],dtype=object))
    lines=['# E218 稳定特征成分诊断实验','',f'- 设备：`{DEVICE}`；诊断子集：{len(meta)} 行；特征维度 945（E65 855 + E146 log-band-power 90）。','- 严格 outer LOSO：每折成分筛选仅使用 outer-train fit rows；held identity 未参与筛选或阈值。','- PGD：eps=0.005/0.01/0.02；随机噪声按配对扰动幅度生成；PGD 为固定外训质心的特征距离最大化诊断。','', '|方案|AUC|F1|known acceptance|unknown rejection|身份正确但拒绝|','|---|---:|---:|---:|---:|---:|']
    for n,v in summary.items(): lines.append(f"|{n}|{v['auc']:.6f}|{v['f1']:.6f}|{v['known_acceptance']:.6f}|{v['unknown_rejection']:.6f}|{v['identity_correct_but_rejected']:.6f}|")
    lines += ['', '- 官方 Test AUC/F1：N/A（无身份标签，未用 Test 选型）。','- 冻结全 Train Test 的 S1 rejection、S2 acceptance：N/A；本诊断未将无标签 Test 作为评估样本。','- 真实限制：原始数据仅提供 Train 注册身份标签，未知身份采用严格 LOSO held identity 代理；完整 PGD 受计算量限制采用每个 subject-task 两行诊断子集。','- 产物：`results.json` 保存每折指标和成分账本；`cache/feature_pairs.npz` 保存配对特征；`models/e218_last_fold_state.npz` 保存最近折筛选状态。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); log('completed')
if __name__=='__main__':
    try: run()
    finally: LOG.close()
