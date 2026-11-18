"""Fold-refit the deployed old-user score (stored whitened prototype whitened again)."""
import runpy,json
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize
H=Path(__file__).resolve().parent
r=runpy.run_path(str(H/'run.py')); ref=r['ref']; rows=[]
for h in ref.ids:
    fit=(ref.f<=.6)&(ref.y!=h); cal=(ref.f>.6)&(ref.f<=.8)&(ref.y!=h); late=ref.f>.8; people=[i for i in ref.ids if i!=h]
    q,art=ref.project(fit); p,v=ref.templates(q,fit,people); p=normalize(p@art['white'],dim=-1)
    cs,_=ref.score(q,p,v,cal,people); ss,pred=ref.score(q,p,v,late,people); rows.append(r['metrics'](ss[0],pred,ref.y[late].cpu().numpy(),h,np.quantile(cs[0],.10)))
s={k:[np.mean([a[k] for a in rows]),min(a[k] for a in rows)] for k in rows[0]}; print(s); (H/'package_metric.json').write_text(json.dumps(dict(folds=rows,mean_min=s),indent=2))
