"""Train-only E260. Reuse E259.evaluate unchanged, substituting its feature projection hook."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'E259_cohort_enroll'
spec=importlib.util.spec_from_file_location('e259',BASE/'run.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
project=b.project

class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.skip=nn.Linear(256,64,bias=False); self.deep=nn.Sequential(nn.Linear(512,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,64))
        nn.init.orthogonal_(self.skip.weight)
    def forward(self,x):
        return F.normalize(self.skip(x[...,:256])+0.15*self.deep(x),dim=-1)

def evaluate(tag,old,new,unknown,seed):
    def learned(fit):
        torch.manual_seed(seed); qw,art=project(fit); pre=((b.X-art['mu'][b.t])/art['sd'][b.t]-art['center'])@art['basis']; x=16*torch.cat([F.normalize(qw),F.normalize(pre)],1)
        net=Encoder().to(x.device); opt=torch.optim.AdamW(net.parameters(),lr=0.0005); rng=np.random.default_rng(seed)
        pools={}
        for task in range(4):
            for who in old:
                ix=torch.where(fit&(b.y==who)&(b.t==task))[0]; blocks=torch.tensor_split(ix[b.order[ix].argsort()],3); assert min(map(len,blocks))>=2
                pools[task,who]=[a.cpu().numpy() for a in blocks]
        with torch.enable_grad():
            for step in range(100):
                task=int(rng.integers(4)); a,c=sorted(rng.choice(3,2,replace=False)); users=rng.choice(old,10,replace=False); support=[]; query=[]
                for who in users:
                    blocks=pools[task,who]; support.append(rng.choice(blocks[a],5,replace=len(blocks[a])<5)); query.append(rng.choice(blocks[c],6,replace=len(blocks[c])<6))
                p=F.normalize(net(x[b.tens(np.stack(support))]).mean(1),dim=-1); z=net(x[b.tens(np.stack(query))]).reshape(-1,64); loss=F.cross_entropy(18*z@p.T,torch.arange(10,device=x.device).repeat_interleave(6)); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); enc=net(x); state={'projection':{k:v.cpu() for k,v in art.items()},'encoder':{k:v.cpu() for k,v in net.state_dict().items()},'seed':seed,'old':old,'new':new,'unknown':unknown}
        torch.save(state,HERE/'weights'/f'{tag}.pt'); return F.pad(enc,(0,192)),art # zeros solely for E259 template buffer compatibility
    b.project=learned
    try:
        with torch.no_grad(): return b.evaluate(old,new,unknown)[0]
    finally: b.project=project

def main():
    (HERE/'weights').mkdir(exist_ok=True); ref=json.loads((BASE/'results.json').read_text()); rows={}; stats={}; keys=['AUC','U95','R90','F1','old_e2e','new_e2e','unknown']; ids=b.ids
    for protocol in ['loso','registration']:
        rs=[]
        for h in range(20):
            rot=ids[h:]+ids[:h]; old,new,unknown=(rot[:14],rot[14:17],rot[17:]) if protocol=='registration' else ([i for i in ids if i!=ids[h]],[],[ids[h]])
            tag=f'{protocol}_{h+1:02}'; r=evaluate(tag,old,new,unknown,26000+h); rs.append(dict(old=old,new=new,unknown_ids=unknown,**r)); print(tag,round(r['AUC'],5),flush=True)
        rows[protocol]=rs; stats[protocol]={k:{'mean':float(np.mean([r[k] for r in rs])),'min':min(r[k] for r in rs)} for k in keys if k in rs[0]}
    def delta(p,k): return stats[p][k]['mean']-ref[p][0][k]['mean']
    gate={k:delta('loso',k)>=0 for k in keys[:3]}; proxy=delta('registration','new_e2e')>0 and all(delta('registration',k)>=0 for k in ['old_e2e','unknown']); passed=all(gate.values()) and proxy
    obj=dict(folds=rows,summary=stats,baseline={p:ref[p] for p in stats},loso_gate=gate,registration_pass=proxy,promote=passed); (HERE/'results.json').write_text(json.dumps(obj,indent=2))
    lines=['# E260：共享残差深度原型编码器','',f'晋级：{passed}；严格 LOSO 门禁：{gate}；注册代理门禁：{proxy}。','', '仅 Train。每折用老用户前 60% 拟合 E259 投影；共享编码器输入白化后和白化前的单位方向，线性跳连加双层非线性分支输出 64 维。100 个 10-way episode，同任务内从早期三等分时间块抽不同块，先块 5-shot 注册、后块 6-query。编码器不学习分数或阈值。', '', '严格 LOSO 为 19 老 + 1 未知；注册代理为 20 次循环轮换，每折 14 老 + 3 新（每人每任务最早 5 条，共 20 条）+ 3 未知。校准只用老用户 60–80%，阈值为固定分数第 10 百分位；评估只用末 20%。U95/R90 分别为晚期已知接受率 95%/90% 对应的未知拒识率，不是部署阈值。F1 为晚期已知闭集 macro-F1，端到端指正确且接受。时间块代理不代表独立跨会话成绩。','', '预设晋级：相对 E259 固定分数，严格 LOSO 三项 ROC 折均值均不下降；注册代理新用户端到端提高且老用户端到端、未知拒识均不下降。下表按折等权；min 为逐指标最差折。','', '|协议|指标|E260 均值|E260 min|E259 固定|差值|E259 cohort|','|---|---|---:|---:|---:|---:|---:|']
    for p in stats:
        for k,s in stats[p].items(): lines.append(f"|{p}|{k}|{s['mean']:.5f}|{s['min']:.5f}|{ref[p][0][k]['mean']:.5f}|{delta(p,k):+.5f}|{ref[p][1][k]['mean']:.5f}|")
    lines+=['','权重 `weights/`，逐折身份、阈值与指标 `results.json`。未读取 Test。']; (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines),flush=True)
if __name__=='__main__': main()
