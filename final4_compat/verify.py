"""Verify the actual submission ZIP against the two immutable reference ZIPs."""
from pathlib import Path
from io import BytesIO
import hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, zipfile
import numpy as np
import torch


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    assert a.keys() == b.keys()
    for k,v in a.items():
        if torch.is_tensor(v):
            assert v.dtype == b[k].dtype and v.device == b[k].device and torch.equal(v,b[k]), k
        else:
            assert v == b[k], k


def clean(text):
    assert not re.search(r'(?i:final|compat|hymn4|[a-z]:[\\/]|/(?:home|Users|mnt|tmp|workspace)/)|(?<![a-zA-Z0-9])E\d+', text), text[:120]


def audit(path, out):
    from build import names
    with zipfile.ZipFile(path) as z:
        expected = {'team_hymn/'+n for n in names}
        assert set(z.namelist()) == expected and len(z.namelist()) == len(expected) == 11
        assert {p.relative_to(out).as_posix() for p in (out/'team_hymn').rglob('*') if p.is_file()} == expected
        for n in z.namelist():
            clean(n); raw=z.read(n); assert raw==(out/n).read_bytes()
            if not n.endswith('.pt'): clean(raw.decode('utf8'))
        assert z.read('team_hymn/model/weights.pt') == z.read('team_hymn/weights.pt')
        raw=z.read('team_hymn/weights.pt'); assert len(raw)<10**9
        with zipfile.ZipFile(BytesIO(raw)) as inner:
            for n in inner.namelist(): clean(n)
        w=torch.load(BytesIO(raw),weights_only=True,map_location='cpu')
        def strings(v):
            if isinstance(v,str): clean(v)
            elif isinstance(v,dict):
                for k,u in v.items(): clean(k); strings(u)
            elif isinstance(v,(list,tuple)):
                for u in v: strings(u)
        strings(w)
        info=json.loads(z.read('team_hymn/team.json'))
        assert all(info[k] for k in 'team_id team_name institution contact architecture description weights new_user_support'.split())
        assert info['team_id']=='team_hymn' and info['weights']=='weights.pt' and info['new_user_support'] is True
        doc=z.read('team_hymn/README.md').decode(); assert doc.startswith('## 1. 方法简述') and re.findall(r'^## (\d+)\.',doc,re.M)==list('1234567')


def worker():
    torch.set_num_threads(4)
    cwd=Path.cwd().resolve(); x=np.load('x.npy'); reg=np.load('reg.npy'); labels=np.repeat([21,22,23],20).astype('int64'); x0=x.copy(); reg0=reg.copy()
    meta=[{'task':['Task1','Task2','Task3','Task4','unmapped'][i%5]} for i in range(128)]
    orig=[cwd/p/'team_hymn' for p in ('final','final2')]; refs=[module(p/'model.py') for p in orig]
    p=cwd/'candidate'/'team_hymn'; entry=module(p/'model.py'); split=module(p/'inference.py'); enroll=module(p/'enroll.py')
    raw=torch.load(p/'weights.pt',weights_only=True)
    assert set(raw)=={'euclidean_gate','cosine_rank'}
    for d,(key,skip) in zip(orig,[('euclidean_gate',('format','source')),('cosine_rank',('protocol',))]):
        source=torch.load(d/'weights.pt',weights_only=True); w=raw[key]; assert source.keys()==w.keys()
        for k in skip: assert isinstance(w[k],str) and isinstance(source[k],str)
        same({k:v for k,v in source.items() if k not in skip},{k:v for k,v in w.items() if k not in skip})
        m=sys.modules[key]; assert Path(m.__file__).resolve().parent==p/'src'
        assert [m._task_index(v) for v in ({'task_folder':'Task2'},{'task':'runs/SSVEP/trial.mat'},{'filename':'EC'},{'order':1},{'task':'Task2'})]==[-1,-1,-1,-1,1]
    # Run the published example in a fresh interpreter, without reference imports.
    code=(p/'README.md').read_text(encoding='utf8').split('```python\n')[1].split('```')[0]
    example=subprocess.check_output([sys.executable,'-I','-c',"import sys,torch;torch.set_num_threads(4);sys.path.insert(0,'.')\n"+code],cwd=p,text=True).strip()
    results=[]; cpu={}
    for dev in ('cpu','cuda:0'):
        r=[m.load_model(d,dev) for m,d in zip(refs,orig)]
        cases=[(e,fn,e.load_model(d,dev)) for e,fn in [(entry,entry.enroll),(split,enroll.enroll)] for d in (p,p/'model')]
        edge=entry.load_model(p,dev); edge['rank']['valid'].fill_(False)
        y=entry.predict_batch(edge,np.zeros_like(x[:1]),None); assert y[0,0]==0 and np.isneginf(y[0,1:]).all()
        frozen=entry.load_model(p,dev)
        for phase in ('before','after'):
            if phase=='after':
                for m,a in zip(refs,r): m.enroll(a,reg,meta[:60],labels)
                for _,fn,a in cases:
                    assert fn(a,reg,meta[:60],labels) is a
                    for k in ('gate','rank'):
                        for f in ('prototypes','valid'):
                            assert torch.equal(a[k][f][:,:20],frozen[k][f][:,:20])
                        same({f:v for f,v in a[k].items() if f not in ('prototypes','valid')},{f:v for f,v in frozen[k].items() if f not in ('prototypes','valid')})
                        assert a[k]['valid'][:,20:].all()
                np.testing.assert_array_equal(reg,reg0)
            for n in (1,7,128):
                g,s=[m.predict_batch(a,x[:n],meta[:n]) for m,a in zip(refs,r)]
                gate=g.argmax(1); rank=s[:,1:].argmax(1)+1; expected=np.where(gate==0,0,rank)
                if dev=='cpu': cpu[phase,n]=expected
                for e,_,a in cases:
                    assert a['gate']['device']==torch.device(dev)==a['rank']['device']
                    y=e.predict_batch(a,x[:n],meta[:n]); assert y.shape==(n,24) and y.dtype==np.float32
                    assert not np.isnan(y).any() and np.isfinite(y).any(1).all()
                    if phase=='before': assert np.isneginf(y[:,21:]).all()
                    np.testing.assert_array_equal(y[:,1:],s[:,1:]); np.testing.assert_array_equal(y.argmax(1),expected)
                    np.testing.assert_array_equal(y[:,0],s[:,1:].max(1)+np.where(gate==0,1,-1).astype('float32'))
                    for k,b in zip(('gate','rank'),r): same(a[k],b)
                np.testing.assert_array_equal(x,x0)
                results.append([dev,phase,n,int((gate==0).sum()),int(((expected>0)&(expected<=20)).sum()),int((expected>=21).sum()),int((expected!=cpu[phase,n]).sum())])
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(),rows=results,example=example)))


def main():
    from scipy.io import loadmat
    import scipy
    out=Path(__file__).resolve().parent; root=out.parent; hashes={}; sha=lambda b:hashlib.sha256(b).hexdigest()
    audit(out/'team_hymn.zip',out)
    with tempfile.TemporaryDirectory() as t:
        tmp=Path(t); assert not tmp.is_relative_to(root)
        for name in ('final','final2',out.name):
            path=root/name/'team_hymn.zip'; hashes[name+'/team_hymn.zip']=sha(path.read_bytes())
            with zipfile.ZipFile(path) as z:
                for k in z.namelist():
                    if k.endswith('weights.pt'): hashes[name+'/'+k]=sha(z.read(k))
                z.extractall(tmp/('candidate' if name==out.name else name))
        rng=np.random.default_rng(4104); x=rng.normal(size=(128,18,1200)).astype('float32'); x[:4]*=1e-7
        trials=sorted((root/'Brainprint_Competition'/'Train').glob('Subject_*/Task*/trial_*.mat'))
        for i,path in zip(range(8,128,2),rng.choice(trials,60,replace=False)): x[i]=loadmat(path)['data'].T
        np.save(tmp/'x.npy',x); np.save(tmp/'reg.npy',rng.normal(size=(60,18,1200)).astype('float32')); shutil.copyfile(__file__,tmp/'verify.py')
        res=json.loads(subprocess.check_output([sys.executable,'-I',str(tmp/'verify.py'),'worker'],cwd=tmp,text=True))
    lines=['# 双分支提交包验证', '', '结论：实际 ZIP 的接口与原算法对照通过。AUC：未评估；F1：未评估。无隐藏标签，不将一致性测试当作识别性能。', '', '## 决赛说明条款对照', '', '|说明行号 / 条款|核对依据与边界|','|---|---|',
    '|1–23，一、二|README 第 3 节记录官方 Train 来源、20 人、11,304 条及预处理；在线特征保留全部 18 通道，300 Hz × 4 秒。不重复训练，未审计历史训练过程。|',
    '|24–61，三、四|旧用户直接预测；三名新用户显式注册，每人 20 条；第 0 列由评测方映射 -1。四类隐藏场景没有本地真值。|',
    f'|62–74、168–179，五、6.4|实测 Windows、Python {sys.version.split()[0]}、torch {torch.__version__}、NumPy {np.__version__}、CUDA runtime {torch.version.cuda}、{res["gpu"]}。SciPy {scipy.__version__} 仅供包外读取 Train；包内依赖仅 torch、numpy 与标准库，不安装依赖。未实测 CentOS 7.8 / Xeon 6348 / RTX 4090。|',
    f'|75–117，两个 6.1 与第一处 6.2|{(out/"team_hymn.zip").stat().st_size:,} 字节 ZIP，单层队伍目录、唯一根 model.py；白名单 11 成员逐字节匹配源树。两份相同权重各 {(out/"team_hymn/weights.pt").stat().st_size:,} 字节，均低于 1 GB 建议值；无数据、缓存或 Docker。|',
    '|118–167，6.2/6.3|CPU 与 cuda:0，各测两组接口 × 两处权重；先预测，再注册，再预测。注册后仅 21–23 原型和掩码可变，旧模板及冻结参数不变。输出 float32、24 列、每行有有限值、无 NaN，输入未改；注册前新用户列为 -inf。|',
    '|180–201，6.5|team.json 八字段、队伍 ID 与权重引用核对；README 从七项说明第一项开篇。文件名、全部文本、反序列化键和字符串、权重归档成员名扫描通过。|',
    '|202–212，6.6|人工检查入口调用链：只从传入目录读取权重，注册与预测不读写文件、不训练、不访问网络/API、不起子进程；不使用测试真值、文件名、目录、序号推断身份。显式任务可用；伪路径、文件名、顺序字段测试不解析。未执行系统调用级沙箱审计。|',
    '|213–240，6.7/6.8|仓库外临时解压实际包，隔离解释器核验导入、weights_only 加载、60 条注册、N=1/7/128、全部标签区间。另起干净进程逐字执行包内 README 示例，输出 `'+res['example']+'`。未验证官方 infer.py、时限或资源峰值。|',
    '|241–257，七、八|不使用 Test、隐藏元信息或测试适配；准确率、未知拒识率、四场景成绩及官方复核均未验证。截止时间、包大小硬限制、校验方式与最终评分以正式通知为准。|', '', '## 数值与算法对照', '', '门控/排序分支标准差下限分别为 1e-12/1e-6，特征值下限 1e-12/1e-8。距离等于阈值时第 0 列优先；排序无有效身份时，两设备零波形实测第 0 列为 0，其余为 -inf。', '', '与两个原 ZIP 的序列化对象逐字段比较，仅替换顶层名称及三个说明字符串；所有张量类型、数值与其他字段相同。对照表每行涵盖四种入口/权重组合：1–23 列与原排序逐元素相等，列 0 与原欧氏门控对应的最高身份分 ±1 相等，最终 argmax 无差异。注册波形为独立生成的 60 条合成数据；预测为 60 条 Train 波形与 68 条合成波形，含低振幅、未知任务。不是跨时段性能实验。', '', '|设备|注册阶段|批量|拒识|旧 ID|新 ID|相对 CPU 判决差异|','|---|---|---:|---:|---:|---:|---:|']
    lines+=['|'+'|'.join(map(str,row))+'|' for row in res['rows']]+['','## SHA-256','']+[f'- `{k}`：`{v}`' for k,v in hashes.items()]+['','重现：项目 `.venv` 中依次执行 `python final4_compat/build.py` 和 `python final4_compat/verify.py`。只提交 `final4_compat/team_hymn.zip`。']
    (out/'验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); print(res,hashes)


if __name__=='__main__':
    worker() if sys.argv[1:]==['worker'] else main()
