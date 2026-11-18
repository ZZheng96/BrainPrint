"""Verify the submitted ZIP; optionally compare an original ZIP passed as argv[1]."""
import importlib.util
import os
from pathlib import Path
import pickletools
import re
import sys
import tempfile
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
BAD = re.compile(r'final|compat|(?<![a-z0-9])E\d{2,}|[a-z]:[\\/]|/home/|/Users/|_v\d+', re.I)
FILES = 'README.md enroll.py inference.py model.py model/weights.pt requirements.txt src/team_hymn_geometry.py team.json weights.pt'.split()


def module(path):
    spec = importlib.util.spec_from_file_location('isolated_' + path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify(t):
    p = t / 'team_hymn'
    with zipfile.ZipFile(ROOT / 'team_hymn.zip') as z:
        assert z.testzip() is None and sorted(z.namelist()) == sorted('team_hymn/' + f for f in FILES)
        for f in z.namelist():
            data = z.read(f)
            assert data == (ROOT / f).read_bytes() and not BAD.search(f), f
            if not f.endswith('.pt'):
                assert not BAD.search(data.decode()), (f, BAD.search(data.decode()))
        z.extractall(t)
    with zipfile.ZipFile(p / 'weights.pt') as z:
        assert all(not BAD.search(f) for f in z.namelist())
        assert all(not BAD.search(a) for _, a, _ in pickletools.genops(z.read('weights/data.pkl')) if isinstance(a, str) and a != 'torch._utils _rebuild_tensor_v2')
    assert (p / 'weights.pt').read_bytes() == (p / 'model/weights.pt').read_bytes()
    w = torch.load(p / 'weights.pt', weights_only=True)
    if BASE:
        with zipfile.ZipFile(BASE) as z:
            z.extractall(t / 'base')
        op = t / 'base/team_hymn'
        before = module(op / 'model.py')
        prev = torch.load(op / 'weights.pt', weights_only=True)
        del prev['protocol']
        assert w.keys() == prev.keys()
        for k, v in prev.items():
            assert torch.equal(v, w[k]) if torch.is_tensor(v) else v == w[k]
    after, split, en = [module(p / f) for f in ('model.py', 'inference.py', 'enroll.py')]
    assert Path(after.predict_batch.__wrapped__.__code__.co_filename).is_relative_to(p)
    exec((p / 'README.md').read_text('utf8').split('```python\n')[1].split('```')[0], {})
    rng = np.random.default_rng(29)
    q = rng.standard_normal((128, 18, 1200), dtype=np.float32)
    r = np.repeat(q[:3], 20, axis=0)
    y = np.repeat([21, 22, 23], 20).astype(np.int64)
    rb, qb, yb = r.copy(), q.copy(), y.copy()
    meta = [{'task': s} for s in ('EC','EO','Imagery','SSVEP')] * 32
    for device in ('cpu','cuda:0'):
        for mod, reg in ((after, after.enroll), (split, en.enroll)):
            for d in (p, p / 'model'):
                a = mod.load_model(d, device)
                assert str(a['device']) == device
                frozen = {k:v.clone() for k,v in a.items() if torch.is_tensor(v)}
                b = reg(a, r, None, y)
                assert all(torch.equal(v,a[k]) for k,v in frozen.items())
                assert all(torch.equal(v,b[k]) for k,v in frozen.items() if k not in ('valid','prototypes'))
                assert torch.equal(b['prototypes'][:,:20],a['prototypes'][:,:20])
                assert not a['valid'][:,20:].any() and b['valid'].all()
                for n in (1,128):
                    pre = mod.predict_batch(a,q[:n],None)
                    post = mod.predict_batch(b,q[:n],None)
                    assert post.shape == (n,24) and np.isfinite(post).all() and np.isneginf(pre[:,21:]).all()
                    assert np.array_equal(pre[:,1:21],post[:,1:21]) and np.array_equal(post.argmax(1)[:min(3,n)],np.arange(21,21+min(3,n)))
                    ids = post[:,1:].argmax(1)+1
                    top = np.sort(post[:,1:],axis=1)[:,-2:]
                    gate = top[:,1]+.5*(top[:,1]-top[:,0])
                    threshold_new = np.where(ids > 20, a['tau']-.06, a['tau'])
                    threshold_old = np.where(ids > 20, a['tau']-.03, a['tau'])
                    expected_new = np.where(gate < threshold_new, 0, ids)
                    expected_old = np.where(gate < threshold_old, 0, ids)
                    assert np.array_equal(post.argmax(1), expected_new)
                    changed = expected_new != expected_old
                    assert np.all(ids[changed] >= 21)
                    assert np.all((gate[changed] >= a['tau']-.06) & (gate[changed] < a['tau']-.03))
                    assert np.array_equal(post.argmax(1)[~changed], expected_old[~changed])
                fallback = mod.predict_batch(b,q[:1],[{'task':'p/SSVEP','task_folder':'EC','label':1}])
                assert np.array_equal(fallback,mod.predict_batch(b,q[:1],None))
        if BASE:
            ref = before.load_model(op,device)
            assert np.array_equal(before.predict_batch(ref,q,meta),after.predict_batch(a,q,meta))
            ref = before.enroll(ref,r,None,y)
            assert np.array_equal(before.predict_batch(ref,q,meta),after.predict_batch(b,q,meta))
        print(device, 'passed')
    assert np.array_equal(rb,r) and np.array_equal(qb,q) and np.array_equal(yb,y)


if __name__ == '__main__':
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as directory:
        cwd = Path.cwd()
        try:
            os.chdir(directory)
            verify(Path(directory).resolve())
        finally:
            os.chdir(cwd)
    print('ZIP and README passed')
