"""本地注册与同批次留出评估；--restore-only 仅恢复，不读取注册样本。"""
import argparse
import copy
import csv
import hashlib
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / 'final3_compat/team_hymn'
DATA = ROOT / 'Brainprint_Competition/new_user_training_20261119'
DELTA = HERE / 'private/enrollment.pt'
IDS = [21, 22, 23]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG))
from model import enroll, load_model, predict_batch


def base_hash():
    archive = PKG.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert name.startswith('team_hymn/') and '..' not in Path(name).parts
            if not name.endswith('/'):
                assert z.read(name) == (PKG.parent / name).read_bytes(), name
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def data():
    with (DATA / 'enroll_index.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    paths = [(DATA / r['file']).resolve() for r in rows]
    assert len(rows) == len(set(paths)) == len({r['sample_id'] for r in rows}) == 60
    assert set(paths) == set((DATA / 'enroll').glob('*.mat'))
    y = np.array([int(r['label']) for r in rows])
    assert Counter(y) == dict.fromkeys(IDS, 20)
    xs, signals = [], set()
    for path in paths:
        assert path.parent == (DATA / 'enroll').resolve()
        a = loadmat(path, variable_names=['data'])['data']
        assert a.shape == (1200, 18) and a.dtype == np.float64 and np.isfinite(a).all()
        signals.add(hashlib.sha256(a.tobytes()).digest())
        xs.append(a.T.astype(np.float32))
    x = np.stack(xs)
    assert len(signals) == 60 and x.shape == (60, 18, 1200) and np.isfinite(x).all()
    return x, y


def equal(a, b):
    return torch.equal(a, b) if torch.is_tensor(a) else a == b


def register(base, frozen, x, y):
    model = enroll(base, x, None, y)
    assert base.keys() == model.keys() == frozen.keys()
    for key, value in frozen.items():
        assert equal(value, base[key]), key
        if key in ('prototypes', 'valid'):
            assert torch.equal(value[:, :20], model[key][:, :20]), key
        else:
            assert equal(value, model[key]), key
    assert not base['valid'][:, 20:].any() and model['valid'].all()
    assert torch.isfinite(model['prototypes']).all()
    return model


def restore():
    saved = torch.load(DELTA, map_location='cpu', weights_only=True)
    assert saved['base_sha256'] == base_hash() and saved['labels'] == IDS
    model = load_model(PKG, 'cuda:0')
    for key in ('prototypes', 'valid'):
        target = model[key][:, 20:23]
        assert target.shape == saved[key].shape and target.dtype == saved[key].dtype
        target.copy_(saved[key])
    assert model['valid'].all() and torch.isfinite(model['prototypes']).all()
    return model


def run():
    digest = base_hash()
    x, y = data()
    base = load_model(PKG, 'cuda:0')
    assert base['device'].type == 'cuda'
    frozen = copy.deepcopy(base)
    pred = np.full(60, -1)
    for i, (train, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=276).split(x, y), 1):
        assert Counter(y[val]) == dict.fromkeys(IDS, 4) and np.all(pred[val] == -1)
        m = register(base, frozen, x[train], y[train])
        scores = predict_batch(m, x[val], None)
        assert scores.shape == (12, 24) and np.isfinite(scores).all()
        assert np.array_equal(scores[:, 1:21], predict_batch(base, x[val], None)[:, 1:21])
        pred[val] = scores.argmax(1)
        print('fold', i, 'correct', (pred[val] == y[val]).sum(), '/12')
    assert np.all(pred >= 0)
    print('correct/reject/old/other_new', (pred == y).sum(), (pred == 0).sum(),
          ((pred > 0) & (pred <= 20)).sum(), ((pred > 20) & (pred != y)).sum())
    print('accuracy', np.mean(pred == y), 'macro-F1', f1_score(y, pred, labels=IDS, average='macro'), 'AUC N/A')
    full = register(base, frozen, x, y)
    DELTA.parent.mkdir(exist_ok=True)
    saved = {k: full[k][:, 20:23].detach().cpu().contiguous().clone() for k in ('prototypes', 'valid')}
    torch.save(dict(saved, labels=IDS, base_sha256=digest), DELTA)
    restored = restore()
    assert all(equal(full[k], restored[k]) for k in full)
    scores = predict_batch(full, x, None)
    assert np.array_equal(scores[:, 1:21], predict_batch(base, x, None)[:, 1:21])
    assert np.array_equal(scores, predict_batch(restored, x, None))
    print('restore verified; base ZIP SHA-256:', digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-only', action='store_true')
    args = parser.parse_args()
    if not __debug__ or not torch.cuda.is_available():
        raise RuntimeError('需要启用断言并使用 CUDA')
    if args.restore_only:
        restore()
        print('已恢复本地注册模型')
    else:
        run()
