"""Build three row-aligned Train waveform views; run directly with CUDA."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'final/team_hymn'
sys.path.insert(0, str(PKG))
from model import _subwindow_feature

DATA = ROOT / 'Brainprint_Competition'
CACHE = ROOT / 'BrainPrint_cache'
OUT = CACHE / 'e262_wave_pair'
# seed, uniform gain half-width, fraction of common average, noise / channel SD
PARAMS = {'train_aug': (26201, .05, .08, .01), 'stress': (26202, .10, .16, .025)}
BATCH = 512  # same boundaries as E253


def load_row(row):
    mat = loadmat(DATA / row.path, variable_names=['data', 'fs', 'session_id', 'task_label'])
    if (mat['fs'].item(), mat['session_id'].item(), mat['task_label'].item()) != (row.fs, row.session, row.task):
        raise ValueError(f'MAT metadata mismatch: {row.path}')
    x = np.asarray(mat['data'], dtype=np.float32).T
    if x.shape != (18, 1200) or not np.isfinite(x).all():
        raise ValueError(f'Invalid waveform: {row.path}')
    return np.ascontiguousarray(x)


def perturb(x, rng, gain, ref, noise):
    g = torch.rand((*x.shape[:2], 1), generator=rng, device=x.device)
    y = x * (1 + gain * (2 * g - 1))
    eps = torch.randn(x.shape, generator=rng, device=x.device)
    return y - ref * y.mean(1, keepdim=True) + noise * x.std(-1, keepdim=True, unbiased=False) * eps


def features(x, cfg):
    windows = x.split(cfg['subwindow_length'], dim=-1)
    assert len(windows) == cfg['n_subwindows'] == 6
    return torch.stack([_subwindow_feature(w, cfg) for w in windows], 1).mean(1)


@torch.inference_mode()
def main():
    meta = pd.read_csv(CACHE / 'train_metadata.csv')
    paths = set(meta.path)
    raw = {p.relative_to(DATA).as_posix() for p in (DATA / 'Train').rglob('*.mat')}
    if len(paths) != len(meta) or paths != raw:
        raise ValueError('Train MAT paths and CSV rows are not one-to-one')
    for row in meta.itertuples():
        if row.path != f'Train/Subject_{row.subject:02d}/{row.task_folder}/trial_{row.trial:03d}.mat':
            raise ValueError(f'CSV key mismatch: {row.path}')
    ref = np.load(CACHE / 'e253_subwindow/train_feat.npy', mmap_mode='r')
    assert ref.shape == (len(meta), 6, 855)
    cfg = torch.load(PKG / 'weights.pt', map_location='cpu', weights_only=True)['feature']
    rng = {k: torch.Generator(device='cuda').manual_seed(v[0]) for k, v in PARAMS.items()}
    OUT.mkdir(exist_ok=True)
    peak = total = 0.
    with TemporaryDirectory(dir=OUT) as tmp, ThreadPoolExecutor(8) as pool:
        dest = {k: np.lib.format.open_memmap(Path(tmp) / f'{k}.npy', mode='w+', dtype='float32', shape=(len(meta), 855)) for k in ('clean', *PARAMS)}
        for i in range(0, len(meta), BATCH):
            rows = meta.iloc[i:i+BATCH]
            x = torch.from_numpy(np.stack(list(pool.map(load_row, rows.itertuples())))).cuda()
            for name in dest:
                y = x if name == 'clean' else perturb(x, rng[name], *PARAMS[name][1:])
                f = features(y, cfg).cpu().numpy()
                if not np.isfinite(f).all():
                    raise ValueError(f'Nonfinite {name} at batch {i}')
                dest[name][i:i+BATCH] = f
                if name == 'clean':
                    expected = ref[i:i+BATCH].mean(axis=1)
                    err = abs(f - expected)
                    peak = max(peak, float(err.max()))
                    total += err.sum(dtype=np.float64)
                    if not np.allclose(f, expected, atol=1e-5, rtol=1e-5):
                        raise ValueError(f'E253 mismatch at batch {i}: max_abs={err.max()}')
        dest.clear()  # close Windows memory maps before renaming
        for p in Path(tmp).glob('*.npy'):
            p.replace(OUT / p.name)
    print(f'{len(meta)} rows; max_abs={peak:.12g}; MAE={total/(len(meta)*855):.12g}')


if __name__ == '__main__':
    main()
