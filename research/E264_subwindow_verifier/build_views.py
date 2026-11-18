"""Build row-aligned six-subwindow Train augmentation/stress caches on GPU."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import runpy

import numpy as np
import pandas as pd
import torch

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
PREV = ROOT / 'research' / 'E262_wave_pair' / 'views.py'
e262 = runpy.run_path(str(PREV))
OUT = ROOT / 'BrainPrint_cache' / 'e264_subwindow_verifier'
BATCH = 512


def six_features(x, cfg):
    length = int(cfg['subwindow_length'])
    return torch.stack([
        e262['_subwindow_feature'](x[..., w * length:(w + 1) * length], cfg)
        for w in range(int(cfg['n_subwindows']))
    ], dim=1)


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('E264 requires CUDA')
    meta = pd.read_csv(ROOT / 'BrainPrint_cache' / 'train_metadata.csv')
    cfg = torch.load(ROOT / 'final' / 'team_hymn' / 'weights.pt', map_location='cpu', weights_only=True)['feature']
    expected = {
        name: np.load(ROOT / 'BrainPrint_cache' / 'e262_wave_pair' / f'{name}.npy', mmap_mode='r')
        for name in e262['PARAMS']
    }
    OUT.mkdir(exist_ok=True)
    rng = {
        name: torch.Generator(device='cuda').manual_seed(params[0])
        for name, params in e262['PARAMS'].items()
    }
    peak = 0.0
    with TemporaryDirectory(dir=OUT) as tmp, ThreadPoolExecutor(8) as pool:
        dest = {
            name: np.lib.format.open_memmap(
                Path(tmp) / f'{name}.npy', mode='w+', dtype='float32',
                shape=(len(meta), 6, 855))
            for name in e262['PARAMS']
        }
        for start in range(0, len(meta), BATCH):
            rows = meta.iloc[start:start + BATCH]
            x = torch.from_numpy(np.stack(list(pool.map(e262['load_row'], rows.itertuples())))).cuda()
            for name, params in e262['PARAMS'].items():
                y = e262['perturb'](x, rng[name], *params[1:])
                feat = six_features(y, cfg)
                mean = feat.mean(1).cpu().numpy()
                if not np.allclose(mean, expected[name][start:start + BATCH], atol=1e-5, rtol=1e-5):
                    raise ValueError(f'{name} mean mismatch at batch {start}')
                dest[name][start:start + BATCH] = feat.cpu().numpy()
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            print(f'features {min(start + BATCH, len(meta))}/{len(meta)}', flush=True)
        dest.clear()
        for path in Path(tmp).glob('*.npy'):
            path.replace(OUT / path.name)
    print(f'saved {OUT}; peak CUDA allocated {peak:.3f} GiB')


if __name__ == '__main__':
    main()
