"""E206: extract per-subwindow subwin6 features (6 x 855) for Train and Test.

Resumable.  Writes ``cache/{split}_subwin6.npy`` with shape (n_trials, 6, 855).
The mean over the 6 sub-windows must reproduce the frozen E171 cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'E200_final_freeze'))
from build_bundle import FEATURE, SUBWIN, load_mat, logspd_subwindow  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)
BASE = ROOT / 'Brainprint_Competition'
METAS = {
    'train': ROOT / 'BrainPrint_cache' / 'train_metadata.csv',
    'test': ROOT / 'BrainPrint_cache' / 'test_metadata.csv',
}
SAVE_EVERY = 1024


def run(split: str, limit: int, batch: int, workers: int) -> dict:
    meta = pd.read_csv(METAS[split])
    n = len(meta) if limit <= 0 else min(limit, len(meta))
    out_path = CACHE / f'{split}_subwin6.npy'
    prog_path = CACHE / f'{split}_subwin6.json'
    dim = FEATURE['dim']
    if not out_path.exists():
        memmap = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32,
                                           shape=(n, SUBWIN, dim))
        memmap.flush()
        del memmap
    else:
        shape = np.load(out_path, mmap_mode='r').shape
        if shape[0] != n:
            raise SystemExit(f'{out_path} has {shape[0]} rows, expected {n}')
    done = int(json.loads(prog_path.read_text()).get('rows_done', 0)) if prog_path.exists() else 0
    store = np.lib.format.open_memmap(out_path, mode='r+')
    paths = [BASE / str(p) for p in meta['path'].to_numpy()[:n]]
    length = FEATURE['subwindow_length']
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(done, n, batch):
            stop = min(start + batch, n)
            arrays = list(pool.map(load_mat, paths[start:stop]))
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                parts = [logspd_subwindow(raw[..., w * length:(w + 1) * length], length)
                         for w in range(SUBWIN)]
                store[start:stop] = torch.stack(parts, dim=1).cpu().numpy()
            if stop % SAVE_EVERY < batch or stop == n:
                store.flush()
                prog_path.write_text(json.dumps({'rows_done': int(stop), 'total': int(n)}))
                rate = (stop - done) / max(time.time() - started, 1e-6)
                print(f'{split}: {stop}/{n} rows, {rate:.1f} rows/s, '
                      f'eta {(n - stop) / max(rate, 1e-6) / 60:.1f} min', flush=True)
    store.flush()
    return {'split': split, 'rows': int(n), 'shape': [int(n), SUBWIN, int(dim)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    info = run(args.split, args.limit, args.batch, args.workers)
    print(info)


if __name__ == '__main__':
    main()
