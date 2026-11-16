"""Self-contained E65 correlation-logSPD (855-d) extraction for the official Test split.

The project venv has no scikit-learn, so this file re-implements the exact E65 feature
math instead of importing research/E65 or research/E36. No labels, no model, no threshold
is used here: it only materialises the feature cache used by the frozen audit.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import loadmat
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'cache'
OUT.mkdir(parents=True, exist_ok=True)
STORE = OUT / 'test_e65_features.npy'

CHANNELS = 18
SAMPLES = 1200
FS = 300.0
BANDS = [[1, 8], [8, 13], [13, 20], [20, 30], [30, 40]]
SHRINKAGE = 0.05
EIG_FLOOR = 1e-12
OFF_DIAG = 1.4142135623730951
ZBAND_EPS = 1e-6


def load_mat(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=['data'])['data'], dtype=np.float32)
    if data.shape != (SAMPLES, CHANNELS):
        raise ValueError('%s: expected (%d,%d) got %s' % (path, SAMPLES, CHANNELS, data.shape))
    if not np.isfinite(data).all():
        raise ValueError('%s: non-finite' % path)
    return np.ascontiguousarray(data.T)


def bandpass(raw: torch.Tensor) -> torch.Tensor:
    freq = torch.fft.rfftfreq(SAMPLES, d=1.0 / FS, device=raw.device)
    masks = []
    for index, (low, high) in enumerate(BANDS):
        upper = freq <= high if index == len(BANDS) - 1 else freq < high
        masks.append((freq >= low) & upper)
    mask = torch.stack(masks).to(raw.dtype)
    spectrum = torch.fft.rfft(raw, dim=-1)
    return torch.fft.irfft(spectrum[:, None] * mask[None, :, None], n=SAMPLES, dim=-1)


def detrend_zband(vec: torch.Tensor) -> torch.Tensor:
    n_bands = len(BANDS)
    bands = vec.reshape(*vec.shape[:-1], n_bands, -1).clone()
    rows, cols = np.triu_indices(CHANNELS)
    diagonal = torch.as_tensor(rows == cols, device=vec.device)
    diag_mean = bands[..., diagonal].mean(dim=-1, keepdim=True)
    bands[..., diagonal] = bands[..., diagonal] - diag_mean
    mean = bands.mean(dim=-1, keepdim=True)
    scale = bands.std(dim=-1, keepdim=True, unbiased=False).clamp_min(ZBAND_EPS)
    return ((bands - mean) / scale).flatten(-2)


def correlation_logspd(raw: torch.Tensor) -> torch.Tensor:
    filtered = bandpass(raw)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    scale = filtered.std(dim=-1, keepdim=True, unbiased=True).clamp_min(1e-12)
    normalized = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    correlation = normalized @ normalized.transpose(-1, -2) / (SAMPLES - 1)
    trace = torch.diagonal(correlation, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(CHANNELS, dtype=raw.dtype, device=raw.device)
    spd = (1.0 - SHRINKAGE) * correlation + SHRINKAGE * trace[..., None, None] / CHANNELS * eye
    values, vectors = torch.linalg.eigh(spd)
    values = values.clamp_min(EIG_FLOOR)
    logspd = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    rows, cols = torch.triu_indices(CHANNELS, CHANNELS, device=raw.device)
    vec = logspd[..., rows, cols]
    vec[..., rows != cols] *= OFF_DIAG
    return detrend_zband(vec.flatten(1))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = pd.read_csv(ROOT / 'BrainPrint_cache' / 'test_metadata.csv')
    n = len(meta)
    print('test rows', n, 'device', device, flush=True)
    if STORE.is_file():
        cached = np.load(STORE, mmap_mode='r')
        if cached.shape == (n, 855):
            print('cache exists', STORE, flush=True)
            return
    root = ROOT / 'Brainprint_Competition'
    paths = [root / str(p) for p in meta['path']]
    out = np.lib.format.open_memmap(STORE, mode='w+', dtype=np.float32, shape=(n, 855))
    batch = 64
    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            arrays = list(pool.map(load_mat, paths[start:stop]))
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
            with torch.no_grad():
                out[start:stop] = correlation_logspd(raw).cpu().numpy()
            if (start // batch) % 20 == 0:
                print('features %d/%d (%.0fs)' % (stop, n, time.time() - started), flush=True)
    out.flush()
    print('done', STORE, flush=True)


if __name__ == '__main__':
    main()
