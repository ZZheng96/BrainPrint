"""E146: phase-based feature family (not derived from the covariance).

Why: every view tested so far (correlation-logSPD, Fisher correlations, covariance
images, frozen DINOv2 embeddings) is a function of the second-order statistics, and
the cross-session instability lives exactly there.  This extractor builds features
that need only phase / amplitude envelopes:

  log band power      5 bands x 18 channels            =  90
  spectral entropy    5 bands x 18 channels            =  90
  PAC (Tort MI)       4 (phase,amp) pairs x 18 ch      =  72
  wPLI                alpha, beta x 153 pairs          = 306
  Hjorth              mobility, complexity x 18 ch     =  36
                                               total   = 594

Both Train and Test go through the identical GPU path, so no Test-side statistic is
involved anywhere.
"""
from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = CFG["data"]["sampling_rate_hz"]
N = CFG["data"]["samples_per_trial"]
C = CFG["data"]["channels"]
BANDS = CFG["bands_hz"]
PAC_PAIRS = CFG["pac_pairs"]
PAC_BINS = CFG["pac_bins"]
WPLI_BANDS = CFG["wpli_bands"]
NB = len(BANDS)
TRIU = torch.triu_indices(C, C, offset=1, device=DEVICE)
DIM = NB * C * 2 + len(PAC_PAIRS) * C + len(WPLI_BANDS) * TRIU.shape[1] + 2 * C


def load_one(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (N, C):
        raise ValueError(f"{path}: 预期 {(N, C)}，实际 {data.shape}")
    return data.T


def band_masks(device):
    freq = torch.fft.rfftfreq(N, d=1.0 / FS, device=device)
    masks = []
    for i, (lo, hi) in enumerate(BANDS):
        upper = freq <= hi if i == len(BANDS) - 1 else freq < hi
        masks.append((freq >= lo) & upper)
    return torch.stack(masks)


def analytic_bands(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """x: B x C x N  ->  analytic signal per band, B x NB x C x N (complex64)."""
    x = x - x.mean(dim=-1, keepdim=True)
    spec = torch.fft.rfft(x, dim=-1)                      # B x C x F
    out = []
    for b in range(NB):
        sb = spec * masks[b][None, None, :].to(spec.dtype)
        # one-sided analytic signal: double the kept bins, then inverse transform
        sb = sb * 2.0
        out.append(torch.fft.ifft(sb, n=N, dim=-1))
    return torch.stack(out, dim=1)


def features_chunk(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    a = analytic_bands(x, masks)                          # B x NB x C x N
    amp = a.abs()
    phase = torch.angle(a)
    feats = []

    # 1) log band power
    power = (amp ** 2).mean(-1)                           # B x NB x C
    feats.append(torch.log(power + 1e-8).permute(0, 2, 1).reshape(B, C * NB))

    # 2) spectral entropy per band (from the band-limited spectrum)
    spec = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1)
    p2 = spec.real ** 2 + spec.imag ** 2
    ent = []
    for b in range(NB):
        m = masks[b]
        pb = p2 * m[None, None, :].to(p2.dtype)
        pb = pb / (pb.sum(-1, keepdim=True) + 1e-12)
        h = -(pb * torch.log(pb + 1e-12)).sum(-1) / np.log(float(m.sum()))
        ent.append(h)
    feats.append(torch.stack(ent, dim=-1).reshape(B, C * NB))

    # 3) PAC, normalized Tort modulation index
    for (lo, hi) in PAC_PAIRS:
        ph = phase[:, lo]                                 # B x C x N
        am = amp[:, hi]
        am = (am - am.min(-1, keepdim=True).values) / (
            (am.max(-1, keepdim=True).values - am.min(-1, keepdim=True).values) + 1e-8)
        bins = torch.clamp((ph / (2 * np.pi) * PAC_BINS).long(), 0, PAC_BINS - 1)
        offset = (torch.arange(B * C, device=DEVICE) * PAC_BINS).reshape(B, C, 1)
        flat = (bins + offset).reshape(-1)
        sums = torch.zeros(B * C * PAC_BINS, device=DEVICE, dtype=am.dtype).index_add_(
            0, flat, am.reshape(-1))
        cnts = torch.zeros_like(sums).index_add_(
            0, flat, torch.ones_like(am.reshape(-1)))
        p = (sums / cnts.clamp(min=1)).reshape(B, C, PAC_BINS)
        p = p / (p.sum(-1, keepdim=True) + 1e-12)
        mi = (np.log(PAC_BINS) + (p * torch.log(p + 1e-12)).sum(-1)) / np.log(PAC_BINS)
        feats.append(mi.reshape(B, C))

    # 4) wPLI per selected band
    for b in WPLI_BANDS:
        z = a[:, b]
        re, im = z.real, z.imag
        prod = (im.unsqueeze(2) * re.unsqueeze(1)
                - re.unsqueeze(2) * im.unsqueeze(1))      # B x C x C x N
        pair = prod[:, TRIU[0], TRIU[1], :]               # B x P x N
        num = pair.mean(-1).abs()
        den = pair.abs().mean(-1)
        feats.append(num / (den + 1e-8))

    # 5) Hjorth mobility / complexity on the broadband (1-40 Hz) signal
    d1 = x[:, :, 1:] - x[:, :, :-1]
    d2 = d1[:, :, 1:] - d1[:, :, :-1]
    var0 = x.var(-1) + 1e-12
    var1 = d1.var(-1) + 1e-12
    var2 = d2.var(-1) + 1e-12
    mob = torch.sqrt(var1 / var0)
    comp = torch.sqrt(var2 / var1) / (mob + 1e-8)
    feats.append(torch.cat([mob, comp], dim=1))

    return torch.cat(feats, dim=1)


def run_split(meta: pd.DataFrame, dest: Path, tag: str):
    masks = band_masks(DEVICE)
    paths = [ROOT / CFG["data"]["root"] / p for p in meta.path]
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32,
                                    shape=(len(paths), DIM))
    chunk = int(CFG["chunk"])
    started = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for st in range(0, len(paths), chunk):
            stop = min(st + chunk, len(paths))
            arr = list(pool.map(load_one, paths[st:stop]))
            x = torch.as_tensor(np.stack(arr), dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                f = features_chunk(x, masks)
            if f.shape[1] != DIM:
                raise ValueError(f"维度不符: {f.shape[1]} != {DIM}")
            out[st:stop] = f.cpu().numpy()
            if stop % 2048 == 0 or stop == len(paths):
                el = time.time() - started
                print(f"{tag} {stop}/{len(paths)}  {el:.0f}s", flush=True)
    out.flush()
    del out


def main():
    tr = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    te = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    print(f"train={len(tr)} test={len(te)} dim={DIM}", flush=True)
    run_split(tr, OUT / "train_phase.npy", "train")
    run_split(te, OUT / "test_phase.npy", "test")
    manifest = {
        "dim": int(DIM),
        "layout": {
            "log_band_power": [0, NB * C],
            "spectral_entropy": [NB * C, 2 * NB * C],
            "pac": [2 * NB * C, 2 * NB * C + len(PAC_PAIRS) * C],
            "wpli": [2 * NB * C + len(PAC_PAIRS) * C,
                     2 * NB * C + len(PAC_PAIRS) * C + len(WPLI_BANDS) * TRIU.shape[1]],
            "hjorth": [DIM - 2 * C, DIM],
        },
        "band_names": CFG["band_names"],
        "pac_pairs": CFG["pac_pair_names"],
        "wpli_bands": [CFG["band_names"][b] for b in WPLI_BANDS],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
