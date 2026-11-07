"""Differentiable GPU covariance features and raw-waveform PGD."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.eeg_features import COV_BANDS, N_CHANNELS
from src.open_set_improvement.core import COV_COLUMNS


class GPUCovExtractor:
    """Extract the six band log-covariance blocks directly on a torch device."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._tri_rows, self._tri_cols = torch.triu_indices(
            N_CHANNELS, N_CHANNELS, device=self.device
        )
        self._off_diagonal = self._tri_rows != self._tri_cols

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        single = tensor.ndim == 2
        if single:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-1] != N_CHANNELS or tensor.shape[-2] not in (600, 1200):
            raise ValueError(f"expected (..., 600|1200, 18), got {tuple(tensor.shape)}")
        tensor = torch.nan_to_num(tensor)
        tensor = tensor - tensor.mean(dim=1, keepdim=True)
        n_samples = tensor.shape[1]
        spectrum = torch.fft.rfft(tensor, dim=1)
        frequencies = torch.fft.rfftfreq(n_samples, d=1.0 / 300.0).to(tensor.device)
        blocks = []
        for low, high in COV_BANDS:
            mask = ((frequencies >= low) & (frequencies <= high)).to(spectrum.dtype)
            band = torch.fft.irfft(spectrum * mask[None, :, None], n=n_samples, dim=1)
            band = band - band.mean(dim=1, keepdim=True)
            covariance = band.transpose(1, 2) @ band / max(n_samples - 1, 1)
            trace = covariance.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
            target = trace[:, None, :] / N_CHANNELS * torch.eye(
                N_CHANNELS, dtype=band.dtype, device=band.device
            )[None]
            covariance = 0.95 * covariance + 0.05 * target
            # 小特征值在 float32 的 eigh 反向传播中可能产生 NaN；与 CPU 特征一致使用 1e-12 下限。
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
            log_covariance = (eigenvectors * eigenvalues.clamp_min(1e-12).log().unsqueeze(1)) @ eigenvectors.transpose(1, 2)
            block = log_covariance[:, self._tri_rows, self._tri_cols].float()
            block = torch.where(self._off_diagonal[None], block * (2.0 ** 0.5), block)
            blocks.append(block)
        output = torch.cat(blocks, dim=1)
        if output.shape[-1] != len(COV_COLUMNS):
            raise RuntimeError(f"unexpected covariance feature shape: {tuple(output.shape)}")
        output = torch.nan_to_num(output)
        return output[0] if single else output


def _as_batch(data: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
    single = tensor.ndim == 2
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-2:] != (1200, N_CHANNELS):
        raise ValueError(f"raw_pgd expects (..., 1200, 18), got {tuple(tensor.shape)}")
    return tensor, single


def raw_pgd(
    model,
    extractor: GPUCovExtractor,
    data: torch.Tensor,
    task,
    targets,
    snr_db: float,
    targeted: bool = False,
    steps: int = 10,
    seed: int = 20261107,
):
    """Run reproducible L2-bounded waveform PGD against the fused model."""
    if steps < 1:
        raise ValueError("steps must be positive")
    clean, single = _as_batch(data, model.device)
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(seed)
    centered = clean - clean.mean(dim=1, keepdim=True)
    signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-8)
    radius = signal_norm / (10.0 ** (float(snr_db) / 20.0))
    delta = torch.randn(clean.shape, generator=generator, device=clean.device)
    delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    delta = delta * radius.view(-1, 1, 1) * torch.rand(
        len(clean), generator=generator, device=clean.device
    ).view(-1, 1, 1)
    target_values = torch.as_tensor(targets, device=clean.device).reshape(-1)
    if target_values.numel() == 1 and len(clean) > 1:
        target_values = target_values.expand(len(clean))
    class_to_index = {int(label): index for index, label in enumerate(model.classes)}
    target_indices = torch.as_tensor(
        [class_to_index[int(label)] for label in target_values.detach().cpu().tolist()],
        dtype=torch.long, device=clean.device,
    )
    step_size = radius / steps * 2.0
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        attacked = clean + delta
        full = extractor(attacked)
        first = extractor(attacked[:, :600])
        second = extractor(attacked[:, 600:])
        scores = model.scores(full, torch.stack((first, second), dim=1), task, "fused")
        loss = F.cross_entropy(scores, target_indices)
        objective = -loss if targeted else loss
        gradient = torch.autograd.grad(objective, delta)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("raw_pgd 出现非有限梯度")
        direction = gradient / gradient.reshape(len(gradient), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
        delta = delta + step_size.view(-1, 1, 1) * direction
        delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
        delta = delta * radius.view(-1, 1, 1)
    with torch.no_grad():
        adversarial = clean + delta
        full = extractor(adversarial)
        first = extractor(adversarial[:, :600])
        second = extractor(adversarial[:, 600:])
        final_scores = model.scores(full, torch.stack((first, second), dim=1), task, "fused")
    if not torch.isfinite(adversarial).all() or not torch.isfinite(final_scores).all():
        raise FloatingPointError("raw_pgd 出现非有限输出")
    diagnostics = {
        "final_scores": final_scores,
        "actual_snr_db": 20 * torch.log10(signal_norm / delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-8)),
        "targeted": bool(targeted),
        "steps": int(steps),
    }
    return (adversarial[0] if single else adversarial), diagnostics
