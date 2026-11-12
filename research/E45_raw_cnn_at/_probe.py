"""E45 受限试验：原始波形 CNN + 对抗训练能否突破 RA 硬边界。

判据（预注册）：白盒 PGD20/SNR20 的**未加阈值直接准确率**若 <0.5，本路线放弃。
对照组为同折 logSPD WCCN（PGD20/SNR20 direct 0.082）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))

from run_e36 import load_config, train_metadata  # noqa: E402
from run_attacks import load_all_raw, stratified_subset  # noqa: E402
from run_e40 import confidence_values  # noqa: E402

device = torch.device("cuda")
e36 = load_config()
metadata = train_metadata(e36)
subjects = metadata["subject"].to_numpy(dtype=int)
trial = metadata["trial"].to_numpy()
maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
forward = trial > 0.8 * maximum
inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
raw = load_all_raw(metadata, e36, device)

all_subjects = np.unique(subjects)
outer_groups = np.array_split(
    np.random.default_rng(e36["nested_protocol"]["outer_seed"]).permutation(all_subjects), 5)
known_subjects = np.setdiff1d(all_subjects, outer_groups[0])
inner_unknown = np.array_split(np.random.default_rng(101).permutation(known_subjects), 2)[0]
enrolled = np.setdiff1d(known_subjects, inner_unknown)
mask = np.isin(subjects, enrolled)
fit_np = np.flatnonzero(mask & ~forward & ~inner_block)
known_np = np.flatnonzero(mask & inner_block)
subset = stratified_subset(metadata, known_np, 8)
fit_index = torch.as_tensor(fit_np, device=device)
class_to_index = {int(v): i for i, v in enumerate(enrolled)}


class EEGNet1d(nn.Module):
    def __init__(self, n_channels=18, n_classes=8, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, 9, stride=2, padding=4), nn.BatchNorm1d(width), nn.ELU(),
            nn.Conv1d(width, width * 2, 9, stride=2, padding=4), nn.BatchNorm1d(width * 2), nn.ELU(),
            nn.Conv1d(width * 2, width * 4, 9, stride=2, padding=4), nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.Conv1d(width * 4, width * 4, 9, stride=2, padding=4), nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(width * 4, n_classes)

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.reshape(len(x), -1).std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        return self.head(self.net(x).squeeze(-1))


def pgd_raw(model, clean, labels, snr, steps, generator):
    centered = clean - clean.mean(dim=-1, keepdim=True)
    signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
    radius = signal_norm / (10.0 ** (float(snr) / 20.0))
    delta = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta = delta * radius.view(-1, 1, 1) * torch.rand(
        len(clean), generator=generator, device=clean.device, dtype=clean.dtype).view(-1, 1, 1)
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        loss = F.cross_entropy(model(clean + delta), labels)
        gradient = torch.autograd.grad(loss, delta)[0]
        direction = gradient / gradient.reshape(len(gradient), -1).norm(
            dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() + (2.0 * radius / steps).view(-1, 1, 1) * direction
        norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
        delta = delta * torch.minimum(torch.ones_like(norm), radius / norm).view(-1, 1, 1)
    return (clean + delta).detach()


model = EEGNet1d(18, len(enrolled)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                             dtype=torch.long, device=device)
rng = np.random.default_rng(20261113)
started = time.time()
eval_index = torch.as_tensor(subset, device=device)
eval_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[subset]],
                              dtype=torch.long, device=device)

for epoch in range(1, 13):
    model.train()
    order = rng.permutation(len(fit_np))
    total = 0.0
    for start in range(0, len(order), 64):
        sel = torch.as_tensor(order[start:start + 64], device=device)
        clean = raw[fit_index[sel]]
        labels = fit_labels[sel]
        generator = torch.Generator(device=device)
        generator.manual_seed(int(20261113 + 1000 * epoch + start))
        snr = 20 if rng.random() < 0.5 else 30
        attacked = pgd_raw(model, clean, labels, snr, 7, generator)
        loss = 0.5 * F.cross_entropy(model(clean), labels) + \
            0.5 * F.cross_entropy(model(attacked), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss) * len(sel)
    model.eval()
    with torch.no_grad():
        logits = model(raw[eval_index])
        clean_acc = float((logits.argmax(1) == eval_labels).float().mean())
    print(f"epoch={epoch} loss={total / len(fit_np):.4f} eval_clean={clean_acc:.4f} "
          f"({time.time() - started:.0f}s)", flush=True)

results = {"clean_acc": clean_acc, "attacks": {}}


class Wrapper:
    def __init__(self, net, classes):
        self.net = net
        self.classes = np.asarray(classes, dtype=int)

    def __call__(self, raw_input):
        # raw_attack 内部把波形转成 float64，这里转回 float32 供 CNN 前向
        return self.net(raw_input.float())


wrapper = Wrapper(model, enrolled)
from run_attacks import raw_attack  # noqa: E402

target_label = np.asarray([
    enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)]
    for v in subjects[subset]], dtype=int)
with torch.no_grad():
    clean_scores = model(raw[eval_index]).cpu().numpy()
threshold = float(np.quantile(confidence_values(clean_scores)["distance_ratio"],
                              0.05, method="higher"))
for base_name, snr in (("untargeted_fgsm", 30), ("untargeted_pgd", 30),
                       ("untargeted_pgd", 20), ("targeted_pgd", 20)):
    scores, diag, _ = raw_attack(
        wrapper, raw[eval_index], subjects[subset], snr, base_name,
        int(20261113 + 900 + snr),
        {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 1.0,
                     "seed_base": 20261113}},
        target=target_label if base_name == "targeted_pgd" else None)
    pred = np.asarray(scores).argmax(1)
    correct = enrolled[pred] == subjects[subset]
    accept = confidence_values(scores)["distance_ratio"] >= threshold
    entry = {"direct_acc": float(np.mean(correct)),
             "RA": float(np.mean(correct & accept)),
             "ASafeR": float(np.mean(correct | ~accept)),
             "target_rate": float(np.mean(enrolled[pred] == target_label))
             if base_name == "targeted_pgd" else None,
             "realized_snr": diag["realized_snr_db_mean"]}
    results["attacks"][f"{base_name}_snr{snr}"] = entry
    print(f"{base_name}_snr{snr}: direct={entry['direct_acc']:.4f} RA={entry['RA']:.4f} "
          f"ASafeR={entry['ASafeR']:.4f} target={entry['target_rate']}", flush=True)

results["criterion"] = "PGD20/SNR20 direct >= 0.5 才继续"
results["passed"] = results["attacks"]["untargeted_pgd_snr20"]["direct_acc"] >= 0.5
(OUT / "probe_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"判定：{'通过，可扩展' if results['passed'] else '未通过，放弃该路线'}", flush=True)
