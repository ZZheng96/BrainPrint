"""E48 探针：提升 CNN 的 clean 准确率（当前 C 0.92-0.95 低于 0.95 门禁）。

单折对比三组配置，目标：clean 直接准确率 ≥0.99 且 PGD20/SNR20 直接准确率不明显下降。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))
sys.path.insert(0, str(OUT.parent / "E45_raw_cnn_at"))

from run_e36 import ZBAND_PATH, load_config, train_metadata  # noqa: E402
from run_attacks import load_all_raw, raw_attack, stratified_subset  # noqa: E402
from run_e40 import FrozenPrototype, confidence_values, fit_prototype  # noqa: E402
from run_e45 import Wrapper, pgd_raw  # noqa: E402

device = torch.device("cuda")
e36 = load_config()
metadata = train_metadata(e36)
subjects = metadata["subject"].to_numpy(dtype=int)
trial = metadata["trial"].to_numpy()
maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
forward = trial > 0.8 * maximum
inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
raw = load_all_raw(metadata, e36, device)
zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)

all_subjects = np.unique(subjects)
outer_groups = np.array_split(
    np.random.default_rng(e36["nested_protocol"]["outer_seed"]).permutation(all_subjects), 5)
known_subjects = np.setdiff1d(all_subjects, outer_groups[0])
inner_unknown = np.array_split(np.random.default_rng(101).permutation(known_subjects), 2)[0]
enrolled = np.setdiff1d(known_subjects, inner_unknown)
mask = np.isin(subjects, enrolled)
fit_np = np.flatnonzero(mask & ~forward & ~inner_block)
known_np = np.flatnonzero(mask & inner_block)
unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
class_to_index = {int(v): i for i, v in enumerate(enrolled)}
fit_index = torch.as_tensor(fit_np, device=device)
fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                             dtype=torch.long, device=device)


class Net(nn.Module):
    def __init__(self, width, n_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(18, width, 9, stride=2, padding=4), nn.BatchNorm1d(width), nn.ELU(),
            nn.Conv1d(width, width * 2, 9, stride=2, padding=4), nn.BatchNorm1d(width * 2), nn.ELU(),
            nn.Conv1d(width * 2, width * 4, 9, stride=2, padding=4), nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.Conv1d(width * 4, width * 4, 9, stride=2, padding=4), nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width * 4, n_classes)

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.reshape(len(x), -1).std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        return self.head(self.dropout(self.net(x).squeeze(-1)))


def train(config, seed=20261113):
    torch.manual_seed(seed)
    model = Net(config["width"], len(enrolled), config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    rng = np.random.default_rng(seed)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = rng.permutation(len(fit_np))
        for start in range(0, len(order), 64):
            sel = torch.as_tensor(order[start:start + 64], device=device)
            clean = raw[fit_index[sel]]
            labels = fit_labels[sel]
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed + 1000 * epoch + start))
            snr = 20 if rng.random() < config["snr20_prob"] else 30
            target = (labels + 1) % len(enrolled) if rng.random() < 0.25 else None
            attacked = pgd_raw(model, clean, labels, snr, 7, generator,
                               target is not None, target)
            clean_loss = F.cross_entropy(model(clean), labels,
                                         label_smoothing=config["label_smoothing"])
            adv_loss = F.cross_entropy(model(attacked), labels)
            loss = config["clean_weight"] * clean_loss + \
                (1 - config["clean_weight"]) * adv_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
    model.eval()
    return model


known_subset = stratified_subset(metadata, known_np, 8)
unknown_subset = stratified_subset(metadata, unknown_np, 8)
known_raw = raw[torch.as_tensor(known_subset, device=device)]
unknown_raw = raw[torch.as_tensor(unknown_subset, device=device)]
true = subjects[known_subset]
target_label = np.asarray([
    enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)] for v in true
], dtype=int)
proto = fit_prototype(zband, fit_np, subjects, enrolled,
                      float(e36["models"]["wccn_shrinkage"]),
                      float(e36["models"]["wccn_eigenvalue_floor"]))
logspd = FrozenPrototype(proto, e36)
dr_known = confidence_values(logspd.scores(known_raw))["distance_ratio"]
dr_unknown = confidence_values(logspd.scores(unknown_raw))["distance_ratio"]
threshold = float(np.quantile(dr_known, 0.05, method="higher"))

configs = [
    {"name": "A_w128_e80_clean0.6", "width": 128, "epochs": 80, "lr": 3e-3,
     "weight_decay": 1e-4, "dropout": 0.0, "snr20_prob": 0.33, "clean_weight": 0.6,
     "label_smoothing": 0.05},
    {"name": "B_w96_e80_clean0.7", "width": 96, "epochs": 80, "lr": 3e-3,
     "weight_decay": 1e-4, "dropout": 0.0, "snr20_prob": 0.5, "clean_weight": 0.7,
     "label_smoothing": 0.0},
    {"name": "C_w128_e120_drop0.2", "width": 128, "epochs": 120, "lr": 2e-3,
     "weight_decay": 1e-4, "dropout": 0.2, "snr20_prob": 0.4, "clean_weight": 0.6,
     "label_smoothing": 0.05},
]
rows = []
for config in configs:
    model = train(config)
    with torch.no_grad():
        cnn_known = model(known_raw).cpu().numpy()
    clean_acc = float(np.mean(enrolled[cnn_known.argmax(1)] == true))
    accept = dr_known >= threshold
    c = float(np.mean((enrolled[cnn_known.argmax(1)] == true) & accept))
    u = float(np.mean(dr_unknown < threshold))
    wrapper = Wrapper(model, enrolled)
    scores, _, waves = raw_attack(
        wrapper, known_raw, true, 20, "untargeted_pgd", 20261113 + 920,
        {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 1.0,
                     "seed_base": 20261113}}, return_waveforms=True)
    correct = enrolled[np.asarray(scores).argmax(1)] == true
    adv_dr = confidence_values(logspd.scores(waves.float()))["distance_ratio"]
    ra = float(np.mean(correct & (dr_known >= threshold)))
    asafer = float(np.mean(correct | (adv_dr < threshold)))
    rows.append({"config": config["name"], "clean_acc": clean_acc, "C": c, "U": u,
                 "PGD20_direct": float(np.mean(correct)), "RA": ra, "ASafeR": asafer})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    del model, wrapper
    torch.cuda.empty_cache()

(OUT / "probe_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
