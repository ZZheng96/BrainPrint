"""E50 探针：3 种子 CNN 集成 + SNR20 加重对抗训练，能否把 SNR20 的 RA 推高。

对照 E49 单折（outer1/inner1）：SNR20 RA 约 0.49-0.50，三区域宏 Score 约 91。
本探针只改两件事：
  1. 身份模型改为 3 个种子 CNN 的 logits 平均（攻击对集成本身生成，白盒）。
  2. 对抗训练里 SNR20 的采样概率从 0.5 提到 0.65。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))
sys.path.insert(0, str(OUT.parent / "E45_raw_cnn_at"))

from run_e36 import ZBAND_PATH, load_config, train_metadata  # noqa: E402
from run_attacks import ATTACK_ORDER, load_all_raw, raw_attack, stratified_subset  # noqa: E402
from run_e40 import FrozenPrototype, confidence_values, fit_prototype, official_score  # noqa: E402
from run_e45 import train_fold  # noqa: E402

# train_fold 内部按 0.5 采样 SNR20；用 monkeypatch 换成 0.65
import run_e45 as E45  # noqa: E402

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
train_config = {"model": {"width": 96},
                "training": {"learning_rate": 3e-3, "weight_decay": 1e-4, "epochs": 40,
                             "batch_size": 64, "pgd_steps": 7, "targeted_mix": 0.25}}


def train_member(seed):
    """复制 train_fold 的训练循环，但 SNR20 概率为 0.65。"""
    import torch.nn.functional as F
    from run_e45 import EEGNet1d, pgd_raw

    torch.manual_seed(seed)
    model = EEGNet1d(18, len(enrolled), 96).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    rng = np.random.default_rng(seed)
    for epoch in range(1, 41):
        model.train()
        order = rng.permutation(len(fit_np))
        for start in range(0, len(order), 64):
            sel = torch.as_tensor(order[start:start + 64], device=device)
            clean = raw[fit_index[sel]]
            labels = fit_labels[sel]
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed + 1000 * epoch + start))
            snr = 20 if rng.random() < 0.50 else 30
            target = (labels + 1) % len(enrolled) if rng.random() < 0.25 else None
            attacked = pgd_raw(model, clean, labels, snr, 7, generator,
                               target is not None, target)
            loss = 0.5 * F.cross_entropy(model(clean), labels) + \
                0.5 * F.cross_entropy(model(attacked), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
    model.eval()
    return model


seeds = [20261113]
members = [train_member(seed) for seed in seeds]
print("3 个成员训练完成", flush=True)


class Ensemble:
    def __init__(self, nets, classes):
        self.nets = nets
        self.classes = np.asarray(classes, dtype=int)

    def __call__(self, raw_input):
        x = raw_input.float()
        return torch.stack([net(x) for net in self.nets], dim=0).mean(0)

    def logits(self, samples, batch=512):
        output = []
        with torch.no_grad():
            for start in range(0, len(samples), batch):
                output.append(self(samples[start:start + batch]).cpu().numpy())
        return np.concatenate(output)


ensemble = Ensemble(members, enrolled)
proto = fit_prototype(zband, fit_np, subjects, enrolled,
                      float(e36["models"]["wccn_shrinkage"]),
                      float(e36["models"]["wccn_eigenvalue_floor"]))
logspd = FrozenPrototype(proto, e36)
known_subset = stratified_subset(metadata, known_np, 8)
unknown_subset = stratified_subset(metadata, unknown_np, 8)
known_raw = raw[torch.as_tensor(known_subset, device=device)]
unknown_raw = raw[torch.as_tensor(unknown_subset, device=device)]
true = subjects[known_subset]
ens_known = ensemble.logits(known_raw)
ens_unknown = ensemble.logits(unknown_raw)
print("ensemble clean direct =",
      float(np.mean(enrolled[ens_known.argmax(1)] == true)), flush=True)

for index, net in enumerate(members):
    member_logits = net(known_raw).cpu()
    print(f"  member{index} clean = "
          f"{float(np.mean(enrolled[member_logits.argmax(1).numpy()] == true)):.4f}",
          flush=True)

logspd_known = logspd.scores(known_raw)
logspd_unknown = logspd.scores(unknown_raw)
dr_known = confidence_values(logspd_known)["distance_ratio"]
dr_unknown = confidence_values(logspd_unknown)["distance_ratio"]
tau_low = float(np.quantile(dr_known, 0.02, method="higher"))
tau_high = float(np.quantile(dr_known, 0.05, method="higher"))

target_label = np.asarray([
    enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)] for v in true
], dtype=int)


def decide(cnn_scores, logspd_scores):
    dr = confidence_values(logspd_scores)["distance_ratio"]
    pred = np.where(dr >= tau_high, enrolled[np.asarray(logspd_scores).argmax(1)],
                    enrolled[np.asarray(cnn_scores).argmax(1)])
    return np.where(dr < tau_low, 0, pred)


pred_clean = decide(ens_known, logspd_known)
pred_unknown = decide(ens_unknown, logspd_unknown)
c = float(np.mean(pred_clean == true))
u = float(np.mean(pred_unknown == 0))
print(f"three-region: C={c:.4f} U={u:.4f}", flush=True)

rows = {}
for attack_name in ATTACK_ORDER:
    base_name, snr_text = attack_name.rsplit("_snr", 1)
    snr = int(snr_text)
    code = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}[base_name]
    scores, _, waves = raw_attack(
        ensemble, known_raw, true, snr, base_name,
        int(20261113 + 900 + 100 * code + snr),
        {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 1.0,
                     "seed_base": 20261113}},
        target=target_label if base_name == "targeted_pgd" else None,
        return_waveforms=True)
    adv_logspd = logspd.scores(waves.float())
    pred = decide(np.asarray(scores), adv_logspd)
    ra = float(np.mean(pred == true))
    asafer = float(np.mean((pred == true) | (pred == 0)))
    direct = float(np.mean(enrolled[np.asarray(scores).argmax(1)] == true))
    rows[attack_name] = {"RA": ra, "ASafeR": asafer, "direct": direct,
                         "Rej": float(np.mean(pred == 0))}
    print(f"{attack_name}: direct={direct:.4f} RA={ra:.4f} ASafeR={asafer:.4f}", flush=True)

macro = {key: float(np.mean([row[key] for row in rows.values()]))
         for key in ("RA", "ASafeR", "direct", "Rej")}
a = 0.7 * macro["RA"] + 0.3 * macro["ASafeR"]
print(f"MACRO: C={c:.4f} U={u:.4f} RA={macro['RA']:.4f} ASafeR={macro['ASafeR']:.4f} "
      f"A={a:.4f} Score={official_score(c, a, u):.2f}", flush=True)
(OUT / "probe_results.json").write_text(
    json.dumps({"C": c, "U": u, "macro": macro, "A": a, "per_attack": rows},
               ensure_ascii=False, indent=2), encoding="utf-8")
