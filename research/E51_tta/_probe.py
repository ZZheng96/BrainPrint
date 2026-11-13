"""E51 探针：测试时增强（时间平移）能否提升受扰动样本的识别率。

训练 1 个 CNN（E49 配置），攻击对基础模型生成，然后在推理时
对 ±40/±80 采样点平移的 5 个视图平均 logits，比较 TTA 前后的 direct/RA/ASafeR。
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
cnn = train_fold(raw, fit_index, fit_labels, len(enrolled), train_config, 20261113, device)
proto = fit_prototype(zband, fit_np, subjects, enrolled,
                      float(e36["models"]["wccn_shrinkage"]),
                      float(e36["models"]["wccn_eigenvalue_floor"]))
logspd = FrozenPrototype(proto, e36)

SHIFTS = (-80, -40, 0, 40, 80)


def logits_tta(model, samples, shifts=SHIFTS, batch=256):
    output = []
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            views = []
            for shift in shifts:
                if shift == 0:
                    views.append(model(chunk))
                else:
                    views.append(model(torch.roll(chunk, shift, dims=-1)))
            output.append(torch.stack(views, dim=0).mean(0).cpu().numpy())
    return np.concatenate(output)


known_subset = stratified_subset(metadata, known_np, 8)
unknown_subset = stratified_subset(metadata, unknown_np, 8)
known_raw = raw[torch.as_tensor(known_subset, device=device)]
unknown_raw = raw[torch.as_tensor(unknown_subset, device=device)]
true = subjects[known_subset]
logspd_known = logspd.scores(known_raw)
logspd_unknown = logspd.scores(unknown_raw)
dr_known = confidence_values(logspd_known)["distance_ratio"]
dr_unknown = confidence_values(logspd_unknown)["distance_ratio"]
tau_low = float(np.quantile(dr_known, 0.02, method="higher"))
tau_high = float(np.quantile(dr_known, 0.05, method="higher"))


def decide(cnn_scores, logspd_scores):
    dr = confidence_values(logspd_scores)["distance_ratio"]
    pred = np.where(dr >= tau_high, enrolled[np.asarray(logspd_scores).argmax(1)],
                    enrolled[np.asarray(cnn_scores).argmax(1)])
    return np.where(dr < tau_low, 0, pred)


def report(tag, cnn_logits_fn):
    cnn_known = cnn_logits_fn(known_raw)
    pred_clean = decide(cnn_known, logspd_known)
    c = float(np.mean(pred_clean == true))
    u = float(np.mean(decide(cnn_logits_fn(unknown_raw), logspd_unknown) == 0))
    rows = {}
    for attack_name in ("untargeted_fgsm_snr30", "untargeted_pgd_snr20", "targeted_pgd_snr20"):
        item = attacks[attack_name]
        pred = decide(cnn_logits_fn(item["waves"]), item["logspd"])
        rows[attack_name] = {
            "RA": float(np.mean(pred == true)),
            "ASafeR": float(np.mean((pred == true) | (pred == 0))),
        }
    macro_ra = float(np.mean([row["RA"] for row in rows.values()]))
    macro_as = float(np.mean([row["ASafeR"] for row in rows.values()]))
    a = 0.7 * macro_ra + 0.3 * macro_as
    print(f"{tag}: C={c:.4f} U={u:.4f} RA={macro_ra:.4f} ASafeR={macro_as:.4f} "
          f"A={a:.4f} Score={official_score(c, a, u):.2f}", flush=True)
    for name, row in rows.items():
        print(f"   {name}: RA={row['RA']:.4f} ASafeR={row['ASafeR']:.4f}", flush=True)
    return {"C": c, "U": u, "RA": macro_ra, "ASafeR": macro_as, "A": a,
            "Score": official_score(c, a, u), "per_attack": rows}


class Wrapper:
    def __init__(self, net, classes):
        self.net = net
        self.classes = np.asarray(classes, dtype=int)

    def __call__(self, x):
        return self.net(x.float())


wrapper = Wrapper(cnn, enrolled)
target_label = np.asarray([
    enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)] for v in true
], dtype=int)
attacks = {}
for attack_name in ("untargeted_fgsm_snr30", "untargeted_pgd_snr20", "targeted_pgd_snr20"):
    base_name, snr_text = attack_name.rsplit("_snr", 1)
    snr = int(snr_text)
    code = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}[base_name]
    scores, _, waves = raw_attack(
        wrapper, known_raw, true, snr, base_name,
        int(20261113 + 900 + 100 * code + snr),
        {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 1.0,
                     "seed_base": 20261113}},
        target=target_label if base_name == "targeted_pgd" else None,
        return_waveforms=True)
    attacks[attack_name] = {"waves": waves.float(), "logspd": logspd.scores(waves.float()),
                            "scores": np.asarray(scores)}

base = report("no_tta", lambda x: cnn(x).detach().cpu().numpy())
tta = report("tta5", lambda x: logits_tta(cnn, x))
(OUT / "probe_results.json").write_text(json.dumps({"base": base, "tta": tta},
                                                   ensure_ascii=False, indent=2),
                                        encoding="utf-8")
