"""E46 探针：在 CNN 嵌入空间做拒识，能否同时保住 RA 与 U。

E45 结果：CNN+AT 的 RA 0.748、ASafeR 0.818，但 softmax 置信度的 U 仅 0.016。
本探针比较四种拒识分数在 clean known / pseudo-unknown / 六攻击上的表现：
  - softmax distance_ratio（E45 现状）
  - 嵌入空间 max_cosine
  - 嵌入空间 distance_ratio
  - 嵌入空间 Mahalanobis（类条件共享协方差）
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

from run_e36 import load_config, train_metadata  # noqa: E402
from run_attacks import ATTACK_ORDER, load_all_raw, raw_attack, stratified_subset  # noqa: E402
from run_e40 import confidence_values  # noqa: E402
from run_e45 import EEGNet1d, Wrapper, train_fold  # noqa: E402

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
unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
class_to_index = {int(v): i for i, v in enumerate(enrolled)}

config = {
    "model": {"width": 96},
    "training": {"learning_rate": 0.003, "weight_decay": 1e-4, "epochs": 40, "batch_size": 64,
                 "pgd_steps": 7, "targeted_mix": 0.25},
    "seed_base": 20261113,
}
fit_index = torch.as_tensor(fit_np, device=device)
fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                             dtype=torch.long, device=device)
model = train_fold(raw, fit_index, fit_labels, len(enrolled), config, 20261113, device)


def embed(net, samples, batch=512):
    output = []
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            x = chunk - chunk.mean(dim=-1, keepdim=True)
            x = x / x.reshape(len(x), -1).std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
            output.append(net.net(x).squeeze(-1).cpu().numpy())
    return np.concatenate(output)


def logits(net, samples, batch=512):
    return np.concatenate([
        net(samples[start:start + batch]).cpu().numpy()
        for start in range(0, len(samples), batch)
    ]) if False else None


# 原型：fit 折嵌入按类均值
fit_embed = embed(model, raw[fit_index])
prototypes = np.stack([fit_embed[np.asarray(fit_labels.cpu()) == index].mean(0)
                       for index in range(len(enrolled))])
prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
residual = fit_embed - prototypes[np.asarray(fit_labels.cpu())]
covariance = np.cov(residual, rowvar=False) + 1e-3 * np.eye(residual.shape[1])
precision = np.linalg.inv(covariance)


def scores_from_embedding(embeddings):
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine = normalized @ prototypes.T
    ordered = np.sort(cosine, axis=1)
    maximum, second = ordered[:, -1], ordered[:, -2]
    distance_ratio = (1.0 - second) / np.maximum(1.0 - maximum, 1e-6)
    delta = embeddings - prototypes.mean(0)
    mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", delta, precision, delta))
    return {"emb_max_cosine": maximum, "emb_distance_ratio": distance_ratio,
            "emb_mahalanobis": mahalanobis}


def summarise(name, values_known, values_unknown):
    threshold = float(np.quantile(values_known, 0.05, method="higher"))
    u = float(np.mean(values_unknown < threshold))
    return {"threshold": threshold, "U": u}


known_subset = stratified_subset(metadata, known_np, 8)
unknown_subset = stratified_subset(metadata, unknown_np, 8)
known_emb = embed(model, raw[torch.as_tensor(known_subset, device=device)])
unknown_emb = embed(model, raw[torch.as_tensor(unknown_subset, device=device)])
known_stats = scores_from_embedding(known_emb)
unknown_stats = scores_from_embedding(unknown_emb)
with torch.no_grad():
    known_softmax = model(raw[torch.as_tensor(known_subset, device=device)]).cpu().numpy()
    unknown_softmax = model(raw[torch.as_tensor(unknown_subset, device=device)]).cpu().numpy()
known_stats["softmax_distance_ratio"] = confidence_values(known_softmax)["distance_ratio"]
unknown_stats["softmax_distance_ratio"] = confidence_values(unknown_softmax)["distance_ratio"]
print("clean direct acc =",
      float(np.mean(enrolled[known_softmax.argmax(1)] == subjects[known_subset])))
for key in known_stats:
    print(f"  {key:26s} " + json.dumps(summarise(key, known_stats[key], unknown_stats[key])))

wrapper = Wrapper(model, enrolled)
raw_known = raw[torch.as_tensor(known_subset, device=device)]
true = subjects[known_subset]
target_label = np.asarray([
    enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)] for v in true
], dtype=int)
rows = {}
for attack_name in ATTACK_ORDER:
    base_name, snr_text = attack_name.rsplit("_snr", 1)
    snr = int(snr_text)
    code = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}[base_name]
    scores, diag, waves = raw_attack(
        wrapper, raw_known, true, snr, base_name,
        int(20261113 + 900 + 100 * code + snr),
        {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 1.0,
                     "seed_base": 20261113}},
        target=target_label if base_name == "targeted_pgd" else None,
        return_waveforms=True)
    adv_emb = embed(model, waves.float())
    adv_stats = scores_from_embedding(adv_emb)
    adv_stats["softmax_distance_ratio"] = confidence_values(np.asarray(scores))["distance_ratio"]
    correct = enrolled[np.asarray(scores).argmax(1)] == true
    entry = {"direct_acc": float(np.mean(correct))}
    for key in known_stats:
        threshold = float(np.quantile(known_stats[key], 0.05, method="higher"))
        reject = adv_stats[key] < threshold
        entry[key] = {"RA": float(np.mean(correct & ~reject)),
                      "ASafeR": float(np.mean(correct | reject)),
                      "Rej": float(np.mean(reject))}
    rows[attack_name] = entry
    print(attack_name, json.dumps({k: v for k, v in entry.items() if k != "direct_acc"}))
(OUT / "probe_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
