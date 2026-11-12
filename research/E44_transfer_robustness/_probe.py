"""E44 探针：迁移攻击下的 RA/ASafeR（更接近官方 Test 的设定）。

官方 Test 的对抗样本是预生成的，参赛者无法对自己的模型重算 PGD，
因此真实设定是「迁移攻击」而非白盒攻击。本探针在同一折上比较：

  - 白盒：攻击直接对本模型生成（E36C 口径，悲观代理）
  - 迁移：攻击对两个不同的替代模型生成，再喂给本模型

替代模型 A：同特征、不同 WCCN 收缩（0.001）
替代模型 B：同特征、Ridge 线性分类器（决策几何不同）
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

from run_e36 import ZBAND_PATH, detrend_zband, gpu_logspd, load_config, train_metadata  # noqa: E402
from run_attacks import DifferentiableWCCN, load_all_raw, raw_attack, stratified_subset  # noqa: E402
from run_e40 import FrozenPrototype, fit_prototype, confidence_values  # noqa: E402

device = torch.device("cuda")
e36 = load_config()
metadata = train_metadata(e36)
subjects = metadata["subject"].to_numpy(dtype=int)
trial = metadata["trial"].to_numpy()
maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
forward = trial > 0.8 * maximum
inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
raw = load_all_raw(metadata, e36, device)
shrinkage = float(e36["models"]["wccn_shrinkage"])
floor = float(e36["models"]["wccn_eigenvalue_floor"])

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
raw_subset = raw[torch.as_tensor(subset, device=device)]
true = subjects[subset]
target_label = np.asarray([
    enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)] for v in true
], dtype=int)

target_proto = fit_prototype(zband, fit_np, subjects, enrolled, shrinkage, floor)
target_model = FrozenPrototype(target_proto, e36)
clean_scores = target_model.scores(raw_subset)


class LinearRaw:
    """原始波形 → zband → 线性分类器（可微，用于生成迁移攻击）。"""

    def __init__(self, weight, bias, classes, config):
        self.weight = weight
        self.bias = bias
        self.classes = np.asarray(classes, dtype=int)
        self.config = config

    def __call__(self, raw_input):
        features = detrend_zband(gpu_logspd(raw_input, self.config), self.config)
        return features @ self.weight.to(features.dtype) + self.bias.to(features.dtype)


# 替代模型 A：同特征、shrinkage=0.001
surrogate_a = DifferentiableWCCN(
    fit_prototype(zband, fit_np, subjects, enrolled, 0.001, floor), e36)
# 替代模型 B：Ridge 闭式解
fit_features = zband[torch.as_tensor(fit_np, device=device)]
fit_labels = torch.as_tensor(
    [int(np.flatnonzero(enrolled == v)[0]) for v in subjects[fit_np]], dtype=torch.long,
    device=device)
onehot = torch.nn.functional.one_hot(fit_labels, len(enrolled)).float()
centered = fit_features - fit_features.mean(0, keepdim=True)
ridge = 1e-2
weight = torch.linalg.solve(
    centered.T @ centered + ridge * torch.eye(centered.shape[1], device=device),
    centered.T @ onehot,
)
bias = onehot.mean(0) - fit_features.mean(0) @ weight
surrogate_b = LinearRaw(weight, bias, enrolled, e36)
# 数值检查：Ridge 本身在 fit 上的准确率
with torch.no_grad():
    acc_b = float(np.mean(
        enrolled[(detrend_zband(gpu_logspd(raw[torch.as_tensor(fit_np[:512], device=device)], e36),
                               e36) @ weight + bias).argmax(1).cpu().numpy()]
        == subjects[fit_np[:512]]))
print(f"Ridge surrogate fit-batch acc={acc_b:.4f}")

white_box = DifferentiableWCCN(target_proto, e36)
attacks = [("untargeted_fgsm", 30), ("untargeted_pgd", 30), ("untargeted_pgd", 20),
           ("targeted_pgd", 30), ("targeted_pgd", 20)]
codes = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
thr = float(np.quantile(confidence_values(clean_scores)["distance_ratio"], 0.05, method="higher"))
clean_acc = float(np.mean(enrolled[clean_scores.argmax(1)] == true))
print(f"target clean direct acc={clean_acc:.4f}  tau(distance_ratio q05)={thr:.4f}")

rows = []
for base_name, snr in attacks:
    for mode, surrogate in (("white_box", white_box), ("transfer_A_shrink", surrogate_a),
                            ("transfer_B_ridge", surrogate_b)):
        seed = int(20261112 + 500000 + 100 * codes[base_name] + snr)
        scores, diag, _ = raw_attack(
            surrogate, raw_subset, true, snr, base_name, seed,
            {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 0.05,
                         "seed_base": 20261112}},
            target=target_label if base_name == "targeted_pgd" else None)
        pred = enrolled[np.asarray(scores).argmax(1)]
        correct = pred == true
        conf = confidence_values(scores)["distance_ratio"]
        accept = conf >= thr
        ra = float(np.mean(correct & accept))
        asafer = float(np.mean(correct | ~accept))
        rows.append({
            "attack": f"{base_name}_snr{snr}", "mode": mode,
            "direct_acc": float(np.mean(correct)),
            "target_rate": float(np.mean(pred == target_label))
            if base_name == "targeted_pgd" else None,
            "RA": ra, "ASafeR": asafer,
            "realized_snr": diag["realized_snr_db_mean"],
        })
        print(f"{base_name}_snr{snr:2d} {mode:18s} direct={rows[-1]['direct_acc']:.4f} "
              f"RA={ra:.4f} ASafeR={asafer:.4f} target_rate={rows[-1]['target_rate']}", flush=True)

(OUT / "probe_results.json").write_text(
    json.dumps({"clean_acc": clean_acc, "threshold": thr, "rows": rows},
               ensure_ascii=False, indent=2), encoding="utf-8")
