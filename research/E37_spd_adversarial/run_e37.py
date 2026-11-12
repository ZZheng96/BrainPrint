"""E37：冻结 WCCN 白化上的 SPD 残差 MLP + ARPL 对抗训练。

设计要点（与 config.json 预注册一致）：
  - 特征：E36 五频带 logSPD → detrend+zband（855 维），只读复用缓存。
  - 白化：fit 折 WCCN（shrinkage 0.05），训练期间冻结。
  - 残差：855 → 256 → 855，GELU，末层零初始化；零残差时与 E36 WCCN 余弦模型完全等价。
  - 原型：可训练、L2 归一化，初始化为同一 fit 折的 WCCN 原型。
  - ARPL：互反点分类损失 + 开空间 MarginRanking 损失（可训练半径）。
  - 三变体：clean_arpl / untargeted_at_arpl / mixed_targeted_untargeted_at_arpl。
  - 对抗训练在冻结白化后的 855 维特征空间做 PGD7，epsilon 由原始波形 PGD20 位移标定。

评估沿用 E36C 六攻击（FGSM/PGD/目标PGD × SNR30/20，PGD20），阈值只在对应内层
clean known + pseudo-unknown 上选择，外层只在全部门禁通过后运行一次。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))

from run_e36 import (  # noqa: E402
    CONFIG_PATH as E36_CONFIG_PATH,
    ZBAND_PATH,
    detrend_zband,
    gpu_logspd,
    load_config as load_e36_config,
    load_train_matrix,
    train_metadata,
)
from run_attacks import (  # noqa: E402
    ATTACK_ORDER,
    DifferentiableWCCN,
    load_all_raw,
    raw_attack,
    stratified_subset,
)

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
COMPONENT_KEYS = ("C", "RA", "ASafeR", "A", "U", "direct_RA", "known_acceptance")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def official_score(c: float, a: float, u: float) -> float:
    return float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0


def confidence_values(scores: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
    return {
        "max_cosine": maximum,
        "margin": maximum - second,
        "distance_ratio": (1.0 - second) / np.maximum(1.0 - maximum, 1e-6),
    }


class ResidualHead(nn.Module):
    """855 → 256 → 855 零初始化残差。"""

    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden)
        self.fc2 = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class RobustModel(nn.Module):
    """冻结白化 + 残差 + 可训练原型 + ARPL 半径。"""

    def __init__(self, whitening: torch.Tensor, prototypes: torch.Tensor, hidden: int):
        super().__init__()
        self.register_buffer("whitening", whitening)
        self.residual = ResidualHead(whitening.shape[0], hidden)
        self.prototypes = nn.Parameter(prototypes.clone())
        self.radius = nn.Parameter(torch.zeros(1))
        self.fc_dtype = whitening.dtype

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        whitened = features.to(self.fc_dtype) @ self.whitening
        combined = whitened + self.residual(whitened)
        return combined / combined.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def embed_whitened(self, whitened: torch.Tensor) -> torch.Tensor:
        """已白化特征直接进入残差，避免重复白化。"""
        combined = whitened + self.residual(whitened)
        return combined / combined.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def scores_whitened(self, whitened: torch.Tensor) -> torch.Tensor:
        return self.embed_whitened(whitened) @ self.normalized_prototypes().T

    def normalized_prototypes(self) -> torch.Tensor:
        return self.prototypes / self.prototypes.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def scores(self, features: torch.Tensor) -> torch.Tensor:
        return self.embed(features) @ self.normalized_prototypes().T

    def arpl_logits(self, features: torch.Tensor) -> torch.Tensor:
        """互反点形式：mean squared L2 distance − dot product。

        单位球上 mean squared distance = 2(1 − cos)/d，故该 logits 与余弦评分
        严格同序（argmax 一致），只改变尺度与梯度形状。
        """
        embedding = self.embed(features)
        prototypes = self.normalized_prototypes()
        squared = (
            (embedding**2).sum(1, keepdim=True)
            - 2 * embedding @ prototypes.T
            + (prototypes**2).sum(1)[None, :]
        ) / embedding.shape[1]
        return squared - embedding @ prototypes.T

    def arpl_logits_centered(self, features: torch.Tensor) -> torch.Tensor:
        """逐样本去均值后的互反点 logits，供交叉熵使用。

        原始互反点 logits 含逐样本常数项（squared 的公共部分），softmax 不变，
        但数值上会与余弦评分出现 argmax 不一致；去均值后与余弦严格同序。
        """
        logits = self.arpl_logits(features)
        return logits - logits.mean(1, keepdim=True)

    def classification_logits(self, features: torch.Tensor) -> torch.Tensor:
        """交叉熵使用的 logits：余弦评分（与推理口径一致）。

        互反点 logits 在单位球上只差一个逐样本常数，softmax 等价，但数值上
        余弦更稳定；开空间损失仍使用互反点距离形式。
        """
        return self.scores(features)

    def open_space_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """开空间损失：真实类距离应小于半径，其余类距离应大于半径。"""
        embedding = self.embed(features)
        prototypes = self.normalized_prototypes()
        squared = ((embedding[:, None, :] - prototypes[None, :, :]) ** 2).mean(-1)
        true_distance = squared.gather(1, labels[:, None]).squeeze(1)
        other_distance = squared.masked_fill(
            F.one_hot(labels, squared.shape[1]).bool(), float("inf")
        ).min(1).values
        radius = self.radius.expand_as(true_distance)
        return (F.margin_ranking_loss(radius, true_distance, torch.ones_like(true_distance),
                                      margin=1.0)
                + F.margin_ranking_loss(radius, other_distance, -torch.ones_like(other_distance),
                                        margin=1.0))

    def cosine_logits(self, features: torch.Tensor) -> torch.Tensor:
        """余弦 logits，用于对抗扰动生成（与 E36 攻击口径一致）。"""
        return self.scores(features)


def fit_wccn(zband: torch.Tensor, fit_np: np.ndarray, subjects: np.ndarray,
             classes: np.ndarray, shrinkage: float, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.as_tensor(fit_np, device=zband.device)
    x = zband[index]
    y = subjects[fit_np]
    class_to_index = {int(label): i for i, label in enumerate(classes)}
    target = torch.as_tensor([class_to_index[int(v)] for v in y], dtype=torch.long, device=x.device)
    means = torch.stack([x[target == i].mean(0) for i in range(len(classes))])
    residual = x - means[target]
    covariance = residual.T @ residual / max(len(x) - len(classes), 1)
    trace = torch.diagonal(covariance).sum()
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace / x.shape[1] * eye
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    whitening = (eigenvectors * eigenvalues.clamp_min(floor).rsqrt().unsqueeze(0)) @ eigenvectors.T
    whitened_means = means @ whitening
    prototypes = whitened_means / whitened_means.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return whitening, prototypes


def calibrate_epsilon(
    raw: torch.Tensor, zband: torch.Tensor, metadata: pd.DataFrame, fit_np: np.ndarray,
    subjects: np.ndarray, classes: np.ndarray, config: dict, e36_config: dict,
    device: torch.device,
) -> dict:
    """用零残差模型的原始波形 PGD20 位移标定特征空间 epsilon。"""
    whitening, prototypes = fit_wccn(
        zband, fit_np, subjects, classes,
        float(e36_config["models"]["wccn_shrinkage"]),
        float(e36_config["models"]["wccn_eigenvalue_floor"]),
    )
    class _Zero(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("whitening", whitening.double())
            self.register_buffer("prototypes", prototypes.double())
            self.classes = np.asarray(classes, dtype=int)

        def scores_from_features(self, features):
            whitened = features @ self.whitening
            normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
            return normalized @ self.prototypes.T

        def features(self, raw_input):
            return detrend_zband(gpu_logspd(raw_input, e36_config), e36_config)

        def __call__(self, raw_input):
            return self.scores_from_features(self.features(raw_input))

    model = _Zero().to(device)
    subset = stratified_subset(metadata, fit_np, 1)
    raw_subset = raw[torch.as_tensor(subset, device=device)]
    true = subjects[subset]
    target = np.asarray([
        classes[(int(np.flatnonzero(classes == label)[0]) + 1) % len(classes)] for label in true
    ], dtype=int)
    displacements = {}
    for snr in (20, 30):
        values = []
        for attack_name in ("untargeted_pgd", "targeted_pgd"):
            seed = int(config["seed_base"] + 100 * ATTACK_CODES[attack_name] + snr)
            _, _, waveforms = raw_attack(
                model, raw_subset, true, snr, attack_name, seed,
                {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 0.05,
                             "seed_base": config["seed_base"]}},
                target=target if attack_name == "targeted_pgd" else None,
                return_waveforms=True,
            )
            with torch.no_grad():
                clean_features = model.features(raw_subset.double())
                attacked_features = model.features(waveforms.double())
                delta = (attacked_features - clean_features) @ whitening.double()
            values.append(delta.norm(dim=1).cpu().numpy())
        displacements[snr] = float(np.quantile(np.concatenate(values), 0.95))
    return {"epsilon_snr20": displacements[20], "epsilon_snr30": displacements[30],
            "n_calibration": int(len(subset))}


def raw_pgd(
    model: RobustModel, raw: torch.Tensor, labels: torch.Tensor, epsilon: float,
    steps: int, targeted: bool, target: torch.Tensor | None, e36_config: dict,
    generator: torch.Generator,
) -> torch.Tensor:
    """原始波形空间的 L2 PGD（端到端穿过 logSPD/zband/白化/残差）。

    特征空间扰动经 logSPD 非线性后会被大幅削弱，因此对抗训练直接在原始波形
    上生成扰动，半径与评估攻击一致（SNR 定义）。
    """
    clean = raw.detach().double()
    centered = clean - clean.mean(dim=-1, keepdim=True)
    signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
    radius = signal_norm / (10.0 ** (float(epsilon) / 20.0))
    delta = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    random_radius = torch.rand(len(clean), generator=generator, device=clean.device,
                               dtype=clean.dtype)
    delta = delta * radius.view(-1, 1, 1) * random_radius.view(-1, 1, 1)
    objective_labels = target if targeted else labels
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        whitened = detrend_zband(gpu_logspd(clean + delta, e36_config), e36_config)
        scores = model.scores_whitened(whitened.to(model.fc_dtype) @ model.whitening)
        loss = F.cross_entropy(scores / 0.05, objective_labels)
        gradient = torch.autograd.grad(-loss if targeted else loss, delta)[0]
        direction = gradient / gradient.reshape(len(gradient), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() + (2.0 * radius / steps).view(-1, 1, 1) * direction
        norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
        factor = torch.minimum(torch.ones_like(norm), radius / norm)
        delta = delta * factor.view(-1, 1, 1)
    return (clean + delta).detach()


def train_variant(
    variant: str, zband: torch.Tensor, raw: torch.Tensor, subjects: np.ndarray,
    fit_np: np.ndarray, known_np: np.ndarray, unknown_np: np.ndarray, classes: np.ndarray,
    epsilon: dict, config: dict, e36_config: dict, seed: int, device: torch.device,
) -> tuple[RobustModel, dict]:
    whitening, prototypes = fit_wccn(
        zband, fit_np, subjects, classes,
        float(e36_config["models"]["wccn_shrinkage"]),
        float(e36_config["models"]["wccn_eigenvalue_floor"]),
    )
    torch.manual_seed(seed)
    model = RobustModel(whitening, prototypes, 256).to(device)
    initial_prototypes = model.normalized_prototypes().detach().clone()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    class_to_index = {int(label): i for i, label in enumerate(classes)}
    fit_index = torch.as_tensor(fit_np, device=device)
    fit_features = zband[fit_index]
    fit_raw = raw[fit_index]
    fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                                 dtype=torch.long, device=device)
    known_index = torch.as_tensor(known_np, device=device)
    unknown_index = torch.as_tensor(unknown_np, device=device)
    batch_size = int(config["training"]["batch_size"])
    rng = np.random.default_rng(seed)
    best = {"score": -1.0, "epoch": -1, "state": None, "metrics": None}
    patience = 0
    history = []
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(fit_np))
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            selected = torch.as_tensor(order[start:start + batch_size], device=device)
            features = fit_features[selected]
            labels = fit_labels[selected]
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed + 1000 * epoch + start))
            if variant == "clean_arpl":
                used = features
            else:
                snr = 20 if rng.random() < 0.5 else 30
                targeted = variant == "mixed_targeted_untargeted_at_arpl" and rng.random() < 0.5
                target = None
                if targeted:
                    target = (labels + 1) % len(classes)
                attacked_raw = raw_pgd(
                    model, fit_raw[selected], labels, float(snr),
                    int(config["feature_attack"]["pgd_steps"]), targeted, target,
                    e36_config, generator,
                )
                used = detrend_zband(gpu_logspd(attacked_raw, e36_config), e36_config)
            logits = model.classification_logits(used)
            loss = F.cross_entropy(logits / float(config["arpl"]["temperature"]), labels)
            loss = loss + float(config["arpl"]["weight"]) * model.open_space_loss(used, labels)
            if variant != "clean_arpl":
                clean_logits = model.classification_logits(features)
                clean_loss = F.cross_entropy(
                    clean_logits / float(config["arpl"]["temperature"]), labels
                )
                clean_loss = clean_loss + float(config["arpl"]["weight"]) * model.open_space_loss(
                    features, labels
                )
                loss = (float(config["training"]["clean_loss_weight_for_at"]) * clean_loss
                        + float(config["training"]["adversarial_loss_weight_for_at"]) * loss)
            # 原型锚定：防止可训练原型漂移破坏零残差等价性
            anchor = float(config["training"]["prototype_anchor_weight"]) * (
                (model.normalized_prototypes() - initial_prototypes) ** 2
            ).sum(1).mean()
            loss = loss + anchor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(selected)
        model.eval()
        with torch.no_grad():
            known_scores = model.scores(zband[known_index]).cpu().numpy()
            unknown_scores = model.scores(zband[unknown_index]).cpu().numpy()
        known_conf = confidence_values(known_scores)["distance_ratio"]
        unknown_conf = confidence_values(unknown_scores)["distance_ratio"]
        threshold = float(np.quantile(known_conf, 0.05, method="lower"))
        known_pred = classes[known_scores.argmax(1)]
        correct = known_pred == subjects[known_np]
        c = float(np.mean(correct & (known_conf >= threshold)))
        u = float(np.mean(unknown_conf < threshold))
        score = float(100 * c**0.40 * u**0.25) if min(c, u) > 0 else 0.0
        history.append({"epoch": epoch, "loss": epoch_loss / len(fit_np), "C": c, "U": u,
                        "score": score})
        print(f"  {variant} epoch={epoch} loss={epoch_loss / len(fit_np):.4f} "
              f"C={c:.4f} U={u:.4f} proxy={score:.2f}", flush=True)
        if score > best["score"] + 1e-9:
            best = {"score": score, "epoch": epoch, "metrics": {"C": c, "U": u},
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= int(config["training"]["early_stopping_patience"]):
                break
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"best_epoch": best["epoch"], "best_metrics": best["metrics"],
                   "history": history}


def predict(model: RobustModel, zband: torch.Tensor, indices: np.ndarray,
            batch: int = 512) -> np.ndarray:
    output = np.empty((len(indices), model.prototypes.shape[0]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            selected = torch.as_tensor(indices[start:start + batch], device=zband.device)
            output[start:start + len(selected)] = model.scores(zband[selected]).cpu().numpy()
    return output


class DifferentiableRobust:
    """原始波形 → 冻结 logSPD/zband → 冻结白化 → 残差 MLP → 原型。"""

    def __init__(self, model: RobustModel, e36_config: dict, classes: np.ndarray):
        self.model = model
        self.e36_config = e36_config
        self.classes = np.asarray(classes, dtype=int)

    def features(self, raw: torch.Tensor) -> torch.Tensor:
        return detrend_zband(gpu_logspd(raw, self.e36_config), self.e36_config)

    def scores_from_features(self, features: torch.Tensor) -> torch.Tensor:
        model = self.model
        whitened = features @ model.whitening.to(features.dtype)
        return model.scores_whitened(whitened.to(model.fc_dtype))

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return self.scores_from_features(self.features(raw))


def attack_seed(base: int, stage: int, outer: int, inner: int, name: str, snr: int) -> int:
    return int(base + 100000 * stage + 10000 * outer + 1000 * inner
               + 100 * ATTACK_CODES[name] + snr)


def attack_subset(metadata: pd.DataFrame, indices: np.ndarray, per_group: int) -> np.ndarray:
    """E36 分层子集，但每个 subject/task 组内随机抽取（不取文件顺序前 N 条）。

    E36 的 stratified_subset 取 trial/path 排序后的前 8 条，属于连续文件块；
    攻击评估若沿用会与训练 fit 块重叠，因此这里按固定种子在组内随机抽样。
    """
    frame = metadata.iloc[indices][["subject", "task"]].copy()
    frame["matrix_index"] = indices
    rng = np.random.default_rng(20261113)
    chosen = []
    for _, group in frame.groupby(["subject", "task"], sort=True):
        values = np.sort(group["matrix_index"].to_numpy(dtype=int))
        take = min(per_group, len(values))
        chosen.append(rng.choice(values, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def run_attacks_for_model(
    model: RobustModel, raw: torch.Tensor, zband: torch.Tensor, metadata: pd.DataFrame,
    known_np: np.ndarray, unknown_np: np.ndarray, subjects: np.ndarray, classes: np.ndarray,
    config: dict, e36_config: dict, stage: int, outer: int, inner: int,
) -> dict:
    per_group = 8
    known_subset = attack_subset(metadata, known_np, per_group)
    unknown_subset = attack_subset(metadata, unknown_np, per_group)
    clean_known = predict(model, zband, known_subset)
    clean_unknown = predict(model, zband, unknown_subset)
    differentiable = DifferentiableRobust(model, e36_config, classes)
    raw_known = raw[torch.as_tensor(known_subset, device=raw.device)]
    true = subjects[known_subset]
    target = np.asarray([
        classes[(int(np.flatnonzero(classes == label)[0]) + 1) % len(classes)] for label in true
    ], dtype=int)
    attacks = {}
    for attack_name in ATTACK_ORDER:
        base_name, snr_text = attack_name.rsplit("_snr", 1)
        snr = int(snr_text)
        seed = attack_seed(config["seed_base"], stage, outer, inner, base_name, snr)
        scores, diagnostics, _ = raw_attack(
            differentiable, raw_known, true, snr, base_name, seed,
            {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 0.05,
                         "seed_base": config["seed_base"]}},
            target=target if base_name == "targeted_pgd" else None,
        )
        attacks[attack_name] = {"scores": scores, "diagnostics": diagnostics,
                                "target": target if base_name == "targeted_pgd" else None}
    return {
        "known_true": subjects[known_subset], "known_scores": clean_known,
        "unknown_scores": clean_unknown, "classes": classes, "attacks": attacks,
        "n_known": int(len(known_subset)), "n_unknown": int(len(unknown_subset)),
    }


def components(run: dict, attack_name: str, score_name: str, threshold: float) -> dict:
    known_conf = confidence_values(run["known_scores"])[score_name]
    known_pred = run["classes"][run["known_scores"].argmax(1)]
    unknown_conf = confidence_values(run["unknown_scores"])[score_name]
    adv_conf = confidence_values(run["attacks"][attack_name]["scores"])[score_name]
    adv_pred = run["classes"][run["attacks"][attack_name]["scores"].argmax(1)]
    clean_accepted = known_conf >= threshold
    adv_accepted = adv_conf >= threshold
    correct = known_pred == run["known_true"]
    adv_correct = adv_pred == run["known_true"]
    c = float(np.mean(correct & clean_accepted))
    ra = float(np.mean(adv_correct & adv_accepted))
    asafer = float(np.mean(adv_correct | ~adv_accepted))
    u = float(np.mean(unknown_conf < threshold))
    return {
        "C": c, "RA": ra, "ASafeR": asafer, "A": 0.7 * ra + 0.3 * asafer, "U": u,
        "direct_RA": float(np.mean(adv_correct)),
        "known_acceptance": float(np.mean(clean_accepted)),
        "Score": official_score(c, 0.7 * ra + 0.3 * asafer, u),
    }


def select_point(runs: list[dict], config: dict) -> dict:
    best = None
    for score_index, score_name in enumerate(config["evaluation"]["threshold_scores"]):
        known = np.concatenate([confidence_values(r["known_scores"])[score_name] for r in runs])
        thresholds = np.unique(np.quantile(known, np.linspace(0.0, 0.05, 101)))
        for threshold in thresholds:
            per_attack = {name: components(run, name, score_name, float(threshold))
                          for run in runs for name in ATTACK_ORDER}
            macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
                     for key in COMPONENT_KEYS}
            macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
            if macro["known_acceptance"] < 0.95 - 1e-12:
                continue
            key = (macro["Score"], macro["RA"], macro["ASafeR"], macro["C"], macro["U"], -score_index)
            if best is None or key > best[0]:
                best = (key, score_name, float(threshold), macro)
    return {"score_name": best[1], "threshold": best[2], "macro": best[3]}


def evaluate_point(runs: list[dict], score_name: str, threshold: float) -> dict:
    per_attack = {}
    for name in ATTACK_ORDER:
        rows = [components(run, name, score_name, threshold) for run in runs]
        per_attack[name] = {key: float(np.mean([row[key] for row in rows]))
                            for key in COMPONENT_KEYS}
        per_attack[name]["Score"] = official_score(
            per_attack[name]["C"], per_attack[name]["A"], per_attack[name]["U"]
        )
    macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
             for key in COMPONENT_KEYS}
    macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
    return {"per_attack": per_attack, "macro": macro}


def clean_metrics(runs: list[dict], score_name: str, threshold: float) -> dict:
    known_true = np.concatenate([r["known_true"] for r in runs])
    known_pred = np.concatenate([r["classes"][r["known_scores"].argmax(1)] for r in runs])
    known_conf = np.concatenate([confidence_values(r["known_scores"])[score_name] for r in runs])
    unknown_conf = np.concatenate([confidence_values(r["unknown_scores"])[score_name] for r in runs])
    classes = np.unique(np.concatenate([r["classes"] for r in runs]))
    accepted = known_conf >= threshold
    final = np.where(accepted, known_pred, 0)
    labels = np.concatenate([np.ones(len(known_conf)), np.zeros(len(unknown_conf))])
    values = np.concatenate([known_conf, unknown_conf])
    fpr, tpr, _ = roc_curve(labels, values)
    return {
        "direct_C": float(np.mean(known_pred == known_true)),
        "identity_macro_f1_direct": float(f1_score(
            known_true, known_pred, labels=classes, average="macro", zero_division=0
        )),
        "C": float(np.mean(final == known_true)),
        "identity_macro_f1": float(f1_score(
            known_true, final, labels=classes, average="macro", zero_division=0
        )),
        "U": float(np.mean(unknown_conf < threshold)),
        "unknown_auroc": float(roc_auc_score(labels, values)),
        "unknown_aupr": float(average_precision_score(1 - labels, -values)),
        "eer": float(np.mean(np.abs(fpr - (1 - tpr))[np.argmin(np.abs(fpr - (1 - tpr)))])) / 2,
        "known_acceptance": float(np.mean(accepted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "inner"), default="inner")
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("E37 要求 CUDA GPU")
    started = time.time()
    device = torch.device("cuda")
    config = load_config()
    e36_config = load_e36_config()
    metadata = train_metadata(e36_config)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(e36_config["nested_protocol"]["outer_seed"]).permutation(all_subjects),
        5,
    )
    variants = args.variants or config["training"]["variants"]
    if args.stage == "smoke":
        variants = [config["smoke"]["variant"]]
        outer_groups = outer_groups[:1]

    raw = load_all_raw(metadata, e36_config, device)
    results = {"status": "running", "experiment": "E37", "date": "2026-11-13",
               "variants": {}, "stage": args.stage}
    for variant in variants:
        folds = []
        for outer_index, unknown_subjects in enumerate(outer_groups):
            outer_fold = outer_index + 1
            known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
            outer_known = np.isin(subjects, known_subjects)
            inner_order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
            inner_splits = list(np.array_split(inner_order, 2))
            if args.stage == "smoke":
                inner_splits = inner_splits[:1]
            runs = []
            infos = []
            for inner_index, inner_unknown in enumerate(inner_splits):
                inner_fold = inner_index + 1
                enrolled = np.setdiff1d(known_subjects, inner_unknown)
                enrolled_mask = np.isin(subjects, enrolled)
                fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
                known_np = np.flatnonzero(enrolled_mask & inner_block)
                unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
                epsilon = calibrate_epsilon(
                    raw, zband, metadata, fit_np, subjects, enrolled, config, e36_config, device
                )
                seed = int(config["seed_base"] + 1000 * outer_fold + inner_fold)
                model, info = train_variant(
                    variant, zband, raw, subjects, fit_np, known_np, unknown_np, enrolled,
                    epsilon, config, e36_config, seed, device,
                )
                info["epsilon"] = epsilon
                run = run_attacks_for_model(
                    model, raw, zband, metadata, known_np, unknown_np, subjects, enrolled,
                    config, e36_config, 1, outer_fold, inner_fold,
                )
                runs.append(run)
                infos.append(info)
                del model
                torch.cuda.empty_cache()
            selection = select_point(runs, config)
            clean = clean_metrics(runs, selection["score_name"], selection["threshold"])
            evaluated = evaluate_point(runs, selection["score_name"], selection["threshold"])
            folds.append({"outer_fold": outer_fold, "train_info": infos,
                          "selection": selection, "clean": clean,
                          "per_attack": evaluated["per_attack"], "macro": evaluated["macro"]})
            print(f"{variant} outer={outer_fold}: Score={evaluated['macro']['Score']:.2f} "
                  f"RA={evaluated['macro']['RA']:.4f} ASafeR={evaluated['macro']['ASafeR']:.4f} "
                  f"C={evaluated['macro']['C']:.4f} U={evaluated['macro']['U']:.4f}", flush=True)
            results["variants"][variant] = {"folds": folds}
            write_json(RESULTS_PATH, results)
        macro = {key: float(np.mean([f["macro"][key] for f in folds])) for key in
                 list(folds[0]["macro"].keys())}
        clean = {key: float(np.mean([f["clean"][key] for f in folds])) for key in
                 list(folds[0]["clean"].keys())}
        results["variants"][variant]["macro_fold_mean"] = macro
        results["variants"][variant]["clean_fold_mean"] = clean
        write_json(RESULTS_PATH, results)
        print(f"{variant} MACRO: Score={macro['Score']:.2f} RA={macro['RA']:.4f} "
              f"ASafeR={macro['ASafeR']:.4f} direct_RA={macro['direct_RA']:.4f} "
              f"C={macro['C']:.4f} U={macro['U']:.4f}", flush=True)
    results["status"] = "complete"
    results["elapsed_seconds"] = time.time() - started
    write_json(RESULTS_PATH, results)
    print(f"完成：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
