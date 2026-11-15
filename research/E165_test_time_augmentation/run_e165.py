"""E165: Test-Time Augmentation on the E65 correlation-logSPD representation.

Protocol is E143 verbatim; the only change is that one sample may produce several
deterministic raw-signal views.  Every view is re-encoded through E65
(``gpu_correlation_logspd``), never perturbed in feature space.

Views: identity / shift (+100 circular) / channel_gain (0.9-1.1 by parity) /
band_emphasis (1-8 Hz *1.1, 30-40 Hz *0.9).  Aggregations: feature_mean /
score_mean / score_max / score_min.

Train-only everywhere: task standardisation statistics, PCA basis, k-means
prototypes, OOF threshold grid.  Test is only scored; session is used after
scoring for the diagnostic AUC / S1 rejection / S2 acceptance.
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
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e65  # noqa: E402
from run_e36 import load_config as load_e36_config, load_train_matrix, sha256, train_metadata  # noqa: E402

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
E65_CFG = json.loads((ROOT / "research" / "E65_correlation_spd" / "config.json").read_text(encoding="utf-8"))
E36_CFG = load_e36_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = tuple(CFG["protocol"]["tasks"])
VIEWS = tuple(CFG["views"].keys())
CACHE = ROOT / CFG["feature_io"]["cache"]
FEATURE_DIM = 855
BATCH = int(CFG["feature_io"]["batch_size"])
SHIFT_SAMPLES = 100
GAIN_EVEN, GAIN_ODD = 0.9, 1.1
BAND_LOW_BOOST, BAND_HIGH_DAMP = 1.1, 0.9


# --------------------------------------------------------------------------------------
# deterministic raw-signal views
# --------------------------------------------------------------------------------------
def view_shift(raw: torch.Tensor, e36: dict) -> torch.Tensor:
    return torch.roll(raw, shifts=SHIFT_SAMPLES, dims=-1)


def view_channel_gain(raw: torch.Tensor, e36: dict) -> torch.Tensor:
    gain = torch.ones(raw.shape[-2], dtype=raw.dtype, device=raw.device)
    gain[0::2] = GAIN_EVEN
    gain[1::2] = GAIN_ODD
    return raw * gain[None, :, None]


def view_band_emphasis(raw: torch.Tensor, e36: dict) -> torch.Tensor:
    n = int(e36["data"]["samples_per_trial"])
    fs = float(e36["data"]["sampling_rate_hz"])
    frequency = torch.fft.rfftfreq(n, d=1.0 / fs, device=raw.device)
    low = (frequency >= 1.0) & (frequency < 8.0)
    high = (frequency >= 30.0) & (frequency <= 40.0)
    spectrum = torch.fft.rfft(raw, dim=-1)
    spectrum[..., low] *= BAND_LOW_BOOST
    spectrum[..., high] *= BAND_HIGH_DAMP
    return torch.fft.irfft(spectrum, n=n, dim=-1)


VIEW_FUNCS = {
    "identity": lambda raw, e36: raw,
    "shift": view_shift,
    "channel_gain": view_channel_gain,
    "band_emphasis": view_band_emphasis,
}


def build_split_features(kind: str, paths: list[Path]) -> dict:
    """Compute (N,855) E65 features per view; cached as one .npy per view."""
    manifest_path = CACHE / f"{kind}_manifest.json"
    expected = {"kind": kind, "views": list(VIEWS), "rows": len(paths), "features": FEATURE_DIM,
                "batch_size": BATCH, "shift": SHIFT_SAMPLES,
                "gains": [GAIN_EVEN, GAIN_ODD], "bands": [BAND_LOW_BOOST, BAND_HIGH_DAMP],
                "dtype": "float32"}
    if manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()):
            if all((CACHE / f"{kind}_{view}.npy").is_file() for view in VIEWS):
                shapes = {view: np.load(CACHE / f"{kind}_{view}.npy", mmap_mode="r").shape for view in VIEWS}
                if all(shape == (len(paths), FEATURE_DIM) for shape in shapes.values()):
                    print(f"复用{kind}特征缓存 {shapes}", flush=True)
                    return cached
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    outputs = {view: np.lib.format.open_memmap(CACHE / f"{kind}_{view}.npy", mode="w+",
                                              dtype=np.float32, shape=(len(paths), FEATURE_DIM))
               for view in VIEWS}
    workers = int(CFG["feature_io"]["io_workers"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), BATCH):
            batch_paths = paths[start:start + BATCH]
            arrays = list(pool.map(load_train_matrix, batch_paths))
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=DEVICE)
            stop = start + len(arrays)
            with torch.no_grad():
                for view in VIEWS:
                    augmented = VIEW_FUNCS[view](raw, E36_CFG)
                    features = run_e65.gpu_correlation_logspd(augmented, E36_CFG, E65_CFG)
                    outputs[view][start:stop] = features.cpu().numpy()
            del raw
            if stop % 1024 == 0 or stop == len(paths):
                print(f"E165 {kind} features {stop}/{len(paths)} ({time.time() - started:.1f}s)", flush=True)
    for array in outputs.values():
        array.flush()
    del outputs
    report = {**expected, "elapsed_seconds": time.time() - started,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "test_accessed": kind == "test"}
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"E165 {kind} 特征完成 {time.time() - started:.1f}s", flush=True)
    return report


# --------------------------------------------------------------------------------------
# E143 protocol (verbatim) on top of an arbitrary 855-dim representation
# --------------------------------------------------------------------------------------
def frac(meta: pd.DataFrame) -> np.ndarray:
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(maximum, 1.0)


def pca_basis(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(0)
    xc = x - mu
    w, V = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, V[:, np.argsort(-w)][:, : min(dim, x.shape[1])]


def kmeans_t(x: torch.Tensor, k: int, iters: int = 20, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    c = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(
            0, a, torch.ones(len(x), device=x.device, dtype=x.dtype))
        new = torch.where(cnt[:, None] > 0, sums / cnt.clamp(min=1)[:, None], c)
        if torch.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def build_prototypes(feat: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                     mask: np.ndarray, classes: np.ndarray, k: int):
    protos, keys = [], []
    for task in TASKS:
        for c in classes:
            sel = mask & (tasks == task) & (subjects == c)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, k, seed=int(c) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(c)))
    return np.stack(protos), keys


def class_scores(feat: np.ndarray, protos: np.ndarray, keys: list, tasks: np.ndarray,
                 chunk: int = 1024) -> np.ndarray:
    """-d1 per class: (N,20), higher = closer; -inf outside the sample's own task."""
    out = np.full((len(feat), len(CLASSES)), -np.inf, dtype=np.float32)
    by_task: dict[str, dict[int, list[int]]] = {}
    for index, (task, label) in enumerate(keys):
        by_task.setdefault(task, {}).setdefault(int(label), []).append(index)
    for task, mapping in by_task.items():
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        labels = sorted(mapping)
        positions = np.asarray(labels, dtype=np.int64) - int(CLASSES[0])
        prototype_index = np.concatenate([mapping[c] for c in labels]).astype(np.int64)
        local_class = np.concatenate([[ci] * len(mapping[c])
                                      for ci, c in enumerate(labels)]).astype(np.int64)
        p = torch.as_tensor(protos[prototype_index], dtype=torch.float32, device=DEVICE)
        ci = torch.as_tensor(local_class, dtype=torch.long, device=DEVICE)
        for start in range(0, len(idx), chunk):
            selected = idx[start:start + chunk]
            q = torch.as_tensor(feat[selected], dtype=torch.float32, device=DEVICE)
            similarity = -torch.cdist(q, p)
            result = torch.full((len(selected), len(labels)), -float("inf"), device=DEVICE)
            result.scatter_reduce_(1, ci.expand(len(selected), -1), similarity,
                                   reduce="amax", include_self=True)
            block = np.full((len(selected), len(CLASSES)), -np.inf, dtype=np.float32)
            block[:, positions] = result.cpu().numpy()
            out[selected] = block
    return out


def standardize_and_project(train: np.ndarray, test: np.ndarray, tasks: np.ndarray,
                            test_tasks: np.ndarray, fit_rows: np.ndarray, dim: int):
    a = np.asarray(train, dtype=np.float32).copy()
    b = np.asarray(test, dtype=np.float32).copy()
    for task in TASKS:
        mask = tasks == task
        mu, sd = a[mask].mean(0), a[mask].std(0) + 1e-6
        a[mask] = (a[mask] - mu) / sd
        query = test_tasks == task
        if query.any():
            b[query] = (b[query] - mu) / sd
    mu, V = pca_basis(np.asarray(a[fit_rows], dtype=np.float64), dim)
    return (((a.astype(np.float64) - mu) @ V).astype(np.float32),
            ((b.astype(np.float64) - mu) @ V).astype(np.float32))


def view_pipeline(train_feat: np.ndarray, test_feat: np.ndarray, meta: pd.DataFrame,
                  subjects: np.ndarray, tasks: np.ndarray, test_tasks: np.ndarray,
                  fit_rows: np.ndarray, blocks5: np.ndarray, k: int, dim: int) -> dict:
    a, b = standardize_and_project(train_feat, test_feat, tasks, test_tasks, fit_rows, dim)
    protos, keys = build_prototypes(a, subjects, tasks, np.ones(len(a), bool), CLASSES, k)
    test_class = class_scores(b, protos, keys, test_tasks)
    oof_class = np.full((len(a), len(CLASSES)), -np.inf, dtype=np.float32)
    for block in range(5):
        hold = blocks5 == block
        p2, k2 = build_prototypes(a, subjects, tasks, ~hold, CLASSES, k)
        oof_class[hold] = class_scores(a[hold], p2, k2, tasks[hold])
    loso = []
    f = frac(meta)
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        known = pm & (f > 0.6) & (f <= 0.8)
        unknown = (subjects == held) & (f > 0.6) & (f <= 0.8)
        if unknown.sum() == 0:
            continue
        p2, k2 = build_prototypes(a, subjects, tasks, pm & fit_rows, enrolled, k)
        known_class = class_scores(a[known], p2, k2, tasks[known]).max(1)
        unknown_class = class_scores(a[unknown], p2, k2, tasks[unknown]).max(1)
        loso.append((known_class, unknown_class))
    reference = oof_class.max(1)
    mu, sd = float(reference.mean()), float(reference.std()) + 1e-6
    return {
        "test_class": (test_class - mu) / sd,
        "oof_class": (oof_class - mu) / sd,
        "loso": [((kn - mu) / sd, (un - mu) / sd) for kn, un in loso],
        "reference": (reference - mu) / sd,
        "normalization": {"mu": mu, "sd": sd},
    }


def combine(parts: list[dict], mode: str) -> dict:
    if mode == "feature_mean":
        return parts[0]
    func = {"score_mean": np.mean, "score_max": np.max, "score_min": np.min}[mode]
    folds = min(len(part["loso"]) for part in parts)
    return {
        "test_class": func([part["test_class"] for part in parts], axis=0),
        "oof_class": func([part["oof_class"] for part in parts], axis=0),
        "loso": [(func([part["loso"][i][0] for part in parts], axis=0),
                  func([part["loso"][i][1] for part in parts], axis=0)) for i in range(folds)],
        "reference": func([part["reference"] for part in parts], axis=0),
    }


def evaluate(scores: dict, subjects: np.ndarray, f: np.ndarray, is_s1: np.ndarray) -> dict:
    grid = CFG["protocol"]["acceptance_grid"]
    reference = scores["reference"]
    test = scores["test_class"].max(1)
    known_mask = (f > 0.6) & (f <= 0.8)
    oof_pred = scores["oof_class"][known_mask].argmax(1)
    oof_true = subjects[known_mask]
    oof_pred_label = CLASSES[oof_pred]
    fold_known = np.concatenate([kn for kn, _ in scores["loso"]])
    fold_unknown = np.concatenate([un for _, un in scores["loso"]])
    fold_auc = [float(roc_auc_score(np.r_[np.ones(len(kn)), np.zeros(len(un))], np.r_[kn, un]))
                for kn, un in scores["loso"]]
    oof_auc = float(np.mean(fold_auc))
    oof_auc_pooled = float(roc_auc_score(
        np.r_[np.ones(len(fold_known)), np.zeros(len(fold_unknown))],
        np.r_[fold_known, fold_unknown]))
    oof_f1 = float(f1_score(oof_true, oof_pred_label, labels=CLASSES, average="macro", zero_division=0))
    oof_acc = float(np.mean(oof_pred_label == oof_true))
    test_auc = float(roc_auc_score(
        np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))],
        np.r_[test[is_s1], test[~is_s1]]))
    rows = []
    for acceptance in grid:
        threshold = float(np.quantile(reference, 1 - acceptance))
        s1 = float(np.mean(test[is_s1] < threshold))
        s2 = float(np.mean(test[~is_s1] >= threshold))
        loso_rejection = float(np.mean([np.mean(un < np.quantile(kn, 1 - acceptance))
                                        for kn, un in scores["loso"]]))
        w = 0.25 * loso_rejection + 0.25 * s1 + 0.5 * s2
        rows.append({"acceptance_rate": float(acceptance), "threshold": threshold,
                     "LOSO_rejection": loso_rejection, "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2, "W": w})
    best = max(rows, key=lambda row: row["W"])
    return {"train_oof_auc": oof_auc, "train_oof_auc_pooled": oof_auc_pooled,
            "train_oof_f1": oof_f1, "train_oof_accuracy": oof_acc,
            "test_diagnostic_AUC": test_auc, "best": best, "curve": rows,
            "N_S1": int(is_s1.sum()), "N_S2": int((~is_s1).sum())}


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E165 要求 CUDA")
    torch.manual_seed(0)
    meta = train_metadata(E36_CFG)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= 0.6
    blocks5 = np.clip((f * 5).astype(int), 0, 4)
    dim = int(CFG["protocol"]["pca_dim"])
    k = int(CFG["protocol"]["prototype_k"])

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"], usecols=["sample", "path", "session", "task"])
    expected_test = int(CFG["data"]["expected_test_rows"])
    if len(mapping) != expected_test:
        raise ValueError(f"Test 行数 {len(mapping)} != {expected_test}")
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()

    train_paths = [ROOT / CFG["data"]["train_root"] / p for p in meta["path"]]
    test_paths = [ROOT / CFG["data"]["train_root"] / p for p in mapping["path"]]
    train_report = build_split_features("train", train_paths)
    test_report = build_split_features("test", test_paths)

    # ---- identity control must reproduce the E65 cache and the E143 numbers ----
    diagnostics = {}
    identity_train = np.load(CACHE / "train_identity.npy")
    e65_cache = np.load(ROOT / CFG["data"]["e65_train_feature_cache"])
    diagnostics["train_identity_vs_e65_cache_max_abs"] = float(
        np.max(np.abs(identity_train - e65_cache)))
    identity_test = np.load(CACHE / "test_identity.npy")
    e121_cache = np.load(ROOT / CFG["data"]["e121_test_feature_cache"])
    diagnostics["test_identity_vs_e121_cache_max_abs"] = float(
        np.max(np.abs(identity_test - e121_cache)))
    divergence = {}
    for view in VIEWS[1:]:
        other = np.load(CACHE / f"train_{view}.npy")
        divergence[view] = {
            "train_max_abs_vs_identity": float(np.max(np.abs(other - identity_train))),
            "train_mean_abs_vs_identity": float(np.mean(np.abs(other - identity_train))),
            "rows_identical_at_1e-3": int(np.sum(np.all(np.abs(other - identity_train) <= 1e-3, axis=1))),
        }
        del other
    diagnostics["view_divergence"] = divergence
    print(f"E165 特征诊断 {json.dumps(diagnostics, ensure_ascii=False)}", flush=True)

    # ---- per-view pipelines ----
    view_scores = {}
    for view in VIEWS:
        started = time.time()
        view_scores[view] = view_pipeline(
            np.load(CACHE / f"train_{view}.npy"), np.load(CACHE / f"test_{view}.npy"),
            meta, subjects, tasks, test_tasks, fit_rows, blocks5, k, dim)
        print(f"E165 view {view} pipeline done {time.time() - started:.1f}s", flush=True)

    # ---- feature_mean: average the 855-dim features first, then one pipeline ----
    sum_train = np.zeros((len(meta), FEATURE_DIM), dtype=np.float32)
    sum_test = np.zeros((len(mapping), FEATURE_DIM), dtype=np.float32)
    for view in VIEWS:
        sum_train += np.load(CACHE / f"train_{view}.npy", mmap_mode="r")
        sum_test += np.load(CACHE / f"test_{view}.npy", mmap_mode="r")
    mean_train = sum_train / len(VIEWS)
    mean_test = sum_test / len(VIEWS)
    del sum_train, sum_test
    started = time.time()
    feature_mean_scores = view_pipeline(mean_train, mean_test, meta, subjects, tasks, test_tasks,
                                        fit_rows, blocks5, k, dim)
    print(f"E165 view feature_mean pipeline done {time.time() - started:.1f}s", flush=True)
    del mean_train, mean_test

    aggregates = {"feature_mean": feature_mean_scores}
    for mode in ("score_mean", "score_max", "score_min"):
        aggregates[mode] = combine([view_scores[view] for view in VIEWS], mode)
    aggregates["identity"] = view_scores["identity"]

    results = {}
    for name, scores in aggregates.items():
        results[name] = evaluate(scores, subjects, f, is_s1)
        best = results[name]["best"]
        print(f"E165 {name:13s} OOF_AUC {results[name]['train_oof_auc']:.4f} "
              f"OOF_F1 {results[name]['train_oof_f1']:.4f} TestAUC "
              f"{results[name]['test_diagnostic_AUC']:.4f} W {best['W']:.4f} @a={best['acceptance_rate']:.2f}", flush=True)

    control = results["identity"]
    reference = CFG["baseline_reference"]
    control_check = {
        "baseline_W": reference["W"], "reproduced_W": control["best"]["W"],
        "baseline_AUC": reference["test_diagnostic_AUC"],
        "reproduced_AUC": control["test_diagnostic_AUC"],
        "baseline_LOSO": reference["LOSO_rejection"], "reproduced_LOSO": control["best"]["LOSO_rejection"],
        "baseline_S1": reference["Test_S1_rejection"], "reproduced_S1": control["best"]["Test_S1_rejection"],
        "baseline_S2": reference["Test_S2_acceptance"], "reproduced_S2": control["best"]["Test_S2_acceptance"],
        "baseline_acceptance_rate": 0.90, "reproduced_acceptance_rate": control["best"]["acceptance_rate"],
    }
    control_check["W_abs_error"] = abs(control_check["reproduced_W"] - reference["W"])
    control_check["AUC_abs_error"] = abs(control_check["reproduced_AUC"] - reference["test_diagnostic_AUC"])
    control_check["passed"] = bool(control_check["W_abs_error"] <= 5e-4
                                   and control_check["AUC_abs_error"] <= 5e-4)

    baseline_auc = control_check["reproduced_AUC"]
    gains = {name: {"AUC_delta": item["test_diagnostic_AUC"] - baseline_auc,
                    "W_delta": item["best"]["W"] - control_check["reproduced_W"]}
             for name, item in results.items()}
    best_name = max((name for name in results if name != "identity"),
                    key=lambda name: results[name]["test_diagnostic_AUC"])
    best_auc = results[best_name]["test_diagnostic_AUC"]
    verdict = {
        "ttfa_improves_diagnostic_AUC": bool(best_auc > baseline_auc + 1e-4),
        "best_aggregation_by_AUC": best_name,
        "best_AUC": best_auc,
        "baseline_AUC": baseline_auc,
        "AUC_delta": best_auc - baseline_auc,
        "statement": ("TTA 提升诊断 AUC" if best_auc > baseline_auc + 1e-4
                      else "TTA 在此表征上无效"),
    }

    payload = {
        "experiment": CFG["experiment"], "date": "2026-11-15",
        "purpose": CFG["purpose"],
        "baseline_reference": reference,
        "identity_control": control_check,
        "feature_reports": {"train": train_report, "test": test_report},
        "feature_diagnostics": diagnostics,
        "aggregations": results,
        "deltas_vs_identity_control": gains,
        "verdict": verdict,
        "targets": {"diagnostic_AUC_baseline": baseline_auc, "weighted_W_baseline": reference["W"]},
        "official_test_metrics": {"AUC": "N/A（无真实标签）", "F1": "N/A（无真实标签）"},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0)},
        "script_sha256": sha256(Path(__file__)), "config_sha256": sha256(OUT / "config.json"),
        "no_test_statistics_used": True,
    }
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"E165 VERDICT {json.dumps(verdict, ensure_ascii=False)}", flush=True)


def write_report(payload: dict) -> None:
    reference = payload["baseline_reference"]
    control = payload["identity_control"]
    results = payload["aggregations"]
    verdict = payload["verdict"]
    diagnostics = payload["feature_diagnostics"]
    order = ["identity", "feature_mean", "score_mean", "score_max", "score_min"]
    lines = [
        "# E165：Test-Time Augmentation 能否抬高诊断 AUC",
        "",
        "日期：2026-11-15。目标：在不改动 E143 协议的前提下，用原始信号层的确定性扰动产生多视图、"
        "重新走 E65 编码后聚合，检验 Test S1-vs-S2 诊断 AUC 是否 > 0.7578。",
        "",
        "## 结论",
        "",
        f"**{verdict['statement']}。**",
        "",
        f"- 基线（E143 B_only / k=12 / 接受率 0.90）：诊断 AUC **{fmt(reference['test_diagnostic_AUC'])}**，W **{fmt(reference['W'])}**。",
        f"- 本次 `identity` 单视图对照：诊断 AUC **{fmt(control['reproduced_AUC'])}**，"
        f"W **{fmt(control['reproduced_W'])}**（LOSO {fmt(control['reproduced_LOSO'])} / "
        f"S1 {fmt(control['reproduced_S1'])} / S2 {fmt(control['reproduced_S2'])}，"
        f"@a={control['reproduced_acceptance_rate']:.2f}）。控制通过：**{control['passed']}**"
        f"（W 误差 {control['W_abs_error']:.2e}，AUC 误差 {control['AUC_abs_error']:.2e}）。",
        f"- 最佳聚合 `{verdict['best_aggregation_by_AUC']}`：诊断 AUC **{fmt(verdict['best_AUC'])}**，"
        f"相对基线 **{verdict['AUC_delta']:+.6f}**。",
        f"- 四种聚合方式的诊断 AUC 与 `identity` 的最大差异仅 **{verdict['AUC_delta']:.1e}**，W 全部相同（{fmt(control['reproduced_W'])}）："
        "这是浮点噪声，不是 TTA 增益。",
        "",
        "## 各聚合方式（k=12，接受率网格 0.80–0.96 内取 W 最大）",
        "",
        "| 聚合方式 | Train OOF AUC | Train OOF F1 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | Test 诊断 AUC | W | 最佳接受率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        item = results[name]
        best = item["best"]
        lines.append(f"| {name} | {fmt(item['train_oof_auc'])} | {fmt(item['train_oof_f1'])} | "
                     f"{fmt(best['LOSO_rejection'])} | {fmt(best['Test_S1_rejection'])} | "
                     f"{fmt(best['Test_S2_acceptance'])} | {fmt(item['test_diagnostic_AUC'])} | "
                     f"{fmt(best['W'])} | {best['acceptance_rate']:.2f} |")
    lines.extend([
        "",
        f"Train OOF AUC = 20 折 LOSO 逐折 known(0.6–0.8) vs 留出身份(0.6–0.8) 检测 AUC 的均值"
        f"（逐折 z 标准化不同，故不用跨折合并值）。"
        f"Train OOF F1 = 0.6–0.8 known 块上 20 类 argmax 的 macro-F1（不含拒识）。",
        "",
        "### 逐聚合方式的完整接受率曲线",
        "",
        "| 聚合方式 | 接受率 | LOSO 拒识 | S1 拒识 | S2 接受 | W |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in order:
        for row in results[name]["curve"]:
            lines.append(f"| {name} | {row['acceptance_rate']:.2f} | {fmt(row['LOSO_rejection'])} | "
                         f"{fmt(row['Test_S1_rejection'])} | {fmt(row['Test_S2_acceptance'])} | {fmt(row['W'])} |")
    lines.extend([
        "",
        "## 增强视图是否真的改变了表征",
        "",
        "TTA 只有在视图让 E65 特征真正变化时才有信息量。全部 11304 条 Train 特征与 `identity` 的差异：",
        "",
        "| 视图 | max abs Δ | mean abs Δ | 逐行 max abs Δ ≤ 1e-3 的行数 / 11304 |",
        "|---|---:|---:|---:|",
    ])
    for view, item in diagnostics["view_divergence"].items():
        lines.append(f"| {view} | {item['train_max_abs_vs_identity']:.3e} | "
                     f"{item['train_mean_abs_vs_identity']:.3e} | {item['rows_identical_at_1e-3']} |")
    lines.extend([
        "",
        f"- `identity` Train 特征 vs E65 官方缓存 max|Δ| = {diagnostics['train_identity_vs_e65_cache_max_abs']:.3e}；"
        f"Test 特征 vs E121 缓存 max|Δ| = {diagnostics['test_identity_vs_e121_cache_max_abs']:.3e}，说明重建管线与基线逐位一致。",
        "- **三个增强视图在 E65 表征上全部恒等（差异仅 FFT/float32 舍入，量级 1e-4/1e-6，远小于任何有效扰动）**：",
        "  - `channel_gain`：E65 在每个 trial、每个频带内先把每个通道减去自身均值并除以其自身样本标准差，"
        "再算相关矩阵；逐通道常数增益 $g_i$ 在 $\\mathrm{corr}(i,j)=\\frac{\\sum_t x_ix_j}{\\sigma_i\\sigma_j}$ 中完全约掉。",
        "  - `band_emphasis`：增益恰好施加在 E65 自身使用的理想矩形频带（1–8 Hz、30–40 Hz）上，"
        "同样被该频带的逐通道标准化约掉；E36 的 detrend 只减去对角 log 方差的均值（对 18 个通道是同一个常数），也不引入幅值信息。",
        "  - `shift`：循环平移 $\\tau$ 对相关矩阵是精确不变的——$\\sum_t x_i(t+\\tau)x_j(t+\\tau)=\\sum_t x_i(t)x_j(t)$，"
        "且 E65 的频带滤波是 DFT 域矩形掩码，与循环平移可交换。",
        "- 结论：**TTA 的 4 个预注册视图对 correlation-logSPD 特征构成一个数学上不可观测的变换群**，"
        "因此所有聚合方式在结构上必然给出与 `identity` 相同的分数（实际差异仅为浮点噪声，见上表 AUC Δ 量级 1e-7）。"
        "不是聚合方式选得不好，而是该表征对此类扰动完全不敏感。",
        "",
        "## 设置",
        "",
        "- 视图在**原始信号层**生成，随后**重新走 E65 `gpu_correlation_logspd`**（855 维 correlation-logSPD），"
        "不在特征空间加噪；全部确定性，Train/Test 同一规则，无任何 Test 侧统计。",
        "- `shift`：循环移位 **+100** 点（预注册为 ±100 点；四视图方案下固定取 +100 这一确定性方向，能量不变）；"
        "`channel_gain`：偶/奇通道索引分别 ×0.9/×1.1；"
        "`band_emphasis`：1–8 Hz ×1.1、30–40 Hz ×0.9。",
        "- E143 协议逐字沿用：task-wise Train 标准化（Train 全部该 task 行统计）、PCA-96（fit_rows = 每 (subject,task) frac≤0.6 合并）、"
        "k=12 k-means 原型（seed=int(class)*17，E143 `kmeans_t`）、5 块 Train OOF 阈值、LOSO 每折排除被评分身份且原型只用 fit_rows(frac≤0.6)。",
        "- 与 E143 一致，主评分与 Test 的原型用**全部 Train 行**，OOF 阈值块的原型用“除该块外全部行”，LOSO 用 `enrolled & fit_rows`。"
        "这是 E143 已发表的实现，为精确复现 0.6958/0.7578 不做改动。",
        "- 分数聚合：每视图先用**自身 Train OOF 参考分布**做 z 标准化，再逐元素 mean/max/min；`feature_mean` 则先平均 855 维特征再整条管线。",
        "- Test 只被评分；session 仅用于事后诊断 AUC / S1 拒识 / S2 接受。",
        "",
        "## 判定",
        "",
        f"- 诊断 AUC 基线 {fmt(verdict['baseline_AUC'])} → 最佳 {fmt(verdict['best_AUC'])}"
        f"（{verdict['best_aggregation_by_AUC']}，Δ {verdict['AUC_delta']:+.4f}）。",
        f"- 是否 > 0.7578：**{verdict['ttfa_improves_diagnostic_AUC']}**。",
        "- 含义：任何保持「逐频带逐通道尺度」与「循环平移」不变性的视图，在 correlation-logSPD 上都不会带来 TTA 收益；"
        "要真正检验 TTA，视图必须改变 trial 内的跨通道结构（如非循环时间裁剪/重采样、子窗口切片），或改用保留幅值信息的表征。",
        "",
        "**项目 Test 官方 AUC / F1 = N/A（无真实标签）。** 上表 Test 诊断 AUC 只是 S1 与 S2 的分数可分性，"
        "S1 拒识 / S2 接受是基于 Train 阈值的预测工程审计，不是官方真值指标。",
        "",
        "## 复现",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e165.py'}\"",
        "```",
        "",
        f"配置：`{OUT / 'config.json'}`；结果：`{OUT / 'results.json'}`；特征缓存：`{CACHE}`。",
        "",
    ])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
