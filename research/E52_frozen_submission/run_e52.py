"""E52：把 E49 三段式系统固化为 Test 提交。

  - CNN 身份模型：全 Train 20 类，fit 块 trial <= 0.8*max，含 PGD7 对抗训练
  - logSPD 拒识模型：同 fit 块 WCCN
  - 阈值：校准块 trial > 0.8*max 的 distance_ratio 的 q0.02 / q0.05（未参与训练）
  - 推理：TTA5 平均 CNN logits；dr 三段式决策
Test 只做预测与无标签审计，Session 不参与决策。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))
sys.path.insert(0, str(OUT.parent / "E45_raw_cnn_at"))

from run_e36 import detrend_zband, gpu_logspd, load_config as load_e36_config, load_train_matrix, train_metadata  # noqa: E402
from run_attacks import load_all_raw  # noqa: E402
from run_e40 import FrozenPrototype, confidence_values, fit_prototype  # noqa: E402
from run_e45 import train_fold  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
TEST_METADATA_PATH = ROOT / "BrainPrint_cache" / "test_metadata.csv"
TEST_ROOT = ROOT / "Brainprint_Competition"
EXPECTED_TEST_ROWS = 22251
SHIFTS = (-80, -40, 0, 40, 80)


def column(frame: pd.DataFrame, name: str) -> str:
    matches = [item for item in frame.columns if item.casefold() == name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"元数据必须且只能包含一个 {name} 字段")
    return matches[0]


def cnn_logits_tta(model, samples, batch=256) -> np.ndarray:
    output = []
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            views = [model(chunk if shift == 0 else torch.roll(chunk, shift, dims=-1))
                     for shift in SHIFTS]
            output.append(torch.stack(views, dim=0).mean(0).cpu().numpy())
    return np.concatenate(output)


def logspd_scores(model, samples, batch=512) -> np.ndarray:
    output = []
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            features = detrend_zband(gpu_logspd(chunk, model.config), model.config)
            whitened = features @ model.whitening.to(features.dtype)
            normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
            output.append((normalized @ model.prototypes.to(features.dtype).T).cpu().numpy())
    return np.concatenate(output)


def load_test_raw(metadata: pd.DataFrame, config: dict, device: torch.device,
                  batch: int = 256) -> torch.Tensor:
    paths = [TEST_ROOT / str(item) for item in metadata["path"]]
    output = torch.empty(
        (len(paths), config["data"]["channels"], config["data"]["samples_per_trial"]),
        dtype=torch.float32, device=device,
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=int(config["features"]["io_workers"])) as pool:
        for start in range(0, len(paths), batch):
            arrays = list(pool.map(load_train_matrix, paths[start:start + batch]))
            output[start:start + len(arrays)] = torch.as_tensor(
                np.stack(arrays), dtype=torch.float32, device=device)
            if (start // batch + 1) % 20 == 0:
                print(f"Test 波形 {start + len(arrays)}/{len(paths)} ({time.time() - started:.0f}s)",
                      flush=True)
    return output


def session_audit(metadata: pd.DataFrame, labels: np.ndarray) -> dict:
    session = metadata[column(metadata, "Session")].astype(str).str.strip().str.casefold()
    codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                         "2": "S2", "s2": "S2", "session2": "S2"})
    if codes.isna().any():
        raise ValueError("离线审计遇到未知 Session 取值")
    output = {"session_used_by_inference": False}
    for code in ("S1", "S2"):
        selected = codes.to_numpy() == code
        output[f"{code}_n"] = int(selected.sum())
        output[f"{code}_rejection_rate"] = (
            float(np.mean(labels[selected] == 0)) if selected.any() else None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=OUT / "e52_predictions.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("E52 要求 CUDA GPU")
    started = time.time()
    device = torch.device("cuda")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    identity = config["identity_model"]
    train_config = {"model": {"width": int(identity["width"])},
                    "training": {"learning_rate": float(identity["learning_rate"]),
                                 "weight_decay": float(identity["weight_decay"]),
                                 "epochs": int(identity["epochs"]),
                                 "batch_size": int(identity["batch_size"]),
                                 "pgd_steps": int(identity["pgd_steps"]),
                                 "targeted_mix": float(identity["targeted_mix"])}}
    e36 = load_e36_config()
    metadata = train_metadata(e36)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    raw = load_all_raw(metadata, e36, device)
    zband = torch.as_tensor(np.load(ROOT / "BrainPrint_cache" / "e36_pure_eeg" /
                                    "train_detrend_zband_cov5.npy"),
                            dtype=torch.float32, device=device)
    fit_np = np.flatnonzero(trial <= 0.8 * maximum)
    cal_np = np.flatnonzero(trial > 0.8 * maximum)
    classes = np.unique(subjects)
    if not np.array_equal(classes, np.arange(1, 21)):
        raise ValueError("最终模型必须是固定 20 类")
    class_to_index = {int(v): i for i, v in enumerate(classes)}
    fit_index = torch.as_tensor(fit_np, device=device)
    fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                                 dtype=torch.long, device=device)
    print(f"训练 CNN：fit={len(fit_np)} cal={len(cal_np)}", flush=True)
    cnn = train_fold(raw, fit_index, fit_labels, len(classes), train_config,
                     int(identity["seed_base"]), device)
    proto = fit_prototype(zband, fit_np, subjects, classes,
                          float(e36["models"]["wccn_shrinkage"]),
                          float(e36["models"]["wccn_eigenvalue_floor"]))
    logspd = FrozenPrototype(proto, e36)

    cal_raw = raw[torch.as_tensor(cal_np, device=device)]
    cal_dr = confidence_values(logspd_scores(logspd, cal_raw))["distance_ratio"]
    cal_acc = float(np.mean(classes[logspd_scores(logspd, cal_raw).argmax(1)]
                            == subjects[cal_np]))
    cal_cnn_acc = float(np.mean(classes[cnn_logits_tta(cnn, cal_raw).argmax(1)]
                                == subjects[cal_np]))
    decision = config["decision"]
    tau_low = float(np.quantile(cal_dr, float(decision["tau_low_q"]), method="higher"))
    tau_high = float(np.quantile(cal_dr, float(decision["tau_high_q"]), method="higher"))
    print(f"校准块：logSPD direct={cal_acc:.4f} CNN(TTA5) direct={cal_cnn_acc:.4f} "
          f"tau_low={tau_low:.4f} tau_high={tau_high:.4f}", flush=True)

    metadata_test = pd.read_csv(TEST_METADATA_PATH)
    if len(metadata_test) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Test 预测数必须为 {EXPECTED_TEST_ROWS}")
    sample_col = column(metadata_test, "Sample")
    path_col = column(metadata_test, "path")
    if metadata_test[sample_col].isna().any() or not metadata_test[sample_col].is_unique:
        raise ValueError("Sample 缺失或不唯一")
    if not metadata_test[path_col].astype(str).str.startswith("Test/").all():
        raise ValueError("Test 元数据含非 Test 路径")

    test_raw = load_test_raw(metadata_test, e36, device)
    test_dr = confidence_values(logspd_scores(logspd, test_raw))["distance_ratio"]
    test_cnn = cnn_logits_tta(cnn, test_raw)
    cnn_pred = classes[test_cnn.argmax(1)]
    logspd_pred = classes[logspd_scores(logspd, test_raw).argmax(1)]
    pred = np.where(test_dr >= tau_high, logspd_pred, cnn_pred)
    labels = np.where(test_dr < tau_low, 0, pred).astype(int)
    if not np.isin(labels, np.arange(21)).all():
        raise RuntimeError("预测标签超出 0–20")

    submission = pd.DataFrame({"Sample": metadata_test[sample_col].to_numpy(), "Label": labels})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)

    audit = {
        "frozen": True,
        "n_rows": int(len(submission)),
        "rule": "dr>=tau_high -> logSPD 身份；tau_low<=dr<tau_high -> CNN(TTA5) 身份；dr<tau_low -> 0",
        "calibration": {"block": "trial > 0.8*max（未参与训练）",
                        "logspd_direct": cal_acc, "cnn_tta_direct": cal_cnn_acc,
                        "tau_low": tau_low, "tau_high": tau_high,
                        "tau_low_q": decision["tau_low_q"],
                        "tau_high_q": decision["tau_high_q"]},
        "regime_counts": {"logspd": int(np.sum(test_dr >= tau_high)),
                          "cnn": int(np.sum((test_dr >= tau_low) & (test_dr < tau_high))),
                          "reject": int(np.sum(test_dr < tau_low))},
        "test_metrics": {key: "N/A（无真实标签）" for key in
                         ("AUC", "F1", "C", "RA", "ASafeR", "U")},
        "session_audit_after_predictions": session_audit(metadata_test, labels),
        "test_labels_used": False,
    }
    RESULTS_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(f"完成，用时 {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
