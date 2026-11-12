"""E42：用同一次 Test 推理产出两份候选 CSV（E36C 规则 / 双门控规则）。

- 身份模型、特征、ECDF 参考全部复用 E36E 冻结实现，不重新训练。
- 阈值只在 Train 的校准块（trial ∈ (0.6, 0.8]，20 类全注册）上求解：
  共享低分位 q，使 (u >= t_u) & (max_cosine >= t_mc) 的联合接受率为 0.95。
- Test 只做预测与无标签审计；Session 字段不参与任何决策。
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

from run_e36 import gpu_logspd, detrend_zband, load_train_matrix  # noqa: E402
from run_e36e_calibration import (  # noqa: E402
    TorchWCCNPrototype,
    distance_ratio,
    load_train_metadata,
    predict_scores,
)
from run_e36e_inference import (  # noqa: E402
    EXPECTED_TEST_ROWS,
    TEST_METADATA_PATH,
    column,
    fit_full_train_model,
    load_train_zband,
    read_json,
)

CONFIG_PATH = ROOT / "research" / "E36_pure_eeg" / "config.json"
CANDIDATE_PATH = ROOT / "research" / "E36_pure_eeg" / "frozen_candidate.template.json"
TEST_ROOT = ROOT / "Brainprint_Competition"
RESULTS_PATH = OUT / "results.json"
DEFAULT_A = OUT / "candidate_e36c_rule.csv"
DEFAULT_B = OUT / "candidate_two_gate_rule.csv"


def max_cosine(scores: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(scores, dtype=np.float64), axis=1)[:, -1]


def solve_shared_quantile(u: np.ndarray, mc: np.ndarray) -> dict:
    best = None
    for q in np.linspace(0.001, 0.05, 200):
        t_u = float(np.quantile(u, q, method="higher"))
        t_mc = float(np.quantile(mc, q, method="higher"))
        acceptance = float(np.mean((u >= t_u) & (mc >= t_mc)))
        if best is None or abs(acceptance - 0.95) < abs(best[0] - 0.95):
            best = (acceptance, t_u, t_mc, float(q))
    return {"quantile": best[3], "t_u": best[1], "t_mc": best[2],
            "acceptance": best[0],
            "single_gate_acceptance": float(np.mean(u >= best[1]))}


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
                np.stack(arrays), dtype=torch.float32, device=device
            )
            if (start // batch + 1) % 20 == 0:
                print(f"Test 原始波形 {start + len(arrays)}/{len(paths)} "
                      f"({time.time() - started:.0f}s)", flush=True)
    return output


def batch_features(raw: torch.Tensor, config: dict, batch: int = 256) -> torch.Tensor:
    output = torch.empty((len(raw), 855), dtype=torch.float32, device=raw.device)
    with torch.no_grad():
        for start in range(0, len(raw), batch):
            chunk = raw[start:start + batch]
            output[start:start + len(chunk)] = detrend_zband(gpu_logspd(chunk, config), config)
    return output


def build_f20_reference(metadata: pd.DataFrame, zband: torch.Tensor, candidate: dict,
                        config: dict, device: torch.device) -> tuple[np.ndarray, str]:
    """20 类前 80% 连续块拟合、后 20% 已知 OOF 的 distance_ratio 排序参考。

    不依赖 E36E 的缓存文件；哈希按本次实际内容计算并写入审计。
    """
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    fit_np = np.flatnonzero(trial <= 0.8 * maximum)
    oof_np = np.flatnonzero(trial > 0.8 * maximum)
    model = TorchWCCNPrototype(
        zband[torch.as_tensor(fit_np, device=device)], subjects[fit_np], np.arange(1, 21),
        float(candidate["wccn_shrinkage"]), float(config["models"]["wccn_eigenvalue_floor"]),
    )
    scores = predict_scores(model, zband, oof_np, device,
                            int(config["models"]["prediction_batch_size"]))
    reference = np.sort(distance_ratio(scores)).astype(np.float64)
    digest = __import__("hashlib").sha256(
        np.ascontiguousarray(reference, dtype=np.float64).tobytes()
    ).hexdigest()
    del model
    torch.cuda.empty_cache()
    return reference, digest


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
            float(np.mean(labels[selected] == 0)) if selected.any() else None
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--csv-b", type=Path, default=DEFAULT_B)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("E42 要求 CUDA GPU")
    started = time.time()
    device = torch.device("cuda")
    config = read_json(CONFIG_PATH)
    candidate = read_json(CANDIDATE_PATH)
    metadata, zband = load_train_zband(config, device)
    reference, reference_hash = build_f20_reference(metadata, zband, candidate, config, device)
    model = fit_full_train_model(metadata, zband, candidate, config)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()

    # 校准块只用 Train：(0.6, 0.8] 连续块，与 E36/E39 的内层块定义一致。
    calibration_np = np.flatnonzero((trial > 0.6 * maximum) & (trial <= 0.8 * maximum))
    calibration_scores = predict_scores(
        model, zband, calibration_np, device, int(config["models"]["prediction_batch_size"])
    )
    calibration_ratio = distance_ratio(calibration_scores)
    calibration_u = np.searchsorted(reference, calibration_ratio, side="right") / len(reference)
    solved = solve_shared_quantile(calibration_u, max_cosine(calibration_scores))
    print(f"校准块 N={len(calibration_np)} q={solved['quantile']:.4f} "
          f"t_u={solved['t_u']:.6f} t_mc={solved['t_mc']:.6f} "
          f"联合接受率={solved['acceptance']:.4f} "
          f"单门控接受率={solved['single_gate_acceptance']:.4f}", flush=True)

    metadata_test = pd.read_csv(TEST_METADATA_PATH)
    if len(metadata_test) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Test 预测数必须为 {EXPECTED_TEST_ROWS}")
    sample_col = column(metadata_test, "Sample")
    path_col = column(metadata_test, "path")
    if metadata_test[sample_col].isna().any() or not metadata_test[sample_col].is_unique:
        raise ValueError("Sample 缺失或不唯一")
    if not metadata_test[path_col].astype(str).str.startswith("Test/").all():
        raise ValueError("Test 元数据含非 Test 路径")

    raw = load_test_raw(metadata_test, config, device)
    features = batch_features(raw, config)
    scores = np.empty((len(features), len(model.classes)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(features), 512):
            chunk = features[start:start + 512]
            scores[start:start + len(chunk)] = model(chunk).cpu().numpy()
    ratio = distance_ratio(scores)
    u = np.searchsorted(reference, ratio, side="right") / len(reference)
    mc = max_cosine(scores)
    direct = model.classes[scores.argmax(1)].astype(int)
    label_a = np.where(u < solved["t_u"], 0, direct).astype(int)
    label_b = np.where((u < solved["t_u"]) | (mc < solved["t_mc"]), 0, direct).astype(int)
    if not np.isin(label_a, np.arange(21)).all() or not np.isin(label_b, np.arange(21)).all():
        raise RuntimeError("预测标签超出 0–20")

    for path, labels, rule in ((args.csv_a, label_a, "e36c_single_gate"),
                               (args.csv_b, label_b, "two_gate")):
        frame = pd.DataFrame({"Sample": metadata_test[sample_col].to_numpy(), "Label": labels})
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    audit = {
        "frozen": True,
        "n_rows": int(len(metadata_test)),
        "reference": {"definition": "20-class first-80% fit, last-20% known OOF distance_ratio, sorted",
                      "sha256": reference_hash, "n": int(len(reference))},
        "calibration_source": "Train 20-class continuous block trial in (0.6, 0.8]",
        "calibration": solved,
        "test_metrics": {key: "N/A（无真实标签）" for key in
                         ("AUC", "F1", "C", "RA", "ASafeR", "U")},
        "rule_a": {"definition": "u < t_u -> 0", "path": str(args.csv_a.resolve()),
                   "rejection_rate": float(np.mean(label_a == 0)),
                   "session_audit": session_audit(metadata_test, label_a)},
        "rule_b": {"definition": "u < t_u 或 max_cosine < t_mc -> 0",
                   "path": str(args.csv_b.resolve()),
                   "rejection_rate": float(np.mean(label_b == 0)),
                   "session_audit": session_audit(metadata_test, label_b)},
        "a_b_disagreement_rate": float(np.mean(label_a != label_b)),
        "test_labels_used": False,
    }
    RESULTS_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(f"完成，用时 {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
