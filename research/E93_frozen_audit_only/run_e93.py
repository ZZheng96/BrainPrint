"""E93: one-shot audit of the E92 frozen 18-channel Test project."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E90_project18_identity_template"))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e90
import run_e65
from run_e36 import load_config as load_e36_config, load_train_matrix

TRAIN_META = ROOT / "BrainPrint_cache/train_metadata.csv"
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"
TEST_CSV = ROOT / "Brainprint_Competition/test.csv"
TEST_META = ROOT / "BrainPrint_cache/test_metadata.csv"
MODEL = OUT / "frozen_model.pt"
MANIFEST = OUT / "frozen_manifest_copy.json"
PRED = OUT / "predictions.csv"
RESULTS = OUT / "results.json"
REPORT = OUT / "report.md"
E92_SCRIPT = ROOT / "research/E92_frozen_project_candidate/run_e92.py"
E91_RESULTS = ROOT / "research/E91_hybrid_identity_unknown/results.json"
E91_SCRIPT = ROOT / "research/E91_hybrid_identity_unknown/run_e91.py"
E90_SCRIPT = ROOT / "research/E90_project18_identity_template/run_e90.py"
E65_SCRIPT = ROOT / "research/E65_correlation_spd/run_e65.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score(proto: np.ndarray, x: np.ndarray | torch.Tensor, device: torch.device, cosine: bool) -> np.ndarray:
    p = torch.as_tensor(proto, dtype=torch.float32, device=device)
    rows = []
    for start in range(0, len(x), 512):
        chunk = x[start:start + 512]
        q = torch.as_tensor(chunk, dtype=torch.float32, device=device)
        if cosine:
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            pp = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            rows.append((q @ pp.T).cpu().numpy())
        else:
            rows.append((-torch.cdist(q, p)).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def scale(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    return np.clip((np.asarray(values, float) - bounds[0]) / max(bounds[1], 1e-8), 0, 1)


def validate_frozen() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("frozen") is not True or manifest.get("test_session_used_for_freeze") is not False:
        raise ValueError("frozen manifest invariant failed")
    if manifest.get("test_index_or_waveform_used_for_freeze") is not False:
        raise ValueError("Test was used for freeze")
    expected_sources = {"e92_script": E92_SCRIPT, "e91_script": E91_SCRIPT, "e90_script": E90_SCRIPT,
                        "e65_script": E65_SCRIPT, "e91_results": E91_RESULTS,
                        "train_metadata": TRAIN_META, "train_features": TRAIN_FEATURES}
    for name, path in expected_sources.items():
        if sha256(path) != manifest["source_sha256"][name]:
            raise ValueError(f"frozen source changed: {name}")
    if sha256(MODEL) != manifest["model_sha256"]:
        raise ValueError("frozen model hash mismatch")
    artifact = torch.load(MODEL, map_location="cpu", weights_only=False)
    if artifact["params"] != manifest["params"]:
        raise ValueError("frozen params mismatch")
    if hashlib.sha256(canonical(manifest["params"]).encode()).hexdigest() != hashlib.sha256(canonical(artifact["params"]).encode()).hexdigest():
        raise ValueError("frozen params hash mismatch")
    if artifact.get("feature_dim") != 855:
        raise ValueError("frozen feature dimension mismatch")
    return manifest, artifact


def run() -> dict:
    if PRED.exists() or RESULTS.exists() or REPORT.exists():
        raise RuntimeError("E93 outputs already exist; refuse rerun")
    manifest, artifact = validate_frozen()
    # First Test access occurs only after copied manifest, model, parameters and hashes validate.
    order = pd.read_csv(TEST_CSV, usecols=["Sample"])
    mapping = pd.read_csv(TEST_META, usecols=["sample", "path"])
    if len(order) != 22251 or len(mapping) != 22251 or not order.Sample.astype(str).equals(mapping["sample"].astype(str)):
        raise ValueError("Test Sample/path mapping invariant failed")
    if not bool(mapping.path.astype(str).str.startswith("Test/").all()):
        raise ValueError("non-Test path")
    device = torch.device("cuda")
    identity, spd, params = artifact["identity_templates"], artifact["spd_prototypes"], artifact["params"]
    labels = np.empty(len(mapping), dtype=np.int64)
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    test_root = ROOT / "Brainprint_Competition"
    for start in range(0, len(mapping), 64):
        stop = min(start + 64, len(mapping))
        raw = torch.as_tensor(np.stack([load_train_matrix(test_root / p) for p in mapping.path.iloc[start:stop]]), dtype=torch.float32, device=device)
        with torch.no_grad():
            feat = run_e65.gpu_correlation_logspd(raw, e36, e65cfg)
        c = score(identity["vectors"], feat, device, True)
        s = score(spd, feat, device, False)
        fused = np.minimum(scale(c.max(1), params["cos_bounds"]), scale(s.max(1), params["spd_bounds"]))
        labels[start:stop] = np.where(fused >= params["threshold"], identity["labels"][c.argmax(1)], 0)
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(PRED, index=False)
    written = pd.read_csv(PRED)
    if list(written.columns) != ["Sample", "Label"] or len(written) != 22251 or not written.Sample.astype(str).equals(order.Sample.astype(str)):
        raise ValueError("predictions.csv completeness invariant failed")
    # Session is read only after the complete predictions CSV is on disk.
    audit_meta = pd.read_csv(TEST_META, usecols=["sample", "session"])
    if not audit_meta["sample"].astype(str).equals(written.Sample.astype(str)):
        raise ValueError("post-write session metadata alignment failed")
    codes = audit_meta.session.astype(str).str.strip().str.casefold().map({"1":"S1", "s1":"S1", "session1":"S1", "2":"S2", "s2":"S2", "session2":"S2"})
    if codes.isna().any():
        raise ValueError("unknown Test session")
    rejected = written.Label.to_numpy() == 0
    audit = {"N": 22251, "total_rejection": int(rejected.sum()), "overall_rejection_rate": float(rejected.mean()), "session_used_by_inference": False, "audit_after_predictions_csv": True}
    for code in ("S1", "S2"):
        mask = codes.to_numpy() == code
        rej = float(rejected[mask].mean())
        audit[f"{code}_N"] = int(mask.sum())
        audit[f"{code}_rejection_rate"] = rej
        audit[f"{code}_acceptance_rate"] = 1.0 - rej
    audit["S1_target_pass"] = audit["S1_rejection_rate"] >= .95
    audit["S2_target_pass"] = audit["S2_acceptance_rate"] >= .98
    result = {"status": "complete_one_shot_frozen_audit", "experiment": "E93 frozen audit only", "test_accessed": True,
              "predictions_csv": str(PRED), "prediction_rows": 22251, "metrics": manifest["test_metrics"],
              "frozen_validation": {"frozen": True, "model_hash_valid": True, "source_hash_valid": True, "params_hash_valid": True},
              "audit": audit, "post_audit_changes": False}
    dump(RESULTS, result)
    REPORT.write_text("# E93 frozen audit-only\n\n## 结论\n\n严格复用 E92 已冻结的 18 通道 E65 特征、模型、参数和 Sample/path 映射，完成唯一一次项目 Test 推理；未修改 E92、模型、阈值或策略。\n\nTest 无真实标签，因此 AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n| 项目 | 真实结果 | 目标 | 是否通过 |\n|---|---:|---:|---|\n| 总拒识 | %d / %d（%.6f） | — | — |\n| S1 拒识率 | %.6f | ≥0.95 | %s |\n| S2 接受率 | %.6f | ≥0.98 | %s |\n\nSession 字段仅在完整 predictions.csv 落盘后读取用于审计，未用于推理、调参、阈值或策略改变。\n" % (audit["total_rejection"], audit["N"], audit["overall_rejection_rate"], audit["S1_rejection_rate"], "通过" if audit["S1_target_pass"] else "未通过", audit["S2_acceptance_rate"], "通过" if audit["S2_target_pass"] else "未通过"), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
