"""E92: frozen E91 quantile-fusion candidate for the 18-channel project Test."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
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
MANIFEST = OUT / "frozen_manifest.json"
PRED = OUT / "predictions.csv"
RESULTS = OUT / "results.json"
REPORT = OUT / "report.md"
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


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")


def time_frac(meta: pd.DataFrame) -> np.ndarray:
    trial = meta.trial.to_numpy(float)
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return trial / np.maximum(maximum, 1.0)


def score(proto: np.ndarray, x: np.ndarray | torch.Tensor, device: torch.device, cosine: bool) -> np.ndarray:
    p = torch.as_tensor(proto, dtype=torch.float32, device=device)
    rows = []
    for start in range(0, len(x), 512):
        q = torch.as_tensor(np.asarray(x[start:start + 512], dtype=np.float32), device=device)
        if cosine:
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            pp = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            rows.append((q @ pp.T).cpu().numpy())
        else:
            rows.append((-torch.cdist(q, p)).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def scale(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    return np.clip((np.asarray(values, float) - bounds[0]) / max(bounds[1], 1e-8), 0, 1)


def build_frozen_train() -> dict:
    meta = pd.read_csv(TRAIN_META)
    x = np.load(TRAIN_FEATURES, mmap_mode="r")
    if len(meta) != 11304 or x.shape != (len(meta), 855) or not np.isfinite(x).all():
        raise ValueError("Train metadata/feature invariant failed")
    if set(meta.subject.astype(int)) != set(range(1, 21)) or set(meta.session) != {"session1"}:
        raise ValueError("Train subject/session invariant failed")
    subjects = meta.subject.to_numpy(int)
    fit = np.flatnonzero(time_frac(meta) <= 0.60)
    classes = np.arange(1, 21, dtype=int)
    identity = run_e90.fit_templates(x, meta, fit, classes, True, bins=3)
    spd = np.stack([np.asarray(x[fit][subjects[fit] == c], dtype=np.float32).mean(0) for c in classes])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("E92 requires CUDA")
    ck = score(identity["vectors"], x[fit], device, True).max(1)
    sk = score(spd, x[fit], device, False).max(1)
    # E91 quantile_fusion is the preregistered strategy; its weight is irrelevant to min fusion.
    params = {"cos_bounds": [float(np.quantile(ck, .01)), float(np.quantile(ck, .99) - np.quantile(ck, .01))],
              "spd_bounds": [float(np.quantile(sk, .01)), float(np.quantile(sk, .99) - np.quantile(sk, .01))],
              "mode": "quantile", "weight": 1.0, "threshold": 0.7971239261751238}
    artifact = {"classes": classes, "identity_templates": identity, "spd_prototypes": spd, "params": params,
                "fit_rule": "all subjects/task/time bins, within-task time fraction <=0.60", "feature_dim": 855}
    torch.save(artifact, MODEL)
    source = {"e92_script": sha256(Path(__file__)), "e91_script": sha256(E91_SCRIPT),
              "e90_script": sha256(E90_SCRIPT), "e65_script": sha256(E65_SCRIPT),
              "e91_results": sha256(E91_RESULTS), "train_metadata": sha256(TRAIN_META),
              "train_features": sha256(TRAIN_FEATURES)}
    manifest = {"frozen": True, "experiment": "E92 frozen project candidate", "strategy": "E91 quantile_fusion",
                "classes": classes.tolist(), "params": params, "model_sha256": sha256(MODEL),
                "source_sha256": source, "test_session_used_for_freeze": False,
                "test_index_or_waveform_used_for_freeze": False, "post_audit_changes_forbidden": True,
                "test_metrics": {k: "N/A" for k in ("AUC", "F1", "C", "RA", "ASafeR", "U", "Official Score")}}
    dump(MANIFEST, manifest)
    return manifest


def run(smoke: bool) -> dict:
    if MODEL.exists() or MANIFEST.exists() or PRED.exists():
        raise RuntimeError("E92 artifacts already exist; refuse rerun or variant change")
    manifest = build_frozen_train()
    result = {"status": "frozen_smoke" if smoke else "frozen_test_pending", "experiment": "E92 frozen project candidate",
              "strategy": "E91 quantile_fusion", "freeze_manifest": str(MANIFEST), "test_accessed": False,
              "metrics": manifest["test_metrics"], "protocol_checks": {"train_only_freeze": True,
              "test_accessed_before_freeze": False, "session_used_for_inference": False,
              "sample_order_or_count_used_for_inference": False, "near_duplicate_pairing": False}}
    if smoke:
        dump(RESULTS, result)
        REPORT.write_text("# E92 frozen project candidate\n\nSmoke 完成：Train-only 冻结清单、模型模板与源码指纹已生成；未读取项目 Test。\n\nTest AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n", encoding="utf-8")
        return result
    # This is the first Test access, after model, params, source hashes and manifest are frozen.
    for name, path in {"e92_script": Path(__file__), "e91_script": E91_SCRIPT, "e90_script": E90_SCRIPT,
                       "e65_script": E65_SCRIPT, "e91_results": E91_RESULTS, "train_metadata": TRAIN_META,
                       "train_features": TRAIN_FEATURES}.items():
        if sha256(path) != manifest["source_sha256"][name]: raise ValueError(f"frozen source changed: {name}")
    order = pd.read_csv(TEST_CSV, usecols=["Sample"])
    paths = pd.read_csv(TEST_META, usecols=["sample", "path"])
    if len(order) != 22251 or len(paths) != 22251 or not np.array_equal(order.Sample.astype(str), paths.sample.astype(str)):
        raise ValueError("Test Sample/path mapping invariant failed")
    if not paths.path.astype(str).str.startswith("Test/").all(): raise ValueError("non-Test path")
    device = torch.device("cuda")
    artifact = torch.load(MODEL, map_location="cpu", weights_only=False)
    identity, spd, params = artifact["identity_templates"], artifact["spd_prototypes"], artifact["params"]
    labels = np.empty(len(paths), dtype=np.int64)
    fused_scores = np.empty(len(paths), dtype=np.float32)
    e36 = load_e36_config(); e65cfg = json.loads((ROOT / "research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    test_root = ROOT / "Brainprint_Competition"
    for start in range(0, len(paths), 64):
        stop = min(start + 64, len(paths))
        raw = torch.as_tensor(np.stack([load_train_matrix(test_root / p) for p in paths.path.iloc[start:stop]]), dtype=torch.float32, device=device)
        with torch.no_grad(): feat = run_e65.gpu_correlation_logspd(raw, e36, e65cfg)
        c = score(identity["vectors"], feat, device, True); s = score(spd, feat, device, False)
        fused = np.minimum(scale(c.max(1), params["cos_bounds"]), scale(s.max(1), params["spd_bounds"]))
        labels[start:stop] = np.where(fused >= params["threshold"], artifact["classes"][c.argmax(1)], 0)
        fused_scores[start:stop] = fused
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(PRED, index=False)
    sessions = pd.read_csv(TEST_META, usecols=["sample", "session"])
    codes = sessions.session.astype(str).str.strip().str.casefold().replace({"1":"S1","s1":"S1","session1":"S1","2":"S2","s2":"S2","session2":"S2"}).to_numpy()
    audit = {"N": len(labels), "overall_rejection_rate": float(np.mean(labels == 0)), "session_used_by_inference": False,
             "audit_after_predictions_and_csv": True, "S1_rejection_target": .95, "S2_acceptance_target": .98}
    for code in ("S1", "S2"):
        mask = codes == code; rej = float(np.mean(labels[mask] == 0)); audit[f"{code}_N"] = int(mask.sum()); audit[f"{code}_rejection_rate"] = rej; audit[f"{code}_acceptance_rate"] = 1 - rej
    audit["S1_rejection_ge_0.95"] = audit["S1_rejection_rate"] >= .95; audit["S2_acceptance_ge_0.98"] = audit["S2_acceptance_rate"] >= .98
    result.update({"status": "complete_one_shot_frozen_engineering_audit", "test_accessed": True,
                   "predictions_csv": str(PRED), "prediction_rows": len(labels), "metrics": manifest["test_metrics"],
                   "offline_session_audit_after_prediction": audit, "post_audit_threshold_or_variant_change": False})
    dump(RESULTS, result)
    REPORT.write_text("# E92 frozen project candidate\n\n## 结论\n\nE91 Train-only quantile fusion 已冻结后执行一次项目 Test 推理。Test 无真实标签，AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n| 项目 | 结果 |\n|---|---:|\n| 总拒识率 | %.6f |\n| S1 拒识率 | %.6f（目标≥0.95：%s） |\n| S2 接受率 | %.6f（目标≥0.98：%s） |\n\nSession 只在预测 CSV 完整落盘后读取；未用于推理、调参、阈值扫描、样本配对或近重复选择。\n" % (audit["overall_rejection_rate"], audit["S1_rejection_rate"], "通过" if audit["S1_rejection_ge_0.95"] else "未通过", audit["S2_acceptance_rate"], "通过" if audit["S2_acceptance_ge_0.98"] else "未通过"), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true"); a = ap.parse_args()
    if a.smoke == a.full: ap.error("choose --smoke or --full")
    print(json.dumps(run(a.smoke), ensure_ascii=False, indent=2))
