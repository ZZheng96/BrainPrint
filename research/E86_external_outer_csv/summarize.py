"""Aggregate and compare outer-test prediction CSVs across external experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

E86_DIR = Path(__file__).resolve().parent
ROOT = E86_DIR.parents[1]
sys.path.insert(0, str(E86_DIR))
import common  # noqa: E402

OUTPUTS = E86_DIR / "outputs"
EXPERIMENTS = [
    ("E73", ROOT / "research" / "E73_external_multisession" / "outputs"),
    ("E74", ROOT / "research" / "E74_multiscale_native22" / "outputs"),
    ("E75", ROOT / "research" / "E75_image_template" / "outputs"),
    ("E76", ROOT / "research" / "E76_open_set_head" / "outputs"),
    ("E77", ROOT / "research" / "E77_score_fusion" / "outputs"),
    ("E78", ROOT / "research" / "E78_identity_calibration" / "outputs"),
    ("E79", ROOT / "research" / "E79_pair_open_set" / "outputs"),
    ("E80", ROOT / "research" / "E80_robust_open_set" / "outputs"),
    ("E81", ROOT / "research" / "E81_evt_open_set" / "outputs"),
    ("E82", ROOT / "research" / "E82_unknown_calibration" / "outputs"),
    ("E84", ROOT / "research" / "E84_hard_negative_pair" / "outputs"),
]
PRIMARY = {
    "E73": "native22_pair", "E74": None, "E75": None, "E76": None, "E77": None,
    "E78": None, "E79": "pair_q05", "E80": None, "E81": None, "E82": "fpir02",
    "E84": "pair_q05",
}
METRICS = ("AUROC", "identity_macro_F1", "unknown_F1", "known_acceptance",
           "unknown_false_acceptance", "unknown_rejection", "C", "U_at_95",
           "DIR_at_FPIR_1", "detection_EER")


def load_experiment(name: str, out_dir: Path) -> dict | None:
    metrics_path = out_dir / "outer_test_metrics.json"
    csv_path = out_dir / "outer_test_predictions.csv"
    if not metrics_path.is_file() or not csv_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(csv_path)
    common.validate_csv_frame(frame)
    methods = list(frame["method"].unique())
    chosen = PRIMARY.get(name) or payload.get("method")
    if chosen not in methods:
        chosen = methods[0]
    selected = frame[frame["method"] == chosen]
    return {
        "experiment": name,
        "method": chosen,
        "methods": methods,
        "csv": str(csv_path.resolve()),
        "csv_sha256": common.sha256(csv_path),
        "n_rows": int(len(selected)),
        "n_groups": int(selected.groupby(["outer_fold", "direction"]).ngroups),
        "max_abs_diff": float(payload.get("reference_check", {}).get("max_abs_diff", 0.0)),
        "group_mean": common.metrics_from_frame(selected) if selected.groupby(["outer_fold", "direction"]).ngroups == 1 else {
            key: float(pd.Series([g[key] for g in common.metrics_by_group(selected).values()]).mean())
            for key in METRICS
        },
        "frame": selected,
    }


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    loaded, missing = {}, []
    for name, out_dir in EXPERIMENTS:
        item = load_experiment(name, out_dir)
        if item is None:
            missing.append(name)
        else:
            loaded[name] = item
    summary = {name: {key: value for key, value in item.items() if key != "frame"}
               for name, item in loaded.items()}
    common.write_json(OUTPUTS / "all_experiments_metrics.json",
                      {"experiments": summary, "missing_csv": missing,
                       "note": "所有指标均由 outer_test_predictions.csv 重算；外部 BCIC IV-2a outer test，非项目 Test"})
    pairs = []
    names = [name for name, _ in EXPERIMENTS if name in loaded]
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = loaded[left], loaded[right]
            try:
                comparison = common.compare_frames(a["frame"], b["frame"])
            except ValueError as error:
                pairs.append({"a": left, "b": right, "skipped": str(error)})
                continue
            pairs.append({"a": left, "b": right, "a_method": a["method"], "b_method": b["method"], **comparison})
    common.write_json(OUTPUTS / "cross_experiment_comparison.json", {"pairs": pairs})
    print(json.dumps({"loaded": list(loaded), "missing": missing,
                      "n_pairs": len(pairs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
