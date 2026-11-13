"""E49：三段式决策的完整嵌套评估。

  distance_ratio >= tau_high -> logSPD 身份（clean 区域，logSPD 更准）
  tau_low <= dr < tau_high   -> CNN 身份（受扰动区域，CNN 更鲁棒）
  distance_ratio < tau_low   -> 拒识（未知身份）

tau_low = clean known dr 的 q0.02，tau_high = q0.05，全部只用 clean known。
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
from sklearn.metrics import f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))
sys.path.insert(0, str(OUT.parent / "E45_raw_cnn_at"))

from run_e36 import ZBAND_PATH, load_config as load_e36_config, train_metadata  # noqa: E402
from run_attacks import ATTACK_ORDER, load_all_raw, raw_attack, stratified_subset  # noqa: E402
from run_e40 import FrozenPrototype, confidence_values, fit_prototype, official_score  # noqa: E402
from run_e45 import Wrapper, train_fold  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
COMBOS = [(0.01, 0.05), (0.02, 0.05), (0.02, 0.20)]
PRIMARY = (0.02, 0.05)
REFERENCE = {"E36C": {"C": 0.9395, "RA": 0.0609, "ASafeR": 0.6818, "U": 1.0, "Score": 59.78},
             "E47": {"C": 0.9148, "RA": 0.5772, "ASafeR": 0.9076, "U": 0.9965, "Score": 84.08}}


def decide(cnn_scores, logspd_scores, classes, tau_low, tau_high):
    dr = confidence_values(logspd_scores)["distance_ratio"]
    cnn_pred = classes[np.asarray(cnn_scores).argmax(1)]
    logspd_pred = classes[np.asarray(logspd_scores).argmax(1)]
    pred = np.where(dr >= tau_high, logspd_pred, cnn_pred)
    pred = np.where(dr < tau_low, 0, pred)
    return pred


def fold_metrics(runs: list[dict], attack: str | None, low_q: float, high_q: float) -> dict:
    rows = []
    for run in runs:
        classes = np.asarray(run["classes"], dtype=int)
        tau_low = float(np.quantile(run["dr_known"], low_q, method="higher"))
        tau_high = float(np.quantile(run["dr_known"], high_q, method="higher"))
        pred_clean = decide(run["cnn_known"], run["logspd_known"], classes, tau_low, tau_high)
        pred_unknown = decide(run["cnn_unknown"], run["logspd_unknown"], classes,
                              tau_low, tau_high)
        if attack is None:
            pred = pred_clean
            ra = float(np.mean(pred == run["known_true"]))
            asafer = 1.0
        else:
            item = run["attacks"][attack]
            pred = decide(item["cnn"], item["logspd"], classes, tau_low, tau_high)
            ra = float(np.mean(pred == run["known_true"]))
            asafer = float(np.mean((pred == run["known_true"]) | (pred == 0)))
        c = float(np.mean(pred_clean == run["known_true"]))
        u = float(np.mean(pred_unknown == 0))
        direct = np.where(run["dr_known"] >= tau_high,
                          classes[run["logspd_known"].argmax(1)],
                          classes[run["cnn_known"].argmax(1)])
        rows.append({
            "C": c, "RA": ra, "ASafeR": asafer, "A": 0.7 * ra + 0.3 * asafer, "U": u,
            "Rej": float(np.mean(pred == 0)),
            "identity_macro_f1": float(f1_score(run["known_true"], pred, labels=classes,
                                                average="macro", zero_division=0)),
            "identity_macro_f1_direct": float(f1_score(run["known_true"], direct,
                                                       labels=classes, average="macro",
                                                       zero_division=0)),
        })
    macro = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
    return macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "inner"), default="inner")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("E49 要求 CUDA GPU")
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
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    raw = load_all_raw(metadata, e36, device)
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(e36["nested_protocol"]["outer_seed"]).permutation(all_subjects), 5)
    if args.stage == "smoke":
        outer_groups = outer_groups[:1]
    results = {"status": "running", "experiment": "E49", "stage": args.stage,
               "primary": {"low_q": PRIMARY[0], "high_q": PRIMARY[1]}, "outer_folds": []}

    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        inner_order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
        inner_splits = list(np.array_split(inner_order, 2))
        if args.stage == "smoke":
            inner_splits = inner_splits[:1]
        runs = []
        for inner_index, inner_unknown in enumerate(inner_splits):
            inner_fold = inner_index + 1
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask = np.isin(subjects, enrolled)
            fit_np = np.flatnonzero(mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(mask & inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            class_to_index = {int(v): i for i, v in enumerate(enrolled)}
            fit_index = torch.as_tensor(fit_np, device=device)
            fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                                         dtype=torch.long, device=device)
            seed = int(identity["seed_base"] + 1000 * outer_fold + inner_fold)
            print(f"outer={outer_fold} inner={inner_fold} 训练", flush=True)
            cnn = train_fold(raw, fit_index, fit_labels, len(enrolled), train_config, seed, device)
            proto = fit_prototype(zband, fit_np, subjects, enrolled,
                                  float(e36["models"]["wccn_shrinkage"]),
                                  float(e36["models"]["wccn_eigenvalue_floor"]))
            logspd = FrozenPrototype(proto, e36)
            known_subset = stratified_subset(metadata, known_np, 8)
            unknown_subset = stratified_subset(metadata, unknown_np, 8)
            known_raw = raw[torch.as_tensor(known_subset, device=device)]
            unknown_raw = raw[torch.as_tensor(unknown_subset, device=device)]
            with torch.no_grad():
                cnn_known = cnn(known_raw).cpu().numpy()
                cnn_unknown = cnn(unknown_raw).cpu().numpy()
            logspd_known = logspd.scores(known_raw)
            logspd_unknown = logspd.scores(unknown_raw)
            run = {
                "outer_fold": outer_fold, "inner_fold": inner_fold, "classes": enrolled,
                "known_true": subjects[known_subset], "cnn_known": cnn_known,
                "cnn_unknown": cnn_unknown,
                "dr_known": confidence_values(logspd_known)["distance_ratio"],
                "logspd_known": logspd_known, "logspd_unknown": logspd_unknown,
                "attacks": {}, "n_known": int(len(known_subset)),
                "n_unknown": int(len(unknown_subset)),
            }
            wrapper = Wrapper(cnn, enrolled)
            true = subjects[known_subset]
            target = np.asarray([
                enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)]
                for v in true
            ], dtype=int)
            for attack_name in ATTACK_ORDER:
                base_name, snr_text = attack_name.rsplit("_snr", 1)
                snr = int(snr_text)
                attack_seed = int(identity["seed_base"] + 100000 + 10000 * outer_fold
                                  + 1000 * inner_fold + 100 * ATTACK_CODES[base_name] + snr)
                cnn_scores, diagnostics, waves = raw_attack(
                    wrapper, known_raw, true, snr, base_name, attack_seed,
                    {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 1.0,
                                 "seed_base": identity["seed_base"]}},
                    target=target if base_name == "targeted_pgd" else None,
                    return_waveforms=True,
                )
                run["attacks"][attack_name] = {
                    "cnn": np.asarray(cnn_scores),
                    "logspd": logspd.scores(waves.float()),
                    "diagnostics": diagnostics,
                }
            runs.append(run)
            print(f"outer={outer_fold} inner={inner_fold} 完成 "
                  f"cnn_clean={np.mean(enrolled[cnn_known.argmax(1)] == run['known_true']):.4f} "
                  f"logspd_clean={np.mean(enrolled[logspd_known.argmax(1)] == run['known_true']):.4f}",
                  flush=True)
            del cnn, wrapper, logspd
            torch.cuda.empty_cache()

        fold = {"outer_fold": outer_fold, "sensitivity": {}}
        for low_q, high_q in COMBOS:
            per_attack = {name: fold_metrics(runs, name, low_q, high_q)
                          for name in ATTACK_ORDER}
            macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
                     for key in per_attack[ATTACK_ORDER[0]]}
            macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
            clean = fold_metrics(runs, None, low_q, high_q)
            fold["sensitivity"][f"low{low_q}_high{high_q}"] = {
                "per_attack": per_attack, "macro": macro, "clean": clean}
        key = f"low{PRIMARY[0]}_high{PRIMARY[1]}"
        fold["per_attack"] = fold["sensitivity"][key]["per_attack"]
        fold["macro"] = fold["sensitivity"][key]["macro"]
        fold["clean"] = fold["sensitivity"][key]["clean"]
        results["outer_folds"].append(fold)
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        macro = fold["macro"]
        print(f"E49 outer={outer_fold}: Score={macro['Score']:.2f} C={macro['C']:.4f} "
              f"RA={macro['RA']:.4f} ASafeR={macro['ASafeR']:.4f} A={macro['A']:.4f} "
              f"U={macro['U']:.4f} F1={macro['identity_macro_f1']:.4f}", flush=True)

    summary = {key: float(np.mean([f["macro"][key] for f in results["outer_folds"]]))
               for key in results["outer_folds"][0]["macro"]}
    clean_summary = {key: float(np.mean([f["clean"][key] for f in results["outer_folds"]]))
                     for key in results["outer_folds"][0]["clean"]}
    sensitivity = {}
    for name in results["outer_folds"][0]["sensitivity"]:
        sensitivity[name] = {
            key: float(np.mean([f["sensitivity"][name]["macro"][key]
                                for f in results["outer_folds"]]))
            for key in results["outer_folds"][0]["sensitivity"][name]["macro"]
        }
    results["macro_fold_mean"] = summary
    results["clean_fold_mean"] = clean_summary
    results["sensitivity_fold_mean"] = sensitivity
    results["reference"] = REFERENCE
    results["delta_vs"] = {
        name: {key: summary[key] - value[key] for key in ("C", "RA", "ASafeR", "U", "Score")}
        for name, value in REFERENCE.items()
    }
    results["status"] = "complete"
    results["elapsed_seconds"] = time.time() - started
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"E49 MACRO: Score={summary['Score']:.2f} C={summary['C']:.4f} RA={summary['RA']:.4f} "
          f"ASafeR={summary['ASafeR']:.4f} A={summary['A']:.4f} U={summary['U']:.4f} "
          f"F1={summary['identity_macro_f1']:.4f}", flush=True)
    print(f"完成：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
