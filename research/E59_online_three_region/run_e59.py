"""E59：E58 三段式 + E55 在线 PGD7 对抗训练。

与 E58 的唯一区别：身份模型训练时每个 mini-batch 都重新生成原始波形 PGD7
（E55 的 true_online 模式），而不是复用缓存。其余（OE、ARPL、三段式决策、
阈值来源、评估协议）完全不变。
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
sys.path.insert(0, str(OUT.parent / "E54_outlier_exposure"))

from run_e36 import ZBAND_PATH, load_config as load_e36_config, train_metadata  # noqa: E402
from run_e40 import FrozenPrototype, confidence_values, fit_prototype, official_score  # noqa: E402
import run_e54 as E54  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
ATTACK_ORDER = ["untargeted_fgsm_snr30", "untargeted_fgsm_snr20",
                "untargeted_pgd_snr30", "untargeted_pgd_snr20",
                "targeted_pgd_snr30", "targeted_pgd_snr20"]
REFERENCE = {"E58": {"C": 0.9714, "RA": 0.9371, "ASafeR": 0.9758, "U": 0.9875, "Score": 96.74}}


def decide(identity_scores, logspd_scores, classes, tau_low, tau_high):
    dr = confidence_values(logspd_scores)["distance_ratio"]
    identity_pred = classes[np.asarray(identity_scores).argmax(1)]
    logspd_pred = classes[np.asarray(logspd_scores).argmax(1)]
    pred = np.where(dr >= tau_high, logspd_pred, identity_pred)
    return np.where(dr < tau_low, 0, pred)


def fold_metrics(runs: list[dict], attack: str | None, low_q: float, high_q: float) -> dict:
    rows = []
    for run in runs:
        classes = np.asarray(run["classes"], dtype=int)
        tau_low = float(np.quantile(run["dr_known"], low_q, method="higher"))
        tau_high = float(np.quantile(run["dr_known"], high_q, method="higher"))
        pred_clean = decide(run["id_known"], run["logspd_known"], classes, tau_low, tau_high)
        pred_unknown = decide(run["id_unknown"], run["logspd_unknown"], classes,
                              tau_low, tau_high)
        if attack is None:
            pred = pred_clean
            ra = float(np.mean(pred == run["known_true"]))
            asafer = 1.0
        else:
            item = run["attacks"][attack]
            pred = decide(item["id"], item["logspd"], classes, tau_low, tau_high)
            ra = float(np.mean(pred == run["known_true"]))
            asafer = float(np.mean((pred == run["known_true"]) | (pred == 0)))
        c = float(np.mean(pred_clean == run["known_true"]))
        u = float(np.mean(pred_unknown == 0))
        direct = np.where(run["dr_known"] >= tau_high,
                          classes[run["logspd_known"].argmax(1)],
                          classes[run["id_known"].argmax(1)])
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
        raise SystemExit("E59 要求 CUDA GPU")
    started = time.time()
    device = torch.device("cuda")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    e54_config = json.loads((OUT.parent / "E54_outlier_exposure" / "config.json")
                            .read_text(encoding="utf-8"))
    override = config["training_override"]
    e54_config["training"]["attack_training_mode"] = override["attack_training_mode"]
    e54_config["training"]["online_pgd_steps"] = int(override["online_pgd_steps"])
    e54_config["training"]["main_targeted_fraction"] = float(override["main_targeted_fraction"])
    e36 = load_e36_config()
    metadata = train_metadata(e36)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    raw = E54.load_all_raw(metadata, e54_config, device)
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(e36["nested_protocol"]["outer_seed"]).permutation(all_subjects), 5)
    if args.stage == "smoke":
        outer_groups = outer_groups[:1]
    results = {"status": "running", "experiment": "E59", "stage": args.stage,
               "identity_model": "E54 oe + E55 online PGD7", "outer_folds": []}

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
            roles = E54.build_inner_roles(known_subjects, inner_unknown, unknown_subjects,
                                          outer_fold, inner_fold, e54_config)
            enrolled = roles["enrolled"]
            mask = np.isin(subjects, enrolled)
            fit_np = np.flatnonzero(mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(mask & inner_block)
            oe_np = np.flatnonzero(np.isin(subjects, roles["oe_only"]) & ~forward & ~inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, roles["calibration"]) & ~forward)
            print(f"outer={outer_fold} inner={inner_fold} 训练 OE-EEGNet(online PGD7) "
                  f"fit={len(fit_np)} oe={len(oe_np)}", flush=True)
            model, info = E54.train_model(
                "oe", raw, subjects, fit_np, oe_np, known_np, unknown_np, enrolled,
                metadata, e54_config, outer_fold, inner_fold,
            )
            proto = fit_prototype(zband, fit_np, subjects, enrolled,
                                  float(e36["models"]["wccn_shrinkage"]),
                                  float(e36["models"]["wccn_eigenvalue_floor"]))
            logspd = FrozenPrototype(proto, e36)
            known_subset = E54.stratified_subset(metadata, known_np, 8)
            unknown_subset = E54.stratified_subset(metadata, unknown_np, 8)
            known_raw = raw[torch.as_tensor(known_subset, device=device)]
            unknown_raw = raw[torch.as_tensor(unknown_subset, device=device)]
            with torch.no_grad():
                id_known = model.scores(known_raw).cpu().numpy()
                id_unknown = model.scores(unknown_raw).cpu().numpy()
            logspd_known = logspd.scores(known_raw)
            logspd_unknown = logspd.scores(unknown_raw)
            run = {
                "outer_fold": outer_fold, "inner_fold": inner_fold, "classes": enrolled,
                "known_true": subjects[known_subset], "id_known": id_known,
                "id_unknown": id_unknown, "logspd_known": logspd_known,
                "logspd_unknown": logspd_unknown,
                "dr_known": confidence_values(logspd_known)["distance_ratio"],
                "attacks": {}, "n_known": int(len(known_subset)),
                "n_unknown": int(len(unknown_subset)),
                "fit_accuracy": info["fit_accuracy"], "best_epoch": info["best_epoch"],
            }
            true = subjects[known_subset]
            target = np.asarray([
                enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)]
                for v in true
            ], dtype=int)
            for attack_name in ATTACK_ORDER:
                base_name, snr_text = attack_name.rsplit("_snr", 1)
                snr = int(snr_text)
                seed = int(e54_config["seed_base"] + 100000 + 10000 * outer_fold
                           + 1000 * inner_fold + 100 * ATTACK_CODES[base_name] + snr)
                scores, diagnostics, waves = E54.raw_attack(
                    model, known_raw, torch.as_tensor(
                        [int(np.flatnonzero(enrolled == v)[0]) for v in true],
                        dtype=torch.long, device=device),
                    snr, base_name, 20, seed,
                    target=torch.as_tensor(
                        [int(np.flatnonzero(enrolled == v)[0]) for v in target],
                        dtype=torch.long, device=device) if base_name == "targeted_pgd" else None,
                    return_waveforms=True,
                )
                run["attacks"][attack_name] = {
                    "id": scores.detach().cpu().numpy(),
                    "logspd": logspd.scores(waves.float()),
                    "diagnostics": diagnostics,
                }
            runs.append(run)
            print(f"outer={outer_fold} inner={inner_fold} 完成 fit_acc={info['fit_accuracy']:.4f} "
                  f"id_clean={np.mean(enrolled[id_known.argmax(1)] == run['known_true']):.4f}",
                  flush=True)
            del model, logspd
            torch.cuda.empty_cache()

        per_attack = {name: fold_metrics(runs, name, 0.02, 0.05) for name in ATTACK_ORDER}
        macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
                 for key in per_attack[ATTACK_ORDER[0]]}
        macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
        clean = fold_metrics(runs, None, 0.02, 0.05)
        results["outer_folds"].append({"outer_fold": outer_fold, "per_attack": per_attack,
                                       "macro": macro, "clean": clean})
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"E59 outer={outer_fold}: Score={macro['Score']:.2f} C={macro['C']:.4f} "
              f"RA={macro['RA']:.4f} ASafeR={macro['ASafeR']:.4f} A={macro['A']:.4f} "
              f"U={macro['U']:.4f} F1={macro['identity_macro_f1']:.4f}", flush=True)

    summary = {key: float(np.mean([f["macro"][key] for f in results["outer_folds"]]))
               for key in results["outer_folds"][0]["macro"]}
    clean_summary = {key: float(np.mean([f["clean"][key] for f in results["outer_folds"]]))
                     for key in results["outer_folds"][0]["clean"]}
    results["macro_fold_mean"] = summary
    results["clean_fold_mean"] = clean_summary
    results["reference"] = REFERENCE
    results["delta_vs_e58"] = {key: summary[key] - REFERENCE["E58"][key]
                               for key in ("C", "RA", "ASafeR", "U", "Score")}
    results["status"] = "complete"
    results["elapsed_seconds"] = time.time() - started
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"E59 MACRO: Score={summary['Score']:.2f} C={summary['C']:.4f} RA={summary['RA']:.4f} "
          f"ASafeR={summary['ASafeR']:.4f} A={summary['A']:.4f} U={summary['U']:.4f} "
          f"F1={summary['identity_macro_f1']:.4f}", flush=True)
    print(f"完成：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
