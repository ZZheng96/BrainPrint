import json, sys, time
from pathlib import Path
import numpy as np, torch

OUT = Path(r"d:\WORK\BCIC2026\BrainPrint\research\E192_oe_three_region_18ch")
sys.path.insert(0, str(OUT))
import run_e192 as E
from sklearn.metrics import f1_score

base = E.load_config()
dev = torch.device("cuda")
meta = E.train_metadata(base)
P = E.build_protocol(meta)
subjects = P["subjects"]
e36 = E.E36.load_config()
e65 = json.loads((E.ROOT / "research" / "E65_correlation_spd" / "config.json").read_text(encoding="utf-8"))
raw = E.E54.load_all_raw(meta, base, dev)
oe = E.load_external(base, dev)
feat = torch.as_tensor(np.load(E.TRAIN_CORR), dtype=torch.float32, device=dev)


def one(cfg, outer, inner):
    _, enrolled, unknown_subjects = E.role_partition(subjects, outer, inner)
    lut = {int(c): i for i, c in enumerate(enrolled)}
    pm = np.isin(subjects, enrolled)
    fit_np = np.flatnonzero(pm & P["fit_rows"])
    known_np = np.flatnonzero(pm & P["window"])
    unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects) & (P["frac"] <= 0.8))
    fit_raw = raw[torch.as_tensor(fit_np, device=dev)].contiguous()
    fit_labels = torch.as_tensor([lut[int(v)] for v in subjects[fit_np]], dtype=torch.long, device=dev)
    model, info = E.train_identity(fit_raw, fit_labels, oe, len(enrolled), cfg,
                                   int(base["seed_base"] + 1000 * outer + inner), dev,
                                   int(cfg["training"]["epochs"]), True, True, "probe")
    known_subset = E.E54.stratified_subset(meta, known_np, 8)
    unknown_subset = E.E54.stratified_subset(meta, unknown_np, 8)
    known_raw = raw[torch.as_tensor(known_subset, device=dev)].contiguous()
    id_known = E.identity_scores(model, known_raw)
    id_unknown = E.identity_scores(model, raw[torch.as_tensor(unknown_subset, device=dev)])
    wccn = E.WCCN(feat, np.flatnonzero(pm & P["fit_rows"]), subjects, enrolled, 0.05, 1e-6)
    log_known = wccn.scores(feat[torch.as_tensor(known_subset, device=dev)])
    log_unknown = wccn.scores(feat[torch.as_tensor(unknown_subset, device=dev)])
    dr = E.distance_ratio(log_known)
    tau_low = float(np.quantile(dr, 0.02, method="higher"))
    tau_high = float(np.quantile(dr, 0.05, method="higher"))
    true = subjects[known_subset]
    pred, _, _ = E.decide(id_known, log_known, enrolled, tau_low, tau_high)
    upred, _, _ = E.decide(id_unknown, log_unknown, enrolled, tau_low, tau_high)
    labels = torch.as_tensor([lut[int(v)] for v in true], dtype=torch.long, device=dev)
    tgt_sub = np.asarray([enrolled[(lut[int(v)] + 1) % len(enrolled)] for v in true])
    targets = torch.as_tensor([lut[int(v)] for v in tgt_sub], dtype=torch.long, device=dev)
    out = {"clean": float(np.mean(pred == true)), "U": float(np.mean(upred == 0)),
           "fit_acc": info["n_fit"], "attacks": {}}
    for name in E.PRIMARY_ATTACKS:
        base_name, snr_text = name.rsplit("_snr", 1)
        steps = 1 if base_name == "untargeted_fgsm" else 20
        seed = int(base["seed_base"] + 100000 + 10000 * outer + 1000 * inner
                   + 100 * E.ATTACK_CODES[base_name] + int(snr_text))
        scores, _, waves = E.E54.raw_attack(model, known_raw, labels, int(snr_text), base_name,
                                            steps, seed,
                                            target=targets if base_name == "targeted_pgd" else None,
                                            batch_size=64, return_waveforms=True)
        log_adv = wccn.scores(E.E65.gpu_correlation_logspd(waves, e36, e65))
        apred, _, _ = E.decide(scores.cpu().numpy(), log_adv, enrolled, tau_low, tau_high)
        out["attacks"][name] = (float(np.mean(apred == true)),
                                float(np.mean((apred == true) | (apred == 0))))
    del model, fit_raw, known_raw
    torch.cuda.empty_cache()
    return out


variants = {
    "A_pgd7_r4_ep44": {"epochs": 44, "cache_pgd_steps": 7, "cache_refresh_epochs": 4},
    "B_pgd5_r5_w6_ep44": {"epochs": 44, "cache_pgd_steps": 5, "cache_refresh_epochs": 5,
                          "adversarial_ce_weight": 0.6, "clean_ce_weight": 0.4},
    "C_pgd3_r6_ep60": {"epochs": 60, "cache_pgd_steps": 3, "cache_refresh_epochs": 6},
}
for key, patch in variants.items():
    cfg = json.loads(json.dumps(base))
    cfg["model"]["temporal_filters"] = 16
    cfg["model"]["separable_filters"] = 32
    cfg["training"].update(patch)
    t0 = time.time()
    rows = []
    for outer in (1, 2):
        rows.append(one(cfg, outer, 1))
    ra = {n: float(np.mean([r["attacks"][n][0] for r in rows])) for n in E.PRIMARY_ATTACKS}
    sf = {n: float(np.mean([r["attacks"][n][1] for r in rows])) for n in E.PRIMARY_ATTACKS}
    print(f"### {key}: cleanC={np.mean([r['clean'] for r in rows]):.4f} "
          f"RA30={ra} worst={min(ra.values()):.4f} "
          f"ASafeR30={sf} worst={min(sf.values()):.4f} {time.time()-t0:.0f}s", flush=True)
