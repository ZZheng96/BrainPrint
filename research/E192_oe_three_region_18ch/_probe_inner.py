import json, sys, time
from pathlib import Path
import numpy as np, torch

OUT = Path(r"d:\WORK\BCIC2026\BrainPrint\research\E192_oe_three_region_18ch")
sys.path.insert(0, str(OUT))
import run_e192 as E

cfg = E.load_config()
dev = torch.device("cuda")
meta = E.train_metadata(cfg)
P = E.build_protocol(meta)
e36 = E.E36.load_config()
e65 = json.loads((E.ROOT / "research" / "E65_correlation_spd" / "config.json").read_text(encoding="utf-8"))
raw = E.E54.load_all_raw(meta, cfg, dev)
oe = E.load_external(cfg, dev)
feat = torch.as_tensor(np.load(E.TRAIN_CORR), dtype=torch.float32, device=dev)

for f1, f2, epochs in [(16, 32, 24), (8, 16, 32)]:
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["model"]["temporal_filters"] = f1
    cfg2["model"]["separable_filters"] = f2
    cfg2["training"]["epochs"] = epochs
    t0 = time.time()
    store = E.run_inner(P, cfg2, raw, oe, feat, e36, e65, dev, "main", [1], [1], True)
    s = E.summarize_runs(store)
    print(f"### f1={f1} f2={f2} ep={epochs}: cleanC={s['clean_C']:.4f} U={s['U']:.4f} "
          f"RA30={s['worst_RA_snr30']:.4f} ASafeR30={s['worst_ASafeR_snr30']:.4f} "
          f"cleanF1={s['clean_macro_f1']:.4f} {time.time()-t0:.0f}s", flush=True)
    for k, v in s["per_attack"].items():
        print("   ", k, f"RA={v['RA']:.4f} ASafeR={v['ASafeR']:.4f}")
