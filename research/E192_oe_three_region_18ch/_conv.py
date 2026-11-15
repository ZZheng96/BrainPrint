import json, sys, time
from pathlib import Path
import numpy as np, torch

OUT = Path(r"d:\WORK\BCIC2026\BrainPrint\research\E192_oe_three_region_18ch")
sys.path.insert(0, str(OUT))
import run_e192 as E
from sklearn.metrics import f1_score

cfg = E.load_config()
dev = torch.device("cuda")
meta = E.train_metadata(cfg)
P = E.build_protocol(meta)
subjects = P["subjects"]
CLASSES = E.CLASSES
lut = {int(c): i for i, c in enumerate(CLASSES)}
raw = E.E54.load_all_raw(meta, cfg, dev)

# block-4 折外：训练 = fit_rows & block!=4（等价于全部 fit_rows），评估 block 4
held = np.flatnonzero(P["blocks"] == 4)
train_rows = np.flatnonzero(P["fit_rows"])
held_raw = raw[torch.as_tensor(held, device=dev)]


def run(epochs, f1, f2, oe, at, tag):
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["model"]["temporal_filters"] = f1
    cfg2["model"]["separable_filters"] = f2
    tr = raw[torch.as_tensor(train_rows, device=dev)].contiguous()
    tl = torch.as_tensor([lut[int(v)] for v in subjects[train_rows]], dtype=torch.long, device=dev)
    t0 = time.time()
    model, info = E.train_identity(tr, tl, oe, len(CLASSES), cfg2, 777, dev, epochs, oe, at, tag)
    ins = E.identity_scores(model, tr).argmax(1)
    hs = E.identity_scores(model, held_raw).argmax(1)
    print(f"{tag}: params={info['parameters']} in-sample={np.mean(np.asarray(CLASSES)[ins]==subjects[train_rows]):.4f} "
          f"block4={np.mean(np.asarray(CLASSES)[hs]==subjects[held]):.4f} "
          f"f1={f1_score(subjects[held], np.asarray(CLASSES)[hs], labels=CLASSES, average='macro'):.4f} "
          f"{time.time()-t0:.0f}s", flush=True)
    del tr, model
    torch.cuda.empty_cache()


oe = E.load_external(cfg, dev)
run(40, 8, 16, False, False, "clean40_f8")
run(80, 8, 16, False, False, "clean80_f8")
run(40, 16, 32, False, False, "clean40_f16")
