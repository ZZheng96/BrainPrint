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
oe = E.load_external(cfg, dev)
cfg2 = json.loads(json.dumps(cfg))
cfg2["model"]["temporal_filters"] = 16
cfg2["model"]["separable_filters"] = 32
train_rows = np.flatnonzero(P["fit_rows"])
eval_rows = np.flatnonzero(P["window"])
tr = raw[torch.as_tensor(train_rows, device=dev)].contiguous()
tl = torch.as_tensor([lut[int(v)] for v in subjects[train_rows]], dtype=torch.long, device=dev)
t0 = time.time()
model, info = E.train_identity(tr, tl, oe, len(CLASSES), cfg2, 777, dev, 50, True, True, "f16_atoe50")
ins = E.identity_scores(model, tr).argmax(1)
es = E.identity_scores(model, raw[torch.as_tensor(eval_rows, device=dev)]).argmax(1)
print(f"params={info['parameters']} in-sample={np.mean(np.asarray(CLASSES)[ins]==subjects[train_rows]):.4f} "
      f"evalblock={np.mean(np.asarray(CLASSES)[es]==subjects[eval_rows]):.4f} "
      f"evf1={f1_score(subjects[eval_rows], np.asarray(CLASSES)[es], labels=CLASSES, average='macro'):.4f} "
      f"{time.time()-t0:.0f}s", flush=True)
