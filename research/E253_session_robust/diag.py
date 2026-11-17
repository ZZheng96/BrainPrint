"""E253d: error structure of the frozen model on the registered Test split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final/team_hymn"
CACHE = ROOT / "BrainPrint_cache/e253_subwindow"
sys.path.insert(0, str(PKG))
import model as M  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sweep import load_artifacts, project, score  # noqa: E402


def main() -> None:
    art = load_artifacts()
    te_feat = np.load(CACHE / "test_feat.npy")
    te = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    labels = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(labels, left_on="sample", right_on="Sample", how="left")
    te_tasks = M._task_indices([{"task": t} for t in te["task"]], len(te))
    p = project(art, te_feat, te_tasks)
    sc = torch.full((len(p), 20), float("inf"))
    for t in range(4):
        r = np.nonzero(te_tasks == t)[0]
        if r.size:
            sc[r] = score(p[r], art["prototypes"][t], "euclid", "mean")
    pred = sc.argmin(1).numpy() + 1
    dmin = sc.min(1).values.numpy()
    lab = te["Label"].to_numpy()
    reg = lab > 0
    acc = pred[reg] == lab[reg]

    print("registered n", int(reg.sum()), "top1", round(acc.mean(), 4))
    df = pd.DataFrame({"subject": lab[reg], "task": te["task"].to_numpy()[reg],
                       "pred": pred[reg], "correct": acc, "dmin": dmin[reg]})
    print("\nper-task accuracy")
    print(df.groupby("task")["correct"].agg(["mean", "size"]).round(3).to_string())
    print("\nper-subject accuracy")
    print(df.groupby("subject")["correct"].agg(["mean", "size"]).round(3).to_string())
    print("\nper-subject median dmin")
    print(df.groupby("subject")["dmin"].median().round(2).to_string())
    print("\ntop predicted classes")
    print(df["pred"].value_counts().head(8).to_string())
    print("\nconfusion top-10 error pairs")
    err = df[~df["correct"]].groupby(["subject", "pred"]).size().sort_values(ascending=False)
    print(err.head(10).to_string())


if __name__ == "__main__":
    main()
