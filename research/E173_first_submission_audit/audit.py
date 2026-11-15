"""审计三次正式提交 CSV 的 Test S1 拒识率与 S2 接受率（session 仅做工程审计）。"""
import pandas as pd

META = "BrainPrint_cache/test_metadata.csv"
FILES = [
    ("第1次 (官方 69.17)", "commit/answer/赞美诗_第1次.csv"),
    ("第2次 (官方 67.94)", "commit/answer/赞美诗_第2次.csv"),
    ("第3次 (官方 26.10)", "commit/answer/赞美诗_第3次.csv"),
]

meta = pd.read_csv(META, usecols=["sample", "session", "task"])
for name, path in FILES:
    d = pd.read_csv(path).rename(columns={"Sample": "sample", "Label": "label"})
    j = meta.merge(d, on="sample", how="left")
    miss = int(j["label"].isna().sum())
    j["label"] = j["label"].fillna(-1).astype(int)
    s1 = j.session.astype(str).str.casefold().eq("session1")
    print("=" * 60)
    print(name, path)
    print("  rows", len(j), "unmatched", miss)
    print("  S1 n", int(s1.sum()), " S2 n", int((~s1).sum()))
    print("  S1 rejection (label==0):", round(float((j.loc[s1, "label"] == 0).mean()), 4))
    print("  S2 acceptance (label!=0):", round(float((j.loc[~s1, "label"] != 0).mean()), 4))
    print("  label 0 total:", int((j["label"] == 0).sum()))
    for t in ["EC", "EO", "Imagery", "SSVEP"]:
        a = j[(j.task == t) & s1]
        b = j[(j.task == t) & (~s1)]
        print(f"    {t:8s} S1 rej {float((a['label'] == 0).mean()):.4f}   "
              f"S2 acc {float((b['label'] != 0).mean()):.4f}")
    print("  label distribution:", j["label"].value_counts().sort_index().to_dict())
