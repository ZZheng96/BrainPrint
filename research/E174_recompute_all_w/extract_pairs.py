"""E174 步骤 2：从 results/metrics JSON 里抽取 (Test S1, Test S2, LOSO) 三元组。

适配历史上的多种命名：
  audit.S1_rejection_rate / S2_acceptance_rate
  train_only_metrics.rule.unknown_rejection + known_acceptance
  threshold.unknown_rejection
  loso_reference.unknown_rejection 等

输出 json_pairs.json，供人工逐条核对工作点一致性。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

S1_KEYS = ("S1_rejection_rate", "Test_S1_rejection", "test_s1_rejection",
           "s1_reject", "s1_rejection", "S1_rejection")
S2_ACC_KEYS = ("S2_acceptance_rate", "Test_S2_acceptance", "test_s2_acceptance",
               "s2_accept", "s2_acceptance", "S2_acceptance")
S2_REJ_KEYS = ("S2_rejection_rate", "test_s2_rejection", "s2_reject")
LOSO_KEYS = ("unknown_rejection", "loso_rejection", "LOSO_rejection",
             "loso_unknown_rejection", "unknown_rejection_mean",
             "heldout_rejection", "unknown_rejection_rate",
             "unknown_rejection_at_95pct_known_acceptance")
KA_KEYS = ("known_acceptance", "acceptance_rate", "target_acceptance",
           "train_outer_accept", "ka", "accept")
N_KEYS = ("N_pseudo_unknown", "N_unknown", "N_external_unknown", "n_unknown")


def num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def pick(d, keys):
    for k in keys:
        if k in d:
            v = num(d[k])
            if v is not None:
                return v, k
    return None, None


def walk(obj, path="", parent=None):
    if isinstance(obj, dict):
        yield path, obj, parent
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k), obj)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]", parent)


def main():
    pairs = []
    for exp in sorted(p for p in (ROOT / "research").iterdir()
                      if p.is_dir() and re.match(r"E\d+", p.name)):
        for jf in sorted(exp.glob("*.json")):
            if jf.name in {"feature_manifest.json", "pixel_manifest.json",
                           "manifest.json", "encoder_config.json", "config.json"}:
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            nodes = list(walk(data))
            # 同一文件内的全局 LOSO 候选（root 级 rule / threshold / loso_reference）
            for path, node, parent in nodes:
                s1, s1k = pick(node, S1_KEYS)
                s2, s2k = pick(node, S2_ACC_KEYS)
                if s2 is None:
                    s2r, s2rk = pick(node, S2_REJ_KEYS)
                    if s2r is not None:
                        s2, s2k = 1.0 - s2r, f"1-{s2rk}"
                if s1 is None or s2 is None:
                    continue
                if not (0 <= s1 <= 1 and 0 <= s2 <= 1):
                    continue
                # LOSO：先本节点，再父节点，再同文件内同层级 rule/threshold
                loso, losok = pick(node, LOSO_KEYS)
                ka, kak = pick(node, KA_KEYS)
                src = path
                if loso is None and parent is not None:
                    loso, losok = pick(parent, LOSO_KEYS)
                    ka2, kak2 = pick(parent, KA_KEYS)
                    ka = ka if ka is not None else ka2
                    src = src + " <- parent"
                if loso is None:
                    # 文件内唯一的 train-only rule / threshold
                    cands = []
                    for p2, n2, _ in nodes:
                        if p2 == path:
                            continue
                        v, k = pick(n2, LOSO_KEYS)
                        if v is not None and re.search(r"rule|threshold|loso|reference|train",
                                                       p2, re.I):
                            ka2, _ = pick(n2, KA_KEYS)
                            cands.append((p2, v, k, ka2))
                    if len(cands) == 1:
                        src = cands[0][0]
                        loso, losok, ka = cands[0][1], cands[0][2], cands[0][3]
                    elif len(cands) > 1:
                        # 多个候选时只认部署块 train_only_metrics.*；仍不唯一则放弃
                        dep = [c for c in cands if c[0].startswith("train_only_metrics")]
                        if len(dep) == 1:
                            src = dep[0][0]
                            loso, losok, ka = dep[0][1], dep[0][2], dep[0][3]
                        else:
                            loso = None
                            losok = f"MULTI({len(cands)})"
                nkey = pick(node, N_KEYS)[1]
                pairs.append({
                    "experiment": exp.name,
                    "file": jf.name,
                    "path": path,
                    "S1": s1, "S1_key": s1k,
                    "S2": s2, "S2_key": s2k,
                    "LOSO": loso, "LOSO_key": losok, "LOSO_src": src,
                    "known_acceptance": ka, "KA_key": kak,
                    "N_unknown_key": nkey,
                })

    print(f"pairs={len(pairs)}")
    (OUT / "json_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")

    by = Counter(p["experiment"] for p in pairs)
    for k, v in sorted(by.items()):
        ws = [0.25 * p["LOSO"] + 0.25 * p["S1"] + 0.5 * p["S2"]
              for p in pairs if p["experiment"] == k and p["LOSO"] is not None]
        print(f"{k:46s} n={v:3d} with_loso={sum(1 for p in pairs if p['experiment']==k and p['LOSO'] is not None):3d}"
              + (f" maxW={max(ws):.4f}" if ws else ""))


if __name__ == "__main__":
    main()
