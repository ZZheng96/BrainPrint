"""E174 步骤 3：按历史实验的统一字段族抽取"已公布的三元组"。

支持字段族：
  best_*   : best_W / best_acceptance / loso_at_best / s1_at_best / s2_at_best
  ref_*    : W_at_ref / ref_acceptance / loso_at_ref / s1_at_ref / s2_at_ref
  090_*    : W_at_090 / loso_at_090 / s1_at_090 / s2_at_090（接受率固定 0.90）
  curve[]  : acceptance_rate + LOSO_rejection + Test_S1_rejection + Test_S2_acceptance

输出 reported_w.json。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SKIP = {"feature_manifest.json", "pixel_manifest.json", "manifest.json",
        "encoder_config.json", "config.json"}


def num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def g(d, *keys):
    for k in keys:
        v = num(d.get(k))
        if v is not None:
            return v
    return None


def walk(obj, path=""):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def extract(exp, jf, data):
    out = []
    for path, d in walk(data):
        cfg = d.get("config") or d.get("name") or d.get("variant") or ""
        # 1) best_* 家族
        w = num(d.get("best_W"))
        l, s1, s2 = g(d, "loso_at_best"), g(d, "s1_at_best"), g(d, "s2_at_best")
        if w is not None and None not in (l, s1, s2):
            out.append(dict(experiment=exp, file=jf.name, path=path, config=str(cfg),
                            acceptance=g(d, "best_acceptance"),
                            LOSO=l, S1=s1, S2=s2, W=w, W_source="best"))
        # 2) ref_* 家族
        w = num(d.get("W_at_ref"))
        l, s1, s2 = g(d, "loso_at_ref"), g(d, "s1_at_ref"), g(d, "s2_at_ref")
        if w is not None and None not in (l, s1, s2):
            out.append(dict(experiment=exp, file=jf.name, path=path, config=str(cfg),
                            acceptance=g(d, "ref_acceptance"),
                            LOSO=l, S1=s1, S2=s2, W=w, W_source="ref"))
        # 3) _090 家族
        w = num(d.get("W_at_090"))
        l, s1, s2 = g(d, "loso_at_090"), g(d, "s1_at_090"), g(d, "s2_at_090")
        if w is not None and None not in (l, s1, s2):
            out.append(dict(experiment=exp, file=jf.name, path=path, config=str(cfg),
                            acceptance=0.90, LOSO=l, S1=s1, S2=s2, W=w, W_source="a090"))
        # 4) curve 家族
        a = g(d, "acceptance_rate", "acceptance", "a", "target_acceptance", "accept_rate")
        l = g(d, "LOSO_rejection", "loso_rejection", "unknown_rejection")
        s1 = g(d, "Test_S1_rejection", "S1_rejection_rate", "S1_rejection", "s1_reject")
        s2 = g(d, "Test_S2_acceptance", "S2_acceptance_rate", "S2_acceptance", "s2_accept")
        ka = g(d, "train_known_acceptance", "known_acceptance")
        if a is None and ka is not None and "TEST" not in path:
            a = ka
        if None not in (a, l, s1, s2) and 0 <= l <= 1 and 0 <= s1 <= 1 and 0 <= s2 <= 1:
            out.append(dict(experiment=exp, file=jf.name, path=path, config=str(cfg),
                            acceptance=a, LOSO=l, S1=s1, S2=s2,
                            W=0.25 * l + 0.25 * s1 + 0.5 * s2, W_source="curve"))
        # 5) 嵌套 LOSO/TEST 家族（E122 three_way.k1.LOSO + .TEST）
        lb = d.get("LOSO") if isinstance(d.get("LOSO"), dict) else None
        tb = d.get("TEST") if isinstance(d.get("TEST"), dict) else None
        if lb and tb:
            l = g(lb, "unknown_rejection_mean", "unknown_rejection")
            s1 = g(tb, "S1_rejection_rate")
            s2 = g(tb, "S2_acceptance_rate")
            a = g(tb, "train_known_acceptance") or g(lb, "known_acceptance_mean")
            if None not in (a, l, s1, s2):
                out.append(dict(experiment=exp, file=jf.name, path=path, config=str(cfg),
                                acceptance=a, LOSO=l, S1=s1, S2=s2,
                                W=0.25 * l + 0.25 * s1 + 0.5 * s2, W_source="nested"))
    return out


def main():
    recs = []
    for exp in sorted(p for p in (ROOT / "research").iterdir()
                      if p.is_dir() and re.match(r"E\d+", p.name)):
        for jf in sorted(exp.glob("*.json")):
            if jf.name in SKIP:
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            recs.extend(extract(exp.name, jf, data))

    # 去重：同实验 + 同 (config, acceptance, S1, S2, LOSO)
    seen, uniq = set(), []
    for r in recs:
        key = (r["experiment"], r["config"], round(r["acceptance"] or -1, 6),
               round(r["S1"], 6), round(r["S2"], 6), round(r["LOSO"], 6))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    print(f"raw={len(recs)} unique={len(uniq)}")
    (OUT / "reported_w.json").write_text(
        json.dumps(uniq, ensure_ascii=False, indent=1), encoding="utf-8")

    best = {}
    for r in uniq:
        cur = best.get(r["experiment"])
        if cur is None or r["W"] > cur["W"]:
            best[r["experiment"]] = r
    for k, r in sorted(best.items(), key=lambda x: -x[1]["W"]):
        print(f"{r['W']:.4f} {k:44s} a={r['acceptance']} S1={r['S1']:.3f} "
              f"S2={r['S2']:.3f} L={r['LOSO']:.3f} [{r['W_source']}] {r['config'][:40]}")


if __name__ == "__main__":
    main()
