"""查看器方案注册表与指标：提交标签 + 可选测试分数 / 训练 OOF，按 key 懒加载并缓存。

仅对比当前最佳与三次提交：
  s1 第 1 次提交：grouped guarded 测试分数 + 训练 blocked OOF（与 submission_task_specific_zscore_grouped_guarded.csv 同 MD5）
  s2 第 2 次提交：research/E10_second_submission/conservative_pair_gate.py 在 s1 分数上做保守配对门控；
     被门控回退到 direct 的行（标签 = direct argmax ≠ grouped argmax）改用 scores_task_specific_zscore.npy，
     其余沿用 grouped 分数；门控只作用于测试集，训练 OOF 沿用 s1
  s3 第 3 次提交：E11 improved_task_threshold full_none_global；只保存了训练 nested blocked OOF（full_none 分数），
     测试分数未落盘 -> 测试仅标签
  s4 第 3 次 rawPGD 备选：E09 未保存测试分数 / OOF -> 仅标签

代理指标（无测试真值）：基于“session1≈未注册、session2≈已注册”的元数据假设与参考方案伪真值，
  U=session1 拒识率；C=session2 上与参考方案非 0 标签一致率；RA=C；ASafeR=(一致+判 0)/参考非 0 的 session2 数。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ANSWER = ROOT / "commit" / "answer"
E11 = ROOT / "research" / "E11_session_blind_candidate" / "improved_task_threshold"
N_SUBJ = 20
LABELS = np.arange(1, N_SUBJ + 1)
DEFAULT = "s1"
PROXY_NOTE = "代理估计（基于 session 假设与参考方案伪真值）"


def _l(a, n: int = 4):
    a = np.round(np.asarray(a, dtype=np.float64), n)
    o = a.astype(object)
    o[~np.isfinite(a)] = None
    return o.tolist()


def _f(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


def _margin(s: np.ndarray) -> np.ndarray:
    top = np.sort(s, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


# ---------- 分数 / OOF 加载器 ----------
def _grouped(cache, meta, labels):
    return np.load(cache / "scores_task_specific_zscore_grouped_guarded.npy")


def _pair_gate(cache, meta, labels):
    g = _grouped(cache, meta, labels)
    d = np.load(cache / "scores_task_specific_zscore.npy")
    use = (labels == d.argmax(1) + 1) & (labels != g.argmax(1) + 1)
    return np.where(use[:, None], d, g)


def _cache_oof(cache, meta):
    tr = meta["train"]
    pred = pd.read_csv(cache / "train_blocked_oof_task_specific_guarded_predictions.csv").set_index("path")["predicted_label"]
    return np.load(cache / "train_blocked_oof_scores_task_specific_guarded.npy"), pred.loc[tr["path"]].to_numpy(int)


def _e11_oof(cache, meta):
    tr = meta["train"]
    z = np.load(E11 / "blocked_oof_scores.npz")
    p = pd.read_csv(E11 / "nested_blocked_oof_predictions.csv", usecols=["subject", "task", "full_none_global"])
    if not (np.array_equal(z["classes"], LABELS) and p[["subject", "task"]].equals(tr[["subject", "task"]])):
        raise ValueError("E11 OOF 与训练元数据未对齐")
    return z["full_none"].astype(np.float32), p["full_none_global"].to_numpy(int)


REGISTRY = {
    "s1": dict(label="第1次提交 (官方 69.17，当前最佳)", official=69.17, test_labels=ANSWER / "赞美诗_第1次.csv",
               test_scores=_grouped, oof=_cache_oof,
               desc="task-specific zscore + 配对分组 + guarded；分数=grouped guarded；OOF=blocked guarded"),
    "s2": dict(label="第2次提交 (官方 67.94)", official=67.94, test_labels=ANSWER / "赞美诗_第2次.csv",
               test_scores=_pair_gate, oof=_cache_oof,
               desc="s1 上的保守配对门控；回退行用 direct 分数，其余 grouped；门控不改训练，OOF 同 s1"),
    "s3": dict(label="第3次提交 (官方 26.10)", official=26.10, test_labels=ANSWER / "赞美诗_第3次.csv",
               test_scores=None, oof=_e11_oof,
               desc="session-blind full_none_global（5% 分位全局阈值）；OOF=E11 nested blocked；测试分数未保存"),
    "s4": dict(label="第3次 rawPGD 备选", official=None, test_labels=ANSWER / "赞美诗_第3次_rawPGD.csv",
               test_scores=None, oof=None, desc="raw-EEG PGD 对抗 ridge；E09 未保存分数，仅标签"),
}


class Scheme:
    def __init__(self, key: str, cache: Path, meta: dict):
        cfg = REGISTRY[key]
        self.key, self.cfg = key, cfg
        te = meta["test"]
        sub = pd.read_csv(cfg["test_labels"]).set_index("Sample")["Label"]
        self.pred = {"test": sub.loc[te["sample"]].to_numpy(int), "train": None}
        self.scores = {"test": cfg["test_scores"](cache, meta, self.pred["test"]) if cfg["test_scores"] else None,
                       "train": None}
        if cfg["oof"]:
            self.scores["train"], self.pred["train"] = cfg["oof"](cache, meta)
        self.margin = {s: None if v is None else _margin(v) for s, v in self.scores.items()}
        agree = None
        if self.scores["test"] is not None:
            nz = self.pred["test"] != 0
            agree = float(np.mean(self.scores["test"][nz].argmax(1) + 1 == self.pred["test"][nz]))
        self.info = {"key": key, "label": cfg["label"], "desc": cfg["desc"], "official": cfg["official"],
                     "test_labels": cfg["test_labels"].name, "has_test_scores": self.scores["test"] is not None,
                     "has_oof": self.pred["train"] is not None, "score_label_agree": agree}


class Registry:
    def __init__(self, cache: Path, meta: dict):
        self.cache, self.meta = cache, meta
        self._s, self._m, self._c = {}, {}, {}
        self._lock = threading.Lock()

    def key(self, k: str | None) -> str:
        k = k or DEFAULT
        if k not in REGISTRY:
            raise ValueError(f"未知方案 {k}")
        return k

    def get(self, k: str | None) -> Scheme:
        k = self.key(k)
        with self._lock:
            if k not in self._s:
                self._s[k] = Scheme(k, self.cache, self.meta)
            return self._s[k]

    def list(self) -> list:
        return [self.get(k).info for k in REGISTRY]

    def metrics(self, k: str | None, ref: str | None) -> dict:
        k, ref = self.key(k), self.key(ref)
        if (k, ref) not in self._m:
            self._m[(k, ref)] = metrics(self.get(k), self.get(ref), self.meta)
        return self._m[(k, ref)]

    def compare(self, a: str | None, b: str | None) -> dict:
        a, b = self.key(a), self.key(b)
        if (a, b) not in self._c:
            self._c[(a, b)] = compare(self.get(a), self.get(b), self.meta)
        return self._c[(a, b)]


# ---------- 指标 ----------
def _macro_f1(y, p) -> float:
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def oof_metrics(sch: Scheme, meta: dict) -> dict | None:
    p, s, m = sch.pred["train"], sch.scores["train"], sch.margin["train"]
    if p is None:
        return None
    tr = meta["train"]
    y, tasks = tr["subject"].to_numpy(int), tr["task"].to_numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    conf = np.zeros((N_SUBJ, N_SUBJ + 1), int)
    np.add.at(conf, (y - 1, p), 1)
    am = s.argmax(1) + 1
    ok = am == y
    hi = float(np.quantile(m, 0.99))
    edges = np.linspace(0, hi, 41)
    sweep = []
    for t in np.unique(np.quantile(m, np.linspace(0, 0.9, 46))):
        keep = m >= t
        q = np.where(keep, am, 0)
        sweep.append({"t": _f(t), "reject": _f(1 - keep.mean()), "acc_accepted": _f(ok[keep].mean()),
                      "f1": _f(_macro_f1(y, q)), "C": _f(np.mean(q == y))})
    return {
        "n": int(len(y)), "acc": _f(np.mean(p == y)), "macro_f1": _f(_macro_f1(y, p)),
        "reject_rate": _f(np.mean(p == 0)), "argmax_acc": _f(ok.mean()),
        "margin_auc": _f(roc_auc_score(ok, m)) if 0 < ok.sum() < len(ok) else None,
        "auc_note": "margin(top1−top2) 区分 OOF argmax 正确 / 错分的 ROC AUC",
        "per_subject": {"precision": _l(prec), "recall": _l(rec), "f1": _l(f1), "n": np.bincount(y - 1, minlength=N_SUBJ).tolist()},
        "confusion": conf.tolist(), "confusion_note": "行=真值 S1–S20，列=预测 0–20",
        "by_task": {t: {"n": int((tasks == t).sum()), "acc": _f(np.mean(p[tasks == t] == y[tasks == t])),
                        "f1": _f(_macro_f1(y[tasks == t], p[tasks == t]))} for t in np.unique(tasks)},
        "margin_hist": {"edges": _l(edges), "correct": np.histogram(np.clip(m[ok], 0, hi), edges)[0].tolist(),
                        "wrong": np.histogram(np.clip(m[~ok], 0, hi), edges)[0].tolist()},
        "sweep": sweep, "sweep_note": "argmax 预测上按 margin 阈值拒识；F1 为 1–20 宏平均（拒识计 FN），C 拒识计错",
    }


def test_metrics(sch: Scheme, meta: dict) -> dict:
    te = meta["test"]
    p, tasks, sess = sch.pred["test"], te["task"].to_numpy(), te["session"].to_numpy()
    cnt = lambda mask: np.bincount(p[mask], minlength=N_SUBJ + 1).tolist()  # noqa: E731
    out = {
        "n": int(len(p)), "counts": cnt(np.ones(len(p), bool)), "reject": int(np.sum(p == 0)),
        "reject_rate": _f(np.mean(p == 0)),
        "by_task": {t: _f(np.mean(p[tasks == t] == 0)) for t in np.unique(tasks)},
        "by_session": {s: _f(np.mean(p[sess == s] == 0)) for s in np.unique(sess)},
        "counts_by_session": {s: cnt(sess == s) for s in np.unique(sess)},
    }
    m = sch.margin["test"]
    if m is not None:
        out["margin_quantiles"] = _l(np.quantile(m[p != 0], [0.05, 0.25, 0.5, 0.75, 0.95])) if (p != 0).any() else None
    return out


def proxy(sch: Scheme, ref: Scheme, meta: dict) -> dict:
    sess = meta["test"]["session"].to_numpy()
    p, r = sch.pred["test"], ref.pred["test"]
    s1, base = sess == "session1", (sess == "session2") & (r != 0)
    u = float(np.mean(p[s1] == 0))
    agree = p[base] == r[base]
    c = float(agree.mean())
    asafer = float(np.mean(agree | (p[base] == 0)))
    score = 100 * c ** 0.4 * (0.7 * c + 0.3 * asafer) ** 0.35 * u ** 0.25
    return {"note": PROXY_NOTE, "ref": ref.key, "U": _f(u), "C": _f(c), "RA": _f(c), "ASafeR": _f(asafer),
            "score": round(score, 3), "n_session1": int(s1.sum()), "n_base": int(base.sum()),
            "official": sch.cfg["official"]}


def metrics(sch: Scheme, ref: Scheme, meta: dict) -> dict:
    return {"scheme": sch.info, "oof": oof_metrics(sch, meta), "test": test_metrics(sch, meta),
            "proxy": proxy(sch, ref, meta)}


def compare(a: Scheme, b: Scheme, meta: dict) -> dict:
    te = meta["test"]
    tasks, sess = te["task"].to_numpy(), te["session"].to_numpy()
    pa, pb = a.pred["test"], b.pred["test"]
    same = pa == pb
    trans = np.zeros((N_SUBJ + 1, N_SUBJ + 1), int)
    np.add.at(trans, (pa, pb), 1)
    idx = np.flatnonzero(~same)
    mg = lambda s: None if s.margin["test"] is None else _l(s.margin["test"][idx], 3)  # noqa: E731
    out = {
        "a": a.info, "b": b.info, "n": int(len(pa)), "agree": _f(same.mean()), "n_diff": int(len(idx)),
        "by_task": {t: _f(same[tasks == t].mean()) for t in np.unique(tasks)},
        "by_session": {s: _f(same[sess == s].mean()) for s in np.unique(sess)},
        "transition": trans.tolist(), "transition_note": "行=A 标签 0–20，列=B 标签 0–20",
        "diff": {"idx": idx.tolist(), "a": pa[idx].tolist(), "b": pb[idx].tolist(), "ma": mg(a), "mb": mg(b),
                 "task": tasks[idx].tolist(), "session": sess[idx].tolist()},
        "test_count_delta": (np.bincount(pb, minlength=N_SUBJ + 1) - np.bincount(pa, minlength=N_SUBJ + 1)).tolist(),
        "oof_delta": None,
    }
    if a.pred["train"] is not None and b.pred["train"] is not None:
        y = meta["train"]["subject"].to_numpy(int)
        ra = precision_recall_fscore_support(y, a.pred["train"], labels=LABELS, zero_division=0)
        rb = precision_recall_fscore_support(y, b.pred["train"], labels=LABELS, zero_division=0)
        out["oof_delta"] = {
            "acc": _f(np.mean(b.pred["train"] == y) - np.mean(a.pred["train"] == y)),
            "macro_f1": _f(_macro_f1(y, b.pred["train"]) - _macro_f1(y, a.pred["train"])),
            "agree": _f(np.mean(a.pred["train"] == b.pred["train"])),
            "subject_acc": _l(rb[1] - ra[1]), "subject_f1": _l(rb[2] - ra[2]),
        }
    return out
