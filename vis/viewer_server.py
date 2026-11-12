"""BrainPrint 脑电波与脑纹查看器的本地数据服务（只读，不写盘）。

数据来源：
  Brainprint_Competition/{Train,Test}/*.mat   原始 18 通道 × 1200 点（300 Hz，4 s）
  BrainPrint_cache/{train,test}_features.npy  log-PSD(18×90) + 6 频带 log-协方差特征
  BrainPrint_cache/ 下的方案分数 / OOF 预测、commit/answer 下的提交标签

脑纹计算（启动时在 GPU 上完成，逐任务）：
  - 每组（训练 / 测试 session1 / 测试 session2）按自身均值方差 z 标准化，等价于赛题方案的 zscore 对齐；
  - 被试模板 = 训练 trial 的平均 z 特征（偏离总体的程度，单位为总体标准差）；
  - Fisher 辨识度 = 被试间方差 / 被试内方差（逐特征）；
  - 嵌入 = PCA(200) → 收缩 LDA(19)，训练集拟合，测试集投影；模板相似度为 LDA 空间余弦；
  - 相似度矩阵：训练前/后半 trial 模板的 Pearson 相关（split），或训练模板 vs 测试 session2 按预测标签分组的模板（cross）。

接口：
  GET /                                   -> viewer.html
  GET /api/schemes                        -> 方案注册表（vis/schemes.py；s1 默认）
  GET /api/summary?scheme=                -> 被试 / 任务统计、OOF 指标、测试预测分布、通道名
  GET /api/metrics?scheme=&ref=           -> OOF 指标 / 测试分布 / 代理分（ref 为伪真值方案）
  GET /api/compare?a=&b=                  -> 两方案一致率、转移矩阵、分歧样本、逐被试 Δ
  GET /api/rows?set=&task=&subject=&session=&scheme=&b=
                                          -> trial 列表（训练集按真实标签筛选，测试集按方案 A 预测标签）
  GET /api/trial/<set>/<idx>?scheme=&b=   -> 波形（原始 + 0.5–90 Hz/50 Hz 陷波滤波）、PSD、协方差、A/B 分数
  （嵌入 / 脑纹 / 相似度与方案无关；cross 相似度与配对按 s1 预测分组）
  GET /api/embed?task=                    -> LDA 二维嵌入点
  GET /api/fingerprint?subject=&task=     -> 被试脑纹（PSD / 协方差 z 偏差、Fisher 辨识度）
  GET /api/similarity?task=&domain=&mode= -> 20×20 模板相似度矩阵

用法：
  python vis/viewer_server.py
  python vis/viewer_server.py --port 8766 --no-browser
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import filtfilt, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_features import (  # noqa: E402
    _FULL_BAND_SOS,
    _NOTCH_A,
    _NOTCH_B,
    COV_BANDS,
    FREQ_MAX,
    FREQ_MIN,
    N_CHANNELS,
    N_PSD_FEATURES,
    load_eeg,
    robust_clip,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemes import DEFAULT, Registry  # noqa: E402

DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"
VIEWER = Path(__file__).resolve().parent / "viewer.html"
SETS = ("train", "test")
TASKS = ("EC", "EO", "Imagery", "SSVEP")
N_SUBJ = 20
N_FREQ = FREQ_MAX - FREQ_MIN + 1
N_BANDS = len(COV_BANDS)
TRIU = np.triu_indices(N_CHANNELS)
OFF = TRIU[0] != TRIU[1]
DOMAINS = {"all": slice(None), "psd": slice(0, N_PSD_FEATURES), "cov": slice(N_PSD_FEATURES, None)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jl(a, n: int = 3):
    """numpy -> 可 JSON 序列化的嵌套列表，非有限值转 null。"""
    a = np.round(np.asarray(a, dtype=np.float64), n)
    if np.isfinite(a).all():
        return a.tolist()
    o = a.astype(object)
    o[~np.isfinite(a)] = None
    return o.tolist()


def cov_matrices(vec: np.ndarray) -> np.ndarray:
    """(..., 6×171) 上三角向量 -> (..., 6, 18, 18) 对称矩阵（还原非对角 √2 缩放）。"""
    v = np.asarray(vec, dtype=np.float32).reshape(*vec.shape[:-1], N_BANDS, -1).copy()
    v[..., OFF] /= np.sqrt(2.0)
    m = np.zeros((*v.shape[:-1], N_CHANNELS, N_CHANNELS), np.float32)
    m[..., TRIU[0], TRIU[1]] = v
    m[..., TRIU[1], TRIU[0]] = v
    return m


def split_feat(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vec[..., :N_PSD_FEATURES].reshape(*vec.shape[:-1], N_CHANNELS, N_FREQ), cov_matrices(vec[..., N_PSD_FEATURES:])


def row_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return F.normalize(a, dim=1) @ F.normalize(b, dim=1).T


def zstats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = x.mean(0), x.std(0)
    return mu, torch.where(sd > 1e-6, sd, torch.ones_like(sd))


class Store:
    def __init__(self):
        t0 = time.time()
        self.meta = {s: pd.read_csv(CACHE / f"{s}_metadata.csv") for s in SETS}
        self.x = {s: np.load(CACHE / f"{s}_features.npy", mmap_mode="r") for s in SETS}
        tr, te = self.meta["train"], self.meta["test"]
        self.reg = Registry(CACHE, self.meta)
        base = self.reg.get(DEFAULT)  # 嵌入 / cross 相似度 / 配对等与方案无关的部分固定用 s1
        self.scheme = base.info
        self.scores, self.pred, self.margin = base.scores, base.pred, base.margin
        self.label = {"train": tr["subject"].to_numpy(int), "test": np.full(len(te), -1)}
        self.task = {s: self.meta[s]["task"].to_numpy() for s in SETS}
        self.session = {s: self.meta[s]["session"].to_numpy() for s in SETS}
        pairs = pd.read_csv(CACHE / "waveform_pairs.csv")
        row = dict(zip(te["sample"], range(len(te))))
        self.pair = {row[a]: (row[b], float(c)) for a, b, c in
                     zip(pairs["sample"], pairs["nearest_sample"], pairs["similarity"])}
        first = loadmat(DATA / te["path"].iloc[0])
        self.channels = [str(c.reshape(-1)[0]) for c in first["channel_names"].reshape(-1)]
        self._fit()
        self.summary_cache = {}
        print(f"[store] 就绪 {time.time() - t0:.1f}s（设备 {DEVICE}，缓存 {CACHE}）", flush=True)

    # ---------- GPU 脑纹拟合 ----------
    def _group(self, s: str, idx: int):
        return (s, self.task[s][idx]) if s == "train" else (s, self.task[s][idx], self.session[s][idx])

    @torch.no_grad()
    def _fit(self):
        self.stats, self.tmpl, self.raw_mean, self.pop_mean, self.fisher = {}, {}, {}, {}, {}
        self.sim, self.centers, self.eig = {}, {}, {}
        self.proj = {s: np.zeros((len(self.meta[s]), N_SUBJ - 1), np.float32) for s in SETS}
        self.tsim = {s: np.zeros((len(self.meta[s]), N_SUBJ), np.float32) for s in SETS}
        for task in TASKS:
            rows = np.flatnonzero(self.task["train"] == task)
            xt = torch.as_tensor(np.asarray(self.x["train"][rows]), device=DEVICE)
            y = torch.as_tensor(self.label["train"][rows] - 1, device=DEVICE)
            n = len(rows)
            mu, sd = zstats(xt)
            self.stats[("train", task)] = (mu.cpu().numpy(), sd.cpu().numpy())
            zt = (xt - mu) / sd
            oh = F.one_hot(y, N_SUBJ).float()
            cnt = oh.sum(0)
            means = oh.T @ zt / cnt[:, None]
            self.tmpl[task] = means
            self.raw_mean[task] = (oh.T @ xt / cnt[:, None]).cpu().numpy()
            self.pop_mean[task] = mu.cpu().numpy()
            within = ((zt - means[y]) ** 2).sum(0) / (n - N_SUBJ)
            between = (cnt[:, None] * means ** 2).sum(0) / (N_SUBJ - 1)
            self.fisher[task] = (between / within.clamp_min(1e-6)).cpu().numpy()

            # PCA -> 收缩 LDA
            torch.manual_seed(0)
            _, _, v = torch.pca_lowrank(zt, q=min(200, n - 1), center=False, niter=4)
            p = zt @ v
            pm = oh.T @ p / cnt[:, None]
            r = p - pm[y]
            sw = r.T @ r / (n - N_SUBJ)
            q = sw.shape[0]
            sw = 0.9 * sw + 0.1 * torch.trace(sw) / q * torch.eye(q, device=DEVICE)
            pc = p.mean(0)
            sb = ((pm - pc) * cnt[:, None]).T @ (pm - pc) / n
            ew, vw = torch.linalg.eigh(sw)
            wh = vw / ew.clamp_min(1e-9).sqrt()
            eb, vb = torch.linalg.eigh(wh.T @ sb @ wh)
            order = torch.argsort(eb, descending=True)[: N_SUBJ - 1]
            w = wh @ vb[:, order]
            lt = (p - pc) @ w
            sign = torch.sign(lt.pow(3).sum(0))  # 固定方向，保证每次启动一致
            w = w * torch.where(sign == 0, torch.ones_like(sign), sign)
            lt = (p - pc) @ w
            centers = (pm - pc) @ w
            self.centers[task] = centers.cpu().numpy()
            self.eig[task] = (eb[order] / eb[order].sum()).cpu().numpy()
            self.proj["train"][rows] = lt.cpu().numpy()
            self.tsim["train"][rows] = (F.normalize(lt, dim=1) @ F.normalize(centers, dim=1).T).cpu().numpy()

            test_z = {}
            for sess in ("session1", "session2"):
                trow = np.flatnonzero((self.task["test"] == task) & (self.session["test"] == sess))
                if not len(trow):
                    continue
                xs = torch.as_tensor(np.asarray(self.x["test"][trow]), device=DEVICE)
                m2, s2 = zstats(xs)
                self.stats[("test", task, sess)] = (m2.cpu().numpy(), s2.cpu().numpy())
                zs = (xs - m2) / s2
                test_z[sess] = (trow, zs)
                ls = (zs @ v - pc) @ w
                self.proj["test"][trow] = ls.cpu().numpy()
                self.tsim["test"][trow] = (F.normalize(ls, dim=1) @ F.normalize(centers, dim=1).T).cpu().numpy()

            # 相似度矩阵：split（训练前/后半）与 cross（训练 vs 测试 session2 预测分组）
            ha, hb = torch.zeros_like(means), torch.zeros_like(means)
            yl = y.cpu().numpy()
            for s_ in range(N_SUBJ):
                k = np.flatnonzero(yl == s_)
                h = len(k) // 2
                ha[s_] = zt[k[:h]].mean(0)
                hb[s_] = zt[k[h:]].mean(0)
            cross = torch.full_like(means, float("nan"))
            if "session2" in test_z:
                trow, zs = test_z["session2"]
                pl = torch.as_tensor(self.pred["test"][trow], device=DEVICE)
                for s_ in range(N_SUBJ):
                    m = pl == s_ + 1
                    if m.any():
                        cross[s_] = zs[m].mean(0)
            for dom, sl in DOMAINS.items():
                self.sim[(task, dom, "split")] = row_corr(ha[:, sl], hb[:, sl]).cpu().numpy()
                self.sim[(task, dom, "cross")] = row_corr(means[:, sl], cross[:, sl]).cpu().numpy()
            print(f"[fit] {task}: train={n}", flush=True)
            del xt, zt, p, test_z
        self.tmpl = {k: v.cpu().numpy() for k, v in self.tmpl.items()}
        for name in ("tmpl", "raw_mean", "fisher"):
            d = getattr(self, name)
            d["ALL"] = np.mean([d[t] for t in TASKS], axis=0)
        self.pop_mean["ALL"] = np.mean([self.pop_mean[t] for t in TASKS], axis=0)
        for dom in DOMAINS:
            for mode in ("split", "cross"):
                self.sim[("ALL", dom, mode)] = np.nanmean([self.sim[(t, dom, mode)] for t in TASKS], axis=0)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---------- 汇总 ----------
    def summary(self, scheme: str | None) -> dict:
        k = self.reg.key(scheme)
        if k not in self.summary_cache:
            self.summary_cache[k] = self._summary(self.reg.get(k))
        return self.summary_cache[k]

    def _summary(self, sch) -> dict:
        tr = self.meta["train"]
        y, p = self.label["train"], sch.pred["train"]
        has = p is not None
        p = p if has else np.full(len(y), -1)
        f1s = []
        for c in range(1, N_SUBJ + 1):
            tp = np.sum((p == c) & (y == c))
            f1s.append(2 * tp / max(1, 2 * tp + np.sum((p == c) & (y != c)) + np.sum((p != c) & (y == c))))
        acc = lambda m: float(np.mean(p[m] == y[m])) if has and m.any() else None  # noqa: E731
        te_pred = sch.pred["test"]
        subjects = []
        for s in range(1, N_SUBJ + 1):
            m = y == s
            row = {"subject": s, "n": int(m.sum()), "acc": acc(m), "tasks": {}}
            for t in TASKS:
                mt = m & (self.task["train"] == t)
                row["tasks"][t] = {"n": int(mt.sum()), "acc": acc(mt)}
            row["test_pred"] = int(np.sum(te_pred == s))
            subjects.append(row)
        return {
            "channels": self.channels,
            "tasks": list(TASKS),
            "bands": [f"{a}-{b}" for a, b in COV_BANDS],
            "freqs": [FREQ_MIN, FREQ_MAX],
            "fs": 300,
            "scheme": sch.info,
            "n_train": int(len(tr)),
            "n_test": int(len(self.meta["test"])),
            "oof_acc": acc(np.ones(len(y), bool)),
            "oof_macro_f1": float(np.mean(f1s)) if has else None,
            "oof_task_acc": {t: acc(self.task["train"] == t) for t in TASKS},
            "subjects": subjects,
            "test_pred_counts": {str(c): int(np.sum(te_pred == c)) for c in range(N_SUBJ + 1)},
            "test_sessions": {str(s): int(np.sum(self.session["test"] == s)) for s in np.unique(self.session["test"])},
            "lda_ratio": {t: jl(self.eig[t][:4], 4) for t in TASKS},
        }

    # ---------- 查询 ----------
    def _sp(self, sch, s: str, idx) -> tuple:
        """方案在 set s 上的 (预测, margin, 分数)；无 OOF / 无分数时预测为 -1、其余为 None。"""
        p = sch.pred[s]
        p = p[idx] if p is not None else np.full(np.shape(idx), -1)
        m, sc = sch.margin[s], sch.scores[s]
        return p, None if m is None else m[idx], None if sc is None else sc[idx]

    def rows(self, s: str, task: str, subject: int | None, session: str, a: str | None = None,
             b: str | None = None) -> dict:
        sa = self.reg.get(a)
        m = np.ones(len(self.meta[s]), bool)
        if task:
            m &= self.task[s] == task
        if subject is not None:
            m &= (self.label[s] if s == "train" else sa.pred[s]) == subject
        if session:
            m &= self.session[s] == session
        idx = np.flatnonzero(m)
        meta = self.meta[s]
        name = (meta["trial"].to_numpy()[idx].tolist() if s == "train" else meta["sample"].to_numpy()[idx].tolist())
        pa, ma, _ = self._sp(sa, s, idx)
        out = {
            "set": s, "scheme": sa.key, "idx": idx.tolist(), "name": name,
            "task": self.task[s][idx].tolist(), "session": self.session[s][idx].tolist(),
            "label": self.label[s][idx].tolist(), "pred": pa.tolist(),
            "margin": None if ma is None else jl(ma, 2),
        }
        if b:
            pb, mb, _ = self._sp(self.reg.get(b), s, idx)
            out.update(scheme_b=b, pred_b=pb.tolist(), margin_b=None if mb is None else jl(mb, 2))
        return out

    def trial(self, s: str, idx: int, a: str | None = None, b: str | None = None) -> dict:
        if not 0 <= idx < len(self.meta[s]):
            raise ValueError("idx 越界")
        m = self.meta[s].iloc[idx]
        data, info = load_eeg(DATA / m["path"])
        clipped, clip = robust_clip(data)
        filt = filtfilt(_NOTCH_B, _NOTCH_A, sosfiltfilt(_FULL_BAND_SOS, clipped, axis=0), axis=0)
        raw = np.nan_to_num(data)
        raw = raw - raw.mean(0)
        feat = np.asarray(self.x[s][idx], dtype=np.float32)
        mu, sd = self.stats[self._group(s, idx)]
        z = (feat - mu) / sd
        logpsd, cov = split_feat(feat)
        psd_z, cov_z = split_feat(z)
        sa = self.reg.get(a)
        pa, ma, sca = self._sp(sa, s, idx)
        out = {
            "set": s, "idx": idx, "path": str(m["path"]), "task": str(self.task[s][idx]),
            "session": str(self.session[s][idx]),
            "name": int(m["trial"]) if s == "train" else str(m["sample"]),
            "label": int(self.label[s][idx]), "scheme": sa.key, "pred": int(pa),
            "margin": None if ma is None else float(ma),
            "unit": "device_native",
            "raw": jl(raw.T, 2), "filt": jl(filt.T, 2),
            "std": jl(filt.std(0), 3), "clip": jl(clip, 4),
            "logpsd": jl(logpsd, 3), "psd_z": jl(psd_z, 2),
            "cov": jl(cov, 3), "cov_z": jl(cov_z, 2),
            "scores": None if sca is None else jl(sca, 3), "tsim": jl(self.tsim[s][idx], 3),
            "emb": jl(self.proj[s][idx][:2], 3),
        }
        if s == "test" and idx in self.pair:
            j, sim = self.pair[idx]
            out["pair"] = {"idx": int(j), "name": str(self.meta["test"]["sample"].iloc[j]), "similarity": round(sim, 5),
                           "pred": int(sa.pred["test"][j])}
        if b:
            pb, mb, scb = self._sp(self.reg.get(b), s, idx)
            out.update(scheme_b=b, pred_b=int(pb), margin_b=None if mb is None else float(mb),
                       scores_b=None if scb is None else jl(scb, 3))
        return out

    def embed(self, task: str) -> dict:
        tr = np.flatnonzero(self.task["train"] == task)
        te = np.flatnonzero(self.task["test"] == task)
        pt, pe = self.proj["train"][tr], self.proj["test"][te]
        return {
            "task": task, "ratio": jl(self.eig[task][:2], 4),
            "train": {"idx": tr.tolist(), "x": jl(pt[:, 0]), "y": jl(pt[:, 1]), "s": self.label["train"][tr].tolist(),
                      "p": self.pred["train"][tr].tolist()},
            "test": {"idx": te.tolist(), "x": jl(pe[:, 0]), "y": jl(pe[:, 1]), "p": self.pred["test"][te].tolist(),
                     "s1": (self.session["test"][te] == "session1").astype(int).tolist()},
            "centers": jl(self.centers[task][:, :2]),
        }

    def fingerprint(self, subject: int, task: str) -> dict:
        k = subject - 1
        psd_z, cov_z = split_feat(self.tmpl[task][k])
        mean_psd, mean_cov = split_feat(self.raw_mean[task][k])
        pop_psd, _ = split_feat(self.pop_mean[task])
        fpsd, fcov = split_feat(self.fisher[task])
        # 当前任务下该被试与其他被试模板的距离（越大越独特）
        t = self.tmpl[task]
        dist = np.linalg.norm(t - t[k], axis=1) / np.sqrt(t.shape[1])
        return {
            "subject": subject, "task": task,
            "psd_z": jl(psd_z, 3), "cov_z": jl(cov_z, 3),
            "mean_psd": jl(mean_psd, 3), "mean_cov": jl(mean_cov, 3), "pop_psd": jl(pop_psd, 3),
            "fisher_psd": jl(fpsd, 3), "fisher_cov": jl(fcov, 3),
            "dist": jl(dist, 3),
        }

    def similarity(self, task: str, domain: str, mode: str) -> dict:
        m = self.sim[(task, domain, mode)]
        ok = np.isfinite(m).all(1)
        hit = int(np.sum((np.nanargmax(np.where(np.isfinite(m), m, -9), 1) == np.arange(N_SUBJ)) & ok))
        return {"task": task, "domain": domain, "mode": mode, "matrix": jl(m, 4), "top1": hit, "valid": int(ok.sum())}


STORE: Store | None = None


def _task(q, allow_all=True) -> str:
    t = q.get("task", ["ALL" if allow_all else "EC"])[0]
    if t not in TASKS and not (allow_all and t == "ALL"):
        raise ValueError(f"非法任务 {t}")
    return t


def _q(q, k: str) -> str | None:
    v = q.get(k, [""])[0]
    if v and STORE is not None:
        STORE.reg.key(v)  # 非法方案立即报错
    return v or None


class Handler(BaseHTTPRequestHandler):
    server_version = "BrainPrintViewer/1.0"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        try:
            if path in ("/", "/index.html", "/viewer.html"):
                self._send(200, VIEWER.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/summary":
                self._json(STORE.summary(_q(q, "scheme")))
            elif path == "/api/schemes":
                self._json({"default": DEFAULT, "schemes": STORE.reg.list()})
            elif path == "/api/metrics":
                self._json(STORE.reg.metrics(_q(q, "scheme"), _q(q, "ref")))
            elif path == "/api/compare":
                self._json(STORE.reg.compare(_q(q, "a"), _q(q, "b")))
            elif path == "/api/rows":
                s = q.get("set", ["train"])[0]
                if s not in SETS:
                    raise ValueError("非法 set")
                task = q.get("task", [""])[0]
                if task and task not in TASKS:
                    raise ValueError("非法任务")
                sub = q.get("subject", [""])[0]
                self._json(STORE.rows(s, task, int(sub) if sub != "" else None, q.get("session", [""])[0],
                                      _q(q, "scheme"), _q(q, "b")))
            elif path.startswith("/api/trial/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[3] not in SETS:
                    raise ValueError("用法 /api/trial/<train|test>/<idx>")
                self._json(STORE.trial(parts[3], int(parts[4]), _q(q, "scheme"), _q(q, "b")))
            elif path == "/api/embed":
                self._json(STORE.embed(_task(q, allow_all=False)))
            elif path == "/api/fingerprint":
                sub = int(q.get("subject", ["1"])[0])
                if not 1 <= sub <= N_SUBJ:
                    raise ValueError("subject 应为 1–20")
                self._json(STORE.fingerprint(sub, _task(q)))
            elif path == "/api/similarity":
                dom = q.get("domain", ["all"])[0]
                mode = q.get("mode", ["split"])[0]
                if dom not in DOMAINS or mode not in ("split", "cross"):
                    raise ValueError("非法 domain / mode")
                self._json(STORE.similarity(_task(q), dom, mode))
            else:
                self._json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    global STORE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    STORE = Store()
    url = f"http://{args.host}:{args.port}/"
    print(f"BrainPrint 查看器：{url}  （Ctrl+C 退出）", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
