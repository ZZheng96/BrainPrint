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
  GET /api/summary                        -> 被试 / 任务统计、OOF 指标、测试预测分布、通道名
  GET /api/rows?set=&task=&subject=&session=
                                          -> trial 列表（训练集按真实标签筛选，测试集按预测标签）
  GET /api/trial/<set>/<idx>              -> 波形（原始 + 0.5–90 Hz/50 Hz 陷波滤波）、PSD、协方差、分数
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

DATA = ROOT / "Brainprint_Competition"
CACHE = next(
    (p for p in (ROOT / "BrainPrint_cache", ROOT.parent / "BrainPrint_cache") if p.exists()),
    ROOT / "BrainPrint_cache",
)
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
    def __init__(self, test_scores: str, test_labels: Path, train_scores: str, train_preds: str):
        t0 = time.time()
        self.meta = {s: pd.read_csv(CACHE / f"{s}_metadata.csv") for s in SETS}
        self.x = {s: np.load(CACHE / f"{s}_features.npy", mmap_mode="r") for s in SETS}
        tr, te = self.meta["train"], self.meta["test"]
        self.scheme = {"test_scores": test_scores, "test_labels": test_labels.name,
                       "train_scores": train_scores, "train_preds": train_preds}
        self.scores = {"train": np.load(CACHE / train_scores), "test": np.load(CACHE / test_scores)}
        oof = pd.read_csv(CACHE / train_preds).set_index("path")["predicted_label"]
        sub = pd.read_csv(test_labels).set_index("Sample")["Label"]
        self.pred = {"train": oof.loc[tr["path"]].to_numpy(int), "test": sub.loc[te["sample"]].to_numpy(int)}
        self.label = {"train": tr["subject"].to_numpy(int), "test": np.full(len(te), -1)}
        self.task = {s: self.meta[s]["task"].to_numpy() for s in SETS}
        self.session = {s: self.meta[s]["session"].to_numpy() for s in SETS}
        self.margin = {}
        for s in SETS:
            top = np.sort(self.scores[s], axis=1)[:, -2:]
            self.margin[s] = top[:, 1] - top[:, 0]
        pairs = pd.read_csv(CACHE / "waveform_pairs.csv")
        row = dict(zip(te["sample"], range(len(te))))
        self.pair = {row[a]: (row[b], float(c)) for a, b, c in
                     zip(pairs["sample"], pairs["nearest_sample"], pairs["similarity"])}
        first = loadmat(DATA / te["path"].iloc[0])
        self.channels = [str(c.reshape(-1)[0]) for c in first["channel_names"].reshape(-1)]
        self._fit()
        self.summary_cache = self._summary()
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
    def _summary(self) -> dict:
        tr = self.meta["train"]
        y, p = self.label["train"], self.pred["train"]
        f1s = []
        for c in range(1, N_SUBJ + 1):
            tp = np.sum((p == c) & (y == c))
            f1s.append(2 * tp / max(1, 2 * tp + np.sum((p == c) & (y != c)) + np.sum((p != c) & (y == c))))
        subjects = []
        for s in range(1, N_SUBJ + 1):
            m = y == s
            row = {"subject": s, "n": int(m.sum()), "acc": float(np.mean(p[m] == s)), "tasks": {}}
            for t in TASKS:
                mt = m & (self.task["train"] == t)
                row["tasks"][t] = {"n": int(mt.sum()), "acc": float(np.mean(p[mt] == s)) if mt.any() else None}
            row["test_pred"] = int(np.sum(self.pred["test"] == s))
            subjects.append(row)
        te_pred = self.pred["test"]
        return {
            "channels": self.channels,
            "tasks": list(TASKS),
            "bands": [f"{a}-{b}" for a, b in COV_BANDS],
            "freqs": [FREQ_MIN, FREQ_MAX],
            "fs": 300,
            "scheme": self.scheme,
            "n_train": int(len(tr)),
            "n_test": int(len(self.meta["test"])),
            "oof_acc": float(np.mean(p == y)),
            "oof_macro_f1": float(np.mean(f1s)),
            "oof_task_acc": {t: float(np.mean(p[self.task["train"] == t] == y[self.task["train"] == t])) for t in TASKS},
            "subjects": subjects,
            "test_pred_counts": {str(c): int(np.sum(te_pred == c)) for c in range(N_SUBJ + 1)},
            "test_sessions": {str(s): int(np.sum(self.session["test"] == s)) for s in np.unique(self.session["test"])},
            "lda_ratio": {t: jl(self.eig[t][:4], 4) for t in TASKS},
        }

    # ---------- 查询 ----------
    def rows(self, s: str, task: str, subject: int | None, session: str) -> dict:
        m = np.ones(len(self.meta[s]), bool)
        if task:
            m &= self.task[s] == task
        if subject is not None:
            m &= (self.label[s] if s == "train" else self.pred[s]) == subject
        if session:
            m &= self.session[s] == session
        idx = np.flatnonzero(m)
        meta = self.meta[s]
        name = (meta["trial"].to_numpy()[idx].tolist() if s == "train" else meta["sample"].to_numpy()[idx].tolist())
        return {
            "set": s, "idx": idx.tolist(), "name": name,
            "task": self.task[s][idx].tolist(), "session": self.session[s][idx].tolist(),
            "label": self.label[s][idx].tolist(), "pred": self.pred[s][idx].tolist(),
            "margin": jl(self.margin[s][idx], 2),
        }

    def trial(self, s: str, idx: int) -> dict:
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
        out = {
            "set": s, "idx": idx, "path": str(m["path"]), "task": str(self.task[s][idx]),
            "session": str(self.session[s][idx]),
            "name": int(m["trial"]) if s == "train" else str(m["sample"]),
            "label": int(self.label[s][idx]), "pred": int(self.pred[s][idx]),
            "margin": float(self.margin[s][idx]),
            "unit": "device_native",
            "raw": jl(raw.T, 2), "filt": jl(filt.T, 2),
            "std": jl(filt.std(0), 3), "clip": jl(clip, 4),
            "logpsd": jl(logpsd, 3), "psd_z": jl(psd_z, 2),
            "cov": jl(cov, 3), "cov_z": jl(cov_z, 2),
            "scores": jl(self.scores[s][idx], 3), "tsim": jl(self.tsim[s][idx], 3),
            "emb": jl(self.proj[s][idx][:2], 3),
        }
        if s == "test" and idx in self.pair:
            j, sim = self.pair[idx]
            out["pair"] = {"idx": int(j), "name": str(self.meta["test"]["sample"].iloc[j]), "similarity": round(sim, 5),
                           "pred": int(self.pred["test"][j])}
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
                self._json(STORE.summary_cache)
            elif path == "/api/rows":
                s = q.get("set", ["train"])[0]
                if s not in SETS:
                    raise ValueError("非法 set")
                task = q.get("task", [""])[0]
                if task and task not in TASKS:
                    raise ValueError("非法任务")
                sub = q.get("subject", [""])[0]
                self._json(STORE.rows(s, task, int(sub) if sub != "" else None, q.get("session", [""])[0]))
            elif path.startswith("/api/trial/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[3] not in SETS:
                    raise ValueError("用法 /api/trial/<train|test>/<idx>")
                self._json(STORE.trial(parts[3], int(parts[4])))
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
    ap.add_argument("--test-scores", default="scores_task_specific_zscore_grouped_guarded.npy")
    ap.add_argument("--test-labels", type=Path,
                    default=ROOT / "commit" / "answer" / "submission_task_specific_zscore_grouped_guarded.csv")
    ap.add_argument("--train-scores", default="train_blocked_oof_scores_task_specific_guarded.npy")
    ap.add_argument("--train-preds", default="train_blocked_oof_task_specific_guarded_predictions.csv")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    STORE = Store(args.test_scores, args.test_labels, args.train_scores, args.train_preds)
    url = f"http://{args.host}:{args.port}/"
    print(f"BrainPrint 查看器：{url}  （Ctrl+C 退出）", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
