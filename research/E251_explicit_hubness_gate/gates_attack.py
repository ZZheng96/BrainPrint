"""E251 white-box PGD gate attack using the full Train-only deployment fit."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E247_train_temporal_loso"))
import run_e247 as R
sys.path.insert(0, str(OUT))
import run_e251 as E251
DEV, DIM = R.DEV, R.DIM


class SPDLog(torch.autograd.Function):
    """Symmetric matrix log whose backward stays finite under repeated eigenvalues.

    torch.linalg.eigh backward divides by eigenvalue gaps and returns NaN when two
    eigenvalues coincide, which is common for 18-channel covariance matrices.  The
    Daleckii-Krein form below uses log1p(d/a)/d and falls back to 1/a at d -> 0.
    """

    @staticmethod
    def forward(ctx, x, floor):
        val, vec = torch.linalg.eigh(x)
        val = val.clamp_min(floor)
        ctx.save_for_backward(val, vec)
        return (vec * val.log().unsqueeze(-2)) @ vec.transpose(-1, -2)

    @staticmethod
    def backward(ctx, dy):
        val, vec = ctx.saved_tensors
        dy = 0.5 * (dy + dy.transpose(-1, -2))
        inner = vec.transpose(-1, -2) @ dy @ vec
        a = val.unsqueeze(-1)
        diff = val.unsqueeze(-2) - a
        safe_a = a.clamp_min(1e-6)
        close = diff.abs() <= 1e-6 * safe_a
        safe_diff = torch.where(close, torch.ones_like(diff), diff)
        ratio = torch.where(close, 1.0 / safe_a, torch.log1p(diff / safe_a) / safe_diff)
        dx = vec @ (ratio * inner) @ vec.transpose(-1, -2)
        return dx, None


def features(x, cfg):
    length = int(cfg["subwindow_length"]); channels = int(cfg["channels"])
    freq = torch.fft.rfftfreq(length, d=1.0 / float(cfg["sampling_rate_hz"]), device=x.device)
    masks = []
    for i, (low, high) in enumerate(cfg["bands_hz"]):
        masks.append((freq >= low) & (freq <= high if i == len(cfg["bands_hz"]) - 1 else freq < high))
    mask = torch.stack(masks).to(x.dtype); n_sub = int(cfg["n_subwindows"]); parts = []
    for w in range(n_sub):
        y = x[..., w * length:(w + 1) * length]
        spectrum = torch.fft.rfft(y, dim=-1)
        filtered = torch.fft.irfft(spectrum[:, None] * mask[None, :, None, :], n=length, dim=-1)
        filtered = filtered - filtered.mean(dim=-1, keepdim=True)
        scale = filtered.std(dim=-1, keepdim=True, unbiased=int(cfg["channel_std_ddof"]) == 1).clamp_min(float(cfg["channel_std_epsilon"]))
        norm = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
        corr = norm @ norm.transpose(-1, -2) / (length - int(cfg["channel_std_ddof"]))
        tr = torch.diagonal(corr, dim1=-2, dim2=-1).sum(-1); eye = torch.eye(channels, dtype=x.dtype, device=x.device)
        shrink = float(cfg["covariance_shrinkage"]); spd = (1-shrink)*corr + shrink*tr[...,None,None]/channels*eye
        logspd = SPDLog.apply(spd, float(cfg["eigenvalue_floor"]))
        rows, cols = torch.triu_indices(channels, channels, device=x.device); diag = rows == cols
        v = logspd[..., rows, cols]; v = torch.where(diag[None,None], v, v * float(cfg["off_diagonal_multiplier"]))
        v = v.reshape(len(v), len(masks), -1); dv = v[..., diag]; v = v.index_copy(-1, torch.nonzero(diag).flatten(), dv-dv.mean(-1,keepdim=True))
        mean = v.mean(-1,keepdim=True); sd = v.std(-1,keepdim=True,unbiased=False).clamp_min(float(cfg["zband_epsilon"]))
        parts.append(((v-mean)/sd).flatten(-2))
    return torch.stack(parts, dim=1).mean(dim=1)
SNRS, STEPS, SEED, CAP = (30, 20), 20, 20261115, 400


def load_mat(path):
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    assert data.shape == (1200, 18)
    return np.ascontiguousarray(data.T)


def fit_deployment(feats, tasks, subj, frac):
    known = np.arange(1, 21)
    fit = np.isin(subj, known) & (frac <= .6)
    records = []
    for held in known:
        remain = known[known != held]
        ifit = np.isin(subj, remain) & (frac <= .4)
        fitted, P, W = E251.fit_base(feats, tasks, subj, ifit, remain)
        iu, pu = E251.score_features(P, np.flatnonzero((subj == held) & (frac > .4) & (frac <= .6)), fitted, W, tasks)
        ik, pk = E251.score_features(P, np.flatnonzero(np.isin(subj, remain) & (frac > .4) & (frac <= .6)), fitted, W, tasks)
        records.append({"unknown_raw": iu, "known_raw": ik, "unknown_pred": pu, "known_pred": pk})
    stats = E251.candidate_stats(records, subj, 0)
    chosen = E251.choose_lambda(stats)
    fitted, P, W = E251.fit_base(feats, tasks, subj, fit, known)
    return fitted, W, chosen, stats


class Scorer(torch.nn.Module):
    def __init__(self, art, fitted, W):
        super().__init__()
        self.cfg = art["feature"]
        s = fitted["space"]
        self.mu, self.sd, self.pm, self.pb = s.mu, s.sd, s.pm, s.pb
        self.W = W
        self.protos = F.normalize(fitted["protos"] @ W, dim=2)
        self.valid = fitted["valid"]
        self.hub_median, self.hub_scale = fitted["hub_median"], fitted["hub_scale"]

    def forward(self, x, task_idx):
        f = features(x, self.cfg)
        z = torch.empty_like(f)
        for t in range(4):
            m = task_idx == t
            if m.any(): z[m] = (f[m] - self.mu[t]) / self.sd[t]
        p = F.normalize(((z - self.pm) @ self.pb) @ self.W, dim=1)
        proto = self.protos
        mat = torch.einsum("nd,tsd->nts", p, proto)
        mat[:, :, ~self.valid] = -1e9
        logits = mat[torch.arange(len(x), device=DEV), task_idx]
        vals, ids = logits.topk(2, dim=1)
        mean_proto = proto[:, self.valid].mean(0)
        ps = mean_proto @ mean_proto.T
        ps.fill_diagonal_(-float("inf"))
        hv = torch.bincount(ps.argmax(1), minlength=int(self.valid.sum())).double()
        all_h = torch.zeros(20, dtype=torch.float64, device=DEV)
        all_h[torch.nonzero(self.valid, as_tuple=False).flatten()] = hv
        hub = (all_h[ids[:, 0]].cpu().numpy() - self.hub_median) / self.hub_scale
        return vals, ids[:, 0] + 1, hub


def pgd(scorer, x, task_idx, true_idx, target_idx, goal, snr, lam):
    gen = torch.Generator(device=DEV).manual_seed(SEED + 1000 * int(snr) + ord(goal))
    xc = x - x.mean(dim=-1, keepdim=True)
    radius = xc.reshape(len(x), -1).norm(dim=1).clamp_min(1e-12) / (10.0 ** (float(snr) / 20.0))
    delta = torch.randn(x.shape, generator=gen, device=DEV)
    delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta *= radius.view(-1, 1, 1) * torch.rand(len(delta), generator=gen, device=DEV).view(-1, 1, 1)
    alpha = 2.0 * radius / STEPS
    for _ in range(STEPS):
        delta = delta.detach().requires_grad_(True)
        vals, ids, hub = scorer(x + delta, task_idx)
        gate = vals[:, 0] - lam * torch.as_tensor(hub, device=DEV, dtype=vals.dtype)
        if goal == "D":
            # E195 convention: gate evasion maximises the refusal gate.
            objective = gate
        else:
            f = features(x + delta, scorer.cfg); z = torch.empty_like(f)
            for t in range(4):
                m = task_idx == t
                if m.any(): z[m] = (f[m] - scorer.mu[t]) / scorer.sd[t]
            p = F.normalize(((z - scorer.pm) @ scorer.pb) @ scorer.W, dim=1)
            mat = torch.einsum("nd,tsd->nts", p, scorer.protos); mat[:, :, ~scorer.valid] = -1e9
            logits = mat[torch.arange(len(x), device=DEV), task_idx]
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(F.one_hot(true_idx, 20).bool(), -1e9)
            margin = wrong.max(1).values if goal == "A" else (torch.logsumexp(wrong, 1) if goal == "B" else logits.gather(1, target_idx[:, None]).squeeze(1))
            objective = (margin - true_logit) / 10.0
        grad = torch.nan_to_num(torch.autograd.grad(objective.sum(), delta)[0],
                                nan=0.0, posinf=0.0, neginf=0.0)
        direction = grad.div(grad.reshape(len(grad), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12))
        delta = delta.detach() + alpha.view(-1, 1, 1) * direction
        norm = delta.reshape(len(delta), -1).norm(dim=1)
        delta *= torch.minimum(torch.ones_like(norm), radius / norm).view(-1,1,1)
    with torch.no_grad():
        vals, ids, hub = scorer(x + delta, task_idx)
        return vals.cpu().numpy(), ids.cpu().numpy(), (vals[:,0].cpu().numpy() - lam * hub)


def metrics(vals, pred, gate, truth, threshold):
    accepted = gate >= threshold; out_pred = np.where(accepted, pred, 0)
    return {"RA": float(np.mean(out_pred == truth)), "ASafeR": float(np.mean((out_pred == truth) | (out_pred == 0))), "accept_rate": float(accepted.mean()), "closed_set_accuracy": float(np.mean(pred == truth)), "wrong_identity_rate": float(np.mean((out_pred != 0) & (out_pred != truth)))}

def main():
    meta = pd.read_csv(R.META_PATH); feats = torch.as_tensor(np.array(np.load(R.FEATURE_PATH, mmap_mode="r"), dtype=np.float32, copy=True), device=DEV)
    R.validate(meta, feats); _, frac = R.temporal_blocks(meta); subj = meta.subject.to_numpy(); tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))
    fitted, W, lam, stats = fit_deployment(feats, tasks, subj, frac)
    art = R.M.load_model(str(ROOT / "final" / "team_hymn"), str(DEV)); scorer = Scorer(art, fitted, W).to(DEV).eval()
    threshold = stats[lam]["threshold"]
    window = (frac > .6) & (frac <= .8); rng = np.random.default_rng(SEED); sel = []
    for t in range(4):
        rows = np.flatnonzero(window & (tasks == t)); sel.append(rows if len(rows) <= CAP else np.sort(rng.choice(rows, CAP, replace=False)))
    subset = np.sort(np.concatenate(sel)); truth = subj[subset]; buf = {(s,g): [] for s in SNRS for g in "ABCD"}; clean_parts=[]; t0=time.time()
    for start in range(0, len(subset), 128):
        rows = subset[start:start+128]; x=torch.stack([torch.as_tensor(load_mat(ROOT / "Brainprint_Competition" / meta.path.iloc[i])) for i in rows]).to(DEV); ti=torch.as_tensor(tasks[rows],device=DEV); tri=torch.as_tensor(truth[start:start+128]-1,device=DEV); target=(tri+1)%20
        with torch.no_grad(): v,p,h=scorer(x,ti); g=v[:,0].cpu().numpy()-lam*h
        clean_parts.append((v[:,0].cpu().numpy(),p.cpu().numpy(),g))
        for snr in SNRS:
            for goal in "ABCD": buf[(snr,goal)].append(pgd(scorer,x,ti,tri,target,goal,snr,lam))
    clean_v=np.concatenate([a for a,_,_ in clean_parts]); clean_p=np.concatenate([a for _,a,_ in clean_parts]); clean_g=np.concatenate([a for _,_,a in clean_parts]); clean=metrics(clean_v,clean_p,clean_g,truth,threshold)
    conditions=[]
    for (snr,goal), parts in buf.items():
        p=np.concatenate([x[1] for x in parts]); g=np.concatenate([x[2] for x in parts]); v=np.concatenate([x[0][:,0] for x in parts]); m=metrics(v,p,g,truth,threshold); m.update({"snr":snr,"goal":goal}); conditions.append(m)
    wr=min(conditions,key=lambda x:x["RA"]); ws=min(conditions,key=lambda x:x["ASafeR"])
    c30=[c for c in conditions if c["snr"]==30]
    wr30=min(c30,key=lambda x:x["RA"]); ws30=min(c30,key=lambda x:x["ASafeR"])
    out={"deployment":"full Train known identities, frac<=0.6; lambda/threshold from Train inner LOSO","candidate":f"lambda_{lam:.2f}","lambda":lam,"threshold":threshold,"n_trials":int(len(subset)),"per_task":{str(t):int((tasks[subset]==t).sum()) for t in range(4)},"clean":clean,"conditions":conditions,"RA":wr["RA"],"ASafeR":ws["ASafeR"],"worst_RA":wr,"worst_ASafeR":ws,"RA_snr30":wr30["RA"],"ASafeR_snr30":ws30["ASafeR"],"worst_RA_snr30":wr30,"worst_ASafeR_snr30":ws30,"protocol":{"subset":"Train 0.6<frac<=0.8, cap 400/task, seed 20261115","snr_db":list(SNRS),"steps":STEPS,"goals":"A/B/C maximise wrong-vs-true identity margin; D maximises the refusal gate (E195 gate-evasion convention)","radius":"demeaned waveform L2 / 10^(SNR/20)","primary_snr_db":30,"snr20":"reference only","gradient":"Daleckii-Krein SPD matrix-log backward; eigh NaN removed","device":str(DEV)},"seconds":time.time()-t0}
    (OUT/"results"/"gates_attack.json").write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps({k:out[k] for k in ("candidate","threshold","n_trials","clean","RA","ASafeR","worst_RA","worst_ASafeR")},indent=2))


if __name__ == "__main__":
    main()
