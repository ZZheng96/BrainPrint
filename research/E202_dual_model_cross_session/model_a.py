from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class IdentityEmbedding(nn.Module):
    def __init__(self, e65_dim: int, e162_dim: int, n_classes: int, out_dim: int = 128):
        super().__init__()
        self.e65 = nn.Sequential(nn.Linear(e65_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.e162 = nn.Sequential(nn.Linear(e162_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(192, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, out_dim))
        self.cls = nn.Linear(out_dim, n_classes)

    def forward(self, x65, x162):
        z = F.normalize(self.fuse(torch.cat([self.e65(x65), self.e162(x162)], dim=1)), dim=1)
        return z, self.cls(z)

def supervised_contrastive(z, y, temperature=0.15):
    sim = z @ z.T / temperature
    mask = torch.eye(len(y), device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & ~mask
    logits = sim.masked_fill(mask, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = pos.sum(1).clamp_min(1)
    return -(logp.masked_fill(~pos, 0).sum(1) / count).mean()

def train_model(x65, x162, y, n_classes, device='cuda', epochs=30, batch_size=256, lr=1e-3, seed=20261116):
    torch.manual_seed(seed)
    model = IdentityEmbedding(x65.shape[1], x162.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x65 = torch.as_tensor(x65, dtype=torch.float32, device=device)
    x162 = torch.as_tensor(x162, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(y)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = order[start:start+batch_size]
            z, logits = model(x65[idx], x162[idx])
            loss = F.cross_entropy(logits, y[idx]) + 0.5 * supervised_contrastive(z, y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z, logits = model(x65, x162)
    return model, z.detach().cpu().numpy(), logits.detach().cpu().numpy()

def embed(model, x65, x162, device='cuda', batch_size=2048):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(x65), batch_size):
            a=torch.as_tensor(x65[i:i+batch_size],dtype=torch.float32,device=device)
            b=torch.as_tensor(x162[i:i+batch_size],dtype=torch.float32,device=device)
            out.append(model(a,b)[0].cpu().numpy())
    import numpy as np
    return np.concatenate(out)
