from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess(raw: torch.Tensor, channels: int) -> torch.Tensor:
    if raw.ndim != 3 or tuple(raw.shape[1:]) != (channels, 1200):
        raise ValueError(f"Expected Bx{channels}x1200, got {tuple(raw.shape)}")
    x = raw.float(); t = torch.linspace(-1, 1, 1200, device=x.device, dtype=x.dtype)
    x = x - x.mean(-1, keepdim=True)
    x = x - ((x*t).sum(-1, keepdim=True)/(t.square().sum()+1e-12))*t
    x = x - x.median(-1, keepdim=True).values
    mad = 1.4826*x.abs().median(-1, keepdim=True).values
    rms = x.square().mean(-1, keepdim=True).sqrt()
    return torch.nan_to_num(x/torch.where(mad > 1e-8, mad, rms).clamp_min(1e-8)).unsqueeze(1)

class E88Spatial18Embedding(nn.Module):
    def __init__(self, config):
        super().__init__(); c=int(config['data']['channels_18']); f=int(config['model']['branch_filters'])
        self.branches=nn.ModuleList([self._branch(c,f,k) for k in (33,65,129)])
        self.depth=nn.Conv2d(f*3,f*3,(1,15),padding=(0,7),groups=f*3,bias=False)
        self.point=nn.Conv2d(f*3,32,1,bias=False); self.bn=nn.BatchNorm2d(32)
        self.projection=nn.Linear(32,int(config['model']['embedding_dim'])); self.dropout=nn.Dropout(float(config['model']['dropout']))
    @staticmethod
    def _branch(c,f,k):
        return nn.Sequential(nn.Conv2d(1,f,(1,k),padding=(0,k//2),bias=False),nn.BatchNorm2d(f),nn.Conv2d(f,f,(c,1),groups=f,bias=False),nn.BatchNorm2d(f),nn.AvgPool2d((1,4)))
    def embed(self, raw):
        x=preprocess(raw,18); x=torch.cat([b(x) for b in self.branches],1)
        x=self.point(self.depth(x)); x=self.dropout(F.elu(self.bn(x)))
        return F.normalize(self.projection(F.adaptive_avg_pool2d(x,(1,1)).flatten(1)),dim=1,eps=1e-12)
