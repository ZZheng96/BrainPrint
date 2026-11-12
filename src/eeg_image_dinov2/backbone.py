"""冻结 DINOv2 骨干特征提取。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDINOv2(nn.Module):
    """冻结的 DINOv2 ViT-B/14，输出 CLS 与 patch token 特征。"""

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m", img_size: int = 224):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=img_size
        )
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.embed_dim = self.backbone.embed_dim
        self.img_size = img_size
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: torch.Tensor, return_tokens: bool = False):
        """images: (B,3,H,W) 值域 [0,1]。返回 CLS (B,D)，可选 patch tokens (B,N,D)。"""
        normalized = (images - self.mean) / self.std
        with torch.no_grad():
            tokens = self.backbone.forward_features(normalized)
        cls = self.backbone.forward_head(tokens, pre_logits=True)
        if return_tokens:
            return cls, tokens[:, 1:, :]
        return cls


@torch.no_grad()
def extract_features(
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
    img_size: int = 224,
    return_tokens: bool = False,
) -> np.ndarray:
    """批量提取冻结特征。images: (N,3,H,W) float32 [0,1]。"""
    model = FrozenDINOv2(model_name, img_size).to(device)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.as_tensor(images[start : start + batch_size], device=device)
        outputs.append(model(batch, return_tokens=return_tokens).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs).astype(np.float32)
