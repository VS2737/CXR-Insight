"""RAD-DINO chest-radiograph feature extraction for research-only v8.

This is a pretrained image encoder, not a diagnostic classifier. Image data is
processed in memory; only study-split feature caches are written by CLI scripts.
"""
from __future__ import annotations
import math
import numpy as np

RAD_DINO_REPO = 'microsoft/rad-dino'


def tokens_from_hidden(hidden, grid: int = 7):
    """Return CLS vector and pooled patch tokens from a DINOv2 forward pass.

    The model's [CLS] token is not a patch. Refuse unknown token layouts rather
    than accidentally pooling registers as spatial locations.
    """
    import torch
    import torch.nn.functional as F
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError('RAD-DINO hidden states must be [1,tokens,features]')
    length = hidden.shape[1] - 1
    side = math.isqrt(length)
    if side * side != length:
        raise ValueError('Unsupported visual token layout (possible register tokens)')
    if length < grid * grid:
        raise ValueError('Patch grid is too small for selected pooling size')
    cls = F.normalize(hidden[0, 0].float(), dim=0)
    patches = hidden[:, 1:, :].transpose(1, 2).reshape(1, hidden.shape[-1], side, side)
    patches = F.adaptive_avg_pool2d(patches.float(), (grid, grid))
    patches = patches.flatten(2).transpose(1, 2).squeeze(0)
    return cls, patches


class RadDinoEncoder:
    def __init__(self, device: str = 'cpu', repo: str = RAD_DINO_REPO, grid: int = 7):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        if grid < 2 or grid > 14:
            raise ValueError('grid must be between 2 and 14')
        self.device = device
        self.grid = grid
        self.repo = repo
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode(self, image):
        import torch
        if image.mode != 'RGB':
            image = image.convert('RGB')
        batch = self.processor(images=image, return_tensors='pt')
        px = batch['pixel_values'].to(self.device)
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.startswith('cuda')):
                hidden = self.model(pixel_values=px).last_hidden_state
            cls, tokens = tokens_from_hidden(hidden.float(), self.grid)
        return cls.cpu().numpy(), tokens.cpu().numpy()


def unit_rows(x):
    z = np.asarray(x, dtype=np.float32)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('Expected finite matrix of feature vectors')
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, 1e-8)
