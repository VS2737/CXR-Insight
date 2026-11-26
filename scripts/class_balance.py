"""Train-only class balancing with explicit missing-label masking."""
import torch

def training_pos_weights(items, cap=5.0):
    """Train-only class balancing; ignore uncertain/unmentioned labels.

    Returns an elementwise positive weight per finding. Never includes val/test.
    """
    from radreport.config import FINDINGS
    from radreport.label_values import parse_binary_chexpert
    positives=torch.zeros(len(FINDINGS),dtype=torch.float32)
    negatives=torch.zeros(len(FINDINGS),dtype=torch.float32)
    for _, row in items:
        for j, name in enumerate(FINDINGS):
            y=parse_binary_chexpert(row.get(name,''))
            if y == 1: positives[j]+=1
            elif y == 0: negatives[j]+=1
    weights=torch.ones_like(positives)
    applicable=(positives > 0)&(negatives > 0)
    weights[applicable]=(negatives[applicable]/positives[applicable]).clamp(min=1.0,max=cap)
    return weights

