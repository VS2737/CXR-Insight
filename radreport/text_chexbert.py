"""Optional 14-class CheXbert text-label adapter for CXR Insight.

Enable with CXR_USE_CHEXBERT=1 after installing f1chexbert and the checkpoint.
The original model uses 0=blank,1=positive,2=negative,3=uncertain for 13 heads.
No Finding is binary. No inference on raw pixels occurs here.
"""
from functools import lru_cache

HEAD_ORDER = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices', 'No Finding',
]
STATES = {0:'NOT_MENTIONED',1:'PRESENT',2:'ABSENT',3:'UNCERTAIN'}

@lru_cache(maxsize=1)
def _load():
    from f1chexbert import F1CheXbert
    model = F1CheXbert(device='cpu')  # avoid occupying the RTX 4060 VRAM
    model.eval()
    return model

def chexbert_claims(text: str) -> dict:
    import torch
    from .config import FINDINGS
    labeler = _load()
    enc = labeler.tokenizer(text or ' ', truncation=True, max_length=512, return_tensors='pt')
    ids = enc['input_ids']
    with torch.inference_mode():
        if hasattr(labeler.model, 'cls_logits'):
            logits = labeler.model.cls_logits(ids)
        else:
            mask = enc.get('attention_mask', torch.ones_like(ids))
            logits = labeler.model(ids, mask)
    if not isinstance(logits,(tuple,list)) or len(logits)!=14:
        raise RuntimeError('Unexpected CheXbert output: expected 14 classification heads')
    output={}
    for i,head in enumerate(logits):
        expected=2 if i==13 else 4
        if head.ndim!=2 or head.shape!=(1,expected):
            raise RuntimeError(f'Unexpected CheXbert head {i} shape {tuple(head.shape)}')
        if i!=13:
            output[HEAD_ORDER[i]]=STATES[int(head.argmax(-1).item())]
    return {name:output.get(name,'NOT_MENTIONED') for name in FINDINGS if name!='No Finding'}
