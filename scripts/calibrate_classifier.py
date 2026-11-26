"""Temperature-scale image classifier against held-out VALIDATION weak labels.

This is dataset-local calibration, NOT external clinical validation. All API
outputs are still drafts, the release gate still requires independent evidence.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from scripts.train_classifier import LabelledImages
from radreport.classifier import load_classifier


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='models/classifier.pt')
    p.add_argument('--root',default='data/processed')
    p.add_argument('--labels',default='data/processed/labels.csv')
    a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,metadata=load_classifier(a.checkpoint,device)
    loader=DataLoader(LabelledImages('val',a.root,a.labels),batch_size=8)
    logits,targets,masks=[],[],[]
    with torch.no_grad():
        for x,y,mask in loader:
            logits.append(model(x.to(device)).cpu())
            targets.append(y);masks.append(mask)
    l,y,m=torch.cat(logits),torch.cat(targets),torch.cat(masks)
    if m.sum().item()<20:raise ValueError('Not enough known validation labels for calibration')
    best_t,best_loss=None,float('inf')
    for temperature in [0.5,0.7,0.85,1.0,1.25,1.5,2.0,3.0,5.0]:
        e=torch.nn.functional.binary_cross_entropy_with_logits(l/temperature,y,reduction='none')
        loss=float((e*m).sum()/m.sum())
        if loss<best_loss: best_t,best_loss=temperature,loss
    metadata.update({'calibrated':True,'temperature':best_t,'calibration_loss':best_loss,
                     'calibration_scope':'IU validation report-derived weak CheXpert labels only',
                     'independent_radiologist_validation':False})
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ckpt['metadata']=metadata
    torch.save(ckpt,a.checkpoint)
    print('Weak-label calibration complete; NOT medically validated:',json.dumps(metadata,indent=2))

if __name__=='__main__':main()
