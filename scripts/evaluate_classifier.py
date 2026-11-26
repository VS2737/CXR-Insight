"""Evaluate weak-label chest X-ray classifier on frozen held-out split.

Produces per-class metrics and explicit evidence-quality warnings. This is NOT
medical sensitivity/specificity or a clinical validation set.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def evaluate_arrays(probabilities, targets, masks, labels, threshold=0.5):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_score, recall_score, f1_score,
                                 balanced_accuracy_score, brier_score_loss)
    p, y, m = map(np.asarray, (probabilities, targets, masks))
    if p.shape != y.shape or y.shape != m.shape or p.ndim != 2 or p.shape[1] != len(labels):
        raise ValueError('Expected aligned (samples, classes) arrays and class names')
    if not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(m).all()):
        raise ValueError('NaN/Inf in evaluation arrays')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be between zero and one')
    rows=[]
    for j, name in enumerate(labels):
        valid=m[:,j] > 0.5
        yy=y[valid,j].astype(int)
        pp=p[valid,j]
        n=len(yy); pos=int(yy.sum()); neg=n-pos
        row={'finding':name,'known_labels':n,'positive_labels':pos,'negative_labels':neg,
             'coverage':round(float(valid.mean()),4) if len(valid) else 0,
             'threshold':threshold,'auroc':None,'average_precision':None,
             'precision':None,'recall':None,'f1':None,'balanced_accuracy':None,
             'brier':None,'predicted_positives':None,
             'note':''}
        if n:
            pred=(pp>=threshold).astype(int)
            row.update({'precision':round(float(precision_score(yy,pred,zero_division=0)),4),
                        'recall':round(float(recall_score(yy,pred,zero_division=0)),4),
                        'f1':round(float(f1_score(yy,pred,zero_division=0)),4),
                        'brier':round(float(brier_score_loss(yy,pp)),4),
                        'predicted_positives':int(pred.sum())})
            if pos and neg:
                row['auroc']=round(float(roc_auc_score(yy,pp)),4)
                row['average_precision']=round(float(average_precision_score(yy,pp)),4)
                row['balanced_accuracy']=round(float(balanced_accuracy_score(yy,pred)),4)
            else:
                row['note']='Only one known class; discrimination metrics undefined.'
        else:
            row['note']='No definitive weak labels for this finding.'
        rows.append(row)
    return rows


def main():
    import torch
    from torch.utils.data import DataLoader
    from radreport.classifier import load_classifier
    from radreport.config import FINDINGS
    from scripts.train_classifier import LabelledImages
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='models/classifier.pt')
    parser.add_argument('--root',default='data/processed')
    parser.add_argument('--labels',default='data/processed/labels.csv')
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--output',default='results/classifier_weak_label_evaluation.json')
    args=parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    network,metadata=load_classifier(args.checkpoint,device)
    data=LabelledImages(args.split,args.root,args.labels)
    loader=DataLoader(data,batch_size=args.batch_size,shuffle=False,num_workers=0)
    probabilities=[]; targets=[]; masks=[]
    with torch.no_grad():
        for x,y,mask in loader:
            logits=network(x.to(device))
            if metadata.get('calibrated'):
                logits=logits/float(metadata.get('temperature',1.0))
            probabilities.append(logits.sigmoid().cpu().numpy())
            targets.append(y.numpy());masks.append(mask.numpy())
    p,y,m=map(np.concatenate,(probabilities,targets,masks))
    metrics=evaluate_arrays(p,y,m,FINDINGS)
    output={'n_images':len(data),'split':args.split,'checkpoint':args.checkpoint,
            'model_version':metadata.get('version','unknown'),
            'calibration_applied':bool(metadata.get('calibrated',False)),
            'per_class':metrics,
            'warning':'Evaluation against report-derived WEAK labels, not independently verified image ground truth. All thresholds exploratory; metrics are NOT clinical sensitivity/specificity.'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2),encoding='utf-8')
    with out.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
