"""Optional ClinicalBERT-based BERTScore F1 on report pairs.

Downloads model weights. This is semantic similarity, not clinical correctness.
"""
import argparse
import csv
import json
from pathlib import Path

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',default='results/predictions.csv')
    parser.add_argument('--output',default='results/clinicalbert_score.json')
    a=parser.parse_args()
    from bert_score import score
    with open(a.predictions,newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No predictions found.')
    g=[r['generated'] for r in rows]
    r=[r['reference'] for r in rows]
    _,_,f=score(g,r,model_type='emilyalsentzer/Bio_ClinicalBERT',num_layers=12,
                batch_size=8,verbose=True)
    metrics={'clinicalbert_score':float(f.mean().item()),'n_reports':len(rows),
             'caveat':'BERTScore semantic similarity using Bio_ClinicalBERT; not image-grounded clinical accuracy.'}
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
