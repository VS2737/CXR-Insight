"""Measure v8 probe against untouched test weak labels; never re-fit on test."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
from radreport.v8_cache import source_manifest
from radreport.v8_probe import prepare_probe_arrays,load_probe,predict_probe
from radreport.config import FINDINGS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',default='data/processed/labels.csv')
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--model',default='models_v8/finding_probe.json')
    p.add_argument('--output',default='results_v8/probe_weak_label_test.json')
    a=p.parse_args()
    probe=load_probe(a.model)
    manifest=source_manifest(a.data_dir)
    if probe.get('train_split_sha256')!=manifest['train_sha256'] or probe.get('val_split_sha256')!=manifest['val_sha256']:
        raise ValueError('Probe trained on a different data split')
    with open(a.labels,encoding='utf-8-sig',newline='') as f:
        labeled={(r['split'],str(r['uid'])):r for r in csv.DictReader(f)}
    X,y,mask,ids=prepare_probe_arrays('test',a.data_dir,a.features,labeled)
    scores=[predict_probe(probe,x) for x in X]
    per=[]
    for j,name in enumerate(FINDINGS):
        info=probe['labels'][name]
        valid=mask[:,j];ground=y[valid,j]
        if not info['trained'] or len(ground)==0:
            per.append({'finding':name,'status':'UNAVAILABLE','n':int(valid.sum())})
            continue
        pvals=np.array([r[name]['weak_label_score'] for r in scores])[valid]
        pred=(pvals>=info['threshold']).astype(int)
        row={'finding':name,'status':'EVALUATED_WEAK_LABELS','n':int(valid.sum()),
             'positives':int(ground.sum()),'threshold_from_val':info['threshold'],
             'precision':round(float(precision_score(ground,pred,zero_division=0)),4),
             'recall':round(float(recall_score(ground,pred,zero_division=0)),4),
             'f1':round(float(f1_score(ground,pred,zero_division=0)),4)}
        if len(np.unique(ground))>1:row['auroc']=round(float(roc_auc_score(ground,pvals)),4)
        per.append(row)
    output={'n_studies':len(X),'split':'test','per_class':per,
            'warning':'Held-out REPORT-DERIVED CheXbert labels, not expert-confirmed image diagnoses.'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
