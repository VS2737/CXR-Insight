"""Weak-label linear probe of RAD-DINO CLS vectors with validation-only thresholds.

Does NOT discover every abnormality. Report-derived labels are imperfect and
missing/uncertain observations are NEVER silently converted to negatives.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from radreport.config import FINDINGS
from radreport.label_values import parse_binary_chexpert
from radreport.v8_cache import load_feature, verify_split_cache, verify_manifest_file, source_manifest
from radreport.data import load_jsonl


def prepare_probe_arrays(split, data_dir, feature_dir, by_split_uid):
    verify_split_cache(feature_dir,split,verify_manifest_file(data_dir,split))
    rows=load_jsonl(Path(data_dir) / f'{split}.jsonl')
    feats=[]; y=[]; masks=[];uids=[]
    for r in rows:
        info=by_split_uid.get((split,str(r['uid'])))
        if info is None:
            continue
        cls,_=load_feature(feature_dir,split,r['uid'])
        target=[];mask=[]
        for label in FINDINGS:
            v=parse_binary_chexpert(info.get(label,''))
            target.append(float(v) if v is not None else 0.)
            mask.append(bool(v is not None))
        feats.append(cls); y.append(target); masks.append(mask);uids.append(str(r['uid']))
    if not feats:
        raise ValueError(f'No {split} features with matching CheXbert weak labels')
    X=np.asarray(feats,dtype=np.float32)
    return X,np.asarray(y,dtype=int),np.asarray(masks,dtype=bool),uids


def fit_probe(Xtr,ytr,Mtr,Xval,yval,Mval, labels=FINDINGS,
              min_train_positives=5, min_train_negatives=5,
              min_val_positives=3,min_val_negatives=3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    tr=np.asarray(Xtr,dtype=np.float32); va=np.asarray(Xval,dtype=np.float32)
    yt=np.asarray(ytr);yv=np.asarray(yval)
    mt=np.asarray(Mtr,dtype=bool);mv=np.asarray(Mval,dtype=bool)
    if tr.ndim!=2 or tr.shape[1]!=768 or va.ndim!=2 or va.shape[1]!=768:
        raise ValueError('Expected [n,768] RAD-DINO CLS feature matrices')
    if yt.shape!=(len(tr),len(labels)) or yv.shape!=(len(va),len(labels)):
        raise ValueError('Label matrices do not match embeddings')
    if mt.shape!=yt.shape or mv.shape!=yv.shape:
        raise ValueError('Label masks do not match labels')
    if not np.isfinite(tr).all() or not np.isfinite(va).all():
        raise ValueError('Nonfinite embeddings')
    fitted={}
    for j,label in enumerate(labels):
        yt_known=yt[mt[:,j],j]
        yv_known=yv[mv[:,j],j]
        nt_pos=int(yt_known.sum());nt_neg=int(len(yt_known)-nt_pos)
        nv_pos=int(yv_known.sum());nv_neg=int(len(yv_known)-nv_pos)
        entry={'trained':False,'train_known':len(yt_known),
               'train_positive':nt_pos,'val_known':len(yv_known),
               'val_positive':nv_pos,'threshold':None,'val_f1':None,
               'val_auroc':None,'reason':''}
        if nt_pos<min_train_positives or nt_neg<min_train_negatives:
            entry['reason']='Insufficient positive/negative training support'
        elif nv_pos<min_val_positives or nv_neg<min_val_negatives:
            entry['reason']='Insufficient validation support to calibrate threshold'
        else:
            classifier=LogisticRegression(C=.05,class_weight='balanced',max_iter=1000,solver='liblinear',random_state=42)
            classifier.fit(tr[mt[:,j]],yt_known)
            probs=classifier.predict_proba(va[mv[:,j]])[:,1]
            # Only validation optimizes the research operating threshold; test
            # is never involved in fitting, tuning, or model promotion.
            candidate_thresholds=np.arange(.20,.801,.025)
            scores=[f1_score(yv_known,probs>=t,zero_division=0) for t in candidate_thresholds]
            best_index=int(np.argmax(scores))
            entry.update({'trained':True,'threshold':round(float(candidate_thresholds[best_index]),4),
                          'val_f1':round(float(scores[best_index]),4),
                          'val_auroc':round(float(roc_auc_score(yv_known,probs)),4),
                          'weights':classifier.coef_[0].astype(float).tolist(),
                          'intercept':float(classifier.intercept_[0]),
                          'reason':'WEAK-LABEL research threshold; not clinically calibrated'})
        fitted[label]=entry
    return {'architecture':'RAD-DINO-CLS-linear-probe-v8',
            'feature_model':'microsoft/rad-dino',
            'labels':fitted,
            'warning':'Report-derived weak labels, limited ontology, validation-only thresholds. Not clinical probabilities.'}


def predict_probe(probe,cls):
    vec=np.asarray(cls,dtype=np.float32)
    if vec.shape!=(768,) or not np.isfinite(vec).all():
        raise ValueError('Expected finite 768-dimensional embedding')
    output={}
    for name,info in probe['labels'].items():
        if not info.get('trained'):
            output[name]={'status':'UNAVAILABLE','reason':info.get('reason','Not trained')}
            continue
        logit=float(np.dot(np.asarray(info['weights'],dtype=np.float32),vec)+info['intercept'])
        p=float(1/(1+np.exp(-np.clip(logit,-50,50))))
        output[name]={'status':'ABOVE_VALIDATION_THRESHOLD' if p>=info['threshold'] else 'BELOW_VALIDATION_THRESHOLD',
                      'weak_label_score':round(p,4), 'threshold':info['threshold']}
    return output


def save_probe(probe,path):
    out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(probe,indent=2),encoding='utf-8')


def load_probe(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('architecture')!='RAD-DINO-CLS-linear-probe-v8':
        raise ValueError('Unexpected probe metadata')
    return data
