"""Check v10 prerequisites without downloading or training any model."""
import argparse
import csv
import json
from pathlib import Path

import torch
from radreport.data import load_jsonl
from radreport.v8_cache import verify_manifest_file,verify_split_cache,load_feature
from radreport.v9_joint import ARCH
from radreport.v9_concepts import CONCEPTS
from radreport.v10_text import clean_pair,VERSION


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--labels',default='data/processed/labels.csv')
    p.add_argument('--v9',default='models_v9')
    a=p.parse_args()
    manifest=verify_manifest_file(a.data_dir,'train')
    verify_manifest_file(a.data_dir,'val')
    for split in ('train','val'):
        verify_split_cache(a.features,split,manifest)
    weights=Path(a.v9)/'generator.pt'
    if not weights.is_file():raise FileNotFoundError('v9 checkpoint missing: '+str(weights))
    saved=torch.load(weights,map_location='cpu',weights_only=True)
    meta=saved['metadata']
    if (meta.get('architecture')!=ARCH or meta.get('concepts')!=list(CONCEPTS) or
          meta.get('train_split_sha256')!=manifest['train_sha256'] or
          meta.get('val_split_sha256')!=manifest['val_sha256']):
        raise ValueError('v9 checkpoint metadata does not match current IU train+val split')
    with open(a.labels,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'uid','split'}.issubset(set(reader.fieldnames or [])):
            raise ValueError('labels.csv must include split and uid')
        label_count=sum(1 for r in reader if r.get('split') in ('train','val'))
    report={}
    for split in ('train','val'):
        rows=load_jsonl(Path(a.data_dir)/f'{split}.jsonl')
        _,token=load_feature(a.features,split,rows[0]['uid'])
        if tuple(token.shape)!=(49,768):raise ValueError('Feature cache has wrong token shape')
        results=[clean_pair(r)[1] for r in rows if r.get('findings') and r.get('impression')]
        report[split]={'studies':len(rows),'usable_after_cleaning':sum(r['usable'] for r in results),
                        'redacted_clauses':sum(r['redacted_clauses'] for r in results)}
    print(json.dumps({'ready':True,'cuda_available':torch.cuda.is_available(),
                      'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      'v9_initialization':str(weights),'weak_label_rows_train_val':label_count,
                      'cleaner':VERSION,'train_val':report,
                      'warning':'Preflight checks files, not model accuracy. IU text labels remain weak evidence.'},indent=2))

if __name__=='__main__':main()
