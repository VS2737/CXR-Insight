"""Research-only nearest neighbor report analogues from the frozen TRAIN split.

Retrieved text is NEVER asserted as a finding in the query image. Retrieval
cannot independently validate disease or safety and may encourage errors.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .v8_cache import load_feature, verify_split_cache, verify_manifest_file
from .v8_vision import unit_rows
from .data import load_jsonl


def build_index(data_dir, features_dir, output_dir):
    manifest=verify_manifest_file(data_dir,'train')
    verify_split_cache(features_dir,'train',manifest)
    rows=load_jsonl(Path(data_dir)/'train.jsonl')
    uids=[str(r['uid']) for r in rows]
    if len(uids)!=len(set(uids)):
        raise ValueError('Expected one study-level frontal image per UID')
    embs=[];meta=[]
    for r in rows:
        cls,_=load_feature(features_dir,'train',r['uid'])
        embs.append(cls)
        meta.append({'uid':str(r['uid']), 'findings':r.get('findings',''),
                     'impression':r['impression']})
    vectors=unit_rows(np.vstack(embs))
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    np.save(out/'train_cls.npy',vectors,allow_pickle=False)
    (out/'train_reports.json').write_text(json.dumps({'source_split':'train',
        'source_sha256':manifest['train_sha256'],'reports':meta,
        'warning':'Reference-like examples from DIFFERENT training studies, not diagnoses for input images.'},
        ensure_ascii=False),encoding='utf-8')
    return len(rows)


class StudyRetriever:
    def __init__(self, folder, expected_train_hash=None):
        out=Path(folder)
        m=json.loads((out/'train_reports.json').read_text(encoding='utf-8'))
        if m.get('source_split')!='train' or (expected_train_hash and m['source_sha256']!=expected_train_hash):
            raise ValueError('Retrieval source is not the frozen training split')
        self.records=m['reports']
        self.vectors=np.load(out/'train_cls.npy',allow_pickle=False).astype('float32')
        if self.vectors.shape!=(len(self.records),768):
            raise ValueError('Retrieval index and metadata do not align')
        self.vectors=unit_rows(self.vectors)
    def search(self,cls,k=3,exclude_uid=None):
        if k<1 or k>20:raise ValueError('k must be 1..20')
        v=unit_rows(np.asarray(cls,dtype='float32')[None])[0]
        similarities=self.vectors@v
        eligible=[i for i,x in enumerate(self.records) if str(x['uid'])!=str(exclude_uid)]
        ranking=sorted(eligible,key=lambda i:float(similarities[i]),reverse=True)[:k]
        return [{'study_uid':self.records[i]['uid'],
                 'visual_similarity':round(float(similarities[i]),4),
                 'reference_findings':self.records[i]['findings'],
                 'reference_impression':self.records[i]['impression'],
                 'source':'TRAIN_STUDY_ANALOGUE_NOT_QUERY_DIAGNOSIS'} for i in ranking]
