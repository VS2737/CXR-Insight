"""Offline regression tests: no downloaded model weights or real medical images."""
import json
import io
import hashlib
import numpy as np
import pytest
import torch
from PIL import Image
from radreport.v8_vision import tokens_from_hidden,unit_rows
from radreport.v8_cache import load_feature,feature_path,verify_split_cache,verify_manifest_file
from radreport.v8_probe import fit_probe,predict_probe,save_probe,load_probe
from radreport.v8_retrieval import build_index, StudyRetriever
from radreport.config import FINDINGS


def test_dino_token_pooling():
    # CLS + 8x8 patch tokens; feature width 768.
    hidden=torch.randn(1,65,768)
    cls,patch=tokens_from_hidden(hidden,grid=7)
    assert cls.shape == (768,) and patch.shape == (49,768)
    assert torch.allclose(torch.linalg.vector_norm(cls),torch.tensor(1.),atol=1e-5)


def test_dino_rejects_register_tokens():
    with pytest.raises(ValueError,match='layout'):
        tokens_from_hidden(torch.randn(1,67,768),grid=7)


def test_feature_cache_rejects_mismatched_manifest(tmp_path):
    split_dir=tmp_path/'train';split_dir.mkdir()
    (split_dir/'cache_info.json').write_text(json.dumps({
        'model':'microsoft/rad-dino','grid':7,'split_sha256':'old'}))
    with pytest.raises(ValueError,match='does not match'):
        verify_split_cache(tmp_path,'train',{'train_sha256':'new'})


def test_feature_cache_is_shape_checked_and_pickle_free(tmp_path):
    path=feature_path(tmp_path,'train','uid-1')
    path.parent.mkdir(parents=True)
    np.savez_compressed(path,cls=np.zeros(768,dtype='float16'),
                        tokens=np.ones((49,768),dtype='float16'))
    cls,tokens=load_feature(tmp_path,'train','uid-1')
    assert cls.shape==(768,) and tokens.shape==(49,768)
    assert feature_path(tmp_path,'test','uid-1')!=path


def test_probe_does_not_fit_classes_without_support():
    x=np.zeros((20,768),dtype='float32')
    yt=np.zeros((20,len(FINDINGS)),dtype=int)
    mv=np.zeros_like(yt,dtype=bool)
    result=fit_probe(x,yt,mv,x[:10],yt[:10],mv[:10])
    assert not any(v['trained'] for v in result['labels'].values())
    assert all(x['status']=='UNAVAILABLE' for x in predict_probe(result,x[0]).values())


def test_probe_uses_train_fit_and_val_thresholds(tmp_path):
    rng=np.random.default_rng(123)
    tr=rng.standard_normal((70,768)).astype('float32')
    va=rng.standard_normal((30,768)).astype('float32')
    j=FINDINGS.index('Pneumonia')
    yt=np.zeros((70,len(FINDINGS)),dtype=int);yv=np.zeros((30,len(FINDINGS)),dtype=int)
    mt=np.zeros_like(yt,dtype=bool);mv=np.zeros_like(yv,dtype=bool)
    yt[:,j]=(tr[:,0]>0).astype(int);yv[:,j]=(va[:,0]>0).astype(int)
    mt[:,j]=True;mv[:,j]=True
    model=fit_probe(tr,yt,mt,va,yv,mv)
    assert model['labels']['Pneumonia']['trained']
    assert model['labels']['Pneumonia']['threshold'] is not None
    path=tmp_path/'probe.json';save_probe(model,path)
    result=predict_probe(load_probe(path),va[0])
    assert result['Pneumonia']['status'] in ('ABOVE_VALIDATION_THRESHOLD','BELOW_VALIDATION_THRESHOLD')
    assert result['Edema']['status']=='UNAVAILABLE'


def test_retriever_only_uses_train_split(tmp_path):
    d=tmp_path/'processed';f=tmp_path/'features';out=tmp_path/'index'
    d.mkdir();(f/'train').mkdir(parents=True)
    train=[{'uid':'train-1','findings':'normal','impression':'No acute finding'},
           {'uid':'train-2','findings':'opacity','impression':'Focal opacity'}]
    (d/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in train))
    sha=hashlib.sha256((d/'train.jsonl').read_bytes()).hexdigest()
    (d/'manifest.json').write_text(json.dumps({'train_sha256':sha}))
    (f/'train'/'cache_info.json').write_text(json.dumps({
        'split_sha256':sha,'model':'microsoft/rad-dino','grid':7}))
    for i,r in enumerate(train):
        cls=np.zeros(768,dtype='float32');cls[i]=1
        path=feature_path(f,'train',r['uid'])
        np.savez_compressed(path,cls=cls,tokens=np.zeros((49,768),dtype='float16'))
    assert build_index(d,f,out)==2
    retriever=StudyRetriever(out,sha)
    q=np.zeros(768,dtype='float32');q[0]=1
    answer=retriever.search(q,k=2,exclude_uid='train-1')
    assert len(answer)==1 and answer[0]['study_uid']=='train-2'
    assert answer[0]['source']=='TRAIN_STUDY_ANALOGUE_NOT_QUERY_DIAGNOSIS'
    with pytest.raises(ValueError,match='not the frozen'):
        StudyRetriever(out,'wrong-hash')


def test_v8_app_model_path_with_mocked_assets(monkeypatch):
    import app.service as s
    class FakeEncoder:
        def encode(self,image):
            return np.zeros(768,dtype='float32'),np.zeros((49,768),dtype='float32')
    class FakeGenerator:
        def sample_reports(self,tokens,tok,n=5,max_new_tokens=168):
            return ['FINDINGS:\nClear lungs.\nIMPRESSION:\nNo acute cardiopulmonary abnormality.']*n
    class FakeRetriever:
        def search(self,feature,k=3):
            return [{'study_uid':'other-train-study','source':'TRAIN_STUDY_ANALOGUE_NOT_QUERY_DIAGNOSIS'}]
    assets={'ready':True,'model_family':'rad-dino-gpt2-v8','model':FakeGenerator(),
            'encoder':FakeEncoder(),'tokenizer':object(),
            'retriever':FakeRetriever(),'probe':None,'classifier':None,
            'classifier_metadata':{},'version':'v8-test','metadata':{'target_mode':'structured'},'device':'cpu'}
    monkeypatch.setattr(s,'load_assets',lambda:assets)
    monkeypatch.setattr(s,'enqueue',lambda *a,**k:None)
    image=Image.new('RGB',(256,256))
    buffer=io.BytesIO();image.save(buffer,format='PNG')
    output=s.analyze(buffer.getvalue(),projection='frontal')
    assert output['model_loaded'] and output['model_version']=='v8-test'
    assert output['formal_report']['report_source']=='MODEL_GENERATED_BOTH'
    assert output['safety']['review_required'] is True
    assert output['v8_retrieved_training_analogues'][0]['study_uid']=='other-train-study'


def test_frozen_jsonl_tampering_detected(tmp_path):
    p=tmp_path/'train.jsonl'
    p.write_text('{"uid":"a"}\n')
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps({'train_sha256':sha}))
    verify_manifest_file(tmp_path,'train')
    p.write_text('{"uid":"b"}\n')
    with pytest.raises(ValueError,match='checksum'):
        verify_manifest_file(tmp_path,'train')
