"""v9 regression tests without large pretrained downloads or medical images."""
import csv
import json
import numpy as np
import pytest
import torch
from torch import nn
from radreport.v9_concepts import CONCEPTS, special_concepts, make_targets, claims_for_audit, compare_reference_generated
from radreport.v9_joint import JointGenerator
from scripts.v9_train_generator import positive_weights, masked_loss, ConceptsDataset
from scripts.v9_compare import paired_bootstrap_delta


def test_structural_abnormality_mentions_are_preserved_without_acuteness_conflation():
    reference='Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries. Clear lungs. No acute pulmonary findings.'
    generated='The heart size and pulmonary vascularity appear within normal limits. The lungs are free of focal airspace disease. No pleural effusion or pneumothorax is seen.'
    audited=compare_reference_generated(reference,generated)
    assert {'Cardiomegaly','Prior Sternotomy','Pulmonary Artery Enlargement'}.issubset(set(audited['omitted_mentions']))
    assert {'Cardiomegaly','Pulmonary Artery Enlargement'}.issubset(set(audited['explicitly_negated_reference_mentions']))
    assert 'Prior Sternotomy' not in audited['explicitly_negated_reference_mentions']


def test_negative_no_acute_not_negative_for_chronic_findings():
    result=claims_for_audit('Borderline cardiomegaly. Previous sternotomy. No acute pulmonary process.')
    assert result['Cardiomegaly']=='PRESENT'
    assert result['Prior Sternotomy']=='PRESENT'


def test_sparse_known_label_masks_and_uncertainty():
    row={'findings':'Enlarged pulmonary arteries. Sternotomy wires. No focal pneumonia.'}
    y,mask=make_targets(row,{'Cardiomegaly':'-1.0','Pneumonia':'0.0'})
    assert len(y)==len(CONCEPTS)==15
    assert mask[CONCEPTS.index('Cardiomegaly')]==0
    assert mask[CONCEPTS.index('Pneumonia')]==1 and y[CONCEPTS.index('Pneumonia')]==0
    assert y[CONCEPTS.index('Prior Sternotomy')]==1
    assert y[CONCEPTS.index('Pulmonary Artery Enlargement')]==1
    assert mask[CONCEPTS.index('Atelectasis')]==0


def test_conflicting_structural_statements_abstain():
    s='Enlarged pulmonary arteries. Normal pulmonary arteries.'
    assert special_concepts(s)['Pulmonary Artery Enlargement'] is None
    assert special_concepts('No acute disease.')['Pulmonary Artery Enlargement'] is None
    assert special_concepts('No acute disease.')['Prior Sternotomy'] is None


def test_only_supported_classes_have_train_loss_weight():
    y=np.zeros((10,15),dtype='float32');mask=np.zeros_like(y)
    mask[:,0]=1; y[:6,0]=1
    mask[:3,1]=1; y[:2,1]=1
    w,eligible,support=positive_weights(y,mask,min_positive=5,min_negative=3)
    assert eligible[0]==1 and eligible[1]==0
    assert support[0]['positive']==6 and support[0]['negative']==4
    assert w.max()<=4


def test_masked_aux_loss_does_not_punish_unknowns():
    logits=torch.zeros((1,15),requires_grad=True)
    y=torch.ones((1,15));known=torch.zeros((1,15));known[:,0]=1
    loss=masked_loss(logits,y,known,torch.ones(15))
    assert float(loss)==pytest.approx(0.693147,rel=1e-5)
    loss.backward()
    assert logits.grad[0,1:].abs().sum().item()==0
    assert logits.grad[0,0].item()!=0


def test_v9_joint_has_finding_head_and_delegates_inference():
    class DummyBase(nn.Module):
        def __init__(self):
            super().__init__();self.bridge=nn.Identity()
        def sample_reports(self,*args,**kwargs):return ['FINDINGS: test.\nIMPRESSION: test.']
    j=JointGenerator(DummyBase())
    result=j.finding_research_scores(torch.zeros((1,49,768)))
    assert set(result)==set(CONCEPTS)
    assert all(0<=v<=1 for v in result.values())
    assert len(j.sample_reports(None,None))==1


def test_paired_ci_same_model_zero_delta():
    rows=[{'reference':'FINDINGS: Borderline cardiomegaly.','generated':'FINDINGS: Borderline cardiomegaly.'},
          {'reference':'FINDINGS: Prior sternotomy.','generated':'FINDINGS: Normal chest.'}]
    result=paired_bootstrap_delta(rows,rows,repeats=20)
    assert result['delta_v9_minus_v8']==0
    assert result['paired_bootstrap_ci95']==[0,0]


def test_v9_inference_api_remains_fail_closed(monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    import app.service as service
    class FakeEncoder:
        def encode(self,image):
            return np.zeros(768,dtype='float32'),np.zeros((49,768),dtype='float32')
    class FakeV9:
        def sample_reports(self,patches,tokenizer,**kw):
            return ['FINDINGS:\nThe heart size is normal.\nIMPRESSION:\nNo acute findings.']*5
        def finding_research_scores(self,patches):
            return {'Cardiomegaly':.77,'Prior Sternotomy':.23}
    assets={'ready':True,'model_family':'rad-dino-gpt2-v9','model':FakeV9(),
            'encoder':FakeEncoder(),'tokenizer':object(),'retriever':None,'probe':None,
            'classifier':None,'classifier_metadata':{},'version':'v9-test',
            'metadata':{'target_mode':'structured'},'device':'cpu'}
    monkeypatch.setattr(service,'load_assets',lambda:assets)
    monkeypatch.setattr(service,'enqueue',lambda *a,**k:None)
    img=Image.new('RGB',(256,256))
    b=io.BytesIO();img.save(b,format='PNG')
    result=service.analyze(b.getvalue(),projection='frontal')
    assert result['model_version']=='v9-test'
    assert result['safety']['review_required'] is True
    assert result['safety']['v9_auxiliary_weak_label_scores']['Cardiomegaly']==.77
    assert result['formal_report']['complete_sections'] is True
    assert 'clinical' in result['safety']['v9_auxiliary_warning'].lower()
