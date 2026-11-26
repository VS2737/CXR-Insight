"""v10 regression tests: text cleanup, weak supervision, evaluation isolation."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from radreport.v10_text import clean_section, clean_pair, encode_clean_target, artifact_flags
from scripts.v10_train_generator import study_sample_weights, validate_config
from scripts.v10_compare import paired_mention_ci, report_quality
from radreport.v9_concepts import compare_reference_generated


def test_deidentified_sternotomy_preserved_without_ghost_tokens():
    clean=clean_section('Midline sternotomy XXXX. Enlarged pulmonary arteries. Clear lungs. Inferior XXXX XXXX.')
    assert clean.text == 'Midline sternotomy. Enlarged pulmonary arteries. Clear lungs.'
    assert clean.redacted_clauses == 1
    assert 'XXXX' not in clean.text


def test_truncated_costophrenic_sentence_dropped_not_guessed():
    clean=clean_section('No pneumothorax. Blunting of left costophrenic XXXX.')
    assert clean.text == 'No pneumothorax.'
    assert clean.removed_clauses==1
    assert 'effusion' not in clean.text.lower()


def test_explicit_surgery_and_abnormality_kept_while_deid_dropped():
    target,info=clean_pair({'uid':'2','findings':'Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries. Clear lungs.',
                            'impression':'No acute pulmonary findings.'})
    assert 'Borderline cardiomegaly.' in target
    assert 'Midline sternotomy.' in target
    assert 'Enlarged pulmonary arteries.' in target
    assert 'No acute pulmonary findings.' in target
    assert info['usable'] and 'XXXX' not in target


def test_cleaner_drops_incomplete_report_without_imputation():
    target,info=clean_pair({'findings':'XXXX XXXX.','impression':'No acute disease.'})
    assert target is None and not info['usable']


def test_numbered_radiology_impressions_not_deleted():
    clean=clean_section('1. Cardiomegaly. 2. Mild interstitial edema. 3. No pneumothorax.')
    assert 'Cardiomegaly.' in clean.text
    assert 'Mild interstitial edema.' in clean.text
    assert 'No pneumothorax.' in clean.text


def test_cleaner_does_not_invent_diagnosis():
    clean=clean_section('No acute disease. XXXX XXXX. Lungs are clear.')
    assert 'pneumonia' not in clean.text.lower()
    assert 'No acute disease.' in clean.text


class TokenStub:
    eos_token_id=50256
    def encode(self,text,add_special_tokens=False):
        return list(range(1,len(text)+1))


def test_tokenization_preserves_impression_when_findings_are_long():
    txt='FINDINGS:\n' + 'Long finding. '*60 + '\nIMPRESSION:\nNo acute disease.'
    ids,attn,labels=encode_clean_target(TokenStub(),txt,max_length=128)
    assert len(ids)==len(attn)==len(labels)==128
    assert (attn==1).sum() > 50
    assert ids[0]==50256
    assert labels[attn==0].eq(-100).all()
    # The suffix includes impression tokens and EOS, rather than truncating before it.
    assert labels[attn==1][-1].item()==50256


def test_tokenization_rejects_unstructured_target():
    with pytest.raises(ValueError):
        encode_clean_target(TokenStub(),'Impression: Normal.',128)


def test_train_sampling_only_uses_supported_weak_positives():
    target=np.array([[1,0],[0,0],[0,1],[1,1]],dtype=np.float32)
    masks=np.ones_like(target)
    supported=np.array([1,0],dtype=np.float32)
    weights=study_sample_weights(target,masks,supported,1.5)
    assert weights.tolist()==[1.5,1.,1.,1.5]
    with pytest.raises(ValueError):study_sample_weights(target,masks,supported,4)


def test_training_refuses_overwriting_v9():
    from argparse import Namespace
    x=Namespace(epochs=3,batch_size=1,accumulation=8,max_length=192,lr=1e-5,
                aux_weight=.35,abnormal_factor=1.5,init_v9='models_v9',output='models_v9',overwrite=False)
    with pytest.raises(ValueError):validate_config(x)


def test_v10_comparison_requires_explicit_final_test(monkeypatch):
    from scripts.v10_compare import main
    monkeypatch.setattr(sys,'argv',['v10_compare','--split','test'])
    with pytest.raises(SystemExit):main()


def test_paired_ci_will_not_claim_gain_for_identical_models():
    a=compare_reference_generated('FINDINGS: Cardiomegaly.','FINDINGS: Clear lungs.')
    old=[{'uid':'1','reference':'FINDINGS: Cardiomegaly.','audit':a}]
    new=[{'uid':'1','reference':'FINDINGS: Cardiomegaly.','audit':a}]
    ci=paired_mention_ci(old,new,repeats=50)
    assert ci['mean_delta_v10_minus_v9']==0 and ci['paired_ci95']==[0,0]


def test_paired_ci_refuses_mismatched_studies():
    with pytest.raises(ValueError):paired_mention_ci([{'uid':'1'}],[{'uid':'2'}])


def test_generated_artifacts_flagged_without_manufacturing_findings():
    f=artifact_flags('FINDINGS:\nHeart is XXXX XXXX.\nIMPRESSION:\nNegative chest x-XXXX.')
    assert 'deidentification_artifact' in f
    assert artifact_flags('FINDINGS:\nLungs clear.\nIMPRESSION:\nNo acute disease.')==[]


def test_v10_api_research_review_on_artifacts(monkeypatch):
    import app.service as service
    class Enc:
        def encode(self,img):return np.zeros(768,dtype='float32'),np.zeros((49,768),dtype='float32')
    class Gen:
        def sample_reports(self,*args,**kwargs):return ['FINDINGS:\nXXXX XXXX.\nIMPRESSION:\nNo acute disease.']*5
        def finding_research_scores(self,*args):return {'Cardiomegaly':.8}
    fake={'ready':True,'model_family':'rad-dino-gpt2-v9','model':Gen(),'encoder':Enc(),
          'tokenizer':object(),'retriever':None,'probe':None,'classifier':None,
          'classifier_metadata':{},'version':'rad-dino-gpt2-v10-research',
          'metadata':{'version':'rad-dino-gpt2-v10-research','target_mode':'structured'},'device':'cpu'}
    monkeypatch.setattr(service,'load_assets',lambda:fake)
    monkeypatch.setattr(service,'enqueue',lambda *args,**kw:None)
    out=io.BytesIO();Image.new('RGB',(256,256)).save(out,format='PNG')
    result=service.analyze(out.getvalue(),projection='frontal')
    assert result['safety']['review_required'] is True
    assert result['safety']['safety_check_passed'] is False
    assert 'deidentification_artifact' in result['safety']['v10_text_quality_flags']
    assert result['formal_report']['is_final'] is False


def test_optional_clean_text_reranker_only_changes_when_malformed():
    from radreport.candidate_selection import select_draft
    original='FINDINGS:\nNegative chest x-XXXX.\nIMPRESSION:\nNo acute disease.'
    alternative='FINDINGS:\nNo focal consolidation.\nIMPRESSION:\nNo acute disease.'
    d=select_draft([original,alternative],'clean_text_v10')
    assert d['index']==1 and d['changed']
    assert 'NOT medical validation' in d['reason']
    assert select_draft([alternative,original],'clean_text_v10')['index']==0


def test_clean_reranker_retains_unclean_when_no_alternate():
    from radreport.candidate_selection import select_draft
    bad='FINDINGS:\nXXXX XXXX.\nIMPRESSION:\nUnknown XXXX.'
    d=select_draft([bad,bad],'clean_text_v10')
    assert d['index']==0 and not d['changed']
