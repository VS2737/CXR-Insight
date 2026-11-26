"""Synthetic regression tests. Do not interpret as clinical validation."""
import csv
from pathlib import Path
import numpy as np
import pytest
from radreport.safety import report_claims, entity_consistency, route, self_consistency
from radreport.candidate_selection import select_draft
from radreport.classifier import region_for_entity, ground_entities
from scripts.audit_generator import audit, load_csv


def test_left_lower_airspace_disease_detected_and_conflicting_normal():
    r = 'Impression: 1. Patchy left lower lobe airspace disease. 2. Clear lungs.'
    claims=report_claims(r)
    assert claims['findings']['Lung Opacity']=='PRESENT'
    assert any(x['type']=='clear_lungs_conflict' for x in claims['contradictions'])
    assert region_for_entity(r,'Lung Opacity')==(0.52,0.48,0.96,0.96)
    e=entity_consistency(r,{'Lung Opacity':.92})
    assert e['status']=='FAIL' and e['contradictions']
    routed=route(r,[r]*5,{'Lung Opacity':.92},{'status':'PROVISIONAL_PASS'},True)
    assert routed['review_required'] and not routed['safety_check_passed']
    assert any('contradictory' in msg.lower() for msg in routed['review_reasons'])


def test_negative_and_positive_same_disease_conflict():
    c=report_claims('No pleural effusion. There is a moderate pleural effusion.')
    assert any(x['type']=='direct_finding_conflict' and x['finding']=='Pleural Effusion'
               for x in c['contradictions'])


@pytest.mark.parametrize('t', ['Left basilar infiltrate.','Patchy left lower lobe opacity.',
                               'Right lower lobe airspace disease.', 'Right upper lobe consolidation.'])
def test_extended_opacity_words(t):
    c=report_claims(t)
    assert any(v=='PRESENT' for v in c['findings'].values())


def test_mild_cardiomegaly_without_acute_abnormality_not_contradiction():
    t='Mild cardiomegaly. No acute cardiopulmonary disease.'
    assert not report_claims(t)['contradictions']


def test_clear_lungs_with_cardiomegaly_not_contradiction():
    assert not report_claims('The lungs are clear. Mild cardiomegaly.')['contradictions']


def test_normal_chest_and_enlarged_heart_conflict():
    assert report_claims('Normal chest. Cardiomegaly present.')['contradictions']


def test_otherwise_clear_lungs_does_not_contradict_localized_opacity():
    t='Left lower lobe opacity. Otherwise clear lungs.'
    assert not report_claims(t)['contradictions']


def test_general_normal_phrases_consistent_without_artificial_opacity_negation():
    texts=['No acute pulmonary disease.','No acute cardiopulmonary abnormality.',
           'No active disease.','No acute chest findings.','No acute cardiopulmonary process.']
    assert self_consistency(texts)>.9


def test_candidate_selector_only_changes_if_original_contradictory():
    samples=['Left lower lobe opacity. Clear lungs.','Left lower lobe opacity.',
             'No acute disease.']
    assert select_draft(samples,'first')['index']==0
    result=select_draft(samples,'noncontradictory')
    assert result['index']==1 and result['changed']
    assert select_draft(samples[1:],'noncontradictory')['index']==0


def test_gradcam_still_returns_overlay_on_contradiction_but_fails(monkeypatch):
    import torch
    import radreport.classifier as c
    cam=np.zeros((224,224),dtype='float32')
    cam[130:204,130:204] = 1
    monkeypatch.setattr(c,'cam_for_entity',lambda *args:cam)
    report='Left lower lobe airspace disease. Clear lungs.'
    result, img=ground_entities(None,torch.zeros(1,3,224,224),report,['Lung Opacity'],projection='frontal')
    assert img is not None and result['status']=='FAIL' and result['contradictions']


def test_bilateral_roi_abstains():
    assert region_for_entity('Bilateral lower lobe airspace disease.','Lung Opacity') is None


def test_generator_audit_text_only():
    result=audit([{'uid':'1','reference':'Left lower lobe airspace disease.',
                    'generated':'Normal chest.'},
                  {'uid':'2','reference':'No active disease.',
                    'generated':'Pneumonia. Clear lungs.'}])
    assert result['n']==2
    assert result['internally_contradictory_report_count']==1
    assert result['generic_normal_drafts_on_reference_abnormal']==1


def test_audit_comparison_uid_guard(tmp_path):
    p=tmp_path/'x.csv';p.write_text('uid,reference,generated\n1,a,b\n',encoding='utf-8')
    assert load_csv(p)[0]['uid']=='1'


def test_distinct_lung_regions_can_legitimately_differ():
    t='No left upper lobe opacity. Right lower lobe opacity.'
    assert not any(x['type']=='direct_finding_conflict' for x in report_claims(t)['contradictions'])


def test_same_region_positive_negative_is_conflict():
    t='No left lower lobe opacity. Patchy left lower lobe opacity.'
    assert any(x['type']=='direct_finding_conflict' for x in report_claims(t)['contradictions'])
