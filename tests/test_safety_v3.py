"""Safety v3 unit and regression tests. No claims of diagnostic accuracy."""
import pytest
from radreport.safety import report_claims, self_consistency, entity_consistency, route
from radreport.classifier import region_for_entity, region_for_report


def test_equivalent_no_acute_phrases_agree():
    variants=['No acute pulmonary disease.','No acute cardiopulmonary abnormality.',
              'No acute cardiopulmonary findings.', 'No active disease.',
              'No acute cardiopulmonary abnormality..']
    assert self_consistency(variants) > .90
    assert entity_consistency(variants[0], {'Pneumonia':.04})['status']=='UNVERIFIED'


def test_normal_claim_flags_possible_omission_not_a_diagnosis():
    r=entity_consistency('Normal chest.',{'Pneumonia':.92})
    assert r['status']=='FAIL' and 'Pneumonia' in r['possible_omissions']
    assert 'review' in r['reason'].lower()


def test_no_acute_disease_does_not_negate_chronic_cardiomegaly():
    x=report_claims('Mild cardiomegaly. No acute cardiopulmonary disease.')
    assert x['findings']['Cardiomegaly']=='PRESENT'
    assert x['normal_scope']=='ACUTE_ONLY'


def test_local_negation_and_hedge_are_preserved():
    x=report_claims('No pleural effusion. Pneumothorax cannot be excluded. Right lower lobe pneumonia.')
    assert x['findings']['Pleural Effusion']=='ABSENT'
    assert x['findings']['Pneumothorax']=='UNCERTAIN'
    assert x['findings']['Pneumonia']=='PRESENT'


def test_report_conflict_scores_low():
    variants=['Right lower lobe pneumonia.']*3+['No evidence of pneumonia.']*2
    assert self_consistency(variants)<.70


def test_positive_and_negative_image_conflicts():
    assert 'Pneumonia' in entity_consistency('Pneumonia.',{'Pneumonia':.10})['unsupported']
    e=entity_consistency('No evidence of pneumonia.',{'Pneumonia':.91})
    assert 'Pneumonia' in e['possible_omissions']


def test_out_of_ontology_abstains():
    r=entity_consistency('Pulmonary fibrosis.',{'Pneumonia':.02})
    assert r['status']=='UNVERIFIED'
    assert 'fibrosis' in r['claims']['uncovered_terms']


def test_per_entity_roi_scoping():
    t='Right lower lobe pneumonia. Left upper lobe pleural effusion.'
    assert region_for_entity(t,'Pneumonia')==(0.04,0.48,0.48,0.96)
    assert region_for_entity(t,'Pleural Effusion')==(0.52,0.12,0.96,0.55)
    assert region_for_report('Right lower lobe pneumonia.')==(0.04,0.48,0.48,0.96)
    assert region_for_entity('Pulmonary edema.','Edema') is None


def test_review_never_auto_approved():
    s=route('Pneumonia.', ['Pneumonia.']*5, {'Pneumonia':.98},
            {'status':'PROVISIONAL_PASS'}, True)
    assert s['review_required'] and s['status']=='MANDATORY_REVIEW'
    assert s['safety_check_passed']


def test_gradcam_abstains_for_unknown_projection(monkeypatch):
    import torch
    from radreport.classifier import ground_entities
    result, cam = ground_entities(None,torch.zeros(1,3,224,224),
                                  'Right lower lobe pneumonia.',['Pneumonia'],projection='unknown')
    assert result['status']=='UNKNOWN' and cam is None


def test_gradcam_finding_specific_roi(monkeypatch):
    import numpy as np
    import torch
    import radreport.classifier as c
    cam = np.zeros((224,224),dtype='float32')
    cam[125:205,20:100]=1.0  # right lower ROI in viewer-left image coords
    monkeypatch.setattr(c,'cam_for_entity', lambda *args:cam)
    result, heatmap = c.ground_entities(None,torch.zeros(1,3,224,224),
                                     'Right lower lobe pneumonia.',['Pneumonia'],projection='frontal')
    assert result['status']=='PROVISIONAL_PASS'
    assert result['details'][0]['area_normalized_enrichment']>1.15
    assert heatmap is cam


def test_gradcam_abstains_without_local_anatomy(monkeypatch):
    import torch
    from radreport.classifier import ground_entities
    result, cam = ground_entities(None,torch.zeros(1,3,224,224),
                                  'Pulmonary edema.',['Edema'],projection='frontal')
    assert result['status']=='UNKNOWN' and cam is None
