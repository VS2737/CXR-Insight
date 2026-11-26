"""Tests for professional-style but strictly nonclinical report rendering."""
import pytest
from radreport.formal_report import parse_generated, build_formal_report, training_target


def test_original_impression_is_kept_without_invented_findings():
    result = parse_generated('Impression: No acute pulmonary disease.')
    assert result['impression'] == 'No acute pulmonary disease.'
    assert 'Not generated separately' in result['findings']
    assert result['complete'] is True


def test_structured_model_sections_are_preserved():
    r = parse_generated('FINDINGS:\nLeft lower lobe opacity.\nIMPRESSION:\nPossible airspace disease.', True)
    assert r['findings'] == 'Left lower lobe opacity.'
    assert r['impression'] == 'Possible airspace disease.'
    assert r['complete'] is True
    assert r['source'] == 'MODEL_GENERATED_BOTH'


def test_both_sections_required_when_structured_expected():
    r = parse_generated('IMPRESSION:\nNo acute process.', True)
    assert not r['complete']
    assert r['source'] == 'STRUCTURED_OUTPUT_MISSING'


def test_missing_impression_section_never_filled_with_guess():
    r = parse_generated('FINDINGS:\nLeft lower lobe airspace opacity.', True)
    assert r['impression'] == 'Not generated.'
    assert not r['complete']


def test_report_does_not_invent_clinical_history_or_comparison():
    r = build_formal_report(raw='Impression: Mild cardiomegaly.', projection='frontal',
        report_id='sample123', model_version='test-v1', trained=True)
    assert r['clinical_history'] == 'Not supplied'
    assert r['comparison'] == 'No prior imaging supplied'
    assert 'not independently verified' in r['technique']
    assert 'Not provided' in r['patient_details']
    assert r['is_final'] is False
    assert 'NOT FOR CLINICAL USE' in r['disclaimer']


def test_untrained_model_has_no_generated_diagnosis():
    r = build_formal_report(raw='Pneumonia', projection='unknown', report_id='id',
        model_version='untrained', trained=False)
    assert r['review_status'] == 'NO_INFERENCE'
    assert r['report_source'] == 'NO_INFERENCE'
    assert 'No model-generated impression' in r['impression']


def test_training_requires_real_findings_for_structured():
    with pytest.raises(ValueError):
        training_target({'findings':'', 'impression':'No acute disease.'}, mode='structured')


def test_training_preserves_iu_findings_and_impression():
    text = training_target({'findings':'Small left effusion.','impression':'Left pleural effusion.'},mode='structured')
    assert text == 'FINDINGS:\nSmall left effusion.\nIMPRESSION:\nLeft pleural effusion.'


def test_report_exposes_review_reasons_without_claiming_safety():
    safety={'status':'MANDATORY_REVIEW','review_reasons':['Possible omission'],
        'entity_check':{'contradictions':[{'finding':'Lung Opacity'}]}}
    r=build_formal_report(raw='Impression: Lungs clear. Left lung opacity.', projection='frontal',
        report_id='1',model_version='v5',trained=True,safety=safety)
    assert any('contradiction' in x.lower() for x in r['quality_flags'])
    assert 'Possible omission' in r['quality_flags']
    assert r['review_status']=='MANDATORY_REVIEW'


def test_lateral_is_not_mislabeled_as_frontal():
    r=build_formal_report(raw='Impression: No acute disease.', projection='lateral',
        report_id='2',model_version='v5',trained=True)
    assert 'Lateral' in r['technique']
    assert 'Frontal' not in r['technique']
