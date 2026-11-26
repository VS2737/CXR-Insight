from radreport.safety import extract_positive_entities, self_consistency, entity_consistency, route
from radreport.config import DISCLAIMER
from scripts.quality_gate import check


def test_negation_does_not_claim_pneumonia():
    assert 'Pneumonia' not in extract_positive_entities('No evidence of pneumonia.')


def test_detects_positive_pneumonia():
    assert 'Pneumonia' in extract_positive_entities('Right lower lobe pneumonia present.')


def test_missing_classifier_requires_review():
    x = route('Right lower lobe pneumonia.', ['Right lower lobe pneumonia.']*5, None, None, True)
    assert x['review_required'] and x['entity_check']['status']=='UNKNOWN'
    assert not x['safety_check_passed']


def test_check_specific_unsupported_entity():
    result=entity_consistency('Pneumonia and cardiomegaly.',{'Pneumonia':.80,'Cardiomegaly':.03})
    assert 'Cardiomegaly' in result['unsupported']
    assert 'Pneumonia' not in result['unsupported']


def test_consistency_identical():
    score=self_consistency(['Right lower lobe pneumonia']*5)
    assert score == 1.0


def test_quality_gate_never_passes_missing_metrics():
    assert check({'BLEU4':.99, 'ROUGE_L_F1':.99})


def test_disclaimer():
    assert 'NOT FOR CLINICAL USE' in DISCLAIMER
