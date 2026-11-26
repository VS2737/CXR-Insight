"""Protect against the common CheXpert 0.0/1.0 import bug."""
import pytest
from radreport.label_values import parse_binary_chexpert

@pytest.mark.parametrize('raw,expected', [
    ('1', 1), ('1.0', 1), ('0', 0), ('0.0', 0),
    ('-1', None), ('-1.0', None), ('', None), ('nan', None),
    (None, None), ('invalid', None),
])
def test_chexpert_floats(raw, expected):
    assert parse_binary_chexpert(raw) == expected

def test_chexpert_scoring_uses_official_float_encoding():
    from scripts.score_chexpert import calculate_metrics
    a={'Pneumonia':'1.0', 'No Finding':'0.0'}
    b={'Pneumonia':'1.0', 'No Finding':'0.0'}
    result=calculate_metrics([a],[b])
    assert result['chexpert_macro_F1'] == 0.5  # zero F1 for shared negative No Finding class
    assert result['entity_inconsistency_proxy_per_report'] == 0
