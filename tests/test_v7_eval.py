"""Synthetic tests; no medical-accuracy claims."""
import numpy as np
from scripts.evaluate_classifier import evaluate_arrays
from scripts.class_balance import training_pos_weights
from scripts.research_readiness import check_files
from scripts.create_review_casebook import build_casebook


def test_metrics_handle_single_class_and_uncertain_masks():
    p=np.array([[.9,.2],[.1,.2],[.8,.1],[.2,.1]])
    y=np.array([[1,0],[0,0],[1,0],[0,0]])
    mask=np.array([[1,1],[1,1],[1,0],[1,0]])
    result=evaluate_arrays(p,y,mask,['Pneumonia','Edema'])
    assert result[0]['auroc']==1
    assert result[0]['f1']==1
    assert result[1]['auroc'] is None
    assert result[1]['known_labels']==2


def test_no_known_label_has_null_metrics():
    r=evaluate_arrays([[.8]],[[0]],[[0]],['Opacity'])[0]
    assert r['recall'] is None and r['note']


def test_training_weights_use_only_known_labels():
    rows=[('one',{'Pneumonia':'1.0'}),('two',{'Pneumonia':'0.0'}),
          ('three',{'Pneumonia':'0'}),('four',{'Pneumonia':'-1.0'})]
    from radreport.config import FINDINGS
    w=training_pos_weights(rows,cap=5)
    assert w[FINDINGS.index('Pneumonia')]==2


def test_readiness_never_certifies_clinical_uses(tmp_path):
    (tmp_path/'models').mkdir();(tmp_path/'models/generator.pt').touch()
    r=check_files(tmp_path)
    assert not r['clinical_use_authorized'] and not r['research_validation_complete']


def test_casebook_html_escaped_and_flags_normal_miss():
    s=build_casebook([{'uid':'<script>','reference':'Left lower lobe opacity.',
                       'generated':'Normal chest.'}])
    assert '&lt;script&gt;' in s and '<script>' not in s
    assert 'Normal-style draft' in s and 'NOT FOR CLINICAL USE' in s


def test_paired_comparison_detects_improvement_and_does_not_confuse_uid_order():
    from scripts.compare_generator_runs import compare
    a={'2':{'uid':'2','reference':'Left lower lobe opacity.','generated':'Normal chest.'},
       '1':{'uid':'1','reference':'No acute disease.','generated':'No acute disease.'}}
    b={'1':a['1'], '2':{'uid':'2','reference':'Left lower lobe opacity.','generated':'Left lower lobe opacity.'}}
    r=compare(a,b,seed=42,repetitions=40)
    assert r['n']==2
    assert r['candidate']['generic_normal_on_abnormal_reference_rate'] < r['baseline']['generic_normal_on_abnormal_reference_rate']


def test_paired_comparison_rejects_reference_change():
    from scripts.compare_generator_runs import compare
    import pytest
    with pytest.raises(ValueError,match='Reference reports changed'):
        compare({'1':{'uid':'1','reference':'a','generated':'b'}},
                {'1':{'uid':'1','reference':'other','generated':'b'}})
