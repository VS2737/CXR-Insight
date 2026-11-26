"""Regression tests for the image + findings + labels report enhancement."""
import pytest
from fastapi.testclient import TestClient
from radreport.report_annotations import annotate_report
from radreport.formal_report import build_formal_report
from app.main import app


def test_positive_findings_and_tags_come_from_generated_text_only():
    r = annotate_report('FINDINGS:\nLeft lower lobe pneumonia. No pleural effusion.\nIMPRESSION:\nPneumonia.')
    assert 'Pneumonia' in {x['name'] for x in r['labels'] if x['polarity']=='PRESENT'}
    assert any('Left lower lobe pneumonia' in x for x in r['observations'])
    assert not any('No pleural effusion' in x for x in r['observations'])
    assert 'Pneumonia' in r['mesh_terms']
    assert 'Pleural Effusion' not in r['mesh_terms']


def test_negative_draft_does_not_become_diagnosed_normal_image():
    r = annotate_report('Impression: No acute cardiopulmonary findings.')
    assert r['observations'] == []
    assert r['normal_statement'] is not None
    assert r['mesh_terms'] == []
    assert all(x['polarity'] == 'UNVERIFIED' for x in r['labels'])


def test_report_derives_no_labels_when_no_inference():
    report = build_formal_report(raw='Left lower lobe pneumonia.', trained=False,
        projection='frontal', report_id='r1', model_version='none')
    assert report['observations']==[]
    assert report['mesh_terms']==[]
    assert report['labels']==[]
    assert report['image_signals']==[]
    assert not report['is_final']


def test_conflict_still_flagged_even_if_safety_arg_missing():
    report = build_formal_report(raw='FINDINGS:\nLeft lower lobe opacity. Lungs are clear.\nIMPRESSION:\nLeft lung opacity.',
        trained=True, projection='frontal', report_id='r2', model_version='v6')
    assert any('contradiction' in x.lower() for x in report['quality_flags'])
    assert 'Lung Diseases' in report['mesh_terms']
    assert report['review_status']=='MANDATORY_REVIEW'


def test_weak_image_scores_remain_separate():
    r = annotate_report('Impression: No acute cardiopulmonary abnormality.',
        {'Pneumonia':0.93,'Atelectasis':0.12})
    assert [x['name'] for x in r['image_signals']]==['Pneumonia']
    assert 'Pneumonia' not in [x['name'] for x in r['labels']]
    assert r['mesh_terms']==[]


def test_tags_not_made_from_out_of_ontology_terms():
    r=annotate_report('Impression: Possible emphysema.')
    assert r['mesh_terms']==[]
    assert 'emphysema' in r['uncovered_terms']


def test_page_has_two_view_print_layout_and_expected_sections():
    with TestClient(app) as client:
        s=client.get('/').text
    for item in ('Observations','MeSH Tags','Findings','Impression','Labels',
                 'r-primary-image','r-secondary-image','print-report'):
        assert item in s


def test_impression_only_report_marks_both_sections_unavailable():
    r = build_formal_report(raw='Impression: No acute process.', projection='frontal',
        report_id='b1', model_version='v5', trained=True)
    assert r['both_sections_generated'] is False
    assert 'Not generated separately' in r['findings']
