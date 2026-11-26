import io
import json
import sqlite3
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.service import load_assets
import radreport.review_db as review_db
from radreport.config import DISCLAIMER


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(review_db,'DB_PATH',tmp_path/'test_reviews.sqlite3')
    monkeypatch.setattr('app.service.MODEL_DIR',tmp_path/'models')
    load_assets.cache_clear()
    with TestClient(app) as cl:yield cl
    load_assets.cache_clear()


def test_health_and_info_labelled(client):
    for endpoint in ('/api/health','/model-info','/api/reviews'):
        r=client.get(endpoint)
        assert r.status_code==200
        assert 'NOT FOR CLINICAL USE' in r.json()['disclaimer']
        assert 'PROHIBITED' in r.headers['X-Clinical-Use']


def test_untrained_mode_never_pretends_to_generate(client):
    bio=io.BytesIO();Image.new('RGB',(240,240),(120,120,120)).save(bio,format='PNG')
    r=client.post('/api/analyze',files={'file':('synthetic.png',bio.getvalue(),'image/png')})
    assert r.status_code==200,r.text
    d=r.json()
    assert d['model_loaded'] is False and d['draft_impression'] is None
    assert d['formal_report']['is_final'] is False
    assert d['formal_report']['report_source']=='NO_INFERENCE'
    assert 'NOT FOR CLINICAL USE' in d['formal_report']['disclaimer']
    assert d['safety']['review_required'] is True
    assert d['safety']['safety_check_passed'] is False
    assert 'NOT FOR CLINICAL USE' in d['disclaimer']
    rows=client.get('/api/reviews').json()['reviews']
    assert len(rows)==1
    assert 'PENDING'==rows[0]['state']
    with sqlite3.connect(review_db.DB_PATH) as db:
        assert 'raw_image' not in db.execute('SELECT sql FROM sqlite_master WHERE name="reviews"').fetchone()[0]


def test_invalid_image_rejected_and_labelled(client):
    r=client.post('/api/analyze',files={'file':('bad.png',b'not a real image','image/png')})
    assert r.status_code==422
    assert 'NOT FOR CLINICAL USE' in r.json()['disclaimer']


def test_signoff_requires_token(client):
    r=client.post('/api/reviews/fake/decision',json={'reviewer':'Test reviewer','decision':'ACCEPT_DRAFT'})
    assert r.status_code==403
    assert 'NOT FOR CLINICAL USE' in r.json()['disclaimer']
