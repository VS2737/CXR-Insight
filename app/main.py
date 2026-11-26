"""FastAPI nonclinical research prototype, with demo-safe behavior pre-training."""
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError
from radreport.config import DISCLAIMER, MAX_UPLOAD_BYTES, SELF_CONSISTENCY_THRESHOLD, IMAGE_ENTITY_THRESHOLD
from radreport.review_db import list_reviews, decide
from .service import analyze, InvalidImage, load_assets

BASE = Path(__file__).resolve().parent
app = FastAPI(title='CXR Report Research Prototype', version='0.1.0',
              description='Educational research only; this system is NOT suitable for clinical decisions.')
app.mount('/static', StaticFiles(directory=str(BASE / 'static')), name='static')


@app.middleware('http')
async def safety_headers(request, call_next):
    response = await call_next(request)
    response.headers['X-Clinical-Use'] = 'PROHIBITED - Educational research prototype'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request, exc):
    return JSONResponse(status_code=exc.status_code, content={'error': str(exc.detail), 'disclaimer': DISCLAIMER})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request, exc):
    return JSONResponse(status_code=422, content={'error': 'Invalid request input', 'disclaimer':DISCLAIMER})


@app.get('/', response_class=HTMLResponse)
async def home():
    return (BASE / 'templates' / 'index.html').read_text(encoding='utf-8')


@app.get('/api/health')
async def health():
    return {'ok':True, 'disclaimer':DISCLAIMER}


@app.get('/model-info')
async def model_info():
    assets = load_assets()
    return {'model_version':assets['version'], 'model_loaded':assets['ready'],
            'training_data_version':assets.get('metadata', {}).get('training_manifest'),
            'calibration_parameters':{'self_consistency_threshold':SELF_CONSISTENCY_THRESHOLD,
                                      'image_entity_threshold':IMAGE_ENTITY_THRESHOLD,
                                      'classifier_calibrated':assets.get('classifier_metadata',{}).get('calibrated',False)},
            'reason':assets.get('reason'), 'target_mode':assets.get('metadata',{}).get('target_mode','impression'),
            'architecture':assets.get('model_family','efficientnet-b0-gpt2-v1'),
            'external_pretraining_used':bool(assets.get('metadata',{}).get('uses_external_pretraining',False)),
            'not_for_clinical_use':True,
            'disclaimer':DISCLAIMER}


@app.post('/api/analyze')
async def api_analyze(file: UploadFile = File(...), projection: str = Form('unknown')):
    if file.content_type not in ('image/png', 'image/jpeg', 'image/jpg'):
        raise HTTPException(415, 'Only PNG or JPEG files supported')
    if projection not in ('unknown', 'frontal', 'lateral'):
        raise HTTPException(422, 'Projection must be unknown, frontal, or lateral.')
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'Image exceeds 12 MB limit')
    try:
        return analyze(payload, projection=projection)
    except InvalidImage as e:
        raise HTTPException(422, str(e))


@app.get('/api/reviews')
async def reviews():
    return {'reviews':list_reviews(), 'notice':'Metadata-only queue; uploaded X-rays and draft text are not saved.',
            'disclaimer': DISCLAIMER}


class ReviewDecision(BaseModel):
    reviewer: str = Field(min_length=2, max_length=120)
    decision: str


@app.post('/api/reviews/{review_id}/decision')
async def review_decision(review_id: str, payload: ReviewDecision, x_review_token: str | None = Header(default=None)):
    expected = os.environ.get('REVIEW_TOKEN')
    if not expected or x_review_token != expected:
        raise HTTPException(403, 'Review approval requires a configured REVIEW_TOKEN; demo has no clinical reviewers.')
    try:
        ok = decide(review_id, payload.reviewer, payload.decision)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not ok: raise HTTPException(404, 'Pending review not found')
    return {'updated':True, 'clinical_approval':False,
            'notice':'Simulated review sign-off only; NOT clinical authorization.', 'disclaimer': DISCLAIMER}
