"""Configuration and non-clinical-use assertions."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv('RAD_MODEL_DIR', ROOT / 'models'))
DATA_DIR = Path(os.getenv('RAD_DATA_DIR', ROOT / 'data'))
DB_PATH = Path(os.getenv('RAD_DB_PATH', ROOT / 'data' / 'review_queue.sqlite3'))
DISCLAIMER = 'AI-GENERATED EDUCATIONAL RESEARCH DRAFT — NOT FOR CLINICAL USE. Requires qualified radiologist review.'
SELF_CONSISTENCY_THRESHOLD = float(os.getenv('SELF_CONSISTENCY_THRESHOLD', '0.70'))
IMAGE_ENTITY_THRESHOLD = float(os.getenv('IMAGE_ENTITY_THRESHOLD', '0.65'))
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_FORMATS = {'PNG', 'JPEG'}
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding']
