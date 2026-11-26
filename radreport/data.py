"""IU X-Ray Kaggle csv parsing, split freezing and data leakage guardrails."""
import csv
import hashlib
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from PIL import Image


def read_iu(input_dir: str, group_col: str = 'patient_id'):
    root = Path(input_dir)
    reports, projections = root / 'indiana_reports.csv', root / 'indiana_projections.csv'
    if not reports.exists() or not projections.exists():
        raise FileNotFoundError('Download indiana_reports.csv and indiana_projections.csv from the IU Kaggle release.')
    image_dir = root / 'images' / 'images_normalized'
    if not image_dir.exists():
        # also accept a flat /images folder
        image_dir = root / 'images'
    with reports.open(encoding='utf-8-sig', newline='') as f:
        rr = {str(row['uid']).strip(): row for row in csv.DictReader(f)}
    with projections.open(encoding='utf-8-sig', newline='') as f:
        pp = list(csv.DictReader(f))
    by_uid = defaultdict(list)
    for p in pp:
        if 'frontal' in p.get('projection', '').lower():
            by_uid[str(p['uid']).strip()].append(p.get('filename', ''))
    output = []
    for uid, r in rr.items():
        report = (r.get('impression') or '').strip()
        if report.lower() in ('', 'nan', 'none'):
            continue
        for filename in by_uid.get(uid, [])[:1]:
            img_path = image_dir / filename
            if not img_path.is_file(): continue
            try:
                with Image.open(img_path) as im:
                    if im.width < 128 or im.height < 128: continue
                    im.verify()
            except Exception: continue
            # UID is study/report identifier; it is NOT automatically a patient identifier.
            pid = (r.get(group_col) or '').strip() if group_col in r else ''
            output.append({'uid': uid, 'patient_id': pid, 'image_path': str(img_path.resolve()),
                           'impression': report, 'findings': (r.get('findings') or '').strip(),
                           'projection': 'frontal'})
    if not output:
        raise ValueError('No usable frontal image/report pairs found. Check dataset layout.')
    return output


def freeze_splits(rows, output_dir, *, seed=42, require_patient_ids=True):
    if require_patient_ids and any(not r['patient_id'] for r in rows):
        raise ValueError('True patient-level split cannot be claimed: public IU Kaggle CSV does not provide verified patient IDs. '
                         'Supply patient_id or pass --allow-study-level-split and report this limitation explicitly.')
    split_kind = 'patient' if all(r['patient_id'] for r in rows) else 'study'
    groups = defaultdict(list)
    for row in rows:
        groups[(row['patient_id'] or row['uid'])].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    count = len(keys)
    n_train, n_val = int(count * .80), int(count * .10)
    bins = {'train': keys[:n_train], 'val': keys[n_train:n_train+n_val], 'test': keys[n_train+n_val:]}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': seed, 'split_level': split_kind, 'group_count': count, 'rows': len(rows)}
    for name, keys in bins.items():
        with (out / (name + '.jsonl')).open('w', encoding='utf-8') as f:
            for key in keys:
                for row in groups[key]:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
        manifest[name + '_sha256'] = hashlib.sha256((out / (name + '.jsonl')).read_bytes()).hexdigest()
        manifest[name + '_count'] = sum(len(groups[k]) for k in keys)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def load_jsonl(filename):
    with open(filename, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
