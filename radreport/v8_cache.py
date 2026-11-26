"""Cache paths and source split validation for RAD-DINO features."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def feature_path(root, split, uid):
    if split not in ('train','val','test'):
        raise ValueError('Expected train, val or test split')
    digest = hashlib.sha256(str(uid).encode('utf-8')).hexdigest()
    return Path(root) / split / f'{digest}.npz'


def verify_split_cache(root, split, manifest):
    meta_file = Path(root) / split / 'cache_info.json'
    if not meta_file.is_file():
        raise FileNotFoundError(f'{meta_file} missing. Build RAD-DINO feature cache first.')
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    if meta.get('split_sha256') != manifest.get(split + '_sha256'):
        raise ValueError(f'{split} feature cache does not match frozen data split')
    if meta.get('model') != 'microsoft/rad-dino' or meta.get('grid') != 7:
        raise ValueError('Unexpected feature cache encoder version')
    return meta


def load_feature(root, split, uid):
    file = feature_path(root, split, uid)
    if not file.is_file():
        raise FileNotFoundError(f'Feature cache missing for study {uid} in {split}: {file}')
    with np.load(file, allow_pickle=False) as f:
        cls = np.asarray(f['cls'], dtype=np.float32)
        tokens = np.asarray(f['tokens'], dtype=np.float32)
    if cls.shape != (768,) or tokens.shape != (49,768) or not np.isfinite(tokens).all() or not np.isfinite(cls).all():
        raise ValueError(f'Invalid RAD-DINO cache tensors: {file}')
    return cls, tokens


def source_manifest(root):
    return json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))


def verify_manifest_file(data_dir, split):
    """Reject modified frozen splits before model fitting or evaluation."""
    manifest = source_manifest(data_dir)
    expected = manifest.get(split + '_sha256')
    target = Path(data_dir) / f'{split}.jsonl'
    if not expected or not target.is_file():
        raise ValueError(f'Frozen {split} split and SHA256 manifest required')
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'{split} JSONL no longer matches frozen split checksum')
    return manifest
