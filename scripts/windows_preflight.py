"""A quick, offline-first check of the local Windows training environment.

Run from project root: python -m scripts.windows_preflight
"""
from pathlib import Path
import importlib.util
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]

def status(name, ok, message):
    print(f"[{'OK' if ok else 'CHECK'}] {name}: {message}")

def main():
    print('CXR Insight Windows training preflight\n')
    status('Python', sys.version_info[:2] == (3, 11), sys.version.split()[0] + ' (project recommends 3.11)')
    status('Platform', True, platform.platform())
    for module in ('torch', 'torchvision', 'transformers', 'PIL', 'sacrebleu', 'rouge_score', 'fastapi'):
        installed = importlib.util.find_spec(module) is not None
        status(f'Package {module}', installed, 'installed' if installed else 'missing')
    try:
        import torch
        status('GPU/CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA-enabled GPU visible to PyTorch')
    except Exception as e:
        status('GPU/CUDA', False, f'cannot check: {type(e).__name__}')
    root = ROOT/'data'/'indiana'
    for name in ('indiana_reports.csv', 'indiana_projections.csv'):
        path = root/name
        status('Dataset '+name, path.is_file(), str(path))
    image_roots = [root/'images'/'images_normalized', root/'images']
    image_found = next((p for p in image_roots if p.exists() and any(p.rglob('*.png'))), None)
    status('Dataset PNG X-rays', bool(image_found), str(image_found) if image_found else 'missing PNGs')
    data = ROOT/'data'/'processed'
    for name in ('train.jsonl','val.jsonl','test.jsonl','manifest.json'):
        status('Processed '+name, (data/name).is_file(), 'ready' if (data/name).is_file() else 'not yet prepared')
    for name in ('generator.pt','classifier.pt'):
        status('Checkpoint '+name,(ROOT/'models'/name).is_file(),'available' if (ROOT/'models'/name).is_file() else 'not yet trained')
    labels = data/'labels.csv'
    status('CheXpert weak labels', labels.is_file(), 'imported' if labels.is_file() else 'not yet imported')
    print('\nThis check does not claim a working medical model or completed evaluation.')

if __name__ == '__main__':
    main()
