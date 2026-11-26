"""Non-invasive readiness check for the coursework demonstration (no model downloads)."""
import json
from pathlib import Path
from radreport.config import ROOT, MODEL_DIR


def main():
    print('CXR Insight: presentation preflight')
    print('Project:',ROOT)
    print('Active model directory:',MODEL_DIR)
    gen=MODEL_DIR/'generator.pt'; clf=MODEL_DIR/'classifier.pt'
    print('Generator checkpoint:', 'PRESENT' if gen.is_file() else 'MISSING')
    print('Image classifier checkpoint:', 'PRESENT' if clf.is_file() else 'MISSING')
    metadata=MODEL_DIR/'generator_metadata.json'
    if metadata.is_file():
        d=json.loads(metadata.read_text(encoding='utf-8'))
        print('Target mode:',d.get('target_mode','impression (legacy checkpoint)'))
        print('Epoch count in checkpoint:',d.get('epochs_completed','not provided'))
        print('Clinical validation:',d.get('clinical_validation',False))
    data=ROOT/'data/processed/test.jsonl'
    print('Frozen test manifest:', 'PRESENT' if data.is_file() else 'MISSING')
    print('SAFETY: All outputs are RESEARCH DRAFTS, never final medical reports.')
    if not gen.is_file():
        print('BLOCKER: generator checkpoint missing; do not claim actual X-ray inference.')

if __name__=='__main__': main()
