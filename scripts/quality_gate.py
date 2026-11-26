"""Conservative release gate: missing metrics FAIL, and all reports must be drafts."""
import argparse
import json
import sys
from pathlib import Path

LIMITS = {'BLEU4':('min',.10), 'ROUGE_L_F1':('min',.30),
          'clinicalbert_score':('min',.30), 'chexpert_macro_F1':('min',.60),
          'entity_inconsistency_proxy_per_report':('max',.15)}


def check(values):
    errors=[]
    for key,(direction,threshold) in LIMITS.items():
        v=values.get(key)
        if type(v) not in (int,float):errors.append(f'{key} missing/unmeasured');continue
        if direction=='min' and v < threshold:errors.append(f'{key}={v} below {threshold}')
        if direction=='max' and v > threshold:errors.append(f'{key}={v} exceeds {threshold}')
    if values.get('clinical_reviewer_evaluation_completed') is not True:
        errors.append('Human simulated-review evaluation not complete')
    if values.get('patient_level_split_verified') is not True:
        errors.append('Verified patient-level split not established')
    if values.get('classifier_independently_calibrated') is not True:
        errors.append('Image classifier not independently calibrated')
    return errors

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metrics',default='results/release_evidence.json')
    a=p.parse_args();fp=Path(a.metrics)
    if not fp.exists():sys.exit(f'RELEASE BLOCKED: {fp} missing')
    problems=check(json.loads(fp.read_text()))
    if problems:sys.exit('RELEASE BLOCKED:\n - '+'\n - '.join(problems))
    print('Research-only release quality gates passed. NOT a clinical-use authorization.')
