"""Honest research readiness snapshot; never asserts clinical certification."""
import json
import argparse
from pathlib import Path


def check_files(root):
    root=Path(root)
    artifacts={
        'report_generator_checkpoint':'models/generator.pt',
        'image_classifier_checkpoint':'models/classifier.pt',
        'test_manifest':'data/processed/test.jsonl',
        'generator_test_metrics':'results/evaluation.json',
        'generator_test_predictions':'results/predictions.csv',
        'classifier_weak_label_test_metrics':'results/classifier_weak_label_evaluation.json',
        'generator_text_audit':'results/generator_audit.json',
        'full_report_model_checkpoint':'models_structured/generator.pt',
    }
    found={name:(root/filename).is_file() for name,filename in artifacts.items()}
    generator={}
    eval_path=root/artifacts['generator_test_metrics']
    if eval_path.exists():
        try:
            generator=json.loads(eval_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeError):
            generator={'error':'Evaluation JSON unreadable'}
    return {'artifacts':found,
            'test_count':generator.get('n'),
            'bleu4':generator.get('BLEU4'),
            'rouge_l_f1':generator.get('ROUGE_L_F1'),
            'presentation_demo_ready':found['report_generator_checkpoint'] and found['image_classifier_checkpoint'],
            'research_validation_complete':False,
            'clinical_use_authorized':False,
            'missing_evidence':['Independent radiologist-reviewed, patient-level external test set',
                                'Validated finding-level false-negative and false-positive rates',
                                'Validated safety-gate sensitivity, specificity and abstention rate',
                                'Prospective calibration, anatomy grounding and drift assessment'],
            'warning':'Functional checkpoints and passing unit tests do not establish medical correctness.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--output',default='results/research_readiness.json')
    a=p.parse_args()
    info=check_files(a.root)
    out=Path(a.root)/a.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
