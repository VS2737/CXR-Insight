import argparse
from radreport.data import read_iu, freeze_splits

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Prepare IU-XRay frontal-only JSONL splits.')
    p.add_argument('--input', default='data/indiana')
    p.add_argument('--output', default='data/processed')
    p.add_argument('--allow-study-level-split', action='store_true', help='Explicitly acknowledge UID != verified patient identity')
    args = p.parse_args()
    rows = read_iu(args.input)
    print(freeze_splits(rows, args.output, require_patient_ids=not args.allow_study_level_split))
