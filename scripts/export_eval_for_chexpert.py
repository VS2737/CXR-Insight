"""Export evaluation report pairs for the official Stanford CheXpert labeler.

Inputs retain the same row order. Both outputs are headerless 1-column CSVs,
which the Stanford CheXpert NLP labeler expects. This merely prepares text;
no fake / inferred medical labels are produced.
"""
import argparse
import csv
from pathlib import Path


def export(predictions, output):
    with open(predictions, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('No evaluation predictions found')
    if not {'reference', 'generated'}.issubset(rows[0]):
        raise ValueError('Expected reference and generated columns')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for field, name in [('reference', 'chexpert_reference_input.csv'),
                        ('generated', 'chexpert_generated_input.csv')]:
        path = out / name
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow([row[field]])
        files[field] = path
    return len(rows), files


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',default='results/predictions.csv')
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    n, files=export(args.predictions,args.output)
    print(f'Exported {n} aligned reference/generated reports for external CheXpert text labelling:')
    for name, path in files.items(): print(f'{name}: {path}')
