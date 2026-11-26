"""Attach official Stanford CheXpert labeler predictions to local record IDs.

Stanford labeler output columns may vary by revision; this requires the 14
canonical names and enforces row-alignment rather than guessing.
"""
import argparse
import csv
from pathlib import Path
from radreport.config import FINDINGS

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True, help='CSV produced by the official Stanford CheXpert labeler')
    p.add_argument('--mapping', default='data/processed/chexpert_uid_map.csv')
    p.add_argument('--output', default='data/processed/labels.csv')
    a = p.parse_args()
    with open(a.mapping, newline='', encoding='utf-8') as f: ids = list(csv.DictReader(f))
    with open(a.labels, newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
    if len(ids) != len(rows): raise ValueError('Labeler output row count does not match input mapping.')
    if any(col not in rows[0] for col in FINDINGS):
        raise ValueError('Labeler output must include all 14 canonical CheXpert observation columns.')
    with open(a.output, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['uid', 'split'] + FINDINGS)
        wr.writeheader()
        for meta, src in zip(ids, rows):
            wr.writerow({'uid': meta['uid'], 'split': meta['split'], **{k:src.get(k,'') for k in FINDINGS}})
    print('Merged structured labels to', a.output)
