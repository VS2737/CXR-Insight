"""Prepare text input and UID mapping for Stanford CheXpert labeler (external)."""
import csv
from pathlib import Path
from radreport.data import load_jsonl

if __name__ == '__main__':
    root = Path('data/processed')
    rows = []
    for split in ('train', 'val', 'test'):
        for row in load_jsonl(root / f'{split}.jsonl'):
            rows.append({'uid': row['uid'], 'split': split, 'report': row['impression']})
    with (root / 'chexpert_input.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        for r in rows: writer.writerow([r['report']])  # headerless per labeler specification
    with (root / 'chexpert_uid_map.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=['uid', 'split'])
        writer.writeheader(); writer.writerows({'uid':r['uid'], 'split':r['split']} for r in rows)
    print(f'Exported {len(rows)} records. Run official Stanford labeler on chexpert_input.csv, then import its output.')
