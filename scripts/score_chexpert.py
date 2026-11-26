"""CheXpert report-label agreement and entity-inconsistency proxy.

Supply labeler-produced CSVs from reference AND predicted reports, in
exactly the same order as results/predictions.csv. Neither measures image-level
hallucinations against independent radiologist-verified image ground truth.
"""
import argparse
import csv
import json
from pathlib import Path
from radreport.config import FINDINGS
from radreport.label_values import parse_binary_chexpert


def calculate_metrics(reference_rows, predicted_rows):
    from sklearn.metrics import f1_score
    if len(reference_rows) != len(predicted_rows) or not reference_rows:
        raise ValueError('Equal nonempty prediction and reference rows required.')
    labels_f1 = {}
    inconsistencies = 0
    for col in FINDINGS:
        true, pred = [], []
        for r, p in zip(reference_rows, predicted_rows):
            rv = parse_binary_chexpert(r.get(col, ''))
            pv = parse_binary_chexpert(p.get(col, ''))
            if rv is not None and pv is not None:
                true.append(rv); pred.append(pv)
            if pv == 1 and rv != 1:
                inconsistencies += 1
        if true:
            labels_f1[col] = float(f1_score(true, pred, zero_division=0))
    if not labels_f1: raise ValueError('No comparable definitive CheXpert labels.')
    return {'chexpert_macro_F1':round(sum(labels_f1.values())/len(labels_f1),4),
            'entity_inconsistency_proxy_per_report':round(inconsistencies/len(reference_rows),4),
            'n_reports':len(reference_rows),'per_label_f1':labels_f1,
            'caveat':'Text-to-text proxy only; does not independently verify image findings.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--references',required=True)
    p.add_argument('--predictions',required=True);p.add_argument('--output',default='results/chexpert_metrics.json')
    a=p.parse_args()
    with open(a.references,encoding='utf-8',newline='') as f: r=list(csv.DictReader(f))
    with open(a.predictions,encoding='utf-8',newline='') as f: g=list(csv.DictReader(f))
    m=calculate_metrics(r,g)
    o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(m,indent=2))
    print(json.dumps(m,indent=2))

if __name__=='__main__':main()
