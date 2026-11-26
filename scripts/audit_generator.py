"""Offline *text* audit of existing generator predictions, no patient data is saved.

Reference reports are imperfect and may have omissions. These are research
proxies, NOT sensitivity, specificity, diagnostic accuracy, or hallucination proof.
"""
import argparse
import csv
import json
from pathlib import Path
from radreport.safety import report_claims


def audit(rows):
    total = len(rows)
    if not total:
        raise ValueError('No prediction/reference pairs provided.')
    contradictions = 0
    ref_positive = generated_positive = 0
    overlap = 0
    unsupported_mentions = 0
    normal_on_ref_abnormal = 0
    ref_abnormal_cases = 0
    cases_with_uncovered = 0
    examples = []
    for row in rows:
        ref = report_claims(row.get('reference',''))
        gen = report_claims(row.get('generated',''))
        generated = {k for k,v in gen['findings'].items() if v=='PRESENT'}
        expected = {k for k,v in ref['findings'].items() if v=='PRESENT'}
        contradictions += int(bool(gen['contradictions']))
        ref_positive += len(expected)
        generated_positive += len(generated)
        overlap += len(expected & generated)
        unsupported_mentions += len(generated - expected)
        ref_abnormal = bool(expected or ref['uncovered_terms'])
        ref_abnormal_cases += int(ref_abnormal)
        generic_normal = (gen['normal_scope'] != 'NONE' and not generated)
        normal_on_ref_abnormal += int(ref_abnormal and generic_normal)
        cases_with_uncovered += int(bool(gen['uncovered_terms']))
        if (gen['contradictions'] or (ref_abnormal and generic_normal)) and len(examples)<8:
            examples.append({'uid':row.get('uid',''), 'issue':('internal_contradiction' if gen['contradictions']
                           else 'generic_normal_vs_reference_abnormal'),
                           'reference':row.get('reference',''), 'generated':row.get('generated','')})
    return {'n':total,
            'internally_contradictory_report_count':contradictions,
            'internally_contradictory_report_rate':round(contradictions/total,4),
            'reference_named_positive_mentions':ref_positive,
            'generated_named_positive_mentions':generated_positive,
            'matched_positive_mentions':overlap,
            'positive_mention_recall_proxy':round(overlap/ref_positive,4) if ref_positive else None,
            'generated_positives_not_in_reference':unsupported_mentions,
            'reference_abnormal_studies_by_text_heuristic':ref_abnormal_cases,
            'generic_normal_drafts_on_reference_abnormal':normal_on_ref_abnormal,
            'generic_normal_draft_rate_on_reference_abnormal':round(normal_on_ref_abnormal/ref_abnormal_cases,4)
                if ref_abnormal_cases else None,
            'generated_reports_with_uncovered_terms':cases_with_uncovered,
            'examples':examples,
            'warning':'Only a lexical comparison against non-adjudicated reference report text; not medical accuracy.'}


def load_csv(p):
    with open(p,newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    for k in ('uid','reference','generated'):
        if not rows or any(k not in r for r in rows):
            raise ValueError(f'Missing column {k} in predictions CSV.')
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',default='results/predictions.csv')
    p.add_argument('--compare', default=None, help='Optional second predictions.csv with same frozen UIDs.')
    p.add_argument('--output', default='results/generator_audit.json')
    a=p.parse_args()
    rows=load_csv(a.predictions)
    report={'baseline':audit(rows)}
    if a.compare:
        other=load_csv(a.compare)
        if [r['uid'] for r in rows] != [r['uid'] for r in other]:
            raise ValueError('UID order differs across predictions. Compare on exactly the same frozen test split.')
        report['candidate']=audit(other)
        for metric in ('internally_contradictory_report_rate','positive_mention_recall_proxy',
                       'generic_normal_draft_rate_on_reference_abnormal'):
            x=report['baseline'][metric]; y=report['candidate'][metric]
            report.setdefault('difference_candidate_minus_baseline',{})[metric]=round(y-x,4) if x is not None and y is not None else None
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
