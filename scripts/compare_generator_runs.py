"""Paired held-out generator comparison, with bootstrap uncertainty intervals.

The same UIDs and references must be present in both files. Never compare a
training split to a test split. Lexical matching is NOT clinical correctness.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from radreport.safety import report_claims


def _rows(filename):
    with open(filename,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if not rows or any(not {'uid','reference','generated'}.issubset(r) for r in rows):
        raise ValueError('Both CSVs require uid, reference, generated')
    if len({r['uid'] for r in rows})!=len(rows):
        raise ValueError('Duplicate UID: must be unique for paired evaluation')
    return {r['uid']:r for r in rows}


def _study_metrics(row):
    r=report_claims(row['reference']);g=report_claims(row['generated'])
    reference_positive={k for k,v in r['findings'].items() if v=='PRESENT'}
    generated_positive={k for k,v in g['findings'].items() if v=='PRESENT'}
    return {
        'internal_contradiction':int(bool(g['contradictions'])),
        'normal_draft_on_abnormal_reference':int(bool(reference_positive or r['uncovered_terms']) and
                                              g['normal_scope']!='NONE' and not generated_positive),
        'positive_mention_overlap':len(reference_positive & generated_positive),
        'positive_reference_mentions':len(reference_positive),
        'unsupported_positive_mentions':len(generated_positive-reference_positive),
    }


def compare(base, candidate, seed=42, repetitions=1000):
    if set(base)!=set(candidate):raise ValueError('Candidate and baseline must contain identical study UIDs')
    uids=sorted(base)
    if any(base[u]['reference']!=candidate[u]['reference'] for u in uids):
        raise ValueError('Reference reports changed between baseline and candidate')
    metrics=['internal_contradiction','normal_draft_on_abnormal_reference',
             'positive_mention_overlap','positive_reference_mentions','unsupported_positive_mentions']
    a=np.array([[ _study_metrics(base[u])[k] for k in metrics] for u in uids],dtype=float)
    b=np.array([[ _study_metrics(candidate[u])[k] for k in metrics] for u in uids],dtype=float)
    rng=np.random.default_rng(seed)
    def summary(x):
        totals=x.sum(axis=0);n=len(x)
        return {'contradiction_rate':totals[0]/n,
                'generic_normal_on_abnormal_reference_rate':totals[1]/n,
                'matched_reference_finding_mentions_per_study':totals[2]/n,
                'positive_mention_recall_proxy':totals[2]/totals[3] if totals[3]>0 else float('nan'),
                'unsupported_positive_mentions_per_study':totals[4]/n}
    before=summary(a);after=summary(b)
    distributions={k:[] for k in before}
    for _ in range(repetitions):
        indices=rng.integers(0,len(uids),len(uids))
        sa=summary(a[indices]);sb=summary(b[indices])
        for k in distributions:
            if np.isfinite(sa[k]) and np.isfinite(sb[k]):
                distributions[k].append(sb[k]-sa[k])
    differences={}
    for k in distributions:
        vals=np.array(distributions[k])
        differences[k]={'candidate_minus_baseline':round(float(after[k]-before[k]),4) if np.isfinite(after[k]-before[k]) else None,
                        'bootstrap_95pct_interval':[round(float(v),4) for v in np.quantile(vals,[.025,.975])] if len(vals) else None}
    return {'n':len(uids),'baseline':{k:round(float(v),4) if np.isfinite(v) else None for k,v in before.items()},
            'candidate':{k:round(float(v),4) if np.isfinite(v) else None for k,v in after.items()},
            'paired_difference':differences,'bootstrap_seed':seed,'bootstrap_repetitions':repetitions,
            'warning':'Lexical reference-report proxies only. A better score does not certify image-level diagnostic accuracy.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='results/predictions.csv')
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',default='results/paired_generator_comparison.json')
    p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args()
    result=compare(_rows(a.baseline),_rows(a.candidate),repetitions=a.bootstrap)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
