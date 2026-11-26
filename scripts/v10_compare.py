"""Paired v9/v10 report audit on the frozen IU VALIDATION split by default.

Use the validation results to select a research candidate; never use TEST
for development/threshold tuning. --final-test explicitly enables TEST only
after choosing the model, and does not establish clinical accuracy.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from radreport.data import load_jsonl
from radreport.formal_report import parse_generated
from radreport.v8_cache import load_feature,verify_manifest_file,verify_split_cache
from radreport.v9_joint import load_v9
from radreport.v9_concepts import compare_reference_generated
from radreport.safety import report_claims
from radreport.v10_text import artifact_flags
from scripts.v9_compare import summarize


def report_quality(rows):
    n=len(rows)
    flagged=sum(bool(artifact_flags(r['raw'])) for r in rows)
    contradictions=sum(bool(report_claims(r['generated'])['contradictions']) for r in rows)
    incomplete=sum(not r['complete_sections'] for r in rows)
    unsupported=sum(len(r['audit']['positive_generated_not_in_reference']) for r in rows)
    false_normal=0; ref_pos=0
    for r in rows:
        if r['audit']['reference_positive']:
            ref_pos+=1
            if report_claims(r['generated'])['normal_scope']!='NONE' and not r['audit']['matched_mentions']:
                false_normal+=1
    return {'artifact_rate':round(flagged/n,4),
            'internal_contradiction_rate':round(contradictions/n,4),
            'incomplete_sections_rate':round(incomplete/n,4),
            'generated_positive_mentions_not_in_reference_per_study':round(unsupported/n,4),
            'generic_normal_with_no_matched_positive_on_reference_positive_rate':round(false_normal/ref_pos,4) if ref_pos else None,
            'reference_positive_studies':ref_pos,
            'warning':'Text-based error proxies only: reference report is not independently verified image ground truth.'}


def paired_mention_ci(old,new,seed=42,repeats=1000):
    if len(old)!=len(new) or not all(a['uid']==b['uid'] for a,b in zip(old,new)):
        raise ValueError('paired predictions must have same study IDs/order')
    diffs=[]
    for a,b in zip(old,new):
        denom=len(a['audit']['reference_positive'])
        if denom:
            diffs.append((len(b['audit']['matched_mentions'])-len(a['audit']['matched_mentions']))/denom)
    if not diffs:
        return {'mean_delta_v10_minus_v9':None,'paired_ci95':None,'positive_study_count':0}
    d=np.array(diffs)
    rng=np.random.default_rng(seed)
    draws=[float(d[rng.integers(0,len(d),len(d))].mean()) for _ in range(repeats)]
    return {'mean_delta_v10_minus_v9':round(float(d.mean()),4),
            'paired_ci95':[round(float(q),4) for q in np.quantile(draws,[.025,.975])],
            'positive_study_count':len(d), 'repeats':repeats}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--v9',default='models_v9')
    p.add_argument('--v10',default='models_v10')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--final-test',action='store_true',help='Explicit final test audit only; never for iterative model selection')
    p.add_argument('--limit',type=int,default=0,help='Quick smoke test; omit for complete split')
    p.add_argument('--output',default='results_v10')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.split=='test' and not a.final_test:
        p.error('TEST split is held out: use --final-test only AFTER choosing the model using VAL')
    if a.final_test and a.split!='test':p.error('--final-test requires --split test')
    manifest=verify_manifest_file(a.data_dir,'train')
    verify_manifest_file(a.data_dir,'val')
    verify_manifest_file(a.data_dir,a.split)
    verify_split_cache(a.features,a.split,manifest)
    rows=load_jsonl(Path(a.data_dir)/f'{a.split}.jsonl')
    if a.limit: rows=rows[:a.limit]
    if not rows:raise ValueError('No rows available')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    predictions={}
    for version,directory in [('v9',a.v9),('v10',a.v10)]:
        model,meta,tokenizer=load_v9(directory,device)
        if (meta.get('train_split_sha256')!=manifest['train_sha256'] or
                meta.get('val_split_sha256')!=manifest['val_sha256']):
            raise ValueError(f'{version}: checkpoint not trained with current frozen train+val split')
        generated=[]
        for idx,row in enumerate(rows):
            _,tokens=load_feature(a.features,a.split,row['uid'])
            patch=torch.from_numpy(tokens).unsqueeze(0).to(device)
            torch.manual_seed(a.seed+idx)
            if device=='cuda':torch.cuda.manual_seed_all(a.seed+idx)
            raw=model.sample_reports(patch,tokenizer,n=5,max_new_tokens=192)[0]
            parsed=parse_generated(raw,structured_expected=True)
            reference='FINDINGS:\n'+str(row.get('findings') or '')+'\nIMPRESSION:\n'+str(row.get('impression') or '')
            gen='FINDINGS:\n'+parsed['findings']+'\nIMPRESSION:\n'+parsed['impression']
            generated.append({'uid':str(row['uid']), 'reference':reference,
                              'generated':gen, 'raw':raw, 'complete_sections':parsed['complete'],
                              'audit':compare_reference_generated(reference,gen)})
            if (idx+1)%25==0:print(f'{version} {idx+1}/{len(rows)}',flush=True)
        predictions[version]=generated
        del model
        if device=='cuda':torch.cuda.empty_cache()
    old,new=predictions['v9'],predictions['v10']
    if [r['uid'] for r in old] != [r['uid'] for r in new]:raise RuntimeError('Unpaired evaluation')
    output_csv=out/f'{a.split}_paired_casebook.csv'
    with output_csv.open('w',newline='',encoding='utf-8-sig') as f:
        names=['uid','reference','v9_generated','v10_generated','reference_positive',
               'v9_omitted','v10_omitted','v9_explicitly_negated','v10_explicitly_negated',
               'v9_unsupported_mentions','v10_unsupported_mentions','v9_artifact_flags','v10_artifact_flags']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for a9,a10 in zip(old,new):
            writer.writerow({'uid':a9['uid'],'reference':a9['reference'],
                'v9_generated':a9['generated'],'v10_generated':a10['generated'],
                'reference_positive':'; '.join(a9['audit']['reference_positive']),
                'v9_omitted':'; '.join(a9['audit']['omitted_mentions']),
                'v10_omitted':'; '.join(a10['audit']['omitted_mentions']),
                'v9_explicitly_negated':'; '.join(a9['audit']['explicitly_negated_reference_mentions']),
                'v10_explicitly_negated':'; '.join(a10['audit']['explicitly_negated_reference_mentions']),
                'v9_unsupported_mentions':'; '.join(a9['audit']['positive_generated_not_in_reference']),
                'v10_unsupported_mentions':'; '.join(a10['audit']['positive_generated_not_in_reference']),
                'v9_artifact_flags':'; '.join(artifact_flags(a9['raw'])),
                'v10_artifact_flags':'; '.join(artifact_flags(a10['raw']))})
    report={'split':a.split,'n':len(rows),'complete_frozen_split':a.limit==0,
            'train_sha256':manifest['train_sha256'],
            'validation_sha256':manifest['val_sha256'],
            'evaluation_sha256':manifest[a.split+'_sha256'],
            'v9':{**summarize(old),**report_quality(old)},
            'v10':{**summarize(new),**report_quality(new)},
            'paired_mention_recall':paired_mention_ci(old,new,seed=a.seed),
            'clinical_validation':False,
            'test_selection_prohibited':True,
            'warning':'Weak report-derived findings, study-level split, no image-adjudicated clinical accuracy. '+
                      ('TEST evaluation: do not make further model/hyperparameter choices based on this result.' if a.split=='test' else 'Validation-only development comparison.')}
    result_file=out/f'{a.split}_comparison.json'
    result_file.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Casebook:',output_csv,'Summary:',result_file)

if __name__=='__main__':main()
