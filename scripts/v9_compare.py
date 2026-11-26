"""Frozen IU TEST v8 vs v9 paired audit (no test-time model selection).

BLEU, ROUGE, mention recall, omissions and paired bootstrap CIs. These are
report-text metrics only, NOT independent clinical diagnostic accuracy.
"""
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from radreport.data import load_jsonl
from radreport.formal_report import parse_generated
from radreport.v8_cache import load_feature,verify_manifest_file,verify_split_cache
from radreport.v8_generator import load_v8
from radreport.v9_joint import load_v9
from radreport.v9_concepts import compare_reference_generated


def summarize(rows):
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer
    refs=[r['reference'] for r in rows]
    pred=[r['generated'] for r in rows]
    rouge=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    bleu=BLEU(effective_order=True).corpus_score(pred,[refs]).score/100
    recall=[];explicit=[];mismatch=[];ref_abnormal=0
    for r in rows:
        audit=compare_reference_generated(r['reference'],r['generated'])
        pos=len(audit['reference_positive'])
        if pos:
            ref_abnormal+=1
            recall.append(len(audit['matched_mentions'])/pos)
            explicit.append(int(bool(audit['explicitly_negated_reference_mentions'])))
            mismatch.append(int(bool(audit['omitted_mentions'])))
    return {'n':len(rows),'BLEU4':round(float(bleu),4),
            'ROUGE_L_F1':round(float(np.mean([rouge.score(ref,gen)['rougeL'].fmeasure for ref,gen in zip(refs,pred)])),4),
            'reference_positive_studies':ref_abnormal,
            'mean_reference_positive_mention_recall':round(float(np.mean(recall)),4) if recall else None,
            'reference_positive_studies_with_omissions_rate':round(float(np.mean(mismatch)),4) if mismatch else None,
            'reference_positive_studies_with_explicit_negations_rate':round(float(np.mean(explicit)),4) if explicit else None}


def paired_bootstrap_delta(v8,v9,seed=42,repeats=500):
    """CI of paired per-study changes in finding mention recall, not clinical sensitivity."""
    a=[];b=[]
    for x,y in zip(v8,v9):
        rx=compare_reference_generated(x['reference'],x['generated'])
        ry=compare_reference_generated(y['reference'],y['generated'])
        denom=len(rx['reference_positive'])
        if denom:
            a.append(len(rx['matched_mentions'])/denom)
            b.append(len(ry['matched_mentions'])/denom)
    if not a:return {'delta':None,'ci95':None,'n_reference_positive':0}
    diff=np.asarray(b)-np.asarray(a)
    rng=np.random.default_rng(seed)
    draws=[float(np.mean(diff[rng.integers(0,len(diff),size=len(diff))])) for _ in range(repeats)]
    return {'delta_v9_minus_v8':round(float(diff.mean()),4),
            'paired_bootstrap_ci95':[round(float(v),4) for v in np.quantile(draws,[.025,.975])],
            'n_reference_positive':len(diff),'repeats':repeats}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--v8',default='models_v8')
    p.add_argument('--v9',default='models_v9')
    p.add_argument('--output',default='results_v9')
    p.add_argument('--limit',type=int,default=0,help='Smoke test only. Omit for the entire 366-study test set.')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    manifest=verify_manifest_file(a.data_dir,'test')
    verify_manifest_file(a.data_dir,'train')
    verify_manifest_file(a.data_dir,'val')
    verify_split_cache(a.features,'test',manifest)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=load_jsonl(Path(a.data_dir)/'test.jsonl')
    rows=rows[:a.limit] if a.limit else rows
    if not rows:raise ValueError('No test studies')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    outputs={}
    for label,path,loader in [('v8',a.v8,load_v8),('v9',a.v9,load_v9)]:
        model,meta,tok=loader(path,device)
        if meta.get('train_split_sha256')!=manifest['train_sha256'] or meta.get('val_split_sha256')!=manifest['val_sha256']:
            raise ValueError(f'{label} checkpoint trained on different frozen splits')
        generated=[]
        for idx,row in enumerate(rows):
            _,features=load_feature(a.features,'test',row['uid'])
            patches=torch.from_numpy(features).unsqueeze(0).to(device)
            torch.manual_seed(a.seed+idx)
            if device=='cuda':torch.cuda.manual_seed_all(a.seed+idx)
            raw=model.sample_reports(patches,tok,n=5,max_new_tokens=168)[0]
            parsed=parse_generated(raw,structured_expected=True)
            reference='FINDINGS:\n'+str(row.get('findings') or '')+'\nIMPRESSION:\n'+str(row.get('impression') or '')
            gen='FINDINGS:\n'+parsed['findings']+'\nIMPRESSION:\n'+parsed['impression']
            audit=compare_reference_generated(reference,gen)
            generated.append({'uid':str(row['uid']),'reference':reference,'generated':gen,'raw':raw,
                              'complete_sections':parsed['complete'],'audit':audit})
            if (idx+1)%25==0:print(f'{label}: {idx+1}/{len(rows)}',flush=True)
        outputs[label]=generated
        del model
        if device=='cuda':torch.cuda.empty_cache()
    if [x['uid'] for x in outputs['v8']]!=[x['uid'] for x in outputs['v9']]:
        raise ValueError('Paired test studies do not match')
    with (out/'paired_casebook.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['uid','reference','v8_generated','v9_generated',
            'reference_positive','v8_omitted','v9_omitted','v8_explicitly_negated','v9_explicitly_negated'])
        writer.writeheader()
        for old,new in zip(outputs['v8'],outputs['v9']):
            writer.writerow({'uid':old['uid'],'reference':old['reference'],
                 'v8_generated':old['generated'],'v9_generated':new['generated'],
                 'reference_positive':'; '.join(old['audit']['reference_positive']),
                 'v8_omitted':'; '.join(old['audit']['omitted_mentions']),
                 'v9_omitted':'; '.join(new['audit']['omitted_mentions']),
                 'v8_explicitly_negated':'; '.join(old['audit']['explicitly_negated_reference_mentions']),
                 'v9_explicitly_negated':'; '.join(new['audit']['explicitly_negated_reference_mentions'])})
    report={'split':'test','n':len(rows),'full_frozen_test_set':not bool(a.limit) and len(rows)==366,
            'train_sha256':manifest['train_sha256'],'test_sha256':manifest['test_sha256'],
            'seed':a.seed,'v8':summarize(outputs['v8']),
            'v9':summarize(outputs['v9']),
            'paired_mention_recall':paired_bootstrap_delta(outputs['v8'],outputs['v9'],a.seed),
            'clinical_validation':False,
            'warning':'All label comparisons are to report TEXT. Images have not been independently adjudicated. IU splits are study-level.'}
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
