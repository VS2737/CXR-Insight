"""Evaluate RAD-DINO generator on held-out IU test studies.

Produces the SAME predictions CSV format as v7 so paired comparisons are
possible. Medical accuracy must not be inferred from lexical report similarity.
"""
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from radreport.data import load_jsonl
from radreport.v8_cache import load_feature,verify_split_cache,verify_manifest_file
from radreport.v8_generator import load_v8
from radreport.formal_report import parse_generated
from radreport.safety import report_claims


def quality_proxies(refs,preds):
    missed=0;disagree=0;contradictions=0;ref_pos=0;matched=0;generated_pos=0
    for ref,pred in zip(refs,preds):
        r=report_claims(ref);g=report_claims(pred)
        R={k for k,v in r['findings'].items() if v=='PRESENT'}
        G={k for k,v in g['findings'].items() if v=='PRESENT'}
        contradictions+=int(bool(g['contradictions']))
        missed+=int(bool(R) and not bool(G) and g['normal_scope']!='NONE')
        disagree+=len(G-R)
        matched+=len(R&G);ref_pos+=len(R);generated_pos+=len(G)
    n=len(preds)
    return {'internal_contradiction_rate':round(contradictions/n,4),
            'generic_normal_on_mentioned_abnormal_reference_rate':round(missed/n,4),
            'unsupported_text_finding_mentions_per_study':round(disagree/n,4),
            'text_finding_mention_recall_proxy':round(matched/ref_pos,4) if ref_pos else None,
            'text_finding_mention_precision_proxy':round(matched/generated_pos,4) if generated_pos else None,
            'warning':'Report-text claim matching only; not image-grounded medical labels.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/processed/test.jsonl')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--model-dir',default='models_v8')
    p.add_argument('--output',default='results_v8')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer
    split=Path(a.input).stem
    if split not in ('val','test'):
        raise ValueError('Only frozen validation/test split supported; train evaluation is not comparable')
    root=Path(a.input).parent
    manifest=verify_manifest_file(root,split)
    verify_manifest_file(root,'train')
    verify_split_cache(a.features,split,manifest)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,meta,tok=load_v8(a.model_dir,device)
    if meta.get('train_split_sha256')!=manifest['train_sha256']:
        raise ValueError('v8 checkpoint trained on a different IU train split')
    rows=load_jsonl(a.input)
    if a.limit: rows=rows[:a.limit]
    if not rows:raise ValueError('No evaluation studies')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    preds=[];refs=[];took=[];incomplete=0
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['uid','reference','generated','raw_report','seconds'])
        writer.writeheader()
        for i,row in enumerate(rows,1):
            _,tokens=load_feature(a.features,split,row['uid'])
            patch=torch.from_numpy(tokens).unsqueeze(0).to(device)
            torch.manual_seed(a.seed+i)
            if device=='cuda': torch.cuda.manual_seed_all(a.seed+i)
            t=time.monotonic()
            sampled=model.sample_reports(patch,tok,n=5,max_new_tokens=168)
            sec=time.monotonic()-t
            parsed=parse_generated(sampled[0],structured_expected=True)
            incomplete+=int(not parsed['complete'])
            impression=parsed['impression']
            writer.writerow({'uid':row['uid'],'reference':row['impression'],
                             'generated':impression,'raw_report':sampled[0],'seconds':sec})
            preds.append(impression);refs.append(row['impression']);took.append(sec)
            if i%20==0:print(f'{i}/{len(rows)}')
    bleu=BLEU(effective_order=True).corpus_score(preds,[refs]).score/100
    rouge=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    score=float(np.mean([rouge.score(ref,pred)['rougeL'].fmeasure for ref,pred in zip(refs,preds)]))
    result={'split':split,'n':len(preds),'full_split':len(rows)==len(load_jsonl(a.input)),
            'train_split_sha256':manifest['train_sha256'],
            'eval_split_sha256':manifest[split+'_sha256'],
            'seed':a.seed,'model_version':meta['version'],
            'BLEU4':round(bleu,4),'ROUGE_L_F1':round(score,4),
            'mean_latency_seconds':round(float(np.mean(took)),3),
            'structured_reports_missing_sections':incomplete,
            'report_claim_proxies':quality_proxies(refs,preds),
            'unmeasured':['independent expert diagnosis accuracy','external validation','CheXbert-F1','clinical sensitivity'],
            'warning':'IU study-level split; automatically generated reports require expert review.'}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
