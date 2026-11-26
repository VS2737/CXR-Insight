"""Evaluate frozen test split. Metrics measured, never hard-coded.

Report-based CheXpert-F1 additionally requires Stanford CheXpert on BOTH
hypotheses and references. This script exports text to support that workflow.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import torch
from PIL import Image
from radreport.data import load_jsonl
from radreport.model import load_generator, TRANSFORM
from radreport.formal_report import parse_generated


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/processed/test.jsonl')
    p.add_argument('--model-dir',default='models')
    p.add_argument('--output',default='results')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=42,help='Seed sampling per study for fairer comparisons.')
    a=p.parse_args()
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer
    from sklearn.metrics import accuracy_score
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model, meta, tokenizer=load_generator(a.model_dir,device)
    rows=load_jsonl(a.input)
    if a.limit: rows=rows[:a.limit]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    preds=[];refs=[];lat=[]; incomplete=0
    target_mode = meta.get('target_mode','impression')
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['uid','reference','generated','raw_report','seconds'])
        wr.writeheader()
        for n,row in enumerate(rows,1):
            with Image.open(row['image_path']) as im:
                inp=TRANSFORM(im.convert('RGB')).unsqueeze(0).to(device)
            torch.manual_seed(a.seed + n)
            if device == 'cuda': torch.cuda.manual_seed_all(a.seed + n)
            t=time.monotonic(); draft=model.sample_reports(inp,tokenizer,n=5,
                max_new_tokens=156 if target_mode=='structured' else 65)
            elapsed=time.monotonic()-t
            parsed = parse_generated(draft[0], structured_expected=target_mode=='structured')
            prediction=parsed['impression']
            incomplete += int(not parsed['complete'])
            wr.writerow({'uid':row['uid'],'reference':row['impression'],'generated':prediction,
                        'raw_report':draft[0],'seconds':elapsed})
            preds.append(prediction);refs.append(row['impression']);lat.append(elapsed)
            if n%20==0:print(f'{n}/{len(rows)}')
    if not preds:raise ValueError('Test split is empty')
    bleu=BLEU(effective_order=True).corpus_score(preds,[refs]).score/100
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    rouge=sum(scorer.score(ref,gen)['rougeL'].fmeasure for ref,gen in zip(refs,preds))/len(preds)
    summary={'n':len(preds),'sampling_seed':a.seed,'target_mode':target_mode,
             'incomplete_generated_reports':incomplete,'BLEU4':round(bleu,4),'ROUGE_L_F1':round(rouge,4),
             'mean_latency_seconds':round(sum(lat)/len(lat),2),
             'CheXpert_F1':'NOT_MEASURED: run official labeler on references and generated reports',
             'clinicalbert_score':'NOT_MEASURED: requires separate validated clinical embedding metric',
             'thresholds_met':'NOT ESTABLISHED until all specified safety and quality metrics measured',
             'checkpoint_metadata':meta}
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
