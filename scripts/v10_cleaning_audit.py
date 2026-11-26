"""Preview cleaned IU TRAIN/VALIDATION targets. No dataset modification."""
import argparse,json
from pathlib import Path
from radreport.data import load_jsonl
from radreport.v8_cache import verify_manifest_file
from radreport.v10_text import clean_pair,VERSION


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--output',default='results_v10/cleaning_audit.json')
    p.add_argument('--examples',type=int,default=5)
    a=p.parse_args()
    report={'cleaner':VERSION,'splits':{},'examples':[],
            'warning':'Only original frozen TRAIN and VAL are used; no raw data edited.'}
    for split in ('train','val'):
        verify_manifest_file(a.data_dir,split)
        rows=load_jsonl(Path(a.data_dir)/f'{split}.jsonl')
        stats=[]
        for row in rows:
            if not row.get('findings') or not row.get('impression'):continue
            cleaned,info=clean_pair(row)
            stats.append(info)
            if info['changed'] and len(report['examples'])<a.examples:
                report['examples'].append({'split':split,'uid':str(row['uid']),
                                          'original_findings':str(row.get('findings','')),
                                          'cleaned_target':cleaned})
        report['splits'][split]={'studies':len(rows),'complete_reports':len(stats),
                 'usable_after_cleaning':sum(x['usable'] for x in stats),
                 'changed_reports':sum(x['changed'] for x in stats),
                 'removed_clauses':sum(x['findings_removed']+x['impression_removed'] for x in stats),
                 'redacted_clauses':sum(x['redacted_clauses'] for x in stats)}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['splits'],indent=2));print('Saved:',path)

if __name__=='__main__':main()
