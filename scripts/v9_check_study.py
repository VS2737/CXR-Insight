"""Find a study across original frozen train/val/test manifests; never assume test."""
import argparse,json
from pathlib import Path
from radreport.data import load_jsonl
from radreport.v8_cache import verify_manifest_file
from radreport.v9_concepts import claims_for_audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--uid',default='2')
    p.add_argument('--data-dir',default='data/processed')
    a=p.parse_args();matches=[]
    for split in ('train','val','test'):
        verify_manifest_file(a.data_dir,split)
        for row in load_jsonl(Path(a.data_dir)/f'{split}.jsonl'):
            if str(row.get('uid'))==str(a.uid):
                text='\n'.join([str(row.get('findings') or ''),str(row.get('impression') or '')])
                matches.append({'split':split,'uid':a.uid,'findings':row.get('findings'),
                                'impression':row.get('impression'),
                                'positive_mentioned_findings':[k for k,v in claims_for_audit(text).items() if v=='PRESENT']})
    print(json.dumps({'study':a.uid,'matches':matches,'warning':'Text labels only; not image interpretation.'},indent=2))
if __name__=='__main__':main()
