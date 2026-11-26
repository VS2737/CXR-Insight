"""Fit per-finding linear probe on TRAIN; select thresholds on VAL only."""
import argparse,csv,json
from pathlib import Path
from radreport.v8_probe import prepare_probe_arrays,fit_probe,save_probe
from radreport.v8_cache import source_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',default='data/processed/labels.csv')
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--output',default='models_v8/finding_probe.json')
    a=p.parse_args()
    with open(a.labels,encoding='utf-8-sig',newline='') as f:
        records={(r['split'],str(r['uid'])):r for r in csv.DictReader(f)}
    train=prepare_probe_arrays('train',a.data_dir,a.features,records)
    val=prepare_probe_arrays('val',a.data_dir,a.features,records)
    result=fit_probe(*train[:3],*val[:3])
    manifest=source_manifest(a.data_dir)
    result['train_split_sha256']=manifest['train_sha256']
    result['val_split_sha256']=manifest['val_sha256']
    save_probe(result,a.output)
    print('Trained:',sum(v['trained'] for v in result['labels'].values()),'of',len(result['labels']),'finding probes.')
    print('UNAVAILABLE labels are deliberately not interpreted as negative.')
    for name,info in result['labels'].items():
        print(f'{name}:', 'F1='+str(info['val_f1']) if info['trained'] else 'ABSTAIN: '+info['reason'])
    print('Saved',a.output)

if __name__=='__main__':main()
