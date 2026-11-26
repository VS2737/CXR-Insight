"""Compute frozen RAD-DINO features on the SAME IU dataset; resumable.

Example: python -m scripts.v8_cache_rad_dino --split all
Cached files contain vectors, not X-ray pixel data. No model fitting occurs here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from radreport.data import load_jsonl
from radreport.v8_cache import feature_path, verify_manifest_file


def cache_split(split, root, output, encoder, overwrite=False):
    manifest = verify_manifest_file(root,split)
    rows = load_jsonl(Path(root) / f'{split}.jsonl')
    out = Path(output) / split
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, row in enumerate(rows, 1):
        dest = feature_path(output, split, row['uid'])
        if dest.exists() and not overwrite:
            count += 1
            continue
        with Image.open(row['image_path']) as im:
            cls, tokens = encoder.encode(im.convert('RGB'))
        if cls.shape != (768,) or tokens.shape != (49,768):
            raise ValueError('Unexpected RAD-DINO dimension; check model checkpoint')
        temp = dest.with_suffix('.tmp.npz')
        np.savez_compressed(temp, cls=cls.astype('float16'), tokens=tokens.astype('float16'))
        temp.replace(dest)
        count += 1
        if i % 25 == 0:
            print(f'{split}: {i}/{len(rows)} (features saved or reused)', flush=True)
    metadata={'model':'microsoft/rad-dino','grid':7,'split':split,
              'split_sha256':manifest.get(split+'_sha256'),'n_studies':len(rows),
              'num_cached':count,'note':'Pretrained on external chest X-ray data; IU train/val/test source unchanged.'}
    (out / 'cache_info.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--split',choices=['train','val','test','all'],default='all')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    import torch
    from radreport.v8_vision import RadDinoEncoder
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading microsoft/rad-dino on',device,'(downloads weights on first use).')
    encoder=RadDinoEncoder(device)
    for split in (['train','val','test'] if a.split=='all' else [a.split]):
        cache_split(split,a.data_dir,a.features,encoder,a.overwrite)

if __name__=='__main__':main()
