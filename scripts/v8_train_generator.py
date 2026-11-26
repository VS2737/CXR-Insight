"""Train radiology-specific RAD-DINO -> GPT-2 cross-attention model.

Same IU data, frozen RAD-DINO visual features, no changes to models/generator.pt.
Windows RTX 4060 recommended: --epochs 8 --batch-size 1 --accumulation 8
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from radreport.data import load_jsonl
from radreport.formal_report import training_target
from radreport.v8_cache import source_manifest, verify_manifest_file, verify_split_cache, load_feature
from radreport.v8_generator import build_v8_model, get_tokenizer, save_v8


class FeatureReportDataset(Dataset):
    def __init__(self, root, features, split, tokenizer, max_length=176):
        self.root=Path(root);self.features=Path(features);self.split=split
        manifest=verify_manifest_file(root,split)
        verify_split_cache(features, split, manifest)
        self.rows = [r for r in load_jsonl(self.root / f'{split}.jsonl')
                     if r.get('findings','').strip().lower() not in ('', 'nan', 'none')
                     and r.get('impression','').strip().lower() not in ('', 'nan', 'none')]
        if not self.rows:
            raise ValueError(f'No complete Findings+Impression reports in {split}')
        self.tokenizer=tokenizer;self.max_length=max_length
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        _, token = load_feature(self.features,self.split,r['uid'])
        target=self.tokenizer.eos_token + training_target(r,'structured') + self.tokenizer.eos_token
        encoded=self.tokenizer(target,padding='max_length',max_length=self.max_length,
                               truncation=True,return_tensors='pt')
        ids=encoded['input_ids'][0]
        mask=encoded['attention_mask'][0]
        labels=ids.clone();labels[mask==0]=-100
        return torch.from_numpy(token),ids,mask,labels


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--output',default='models_v8')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--max-length',type=int,default=176)
    p.add_argument('--train-last-blocks',type=int,default=2)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if Path(a.output).resolve() == Path('models').resolve():
        raise ValueError('Do not overwrite existing baseline models folder')
    if a.epochs < 1 or a.batch_size < 1 or a.accumulation < 1 or a.max_length < 128:
        raise ValueError('Invalid training hyperparameters')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Training device:',device,'; RAD-DINO features precomputed; GPT-2 cross-attention fine-tuning.')
    tok=get_tokenizer()
    tr=FeatureReportDataset(a.data_dir,a.features,'train',tok,a.max_length)
    va=FeatureReportDataset(a.data_dir,a.features,'val',tok,a.max_length)
    train=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0)
    valid=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=0)
    m=build_v8_model(pretrained=True,train_last_blocks=a.train_last_blocks).to(device)
    trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'Vision encoder frozen; {len(tr)} train / {len(va)} validation studies; trainable params: {trainable:,}')
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=(device=='cuda'))
    best=float('inf');stale=0
    for ep in range(a.epochs):
        m.train();total=0;opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train):
            patches,ids,mask,labels=[x.to(device) for x in batch]
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):
                loss=m(patches,ids,mask,labels).loss
            scaler.scale(loss/a.accumulation).backward()
            if (i+1)%a.accumulation==0 or (i+1)==len(train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            total+=float(loss.detach().cpu())
            if (i+1)%200==0:
                print(f'Epoch {ep+1}: batch {i+1}/{len(train)}',flush=True)
        m.eval(); val_total=0
        with torch.no_grad():
            for batch in valid:
                patches,ids,mask,labels=[x.to(device) for x in batch]
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):
                    val_total+=float(m(patches,ids,mask,labels).loss.detach().cpu())
        vl=val_total/len(valid); tl=total/len(train)
        print(f'Epoch {ep+1}/{a.epochs} train={tl:.4f} val={vl:.4f}',flush=True)
        if vl < best-1e-4:
            best=vl;stale=0
            meta={'architecture':'rad-dino-gpt2-v8','version':'rad-dino-gpt2-v8-research',
                  'encoder':'microsoft/rad-dino','decoder':'gpt2-cross-attention',
                  'grid':7,'target_mode':'structured','training_data':'IU X-Ray',
                  'uses_external_pretraining':True,
                  'frozen_visual_encoder':True,'train_last_blocks':a.train_last_blocks,
                  'epochs_completed':ep+1,'best_val_loss':best,'seed':a.seed,
                  'train_split_sha256':source_manifest(a.data_dir)['train_sha256'],
                  'val_split_sha256':source_manifest(a.data_dir)['val_sha256'],
                  'validation_only_checkpoint_choice':True,
                  'clinical_validation':False,'review_required':True}
            save_v8(m,a.output,meta)
            print('Saved best NEW v8 checkpoint.',flush=True)
        else:
            stale+=1
            if a.patience and stale>=a.patience:
                print('Early stop: no validation-loss improvement.');break

if __name__=='__main__':main()
