"""CXR Insight v10: cleaned-target, weak finding-aware v9 continuation.

Runs on original frozen IU train and val splits only. Keeps models_v9 intact.
No test-set tuning, no invented clinical ground truth, no external dataset.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from scripts.v8_train_generator import FeatureReportDataset
from scripts.v9_train_generator import ConceptsDataset, positive_weights, masked_loss
from radreport.v8_cache import verify_manifest_file, load_feature
from radreport.v9_joint import load_v9, save_v9, ARCH
from radreport.v10_text import clean_pair, encode_clean_target, VERSION


class CleanFeatureDataset(FeatureReportDataset):
    """Keeps immutable source data; cleans only decoded *training targets*."""
    def __init__(self, root, features, split, tokenizer, max_length=192):
        super().__init__(root,features,split,tokenizer,max_length)
        filtered=[]; cleaned=[]; reports=[]
        for row in self.rows:
            target,audit=clean_pair(row)
            reports.append(audit)
            if target is not None:
                filtered.append(row)
                cleaned.append(target)
        self.rows=filtered
        self.targets=cleaned
        self.cleaning_stats={
            'source_rows':len(reports), 'usable_rows':len(filtered),
            'dropped_rows':len(reports)-len(filtered),
            'rows_changed':sum(x['changed'] for x in reports),
            'removed_clauses':sum(x['findings_removed']+x['impression_removed'] for x in reports),
            'redacted_clauses':sum(x['redacted_clauses'] for x in reports),
            'cleaner':VERSION,
        }
        if not self.rows:raise ValueError(f'No usable reports after cleaning {split}')

    def __getitem__(self, index):
        r=self.rows[index]
        _,tokens=load_feature(self.features,self.split,r['uid'])
        ids,mask,labels=encode_clean_target(self.tokenizer,self.targets[index],self.max_length)
        return torch.from_numpy(tokens),ids,mask,labels


def study_sample_weights(target,mask,supported,abnormal_factor=1.5):
    """Train-only weak-positive sampling; cap oversampling and never infer negatives."""
    if not 1 <= abnormal_factor <= 2:
        raise ValueError('abnormal_factor must be between 1 and 2')
    known_positives=(target>0.5)&(mask>0.5)&(supported[None,:]>0.5)
    return np.where(known_positives.any(axis=1),abnormal_factor,1.0).astype('float64')


def validate_config(a):
    if a.epochs < 1 or a.batch_size < 1 or a.accumulation < 1 or a.max_length < 128:
        raise ValueError('Invalid epochs/batch/accumulation/max_length')
    if not 0 < a.lr <= 1e-3 or not 0<=a.aux_weight<=1:
        raise ValueError('Invalid loss weighting or learning rate')
    if not 1<=a.abnormal_factor<=2:
        raise ValueError('abnormal factor must be in [1,2]')
    prohibited={Path('models').resolve(),Path('models_v8').resolve(),Path(a.init_v9).resolve()}
    if Path(a.output).resolve() in prohibited:
        raise ValueError('Output cannot overwrite original models/v8/v9')
    if (Path(a.output)/'generator.pt').exists() and not a.overwrite:
        raise FileExistsError('Output checkpoint already exists; choose a new --output directory or explicitly use --overwrite')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--labels',default='data/processed/labels.csv')
    p.add_argument('--init-v9',default='models_v9')
    p.add_argument('--output',default='models_v10')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--max-length',type=int,default=192)
    p.add_argument('--lr',type=float,default=8e-6)
    p.add_argument('--aux-weight',type=float,default=.35)
    p.add_argument('--abnormal-factor',type=float,default=1.5)
    p.add_argument('--patience',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    validate_config(args)
    tr_manifest=verify_manifest_file(args.data_dir,'train')
    verify_manifest_file(args.data_dir,'val')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if device=='cuda':torch.cuda.manual_seed_all(args.seed)
    model,v9meta,tok=load_v9(args.init_v9,device)
    if (v9meta.get('train_split_sha256')!=tr_manifest['train_sha256'] or
            v9meta.get('val_split_sha256')!=tr_manifest['val_sha256']):
        raise ValueError('v9 checkpoint and current IU frozen train/val splits differ')
    base_tr=CleanFeatureDataset(args.data_dir,args.features,'train',tok,args.max_length)
    base_val=CleanFeatureDataset(args.data_dir,args.features,'val',tok,args.max_length)
    train=ConceptsDataset(base_tr,args.labels,'train')
    valid=ConceptsDataset(base_val,args.labels,'val')
    weights,supported,support=positive_weights(train.target,train.known)
    print('Training:',len(train),'Validation:',len(valid),'Device:',device,flush=True)
    print('Cleaning (train/val only):',json.dumps({'train':base_tr.cleaning_stats,'val':base_val.cleaning_stats}),flush=True)
    print('Train-only weak-label concept support:',json.dumps(support),flush=True)
    sam_w=study_sample_weights(train.target,train.known,supported,args.abnormal_factor)
    sampler=WeightedRandomSampler(torch.from_numpy(sam_w),len(sam_w),replacement=True,
                                  generator=torch.Generator().manual_seed(args.seed))
    train_loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=0)
    val_loader=DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0)
    class_w=torch.as_tensor(weights,device=device)
    class_supported=torch.as_tensor(supported,device=device)
    opt=torch.optim.AdamW((t for t in model.parameters() if t.requires_grad),lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    best=float('inf');stale=0
    for epoch in range(args.epochs):
        history={}
        for phase,loader in [('train',train_loader),('val',val_loader)]:
            model.train(phase=='train');tot=txt=auxsum=0.
            if phase=='train':opt.zero_grad(set_to_none=True)
            for idx,batch in enumerate(loader):
                patches,ids,mask,labels,y,known=(t.to(device) for t in batch)
                known=known*class_supported
                with torch.set_grad_enabled(phase=='train'):
                    with torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
                        text_loss,logits=model(patches,ids,mask,labels)
                        finding_loss=masked_loss(logits,y,known,class_w)
                        loss=text_loss+args.aux_weight*finding_loss
                    if phase=='train':
                        scaler.scale(loss/args.accumulation).backward()
                        if (idx+1)%args.accumulation==0 or idx+1==len(loader):
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_((t for t in model.parameters() if t.requires_grad),1.)
                            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                tot+=float(loss.detach());txt+=float(text_loss.detach());auxsum+=float(finding_loss.detach())
                if phase=='train' and (idx+1)%250==0:
                    print(f'Epoch {epoch+1}: {idx+1}/{len(loader)} steps',flush=True)
            history[phase]={'total':tot/len(loader),'text':txt/len(loader),'aux':auxsum/len(loader)}
        print(f'Epoch {epoch+1}/{args.epochs}: {json.dumps(history)}',flush=True)
        if history['val']['total']<best-1e-4:
            best=history['val']['total'];stale=0
            meta={**v9meta,
                  'architecture':ARCH,'version':'rad-dino-gpt2-v10-research',
                  'init_checkpoint':str(args.init_v9),'training_revision':'v10-cleaned-targets-aux-labels',
                  'target_cleaner':VERSION,'cleaning_stats':{'train':base_tr.cleaning_stats,'val':base_val.cleaning_stats},
                  'weak_label_support':support,'weak_labels_sha256':hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
                  'abnormal_factor':args.abnormal_factor,'aux_weight':args.aux_weight,'lr':args.lr,
                  'best_epoch':epoch+1,'best_val_loss_cleaned':best,
                  'train_split_sha256':tr_manifest['train_sha256'],'val_split_sha256':tr_manifest['val_sha256'],
                  'clinical_validation':False,'review_required':True,
                  'validation_only_checkpoint_choice':True,
                  'warning':'IU report-derived weak labels are not independently adjudicated; requires human review.'}
            save_v9(model,args.output,meta)
            print('Saved v10 candidate (v9 unchanged).',flush=True)
        else:
            stale+=1
            if args.patience and stale>=args.patience:
                print('Early stopping on validation cleaned-target objective.');break
    print('No test split used in training or checkpoint selection.')

if __name__=='__main__':main()
