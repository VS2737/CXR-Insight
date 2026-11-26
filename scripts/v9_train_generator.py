"""Continue v8 on SAME frozen IU train split with sparse finding-level loss.

Windows: python -m scripts.v9_train_generator --epochs 3 --batch-size 1 --accumulation 8
Keeps models_v8/ untouched. No medical-accuracy claims without external validation.
"""
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scripts.v8_train_generator import FeatureReportDataset
from radreport.v8_cache import source_manifest,verify_manifest_file
from radreport.v9_joint import from_v8,save_v9,ARCH
from radreport.v9_concepts import CONCEPTS,make_targets


class ConceptsDataset(Dataset):
    def __init__(self, base, label_file, split):
        self.base=base
        with open(label_file,newline='',encoding='utf-8-sig') as f:
            label_map={(str(r.get('split','')),str(r.get('uid',''))):r for r in csv.DictReader(f)}
        self.target=[]; self.known=[]
        for row in base.rows:
            y,mask=make_targets(row,label_map.get((split,str(row['uid']))))
            self.target.append(y);self.known.append(mask)
        self.target=np.asarray(self.target,dtype=np.float32)
        self.known=np.asarray(self.known,dtype=np.float32)
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        return (*self.base[i],torch.tensor(self.target[i]),torch.tensor(self.known[i]))


def positive_weights(target,mask,cap=4.0,min_positive=5,min_negative=5):
    """Train-only weights; under-supported classes abstain from auxiliary fit."""
    count_pos=(target*mask).sum(axis=0);known=mask.sum(axis=0);count_neg=known-count_pos
    supported=(count_pos>=min_positive)&(count_neg>=min_negative)
    weights=np.where(count_pos>0,np.minimum(cap,count_neg/np.maximum(count_pos,1)),1.)
    return weights.astype(np.float32),supported.astype(np.float32),\
        [{'name':k,'positive':int(p),'negative':int(n),'supported':bool(s)} for k,p,n,s in zip(CONCEPTS,count_pos,count_neg,supported)]


def masked_loss(logits,y,known,pos_weights):
    mask=known.float()
    errors=F.binary_cross_entropy_with_logits(logits.float(),y.float(),pos_weight=pos_weights.float(),reduction='none')
    return (errors*mask).sum()/mask.sum().clamp_min(1.)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--labels',default='data/processed/labels.csv')
    p.add_argument('--init-v8',default='models_v8')
    p.add_argument('--output',default='models_v9')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--max-length',type=int,default=192)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--aux-weight',type=float,default=.25)
    p.add_argument('--patience',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if not (0 < args.lr <= 1e-3 and 0 <= args.aux_weight <= 1 and args.epochs>0 and args.batch_size>0 and args.accumulation>0 and args.max_length>=128):
        p.error('Unsafe or invalid hyperparameters')
    forbidden={Path('models').resolve(),Path(args.init_v8).resolve()}
    if Path(args.output).resolve() in forbidden:
        p.error('Output must be a new folder, not the v8 model or old baseline')
    manifest=verify_manifest_file(args.data_dir,'train')
    verify_manifest_file(args.data_dir,'val')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu': print('WARNING: CPU full-model training will be slow.')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    model,v8_meta,tok=from_v8(args.init_v8,device)
    if v8_meta.get('train_split_sha256')!=manifest['train_sha256'] or v8_meta.get('val_split_sha256')!=manifest['val_sha256']:
        raise ValueError('Original v8 checkpoint is not from current frozen IU split')
    train=ConceptsDataset(FeatureReportDataset(args.data_dir,args.features,'train',tok,args.max_length),args.labels,'train')
    valid=ConceptsDataset(FeatureReportDataset(args.data_dir,args.features,'val',tok,args.max_length),args.labels,'val')
    weights,supported,support=positive_weights(train.target,train.known)
    print('Concept support from TRAIN only:',json.dumps(support,indent=2))
    weights=torch.tensor(weights,device=device)
    supported=torch.tensor(supported,device=device)
    loaders={'train':DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0),
             'val':DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0)}
    optim=torch.optim.AdamW((q for q in model.parameters() if q.requires_grad),lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    best=float('inf');stale=0
    for ep in range(args.epochs):
        recorded={}
        for split,loader in loaders.items():
            model.train(split=='train');total=total_text=total_aux=0.; optim.zero_grad(set_to_none=True)
            for idx,batch in enumerate(loader):
                patches,ids,attn,labels,y,known=[item.to(device) for item in batch]
                known=known*supported
                with torch.set_grad_enabled(split=='train'):
                    with torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
                        text_loss,logits=model(patches,ids,attn,labels)
                        aux=masked_loss(logits,y,known,weights)
                        loss=text_loss+args.aux_weight*aux
                    if split=='train':
                        scaler.scale(loss/args.accumulation).backward()
                        if (idx+1)%args.accumulation==0 or idx+1==len(loader):
                            scaler.unscale_(optim)
                            torch.nn.utils.clip_grad_norm_((q for q in model.parameters() if q.requires_grad),1.)
                            scaler.step(optim);scaler.update();optim.zero_grad(set_to_none=True)
                total+=float(loss.detach());total_text+=float(text_loss.detach());total_aux+=float(aux.detach())
                if split=='train' and (idx+1)%250==0:
                    print(f'Epoch {ep+1} train step {idx+1}/{len(loader)}',flush=True)
            recorded[split]={'combined':total/len(loader),'text':total_text/len(loader),'aux':total_aux/len(loader)}
        print(f'Epoch {ep+1}/{args.epochs}: {json.dumps(recorded)}',flush=True)
        if recorded['val']['combined']<best-1e-4:
            best=recorded['val']['combined'];stale=0
            meta={'architecture':ARCH,'version':'rad-dino-gpt2-v9-research',
                  'grid':7,'target_mode':'structured','encoder':'microsoft/rad-dino',
                  'init_checkpoint':str(args.init_v8),'train_last_blocks':v8_meta.get('train_last_blocks',2),
                  'concepts':list(CONCEPTS),'concept_support':support,'aux_weight':args.aux_weight,
                  'epoch':ep+1,'best_val_combined_loss':best,'best_val_text_loss':recorded['val']['text'],
                  'seed':args.seed,'train_split_sha256':manifest['train_sha256'],
                  'val_split_sha256':manifest['val_sha256'],
                  'weak_labels_sha256':hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
                  'clinical_validation':False,'review_required':True,
                  'warning':'Report-derived weak finding labels; auxiliary scores are NOT clinical probabilities.'}
            save_v9(model,args.output,meta)
            print('Saved best v9 experimental checkpoint.',flush=True)
        else:
            stale+=1
            if args.patience and stale>=args.patience:
                print('Early stopping on validation combined loss.');break

if __name__=='__main__':main()
