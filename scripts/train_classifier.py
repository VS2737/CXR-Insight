"""Train DenseNet-121 on weak report-derived CheXpert labels; uncertainties masked."""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from PIL import Image
from radreport.classifier import ImageFindingClassifier
from radreport.model import TRANSFORM
from radreport.config import FINDINGS
from radreport.data import load_jsonl
from radreport.label_values import parse_binary_chexpert


class LabelledImages(Dataset):
    def __init__(self, split, root, labels):
        self.items = []
        with open(labels, newline='', encoding='utf-8') as f:
            found = {(r['split'], r['uid']): r for r in csv.DictReader(f)}
        for row in load_jsonl(Path(root)/f'{split}.jsonl'):
            match = found.get((split, row['uid']))
            if match: self.items.append((row['image_path'], match))
        if not self.items: raise ValueError(f'No joined labelled {split} items; run official CheXpert labeller/import first.')
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, row = self.items[i]
        with Image.open(path) as im: x=TRANSFORM(im.convert('RGB'))
        target, mask = [], []
        for k in FINDINGS:
            value = parse_binary_chexpert(row.get(k, ''))
            target.append(float(value) if value is not None else 0.0)
            mask.append(1.0 if value is not None else 0.0) # exclude -1 uncertain and unmentioned
        return x, torch.tensor(target), torch.tensor(mask)


from scripts.class_balance import training_pos_weights

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', default='data/processed')
    p.add_argument('--labels', default='data/processed/labels.csv')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--output', default='models/classifier.pt')
    p.add_argument('--balanced-loss', action='store_true', help='Train-only masked positive weighting (experimental).')
    p.add_argument('--pos-weight-cap', type=float, default=5.0)
    p.add_argument('--freeze-backbone', action='store_true', help='Train final DenseNet layer only, a faster baseline.')
    p.add_argument('--patience', type=int, default=0, help='Early-stop after this many non-improving validation epochs; 0 disables.')
    a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train=DataLoader(LabelledImages('train', a.root, a.labels),batch_size=a.batch_size,shuffle=True)
    val=DataLoader(LabelledImages('val', a.root, a.labels),batch_size=a.batch_size)
    model=ImageFindingClassifier(pretrained=True).to(device)
    print(f'Using {len(train.dataset)} labelled train images and {len(val.dataset)} validation images')
    if a.freeze_backbone:
        for weight in model.network.features.parameters(): weight.requires_grad=False
    if a.pos_weight_cap < 1 or a.pos_weight_cap > 20: raise ValueError('pos-weight-cap must be 1..20')
    pos_weight=(training_pos_weights(train.dataset.items,a.pos_weight_cap).to(device)
                if a.balanced_loss else None)
    if a.balanced_loss: print('Train-only class weights:',pos_weight.cpu().tolist())
    optim=torch.optim.AdamW([w for w in model.parameters() if w.requires_grad],lr=a.lr)
    best=float('inf'); stale=0
    for ep in range(a.epochs):
        model.train(); losses=[]
        for x,y,mask in train:
            x,y,mask=x.to(device),y.to(device),mask.to(device)
            optim.zero_grad()
            logits=model(x)
            per_label=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=pos_weight)
            if mask.sum().item() == 0:
                raise ValueError('Batch has zero definitive labels. Check the CheXpert CSV values (0.0, 1.0, -1.0).')
            loss=(per_label*mask).sum()/mask.sum()
            loss.backward();optim.step();losses.append(loss.item())
        model.eval();v=[]
        with torch.no_grad():
            for x,y,mask in val:
                x,y,mask=x.to(device),y.to(device),mask.to(device)
                err=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y,reduction='none')
                if mask.sum().item():
                    v.append(float((err*mask).sum()/mask.sum()))
        if not v: raise ValueError('Validation split has zero definitive labels.')
        cur=sum(v)/len(v);print(f'Epoch {ep+1}: train {sum(losses)/len(losses):.4f}, val {cur:.4f}')
        if cur<best:
            best=cur;stale=0
            out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'state_dict':model.state_dict(), 'metadata':{'version':'densenet121-chexpert-weak-labels-v1',
                         'clinical_validation':False,'calibrated':False,
                         'warning':'Uncalibrated image probabilities; provisional research use only',
                         'balanced_loss':a.balanced_loss,'pos_weight_cap':a.pos_weight_cap if a.balanced_loss else None,
                         'freeze_backbone':a.freeze_backbone}},out)
            print('Saved classifier checkpoint')
        else:
            stale+=1
            if a.patience and stale>=a.patience:
                print(f'Early stop: validation did not improve for {stale} epochs');break
    print('IMPORTANT: classifier outputs are uncalibrated; conduct per-class calibration before treating checks as validated.')

if __name__=='__main__':main()
