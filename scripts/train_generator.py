"""Train EfficientNet->GPT2 report generator. Requires GPU and installed ML requirements."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from radreport.model import ReportGenerator, get_tokenizer, TRANSFORM, save_generator
from radreport.data import load_jsonl
from radreport.config import FINDINGS
from radreport.formal_report import training_target
from radreport.label_values import parse_binary_chexpert


class IUDataset(Dataset):
    def __init__(self, jsonl, tokenizer, max_length=96, target_mode='impression'):
        self.rows = load_jsonl(jsonl)
        self.target_mode = target_mode
        if target_mode == 'structured':
            self.rows = [r for r in self.rows if str(r.get('findings') or '').strip().lower() not in ('', 'nan', 'none')]
            if not self.rows:
                raise ValueError('Structured mode needs IU reports containing actual Findings and Impression text.')
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r['image_path']) as im: x = TRANSFORM(im.convert('RGB'))
        # EOS acts as a start marker, consistent with sample_reports() inference.
        text = self.tokenizer.eos_token + training_target(r, self.target_mode) + self.tokenizer.eos_token
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                             return_tensors='pt')
        ids, mask = enc['input_ids'][0], enc['attention_mask'][0]
        label = ids.clone(); label[mask == 0] = -100
        return x, ids, mask, label


def make_balanced_sampler(dataset, labels_path, abnormal_weight, seed):
    """Oversample weak-label positive *training* studies only, never test studies.

    Experimental anti-normal-collapse sampling; may increase false positives.
    """
    from collections import Counter
    with open(labels_path, newline='', encoding='utf-8') as f:
        labels = list(csv.DictReader(f))
    by_uid = {str(r['uid']):r for r in labels if r.get('split')=='train'}
    weights = []
    positives = 0
    missing = 0
    for row in dataset.rows:
        info=by_uid.get(str(row['uid']))
        if not info:
            weights.append(1.0)
            missing += 1
            continue
        abnormal=any(parse_binary_chexpert(info.get(name, '')) == 1
                     for name in FINDINGS if name not in ('No Finding','Support Devices'))
        positives += int(abnormal)
        weights.append(abnormal_weight if abnormal else 1.0)
    if positives == 0:
        raise ValueError('No positive training weak-label examples found; refusing to claim balanced sampling.')
    print(f'Experimental train-only sampling: {positives} positive studies, {missing} missing labels, '
          f'abnormal weight={abnormal_weight}; no test labels used.')
    generator=torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights,num_samples=len(dataset),replacement=True,generator=generator)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--output', default='models')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max-length', type=int, default=96)
    p.add_argument('--target-mode', choices=['impression','structured'], default='impression',
                   help='Structured requires true IU Findings and Impression report text; train into a NEW directory.')
    p.add_argument('--unfreeze-encoder', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--track-mlflow', action='store_true')
    p.add_argument('--resume-from', default=None, help='Previously trained generator.pt checkpoint; optimizer starts fresh.')
    p.add_argument('--balanced-sampling', action='store_true', help='Experimental train-only weak-label oversampling.')
    p.add_argument('--weak-labels', default='data/processed/labels.csv')
    p.add_argument('--abnormal-weight', type=float, default=2.0)
    args = p.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('DEVICE:', device, '(GPU recommended)')
    if args.target_mode == 'structured' and Path(args.output).resolve() == Path('models').resolve():
        raise ValueError('Structured training requires a NEW --output directory, e.g. models_structured; do not overwrite original weights.')
    if args.target_mode == 'structured' and args.max_length < 128:
        raise ValueError('Use --max-length 160 or 192 for Findings + Impression targets.')
    if args.abnormal_weight < 1 or args.abnormal_weight > 4:
        raise ValueError('--abnormal-weight must be between 1.0 and 4.0')
    if args.resume_from and Path(args.output).resolve() == Path(args.resume_from).resolve().parent:
        raise ValueError('Use a NEW --output folder when continuing from a checkpoint, preserving your original model.')
    tokenizer = get_tokenizer()
    train_set = IUDataset(f'{args.data_dir}/train.jsonl', tokenizer, args.max_length, args.target_mode)
    sampler = (make_balanced_sampler(train_set, args.weak_labels, args.abnormal_weight, args.seed)
               if args.balanced_sampling else None)
    tr = DataLoader(train_set, batch_size=args.batch_size,
                    shuffle=(sampler is None), sampler=sampler, num_workers=0)
    va = DataLoader(IUDataset(f'{args.data_dir}/val.jsonl', tokenizer, args.max_length, args.target_mode),
                    batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.resume_from:
        # Load only a checkpoint you generated or trust. Checkpoints can contain pickle.
        checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        model = ReportGenerator(pretrained=False, freeze_encoder=not args.unfreeze_encoder)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print('Continuing weights from', args.resume_from, '(optimizer is reset).')
    else:
        model = ReportGenerator(pretrained=True, freeze_encoder=not args.unfreeze_encoder)
    model = model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    if args.track_mlflow:
        import mlflow
        mlflow.set_experiment('RadiologyReportGeneration')
        mlflow.start_run()
        mlflow.log_params(vars(args))
    try:
        best = float('inf')
        for epoch in range(args.epochs):
            model.train(); opt.zero_grad(set_to_none=True)
            total = 0
            for step, batch in enumerate(tr):
                x, ids, mask, labels = [v.to(device) for v in batch]
                with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                    loss = model(x, ids, mask, labels).loss
                    scaled = loss / args.accumulation
                scaler.scale(scaled).backward()
                if (step + 1) % args.accumulation == 0 or step + 1 == len(tr):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                total += float(loss.item())
            model.eval(); val_sum = 0
            with torch.no_grad():
                for batch in va:
                    x, ids, mask, labels = [v.to(device) for v in batch]
                    with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                        val_sum += model(x, ids, mask, labels).loss.item()
            train_loss, val_loss = total / max(1, len(tr)), val_sum / max(1, len(va))
            print(f'Epoch {epoch+1}/{args.epochs}: train={train_loss:.4f} validation={val_loss:.4f}')
            if args.track_mlflow:
                mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch + 1)
            if val_loss < best:
                best = val_loss
                manifest = json.loads((Path(args.data_dir) / 'manifest.json').read_text())
                metadata = {'version':'efficientnet-b0-gpt2-structured-v5-experimental' if args.target_mode=='structured' else 'efficientnet-b0-gpt2-v1', 'seed':args.seed, 'training_manifest':manifest,
                            'validation_loss':best, 'epochs_completed':epoch+1,
                            'encoder':'EfficientNet-B0', 'decoder':'GPT-2 with randomly initialized cross attention',
                            'review_required':True, 'clinical_validation':False,
                            'resume_from':str(args.resume_from) if args.resume_from else None,
                            'target_mode':args.target_mode, 'max_target_tokens':args.max_length,
                            'optimizer_resumed':False,
                            'balanced_sampling':args.balanced_sampling,
                            'abnormal_weight':args.abnormal_weight if args.balanced_sampling else None}
                save_generator(model, args.output, metadata)
                print('Saved best checkpoint.')
    finally:
        if args.track_mlflow: mlflow.end_run()

if __name__ == '__main__': main()
