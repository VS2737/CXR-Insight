"""Experimental joint report-generation + sparse finding-supervision head.

Uses a completed RAD-DINO v8 checkpoint as initialization. The vision encoder
is frozen/cached. No image label here is independent clinical ground truth.
"""
from pathlib import Path
import json
import torch
from torch import nn
from radreport.v8_generator import build_v8_model, get_tokenizer
from radreport.v9_concepts import CONCEPTS

ARCH = 'rad-dino-gpt2-v9-concepts'


class JointGenerator(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.finding_head = nn.Linear(768,len(CONCEPTS))

    def forward(self, patches, ids, mask, labels):
        memory = self.base.bridge(patches.float())
        output = self.base.decoder(input_ids=ids,attention_mask=mask,
                                   encoder_hidden_states=memory,labels=labels,use_cache=False)
        logits = self.finding_head(memory.mean(dim=1))
        return output.loss,logits

    def sample_reports(self, patches, tokenizer, **kwargs):
        return self.base.sample_reports(patches,tokenizer,**kwargs)

    @torch.no_grad()
    def finding_research_scores(self,patches):
        self.eval()
        vector=self.base.bridge(patches.float()).mean(dim=1)
        return {name:float(p) for name,p in zip(CONCEPTS,
                self.finding_head(vector).sigmoid()[0].cpu().tolist())}


def from_v8(folder='models_v8',device='cpu'):
    path=Path(folder)/'generator.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    meta=saved['metadata']
    if meta.get('architecture')!='rad-dino-gpt2-v8' or meta.get('grid')!=7:
        raise ValueError('Expected verified v8 RAD-DINO checkpoint')
    base=build_v8_model(pretrained=False,train_last_blocks=meta.get('train_last_blocks',2))
    base.load_state_dict(saved['state_dict'],strict=True)
    return JointGenerator(base).to(device),meta,get_tokenizer()


def save_v9(model,folder,metadata):
    path=Path(folder);path.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'metadata':metadata},path/'generator.pt')
    (path/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


def load_v9(folder='models_v9',device='cpu'):
    saved=torch.load(Path(folder)/'generator.pt',map_location='cpu',weights_only=True)
    meta=saved['metadata']
    if meta.get('architecture')!=ARCH or meta.get('grid')!=7 or meta.get('concepts')!=list(CONCEPTS):
        raise ValueError('Unexpected v9 checkpoint or concept vocabulary')
    base=build_v8_model(pretrained=False,train_last_blocks=meta.get('train_last_blocks',2))
    model=JointGenerator(base)
    model.load_state_dict(saved['state_dict'],strict=True)
    return model.to(device).eval(),meta,get_tokenizer()
