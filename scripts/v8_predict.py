"""Generate a structured research draft from a frontal CXR with the v8 encoder."""
import argparse,json
from PIL import Image
import torch
from radreport.v8_vision import RadDinoEncoder
from radreport.v8_generator import load_v8
from radreport.formal_report import parse_generated
from radreport.safety import route


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--model-dir',default='models_v8')
    p.add_argument('--projection',choices=['frontal','unknown','lateral'],default='unknown')
    a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,meta,tok=load_v8(a.model_dir,device)
    encoder=RadDinoEncoder(device)
    with Image.open(a.image) as im:
        _,patches=encoder.encode(im.convert('RGB'))
    torch.manual_seed(42)
    samples=model.sample_reports(torch.from_numpy(patches).unsqueeze(0).to(device),tok,n=5)
    parsed=parse_generated(samples[0],structured_expected=True)
    safety=route(samples[0],samples,None,None,True)
    print(json.dumps({'source':'RAD-DINO GPT-2 v8 RESEARCH CHECKPOINT',
                      'findings':parsed['findings'],'impression':parsed['impression'],
                      'complete':parsed['complete'],'self_consistency':safety['self_consistency'],
                      'review_required':True,'review_reasons':safety['review_reasons'],
                      'disclaimer':'Not for clinical use. All outputs require qualified review.'},indent=2))
if __name__=='__main__':main()
