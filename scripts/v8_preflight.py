"""Read-only Windows setup and evidence checklist for experimental v8."""
from pathlib import Path
import json,sys


def main():
    print('CXR Insight v8 research preflight — NOT FOR CLINICAL USE')
    print('Python:',sys.version.split()[0])
    try:
        import torch
        print('PyTorch:',torch.__version__, '| CUDA:',torch.cuda.is_available())
        if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
    except ImportError:print('PyTorch missing')
    try:
        import transformers
        print('Transformers:',transformers.__version__)
    except ImportError:print('Transformers missing; install requirements-v8.txt')
    try:
        from radreport.v8_cache import verify_manifest_file,verify_split_cache
        for split in ('train','val','test'):
            try:
                manifest=verify_manifest_file('data/processed',split)
                print('IU',split,'split: VERIFIED, studies:',manifest.get(split+'_count'))
            except (OSError,ValueError,KeyError,json.JSONDecodeError) as e:
                print('IU',split,'split: NOT READY:',e)
                continue
            try:
                verify_split_cache('data/v8_rad_features',split,manifest)
                print('RAD-DINO',split,'features: INDEX READY (individual vector files checked during training)')
            except (OSError,ValueError) as e:
                print('RAD-DINO',split,'features: NOT READY:',e)
    except Exception as e:
        print('Data/split status unavailable:',type(e).__name__,e)
    for path in ('models/generator.pt','models/classifier.pt',
                 'models_v8/generator.pt','models_v8/finding_probe.json',
                 'models_v8/retrieval/train_reports.json'):
        print(path,':','EXISTS' if Path(path).is_file() else 'NOT PRESENT')
    print('No results above claim medical accuracy. Preserve original models/ as a fallback.')

if __name__=='__main__':main()
