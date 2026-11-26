"""Lazy, memory-only image inference. No image or generated text persisted."""
import io
import math
import os
import uuid
from pathlib import Path
from functools import lru_cache
from PIL import Image, ImageOps, UnidentifiedImageError
from radreport.config import MODEL_DIR, ALLOWED_FORMATS, DISCLAIMER
from radreport.safety import route, extract_positive_entities
from radreport.review_db import enqueue
from radreport.formal_report import build_formal_report


class InvalidImage(ValueError): pass


def image_from_bytes(payload: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise InvalidImage('Upload PNG or JPEG only.')
            if im.width < 128 or im.height < 128:
                raise InvalidImage('Image must be at least 128 × 128 pixels.')
            if im.width * im.height > 25_000_000:
                raise InvalidImage('Image resolution exceeds safety limit.')
            im = ImageOps.exif_transpose(im)
            return im.convert('RGB').copy()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        if isinstance(e, InvalidImage): raise
        raise InvalidImage('Could not decode a supported image.') from e


def image_stats(img: Image.Image) -> dict:
    import numpy as np
    arr = np.asarray(img.resize((64, 64)).convert('L'), dtype='float32') / 255
    return {'brightness': round(float(arr.mean()), 5),
            'contrast': round(float(arr.std()), 5),
            'width': img.width, 'height': img.height}


@lru_cache(maxsize=1)
def load_assets():
    # Opt-in v8 research model; never silently swap weights or claim
    # improved medical accuracy. Use CXR_V8_MODE=1 only after training v8.
    use_v10 = os.getenv('CXR_V10_MODE', '0') == '1'
    use_v9 = use_v10 or os.getenv('CXR_V9_MODE', '0') == '1'
    if use_v9 or os.getenv('CXR_V8_MODE', '0') == '1':
        try:
            import torch
            from radreport.v8_vision import RadDinoEncoder
            from radreport.v8_generator import load_v8
            from radreport.v9_joint import load_v9
            from radreport.classifier import load_classifier
            from radreport.v8_probe import load_probe
            from radreport.v8_retrieval import StudyRetriever
            directory = (os.getenv('CXR_V10_MODEL_DIR','models_v10') if use_v10 else os.getenv('CXR_V9_MODEL_DIR', 'models_v9')) if use_v9 else os.getenv('CXR_V8_MODEL_DIR', 'models_v8')
            if not (Path(directory) / 'generator.pt').is_file():
                raise FileNotFoundError(f'{directory}/generator.pt missing; train requested model first or disable its mode.')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, meta, tokenizer = (load_v9(directory, device) if use_v9 else load_v8(directory, device))
            encoder = RadDinoEncoder(device)
            class_path = MODEL_DIR / 'classifier.pt'
            classifier, cmeta = (load_classifier(class_path, device) if class_path.exists() else (None, {}))
            probe_file = (Path('models_v8') if use_v9 else Path(directory)) / 'finding_probe.json'
            probe = load_probe(probe_file) if probe_file.is_file() else None
            if probe and (probe.get('train_split_sha256') != meta.get('train_split_sha256')
                          or probe.get('val_split_sha256') != meta.get('val_split_sha256')):
                raise ValueError('V8 finding probe and generator were trained on different IU splits')
            retrieval_dir = (Path('models_v8') if use_v9 else Path(directory)) / 'retrieval'
            retriever = (StudyRetriever(retrieval_dir, meta.get('train_split_sha256'))
                         if (retrieval_dir / 'train_reports.json').is_file() else None)
            return {'ready':True,'model_family':'rad-dino-gpt2-v9' if use_v9 else 'rad-dino-gpt2-v8',
                    'model':model,'encoder':encoder,'tokenizer':tokenizer,
                    'probe':probe,'retriever':retriever,
                    'classifier':classifier,'classifier_metadata':cmeta,
                    'version':meta.get('version','rad-dino-gpt2-v8'),
                    'metadata':meta,'device':device}
        except Exception as exc:
            return {'ready':False,'model':None,'tokenizer':None,'encoder':None,
                    'classifier':None,'classifier_metadata':{},'version':'v10-load-failed' if use_v10 else ('v9-load-failed' if use_v9 else 'v8-load-failed'),
                    'reason':f'RAD-DINO checkpoint could not load: {type(exc).__name__}: {exc}'}
    gen_path = MODEL_DIR / 'generator.pt'
    class_path = MODEL_DIR / 'classifier.pt'
    if not gen_path.exists():
        return {'ready':False, 'model':None, 'tokenizer':None,
                'classifier':None, 'classifier_metadata':{}, 'version':'untrained',
                'reason':'Train the generator checkpoint before image-to-report inference.'}
    try:
        import torch
        from radreport.model import load_generator
        from radreport.classifier import load_classifier
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, meta, tokenizer = load_generator(str(MODEL_DIR), device)
        classifier, cmeta = (load_classifier(class_path, device)
                              if class_path.exists() else (None, {}))
        return {'ready':True,'model':model, 'tokenizer':tokenizer,
                'classifier':classifier,'classifier_metadata':cmeta,
                'version':meta.get('version','unknown'), 'metadata':meta,
                'device':device}
    except Exception as e:
        # Do not silently claim a failed checkpoint produced inference.
        return {'ready':False, 'model':None, 'tokenizer':None,
                'classifier':None, 'classifier_metadata':{}, 'version':'load-failed',
                'reason':f'Checkpoint could not be loaded: {type(e).__name__}: {e}'}


def analyze(payload: bytes, projection: str = "unknown") -> dict:
    img = image_from_bytes(payload)
    stats = image_stats(img)
    assets = load_assets()
    draft = None
    variants = []
    image_scores = None
    grounding = None
    overlay = None
    selection = None
    cls_feature = None
    probe_evidence = None
    similar_studies = None
    research_finding_scores = None
    if assets['ready']:
        import torch
        from radreport.classifier import predict_image, ground_entities, make_cam_overlay
        # Legacy decoder and DenseNet use 224px inputs; RAD-DINO v8 uses its
        # own 518px processor, avoiding unnecessary imports / extra memory.
        tensor = None
        if assets.get('model_family') not in ('rad-dino-gpt2-v8','rad-dino-gpt2-v9'):
            from radreport.model import TRANSFORM
            tensor = TRANSFORM(img).unsqueeze(0).to(assets['device'])
        target_mode = assets.get('metadata', {}).get('target_mode', 'impression')
        max_tokens = (192 if assets.get('metadata',{}).get('version','').startswith('rad-dino-gpt2-v10')
                      else 168 if target_mode == 'structured' else 65)
        if assets.get('model_family') in ('rad-dino-gpt2-v8','rad-dino-gpt2-v9'):
            import numpy as np
            cls_feature, patch_features = assets['encoder'].encode(img)
            token_tensor = torch.from_numpy(patch_features).unsqueeze(0).to(assets['device'])
            variants = assets['model'].sample_reports(token_tensor, assets['tokenizer'], n=5, max_new_tokens=max_tokens)
            if assets.get('model_family') == 'rad-dino-gpt2-v9':
                research_finding_scores = assets['model'].finding_research_scores(token_tensor)
            if assets.get('probe') is not None:
                from radreport.v8_probe import predict_probe
                probe_evidence = predict_probe(assets['probe'], cls_feature)
            if assets.get('retriever') is not None:
                similar_studies = assets['retriever'].search(cls_feature, k=3)
        else:
            variants = assets['model'].sample_reports(tensor, assets['tokenizer'], n=5, max_new_tokens=max_tokens)
        from radreport.candidate_selection import select_draft
        selection = select_draft(variants, mode=os.getenv('CXR_DRAFT_SELECTION', 'first'))
        draft = variants[selection['index']] if selection['index'] is not None else None
        classifier = assets['classifier']
        # An uncalibrated classifier is NOT sufficient to pass an entity safety check.
        calibrated = assets['classifier_metadata'].get('calibrated') is True
        if classifier is not None:
            if tensor is None:
                from radreport.model import TRANSFORM
                tensor = TRANSFORM(img).unsqueeze(0).to(assets['device'])
            if calibrated:
                image_scores = predict_image(classifier, tensor, float(assets['classifier_metadata'].get('temperature', 1.0)))
                entities = extract_positive_entities(draft)
                grounding, cam = ground_entities(classifier, tensor, draft, entities, projection=projection)
                # Keep the report audit separate from an exploratory classifier preview.
                from radreport.config import FINDINGS
                from radreport.classifier import cam_for_entity
                import numpy as np
                from math import isfinite

                target = None
                if cam is not None:
                    target = next((d['entity'] for d in grounding.get('details', [])
                                   if 'region' in d or d.get('reason') == 'Degenerate zero Grad-CAM map.'), None)
                if cam is None and projection == 'frontal':
                    candidates = [name for name in entities if name in FINDINGS]
                    if not candidates:
                        candidates = [name for name in FINDINGS
                                      if isfinite(image_scores.get(name, float('nan')))]
                    if candidates:
                        target = max(candidates, key=image_scores.get)
                        cam = cam_for_entity(classifier, tensor, FINDINGS.index(target))
                if cam is not None and np.isfinite(cam).all() and float(np.ptp(cam)) > 1e-8:
                    overlay = make_cam_overlay(img, cam)
                    grounding['preview_label'] = 'Classifier Grad-CAM: ' + (target or 'class response')
                    if not entities:
                        grounding['preview_label'] += ' (exploratory; no positive report finding)'
                elif cam is not None:
                    grounding['reason'] += ' The classifier produced a flat map; no overlay is available.'
            else:
                grounding = {'status':'UNKNOWN', 'reason':'Image classifier has not been independently calibrated.'}
    safety = route(draft or '', variants, image_scores, grounding, assets['ready'])
    if assets.get('metadata',{}).get('version','').startswith('rad-dino-gpt2-v10'):
        from radreport.v10_text import artifact_flags
        quality_flags = artifact_flags(draft or '')
        safety['v10_text_quality_flags'] = quality_flags
        if quality_flags:
            safety['review_reasons'].append('Generated text contains incomplete sections or redaction artifacts; manual review required')
            safety['safety_check_passed'] = False
        safety['v10_quality_warning'] = 'Text-format checks are not proof of accurate image interpretation.'
    if research_finding_scores is not None:
        safety['v9_auxiliary_weak_label_scores'] = research_finding_scores
        safety['v9_auxiliary_warning'] = 'Uncalibrated report-derived weak-label scores, NOT clinical probabilities. Never interpret as diagnoses or proof of normal findings.'
    if probe_evidence is not None:
        # Weak probe signals can trigger extra human review, not clinical passes.
        positive = set(extract_positive_entities(draft or ''))
        possible_missing = []
        for name, entry in probe_evidence.items():
            if name in ('No Finding','Support Devices'):
                continue
            if entry.get('status') == 'ABOVE_VALIDATION_THRESHOLD' and name not in positive:
                possible_missing.append(name)
        safety['rad_dino_probe'] = {
            'findings':probe_evidence,
            'possibly_omitted_by_generator':possible_missing,
            'warning':'Weak report-derived label scores, NOT independently verified X-ray diagnoses.'}
        if possible_missing:
            safety['review_reasons'].append('RAD-DINO weak-label probe suggests possible omissions; expert review required')
            safety['safety_check_passed'] = False
    if assets['ready'] and not assets.get('classifier'):
        safety['review_reasons'].append('Independent image classifier checkpoint missing')
    if assets['ready'] and not assets['classifier_metadata'].get('calibrated'):
        safety['review_reasons'].append('Image classifier not calibrated')
    rid = uuid.uuid4().hex
    formatted = build_formal_report(raw=draft, projection=projection, report_id=rid[:12],
        model_version=assets['version'], trained=assets['ready'],
        structured_expected=assets.get('metadata',{}).get('target_mode') == 'structured',
        safety=safety, image_scores=image_scores)
    enqueue(rid, assets['version'], stats, safety)
    return {'id':rid, 'draft_impression':draft, 'decodings':variants,
            'draft_selection':selection,
            'image_stats':stats, 'projection':projection, 'model_loaded':assets['ready'],
            'model_version':assets['version'], 'model_status': assets.get('reason', 'Checkpoint loaded'),
            'safety':safety, 'gradcam_overlay':overlay, 'formal_report':formatted,
            'v8_retrieved_training_analogues':similar_studies,
            'v8_information':'Retrieved examples are from other training studies and are NOT diagnoses for the uploaded X-ray.' if similar_studies else None,
            'prototype':True, 'disclaimer':DISCLAIMER}
