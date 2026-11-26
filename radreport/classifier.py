"""Independent radiograph classifier and class-conditional Grad-CAM audit.

Labels should be generated using official CheXpert report labeler, NOT by
pretending that CheXpert NLP labels images. Weak report-derived labels require
independent validation before any claim about image-grounding accuracy.
"""
import re
import numpy as np
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
from .config import FINDINGS


class ImageFindingClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.network = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.network.classifier = nn.Linear(self.network.classifier.in_features, len(FINDINGS))

    def forward(self, x):
        return self.network(x)


def load_classifier(path, device):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    m = ImageFindingClassifier(pretrained=False)
    m.load_state_dict(blob['state_dict'])
    return m.to(device).eval(), blob.get('metadata', {})


def predict_image(classifier, img_tensor, temperature=1.0):
    with torch.no_grad():
        p = (classifier(img_tensor) / temperature).sigmoid().squeeze(0).cpu().tolist()
    return dict(zip(FINDINGS, [round(v, 4) for v in p]))


# Approximate research ROIs for frontal images ONLY. They are not segmentations.
# Viewer-left corresponds to the patient's right on a standard displayed frontal CXR.
LOCATIONS = {
    r'right (?:lower|basilar|base\b|lung base|lower (?:lung )?zone|lower lobe)': (0.04, 0.48, 0.48, 0.96),
    r'left (?:lower|basilar|base\b|lung base|lower (?:lung )?zone|lower lobe)': (0.52, 0.48, 0.96, 0.96),
    r'right (?:upper|apical|apex)': (0.04, 0.12, 0.48, 0.55),
    r'left (?:upper|apical|apex)': (0.52, 0.12, 0.96, 0.55),
    r'right (?:middle|mid)': (0.04, 0.32, 0.48, 0.76),
    r'left (?:middle|mid)': (0.52, 0.32, 0.96, 0.76),
}
HEART_REGION = (0.28, 0.35, 0.78, 0.94)
LOCALIZED_FINDINGS = {
    'Atelectasis', 'Consolidation', 'Lung Lesion', 'Lung Opacity',
    'Pleural Effusion', 'Pneumonia', 'Pneumothorax',
}


def _clause_for_finding(report, entity):
    """Identify a report clause containing this named finding, if possible."""
    from .safety import ALIASES
    from .safety import report_claims
    for clause in re.split(r'[.;\n]+', report):
        if (any(re.search(p, clause, re.I) for p in ALIASES.get(entity, []))
                and report_claims(clause)['findings'].get(entity) == 'PRESENT'):
            return clause
    return ''


def region_for_entity(report, entity):
    if entity == 'Cardiomegaly':
        return HEART_REGION
    if entity not in LOCALIZED_FINDINGS:
        return None  # diffuse findings require different anatomy models
    local = _clause_for_finding(report, entity)
    # Bilateral and multiple distinct regions are not represented faithfully by
    # a single one-sided rectangle. Abstain rather than claim anatomical proof.
    if re.search(r'\b(?:bilateral|bibasilar|both lungs|multifocal)\b', local, re.I):
        return None
    for pattern, area in LOCATIONS.items():
        if re.search(pattern, local, flags=re.I):
            return area
    return None


def region_for_report(report):
    """Backward-compatible helper used in the original diagnostic script."""
    from .safety import extract_positive_entities
    for entity in extract_positive_entities(report):
        region = region_for_entity(report, entity)
        if region is not None:
            return region
    return None


def cam_for_entity(classifier, input_tensor, entity_idx):
    """Compute a class-specific gradient-weighted activation heatmap.

    It is an explanation of this weak-label classifier, NOT medical localization.
    """
    classifier.zero_grad(set_to_none=True)
    with torch.enable_grad():
        feature_maps = classifier.network.features(input_tensor.detach().clone().requires_grad_(True))
        feature_maps.retain_grad()
        x = torch.nn.functional.relu(feature_maps, inplace=False)
        logits = classifier.network.classifier(torch.nn.functional.adaptive_avg_pool2d(x, (1,1)).flatten(1))
        logits[0, entity_idx].backward()
        weights = feature_maps.grad.detach().mean(dim=(2,3), keepdim=True)
        cam = torch.relu((weights * feature_maps.detach()).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)
        cam = cam[0,0].float().cpu().numpy()
    if not np.isfinite(cam).all() or (cam.max() - cam.min()) < 1e-8:
        return np.zeros((224,224), dtype='float32')
    return ((cam - cam.min()) / (cam.max() - cam.min())).astype('float32')


def ground_entities(classifier, input_tensor, report, entities,
                    min_region_fraction=0.28, projection='frontal'):
    """Per-finding Grad-CAM ROI audit with truthful abstention.

    projection='unknown'/'lateral' => no frontal regional judgement.
    Legacy calls without projection default to frontal for diagnostic compatibility.
    This heuristic requires bounding-box validation before scientific claims.
    """
    if not entities:
        return {'status':'UNKNOWN', 'reason':'No positive finding to localize.'}, None
    from .safety import report_contradictions
    conflicts = report_contradictions(report)
    # Still generate the heatmap for inspectability, but a saliency map must not
    # override an internal text contradiction. The final verdict stays FAIL.
    if projection != 'frontal':
        return {'status':'UNKNOWN', 'reason':'Frontal projection not confirmed; ROI comparison was skipped.'}, None
    details, first_cam = [], None
    for entity in entities:
        if entity not in FINDINGS:
            return {'status':'UNKNOWN', 'reason':'Claim is outside the classifier ontology.'}, None
        roi = region_for_entity(report, entity)
        if roi is None:
            details.append({'entity':entity, 'verdict':'UNKNOWN',
                            'reason':'Specific localized anatomical region not provided or not supported.'})
            continue
        cam = cam_for_entity(classifier, input_tensor, FINDINGS.index(entity))
        if first_cam is None:
            first_cam = cam
        total = float(cam.sum())
        if total <= 1e-8:
            details.append({'entity':entity,'verdict':'UNKNOWN', 'reason':'Degenerate zero Grad-CAM map.'})
            continue
        x0,y0,x1,y1 = [int(v*224) for v in roi]
        mass = float(cam[y0:y1,x0:x1].sum()/total)
        area_fraction = float((x1-x0)*(y1-y0)/(224*224))
        enrichment = mass/max(area_fraction,1e-6)
        # Raw mass alone unfairly rewards large ROIs; normalize by ROI area.
        # These cutoffs are exploratory, NOT calibrated localization criteria.
        if mass < .12 or enrichment < .70:
            verdict='FAIL'
        elif mass >= min_region_fraction and enrichment >= 1.15:
            verdict='PROVISIONAL_PASS'
        else:
            verdict='UNKNOWN'
        details.append({'entity':entity, 'verdict':verdict,
                        'region_cam_mass':round(mass,4),
                        'region_area_fraction':round(area_fraction,4),
                        'area_normalized_enrichment':round(enrichment,4),
                        'region':list(roi)})
    if any(x['verdict']=='FAIL' for x in details):
        status='FAIL'
    elif details and all(x['verdict']=='PROVISIONAL_PASS' for x in details):
        status='PROVISIONAL_PASS'
    else:
        status='UNKNOWN'
    if conflicts:
        status = 'FAIL'
    return {'status':status, 'details':details,
            'reason': ('Internal report contradiction; Grad-CAM cannot resolve it.' if conflicts else
                       'Research-only frontal ROI heuristic; saliency does not verify anatomical pathology.'),
            'projection':projection, 'contradictions':conflicts}, first_cam


def make_cam_overlay(pil_img: Image.Image, cam):
    import io, base64
    if cam is None:
        return None
    original = np.asarray(pil_img.convert('RGB').resize((224,224)), dtype='float32')
    red = np.zeros_like(original)
    red[...,0] = 255*np.clip(cam,0,1)
    red[...,1] = 45*np.clip(cam,0,1)
    blended = np.uint8(np.clip(.74*original + .26*red,0,255))
    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
