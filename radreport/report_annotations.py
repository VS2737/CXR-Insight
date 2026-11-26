"""Conservative presentation annotations, derived from UNVERIFIED model outputs.

These are not radiologist observations or validated MeSH classifications. The
source of each label is explicit; we never turn an unmentioned finding into an
image diagnosis. Image classifier scores are separate and weakly supervised.
"""
from __future__ import annotations

import re
from .config import FINDINGS, IMAGE_ENTITY_THRESHOLD
from .safety import ALIASES, report_claims, _local_state

# Limited controlled-vocabulary *suggestions*. These strings are not MeSH IDs,
# do not prove a diagnosis, and must be reviewed before clinical indexing.
MESH_SUGGESTIONS = {
    'Atelectasis': 'Pulmonary Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Lung Diseases',
    'Edema': 'Pulmonary Edema',
    'Fracture': 'Fractures, Bone',
    'Lung Opacity': 'Lung Diseases',
    'Pleural Effusion': 'Pleural Effusion',
    'Pleural Other': 'Pleural Diseases',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
}


def _finding_sentences(text: str, positives: set[str]) -> list[str]:
    """Quote brief sentences containing unnegated *claimed* positive findings.

    Source text is never expanded into a new pathology assertion. Conservative:
    if a clause contains a negative cue for that entity, do not list it here.
    """
    clean = re.sub(r'(?im)^\s*(?:findings|impression)\s*:\s*', '', text or '')
    chunks = re.split(r'(?<=[.!?])\s+(?=(?:\d+\.\s*)?[A-Z])|\n+', clean)
    observations: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r'^\s*\d+\.\s*', '', chunk.strip())
        if not chunk:
            continue
        for entity in positives:
            for pattern in ALIASES.get(entity, []):
                matches = list(re.finditer(pattern, chunk, re.I))
                if any(_local_state(chunk, m.start(), m.end()) == 'PRESENT' for m in matches):
                    if chunk not in observations:
                        observations.append(chunk)
                    break
    return observations[:12]


def annotate_report(text: str | None, image_scores: dict | None = None) -> dict:
    """Return labels/observations to PRESENT in a report (no diagnostic claim)."""
    content = (text or '').strip()
    if not content:
        return {
            'observations': [], 'labels': [], 'mesh_terms': [], 'image_signals': [],
            'normal_statement': None, 'contradictions': [], 'uncovered_terms': [],
            'text_label_note': 'No model text available. Nothing can be labeled.',
            'image_signal_note': 'No image findings verified.',
        }
    claims = report_claims(content)
    positives = {name for name, polarity in claims['findings'].items() if polarity == 'PRESENT'}
    labels = [
        {'name': name, 'polarity': polarity, 'source': 'AI-generated report text'}
        for name, polarity in claims['findings'].items()
        if polarity != 'NOT_MENTIONED'
    ]
    mesh = sorted({MESH_SUGGESTIONS[name] for name in positives if name in MESH_SUGGESTIONS})
    observations = _finding_sentences(content, positives)
    scope = claims['normal_scope']
    normal = {
        'GENERAL': 'The draft contains a general normal-chest statement.',
        'ACUTE_ONLY': 'The draft states no acute abnormality; chronic abnormalities are not excluded.',
        'LUNGS_ONLY': 'The draft contains a clear-lungs statement.',
    }.get(scope)
    if normal and not labels:
        labels.append({'name':'Normal/negative statement (draft claim)',
                       'polarity':'UNVERIFIED', 'source':'AI-generated report text'})
    # Do not treat image classifier scores as labels derived from report text;
    # expose them separately with a calibrated-weak-supervision disclosure.
    image_signals = []
    if image_scores is not None:
        candidates = sorted(
            [(name, float(score)) for name, score in image_scores.items()
             if name in FINDINGS and name != 'No Finding' and score is not None],
            key=lambda x: x[1], reverse=True)
        image_signals = [
            {'name': name, 'score': round(score, 3),
             'threshold': IMAGE_ENTITY_THRESHOLD, 'source': 'Weak-label X-ray classifier'}
            for name, score in candidates if score >= IMAGE_ENTITY_THRESHOLD
        ][:6]
    return {
        'observations': observations,
        'labels': labels,
        'mesh_terms': mesh,
        'image_signals': image_signals,
        'normal_statement': normal,
        'contradictions': claims['contradictions'],
        'uncovered_terms': claims['uncovered_terms'],
        'text_label_note': 'Observations/labels are EXTRACTED from the AI-written draft; they are not independent image findings.',
        'image_signal_note': ('Separate weak-label classifier scores, not confirmed diagnoses or calibrated clinical probabilities.'
                              if image_scores is not None else
                              'No eligible image-classifier scores available. This does not mean the X-ray is normal.'),
    }
