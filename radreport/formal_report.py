"""Presentation-friendly, fail-closed structured radiology DRAFT formatting.

Important: typography and formatting are NOT medical validation. Never invent
patient demographics, clinical history, comparison images or unseen findings.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from .report_annotations import annotate_report

DISCLAIM = 'AI-GENERATED EDUCATIONAL RESEARCH DRAFT — NOT FOR CLINICAL USE. Qualified radiologist review required.'
_SECTION = re.compile(r'(?im)^\s*(?:\*\*)?\b(FINDINGS|IMPRESSION)\b(?:\*\*)?\s*:', re.I)


def training_target(row: dict, mode: str = 'impression') -> str:
    """Use actual training-set report fields, never synthetic positive findings."""
    impression = str(row.get('impression') or '').strip()
    if not impression or impression.lower() in ('nan', 'none'):
        raise ValueError('Missing reference impression')
    if mode == 'impression':
        return 'Impression: ' + impression
    if mode != 'structured':
        raise ValueError('Unsupported target mode')
    findings = str(row.get('findings') or '').strip()
    if not findings or findings.lower() in ('nan', 'none'):
        raise ValueError('Structured training requires actual findings and impression in source data')
    return 'FINDINGS:\n' + findings + '\nIMPRESSION:\n' + impression


def parse_generated(raw: str | None, structured_expected: bool = False) -> dict:
    """Copy generated claims faithfully; do not auto-correct or invent observations."""
    raw = (raw or '').strip()
    if not raw:
        return {'findings': 'No model-generated findings available.',
                'impression': 'No model-generated impression available.',
                'source': 'NO_INFERENCE', 'complete': False}
    tags = list(_SECTION.finditer(raw))
    fields: dict[str,str] = {}
    for i, tag in enumerate(tags):
        value = raw[tag.end(): tags[i+1].start() if i+1 < len(tags) else len(raw)].strip()
        fields[tag.group(1).upper()] = value
    if 'FINDINGS' in fields:
        # Missing either section is explicitly incomplete. Never fill it with guesses.
        findings = fields.get('FINDINGS') or 'Not generated.'
        impression = fields.get('IMPRESSION') or 'Not generated.'
        return {'findings':findings, 'impression':impression, 'source':'MODEL_GENERATED_BOTH',
                'complete':bool(fields.get('FINDINGS') and fields.get('IMPRESSION'))}
    impression = fields.get('IMPRESSION') or re.sub(r'(?i)^\s*impression\s*:\s*', '', raw).strip()
    return {'findings':'Not generated separately. The current checkpoint was trained on impressions only; see Impression.',
            'impression': impression or 'No model-generated impression available.',
            'source':'IMPRESSION_ONLY_MODEL' if not structured_expected else 'STRUCTURED_OUTPUT_MISSING',
            'complete':not structured_expected}


def build_formal_report(*, raw: str | None, projection: str, report_id: str,
                        model_version: str, trained: bool, structured_expected: bool = False,
                        safety: dict | None = None, image_scores: dict | None = None) -> dict:
    parsed = parse_generated(raw if trained else None, structured_expected=structured_expected)
    view = {'frontal':'Frontal view selected (PA/AP not differentiated)',
            'lateral':'Lateral view selected', 'unknown':'Projection not confirmed'}.get(projection,'Projection not confirmed')
    safety = safety or {}
    annotations = annotate_report(raw if trained else None, image_scores if trained else None)
    status = safety.get('status', 'MANDATORY_REVIEW')
    notes = []
    if not parsed['complete']:
        notes.append('Full Findings and Impression text is incomplete or unavailable; manual completion required.')
    entity = safety.get('entity_check', {})
    if (annotations['contradictions'] or entity.get('contradictions')
            or entity.get('claims', {}).get('contradictions')):
        notes.append('Internal contradiction(s) detected. Do not use as a finalized report.')
    if safety.get('review_reasons'):
        notes.extend(safety['review_reasons'])
    notes = list(dict.fromkeys(notes))
    return {
        'report_title': 'CHEST RADIOGRAPH — AI RESEARCH DRAFT',
        'report_id': report_id,
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'patient_details': 'Not provided; no identifying data stored',
        'clinical_history': 'Not supplied',
        'examination': 'Chest radiograph',
        'technique': f'One uploaded radiograph. {view}; not independently verified by software.',
        'comparison': 'No prior imaging supplied',
        'findings': parsed['findings'],
        'impression': parsed['impression'],
        'observations': annotations['observations'],
        'labels': annotations['labels'],
        'mesh_terms': annotations['mesh_terms'],
        'image_signals': annotations['image_signals'],
        'normal_statement': annotations['normal_statement'],
        'report_annotations': annotations,
        'annotations_note': annotations['text_label_note'],
        'image_signals_note': annotations['image_signal_note'],
        'model_version': model_version,
        'report_source': parsed['source'],
        'complete_sections': parsed['complete'],
        'both_sections_generated': bool(parsed['source'] == 'MODEL_GENERATED_BOTH' and parsed['complete']),
        'review_status': status if trained else 'NO_INFERENCE',
        'radiologist': 'Not reviewed / no clinical sign-off',
        'quality_flags': notes,
        'disclaimer': DISCLAIM,
        'is_final': False,
    }
