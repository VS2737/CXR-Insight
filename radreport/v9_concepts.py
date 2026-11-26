"""Sparse report-derived concepts for experimental finding-aware training.

These are NOT radiologist-verified image labels. Unknown/uncertain labels are
always masked out, rather than treated as negatives. Only IU TRAIN labels may
be used for fitting. No-acute-disease is NOT synonymous with normal anatomy.
"""
from __future__ import annotations
import re
from radreport.config import FINDINGS
from radreport.label_values import parse_binary_chexpert
from radreport.safety import report_claims

BASE = tuple(x for x in FINDINGS if x != 'No Finding')
EXTRA = ('Pulmonary Artery Enlargement', 'Prior Sternotomy')
CONCEPTS = BASE + EXTRA

VASCULAR_POS = re.compile(r'\b(?:enlarged|dilat(?:ed|ation)|prominent|ectatic)\s+pulmonary\s+arter(?:y|ies)\b|\bpulmonary arterial enlargement\b', re.I)
VASCULAR_NEG = re.compile(r'\b(?:normal|within normal limits|unremarkable)\s+pulmonary\s+(?:arter(?:y|ies)|vascularity)\b|\bpulmonary vascularity\s+(?:is |appears? )?(?:normal|within normal limits)\b|\bno (?:pulmonary )?arterial enlargement\b', re.I)
STERN_POS = re.compile(r'\b(?:median\s+)?sternotomy(?:\s+wires)?\b|\bstatus post sternotomy\b', re.I)
STERN_NEG = re.compile(r'\bno (?:evidence of |prior )?sternotomy\b',re.I)
NORMAL_HEART = re.compile(r'\bheart size and pulmonary vascularity\s+(?:appear|are)\s+within normal limits\b|\bheart\s+(?:size\s+)?(?:is\s+|appears?\s+)?(?:normal|within normal limits)\b|\b(?:normal|not enlarged)\s+(?:cardiac silhouette|heart size)\b', re.I)


def special_concepts(text: str) -> dict[str, int | None]:
    text = text or ''
    out = {}
    for name, positive, negative in ((EXTRA[0],VASCULAR_POS,VASCULAR_NEG),(EXTRA[1],STERN_POS,STERN_NEG)):
        pos = bool(positive.search(text)); neg = bool(negative.search(text))
        # Contradiction: do not silently choose one side.
        out[name] = None if pos and neg else 1 if pos else 0 if neg else None
    return out


def make_targets(row: dict, label_row: dict | None) -> tuple[list[float],list[float]]:
    """One-hot-style binary targets plus masks for *known* weak labels only."""
    label_row = label_row or {}
    findings = str(row.get('findings') or '')
    result = []
    for name in BASE:
        result.append(parse_binary_chexpert(label_row.get(name)))
    extra = special_concepts(findings)
    result.extend(extra[name] for name in EXTRA)
    y = [float(v) if v is not None else 0.0 for v in result]
    mask = [1.0 if v is not None else 0.0 for v in result]
    return y,mask


def claims_for_audit(text: str) -> dict[str,str]:
    """Extract mention polarity from text; never use as true image diagnosis."""
    states = dict(report_claims(text)['findings'])
    for name,value in special_concepts(text).items():
        states[name] = 'PRESENT' if value == 1 else 'ABSENT' if value == 0 else 'NOT_MENTIONED'
    if NORMAL_HEART.search(text or '') and states.get('Cardiomegaly') == 'NOT_MENTIONED':
        states['Cardiomegaly'] = 'ABSENT'
    return states


def compare_reference_generated(reference: str, generated: str) -> dict:
    """Finding-mention comparison, NOT medically confirmed accuracy."""
    ref = claims_for_audit(reference)
    gen = claims_for_audit(generated)
    present = {k for k,v in ref.items() if v == 'PRESENT'}
    matched = sorted(k for k in present if gen.get(k) == 'PRESENT')
    omitted = sorted(k for k in present if gen.get(k) != 'PRESENT')
    contradicted = sorted(k for k in present if gen.get(k) == 'ABSENT')
    unsupported = sorted(k for k,v in gen.items() if v == 'PRESENT' and ref.get(k) != 'PRESENT')
    return {'reference_positive':sorted(present),'matched_mentions':matched,
            'omitted_mentions':omitted,'explicitly_negated_reference_mentions':contradicted,
            'positive_generated_not_in_reference':unsupported,
            'warning':'Text mention comparison only. Unmentioned does not mean medically absent.'}
