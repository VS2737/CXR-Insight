"""Finding-aware, fail-closed research checks for CXR Insight v4.

NOT a clinical validator. The text claim parser is deterministic and incomplete;
image probabilities come from a weak-label classifier. No result authorizes care.
Compatible with the original `extract_positive_entities`, `route`, and API keys.
"""
from __future__ import annotations

import re
import os
from itertools import combinations
from .config import FINDINGS, SELF_CONSISTENCY_THRESHOLD, IMAGE_ENTITY_THRESHOLD

# Coverage is limited to the CheXpert / CheXbert ontology. Use explicit abstention
# for potentially meaningful terms outside that ontology.
ALIASES = {
    'Atelectasis': [r'atelecta\w+', r'collapsed (?:lung|lobe)'],
    'Cardiomegaly': [r'cardiomegal\w+', r'enlarged (?:cardiac|heart) silhouette', r'heart (?:is )?enlarged'],
    'Consolidation': [r'consolidat\w+'],
    'Edema': [r'(?:pulmonary )?(?:edema|oedema)'],
    'Enlarged Cardiomediastinum': [r'wid(?:e|ened) mediastin\w+'],
    'Fracture': [r'fractur\w+'],
    'Lung Lesion': [r'(?:lung|pulmonary) (?:lesion|mass|nodule)s?'],
    'Lung Opacity': [r'\bopacit(?:y|ies)\b', r'\bopacification\b',
                     r'\bair[ -]?space (?:disease|process|infiltrat\w*|opacit\w*)\b',
                     r'\bpulmonary (?:infiltrat\w*|opacit\w*)\b',
                     r'\binfiltrat(?:e|es|ion|ions|ive)\b', r'\bair[ -]?space (?:densit\w*|chang\w*)\b'],
    'Pleural Effusion': [r'pleural effusion\w*', r'pleural fluid'],
    'Pleural Other': [r'pleural thicken\w+'],
    'Pneumonia': [r'pneumoni\w+', r'pneumonic'],
    'Pneumothorax': [r'pneumothora\w+'],
    'Support Devices': [r'(?:endotracheal|chest) tube', r'central (?:venous )?catheter', r'pacemaker'],
}
NEGATION_BEFORE = re.compile(r'\b(?:no|without|absence of|negative for|free of|no evidence of|no sign of|without evidence of)\b', re.I)
NEGATION_AFTER = re.compile(r'\b(?:not (?:seen|identified|present)|is absent|are absent|ruled out)\b', re.I)
HEDGE = re.compile(r'\b(?:possible|possibly|may (?:be|represent)|might be|cannot (?:be )?exclud(?:e|ed)|could represent|questionable|suspicious for|suggestive of|not excluded|difficult to exclude)\b', re.I)
LUNG_CLEAR = re.compile(r'\b(?:clear lungs|lungs (?:are|appear) clear|both lungs (?:are|appear) clear|no focal (?:lung|pulmonary|airspace) (?:disease|opacity|infiltrate))\b', re.I)
ACUTE_NORMAL = re.compile(r'\b(?:no acute (?:cardiopulmonary|pulmonary|chest)(?: (?:abnormalit\w*|disease|process|findings?))?|no (?:active|acute) disease|negative (?:chest|for acute (?:cardiopulmonary|pulmonary))|no evidence of active disease)\b', re.I)
GENERAL_NORMAL = re.compile(r'\b(?:normal chest|unremarkable (?:chest|radiograph|lungs)|no (?:cardiopulmonary|pulmonary) abnormalit\w*)\b', re.I)
OUT_OF_ONTOLOGY = re.compile(r'\b(?:fibrosis|emphysema|bronchiectasis|interstitial lung disease|lymphadenopathy|aortic aneurysm|hiatal hernia)\b', re.I)
CLAUSES = re.compile(r'[.;\n]+|\b(?:but|however|although)\b', re.I)
STATUS_RANK = {'NOT_MENTIONED': 0, 'ABSENT': 1, 'UNCERTAIN': 2, 'PRESENT': 3}
LUNG_FINDINGS = {'Atelectasis','Consolidation','Edema','Lung Lesion','Lung Opacity','Pneumonia','Pneumothorax'}
ACUTE_FINDINGS = {'Consolidation','Lung Opacity','Pneumonia','Pneumothorax'}



def _local_state(clause: str, start: int, end: int) -> str:
    """Lightweight local negation/hedge handling; not a full clinical NLP parser."""
    before = clause[max(0, start - 75):start]
    after = clause[end:end + 44]
    # Do not propagate negation across a contrast delimiter ("no effusion but opacity").
    before = re.split(r'\b(?:but|however|although|except)\b', before, flags=re.I)[-1]
    # "cannot exclude" is uncertain, not a definite negative.
    if HEDGE.search(before) or HEDGE.search(after):
        return 'UNCERTAIN'
    if NEGATION_BEFORE.search(before) or NEGATION_AFTER.search(after):
        return 'ABSENT'
    return 'PRESENT'


def _contradictions(states: dict, mentions: dict, scope: str, clear_lungs: bool,
                    locations: dict) -> list[dict]:
    """Return review flags, not diagnostic truth statements."""
    conflicts = []
    for name, polarities in mentions.items():
        if 'PRESENT' in polarities and 'ABSENT' in polarities:
            pos = set(locations[name].get('PRESENT', []))
            neg = set(locations[name].get('ABSENT', []))
            # Two different explicitly specified regions can both be true.
            # Unknown-region claims cannot be safely disambiguated.
            if pos and neg and None not in pos and None not in neg and pos.isdisjoint(neg):
                continue
            conflicts.append({'type':'direct_finding_conflict','finding':name,
                              'reason':f'{name} is both asserted and negated.'})
    for name, state in states.items():
        if state != 'PRESENT':
            continue
        if scope == 'GENERAL':
            conflicts.append({'type':'normal_chest_conflict','finding':name,
                              'reason':f'General normal chest claim conflicts with asserted {name}.'})
        elif clear_lungs and name in LUNG_FINDINGS:
            conflicts.append({'type':'clear_lungs_conflict','finding':name,
                              'reason':f'Clear-lungs claim conflicts with asserted {name}.'})
        elif scope == 'ACUTE_ONLY' and name in ACUTE_FINDINGS:
            conflicts.append({'type':'acute_normal_conflict','finding':name,
                              'reason':f'No-acute-disease claim may conflict with asserted {name}.'})
    # Only one conflict per (type, finding). Do not pretend this catches every nuance.
    return list({(x['type'],x['finding']):x for x in conflicts}.values())


def report_claims(text: str) -> dict:
    """Extract finding polarity, normal claims, and internally contradictory text.

    These are heuristic *review triggers*. No label here certifies radiology truth.
    Negative wording is not assumed to prove a normal image.
    """
    content = text or ''
    states = {name: 'NOT_MENTIONED' for name in FINDINGS if name != 'No Finding'}
    mentions = {name: set() for name in states}
    locations = {name: {'PRESENT': [], 'ABSENT': []} for name in states}
    for clause in CLAUSES.split(content):
        clause = clause.strip()
        if not clause:
            continue
        for name, patterns in ALIASES.items():
            for pattern in patterns:
                for match in re.finditer(pattern, clause, flags=re.I):
                    state = _local_state(clause, match.start(), match.end())
                    mentions[name].add(state)
                    if state in ('PRESENT','ABSENT'):
                        location_match = re.search(r'\b(left|right)\s+(upper|lower|middle|mid|basilar|apical|base)\b',
                                                   clause, flags=re.I)
                        location = (location_match.group(1).lower() + ' ' + location_match.group(2).lower()) \
                            if location_match else None
                        locations[name][state].append(location)
                    if STATUS_RANK[state] > STATUS_RANK[states[name]]:
                        states[name] = state
    # "Otherwise clear lungs" is compatible with a specifically described opacity.
    clear_lungs = any(not re.search(r'\b(?:otherwise|remaining|rest of the)\b',
                                    content[max(0,m.start()-30):m.start()], re.I)
                      for m in LUNG_CLEAR.finditer(content))
    scope = ('GENERAL' if GENERAL_NORMAL.search(content) else
             'LUNGS_ONLY' if clear_lungs else
             'ACUTE_ONLY' if ACUTE_NORMAL.search(content) else 'NONE')
    # A report can say "lungs clear" + "no acute process"; keep both signals.
    acute_claim = bool(ACUTE_NORMAL.search(content))
    conflicts = _contradictions(states, mentions, scope, clear_lungs, locations)
    if acute_claim and scope in ('LUNGS_ONLY', 'GENERAL'):
        for name in ACUTE_FINDINGS:
            if states.get(name) == 'PRESENT' and not any(x['finding']==name for x in conflicts):
                conflicts.append({'type':'acute_normal_conflict','finding':name,
                                  'reason':f'No-acute claim may conflict with {name}.'})
    uncovered = sorted(set(m.group(0).lower() for m in OUT_OF_ONTOLOGY.finditer(content)))
    parser = 'rule_based_research_parser_v4'
    issue = None
    if os.getenv('CXR_USE_CHEXBERT','0') == '1':
        try:
            from .text_chexbert import chexbert_claims
            # CheXbert augments the finding labels. Keep lexical contradiction
            # checks even if CheXbert smooths conflicting sentences into one label.
            bert_states = chexbert_claims(content)
            for name, state in bert_states.items():
                if state != 'NOT_MENTIONED' or states[name] == 'NOT_MENTIONED':
                    states[name] = state
            parser = 'chexbert_plus_lexical_conflict_v4'
        except Exception as exc:
            issue = f'CheXbert unavailable: {type(exc).__name__}. Rule parser fallback; review required.'
            uncovered.append('chexbert_unavailable')
    return {'findings':states, 'normal_scope':scope, 'uncovered_terms':uncovered,
            'parser':parser, 'parser_issue':issue, 'contradictions':conflicts,
            'has_clear_lungs_claim':clear_lungs, 'has_no_acute_claim':acute_claim}


def report_contradictions(text: str) -> list[dict]:
    return report_claims(text)['contradictions']


def extract_positive_entities(text: str) -> list[str]:
    """Backward-compatible positive finding list for the visual grounding code."""
    states = report_claims(text)['findings']
    return [name for name, status in states.items() if status == 'PRESENT']


def _pair_agreement(a: dict, b: dict) -> float:
    """Score factual contradictions; identical wording is neither needed nor rewarded."""
    states_a, states_b = a['findings'], b['findings']
    mentioned = {k for k in states_a if states_a[k] != 'NOT_MENTIONED' or states_b[k] != 'NOT_MENTIONED'}
    if mentioned:
        per_entity = []
        for k in mentioned:
            x, y = states_a[k], states_b[k]
            if x == y:
                per_entity.append(1.0)
            elif 'NOT_MENTIONED' in (x, y):
                per_entity.append(.25)  # one sample omits the finding
            elif 'UNCERTAIN' in (x, y):
                per_entity.append(.4)
            else:
                per_entity.append(0.0)  # explicit positive vs explicit negative
        finding_score = sum(per_entity) / len(per_entity)
    else:
        finding_score = 1.0
    n_a, n_b = a['normal_scope'], b['normal_scope']
    if n_a == n_b:
        normal_score = 1.0
    elif 'NONE' in (n_a, n_b):
        normal_score = 0.4
    else:
        normal_score = 0.85  # normal and no-acute broadly agree but aren't identical
    # Internal inconsistencies are reported separately. Agreement across five
    # copies of the SAME contradictory report can remain 100%, which is why
    # stability must never be presented as clinical validity.
    return 0.85 * finding_score + 0.15 * normal_score


def self_consistency_details(texts: list[str]) -> dict:
    if len(texts) < 2:
        return {'score': None, 'reason': 'At least two independently sampled reports required.', 'discordant_pairs': []}
    parsed = [report_claims(t) for t in texts]
    scores = []
    differing = []
    for i, j in combinations(range(len(parsed)), 2):
        val = _pair_agreement(parsed[i], parsed[j])
        scores.append(val)
        if val < .75:
            differing.append({'a': i+1, 'b': j+1, 'finding_agreement': round(val, 3)})
    return {'score': round(sum(scores)/len(scores), 4), 'method': 'finding_polarity_pairwise_v4',
            'n_decodings': len(texts), 'discordant_pairs': differing[:15],
            'internally_contradictory_decodings': [i+1 for i,c in enumerate(parsed) if c['contradictions']],
            'warning': 'Agreement measures generation stability, NOT whether the image supports the report.'}


def self_consistency(texts: list[str]) -> float | None:
    return self_consistency_details(texts)['score']


def entity_consistency(report: str, image_scores: dict | None, threshold: float = IMAGE_ENTITY_THRESHOLD):
    claims = report_claims(report)
    named = [name for name, status in claims['findings'].items() if status == 'PRESENT']
    negatives = [name for name, status in claims['findings'].items() if status == 'ABSENT']
    uncertainties = [name for name, status in claims['findings'].items() if status == 'UNCERTAIN']
    if claims['contradictions']:
        return {'status':'FAIL', 'claimed_entities':named, 'negated_entities':negatives,
                'uncertain_entities':uncertainties, 'unsupported':[], 'possible_omissions':[],
                'claims':claims, 'contradictions':claims['contradictions'],
                'reason':'Report contains internally contradictory statements; mandatory review required.'}
    if image_scores is None:
        return {'status':'UNKNOWN', 'claimed_entities':named, 'negated_entities':negatives,
                'uncertain_entities':uncertainties, 'unsupported':named,
                'reason':'Image classifier missing or not weak-label calibrated; no image-based comparison.',
                'claims': claims}
    unsupported = []
    possible_omissions = []
    for name in named:
        if name not in image_scores or image_scores[name] < threshold:
            unsupported.append(name)
    # Explicitly negated diagnoses should be challenged if the weak classifier strongly disagrees.
    for name in negatives:
        if image_scores.get(name, 0.0) >= max(.85, threshold):
            possible_omissions.append(name)
    # Normal reports are the largest weakness in the original code. Never call them
    # image-verified: inspect high classifier scores as *possible* missed findings.
    if claims['normal_scope'] != 'NONE':
        for name, p in image_scores.items():
            if name not in ('No Finding', 'Support Devices') and p >= max(.85, threshold) and name not in named:
                if name not in possible_omissions:
                    possible_omissions.append(name)
    if unsupported or possible_omissions:
        status = 'FAIL'  # conflict flags require review, do NOT prove image truth
        reason = 'Weak image classifier conflicts with one or more named/negated claims; manual review needed.'
    elif claims['uncovered_terms'] or claims['parser_issue']:
        status = 'UNVERIFIED'
        reason = 'Potential findings outside the supported 14-label ontology.'
    elif not named:
        status = 'UNVERIFIED'
        reason = 'No positive finding to confirm; normal/negative statements cannot be certified by this classifier.'
    else:
        status = 'PROVISIONAL_PASS'
        reason = 'Named finding has weak-label image-model support, not independently verified ground truth.'
    return {'status':status, 'claimed_entities':named, 'negated_entities': negatives,
            'uncertain_entities':uncertainties, 'unsupported':unsupported,
            'possible_omissions':possible_omissions, 'claims': claims,
            'contradictions':claims['contradictions'],
            'reason':reason, 'classifier_threshold':threshold,
            'warning':'Image score is research model output, NOT a diagnosis.'}


def route(report: str, variants: list[str], image_scores: dict | None,
          grounding: dict | None, model_ready: bool, threshold: float = SELF_CONSISTENCY_THRESHOLD):
    entity = entity_consistency(report, image_scores)
    details = self_consistency_details(variants)
    score = details['score']
    if grounding is None:
        grounding = {'status':'UNKNOWN', 'reason':'Class-specific Grad-CAM could not be computed.'}
    blockers = []
    if not model_ready: blockers.append('No trained generator checkpoint')
    if entity.get('contradictions') or entity.get('claims',{}).get('contradictions'):
        blockers.append('Internally contradictory generated impression')
    if entity['status'] != 'PROVISIONAL_PASS': blockers.append('Entity-image agreement not verified / possible omission')
    if score is None or score < threshold: blockers.append('Low or unavailable finding-level self-consistency')
    if grounding.get('status') != 'PROVISIONAL_PASS': blockers.append('Grad-CAM localization unverified or flagged')
    # Research prototype: never auto-publish regardless of provisional flags.
    return {'entity_check':entity, 'self_consistency':score, 'self_consistency_details':details,
            'grounding':grounding, 'safety_check_passed':not blockers,
            'review_required':True, 'review_reasons':blockers or ['All research drafts still require human review'],
            'status':'MANDATORY_REVIEW', 'safety_version':'v4-contradiction-aware'}
