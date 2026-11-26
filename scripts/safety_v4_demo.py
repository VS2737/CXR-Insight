"""Lightweight controlled examples. This is NOT an image-level medical validation."""
import json
from radreport.safety import report_claims, route
from radreport.candidate_selection import select_draft
from radreport.classifier import region_for_entity

cases = [
    ('Contradictory impression', 'Patchy left lower lobe airspace disease. Clear lungs.'),
    ('Plausible no-acute with chronic finding', 'Mild cardiomegaly. No acute cardiopulmonary disease.'),
    ('Bilateral localization abstains', 'Bilateral lower lobe airspace disease.'),
]
for title, text in cases:
    claims=report_claims(text)
    positives=[n for n,v in claims['findings'].items() if v == 'PRESENT']
    print('\n'+title+'\n'+text)
    print(json.dumps({'positives':positives,'contradictions':claims['contradictions'],
                      'rois':{n:region_for_entity(text,n) for n in positives}},indent=2))

bad = 'Patchy left lower lobe airspace disease. Clear lungs.'
variants=[bad,'Patchy left lower lobe airspace disease.','No acute disease.',
          'Left lower lobe opacity.','Patchy left lower lobe opacity.']
print('\nSafety route example:')
print(json.dumps(route(bad,variants,{'Lung Opacity':.8},None,True),indent=2))
print('\nExperimental draft selection:')
print(json.dumps(select_draft(variants, 'noncontradictory'),indent=2))
