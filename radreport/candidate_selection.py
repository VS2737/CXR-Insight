"""Optional contradiction-aware candidate choice for coursework experiments.

Selection is NOT diagnostic verification; maintain the original five raw samples.
Avoid quietly replacing a clinically relevant anomaly with common normal language.
Default app mode remains 'first'. The experimental 'noncontradictory' mode only
replaces the first candidate if it has a clearly detectable internal conflict.
"""
from __future__ import annotations
from .safety import report_claims


def select_draft(texts: list[str], mode: str = 'first') -> dict:
    if not texts:
        return {'index': None, 'original_index': None, 'changed': False,
                'reason': 'No generated candidates.', 'candidates': []}
    diagnostics = []
    for i, text in enumerate(texts):
        claims = report_claims(text)
        diagnostics.append({'index':i, 'contradiction_count':len(claims['contradictions']),
                            'uncovered_terms':claims['uncovered_terms'],
                            'parser_issue':claims['parser_issue'],
                            'text_length':len(text.strip())})
    index = 0
    reason = 'Original first decoding retained (default; no candidate selection).'
    if mode == 'clean_text_v10':
        from .v10_text import artifact_flags
        diagnostics = [dict(d, text_quality_flags=artifact_flags(t)) for d,t in zip(diagnostics,texts)]
        if diagnostics[0]['text_quality_flags']:
            alternatives = [d for d in diagnostics[1:] if not d['text_quality_flags']
                            and not d['contradiction_count'] and not d['uncovered_terms']
                            and not d['parser_issue']]
            if alternatives:
                index = alternatives[0]['index']
                reason = 'Selected cleanly formatted alternate draft; selection is NOT medical validation.'
            else:
                reason = 'No clean alternate found; original draft retained and flagged.'
        else:
            reason = 'First draft has complete sections and no obvious redaction artifacts; retained.'
    elif mode == 'noncontradictory' and diagnostics[0]['contradiction_count']:
        # Abstain from any re-ranking if all samples are contradictory.
        good = [x for x in diagnostics if (x['contradiction_count'] == 0
                and x['text_length'] >= 12 and not x['uncovered_terms']
                and not x['parser_issue'])]
        if good:
            # Keep a deterministic original-order tie break. Do not select by
            # model confidence as the weak classifier is not validated.
            index = good[0]['index']
            reason = 'First decoding has an internal contradiction; selected an uncontradicted sample for review only.'
        else:
            reason = 'No clearly uncontradicted alternative; first decoding retained and flagged.'
    elif mode not in ('first','noncontradictory','clean_text_v10'):
        reason = 'Unsupported selection mode; first decoding retained.'
    return {'index':index, 'original_index':0, 'changed':index!=0,
            'mode':mode, 'reason':reason, 'candidates':diagnostics,
            'warning':'Candidate selection is NOT image-grounded diagnostic validation.'}
