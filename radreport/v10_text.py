"""Loss-aware cleanup of de-identification artifacts in IU training targets.

This NEVER edits original CSV or frozen train/val/test JSONLs. It is for
training target text only, not automatic clinical correction of model reports.
Clauses with missing medical content are discarded rather than guessed.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

REDACTION = re.compile(r'\b(?:x{3,}|__+|\[redacted\]|<unk>)\b', re.I)
NEAR_REDACTION = re.compile(r'(?i)\b(?:chest\s+x[-\s]*x{2,}|\bunknown\s+xxxx)')
MIDLINE_STERN = re.compile(r'(?i)\b((?:midline|median|prior|previous|status\s+post)\s+sternotomy|sternotomy\s+wires|sternotomy)\s+(?:x{3,}\s*)+(?=\.|$)')
SOFT_BLANK = re.compile(r'\s+')
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)
VERSION = 'v10_iu_conservative_cleaner_1'


@dataclass(frozen=True)
class CleanResult:
    text: str
    retained_clauses: int
    removed_clauses: int
    redacted_clauses: int
    changed: bool


def clean_section(section: str | None) -> CleanResult:
    """Drop redaction-damaged statements; retain safe standalone surgery mentions.

    Never invent a diagnosis or silently negate a finding. Split on sentence
    boundaries while avoiding the common `1. 2.` numbering convention.
    """
    source = str(section or '').strip()
    if not source or source.lower() in ('none', 'nan'):
        return CleanResult('',0,0,0,bool(source))
    source = source.replace('\r', '\n')
    # Numbered findings are separate clauses; remove index only, not content.
    source = re.sub(r'(?<!\S)\d+\.\s*', '', source)
    source = re.sub(r'(?<=\.)\s+(?=\d+\.\s+)', '\n', source)
    clauses = re.split(r'(?<=[.!?])\s+|\n+', source)
    kept=[];removed=0;redacted=0
    seen=set()
    for raw in clauses:
        s=SOFT_BLANK.sub(' ',raw).strip(' \t\r\n')
        if not s:continue
        s = MIDLINE_STERN.sub(r'\1',s)
        affected=bool(REDACTION.search(s) or NEAR_REDACTION.search(s))
        if affected:
            redacted+=1;removed+=1;continue
        # A lonely de-ID remnant / numbered marker is not a useful sentence.
        if len(WORD.findall(s)) < 2 and s.strip(' .!?').lower() not in (
                'cardiomegaly','atelectasis','pneumonia','pneumothorax','edema',
                'emphysema','consolidation','sternotomy','hyperinflation'):
            removed+=1;continue
        s=s.strip()
        if not s.endswith(('.', '!', '?')):
            s += '.'
        key=s.lower()
        if key in seen:continue
        seen.add(key);kept.append(s)
    cleaned=' '.join(kept)
    return CleanResult(cleaned,len(kept),removed,redacted,cleaned != source)


def clean_pair(row:dict) -> tuple[str | None, dict]:
    """Return structured training target or None when content is insufficient."""
    f=clean_section(row.get('findings'))
    i=clean_section(row.get('impression'))
    data={'uid':str(row.get('uid','')),'findings_removed':f.removed_clauses,
          'impression_removed':i.removed_clauses,'redacted_clauses':f.redacted_clauses+i.redacted_clauses,
          'retained_findings':f.retained_clauses,'retained_impression':i.retained_clauses,
          'changed': f.changed or i.changed,'usable':bool(f.text and i.text)}
    if not data['usable']:
        return None,data
    return f'FINDINGS:\n{f.text}\nIMPRESSION:\n{i.text}',data


def artifact_flags(text: str | None) -> list[str]:
    """Only describe visible formatting problems; never auto-repair predictions."""
    txt=text or ''
    flags=[]
    if REDACTION.search(txt) or NEAR_REDACTION.search(txt):
        flags.append('deidentification_artifact')
    if not re.search(r'(?im)^\s*FINDINGS\s*:',txt) or not re.search(r'(?im)^\s*IMPRESSION\s*:',txt):
        flags.append('missing_structured_section')
    return flags


def encode_clean_target(tokenizer, structured: str, max_length: int = 192):
    """Tokenize both sections with an explicit budget reserved for Impression.

    Avoid the v9 problem where a long Findings paragraph could truncate the
    Impression entirely. Returns (ids, attention_mask, labels), each 1-D Tensor.
    """
    import torch
    if max_length < 128:
        raise ValueError('max_length must be >=128')
    marker='\nIMPRESSION:\n'
    if not structured.startswith('FINDINGS:\n') or marker not in structured:
        raise ValueError('cleaned structured target required')
    findings,impression=structured[len('FINDINGS:\n'):].split(marker,1)
    eos=tokenizer.eos_token_id
    header=[eos] + tokenizer.encode('FINDINGS:\n',add_special_tokens=False)
    between=tokenizer.encode(marker,add_special_tokens=False)
    ft=tokenizer.encode(findings,add_special_tokens=False)
    it=tokenizer.encode(impression,add_special_tokens=False)
    fixed=len(header)+len(between)+1
    budget=max_length-fixed
    if budget < 60:raise ValueError('Not enough token budget for report sections')
    # At least 35% of available tokens are reserved for Impression, even if
    # Findings is long. A short impression frees remaining tokens for Findings.
    reserved=min(len(it),max(32,int(.35*budget)))
    available_findings=budget-reserved
    ft=ft[:available_findings]
    it=it[:budget-len(ft)]
    ids=header+ft+between+it+[eos]
    attention=[1]*len(ids)
    labels=ids+[-100]*(max_length-len(ids))
    ids=ids+[eos]*(max_length-len(ids))
    attention=attention+[0]*(max_length-len(attention))
    return (torch.tensor(ids,dtype=torch.long),torch.tensor(attention,dtype=torch.long),
            torch.tensor(labels,dtype=torch.long))
