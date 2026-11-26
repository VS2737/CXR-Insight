"""Generate a printable offline comparison of reference and AI drafts from test data.
Only use de-identified research reports; escapes text and includes warnings.
"""
import argparse
import csv
import html
import json
from pathlib import Path
from radreport.safety import report_claims


def build_casebook(rows, limit=20):
    blocks=[]
    for row in rows[:limit]:
        reference=row.get('reference','') or ''
        generated=row.get('generated','') or ''
        a=report_claims(reference);b=report_claims(generated)
        ref_positive={k for k,v in a['findings'].items() if v=='PRESENT'}
        gen_positive={k for k,v in b['findings'].items() if v=='PRESENT'}
        missed=sorted(ref_positive-gen_positive)
        added=sorted(gen_positive-ref_positive)
        flags=[]
        if missed: flags.append('Reference-mentioned findings absent from draft: '+', '.join(missed))
        if added: flags.append('Draft findings not mentioned in reference: '+', '.join(added))
        if b['contradictions']: flags.append('Internal text contradiction detected')
        if b['normal_scope']!='NONE' and ref_positive: flags.append('Normal-style draft despite abnormal reference mentions')
        if not flags: flags=['No lexical discrepancy detected; correctness still unverified']
        flags_html=''.join('<li>'+html.escape(f)+'</li>' for f in flags)
        blocks.append('<section class="case"><header>Study '+html.escape(str(row.get('uid','unknown')))+
          '</header><div class="cols"><div><h3>Reference impression</h3><p>'+html.escape(reference)+
          '</p></div><div><h3>AI draft</h3><p>'+html.escape(generated)+
          '</p></div></div><h3>Review triggers</h3><ul>'+flags_html+'</ul></section>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CXR Insight — Research Casebook</title>
<style>body{font:15px/1.55 system-ui,Arial;color:#1b3546;max-width:1060px;margin:40px auto;padding:0 22px}
h1{color:#143f51}.notice{border:1px solid #ba8b42;background:#fff7e7;padding:14px;border-radius:8px}
.case{break-inside:avoid;border:1px solid #d5e1e9;padding:20px;margin:20px 0;border-radius:10px}header{font-size:18px;font-weight:700}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}.cols>div{background:#f4f8fa;padding:12px;border-radius:8px}h3{font-size:13px;margin:4px 0;text-transform:uppercase;letter-spacing:.05em}p{white-space:pre-wrap}li{margin:3px 0}@media print{body{margin:5mm} .case{page-break-inside:avoid}}</style></head><body>
<h1>CXR Insight — Held-out Research Casebook</h1><p class="notice"><strong>NOT FOR CLINICAL USE.</strong> Text comparisons are heuristic; absence from a reference does not prove hallucination. No images or personal information are included. All drafts require qualified review.</p>'''+''.join(blocks)+'</body></html>'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='results/predictions.csv')
    parser.add_argument('--limit',type=int,default=20)
    parser.add_argument('--output',default='results/review_casebook.html')
    args=parser.parse_args()
    with open(args.input,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if not rows or not {'uid','reference','generated'}.issubset(rows[0]):
        raise ValueError('Need predictions.csv with uid/reference/generated headers')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(build_casebook(rows,args.limit),encoding='utf-8')
    print(f'Created {out} with {min(args.limit,len(rows))} research study comparisons')

if __name__=='__main__':main()
