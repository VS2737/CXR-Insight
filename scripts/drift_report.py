"""Monitor metadata-only image brightness / contrast using Evidently 0.4.x.

Reference CSV must have brightness,contrast columns and contain de-identified
reference image STATISTICS, not images or individual identifiers.
"""
import argparse
import json
import sqlite3
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--database',default='data/review_queue.sqlite3')
    p.add_argument('--output',default='results/drift_report.html')
    a=p.parse_args()
    reference=pd.read_csv(a.reference)[['brightness','contrast']].dropna()
    with sqlite3.connect(a.database) as db:
        rows=db.execute('SELECT image_stats FROM reviews').fetchall()
    current=pd.DataFrame([json.loads(x[0]) for x in rows])[['brightness','contrast']].dropna()
    if len(reference)<20 or len(current)<20:
        raise ValueError('At least 20 reference and current observations required.')
    r=Report(metrics=[DataDriftPreset()]);r.run(reference_data=reference,current_data=current)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);r.save_html(a.output)
    print(a.output)
