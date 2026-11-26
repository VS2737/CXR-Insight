"""SQLite queue: DOES NOT persist images, report text, or identifying patient information."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .config import DB_PATH


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute('''CREATE TABLE IF NOT EXISTS reviews (
       id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, state TEXT NOT NULL,
       model_version TEXT NOT NULL, image_stats TEXT NOT NULL,
       safety_summary TEXT NOT NULL, reviewer TEXT, decision TEXT, reviewed_at TEXT)''')
    connection.commit()
    return connection


def enqueue(identifier: str, model_version: str, image_stats: dict, safety: dict):
    summary = {k: safety.get(k) for k in ('self_consistency', 'review_reasons', 'safety_check_passed')}
    with connect() as db:
        db.execute('INSERT INTO reviews (id,timestamp,state,model_version,image_stats,safety_summary) VALUES (?,?,?,?,?,?)',
                   (identifier, datetime.now(timezone.utc).isoformat(), 'PENDING', model_version,
                    json.dumps(image_stats), json.dumps(summary)))


def list_reviews(limit: int = 50):
    with connect() as db:
        rows = db.execute('SELECT id,timestamp,state,model_version,image_stats,safety_summary,reviewer,decision,reviewed_at FROM reviews ORDER BY timestamp DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]


def decide(identifier: str, reviewer: str, decision: str):
    if decision not in ('ACCEPT_DRAFT', 'REVISE_DRAFT', 'REJECT_DRAFT'):
        raise ValueError('Invalid decision')
    if not reviewer.strip():
        raise ValueError('Reviewer name is required')
    with connect() as db:
        c = db.execute('UPDATE reviews SET state=?, reviewer=?, decision=?, reviewed_at=? WHERE id=? AND state=?',
                       ('REVIEWED', reviewer.strip(), decision, datetime.now(timezone.utc).isoformat(), identifier, 'PENDING'))
        return c.rowcount > 0
