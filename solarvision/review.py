import math
import sqlite3
from datetime import datetime, timezone

class ReviewQueue:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, score REAL NOT NULL,
          priority TEXT NOT NULL, status TEXT NOT NULL, decision TEXT, note TEXT);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, id TEXT, time TEXT,
          action TEXT, note TEXT);
        ''')

    def add(self, ident, score, threshold):
        if not ident or not math.isfinite(score) or not 0 <= score <= 1 or not 0 <= threshold <= 1:
            raise ValueError('Invalid case')
        with self.db:
            self.db.execute('INSERT INTO cases VALUES(?,?,?,?,NULL,NULL)',
                            (ident, score, 'FLAGGED' if score >= threshold else 'LOW_SCORE', 'QUEUED'))
            self.db.execute('INSERT INTO events(id,time,action,note) VALUES(?,?,?,?)',
                            (ident, datetime.now(timezone.utc).isoformat(), 'QUEUED', 'Model screening; human decision pending'))

    def review(self, ident, decision, note):
        if decision not in ('defect', 'clear', 'uncertain') or not note.strip():
            raise ValueError('A valid decision and note are required')
        with self.db:
            row = self.db.execute('SELECT status FROM cases WHERE id=?', (ident,)).fetchone()
            if row is None or row['status'] != 'QUEUED':
                raise ValueError('Case missing or already reviewed')
            self.db.execute("UPDATE cases SET status='REVIEWED',decision=?,note=? WHERE id=?", (decision, note, ident))
            self.db.execute('INSERT INTO events(id,time,action,note) VALUES(?,?,?,?)',
                            (ident, datetime.now(timezone.utc).isoformat(), 'REVIEWED', note))

    def cases(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM cases ORDER BY score DESC,id')]

    def close(self):
        self.db.close()
