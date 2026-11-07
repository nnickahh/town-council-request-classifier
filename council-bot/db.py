import sqlite3
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    category TEXT,
    description TEXT NOT NULL,
    location TEXT,
    photo_id TEXT,
    town_council TEXT,
    lat REAL,
    lng REAL,
    ai_diagnosis TEXT,
    severity TEXT DEFAULT 'normal',
    subscribers TEXT DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open',
    staff_note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    sources TEXT,
    answered INTEGER NOT NULL,
    feedback INTEGER,
    handled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS forum_topics (
    town_council TEXT PRIMARY KEY,
    thread_id INTEGER NOT NULL
);
"""


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


import json

def init():
    with conn() as c:
        c.executescript(SCHEMA)
        # Migrate existing database columns
        cols = [r[1] for r in c.execute("PRAGMA table_info(tickets)").fetchall()]
        migrations = [
            ("town_council", "TEXT"),
            ("lat", "REAL"),
            ("lng", "REAL"),
            ("ai_diagnosis", "TEXT"),
            ("severity", "TEXT DEFAULT 'normal'"),
            ("subscribers", "TEXT DEFAULT '[]'"),
        ]
        for col, col_type in migrations:
            if col not in cols:
                try:
                    c.execute(f"ALTER TABLE tickets ADD COLUMN {col} {col_type}")
                except Exception:
                    pass


def create_ticket(user_id, category, description, location, photo_id, town_council=None, lat=None, lng=None, ai_diagnosis=None, severity="normal"):
    with conn() as c:
        subs = json.dumps([user_id])
        cur = c.execute(
            "INSERT INTO tickets (user_id, category, description, location, photo_id, town_council, lat, lng, ai_diagnosis, severity, subscribers) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, category, description, location, photo_id, town_council, lat, lng, ai_diagnosis, severity, subs),
        )
        return cur.lastrowid


def find_duplicate_ticket(category, location, max_hours=24):
    """Finds an existing open ticket for the same issue and location."""
    if not location or len(location) < 3:
        return None
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM tickets WHERE status != 'resolved' AND category = ? ORDER BY id DESC LIMIT 20",
            (category,),
        ).fetchall()
        loc_clean = location.lower()
        for r in rows:
            r_loc = (r["location"] or "").lower()
            if not r_loc:
                continue
            # If same postal code or overlapping block/street
            import re
            p1 = re.search(r"\b\d{6}\b", loc_clean)
            p2 = re.search(r"\b\d{6}\b", r_loc)
            if p1 and p2 and p1.group(0) == p2.group(0):
                return r
            if loc_clean in r_loc or r_loc in loc_clean:
                return r
    return None


def subscribe_to_ticket(ticket_id, user_id):
    """Links an additional resident to an existing ticket to receive resolution alerts."""
    with conn() as c:
        row = c.execute("SELECT subscribers FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row:
            try:
                subs = json.loads(row[0] or "[]")
            except Exception:
                subs = []
            if user_id not in subs:
                subs.append(user_id)
                c.execute("UPDATE tickets SET subscribers = ? WHERE id = ?", (json.dumps(subs), ticket_id))


def get_ticket(ticket_id):
    with conn() as c:
        return c.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()


def set_status(ticket_id, status, note=None):
    with conn() as c:
        c.execute(
            "UPDATE tickets SET status = ?, staff_note = COALESCE(?, staff_note), "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, note, ticket_id),
        )


def log_qa(user_id, question, answer, sources, answered):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO logs (user_id, question, answer, sources, answered) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, question, answer, sources, int(answered)),
        )
        return cur.lastrowid


def set_feedback(log_id, value):
    with conn() as c:
        c.execute("UPDATE logs SET feedback = ? WHERE id = ?", (value, log_id))


def get_topic(council: str):
    with conn() as c:
        row = c.execute("SELECT thread_id FROM forum_topics WHERE town_council = ?", (council,)).fetchone()
        return row[0] if row else None


def save_topic(council: str, thread_id: int):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO forum_topics (town_council, thread_id) VALUES (?, ?)", (council, thread_id))


def list_tickets(statuses=None):
    q, params = "SELECT * FROM tickets", []
    if statuses:
        q += " WHERE status IN (%s)" % ",".join("?" * len(statuses))
        params = list(statuses)
    with conn() as c:
        return c.execute(q + " ORDER BY id DESC", params).fetchall()


def list_unanswered():
    with conn() as c:
        return c.execute("SELECT * FROM logs WHERE answered = 0 AND handled = 0 ORDER BY id DESC").fetchall()


def list_downvoted():
    with conn() as c:
        return c.execute("SELECT * FROM logs WHERE feedback = 0 ORDER BY id DESC").fetchall()


def mark_handled(log_id):
    with conn() as c:
        c.execute("UPDATE logs SET handled = 1 WHERE id = ?", (log_id,))


def stats():
    with conn() as c:
        one = lambda q: c.execute(q).fetchone()[0]
        return {
            "tickets": one("SELECT COUNT(*) FROM tickets"),
            "open": one("SELECT COUNT(*) FROM tickets WHERE status != 'resolved'"),
            "resolved": one("SELECT COUNT(*) FROM tickets WHERE status = 'resolved'"),
            "questions": one("SELECT COUNT(*) FROM logs"),
            "answered": one("SELECT COUNT(*) FROM logs WHERE answered = 1"),
            "up": one("SELECT COUNT(*) FROM logs WHERE feedback = 1"),
            "down": one("SELECT COUNT(*) FROM logs WHERE feedback = 0"),
        }
