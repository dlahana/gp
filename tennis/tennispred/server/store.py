"""SQLite storage for invites, rating rounds, votes and drafts."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS invites (code TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT, invite TEXT NOT NULL, name_a TEXT NOT NULL, name_b TEXT NOT NULL,
    shown TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, round_id INTEGER NOT NULL UNIQUE REFERENCES rounds(id),
    invite TEXT NOT NULL, chosen INTEGER, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY, token TEXT NOT NULL, day TEXT NOT NULL, tour TEXT NOT NULL, matches TEXT NOT NULL,
    tweets TEXT NOT NULL, nicknames_used TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL,
    tweet_ids TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    # invites
    def add_invite(self, code: str, name: str) -> None:
        self._exec("INSERT INTO invites VALUES (?, ?, ?)", (code, name, now()))

    def invite(self, code: str) -> sqlite3.Row | None:
        return self._exec("SELECT * FROM invites WHERE code = ?", (code,)).fetchone()

    # rating
    def add_round(self, invite: str, a: str, b: str, shown: list[str]) -> int:
        return self._exec("INSERT INTO rounds (invite, name_a, name_b, shown, created_at) VALUES (?, ?, ?, ?, ?)",
                          (invite, a, b, json.dumps(shown), now())).lastrowid

    def round(self, round_id: int) -> sqlite3.Row | None:
        return self._exec("SELECT * FROM rounds WHERE id = ?", (round_id,)).fetchone()

    def add_vote(self, round_id: int, invite: str, chosen: int | None) -> bool:
        try:
            self._exec("INSERT INTO votes (round_id, invite, chosen, created_at) VALUES (?, ?, ?, ?)",
                       (round_id, invite, chosen, now()))
            return True
        except sqlite3.IntegrityError:
            return False   # already voted on this round

    def votes(self) -> list[sqlite3.Row]:
        return self._exec("SELECT r.name_a, r.name_b, r.shown, v.chosen, v.invite FROM votes v "
                          "JOIN rounds r ON r.id = v.round_id ORDER BY v.id").fetchall()

    def vote_count(self, invite: str | None = None) -> int:
        if invite is None:
            return self._exec("SELECT COUNT(*) FROM votes").fetchone()[0]
        return self._exec("SELECT COUNT(*) FROM votes WHERE invite = ?", (invite,)).fetchone()[0]

    # drafts
    def add_draft(self, d: dict) -> None:
        self._exec("INSERT INTO drafts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (d["id"], d["token"], d["day"], d["tour"], json.dumps(d["matches"]), json.dumps(d["tweets"]),
                    json.dumps(d["nicknames_used"]), d["source"], "pending", None, now(), now()))

    def draft(self, draft_id: str) -> dict | None:
        row = self._exec("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("matches", "tweets", "nicknames_used", "tweet_ids"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def update_draft(self, draft_id: str, **fields) -> None:
        for k in ("tweets", "nicknames_used", "tweet_ids"):
            if k in fields:
                fields[k] = json.dumps(fields[k])
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE drafts SET {cols}, updated_at = ? WHERE id = ?", (*fields.values(), now(), draft_id))
