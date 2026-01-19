from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os
import re
import time
import sqlite3
import hashlib
import base64

def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-z0-9_]+", text)
    return [t for t in toks if len(t) >= 3]

def _keyword_score(query: str, doc: str) -> float:
    q = (query or "").lower().strip()
    d = (doc or "").lower()
    q_toks = set(_tokenize(q))
    if not q_toks:
        return 0.0
    d_toks = set(_tokenize(d))
    overlap = len(q_toks.intersection(d_toks))
    base = overlap / max(len(q_toks), 1)
    if q and q in d:
        base += 0.25
    return float(base)

def _pbkdf2_hash_password(password: str, salt: bytes, rounds: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)

class LongTermMemorySQLite:
    """
    User-scoped long-term memory stored in SQLite.
    Retrieval is keyword overlap + recency bias (no embeddings).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._conn() as con:
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    salt_b64 TEXT NOT NULL,
                    pw_hash_b64 TEXT NOT NULL,
                    created_ts REAL NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    created_ts REAL NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_ts ON memories(user_id, created_ts)")
            con.commit()

    def create_user(self, username: str, password: str) -> int:
        salt = os.urandom(16)
        pw_hash = _pbkdf2_hash_password(password, salt)
        with self._conn() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO users(username, salt_b64, pw_hash_b64, created_ts) VALUES (?, ?, ?, ?)",
                (
                    username,
                    base64.b64encode(salt).decode("utf-8"),
                    base64.b64encode(pw_hash).decode("utf-8"),
                    time.time(),
                ),
            )
            con.commit()
            return int(cur.lastrowid)

    def authenticate(self, username: str, password: str) -> Optional[int]:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute("SELECT id, salt_b64, pw_hash_b64 FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            if not row:
                return None
            user_id, salt_b64, pw_hash_b64 = row
            salt = base64.b64decode(salt_b64)
            expected = base64.b64decode(pw_hash_b64)
            got = _pbkdf2_hash_password(password, salt)
            return int(user_id) if got == expected else None

    def create_or_auth(self, username: str, password: str) -> int:
        uid = self.authenticate(username, password)
        if uid is not None:
            return uid
        return self.create_user(username, password)

    def add_memory(self, user_id: int, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._conn() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO memories(user_id, created_ts, text) VALUES (?, ?, ?)",
                (user_id, time.time(), text),
            )
            con.commit()

    def search(
        self,
        user_id: int,
        query: str,
        k: int = 5,
        min_score: float = 0.15,
        candidate_limit: int = 300,
    ) -> List[Tuple[float, str]]:
        q = (query or "").strip()
        if not q:
            return []

        with self._conn() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT created_ts, text FROM memories WHERE user_id=? ORDER BY created_ts DESC LIMIT ?",
                (user_id, int(candidate_limit)),
            )
            rows = cur.fetchall()

        now = time.time()
        scored: List[Tuple[float, str]] = []
        for ts, text in rows:
            s = _keyword_score(q, text)
            age_days = max((now - float(ts)) / 86400.0, 0.0)
            recency_boost = 0.10 * (1.0 / (1.0 + age_days / 30.0))
            s2 = s + recency_boost
            if s >= min_score:
                scored.append((s2, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

def update_short_term_memory(memory: Dict, user_q: str, assistant_a: str, max_recent_turns: int = 6) -> Dict:
    """
    Keeps last `max_recent_turns` messages intact (as a list of {role, content}),
    and leaves older messages in `older_summary` (caller decides when/how to summarize).
    """
    recent = memory.get("recent_turns", [])
    recent = recent + [{"role": "user", "content": user_q}, {"role": "assistant", "content": assistant_a}]
    memory["recent_turns"] = recent[-max_recent_turns:]
    memory.setdefault("older_summary", "")
    return memory
